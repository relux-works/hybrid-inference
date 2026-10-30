# M0.2 append-only sessions: scope decision

Run: `hybrid-b-r10-261030`, 2026-10-30 UTC. Branch: `m02-append-only-sessions`, based on signed main `edb31a7edfa17cb83faadd0a06a700a0291c6fb8` fetched read-only from `/home/agent/Werkstatt/hybrid-prototype.git`.

**Status: blocked before feature implementation.** The user's setup says: "If you need an engine or `llama-server` change, stop and report what and why first." The existing chat route cannot consume an authoritative token journal. Its raw completion alternative accepts exact IDs but does not retain the native reasoning/tool parser, tool grammar or chat message boundaries. A proxy implementation of those systems would be a separate design and would duplicate the existing server implementation. This report requests a small llama-server API extension instead. No engine, server, codec or prefill-node files were changed. The current proxy behavior is unchanged.

## Verified constraints

| Constraint | Source | Native control |
| --- | --- | --- |
| Chat always uses its template-rendered prompt. A supplied `prompt` token array cannot override it. | `tools/server/server-common.cpp:1515` sets `llama_params["prompt"]`; `:1550` does not overwrite an existing key. `tools/server/server-context.cpp:6082` uses these parsed parameters. | Supply two different token prefixes. `/completion` evaluates the supplied IDs and hash exactly. `/v1/chat/completions` instead evaluates the independently tokenized `/apply-template` result and its hash. Both templates reproduce this. |
| Raw completion has no native chat session. | `tools/server/server-context.cpp:6053` passes no chat session; `:6082` constructs one for chat. `tools/server/server-task.h:308` skips chat sampling, format and spans when no template is attached. | Exact raw token input is a passing positive control, not evidence of chat compatibility. The missing parser/grammar/spans are established by source inspection, not a tool-output generation test. |
| Nonstream sampled IDs are available. | `tools/server/server-context.cpp:2159`, `:2437`; `tools/server/server-task.cpp:445`. | With `return_tokens=true, verbose=true`, nonstream chat returns eight sampled IDs in `__verbose.tokens`; slot info holds the prompt plus the first seven. |
| Verbose streaming does not expose those sampled IDs. | `tools/server/server-task.cpp:1114` emits chat diffs and optional logprobs, without native token IDs. `tools/server/server-context.cpp:2433` explicitly clears final stream tokens. | With the same flags and eight generated tokens, final `__verbose.tokens` is empty, partial chunks contain no native token list, and slot info contains only seven evaluated generated tokens. Slot info therefore cannot recover the final sampled ID. |
| Logprobs cannot close that gap for tool streams. | `tools/server/server-common.cpp:1537`. | `tools=[valid_function], tool_choice="none", stream=true, logprobs=true, top_logprobs=1` returns exactly 400 with `logprobs is not supported with tools + stream`. |
| `/apply-template` returns prompt text, not message offsets or token boundaries. | `tools/server/server-context.cpp:6436`. | Probe uses the returned text as the independent chat-prompt oracle. General incremental suffix exactness was not established. |

The missing IDs concern sampled versus evaluated state. A durable journal can already record the evaluated IDs from slot info, but the final sampled token must also be available without re-tokenizing client text before a later append. Capturing nonstream IDs and then fabricating a streamed response would lose live upstream streaming. Using the raw route would require a separate reasoning/tool parser and generation-constraint integration in the proxy. Neither alternative was implemented.

## Proposed scope change and design

The requested decision is authorization to change **llama-server only**, in addition to the proxy. No core engine or A-owned prefill-node change is proposed:

1. Add an optional native `prompt_tokens` array to chat completion requests. Construct the existing `common_chat_session` from messages/tools/settings as today, but use the validated array for task token input. Retain native chat output parsing, grammar/sampling and delimiter-based message spans. Reject invalid types, token IDs and lengths before slot mutation. Ordinary chat requests retain their current behavior.
2. When `return_tokens=true`, retain the complete sampled ID vector in the final stream result's existing verbose native data, rather than clearing it solely because the request streams. This reuses `return_tokens` and `verbose`; it does not require logprobs or a new parser. The final list must include stop/EOG tokens and tokens whose text was buffered or suppressed. The native stream controls must cover those cases before acceptance.

These are proposed changes for approval, not implemented patches or a proved sufficient M0.2 design. The source sites above make the proposal concrete. The HTTP API between the proxy and A's prefill node, and the state-delta format, need no change for the documented fork fallback.

After the scope decision, the proxy design should retain its M0.1 per-session FIFO and pinned slot lease, and commit a message journal beside its exact token journal. Store message identity, incoming versus generated segments, evaluated and sampled boundaries, template/settings identity and user-boundary checkpoint positions. Ignore a missing `reasoning_content` only when comparing an otherwise matching assistant message; never ignore edits to user/tool content or supplied conflicting reasoning. A generated final token must be reconciled with the evaluated slot position before metering or replay claims it as applied.

For a matched history, keep every old journal ID and append only the newly rendered suffix; send the authoritative combined IDs to chat without re-tokenizing old generated text. A safe suffix algorithm is still required. Full re-render length subtraction alone is invalid when earlier reasoning disappears. Do not infer message offsets by searching untrusted user text for delimiters. Establish boundaries with the active template/tokenizer and prove the new suffix at its message position, including tool groups, generation prefixes and BPE boundaries. If existing template output cannot establish that unambiguously, an explicit template-suffix/boundary API would need its own scope decision. No exactness guarantee is claimed by this diagnostic run.

Meter actual local prompt evaluations and remote applied ranges, with generated-token reconciliation reported separately. A real rewrite must use a recorded valid checkpoint at or before divergence or take the documented fork; explicit reuse without an allowed transition must return 409. The present explicit mismatch path raises `ValueError` and is mapped to 400 (`store.py:103`, `proxy.py:537`); that must be corrected during implementation, not described as already meeting the new 409 requirement.

## Checkpoint findings

Native checkpoints do cover the local hybrid slot. This is upstream functionality at the pinned base, not an A-only addition. `tools/server/server-context.cpp:4473` enables them for recurrent/limited-removal/SWA memory; `:4579` splits batches at user starts; `:4663` saves before evaluating that batch. `:4330` selects a usable checkpoint on a later prompt divergence. `tools/server/server-task.h:316` derives message spans from the task tokens using the chat session's delimiters. Raw completion has no such chat session or user spans.

The probe enables `--ctx-checkpoints 32 --checkpoint-min-step 0`, evaluates two user turns, edits only the later user question, and requires the restored cache position to equal the independently tokenized prefix ending immediately before that user's `<|im_start|>user\n`. This literal delimiter is used only for the two named Qwen templates in this fixture, not as a proposed generic product parser. It also verifies every final edited token and its prefix hash.

| Template on the 0.8B model | Latest user boundary | Restored `cache_n` | Server trace |
| --- | ---: | ---: | --- |
| Model's Qwen3.5 GGUF template | 120 | 120 | `restored context checkpoint ... n_tokens = 120, n_past = 120` |
| Supplied Qwen3.8-27B template | 124 | 124 | `restored context checkpoint ... n_tokens = 124, n_past = 124` |

The current proxy erases a mismatch at `proxy.py:399` before this native recovery can run. Its documented M0.1 launch also sets `--ctx-checkpoints 0`. M0.2 should enable checkpoints and preserve the slot for native rollback once the journal and authoritative-input path exist. The controls establish two simple local rewrite cases; they do not prove compaction, eviction/restart persistence, checkpoint eviction, an arbitrary earlier divergence, or an imported offloaded boundary. Native save/restore persists checkpoint appendices (`server-context.cpp:2805`), but that path was inspected only, not exercised in this run.

## What the prefill node would need

Today's section 5 has current-state snapshot, create, extend, info and delete; it cannot restore or fork an older checkpoint. A future operation would need a client-scoped checkpoint identity, model identity, token position and prefix hash, and an atomic restore or fork to that identity. Foreign checkpoint/session ownership must return 404 and stale/incompatible state must fail before mutation. FIFO ordering must include the operation. This is a proposal for A and the orchestrator, not a unilateral shared-contract change.

The authorized fallback remains creating a fresh remote session and replaying the chosen journal prefix. A usable local checkpoint can still avoid local full prefill, but the new remote session pays for replay. Without a usable local checkpoint an inferred rewrite forks; explicit reuse without an applicable append/rollback returns 409. None of these M0.2 transitions were implemented or tested here.

## Commands, identity and exit codes

All commands ran from `/home/agent/Werkstatt/hybrid-b/llama.cpp`. Outputs are under `.temp/hybrid-b-r10-261030/`; `gates.jsonl` records the actual argument vector, UTC start, duration, log and process exit code. `run.py` only invokes a standalone subprocess and redirects its output; it does not pipe a gate through `tee` or mask its exit status.

Fetch: `git fetch /home/agent/Werkstatt/hybrid-prototype.git main:refs/remotes/prototype/main`, exit 0. Branch: `git switch -c m02-append-only-sessions edb31a7e`, exit 0. Free root space before native work: 40,188,350,464 bytes. No configure/build, installation, download or push ran.

Existing binary identity: `llama-server --version` reports build 12044, commit `328cfb23cf`, GNU 15.2.0, Linux x86_64. `git diff 328cfb23cf edb31a7e -- src tools/server common include` is empty. Server SHA-256 `b88d7a41a493de8fb1089218954c4fd0c953d6ee0de66f228c8b7d9837340a9e`; libllama SHA-256 `1ad5d26ba8388d05ca87d8872dd6dcbc104694c24b38dff93ee7ae44844c8865`. The only model's SHA-256 is `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`. Qwen3.8 template SHA-256 is `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.

```sh
python3 .temp/hybrid-b-r10-261030/run.py api-default-02 python3 tools/hybrid-proxy/probe_append_api.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r10-261030/api-default-02
python3 .temp/hybrid-b-r10-261030/run.py api-target-02 python3 tools/hybrid-proxy/probe_append_api.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --artifacts .temp/hybrid-b-r10-261030/api-target-02
python3 .temp/hybrid-b-r10-261030/run.py chat-token-prerequisite-red-01 python3 tools/hybrid-proxy/probe_append_api.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r10-261030/chat-token-prerequisite-red-01 --require-chat-token-input
python3 .temp/hybrid-b-r10-261030/run.py baseline-proxy-01 python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v
python3 .temp/hybrid-b-r10-261030/run.py syntax-01 python3 -m py_compile tools/hybrid-proxy/probe_append_api.py
python3 .temp/hybrid-b-r10-261030/run.py delivery-check-01 python3 .temp/hybrid-b-r10-261030/check_delivery.py
ty --version > .temp/hybrid-b-r10-261030/ty-readiness-01.log 2>&1
editorconfig-checker --version > .temp/hybrid-b-r10-261030/editorconfig-readiness-01.log 2>&1
```

| Gate | Exit | Meaning |
| --- | ---: | --- |
| `api-default-01`, `api-target-01` | 0 each | Initial controls: exact raw versus re-rendered chat, sampled-ID visibility, tool/logprobs rejection and nonzero checkpoint reuse. Superseded by the stronger boundary assertion below. |
| `api-default-02`, `api-target-02` | 0 each | Same controls plus exact latest user-boundary restore and final token/hash equality. |
| `chat-token-prerequisite-red-01` | **1, expected red** | Fails on `M0.2 prerequisite: chat must evaluate authoritative token IDs while retaining its chat session`. All preceding controls ran successfully; it is not a setup failure. |
| `baseline-proxy-01` | 0 | All 38 existing tests, original names and assertions retained, 70.494 seconds reported by unittest. This qualifies the unchanged proxy baseline, not M0.2. |
| `syntax-01` | 0 | New diagnostic compiles with the installed Python. |
| `delivery-check-01` | 0 | Changed file formats/whitespace, unchanged protected source, all probe PIDs absent. The script is a task-local check, not a product acceptance gate. |
| `ty --version`, `editorconfig-checker --version` | 127 each | Executables absent; their checks are unrun. Readiness logs retain the errors. |

For a fresh checkout, omit the `python3 .temp/.../run.py LABEL` prefix to run the committed diagnostic directly; its own artifacts include command/identity/results/server PID and cleanup. The source beside each probe states the precise positive or negative claim and the named-template limit.

## Coverage and remaining gates

| User requirement | Status |
| --- | --- |
| Exact generated IDs and message journal | Nonstream ID source verified; streamed tool-compatible ID source blocked. No message journal implemented. |
| Matched-history append and exact new-message tokens | Blocked on authoritative chat input; suffix guarantee still unestablished. |
| Local checkpoint rollback | Native two-user edit control passes at 120/124; proxy compaction/rollback transition unimplemented. |
| Remote historical restore/fork | API need documented; no A edits. Fresh-session fallback remains planned. |
| Per-request local/remote metering; explicit 409 semantics | Unimplemented. |
| Tests 1-5: three-user/tool loops, both thinking modes, target reasoning present/omitted, compaction/isolation, exact greedy64 | **Not run** as M0.2 gates; no candidate exists. Probe's eight-token API controls are not substitutes. |
| Test 6: 38 proxy regressions | Pass on unchanged runtime. Real 2 clients x 2 sessions/section 11 **not rerun**, because feature implementation stopped. Prior M0.1 evidence is not presented as M0.2 evidence. |
| Test 7: requested M0.1 conversation witnesses 1 and 3 | **Not run.** The retained red log is a different, narrower chat-token-input prerequisite witness. |
| Test 8: three-user total prefilled tokens, M0.1 versus M0.2 | **Unavailable** for both requested conversations; no values fabricated. |

The brief supplies no surface table. This is an explicit coverage-map gap; no full feature handoff is claimed. No out-of-contract requirement was silently dropped: all remaining requirements above are pending the scope decision. The diff contains only a native diagnostic and this proxy documentation, inside `tools/hybrid-proxy/`; all previously committed reviewer tests remain present and green. Self-review used scope, security, general, design and test checklists from `skills/code-review/SKILL.md`. No reusable infrastructure or dependency was added. The repository-wide `ty` and `editorconfig-checker` jobs were not run: neither executable is installed, and no installation was authorized. Syntax, diff whitespace and the applicable file-format rules were checked locally; this is not a substitute for those hosted jobs. Model-naming checks are outside this diff, which adds no model code.

The only servers started were the five probe processes, each recorded under its artifact directory and terminated/waited by its own PID in `finally`. Cleanup records and a separate PID readback are retained in the task artifacts. Small contexts (4096 total, two slots), two CPU threads, no benchmarks. This is a reviewed scope report and diagnostic, not an accepted M0.2 implementation.

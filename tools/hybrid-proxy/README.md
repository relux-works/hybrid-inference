# Hybrid client proxy prototype

For private deployment, seeded chat benchmarks, opt-in content-free timing traces and real-node equivalence checks, see [the hybrid benchmark guide](../hybrid-bench/README.md). Benchmark switches require explicit operator configuration; ordinary requests use the existing routing policy.

This directory contains the Python client side of the hybrid prefill/decode prototype. Run commands from the repository root. Python 3.10+ and its standard library are sufficient for the proxy. The synthetic stub additionally calls the already installed `libxxhash` through `ctypes`; it does not install a Python package. The real delta reader is shared with producer A in `tools/hybrid`.

## M0.2 append-only sessions

Text user/tool turns use a persistent message journal beside the exact token journal. Matching uses canonical JSON for all message fields, preserving boolean/integer/float/string types and null versus missing keys recursively. Tool-argument JSON ignores whitespace, object key order and equivalent Unicode escapes; only omitted assistant `reasoning_content` is ignored, while retaining the original reasoning in both journals. Explicit edits use the latest usable user checkpoint; inferred edits fork, borrowing a same-client native checkpoint when possible. An explicit rewrite without a usable checkpoint returns 409. See [m02-append-only-261030.md](m02-append-only-261030.md) for design, lock order, qualification and exact exits.

The proxy asks `/apply-template` to render a nonce at the preceding assistant boundary. It accepts a suffix only if the real render ends with exactly those bytes and token IDs. Only that checked suffix is appended to the existing IDs; generated history is never re-tokenized for reuse. The qualified framing is ChatML with Qwen user/tool/assistant frames. Unsupported or ambiguous frames fail explicitly. Message boundaries include a token index and a byte offset within that token, so a BPE token spanning a textual boundary is represented honestly. Generation reconciliation evaluates the final sampled token before the journal commits; it is metered separately from prompt prefill. A temporary client-scoped admission map keeps an inferred follow-up on its FIFO session during the gap between stream DONE and durable commit. Checkpoint rewrites replay unchanged generated segments as original IDs before rendering the changed suffix.

The server bridge uses opt-in `prompt_tokens` on chat requests, preserving native reasoning/tool parsing and grammar. Arrays must be nonempty, use integer vocabulary IDs, and fit the slot context. The existing server completion guard still reserves one token, including for `n_predict=0`; `n_ctx-1` is the usable prompt limit. With `return_tokens=true, verbose=true`, a chat stream's final `__verbose.tokens` includes every sampled ID, including hidden stop/EOG/buffered tokens. The proxy consumes that vector internally and forwards ordinary OpenAI chat output. Clients cannot inject their own `prompt_tokens` through the proxy.

```sh
python3 tools/hybrid-proxy/test_append_api.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/append-api
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/append-conversation --thinking on
python3 tools/hybrid-proxy/test_remote_append.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/remote-append
python3 tools/hybrid-proxy/test_message_identity.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/typed-identity
```

Use a fresh artifact directory for each run. On the shared host, group native artifact paths below one task scratch directory; retain hashes/metrics and remove the state dumps after stopping the servers. These tools start existing CPU binaries and stop them by PID, never build. The API tool checks malformed arrays, exact native bytes, parsed output and complete streamed IDs. The conversation tool compares three user turns with real tool loops against an independent frame/token oracle and a solo context with the same partition; `--thinking off`, `--inferred`, `--omit-reasoning` and `--template PATH` select the other qualification cases. The remote tool checks fresh-session replay, retirement, real delta metering and no-checkpoint transitions. The identity tool checks both directions of scalar/nested edits against native checkpoint rollback, plus identical/key-order/Unicode append controls; `--no-checkpoint-control` requires conflicts with unchanged native bytes. Its pure identity and admission tests also run in the proxy unit suite. Commands, identities, results, owned PIDs and resource/cleanup records go under `--artifacts`. `native_fixture.py` provides their shared process lifecycle.

`probe_append_api.py` is the smaller capability probe: add `--require-chat-token-input` to require both new capabilities. The original expected-red scope probe is preserved at commit `d079547f`; the current probe detects the extended API. M0.1 conversation witnesses use `test_append_sessions.py --proxy-source DIRECTORY --witness` with modules extracted from `edb31a7e` and must exit 1 for full re-prefill, not setup errors.

## Run the synthetic path

The example contains public fixture credentials, not deployment credentials. Start these three commands in separate terminals:

```sh
mkdir -p .temp/hybrid-b-r1-261030/slots
python3 tools/hybrid-proxy/stub_server.py --role prefill --port 8081 --database .temp/hybrid-b-r1-261030/stub.sqlite3
python3 tools/hybrid-proxy/stub_server.py --role decode --port 8080 --slot-path .temp/hybrid-b-r1-261030/slots
python3 tools/hybrid-proxy/proxy.py --config tools/hybrid-proxy/config.example.json --database .temp/hybrid-b-r1-261030/client.sqlite3 --port 8082
```

```sh
curl --no-buffer http://127.0.0.1:8082/v1/chat/completions -H 'Authorization: Bearer fixture-a' -H 'Content-Type: application/json' -d '{"model":"stub","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

The response includes `X-Hybrid-Session`. Send it on later requests for explicit association. Without it, the proxy first matches client-scoped message journals, then uses cumulative SHA-256 token hashes at 256-token boundaries for legacy sessions. Divergent inferred histories get a fresh session, using a copied local checkpoint when available. An explicit ID owned by another client returns 404. Explicit IDs must contain only ASCII letters, digits, hyphens and underscores, up to 128 characters.

The stub has a clearly labelled synthetic token body. It cannot be imported by the real engine. Its HTTP state-machine and corruption tests do not establish inference equivalence or real prefill-node isolation.

## Run with the engine

Set `model_sha256` to the SHA-256 of the one shared GGUF. Point `prefill_url` at A's prefill node and `decode_url` at the patched llama-server. Replace the fixture credentials in a private configuration file, and restrict its permissions. Increase `prefill_timeout` to cover real CPU prefill during correctness runs (for example, 120 seconds); its default of 0.5 seconds bounds waits when using an unavailable development server. `slot_path` must be the same local directory passed to llama-server as `--slot-save-path`, and `slots` must match its `--parallel` value. Keep context shifting, speculative decoding, automatic idle-slot caching, and RAM prompt caching disabled for this prototype (`--no-context-shift --no-cache-idle-slots --cache-ram 0 --ctx-checkpoints 32 --checkpoint-min-step 0`). Text chat and upstream chat tool parsing are forwarded; multimodal inputs and multiple choices are not supported. Anthropic Messages is not implemented.

The decode node must expose `GET /slots/{id}/info` with `n`, `prefix_hash`, and `tokens`, and binary `POST /slots/{id}?action=import_delta`. The import uses A's shared checked reader. The proxy offloads all but the final prompt token, then submits the chat request with authoritative token IDs locally. That last token produces the client's exact logits; memory serialization alone does not carry logits.

SQLite stores each client's exact token and message journals, message/segment boundaries, sampled/evaluated positions, template/settings identities, checkpoint candidates, block hashes, remote acknowledgement position, slot, and last-use time. The two journals commit together after generation reconciliation. Intermediate turn state stays in memory; uncommitted native extensions are discarded on verification. Historical remote sessions retired by a committed rewrite are deleted through bounded background replay. This contains conversation data; keep the database and slot directory private. Shadow replay runs in a background thread, and also precedes a critical extend. A 202 only acknowledges queuing; GET establishes the applied boundary. After a lost acknowledgement or restart, the boundary and hash determine the remaining suffix, preventing duplicate application. If a RAM-only server lost the session, the proxy creates a replacement remote session and replays the complete durable history. If the server is unavailable, the request uses local prefill. After a restart, the proxy checks slot state and the remote prefix before reuse. Slot eviction saves the least recently used session and restores it on demand; missing parked files fall back to local prefill from durable tokens.

A 409 logs both hashes. A behind server receives journal replay; an ahead server is recovered through its snapshot and the local slot restore action. The restored prefix must match the incoming history. Incompatible model/prefix metadata returns 409 before restore. A snapshot mismatch or journal failure restores the prior decoder snapshot. Recovery GET transport errors and retryable HTTP failures continue locally, then replay on reconnect. Different sessions decode concurrently up to the configured slot count. Each session has a FIFO ticket queue. Requests reserve a client admission position before template/tokenizer HTTP; those HTTP calls run concurrently, and their completed mapping results enter session queues in arrival order. A slow earlier classification can delay later admission for that client, while other clients, already admitted streams and replay keep progressing. Failed classification removes its admission position. The queued request reloads its journal before use, so a completed predecessor can turn an inferred request into a fork. Configure `slots` to match the decoder's `--parallel` pool. A session with a running request pins its slot through prepare, recovery, import, streaming and journal commit. Eviction chooses only an unpinned slot; it reserves that slot under a short lock, then saves/erases/restores over HTTP after releasing the shared lock. When all slots are unavailable, a new session waits up to 30 seconds and returns `503` with `Retry-After: 1`.

The store, slot table, verification flags and link EWMA share one short reentrant lock. No HTTP or stream runs under it. Session ticket conditions release their mutex before running a turn; the only nested mutex order is shared state then ticket condition during reservation. No path waits for session ownership while holding shared state. Replay skips busy sessions, locks one session at a time, reloads its journal, and never holds the shared lock over remote HTTP. Journal saves do not update slot bindings; only the slot allocator changes those, under a SQLite unique index. This prevents a stale replay save from undoing eviction.

Before native slot operations and before releasing a lease, the proxy checks the existing `GET /slots` activity route. Failed idle verification quarantines that slot until a later successful probe, without blocking another healthy slot. If a downstream stream reader disconnects, the proxy drains the upstream completion and commits its evaluated tokens before releasing the lease. This preserves the native processing boundary; disconnect does not immediately cancel generation. Snapshot/rollback files stay protected by the session ticket and slot lease.

## Policy and trace replay

The policy uses a depth-indexed local prefill table, server rate, link EWMA, fixed overhead, KV bytes/token, recurrent-state bytes, efficiency, margin and hysteresis. The defaults are estimates from the specification; configure measured rates for the target Mac and server. Transfer EWMA measures body bytes after response headers arrive, excluding server prefill time; it is application-level throughput, not a physical link measurement. `policy.always=true` forces an offload attempt when there are new transferable tokens; offline requests still run locally.

```sh
python3 tools/hybrid-proxy/policy.py .temp/turns.jsonl --config .temp/policy.json
```

The trace contains JSON objects with `delta`, `depth`, and optional `server_factor`; the config contains `Policy` fields. Output is JSONL with the selected path and both predicted times. The tool runs no benchmarks.

## Tests and artifacts

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v
python3 -m py_compile tools/hybrid-proxy/*.py
```

The tests use ephemeral localhost ports and temporary SQLite/slot fixtures. They cover policy decisions, explicit and inferred mapping, shared-prefix forks, restart persistence, streaming, offline/lost-ack replay, 409/snapshot recovery, foreign-client 404s on all prefill session operations, 3 clients x 4 interleaved sessions, and atomic synthetic corruption rejection. See `RESULTS.md` for exact run evidence and engine limitations. Runtime databases, slots, diagnostics and test logs belong under `.temp/hybrid-b-r1-261030/`.

Build the four named correctness targets listed below under the shared flock, with CPU-only settings and `-j2`. Check exact free bytes and retain the 25 GiB start/abort floor. Do not start a build while the orchestrator's disk hold is active.

```sh
build/bin/llama-hybrid-check-append ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf ~/Werkstatt/hybrid-models/golden/prompt-512 ~/Werkstatt/hybrid-models/golden/prompt-2048 ~/Werkstatt/hybrid-models/golden/prompt-8192
python3 tools/hybrid-proxy/test_decode.py --binary build/bin/llama-server --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --golden ~/Werkstatt/hybrid-models/golden/prompt-512 --artifacts .temp/hybrid-b-r1-261030/native-http
# Add A's existing prefill binary to exercise the complete real proxy path too:
python3 tools/hybrid-proxy/test_decode.py --binary build/bin/llama-server --prefill-binary /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-hybrid-prefill --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --golden ~/Werkstatt/hybrid-models/golden/prompt-512 --artifacts .temp/hybrid-b-r1-261030/real-e2e
```

The C++ checker compares incremental append with plain local prefill and A's 64-token golden sequence. The HTTP checker independently checks the actual slot routes, both slots' native save bytes after rejection, cross-slot import from A's one-stream source, actual cache reuse, and 64-token equality for the golden prompt (EOS masked consistently on both HTTP paths). With `--prefill-binary`, it also checks the full real proxy path for two clients with two sessions each, captures actual imported deltas, checks both numerical references described below, and exercises streaming forks, foreign-client 404, offline continuation and replay after losing the server RAM store. Native layout probes are bounded to the supplied Qwen35 fixtures. These commands are not passing evidence until `RESULTS.md` records their exit status.

## Review fix round and numerical checks

See [fixes-r2-261030.md](fixes-r2-261030.md) for the current signed fix, exact commands, build evidence and finite coverage, and [fixes-r1-261030.md](fixes-r1-261030.md) for the prior round. Historical build-hold notes in RESULTS.md are superseded by those rounds.

Build targets for the real path now include `llama-server`, `llama-hybrid-check-append`, `llama-hybrid-check-guards` and `llama-hybrid-check-distributions`. All builds use CPU `-j2`, the shared flock and the exact 25 GiB free-space guard. The end-to-end harness uses two proxy slots and a third independent local-reference slot, and stops its servers by PID.

Spec v0.3.4 section 11 distinguishes exact 64-token same-partition equivalence from whole-prompt numerical closeness. The numerical checker imports each actual delta and compares it with two independent token-only local contexts. It teacher-forces the same 64 tokens into imported and whole-prompt contexts, requiring mean KL <=0.01 nats/token and top1 agreement >=95%. Free-running whole-prompt divergence is retained as information. The real harness automatically calls the numerical checker beside `--binary`, or at `--distribution-checker`.

```sh
python3 tools/hybrid/test_recurrent_append.py --library build/bin/libllama.so --malformed
build/bin/llama-hybrid-check-guards ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf ~/Werkstatt/hybrid-models/golden/prompt-512
build/bin/llama-hybrid-check-distributions ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf ~/Werkstatt/hybrid-models/golden/prompt-512.tokens.json ~/Werkstatt/hybrid-models/golden/prompt-2048.tokens.json ~/Werkstatt/hybrid-models/golden/prompt-8192.tokens.json
```

| Tool | Purpose and entry point | Artifacts |
|---|---|---|
| Python standard library | Proxy/stub CLIs and unittest commands above | SQLite, slots and logs under `.temp/` |
| CMake and C++ compiler | CPU engine and the four named targets, under the shared flock | `build/`; monitor/build logs under `.temp/` |
| Native checkers | Append, recurrent rejection, SWA guard and distribution commands above | stdout/stderr logs under `.temp/` |
| Git | Scoped signed local commits (`git commit -S`), no pushes | Local feature-branch history |

## Second fix round controls

Build the additional targets `llama-hybrid-check-client` and `llama-hybrid-check-codec` under the same lock and disk guard. The codec controls retain A80013538's valid maximum int32-token header and intended budget failures. The capacity control fills a 512-cell unified KV cache so the valid append fails after validation, then checks both sequences' native bytes and a successful retry after freeing cells.

```sh
build/bin/llama-hybrid-check-codec
build/bin/llama-hybrid-check-client ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --capacity ~/Werkstatt/hybrid-models/golden/prompt-512
python3 tools/hybrid-proxy/test_header_budget.py --binary build/bin/llama-server --prefill-binary /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-hybrid-prefill --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r6-261030/header-http
```

The HTTP header control generates one real 16,384-token state with A's existing binary, checks both header-budget rejections against saved native slot bytes, then imports the valid maximum. It stops both servers by recorded PID. The client checker also supplies the requested bounded append-cost measurement: `--prepare GOLDEN_DIR OUTPUT_DIR` creates three fixed-suffix fixtures, then `--measure STEM N0` measures one append in a fresh process and verifies source equality outside the timing interval. Exact before/after commands, RSS methodology and limits are in the round-2 report. All outputs go under `.temp/`.

If snapshot rollback fails, the proxy returns HTTP 503 and retains `SESSION-resync-backup.bin` in the configured slot directory. This file also marks that binding untrusted across a proxy restart. The next request erases the dirty slot and supplies tokens for a full prefill; the marker is removed after the decoder's evaluated tokens and the durable journal agree. Ordinary prefill-node outages still complete locally and replay on reconnect.

## M0.1 concurrency controls

See [m01-concurrency-261030.md](m01-concurrency-261030.md) for the signed candidate, lock order, exact exits, native M0 red witness and limits. The standard unittest command includes FIFO, contention, pinned eviction, the actual 30-second timeout, replay/eviction races, recovery and disconnect controls. Native controls use existing binaries; they never configure or build the engine.

```sh
python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/native-concurrency
```

The native helper starts `llama-server --parallel 2`, compares two simultaneous 64-token greedy streams with solo runs, and checks both overlapping token windows and two processing native slots. It then runs four threads, four sessions and three ordinary user turns per session through two slots, auditing tokens and prefix hashes while each lease is pinned. It records source/binary hashes, stream windows, journal checks and owned PID under its artifact directory, and stops that PID in `finally`. `--proxy-source DIRECTORY` selects an extracted read-only M0 proxy for the expected-red overlap witness.

M0.2 replaces the old assistant-continuation workaround with ordinary user turns. The audit requires every sampled ID to be evaluated and present in the durable journal before releasing the lease. This helper still checks overlapping streams, exclusive slots and parked-session restore. The append helper separately qualifies template divergence and user-boundary rewrites. Requests that cannot establish a supported message boundary, including a same-history assistant continuation with no new message, return 409 under explicit reuse.


### Bounded extends and heterogeneous quantization

`extend_chunk_tokens` limits each critical append/import. Critical chunks use `(512 MiB - recurrent_bytes - 1 MiB) // (kv_bytes_per_token + 24)`, also limited by `(1 MiB - 4096) // 11` for the JSON header. The 24 bytes are native per-cell metadata for one sequence; the 1 MiB reserve covers fixed serialization overhead. The boundary includes this reserve: one more token violates the reserved planning bound, although it can still fit the raw body cap. The launcher writes separate `delta_geometry` values from GGUF metadata; policy cost estimates do not control this safety bound. Without launcher geometry the proxy and benchmark use a conservative 27B bound (34,816 KV bytes/token, 157,000,000 recurrent bytes), safe also for 0.8B (6,528 KV bytes/token, 20,201,472 recurrent bytes). Explicit chunk sizes above the planning bound refuse startup. These are the supported 0.8B/27B geometries; other geometry needs verified `delta_geometry` before use. Each import verifies its resulting prefix. A failed middle append retains valid preceding imports and falls back through existing local/replay recovery. Benchmark traces include every attempted chunk and content-free fallback causes, including RAM budget refusals.

Live node health is checked before sync. Native bytes/checksums and decoder layout validation remain intact. Whole snapshots are not translated; ahead-state recovery uses local continuation. Exact model SHA-256 is required by default. Experimental `--hetero-quant` (`hetero_quant=true` in the proxy) admits a different SHA only after a matching layout fingerprint; the proxy then rewrites the delta envelope SHA, bypassing the native SHA guard. A layout match establishes shapes only, not base-weight identity or numerical equivalence. Quality is unverified and uncalibrated: the r11 CPU Q4_0 node / Q8_0 client measurement gave mean KL 0.0145 nats/token (above the 0.01 reference) and top-1 62/64. The Q8 file was a requantization of Q4 weights, not an independent higher-precision source. Heterogeneous benchmark results remain uncalibrated even when their measured values meet the reference. See the [benchmark guide](../hybrid-bench/README.md) for setup, recorded identities and uncalibrated accuracy checks.

With `benchmark=true`, authenticated `DELETE /v1/sessions/ID` releases that client's current and retired remote sessions and prevents their background recreation. A local-mode marker is persisted before generation, so interruption cannot enable remote replay. Ordinary responses have no benchmark fields and exactly one streaming DONE marker; default GET behavior remains HTTP 501.

## Captured chat-template qualification

The journal probes live message frames for Qwen ChatML, Gemma 4 and Muse Glimmer. It verifies the family against a live user/assistant/user render and, for Gemma and Glimmer, the template source and BOS/EOS properties. Unknown frames reject with 409. Plain token pieces must reproduce every rendered byte. One tokenizer-added BOS is accepted only when its declared single-token ID precedes exactly the plain IDs; its byte width is zero. A visible BOS cannot also be a leading special.

Normal reply boundaries are `<|im_end|>`, `<turn|>` and `<|eot|>`, respectively. Gemma tool replies continue inside a model turn at the sampled `<|tool_response>` opener. Both forms retain a single-token ending check, exact text/token suffix checks and anchor overlap. User checkpoint headers come from the same live message probes.

`fixtures/templates/*.json` are unmodified captures from real decoders at b12094-74f8f4c3a. `FixtureTemplate` in `stub_server.py` supplies text renderers for the captured shapes, default/disabled thinking and assistant continuations. Token replay accepts captured text and aligned captured spans only; it is not a model tokenizer. Renderers are checked against all 96 captured renders before their marker probes serve as test evidence. Tools with root-level definitions, arbitrary multimodal inputs and uncaptured tokenizations are outside this fixture harness.

```sh
mkdir -p .temp/template-tests/tmp
TMPDIR="$PWD/.temp/template-tests/tmp" nice -n 10 python3 -B -m unittest discover -s tools/hybrid-proxy -p test_templates.py -v
TMPDIR="$PWD/.temp/template-tests/tmp" nice -n 10 python3 -B tools/hybrid-proxy/template_controls.py --output .temp/template-tests/controls
```

The controls output directory must be new. It contains copied-source identities, owned process IDs, exact commands, logs, assertion records and `controls.json`. Identity must survive; setup errors, skipped tests and wrong assertion failures receive no kill credit. Results and finite GPU qualification gaps are in [the r19 report](RESULTS-hybrid-a-r19-templates-261101.md). Gemma drops the bench's unmatched tool response; the workload is unchanged pending the decision recorded in [UNRESOLVED_QUESTIONS.md](../../UNRESOLVED_QUESTIONS.md).

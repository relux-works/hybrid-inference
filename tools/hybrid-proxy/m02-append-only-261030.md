# M0.2 append-only sessions

Status: ready for review. Requested native, proxy and end-to-end gates passed; the three M0.1 witnesses failed for the intended full re-prefill. No push or independent acceptance is claimed.

Run `hybrid-b-r11-261030`, 2026-10-30 UTC, branch `m02-append-only-sessions`. This continues the signed r10 scope report `d079547f8f`; the user explicitly approved the two server extensions and proxy implementation. The base is landed M0.1 main `edb31a7e`. Specification: the lane's v0.3.7 section 4a and the r11 conditions. No `src/`, codec, prefill-node or A-workbench edits, new dependencies, configure steps or pushes.

## Signed implementation

| Commit | Change |
| --- | --- |
| `ef88f29066fa2e8652bd0d729410fc2425b017ff` | Two opt-in server changes and native API controls. |
| `0d0cb2bead06b4768edd6a01e292a0b3d4fe7f43` | Exact token/message journals, checked suffix rendering, checkpoint transitions, replay and metering. |
| `9c9c18e29a165fe6e5b4c0569206dfb704e61274` | Preserve unchanged generated IDs through a checkpoint rewrite; retain complete segment coverage. |
| `b20e7b0a657b95bb669d0bb6b9312c7b96656123` | Keep inferred admission on the same session between stream DONE and durable journal commit. |
| `154999a1dfd56dc5e2324c05f4d6e345e6b87bb3` | Keep the documented synthetic decoder compatible with the bridge; real-message protocol regression. |

All carry the configured host identity and signatures. The report/API-documentation commit is the subsequent branch tip. Runtime source is frozen at `b20e7b0a65`; the synthetic decoder additionally uses `154999a1df`. Each handoff conversation artifact records module paths and SHA-256 identities. `server-context.cpp` is exactly the built `ef88f29066` blob `ef7550b0b57e944490bdb6390730e1fd95812a2d`.

## Server bridge

`tools/server/server-context.cpp` changes by 17 lines. Chat still constructs `common_chat_session` from messages, tools and settings. An optional `prompt_tokens` array replaces the task's token input after validation: nonempty array, integer IDs, every ID below the vocabulary size, length no greater than slot context. Converting negative integers to unsigned places them above the vocabulary bound. Validation finishes before task submission. Calls without the field take the existing path.

For `TASK_RESPONSE_TYPE_OAI_CHAT`, a stream with both `return_tokens=true` and `verbose=true` retains the complete `slot.generated_tokens` vector in its final `__verbose.tokens`. `process_token` already records IDs before stop/EOG handling and UTF-8 buffering, so the final vector includes suppressed and buffered output. Other stream shapes remain unchanged. The proxy consumes the native data and forwards normal OpenAI reasoning, content and tool-call deltas.

The upstream completion guard reserves one token even for `n_predict=0`: an exactly full context is rejected with unchanged slot info. This existing behavior is retained; the tested usable maximum is `n_ctx-1`. The bridge is for text-only nodes. [Server options](../server/README.md) document both extensions.

## Two journals and admission

SQLite adds one nullable JSON `journal` column; existing token journals and client ownership survive migration. Completed rows contain:

- Exact evaluated token IDs, block hashes and native prefix hash; message identities with client-omitted reasoning restored from otherwise matching recorded assistant messages.
- Incoming and generated segments, original sampled ID vectors, and replay annotations. Checkpoint clipping preserves full token coverage without gaps or duplicate spans.
- Every message boundary as a token index plus a byte offset within that token. This represents BPE pieces crossing text boundaries without inventing a token boundary.
- Evaluated and sampled boundaries, template/render-settings identity, complete request-settings hash, user checkpoint candidates, per-request meters and retired remote IDs.

`return_tokens` plus final verbose data is the generated-ID source. Slot info independently verifies the evaluated prefix. The last sampled token is normally not evaluated by llama-server; the proxy evaluates that exact ID through existing `/completion` with `n_predict=0`, then checks native tokens/hash again. Both journals commit in one SQLite transaction only after this reconciliation. Intermediate turn saves do not replace the last completed pair.

A transient, client-scoped admission map lets an inferred follow-up find its session after stream DONE but before the first journal commit. It is used only for admission, never durable replay. A longer completed message match wins over a shorter pending prefix. After its FIFO ticket is acquired, a request reloads the completed journal and resolves an append, rewrite or fork. Unknown native extensions after an interrupted commit are discarded rather than adopted into completed message history.

An absent assistant `reasoning_content` is ignored only for an otherwise matching assistant message. Supplied conflicting reasoning, user edits, tool-output edits and tool-call identity changes are real divergences. JSON tool arguments compare as values. Template, tool-schema and render-setting changes do not silently reuse an incompatible journal. Tool selection can change per turn; native grammar remains responsible for the new generation.

## Guarantee for new-message tokens

The proxy uses the active server template and tokenizer, not a Python Jinja replacement or a tool parser. For an append it replaces the preceding assistant's content with a fresh nonce in a probe render, then takes the suffix after that nonce. It accepts that suffix only when the real full render ends with exactly the same bytes and encoded IDs. ChatML closing framing is checked explicitly. An already sampled identical `im_end` token is not appended twice.

Only the accepted suffix is joined to the old authoritative IDs. Old generated IDs are never reconstructed from returned text. Boundary probes locate actual message frames; they do not search untrusted user text for a delimiter. Token pieces must reconstruct the real rendered bytes and IDs. A separate render without the generation prompt establishes that prompt's boundary. Ambiguous or unsupported frames fail explicitly.

The qualified families are the supplied Qwen3.5 and Qwen3.8 ChatML templates. The native test's independent new-token oracle spells their user/tool/generation frames directly from the supplied template files. It does not call the proxy renderer. A separate solo slot repeats the actual per-turn partition and final-generated-token evaluation. Every final answer in the three-user tool conversation compares 64 greedy IDs; tool-call generations compare their entire sampled vector and parsed calls.

## Checkpoints, forks and recovery

Upstream native checkpoints cover hybrid/recurrent slots: user spans split prefill batches, partial state is captured before the next batch, and a later request can restore a prior checkpoint. Enable `--ctx-checkpoints 32 --checkpoint-min-step 0`. The proxy no longer erases a planned mismatch before native rollback can run.

For a genuine explicit rewrite, choose the latest recorded user boundary at or before message divergence. Save a rollback file, submit a zero-generation trial ending just after that user header, and require actual native `cache_n` to equal the chosen checkpoint. Verify trial IDs/hash. If unavailable, restore the backup and return409. A rollback failure retains its backup, revokes verification and invalidates the slot binding; even a backup-rename failure leaves a recognized marker. A later lease clears untrusted state.

Where an unchanged generated turn follows the checkpoint but precedes the edit, replay its exact old IDs through its last sampled boundary. Render only the changed suffix after that anchor. This preserves old reasoning and noncanonical tokenization even when the client omitted reasoning. Prefix/replayed segments are clipped at the checkpoint while retaining their original sampled IDs. The native compaction test checks the actual old generated prefix, not a second rendering of it.

An inferred divergence creates a child session. If a same-client parent has a checkpoint candidate, copy its existing native slot file while its session/slot are pinned, release the parent, then lease a child slot and validate native rollback there. The parent journal/state stays intact. If the copied checkpoint is unusable, reset only the child and take the documented fresh-session full-prefill fallback. This also works with fewer slots than sessions. Explicit reuse without a supported transition returns409, including the former legacy 400 mismatch path.

Only rare rewrites/forks save a whole-slot rollback/copy file for the journal transition. Ordinary resident appends do not; existing LRU parking can still save slot files. Native import clears checkpoints, so imported state is not advertised as carrying user checkpoints. Candidate positions are validated on use; eviction, restore and unavailable checkpoints are treated explicitly.

A's current API cannot restore/fork a historical checkpoint. A successful local rewrite therefore creates a fresh remote session and replays its chosen prefix. A future API would need a client-scoped checkpoint identity, model, position/hash and atomic ordered restore/fork. No shared-contract change was made. Old remote state is retired only after the new journal commits; bounded replay retries deletion, including after an outage. The real-node control verifies fresh replay, old-session404 and intact parent state on an inferred fork.

Lock order remains shared metadata then ticket-condition during reservation; no code waits for session ownership while holding shared metadata. HTTP, template work, backup files, streaming and replay run under only their session ticket and pinned slot lease. An inferred fork never holds two session tickets or slot leases while waiting for the child. Replay skips active/queued sessions. The pending admission map is removed before the session ticket is released.

## Metering

`prefilled = local_evaluated + remote_applied`. Ordinary appends require this to equal `new_tokens`; the tests compare the count and every appended ID with the independent oracle. New tokens include template framing and the new generation prefix. `generated_evaluated` records the final sampled-token evaluation separately. A rewrite additionally reports `replayed_prefix`; its prefill includes that replay plus the changed suffix. Logs identify session, local/remote/generated counts and rollback position.

Critical and shadow acknowledgement logs include actual `n0,n1` ranges. `remote_replay_acknowledged` comes from GET-confirmed progress, never from assuming a 202 means applied. A remote fresh-session prefix replay is separate from newly offloaded prompt work. Lost acknowledgements remain bounded by the existing prefix/hash protocol.

## Qualification and known-violation controls

The final gate table and command ledger below are generated from actual process exit records. Initial iterations are retained separately; a nonzero result is never relabelled green.

| Surface | Evidence |
| --- | --- |
| Server positive | Rendered tokens versus message input: identical sampled IDs, content, reasoning and tool functions/arguments. Native-generated transport tool IDs are random, so that one field is normalized while requiring nonempty IDs. Streamed final IDs equal nonstream IDs for reasoning, a real tool call, stop-string suppression and buffered Unicode/EOG. A distinct valid array proves the field is used. Highest vocabulary ID and usable context boundary accepted. |
| Server negative/legacy | Ten malformed cases: string, null, empty, bool, float, negative, minimum signed integer, vocabulary boundary, unsigned overflow and excessive length. Each400 preserves info and complete native bytes. Legacy requests before/after build have identical sampled/parsed output and native slot-file bytes. |
| r10 test1 | Old template, thinking on/off, three real user/tool loops. Every request prefills only the new framed input; all three final answers are greedy64/64 against independent solo contexts. |
| r10 tests2-3 | Target 27B template on 0.8B weights, reasoning returned and omitted. Same session, exact appended IDs and greedy64/64; stored old reasoning survives later omitted echoes. |
| r10 tests4-5 | Dropped second tool result, earlier user edit, checkpoint-backed inferred compaction, unavailable checkpoint, native prefix/hash, segment coverage and neighbor isolation. Foreign explicit client404. A controlled stream-DONE/commit gap proves inferred FIFO reuse. |
| r10 test6 | All original38 tests retained; former mismatch status assertion updated400->409 as required. Added journal/stub controls plus native concurrency and the real2x2 section11 run. |
| r10 test7 | Three M0.1 witness processes exit1 with the named full re-prefill assertion. Later user requests must have `prefilled == full_prompt_length > new_message_tokens`; unrelated setup errors do not satisfy the witness. |
| r10 test8 | Actual prefilled-token totals are tabulated below, with generated-token reconciliation separated. |
| Real remote transitions | Local checkpoint rewrite, fresh remote prefix replay and retired-state deletion; subsequent delta append has local1+remote28=new29. Explicit no-checkpoint409 preserves native bytes; inferred fork uses a fresh remote and leaves the parent unchanged. |
| Regression state safety | Atomic dual-journal staging, missing reasoning versus conflicting supplied reasoning, edited user/tool content, nonce collision, probe-dependent render, tokenizer-boundary mismatch, segment clipping, SQLite migration/reopen, remote retirement outage, failed rewrite rollback plus failed rename, unknown native extension, external token injection and pending admission ownership/ranking. |

The conversation fixture uses a 16-token reasoning budget and disables parallel tool calls to obtain bounded real tool loops on the small weights. Tool calls are genuinely generated and native-parsed, not injected text. The same fixture is used for M0.1 and M0.2. M0.1 modules are extracted read-only from `edb31a7e`; both versions use the same rebuilt native binary. The server opt-ins expose capture metadata without changing their generated IDs, and the separate legacy byte control checks the unchanged request path.

The native concurrency helper retains its overlap/solo, four-thread/four-session/two-slot and all16 per-request audits. It now uses ordinary user turns instead of the old assistant-continuation workaround, enables checkpoints and audits the complete sampled journal after reconciliation. The same-session FIFO/unit tests keep their original names. New failures are covered by named executable controls rather than silently relaxing assertions.

## Limits

- CPU and the single supplied Qwen3.5-0.8B-Q4_0 model only. The 27B template was applied to those small weights; no other model was downloaded. CUDA, Metal, 27B weights, large-depth costs and hardware link rates remain unqualified.
- Qualified incremental framing is Qwen/ChatML text. Multimodal inputs, multiple choices, Anthropic Messages, other template framing and same-history assistant continuation without a new message are unsupported. Explicit reuse without a supported transition returns409. The synthetic node supports simple message/protocol controls; it is not an inference or checkpoint-equivalence substitute.
- Slot loss can require recovery prefill. Rare full rollback/fork files have not been cost-qualified at 110k tokens. Native checkpoints can be evicted or unavailable and are checked before accepting rollback.
- SQLite commits the token/message pair atomically; it is not an atomic transaction with another process's decoder memory. Process-crash transactions, native crash during rollback, arbitrary corrupted local SQLite/files and cross-process ownership of a slot pool are not qualified. Synthetic rollback-outage controls are distinguished from real native byte controls.
- The prefill node still has no historical checkpoint operation; remote prefix replay can be substantial. Imported state clears user checkpoints. Session/queue/journal/park-file TTL remains future work.
- Upstream `ty` and `editorconfig-checker` executables are absent (readiness exit127 each); no installation was attempted. Python syntax, local file-format checks and diff whitespace ran. No hosted check or independent review is claimed.
- The brief has no surface table. This is an explicit coverage-map gap, not an assertion of universal coverage. All requested r10/r11 rows are accounted for above; no requirement was silently treated as out of contract. The rework stays inside `tools/server` and `tools/hybrid-proxy`.

The original r10 blocker and red prerequisite remain in signed history at `d079547f`. Its capability probe is now updated to recognize the approved bridge; the full M0.1 conversation red logs remain separate witnesses.

## Build and source identities

There were two incremental attempts, both only `llama-server`, CPU `-j2`, under the shared `flock` and exact-byte disk monitor. No configure or engine/checker rebuild ran. Attempt 1 exited2 because this checkout's JSON wrapper lacks `is_number_unsigned`; the final validator uses its existing integer predicate and unsigned conversion. Attempt 2 exited0. The cache is Release, GGML_NATIVE=ON, GGML_CUDA=OFF; the existing build configuration was retained.

The successful attempt started with 40,057,217,024 available bytes; its minimum observation was 40,054,181,888. The abort floor was exactly25GiB = 26,843,545,600 bytes. The resulting build tree occupies 86,166,193 bytes. A later delivery observation was 28,864,172,032 bytes; no later build was attempted. The shared host's free space can change independently.

| Object | SHA-256 |
| --- | --- |
| Before `build/bin/llama-server` | `b88d7a41a493de8fb1089218954c4fd0c953d6ee0de66f228c8b7d9837340a9e` |
| After `build/bin/llama-server` | `acdf3fc833c655cc788e049fbe846e925fd68479a4384e5234644acb97a69a7c` |
| After `build/bin/libllama-server-impl.so` | `d567ac3b58a6f7556aaf24e79e518cb3e69d8d8b80ff64e28e029f9b4ffad81d` |
| After `build/bin/libllama.so` | `1ad5d26ba8388d05ca87d8872dd6dcbc104694c24b38dff93ee7ae44844c8865` |
| After `build/bin/llama-hybrid-prefill` | `5af5b1901d918b9db7eb6d5b94fb79d238f625ee3cb51b660f4c64f40de80240` |
| After `build/bin/llama-hybrid-check-distributions` | `ce6de8ea0b6d2fd37511f903adea0b66d9037c9fccb27878a6e1326f61fa4814` |

The server was built while HEAD still described the r10 report, with the reviewed17-line source patch in the worktree. Its displayed build fingerprint therefore contains `d079547f8f`; that is not the full source identity. The compiled server-context blob is `ef7550b0b57e944490bdb6390730e1fd95812a2d`, identical at `ef88f29066` and the delivered head. `ldd` resolves the implementation/core/ggml libraries to B's build directory. The core library SHA is unchanged. Existing B-owned prefill and distribution binaries were reused read-only.

The pre-build implementation-library hash was not captured. The actual legacy request/slot-byte control was executed before rebuilding, and those bytes are retained. The copied pre-build launcher now resolves the updated dynamic implementation and is not presented as an old implementation rerun.

Model SHA-256: `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`. The supplied 27B template, revision1d4bf0f2, hashes to `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`. The supplied old-template file hashes to `7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67`; the old-template native fixture uses the GGUF default. Each native artifact has its exact server argv and model/binary identity. Conversation helpers use three4096-token slots, CPU2 threads and batch/ubatch256, checkpoint32/min-step0. The existing section11 helper keeps its independently specified checkpoint-free same-partition fixture.

## Prefilled-token totals and rollback observations

These totals cover the six requests of the three-user-turn conversation, not the subsequent compaction/isolation controls. The initial system/tools/user prompt is included. All six M0.2 requests assert actual `prefilled == expected_new` and exact appended IDs. Each M0.2 conversation additionally evaluates six final sampled tokens, recorded separately from prompt prefill.

| Template / reasoning echo | M0.1 actual prompt prefill | M0.2 actual prompt prefill | Independent new-input oracle | M0.2 generation reconciliation |
| --- | ---: | ---: | ---: | ---: |
| Old GGUF, thinking on | 1530 | 524 | 524 | 6 |
| Old GGUF, thinking off | 2102 | 536 | 536 | 6 |
| 27B template, reasoning omitted | 3264 | 562 | 562 | 6 |
| 27B template, reasoning returned | not requested | 562 | 562 | 6 |

The old proxy also re-evaluates a held generation tail in later requests; it has no separate M0.2 meter. The table preserves its observed prompt-evaluation total rather than subtracting guessed work. Each witness asserts the later user turns require the entire newly rendered prompt, with new sessions where their prefix diverges.

The dropped-second-tool-result checkpoint trials produced:

| Case | Actual rollback n | Replayed original IDs | Changed suffix | Actual prefilled |
| --- | ---: | ---: | ---: | ---: |
| old-on | 469 | 76 | 224 | 300 |
| old-off | 455 | 61 | 262 | 323 |
| target-returned | 507 | 76 | 284 | 360 |
| target-omitted | 507 | 76 | 236 | 312 |

These are nonzero user boundaries at/before divergence. Every resulting native prefix/hash is checked, and unchanged generated spans through the post-checkpoint anchor retain their original IDs. The target-omitted case intentionally differs in its changed suffix because previously omitted client reasoning is not treated as a newly supplied edit.

## Final gate results

Commands run from `/home/agent/Werkstatt/hybrid-b/llama.cpp`. All logs/artifacts are below `.temp/hybrid-b-r11-261030/`. The table lists the final source-bound acceptance runs; the complete chronological command ledger follows. Each command is a standalone process, its stdout/stderr goes directly to its named log, and `run.py` preserves its real exit code without a pipe or tee.

| Gate / artifact label | Exit | What the assertions establish |
| --- | ---: | --- |
| `server-api-qualified` | 0 | Opt-in API equivalence, malformed native-byte preservation, stop/EOG/UTF-8 IDs and pre-build legacy byte control. |
| `matrix-old-on-handoff` | 0 | Old template thinking on; tool loops, solo exact64, rewrites/isolation and inferred commit-gap control. |
| `matrix-old-off-handoff` | 0 | Old template thinking off, inferred sessions; same checks. |
| `matrix-target-returned-handoff` | 0 | Target template, supplied reasoning; same checks. |
| `matrix-target-omitted-handoff` | 0 | Target template, omitted reasoning, inferred sessions; same checks. |
| `matrix-no-checkpoint-handoff` | 0 | Explicit unsupported rewrite409 restores native bytes; inferred child takes documented fork. |
| `witness-m01-old-on-handoff` | 1 | Expected red: later user full re-prefill, not a setup failure. |
| `witness-m01-old-off-handoff` | 1 | Expected red: later user full re-prefill, not a setup failure. |
| `witness-m01-target-omitted-handoff` | 1 | Expected red: later user full re-prefill, not a setup failure. |
| `matrix-handoff` | 0 | Driver requires the five candidate0 and three witness1 exits and completed artifacts. |
| `regressions-delivery` | 0 | All55 proxy tests: original38 retained plus17 journal/bridge controls. |
| `concurrency-native-handoff` | 0 | Two solo64/64 comparisons, simultaneous native processing, 2.267774s overlap, all16 contention token/hash audits. |
| `remote-append-handoff` | 0 | Real checkpoint/fresh replay, remote retirement404, actual remote28+local1=new29 and no-checkpoint transitions. |
| `e2e-section11-handoff` | 0 | Real2clients x2sessions;16 malformed imports byte-atomic; one-token endpoints, portable restore, offline/restart replay and numeric checks. |
| `capability-probe-final` | 0 | Both extended API capabilities detected on the built binary. |
| `syntax-delivery` | 0 | Changed Python compiles. |
| `delivery-check-02` | 0 | Final module/test hashes, scope/format, source blob, gate assertions and65 recorded PIDs absent. |
| `diff-whitespace-delivery` | 0 | Diff has no whitespace errors. |

Section11's distribution checker exited0 (invoked by the end-to-end process, with full child log retained):

| Conversation, each of clients a and b | Same-partition greedy | Whole-prompt teacher-forced mean KL | Teacher-forced top1 | Whole-prompt free-running observation |
| --- | --- | ---: | --- | --- |
| Counting | 64/64 | 8.67437e-15 | 64/64 | 64/64 |
| Colors | 64/64 | 0.00203662 | 63/64 | 38/64; first divergence38 |

Teacher-forced bounds and exact same-partition checks pass. Whole-prompt free-running equality is not claimed for colors. KL-only and top1-only violation controls are rejected by the checker. The separate real-remote multi-turn test establishes journal/range/rollback behavior, not a new multi-turn KL qualification.

| Known violation / control | Narrowed property | Named gate that rejects it | Bound |
| --- | --- | --- | --- |
| M0.1 old template, on/off | New-user input must append without re-rendering history | `test_append_sessions.py --witness` / old-on and old-off handoff logs | Both exit1 for the full-prefill assertion; no claim for arbitrary templates. |
| M0.1 target template, omitted reasoning | Missing reasoning must retain old exact IDs | `test_append_sessions.py --witness` / target-omitted handoff log | Exit1 for full prefill on the same small model/template fixture. |
| KL-only and top1-only violations | Numeric checker must enforce each independent bound | `check-distributions` controls in `distribution-check.log` | Both known violations rejected; not arbitrary numerical corruption. |

No surviving narrowing mutant was used to justify a passing gate. Product negative controls additionally include wrong types/ranges, reasoning conflicts, user/tool rewrites, foreign ownership, unavailable checkpoints, rollback/rename failure, transient admission and nonce/tokenizer ambiguity.

Initial nonzero iterations remain in the ledger: build-server-01=2 (JSON-wrapper predicate), server-api-01=1 (random transport tool ID compared too strictly), append-old-off-01=1 (tool-injected system layout not yet supported), append-target-returned-01=1 (unbounded thinking exhausted the tool-generation budget), server-api-final=1 (upstream reserve-one context guard), journal-unit-final=1 (volatile stub access timestamp compared as state), delivery-check-01=1 (audit's expected witness text differed from its actual wording). Each was corrected and rerun with the same or stronger intended invariant; none counts as a passing gate. The three deliberate witness failures retain exit1 throughout.

## Cleanup, resources and self-review

`delivery-check-02` reads every recorded fixture server/prefill PID, the section11 PID lists and both CMake PIDs. All65 distinct PIDs are absent; `cleanup-readback.json` retains their origins. Fixtures stop their own processes by PID in `finally`. No pattern kill or foreign-process stop was used.

Observed maximum recorded native fixture high-water RSS was 2,041,444KiB for a server, and 532,996KiB for a Python driver. These are per-process observations from `/proc`, not an aggregate run peak or performance benchmark. Build aggregate RSS was not measured. Contexts remain bounded and only the small model was loaded.

Self-review applied the repository code-review skill to the changed scope and checked parser/task validation order, stop/EOG capture, message ownership, suffix provenance, backup failure paths and session/slot lock ordering. It found and fixed two additional failures before handoff: re-tokenization of an unchanged generated segment after checkpoint rollback, and inferred-session loss in the stream-DONE/first-commit gap. Their signed fixes and native/pure regressions are listed above. Review by A and the orchestrator is still pending. README documents tool entry points, state privacy, run configuration and finite qualification limits; unresolved items are listed in `UNRESOLVED_QUESTIONS.md`.

## Exact chronological command ledger

Artifact paths in commands are the actual qualification paths. Re-running a fixture requires a new `--artifacts` directory to avoid overwriting its retained evidence. For witnesses, extract the four M0.1 modules using `git show edb31a7e:tools/hybrid-proxy/NAME.py` for NAME=proxy,store,policy,concurrency into the `--proxy-source` directory. This changes no workbench. Native helpers record their full server argv in `identity.json`; remote/end-to-end helpers also record prefill argv and PID lists. `run_matrix_handoff.py` only sequences the individually listed commands and checks their expected exits.

The ephemeral wrapper `python3 .temp/hybrid-b-r11-261030/run.py LABEL COMMAND...` redirects COMMAND directly to LABEL.log and appends its actual result to gates.jsonl. The exact COMMAND argv is reproduced here, with UTC start and real exit:

`legacy-before-01`; 2026-10-30T15:23:24Z; exit **0**; log `legacy-before-01.log`.

```sh
python3 tools/hybrid-proxy/test_append_api.py --binary .temp/hybrid-b-r11-261030/llama-server-before --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/legacy-before-01 --legacy-only
```

`build-server-01`; 2026-10-30T15:23:43Z; exit **2**; log `build-server-01.log`.

```sh
flock /home/agent/Werkstatt/hybrid-models/.build.lock python3 .temp/hybrid-b-r1-261030/build_guard.py --minimum-start-gib 25 --abort-gib 25 --log-prefix .temp/hybrid-b-r11-261030/build-server-01 -- /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-server
```

`build-server-02`; 2026-10-30T15:26:47Z; exit **0**; log `build-server-02.log`.

```sh
flock /home/agent/Werkstatt/hybrid-models/.build.lock python3 .temp/hybrid-b-r1-261030/build_guard.py --minimum-start-gib 25 --abort-gib 25 --log-prefix .temp/hybrid-b-r11-261030/build-server-02 -- /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-server
```

`server-api-01`; 2026-10-30T15:30:00Z; exit **1**; log `server-api-01.log`.

```sh
python3 tools/hybrid-proxy/test_append_api.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/server-api-01 --baseline .temp/hybrid-b-r11-261030/legacy-before-01
```

`server-api-02`; 2026-10-30T15:30:55Z; exit **0**; log `server-api-02.log`.

```sh
python3 tools/hybrid-proxy/test_append_api.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/server-api-02 --baseline .temp/hybrid-b-r11-261030/legacy-before-01
```

`regressions-01`; 2026-10-30T15:33:27Z; exit **0**; log `regressions-01.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v
```

`journal-unit-01`; 2026-10-30T15:34:43Z; exit **0**; log `journal-unit-01.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p test_journal.py -v
```

`append-old-on-01`; 2026-10-30T15:38:31Z; exit **0**; log `append-old-on-01.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/append-old-on-01 --thinking on
```

`append-old-off-01`; 2026-10-30T15:44:03Z; exit **1**; log `append-old-off-01.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/append-old-off-01 --thinking off --inferred
```

`journal-unit-02`; 2026-10-30T15:46:15Z; exit **0**; log `journal-unit-02.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p test_journal.py -v
```

`append-old-off-02`; 2026-10-30T15:46:42Z; exit **0**; log `append-old-off-02.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/append-old-off-02 --thinking off --inferred
```

`journal-unit-03`; 2026-10-30T15:47:47Z; exit **0**; log `journal-unit-03.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p test_journal.py -v
```

`server-api-03`; 2026-10-30T15:48:40Z; exit **0**; log `server-api-03.log`.

```sh
python3 tools/hybrid-proxy/test_append_api.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/server-api-03 --baseline .temp/hybrid-b-r11-261030/legacy-before-01
```

`append-target-returned-01`; 2026-10-30T15:49:38Z; exit **1**; log `append-target-returned-01.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --artifacts .temp/hybrid-b-r11-261030/append-target-returned-01 --thinking on
```

`append-target-returned-02`; 2026-10-30T15:50:28Z; exit **0**; log `append-target-returned-02.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --artifacts .temp/hybrid-b-r11-261030/append-target-returned-02 --thinking on
```

`server-api-final`; 2026-10-30T15:52:18Z; exit **1**; log `server-api-final.log`.

```sh
python3 tools/hybrid-proxy/test_append_api.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/server-api-final --baseline .temp/hybrid-b-r11-261030/legacy-before-01
```

`journal-unit-04`; 2026-10-30T15:56:06Z; exit **0**; log `journal-unit-04.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p test_journal.py -v
```

`server-api-final-02`; 2026-10-30T15:56:59Z; exit **0**; log `server-api-final-02.log`.

```sh
python3 tools/hybrid-proxy/test_append_api.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/server-api-final-02 --baseline .temp/hybrid-b-r11-261030/legacy-before-01
```

`append-old-on-final`; 2026-10-30T16:01:38Z; exit **0**; log `append-old-on-final.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/append-old-on-final --thinking on
```

`journal-unit-final`; 2026-10-30T16:05:12Z; exit **1**; log `journal-unit-final.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p test_journal.py -v
```

`matrix-old-on`; 2026-10-30T16:06:41Z; exit **0**; log `matrix-old-on.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-old-on --thinking on
```

`matrix-old-off`; 2026-10-30T16:07:12Z; exit **0**; log `matrix-old-off.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-old-off --thinking off --inferred
```

`matrix-target-returned`; 2026-10-30T16:07:43Z; exit **0**; log `matrix-target-returned.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-target-returned --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --thinking on
```

`matrix-target-omitted`; 2026-10-30T16:08:16Z; exit **0**; log `matrix-target-omitted.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-target-omitted --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --thinking on --omit-reasoning --inferred
```

`matrix-no-checkpoint`; 2026-10-30T16:08:49Z; exit **0**; log `matrix-no-checkpoint.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-no-checkpoint --thinking off --no-checkpoint-control
```

`witness-m01-old-on`; 2026-10-30T16:08:58Z; exit **1**; log `witness-m01-old-on.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/witness-m01-old-on --thinking on --proxy-source /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/m01-proxy --witness
```

`witness-m01-old-off`; 2026-10-30T16:09:12Z; exit **1**; log `witness-m01-old-off.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/witness-m01-old-off --thinking off --proxy-source /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/m01-proxy --witness
```

`journal-unit-final-02`; 2026-10-30T16:09:23Z; exit **0**; log `journal-unit-final-02.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p test_journal.py -v
```

`witness-m01-target-omitted`; 2026-10-30T16:09:26Z; exit **1**; log `witness-m01-target-omitted.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/witness-m01-target-omitted --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --thinking on --omit-reasoning --proxy-source /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/m01-proxy --witness
```

`matrix-01`; 2026-10-30T16:06:41Z; exit **0**; log `matrix-01.log`.

```sh
python3 .temp/hybrid-b-r11-261030/run_matrix.py
```

`remote-append-01`; 2026-10-30T16:12:58Z; exit **0**; log `remote-append-01.log`.

```sh
python3 tools/hybrid-proxy/test_remote_append.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/remote-append-01
```

`regressions-final`; 2026-10-30T16:17:53Z; exit **0**; log `regressions-final.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v
```

`matrix-old-on-final`; 2026-10-30T16:19:35Z; exit **0**; log `matrix-old-on-final.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-old-on-final --thinking on
```

`matrix-old-off-final`; 2026-10-30T16:20:06Z; exit **0**; log `matrix-old-off-final.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-old-off-final --thinking off --inferred
```

`matrix-target-returned-final`; 2026-10-30T16:20:36Z; exit **0**; log `matrix-target-returned-final.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-target-returned-final --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --thinking on
```

`matrix-target-omitted-final`; 2026-10-30T16:21:09Z; exit **0**; log `matrix-target-omitted-final.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-target-omitted-final --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --thinking on --omit-reasoning --inferred
```

`matrix-no-checkpoint-final`; 2026-10-30T16:21:45Z; exit **0**; log `matrix-no-checkpoint-final.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-no-checkpoint-final --thinking off --no-checkpoint-control
```

`witness-m01-old-on-final`; 2026-10-30T16:21:53Z; exit **1**; log `witness-m01-old-on-final.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/witness-m01-old-on-final --thinking on --proxy-source /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/m01-proxy --witness
```

`witness-m01-old-off-final`; 2026-10-30T16:22:08Z; exit **1**; log `witness-m01-old-off-final.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/witness-m01-old-off-final --thinking off --proxy-source /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/m01-proxy --witness
```

`witness-m01-target-omitted-final`; 2026-10-30T16:22:23Z; exit **1**; log `witness-m01-target-omitted-final.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/witness-m01-target-omitted-final --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --thinking on --omit-reasoning --proxy-source /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/m01-proxy --witness
```

`matrix-final`; 2026-10-30T16:19:35Z; exit **0**; log `matrix-final.log`.

```sh
python3 .temp/hybrid-b-r11-261030/run_matrix_final.py
```

`syntax-final`; 2026-10-30T16:23:50Z; exit **0**; log `syntax-final.log`.

```sh
python3 -m py_compile tools/hybrid-proxy/proxy.py tools/hybrid-proxy/store.py tools/hybrid-proxy/journal.py tools/hybrid-proxy/native_fixture.py tools/hybrid-proxy/test_append_api.py tools/hybrid-proxy/test_append_sessions.py tools/hybrid-proxy/test_remote_append.py tools/hybrid-proxy/test_journal.py tools/hybrid-proxy/probe_append_api.py tools/hybrid-proxy/test_concurrency_native.py
```

`concurrency-native-final`; 2026-10-30T16:24:13Z; exit **0**; log `concurrency-native-final.log`.

```sh
python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/concurrency-native-final
```

`e2e-section11-final`; 2026-10-30T16:27:15Z; exit **0**; log `e2e-section11-final.log`.

```sh
python3 tools/hybrid-proxy/test_decode.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --golden /home/agent/Werkstatt/hybrid-models/golden/prompt-512 --artifacts .temp/hybrid-b-r11-261030/e2e-section11-final
```

`capability-probe-final`; 2026-10-30T16:30:35Z; exit **0**; log `capability-probe-final.log`.

```sh
python3 tools/hybrid-proxy/probe_append_api.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/capability-probe-final --require-chat-token-input
```

`server-api-qualified`; 2026-10-30T16:32:46Z; exit **0**; log `server-api-qualified.log`.

```sh
python3 tools/hybrid-proxy/test_append_api.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/server-api-qualified --baseline .temp/hybrid-b-r11-261030/legacy-before-01
```

`remote-append-qualified`; 2026-10-30T16:38:23Z; exit **0**; log `remote-append-qualified.log`.

```sh
python3 tools/hybrid-proxy/test_remote_append.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/remote-append-qualified
```

`regressions-qualified`; 2026-10-30T16:38:07Z; exit **0**; log `regressions-qualified.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v
```

`rewrite-prefix-qualified`; 2026-10-30T16:45:01Z; exit **0**; log `rewrite-prefix-qualified.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --artifacts .temp/hybrid-b-r11-261030/rewrite-prefix-qualified --thinking on --omit-reasoning --inferred
```

`journal-rewrite-qualified`; 2026-10-30T16:47:04Z; exit **0**; log `journal-rewrite-qualified.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p test_journal.py -v
```

`rewrite-segments-qualified`; 2026-10-30T16:51:09Z; exit **0**; log `rewrite-segments-qualified.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --artifacts .temp/hybrid-b-r11-261030/rewrite-segments-qualified --thinking on --omit-reasoning --inferred
```

`matrix-old-on-release`; 2026-10-30T16:54:56Z; exit **0**; log `matrix-old-on-release.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-old-on-release --thinking on
```

`matrix-old-off-release`; 2026-10-30T16:55:28Z; exit **0**; log `matrix-old-off-release.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-old-off-release --thinking off --inferred
```

`regressions-release`; 2026-10-30T16:54:55Z; exit **0**; log `regressions-release.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v
```

`matrix-target-returned-release`; 2026-10-30T16:55:59Z; exit **0**; log `matrix-target-returned-release.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-target-returned-release --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --thinking on
```

`matrix-target-omitted-release`; 2026-10-30T16:56:34Z; exit **0**; log `matrix-target-omitted-release.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-target-omitted-release --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --thinking on --omit-reasoning --inferred
```

`matrix-no-checkpoint-release`; 2026-10-30T16:57:15Z; exit **0**; log `matrix-no-checkpoint-release.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-no-checkpoint-release --thinking off --no-checkpoint-control
```

`witness-m01-old-on-release`; 2026-10-30T16:57:24Z; exit **1**; log `witness-m01-old-on-release.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/witness-m01-old-on-release --thinking on --proxy-source /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/m01-proxy --witness
```

`witness-m01-old-off-release`; 2026-10-30T16:57:38Z; exit **1**; log `witness-m01-old-off-release.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/witness-m01-old-off-release --thinking off --proxy-source /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/m01-proxy --witness
```

`witness-m01-target-omitted-release`; 2026-10-30T16:57:53Z; exit **1**; log `witness-m01-target-omitted-release.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/witness-m01-target-omitted-release --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --thinking on --omit-reasoning --proxy-source /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/m01-proxy --witness
```

`matrix-release`; 2026-10-30T16:54:56Z; exit **0**; log `matrix-release.log`.

```sh
python3 .temp/hybrid-b-r11-261030/run_matrix_release.py
```

`inflight-unit-qualified`; 2026-10-30T17:03:14Z; exit **0**; log `inflight-unit-qualified.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p test_journal.py -v
```

`commit-window-qualified`; 2026-10-30T17:03:13Z; exit **0**; log `commit-window-qualified.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --artifacts .temp/hybrid-b-r11-261030/commit-window-qualified --thinking on --omit-reasoning --inferred
```

`matrix-old-on-handoff`; 2026-10-30T17:07:16Z; exit **0**; log `matrix-old-on-handoff.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-old-on-handoff --thinking on
```

`matrix-old-off-handoff`; 2026-10-30T17:07:49Z; exit **0**; log `matrix-old-off-handoff.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-old-off-handoff --thinking off --inferred
```

`regressions-handoff`; 2026-10-30T17:07:15Z; exit **0**; log `regressions-handoff.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v
```

`matrix-target-returned-handoff`; 2026-10-30T17:08:22Z; exit **0**; log `matrix-target-returned-handoff.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-target-returned-handoff --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --thinking on
```

`matrix-target-omitted-handoff`; 2026-10-30T17:08:58Z; exit **0**; log `matrix-target-omitted-handoff.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-target-omitted-handoff --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --thinking on --omit-reasoning --inferred
```

`matrix-no-checkpoint-handoff`; 2026-10-30T17:09:33Z; exit **0**; log `matrix-no-checkpoint-handoff.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/matrix-no-checkpoint-handoff --thinking off --no-checkpoint-control
```

`witness-m01-old-on-handoff`; 2026-10-30T17:09:41Z; exit **1**; log `witness-m01-old-on-handoff.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/witness-m01-old-on-handoff --thinking on --proxy-source /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/m01-proxy --witness
```

`witness-m01-old-off-handoff`; 2026-10-30T17:09:56Z; exit **1**; log `witness-m01-old-off-handoff.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/witness-m01-old-off-handoff --thinking off --proxy-source /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/m01-proxy --witness
```

`witness-m01-target-omitted-handoff`; 2026-10-30T17:10:11Z; exit **1**; log `witness-m01-target-omitted-handoff.log`.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/witness-m01-target-omitted-handoff --template /home/agent/lanes/hybrid/.temp/research/qwen3.8-27b-chat_template.jinja --thinking on --omit-reasoning --proxy-source /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r11-261030/m01-proxy --witness
```

`matrix-handoff`; 2026-10-30T17:07:16Z; exit **0**; log `matrix-handoff.log`.

```sh
python3 .temp/hybrid-b-r11-261030/run_matrix_handoff.py
```

`stub-bridge-qualified`; 2026-10-30T17:14:40Z; exit **0**; log `stub-bridge-qualified.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p test_journal.py -v
```

`e2e-section11-handoff`; 2026-10-30T17:17:28Z; exit **0**; log `e2e-section11-handoff.log`.

```sh
python3 tools/hybrid-proxy/test_decode.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --golden /home/agent/Werkstatt/hybrid-models/golden/prompt-512 --artifacts .temp/hybrid-b-r11-261030/e2e-section11-handoff
```

`regressions-delivery`; 2026-10-30T17:17:27Z; exit **0**; log `regressions-delivery.log`.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v
```

`concurrency-native-handoff`; 2026-10-30T17:20:57Z; exit **0**; log `concurrency-native-handoff.log`.

```sh
python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/concurrency-native-handoff
```

`syntax-delivery`; 2026-10-30T17:26:14Z; exit **0**; log `syntax-delivery.log`.

```sh
python3 -m py_compile tools/hybrid-proxy/proxy.py tools/hybrid-proxy/store.py tools/hybrid-proxy/journal.py tools/hybrid-proxy/stub_server.py tools/hybrid-proxy/probe_append_api.py tools/hybrid-proxy/test_journal.py tools/hybrid-proxy/test_append_sessions.py tools/hybrid-proxy/native_fixture.py tools/hybrid-proxy/test_remote_append.py tools/hybrid-proxy/test_append_api.py tools/hybrid-proxy/test_concurrency_native.py
```

`remote-append-handoff`; 2026-10-30T17:26:13Z; exit **0**; log `remote-append-handoff.log`.

```sh
python3 tools/hybrid-proxy/test_remote_append.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r11-261030/remote-append-handoff
```

`delivery-check-01`; 2026-10-30T17:30:13Z; exit **1**; log `delivery-check-01.log`.

```sh
python3 .temp/hybrid-b-r11-261030/delivery_check.py
```

`delivery-check-02`; 2026-10-30T17:30:23Z; exit **0**; log `delivery-check-02.log`.

```sh
python3 .temp/hybrid-b-r11-261030/delivery_check.py
```

`diff-whitespace-delivery`; 2026-10-30T17:30:55Z; exit **0**; log `diff-whitespace-delivery.log`.

```sh
git diff --check
```

`delivery-check-report`; 2026-10-30T17:32:17Z; exit **0**; log `delivery-check-report.log`.

```sh
python3 .temp/hybrid-b-r11-261030/delivery_check.py
```

`diff-whitespace-report`; 2026-10-30T17:32:17Z; exit **0**; log `diff-whitespace-report.log`.

```sh
git diff --check
```

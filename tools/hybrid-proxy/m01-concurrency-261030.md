# M0.1 proxy concurrency (O4)

Run `hybrid-b-r9-261030`, 2026-10-30 UTC. **Ready for review**, with the precise admission and model-template bounds below. Branch `m01-proxy-concurrency` starts at accepted M0 `7d37eab007bc2806c09d3a020d3308f54c6102cb`. Implementation commit: `5199481b49bab9c6c40cbca94df225c4ff9f3817` (signature G). A separate arrival-order correction follows initial implementation `1d53c0f4887ba74b01c01007d167c197ea9c6677`; the evidence commit follows both. Nothing was pushed.

Contract: `/home/agent/lanes/hybrid/docs/prototype-spec.md` v0.3.6, section 4a, SHA-256 `1bfff92e07765898cb19b6167e6ca27ec8cb16662d990685efd45a40de3867ac`. The historical M0 spec/verdict in the branch is preserved; this change touches only `tools/hybrid-proxy/`. No engine/server edit, configure, build, installation, model download or A-workbench write was needed. Existing native binaries are read-only M0 build `328cfb23cf`.

## Design and lock order

1. Each session has a FIFO ticket queue (`concurrency.py`). A ticket owns the session through preparation, recovery, decode and journal finalization, but its condition mutex is released throughout that work. All requests first reserve a client admission position before template/tokenizer HTTP. Those HTTP calls run concurrently; completed mapping results retire into session queues in arrival order, preventing a fast later tokenizer from overtaking either an explicit or inferred predecessor. Classification failure removes its reservation. Later requests from that client may wait for earlier classification, while other clients, already admitted streams and replay remain independent. A queued request reloads its journal and either continues the fresh prefix or resolves a fork. Foreign explicit IDs return 404 before joining the queue.

2. `Proxy.lock` is `Store.lock`, a reentrant mutex shared by SQLite access, slot reservations, verification flags, queue lookup/reservation and link-rate EWMA. Its critical sections perform metadata/SQLite work only. Network calls, file snapshots, stream reads/writes and waits for session ownership run outside it. `slot_available.wait()` releases this mutex. Session reservation briefly nests shared mutex -> ticket condition; ticket enter/exit never hold their condition while acquiring the shared mutex. There is no opposite nested edge.

3. A lease atomically selects and pins an owned/free/LRU unpinned slot. Parking, erasing, restoring and validating happen after releasing the shared mutex, while the reservation remains pinned. Its owner keeps the lease through import, full stream and `finish`. Other sessions may acquire other slots. The allocator reloads bindings after waiting, and clears the old binding only after saving its snapshot. SQLite has a UNIQUE slot index. Generic journal saves deliberately omit `slot`; only `set_slot` changes bindings, so a stale replay copy cannot undo eviction.

4. The existing native `GET /slots` route establishes that a reserved slot is idle before slot operations and before release. An unverifiable slot is quarantined until a successful idle probe; another healthy slot remains usable. When all candidates are unavailable, acquisition waits on a monotonic 30-second deadline, then returns 503 with `Retry-After: 1`. The per-session queue wait is separate from this slot deadline.

5. Replay reserves only an idle session queue, skips busy sessions, reloads its journal and performs HTTP without shared ownership. Replay does not pin native state because it only updates the remote shadow; eviction can park that inactive session without overwriting its journal. Recovery and rollback filenames remain protected by the session ticket plus pinned native lease. Existing compatibility checks, rollback backups and dirty-binding rules are retained.

6. A downstream streaming write failure stops downstream writes but drains upstream and journals all evaluated tokens before releasing its slot. An upstream error invalidates verification; lease cleanup waits for idle or quarantines the slot. This avoids reusing a still-processing native sequence. Immediate generation cancellation is not added.

`R` below means `.temp/hybrid-b-r9-261030`. `R/shared-lock-audit.json` records all shared critical sections; AST and source inspection found no network call or yield under them. The source/binary/cleanup manifest is `R/source-binary-manifest.json`. Queue retention follows the existing unbounded session table; no TTL work is included.

## Acceptance evidence

| Brief item | Named test / actual assertion | Result |
|---|---|---|
| 1. Native overlap | `test_concurrency_native.py`: two real streams on `--parallel 2`; sampled token IDs equal independent solo streams for all 64 tokens. Actual token SSE windows overlap and independently polled native slots both process. | Final green: overlap 3.006399 s, both slots processing. |
| 2. Contention | `test_contention_bindings_and_exact_journals`: six workers/sessions, four turns each, two slots. Pinned finish hooks compare slot tokens/hash to the exact expected prompt plus evaluated tokens; all 24 checks pass. Replay twice yields exact remote counts/hashes without duplication. Native helper additionally runs four threads/sessions, three requests each, two slots. | Native audits all 16 requests (4 solo/overlap + 12 contended), including save/restore. |
| 3. Eviction | `test_eviction_skips_pinned_slot`; `test_all_pinned_timeout_and_release`. The active neighbor is unchanged, an idle one can be parked, both pinned slots exclude a new session until 503, then releasing one admits a waiter while the other stays pinned. | Measured 30.009936 s; 503 and Retry-After 1. Independent literal assertion requires the configured deadline to equal 30. |
| 4. Same session | `test_same_session_fifo`: first plus two queued turns enter strictly in order, never together, with exact final journal. `test_same_session_stale_prompt_rejected` rejects stale explicit history without overwrite. | Pass; FIFO rotation mutant fails the second-turn entry assertion. |
| 4. Inferred IDs | `test_inferred_session_continues_after_queued_predecessor`; `test_inferred_queued_fork_reselects_after_predecessor`. | Pass; continuation retains its ID, divergent completed history creates a distinct fork with separate journals. |
| 5. Replay | `test_replay_and_other_live_request_overlap` blocks replay HTTP and still starts another live stream; replay finishes without waiting on that busy stream. `test_replay_does_not_resurrect_evicted_binding` parks a replaying session and then finishes its old journal save. | Pass; slot binding remains transferred and both journals remain correct. |
| 6. Regression | All 21 prior proxy test names/assertions retained; final full suite contains 38 tests. Real integrated prefill+decode 2 clients x 2 sessions, native malformed controls and section-11 numerical checker. | All green, details below. |
| 7. M0 witness | Identical native test file against extracted M0 proxy/store/policy, checked against git blobs. Both solo comparisons pass first, then overlap fails on the intended serialization assertion. | Expected exit 1: overlap -0.425029 s, neither sample has two processing slots. |
| Extra failure paths | `test_native_busy_guard_does_not_block_other_slot`; `test_failed_idle_probe_quarantines_then_recovers`; `test_disconnected_reader_drains_and_commits_journal`; `test_store_rejects_duplicate_slot_binding`; `test_recovery_is_atomic_per_session_and_concurrent`. | Pass with positive reuse/other-slot controls. Snapshot recovery leaves the old journal/state while blocked, yet another session progresses. |

The default unit command does not execute the opt-in native helpers. Native checks were run explicitly below. The final native green and M0 red identity files have the same `test_sha256`; green module hashes equal the committed runtime files. The 38-test stdlib trace reports executed-line coverage: proxy 89.8%, store 96.4%, FIFO utility 86.7%. This is line coverage, not exhaustive interleaving/branch proof; cancellation during FIFO entry and process-crash transactions remain outside the evidence.

### Real end-to-end numerical results

| Conversation | Same-partition greedy | Whole-prompt mean KL | Whole-prompt top-1 |
|---|---|---|---|
| `conversation-a-0.tokens.json` | 64/64 | 8.67437e-15 | 64/64 |
| `conversation-a-1.tokens.json` | 64/64 | 0.00203662 | 63/64 |
| `conversation-b-0.tokens.json` | 64/64 | 8.67437e-15 | 64/64 |
| `conversation-b-1.tokens.json` | 64/64 | 0.00203662 | 63/64 |

Thresholds remain KL <= 0.01 nats/token and top-1 >= 95%. Both colors conversations have whole-prompt free-running first divergence at zero-based index 38; both counting conversations match 64/64. These informational divergences do not replace either numerical gate. The real harness also passes foreign 404, streaming forks, offline completion, RAM-node restart/replay, native import/info, portable restore, single-token boundaries and sixteen native HTTP rejection/neighbor-byte controls.

## Exact commands and real exits

Commands run from `/home/agent/Werkstatt/hybrid-b/llama.cpp`. Each row ran as a standalone subprocess through `python3 R/run.py LABEL COMMAND...`, with direct output redirection to `R/LABEL.log` and its actual exit, UTC time and duration stored in `R/gates.jsonl`. No `tee` or pipeline masks the status. The fetch was `git fetch /home/agent/Werkstatt/hybrid-prototype.git prototype-m0:refs/remotes/prototype/prototype-m0` (0), and branch creation was `git switch -c m01-proxy-concurrency 7d37eab0` (0).

| Label / log | Exact command | Exit |
|---|---|---|
| `native-m0-witness-01` | `python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --proxy-source .temp/hybrid-b-r9-261030/m0-proxy --artifacts .temp/hybrid-b-r9-261030/native-m0-witness-01` | **1** |
| `native-m0-witness-02` | `python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --proxy-source .temp/hybrid-b-r9-261030/m0-proxy --artifacts .temp/hybrid-b-r9-261030/native-m0-witness-02` | **1** |
| `synthetic-m0-order-red-01` | `python3 -m unittest discover -s tools/hybrid-proxy -p test_concurrency.py -k same_session_fifo -v` | **1** |
| `proxy-regression-01` | `python3 -m unittest discover -s tools/hybrid-proxy -p test_proxy.py -v` | **1** |
| `proxy-regression-02` | `python3 -m unittest discover -s tools/hybrid-proxy -p test_proxy.py -v` | **1** |
| `native-overlap-01` | `python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r9-261030/native-overlap-01` | **0** |
| `concurrency-01` | `python3 -m unittest discover -s tools/hybrid-proxy -p test_concurrency.py -v` | **1** |
| `proxy-regression-03` | `python3 -m unittest discover -s tools/hybrid-proxy -p test_proxy.py -v` | **0** |
| `concurrency-02` | `python3 -m unittest discover -s tools/hybrid-proxy -p test_concurrency.py -v` | **1** |
| `real-e2e-01` | `python3 tools/hybrid-proxy/test_decode.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --golden /home/agent/Werkstatt/hybrid-models/golden/prompt-512 --artifacts .temp/hybrid-b-r9-261030/real-e2e-01` | **0** |
| `concurrency-extra-01` | `python3 -m unittest discover -s tools/hybrid-proxy -p test_concurrency.py -k test_re -v` | **1** |
| `concurrency-idle-01` | `python3 -m unittest discover -s tools/hybrid-proxy -p test_concurrency.py -k guard -v` | **1** |
| `concurrency-idle-02` | `python3 -m unittest discover -s tools/hybrid-proxy -p test_concurrency.py -k guard -v` | **0** |
| `concurrency-extra-02` | `python3 -m unittest discover -s tools/hybrid-proxy -p test_concurrency.py -k test_re -v` | **0** |
| `concurrency-quarantine-01` | `python3 -m unittest discover -s tools/hybrid-proxy -p test_concurrency.py -k failed_idle -v` | **0** |
| `proxy-all-trace-01` | `python3 -m trace --count --summary --missing --coverdir .temp/hybrid-b-r9-261030/coverage --ignore-dir /usr:/lib --module unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v` | **0** |
| `native-concurrency-02` | `python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r9-261030/native-concurrency-02` | **1** |
| `mutant-binding-01` | `python3 -m unittest discover -s .temp/hybrid-b-r9-261030/mutants/binding -p test_concurrency.py -k resurrect -v` | **1** |
| `mutant-pins-01` | `python3 -m unittest discover -s .temp/hybrid-b-r9-261030/mutants/pins -p test_concurrency.py -k all_pinned -v` | **1** |
| `mutant-fifo-01` | `python3 -m unittest discover -s .temp/hybrid-b-r9-261030/mutants/fifo -p test_concurrency.py -k same_session_fifo -v` | **1** |
| `native-concurrency-03` | `python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r9-261030/native-concurrency-03` | **1** |
| `mutant-pins-02` | `python3 -m unittest discover -s .temp/hybrid-b-r9-261030/mutants/pins -p test_concurrency.py -k all_pinned -v` | **1** |
| `native-m0-witness-03` | `python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --proxy-source .temp/hybrid-b-r9-261030/m0-proxy --artifacts .temp/hybrid-b-r9-261030/native-m0-witness-03` | **1** |
| `native-concurrency-04` | `python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r9-261030/native-concurrency-04` | **1** |
| `native-concurrency-05` | `python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r9-261030/native-concurrency-05` | **0** |
| `native-m0-final-red-01` | `python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --proxy-source .temp/hybrid-b-r9-261030/m0-proxy --artifacts .temp/hybrid-b-r9-261030/native-m0-final-red-01` | **1** |
| `native-final-01` | `python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r9-261030/native-final-01` | **0** |
| `proxy-all-trace-02` | `python3 -m trace --count --summary --missing --coverdir .temp/hybrid-b-r9-261030/coverage-final --ignore-dir /usr:/lib --module unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v` | **0** |
| `inferred-arrival-01` | `python3 -m unittest discover -s tools/hybrid-proxy -p test_concurrency.py -k tokeniz -v` | **0** |
| `native-arrival-final-01` | `python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r9-261030/native-arrival-final-01` | **1** |
| `proxy-all-trace-03` | `python3 -m trace --count --summary --missing --coverdir .temp/hybrid-b-r9-261030/coverage-arrival --ignore-dir /usr:/lib --module unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v` | **0** |
| `real-e2e-arrival-01` | `python3 tools/hybrid-proxy/test_decode.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --golden /home/agent/Werkstatt/hybrid-models/golden/prompt-512 --artifacts .temp/hybrid-b-r9-261030/real-e2e-arrival-01` | **0** |
| `native-m0-arrival-red-01` | `python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --proxy-source .temp/hybrid-b-r9-261030/m0-proxy --artifacts .temp/hybrid-b-r9-261030/native-m0-arrival-red-01` | **1** |
| `native-arrival-final-02` | `python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r9-261030/native-arrival-final-02` | **0** |
| `mutant-final-binding-01` | `python3 -m unittest discover -s .temp/hybrid-b-r9-261030/mutants-final/binding -p test_concurrency.py -k resurrect -v` | **1** |
| `mutant-final-pins-01` | `python3 -m unittest discover -s .temp/hybrid-b-r9-261030/mutants-final/pins -p test_concurrency.py -k all_pinned -v` | **1** |
| `mutant-final-admission-01` | `python3 -m unittest discover -s .temp/hybrid-b-r9-261030/mutants-final/admission -p test_concurrency.py -k out_of_order -v` | **1** |
| `mutant-final-fifo-01` | `python3 -m unittest discover -s .temp/hybrid-b-r9-261030/mutants-final/fifo -p test_concurrency.py -k same_session_fifo -v` | **1** |
| `delivery-verification-01` | `python3 .temp/hybrid-b-r9-261030/verify_delivery.py` | **0** |

The real end-to-end subprocess additionally ran the following native gate and asserted its actual exit 0; its output is `R/real-e2e-arrival-01/distribution-check.log`:

```sh
build/bin/llama-hybrid-check-distributions /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r9-261030/real-e2e-arrival-01/conversation-a-0.tokens.json /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r9-261030/real-e2e-arrival-01/conversation-a-1.tokens.json /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r9-261030/real-e2e-arrival-01/conversation-b-0.tokens.json /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r9-261030/real-e2e-arrival-01/conversation-b-1.tokens.json
```

The M0 directory is reproducible with `git show 7d37eab0:tools/hybrid-proxy/NAME.py > R/m0-proxy/NAME.py` for `NAME=proxy,store,policy`, after creating that task-owned directory. The native helper imports that directory only when `--proxy-source` is supplied; it does not change the M0 tree. The extraction bytes were rechecked against git before handoff.

## Red controls and earlier failures

| Plant | What it narrows the gate to | Named failure | Result and bound |
|---|---|---|---|
| Accepted M0 whole-request global lock | Concurrency must exist in actual native decoding; solo correctness alone is insufficient. | `native-m0-arrival-red-01`: decode windows serialized; expected concurrent sessions. | Killed, actual exit 1 after both 64/64 solo controls. |
| Rotate queued tickets on release | Session order must be FIFO even with two pending successors. | `mutant-final-fifo-01`, `test_same_session_fifo`: second turn does not enter before third. | Killed, exit 1. Does not test every cancellation interleaving. |
| Save stale journal slot field | Eviction must survive a concurrent old replay save. | `mutant-final-binding-01`, `test_replay_does_not_resurrect_evicted_binding`: replay gets a UNIQUE-slot integrity error instead of retaining the transferred binding. | Killed, exit 1; the runtime fix separates bindings from journal saves. |
| Ignore pin exclusion at capacity | A saturated new session must not attempt to park a live slot. | `mutant-final-pins-01`, `test_all_pinned_timeout_and_release`: fixture guard reports `a pinned streaming slot was mutated` on the attempted save, and expected 503 fails. | Killed, exit 1. The deliberate guard raises in the HTTP fixture thread before mutation; this is specific violation detection, not an unrelated native crash. |
| Bypass ordered mapping-result retirement | A faster inferred tokenizer must not overtake earlier classification. | `mutant-final-admission-01`, `test_inferred_fifo_with_out_of_order_tokenization`: later admission retires while the first remains unresolved (1 pending ticket instead of 2). | Killed, exit 1; separate from session turn-release order. |
| Preliminary unconditional pin bypass | Concurrent reservation must exclude a slot whose binding is not yet published. | `mutant-pins-01`, `test_all_pinned_timeout_and_release`: UNIQUE index rejects the second claimant before both streams start. | Rejected at reservation; not counted as the later streaming-eviction witness. |

All final plants are private copies under `R/mutants-final/` (earlier controls under `R/mutants/`); production files were not mutated. No plant survived these scoped controls. A fourth final plant bypasses ordered classification retirement: `mutant-final-admission-01` fails `test_inferred_fifo_with_out_of_order_tokenization` because the later classification retires while its predecessor is still unresolved (1 pending ticket instead of 2), exit 1. This narrows the inferred-arrival gate separately from per-session turn release. The first pin plant (`mutant-pins-01`) was already caught during concurrent initial reservation by the UNIQUE index, before reaching streaming saturation; it is not counted as the later-stage eviction witness. The revised bounded plant preserves initial reservation and bypasses the pin filter only at capacity.

Earlier nonzero results are retained rather than relabeled as green:

- `native-m0-witness-01` sampled the empty chat-stream verbose token field and failed token-count setup. It is not the O4 witness. The corrected instrument reads raw token IDs from the existing streamed logprob field; witness02, witness03 and final-red all reach and fail the intended overlap assertion.
- `synthetic-m0-order-red-01` used a new queue-inspection fixture against the pre-implementation proxy and failed on missing new API/held fixture work. It is not accepted as an independent FIFO witness; the native M0 test is the required red witness.
- `proxy-regression-01` exposed that the old stub advertised no initialized native slots to the new idle probe. The fixture now models two idle slots and existing GET /slots. The attempted stop found no remaining owned process because that test had already exited; no foreign process was killed.
- `proxy-regression-02` exposed a test-boundary assumption: Content-Length body receipt can precede handler lease cleanup, so immediate manual replay skips a still-busy session. The HTTP fixture now waits behind a session ticket before its manual assertions; no original assertion/test name was dropped. `proxy-regression-03` passes all 21. The real harness replaces its external global replay lock with the same per-session completion boundary.
- `concurrency-01`/`02` also compared volatile fixture `last_used` timestamps in a state-invariance assertion. The corrected assertion explicitly checks count, token IDs and prefix hash. Earlier replay timing uses the fixture completion boundary above. `concurrency-extra-01` and `concurrency-idle-01` failed test-module import after a mechanical patch split a try/finally; that syntax error was corrected before all final suites.
- `native-concurrency-02`/`03` rejected explicit reuse after adding a new user turn: this model template removes earlier reasoning tags, so its rendered token prefix actually changes. `native-concurrency-04` used adjacent trailing assistant messages, which the existing engine rejects. The valid native fixture grows one final assistant message for its continuations; it preserves the token prefix, exercises real native parking/restore, and passes in `05` and `native-arrival-final-02`. No product workaround or engine/template change was introduced.

## Identity, scope and limits

Model: Qwen3.5-0.8B-Q4_0.gguf only, SHA-256 `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`. Runtime source hashes are in the manifest. The following binaries/libraries remained unchanged throughout; no build ran:

| Native artifact | SHA-256 |
|---|---|
| `build/bin/llama-server` | `b88d7a41a493de8fb1089218954c4fd0c953d6ee0de66f228c8b7d9837340a9e` |
| `build/bin/llama-hybrid-prefill` | `5af5b1901d918b9db7eb6d5b94fb79d238f625ee3cb51b660f4c64f40de80240` |
| `build/bin/llama-hybrid-check-distributions` | `ce6de8ea0b6d2fd37511f903adea0b66d9037c9fccb27878a6e1326f61fa4814` |
| `build/bin/libllama.so` | `1ad5d26ba8388d05ca87d8872dd6dcbc104694c24b38dff93ee7ae44844c8865` |

Self-review additionally found that the initial implementation admitted inferred requests in tokenizer-completion order. Before handoff, the arrival-order correction added per-client ordered mapping-result retirement and the two controls above; final native, full-suite and end-to-end runs were repeated for that changed runtime identity. `native-arrival-final-01` then observed an ambiguous counting answer select a word versus a digit at the first token between solo and concurrent runs, while native token/journal integrity assertions still passed. The cause was not isolated; dynamic batching is a possible contributor, not a proven diagnosis. The final fixture explicitly requests decimal digits and comma-space formatting, without grammar constraints or forced tokens, and passes all 64 positions. The divergent log remains retained; arbitrary-prompt bit-exactness under changing concurrent batch shape is not claimed.

`git diff --exit-code 7d37eab0 -- src include tools/server tools/hybrid CMakeLists.txt tools/CMakeLists.txt` exited 0. Runtime/test/docs changes are entirely under the requested proxy module. All original reviewer tests remain under their names. Both source/binary manifests, identical native test hashes, extracted M0 module blobs, staged whitespace and signed commit checks were verified locally.

Arrival order is assigned when a validated request enters preparation, before tokenizer HTTP. `test_inferred_fifo_with_out_of_order_tokenization` holds the earlier tokenizer, lets the later one finish, and requires the original inferred session to run in order; another client progresses while classification is blocked. `test_failed_tokenizer_releases_admission` proves a failed earlier classification cannot strand later valid work. This is request-admission FIFO, not ordering of partial TCP uploads. No shared mutex is held during HTTP or while awaiting session ownership.

The native contended multi-request fixture is an initial request plus two continuations of one assistant message. It does not qualify ordinary new-user-turn explicit reuse with this GGUF: its template intentionally changes historical reasoning serialization, so explicit mismatches reject and inferred histories fork. The failed validity controls and extracted template are retained in `R/model-chat-template.jinja` and native failure artifacts.

The proxy must exclusively own its configured decoder slot pool; unrelated consumers may use slots outside that pool, as the end-to-end reference does. Cross-process proxy ownership of the same SQLite/slots, immediate cancellation on reader disconnect, process-crash atomicity between native memory and SQLite, maximum simultaneous large state-transfer memory, CUDA/Metal, 27B and the 50-prompt 32k/128k matrix are unqualified. Existing O5 double-failure recovery is unchanged. No external landing property suite or attack-surface table was supplied; these are finite CPU correctness gates, not universal interleaving or performance claims.

All 19 recorded native server PIDs are absent from `/proc` at the manifest observation time; every native helper terminates/reaps its own PID in finally, and the end-to-end prefill restart records each PID. `R/cleanup-final.json` lists them. Synthetic HTTP servers are task-owned threads joined by fixture teardown. Available disk at the scope/identity audit: 40,473,874,432 bytes. No push or pull request was attempted; A/orchestrator review remains pending.

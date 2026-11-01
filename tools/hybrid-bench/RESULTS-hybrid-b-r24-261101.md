# R24: cleanup-safe checker evidence

Ready for review on local branch `m2-cleanup-safe`, based on the requested PR #9 head `f65c59483b699ab5cb3b3de32a677fc1fbd7f8b1`, fetched from `/home/agent/Werkstatt/hybrid-prototype.git m2-noise-floor`. No push, PR submission, upstream review or landing. Owner-specified signed local delivery and co-author trailer apply.

Evidence directory: `.temp/hybrid-b-r24-261101/` (`RUN` below). Code/docs/tests identity is `source-identity.json`, SHA-256 `f991a3f76ad888a4d1d49a1907ab695cee510fec96a4841d630389b5d81d5158`. Findings and limits are persisted in `self-review.md` and the existing `.temp/tasks.md`; no task-board is configured here (`board-goal-01.log`, exit 1).

## Behavior

The checker native exit and exact result JSON are saved in a `checker` JSONL row, then its validated `equivalence` row is appended. Both are flushed and fsynced before DELETE or temporary-directory cleanup. Final JSONL publication uses a flushed temporary file and atomic replacement, preserving the journal if a later write fails. Native thresholds, status rules and partitions are unchanged.

Exhausted cleanup attempts add a separate `release_error` containing scope, exception type, credential-redacted message, the session being reconciled and known node session IDs that may remain. Unknown node IDs are explicit. JSONL, bundle and Markdown retain checker metrics and verdicts; summary `release_failed` is true and overall exit is 2 (130 remains authoritative for interruption). Cleanup cannot suppress an earlier failure or interrupt.

Release/DELETE makes at most three transport attempts with 0.5 s and 1 s backoff. HTTP responses, including 401/403/503, are terminal; 404 means already released. A lost acknowledgement followed by 404 is covered. No extend/generation retry was added.

The small proxy change is necessary: its previous DELETE handler converted a node transport exception to HTTP 503, hiding both retry eligibility and node IDs. It now retries transport failures before that conversion and returns remaining current/retired IDs on failure. `tools/hybrid-proxy/release.py` shares retry and diagnostic behavior between proxy and benchmark. Other changes stay in `tools/hybrid-bench/`.

README now recommends `ServerAliveInterval 15`, `ServerAliveCountMax 8`, `ExitOnForwardFailure` and a tunnel supervisor for unstable links. It documents client-token-authenticated GET `/v1/sessions`, selected DELETEs, 404 handling and verification. Its runnable example reads credentials from a private file.

## Validation

All runs used CPU, `nice -n 10`, the existing local 0.8B Q4_0 model and no downloads. Model SHA-256: `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`.

| Command / log under RUN | Real exit | Evidence |
|---|---:|---|
| `python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v`, `bench-tests-05.log` | 0 | 39 tests. Socket-level permanent DELETE failure retains passing and failing synthetic checker outputs; transient recovery, lost acknowledgement/404, 401, original-error preservation, redaction and partial-write journal protection. |
| `python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v`, `proxy-tests-01.log` | 0 | 69 tests; existing generation/state/replay contracts and new proxy release cases. |
| `python3 -m unittest discover -s tools/hybrid-proxy -p test_benchmark.py -v`, `proxy-release-tests-02.log` | 0 | 8 tests on final test source; transient success, bounded permanent transport failure, one-attempt 401/503 and leftover IDs. |
| `python3 -m unittest discover -s tools/hybrid-eval -p test_eval.py -v`, `eval-tests-01.log` | 0 | 11 evaluator regressions; it imports the mode-release helper. |
| `python3 $RUN/build.py`, `build-01.log`, `build-record.json` | 0 | Existing node-guide CMake, shared `/home/agent/Werkstatt/hybrid-models/.build.lock`, CPU `-j2`, named checker/prefill/server targets only. Minimum sampled free bytes 36,288,290,816, above 25 GiB. |
| `python3 tools/hybrid-bench/test_integrated.py --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --checker-binary build/bin/llama-hybrid-bench-check --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts $RUN/native-01`, `native-01.log` | 0 | Full existing native suite: noise references, checksum rejection, checksum-valid lossy/corrupt logit rejection, default limits, delta-size slope, both link settings, fresh/append checks, LRU replay and budget refusal. |
| Same native command with `--artifacts $RUN/native-02 --case timing`, `native-02.log` | 0 | Final product source after atomic JSONL change: four offload calls and two real checks per link setting; no retained sessions. |
| `python3 $RUN/native_cleanup.py`, `native-cleanup-02.log` | 0 | Real native checker plus isolated final-DELETE outage: benchmark exits **2**, checker exits **0/0**, both same-partition results **8/8**, three attempts, separate URLError/leftover-ID row, explicit cleanup DELETE leaves zero sessions. |
| `python3 $RUN/controls.py`, `controls-04.log` | 0 | Historical result partially invalidated by r6j: three detached-mock kills are void. Corrected controls and identity evidence are in r18 below. |
| `python3 $RUN/doc_check.py`, `doc-check-01.log` | 0 | Exact README snippet against authenticated fake node: GET, selected DELETE, 404, empty list, no token in output. |
| `python3 -m py_compile` on all five changed Python files; `git diff --check` | 0 | Syntax and whitespace. Manual changed-line review also checked ASCII, final newlines and test-name retention. |
| `python3 $RUN/audit.py`, `audit-01.log`; port/PID check, `ports-01.log` | 0 | Seven public bundles and manifests verified, fixture credentials excluded, 26 recorded native PIDs inactive, 15 recorded native ports closed, build driver/child absent. |

The locked build used existing UI-off/CUDA-off/Metal-off/CURL-off configuration. `build-record.json` records the dirty Python candidate at the pinned base, exact command/PID, binary/library hashes and free-byte samples. Native source is unchanged; this is not a claim that the stale launcher build stamp identifies the candidate. Audit verified runtime binaries/libraries unchanged after the build.

Native negative subcommands honestly returned 2 for invalid checksum/backend/device and RAM refusal, and 3 for checksum-valid corrupted/lossy state. The native harness asserted these intended rejections; it did not convert their exits to success. Earlier green bench logs (37/38/39 tests) remain available; the final count is 39. All 33 prior bench test names and all seven prior proxy-benchmark names remain.

## Narrowing controls

The original `controls-04` claimed seven kills; r6j correctly invalidated `equivalence-not-journaled`, `checker-outcome-not-journaled` and `ignore-release-exit` because the same detached namespace also killed identity. Those three historical credits are withdrawn. The other four original controls were valid. The table below is the new r18 evidence from `.temp/hybrid-a-r18-261101/controls-02/controls.json`, with copied-source imports, fresh Python processes and test mocks bound to the imported module.

| Mutant/control | Narrows the gate to | Named failing test | Real outcome / surviving bound |
|---|---|---|---|
| `identity` | Unchanged source under the same mocks | None; all 9 `CleanupTests` pass | Survives, exit 0; validates this finite mocked cleanup suite only |
| `equivalence-not-journaled` | Raw checker output alone | `test_checker_result_survives_final_delete`: journal stages differ | Killed, exit 1 |
| `checker-outcome-not-journaled` | Validated metrics alone | `test_checker_result_survives_final_delete`: journal stages differ | Killed, exit 1 |
| `truncate-final-journal` | Partial final write may overwrite durable evidence | `test_bundle_failure_keeps_journal`: durable checker journal changed | Killed, exit 1 |
| `no-transport-retry` | First connection failure becomes terminal | `test_transient_delete_recovers`: 2 != 0 | Killed, exit 1 |
| `retry-401` | HTTP refusal treated as transient | `test_unauthorized_delete_not_retried`: 3 != 1 attempts | Killed, exit 1 |
| `ignore-release-exit` | Failed DELETE can leave overall success | `test_checker_result_survives_final_delete`: 0 != 2 and 3 != 2 | Killed, exit 1 |
| `suppress-active-error` | Cleanup can swallow prior exception | `test_release_preserves_original_error_and_redacts`: Refusal not raised | Killed, exit 1 |
| `interrupt-not-journaled` | Ctrl-C preserves checker results but loses leftover IDs | `test_interrupt_during_checker_release`: release_failed is false | Killed, exit 1 |
| `short-mode-timeout` | Mode release inherits the checker timeout | `test_release_timeout_and_mode_interrupt`: 10 != 30 | Killed, exit 1 |

The identity control runs first and must survive; a failure stops the run. A source plant must replace exactly one occurrence. Kill credit requires exit 1, no unittest errors/skips, a failure in the named test and its expected assertion text. Each worker records its imported source path and verifies that `main` resolves globals from the module the tests mock, including the imported `bundle` alias. These are bounded instrument controls, not exhaustive mutation coverage.

## Failed attempts and limits

- `native-cleanup-01.log`: fault harness exited 1 before reaching the checker because it used `Request.method` on default GET. The benchmark honestly produced an aborted exit-2 bundle. Corrected harness uses `get_method()`; the successful second run is the native fault evidence.
- `controls-03.log`: exited 1 because the atomic-write plant did not replace the test module's imported `bundle` alias. That survival was not credited. Correcting that alias in `controls-04` did not repair the detached globals in three other plants; the old seven-kill claim and the corresponding earlier-run credits for those three plants are withdrawn. See r18.
- PATH CMake was absent (127); the existing node-guide local executable passed readiness and built successfully. `ty` and `editorconfig-checker` are absent (127); their checks are **unrun**, not green. No tools were installed.
- The fake-node unit suite uses synthetic checker JSON to isolate evidence persistence; the native runs separately exercise the actual checker. CPU K=8/depth-132-class tests do not qualify 40k, M3 Max/Metal, CUDA, 5G/SSH reconnection or section-11 acceptance. No target-machine performance claim.
- A proxy release can outlive the client's per-attempt timeout. An unreachable proxy cannot provide node IDs; authenticated node reconciliation remains required. Ambiguous lost responses mean IDs **may** remain. Filesystem/power loss outside successful fsync/atomic-replace guarantees and uncatchable termination are not qualified.
- No surface table was supplied (brief gap); the named contracts above are finite coverage. No supplied surface rows were silently treated as out of contract. No previously committed reviewer test was removed. This is self-reviewed local delivery; independent acceptance and hosted checks are pending.

## Artifacts and cleanup

Real cleanup-failure bundle: `native-cleanup-02/proxy/hybrid-bench-native-cleanup-20261101T132625294214Z.tar.gz`, SHA-256 `1a79b191c6b4041ec277b341cbf78a05f97ce7a6f01e39667b3f54a64e2626b8`. Final passing timing bundles: no-link `bb60683bd5de96aef33cee4381298bc19b7af2c57d33d3534329a9cca918ebd9`, link `24b483ac41a26e33c19ceb4dd77effc6a8519ea7c1bfe8bfd1037f634f3b0016`. All seven bundle paths/hashes and observation time are in `audit.json`.

Audit at 2026-11-01 13:28:52 UTC hashed then removed 50 owned native/config/state files, 123,298,852 bytes. Cleanup manifest SHA-256: `595274bb4b487e8f3299ce202a4506b4dfe98dbc1c561a70d32e97e4b3b31f2d`. Logs, command/PID records, metadata and sanitized bundles remain. No shared models, other workbenches or foreign processes were changed. No session links included.

## r18 fixes

Run `hybrid-a-r18-261101` continues dev-b's exact `bcae223c9b06c2284f0d1bbc2fce2f77a67effb1` on `m2-cleanup-safe`, without rebase; its PR #9 base remains `f65c59483b699ab5cb3b3de32a677fc1fbd7f8b1`. Review input: `/home/agent/Werkstatt/hybrid-review-proto/llama.cpp/.temp/REVIEW-cleanup-safe-claude.md`. Local signed handoff only, no push. Evidence root (`R18`): `.temp/hybrid-a-r18-261101/`. Tested source/docs manifest: `source-identity.json`, SHA-256 `45d3cb1cc7ec050fe2c4000850fd5ddd1293ef08109a42b99e7a787aca44f965` (excludes this report).

F1 is repaired in tracked `tools/hybrid-bench/controls.py`, replacing the old scratch-only harness. It copies the Python source files into an isolated temporary tree, changes the source on disk and imports it in a fresh subprocess. Identity survives; all seven original mutants now fail on their intended assertions, with two additional controls for the notes below. Socket-handler journal snapshots are asserted in the test thread, so a persistence violation reports a direct assertion rather than a handler crash followed by an unrelated zero-check count.

N1: `release_session` records a sanitized `KeyboardInterrupt` release-error row and then re-raises. Tests inject interruption in both DELETE and retry backoff, on both fresh and append-chain checker releases. The durable JSONL and identical archived JSONL retain every completed checker result and the remaining checker ID, the Markdown identifies it, and exit 130 remains authoritative. A mode-release interrupt records its client session ID and explicitly unknown node IDs; no response means those IDs cannot be recovered from the proxy. Repeated interruption during recording, fsync failure or uncatchable termination cannot guarantee a durable row. Normal successful release remains the positive control.

N2: `test_checker_result_survives_fresh_delete` adds a real socket-drop outage on `checker-1`, for checker exits 0 and 3. Both fresh and append-chain results survive, only the fresh node ID remains, three attempts occur, and the overall exit is 2. The existing no-failure and transient-recovery controls remain green.

N4: restored the client-to-proxy release timeout from 10 s to the original 30 s per attempt; checker DELETE remains 10 s. A successful proxy release may need to delete current and retired sessions, so sharing the shorter checker budget was unjustified. Both retain at most three transport attempts with 0.5/1 s backoff: approximately 91.5 s and 31.5 s when every attempt reaches its timeout, excluding scheduling overhead. The README and a positive/negative timeout test pin that distinction. A proxy node request can still outlive the client timeout; unknown IDs remain an honest reconciliation requirement.

| Command / evidence under R18 | Real exit | Claim |
|---|---:|---|
| `nice -n 10 python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v`, `bench-tests-02.log` | 0 | 42 tests; existing names retained, new fresh outage, interrupt and timeout contracts |
| `nice -n 10 python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v`, `proxy-tests-01.log` | 0 | 69 tests; proxy product source unchanged in r18 |
| `nice -n 10 python3 -m unittest discover -s tools/hybrid-eval -p test_eval.py -v`, `eval-tests-01.log` | 0 | 11 evaluator regressions; shared mode-release helper |
| `nice -n 10 python3 tools/hybrid-bench/controls.py --output $R18/controls-02`, `controls-02.log` | 0 | Identity survives, nine nonidentity mutants killed; individual mutant exits remain 1 |
| `nice -n 10 python3 $R18/harness_negative.py`, `harness-negative-02.log`, `harness-negative.json` | 0 | Two isolated instrument negatives verified: import/setup failure produces only invalid identity and no credit; a semantically identical record-call plant survives (bound: that unchanged call under these mocks, no weakened persistence claim). Both child harnesses really exit 1, as required. |
| `PYTHONPYCACHEPREFIX=$R18/pycache nice -n 10 python3 -m py_compile tools/hybrid-bench/hybrid_bench.py tools/hybrid-bench/test_bench.py tools/hybrid-bench/controls.py`, `compile-01.log` | 0 | Changed Python files compile; no native source/build changes |
| `nice -n 10 python3 $R18/code_style.py`, `code-style-01.log` | 0 | Exact model-naming script extracted from the repository workflow, 166 mappings |
| `git diff --check`, `diff-check-01.log` | 0 | Final whitespace check; source hashes and original test-name retention separately checked |

Expected-red and corrected attempts remain visible. Before the product fix, `cleanup-before-01.log` exits 1 with six assertion failures and one missing-row error: Ctrl-C loses the release-error row, and mode timeout is 10 instead of 30. The first new harness run, `controls-01.log`, exits 1 because the atomic-write plant failed the intended journal assertion but its expected text was too specific; that run marks it invalid and credits no kill for that row. The test now names the durable-journal assertion explicitly and `controls-02` passes. `ty --version` and `editorconfig-checker --version` each exit 127 (readiness-02 logs); type/editorconfig checks are unrun, no installation or download attempted.

Self-review covers scope, security, general conventions, tests and copied-source control design; details are in `self-review.md` and the existing `.temp/tasks.md`. No supplied leaf surface table (brief gap); finite coverage maps F1 to identity/seven original plants, N1 to both interrupt tests/control, N2 to the fresh failure test, and N4 to the timeout test/control. No existing reviewer/producer test name is removed (`test-retention.json`). The root README change is only the required tool-documentation link. No out-of-contract supplied surface rows are claimed. N3's single real-proxy composition test and N5's unusual non-transport proxy exception remain unqualified notes; no new claim is made for them.

All r18 test/control runs are short, CPU-only and `nice -n 10`; no models are loaded, no native/model performance run or rebuild is repeated, and there are no downloads. Historical native evidence above belongs to the earlier candidate; r18 does not promote it to a new native run. Required Python compilation and affected unit suites ran directly with their real exit codes. Independent review/acceptance remains pending; ready for review.

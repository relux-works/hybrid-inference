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
| `python3 $RUN/controls.py`, `controls-04.log` | 0 | Seven narrowing mutants killed by named assertions; table below. |
| `python3 $RUN/doc_check.py`, `doc-check-01.log` | 0 | Exact README snippet against authenticated fake node: GET, selected DELETE, 404, empty list, no token in output. |
| `python3 -m py_compile` on all five changed Python files; `git diff --check` | 0 | Syntax and whitespace. Manual changed-line review also checked ASCII, final newlines and test-name retention. |
| `python3 $RUN/audit.py`, `audit-01.log`; port/PID check, `ports-01.log` | 0 | Seven public bundles and manifests verified, fixture credentials excluded, 26 recorded native PIDs inactive, 15 recorded native ports closed, build driver/child absent. |

The locked build used existing UI-off/CUDA-off/Metal-off/CURL-off configuration. `build-record.json` records the dirty Python candidate at the pinned base, exact command/PID, binary/library hashes and free-byte samples. Native source is unchanged; this is not a claim that the stale launcher build stamp identifies the candidate. Audit verified runtime binaries/libraries unchanged after the build.

Native negative subcommands honestly returned 2 for invalid checksum/backend/device and RAM refusal, and 3 for checksum-valid corrupted/lossy state. The native harness asserted these intended rejections; it did not convert their exits to success. Earlier green bench logs (37/38/39 tests) remain available; the final count is 39. All 33 prior bench test names and all seven prior proxy-benchmark names remain.

## Narrowing controls

| Mutant | Narrows the gate to | Named failing test | Surviving bound |
|---|---|---|---|
| `equivalence-not-journaled` | Raw checker output alone | `test_checker_result_survives_final_delete` | None; killed |
| `checker-outcome-not-journaled` | Validated metrics alone | `test_checker_result_survives_final_delete` | None; killed |
| `truncate-final-journal` | Partial final write may overwrite durable evidence | `test_bundle_failure_keeps_journal` | None; killed |
| `no-transport-retry` | First connection failure becomes terminal | `test_transient_delete_recovers` | None; killed |
| `retry-401` | HTTP refusal treated as transient | `test_unauthorized_delete_not_retried` | None; killed |
| `ignore-release-exit` | Failed DELETE can leave overall success | `test_checker_result_survives_final_delete` | None; killed |
| `suppress-active-error` | Cleanup can swallow prior exception | `test_release_preserves_original_error_and_redacts` | None; killed |

`controls.json` and per-mutant logs retain exact failure names. These are bounded instrument controls, not exhaustive mutation coverage.

## Failed attempts and limits

- `native-cleanup-01.log`: fault harness exited 1 before reaching the checker because it used `Request.method` on default GET. The benchmark honestly produced an aborted exit-2 bundle. Corrected harness uses `get_method()`; the successful second run is the native fault evidence.
- `controls-03.log`: exited 1 because the new atomic-write plant did not replace the test module's imported `bundle` alias. The test survived that unapplied plant, so it was not credited as a kill. Corrected both bindings; `controls-04.log` is green. Original six controls were green in runs 01/02.
- PATH CMake was absent (127); the existing node-guide local executable passed readiness and built successfully. `ty` and `editorconfig-checker` are absent (127); their checks are **unrun**, not green. No tools were installed.
- The fake-node unit suite uses synthetic checker JSON to isolate evidence persistence; the native runs separately exercise the actual checker. CPU K=8/depth-132-class tests do not qualify 40k, M3 Max/Metal, CUDA, 5G/SSH reconnection or section-11 acceptance. No target-machine performance claim.
- A proxy release can outlive the client's per-attempt timeout. An unreachable proxy cannot provide node IDs; authenticated node reconciliation remains required. Ambiguous lost responses mean IDs **may** remain. Filesystem/power loss outside successful fsync/atomic-replace guarantees and uncatchable termination are not qualified.
- No surface table was supplied (brief gap); the named contracts above are finite coverage. No supplied surface rows were silently treated as out of contract. No previously committed reviewer test was removed. This is self-reviewed local delivery; independent acceptance and hosted checks are pending.

## Artifacts and cleanup

Real cleanup-failure bundle: `native-cleanup-02/proxy/hybrid-bench-native-cleanup-20261101T132625294214Z.tar.gz`, SHA-256 `1a79b191c6b4041ec277b341cbf78a05f97ce7a6f01e39667b3f54a64e2626b8`. Final passing timing bundles: no-link `bb60683bd5de96aef33cee4381298bc19b7af2c57d33d3534329a9cca918ebd9`, link `24b483ac41a26e33c19ceb4dd77effc6a8519ea7c1bfe8bfd1037f634f3b0016`. All seven bundle paths/hashes and observation time are in `audit.json`.

Audit at 2026-11-01 13:28:52 UTC hashed then removed 50 owned native/config/state files, 123,298,852 bytes. Cleanup manifest SHA-256: `595274bb4b487e8f3299ce202a4506b4dfe98dbc1c561a70d32e97e4b3b31f2d`. Logs, command/PID records, metadata and sanitized bundles remain. No shared models, other workbenches or foreign processes were changed. No session links included.

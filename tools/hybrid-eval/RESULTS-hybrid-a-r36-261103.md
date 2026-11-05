# R36: transient RULER native slots

Ready for review against accepted `b63d901932340edab431986dd1497f88ccbd4839` on the existing private-fork branch `m3-ruler-eval`. CPU only on agents1; no GPU, downloads, native builds or push. This fixes resource lifetime without changing generated inputs, arm ordering, inference parameters, scoring or paired analysis. The GPU incident and its 17 retained rows are operator-reported; this task did not access or relaunch that node.

## Save-path audit and root cause

| Path | File / purpose | RULER lifetime |
|---|---|---|
| `ruler_eval.run_example` | No native save call. Creates independent `ruler-<uuid>` sessions, one per arm; only specs, events and committed rows are written by the evaluator. | No arm needs another arm's native state as a base. Specs/results intentionally persist. |
| `Proxy.slot_lease` | `<sid>.bin`, saves an inactive decoder occupant before another session takes its slot; restores a parked session if needed later. | Root cause: benchmark DELETE marked the session closed but retained its slot binding. The next arm saved the closed occupant. Neither DELETE nor later code removed the parked file. |
| `Proxy.prepare`, fork branch | `<child-id>.bin`, copies a parent session as a fork's initial base. | Not used by fresh explicit one-turn RULER IDs. It remains needed for ordinary message-history forks. |
| `Proxy.prepare_messages`, rollback branch | `<sid>-rewrite-backup.bin`, transactional backup before changing a prior user boundary. | Not used by these first-turn examples. Normal rollback/commit removes it; failed restore retains it until explicit session release. |
| `Proxy.resync` | `<sid>-resync.bin.tmp`, streamed node snapshot export; renamed to `<sid>-resync.bin` for restore. `<sid>-resync-backup.bin` protects the current decoder while restoring the export. | Exceptional recovery path, not the normal first-turn route. Finally removes export/temp and normally the backup; failed rollback retains the backup until explicit release. Node snapshot serialization itself is in RAM (`tools/hybrid/prefill.cpp`). |
| `Proxy.prepare_session` / `import_transaction` | `<sid>-delta-backup.bin`, saves the pre-import decoder before applying the last offload chunk. | Used by offload. Transaction/finish normally removes it. Failed rollback retains recovery state until explicit release. Delta HTTP bodies and staged imports do not create additional export files. |
| `setup-client.sh` -> `manage.py client` | Creates `slots/` and passes `--slot-save-path` to the decoder. SQLite token/message journals and service logs live in the state directory. | No launcher loop saves native sessions. Journals/logs intentionally persist; slot saves above are requested by the proxy. Node/client stop and state moves are operator actions, documented in the runbook. |

DELETE now holds session ownership, waits for any competing slot lease, pins and erases the idle decoder, detaches its durable slot binding, then unlinks only its six known file names. Removing the binding is essential: unlinking alone would allow the next lease to save the same closed session again. Cleanup precedes remote DELETE, so a remote outage does not retain local files. Native/OS cleanup failures return refusal and can be retried; foreign-client DELETE retains its original 404 and cannot touch files.

Every normal arm release removes its native files before the next arm, earlier than row commit. A per-example atomic recovery ledger records all three fresh IDs before requests. The outer finally also reconciles every ID on errors, interrupted streams, row-write failures, Ctrl-C and SIGTERM. All arms are attempted even when one release raises. Failed cleanup retains the ledger and blocks subsequent inference. SIGKILL/power loss requires relaunch; the next invocation reconciles the ledger before skipping rows or starting another example. Fresh attempts use fresh UUIDs.

## Disk admission and resume

The CLI requires `--saved-session-bytes S`, an operator-supplied upper bound including output tokens and native overhead, and defaults to a 1 GiB margin. Both slot volumes must have `free >= 2*S + margin`, checked before preparing an uncommitted example and again immediately before session allocation. A refusal states all byte counts and leaves previous committed rows intact. Successful admission records the budget in the event log. Concurrent unrelated disk consumption and an underestimated S remain explicit limits.

The runbook scales the incident's observed 442685280-byte 8k native file conservatively for this model/build, including the 128-token output budget plus two positions. The 8k command uses S=449710315 and a 1973162454-byte floor. This is not a universal estimator; the operator must verify geometry/pilot high-water sizes before deeper spending.

Source hashes remain a resume guard. The explicit `--resume-slot-cleanup` flag admits only an incomplete bundle whose changed source keys are the four repaired Python files and whose old hashes match the accepted head exactly. All other identity/source fields must match. The original manifest, transition record and retained row-file hashes are preserved and archived. Existing rows still pass their original checksum/spec guards and are never rewritten or rerun. Completed old pilots remain historical bundles and are not migrated. The runbook gives the incident-specific stop (already performed), move-three-state-directories, relaunch-three-services, then resume-same-results-directory sequence.

## CPU evidence

Commands run as standalone processes with `PYTHONDONTWRITEBYTECODE=1`, on-disk `TMPDIR=.temp/hybrid-a-r36-261103/tmp` and niced test processes. Evidence is under `.temp/hybrid-a-r36-261103/` in this worktree. Python is 3.14.4; reused the existing RULER venv and dependencies, with no installs. Free disk samples stayed above 28 decimal GB; final sample is recorded in delivery evidence.

| Gate | Result / real exit code | Evidence |
|---|---|---|
| RULER, including disk-floor and 17-row resume | 17 tests, exit 0 | `ruler-04.log` |
| Existing synthetic eval | 11 tests, exit 0 | `eval-01.log` |
| Existing journal | 18 tests, exit 0 | `journal-01.log` |
| Existing template fixtures | 43 tests, exit 0, 5 optional skips | `templates-01.log` |
| Full proxy discovery, including lifecycle/ownership/concurrency | 130 tests, exit 0, 5 optional skips | `proxy-all-01.log` |
| Benchmark suite | 55 tests, exit 0, 2 optional skips | `bench-01.log` |
| RULER evaluator/test type check (ty 0.0.88, explicit venv/search paths) | Exit 0 | `type-check-06.log` |
| Proxy type check and accepted-source baseline | Both exit 1, identical 38 diagnostic messages; no proxy-wide type-clean claim | `type-proxy-01.log`, `type-proxy-baseline-01.log`, comparison in `python-validation-01.log` |
| Python compilation without bytecode, task text checks, exact predecessor hashes | Exit 0 | `python-validation-01.log` |
| Runbook Bash syntax / disk arithmetic / documentation text | Exit 0 | `documentation-check-02.log` |
| Maintained-scope whitespace | Exit 0 | `staged-diff-check-02.log` |

The syntax validation is the Python build check for this delta. No native behavior/build changed; native inference was not rerun. The upstream model-naming style gate is inapplicable to this Python/docs delta. Standalone editorconfig-checker is unavailable; task-local LF/final-newline/trailing-whitespace checks passed. The pinned upstream vendor bytes are untouched.

`BenchmarkProxyTests.test_ruler_slot_files_bounded_after_committed_rows` executes six three-arm examples against the existing HTTP fixture with tiny real native-slot writes, asserting zero RULER slot count and bytes after each committed-row marker, and preserving an unrelated file. These are stub-native HTTP fixtures, not real model inference. The companion `test_release_removes_owned_slot_files_and_detaches` plants each known parked/export/backup file and checks ownership, detachment, removal and idempotence. `test_release_unlink_failure_refuses_and_retry_cleans` establishes a visible, retryable cleanup failure; existing remote-release fault tests also assert absent local bindings/files.

`RulerTests.test_resume_slot_cleanup_reuses_17_committed_rows_and_disk_refusal` interrupts before row 18, retains exactly 17 rows, simulates the accepted predecessor manifest, refuses the next example one byte below the floor, then resumes only index 17. All 17 row byte strings stay unchanged; migrated provenance enters the archive. Other named tests cover exact-floor admission on both volumes, invalid budgets, errors/abort cleanup, hard-stop ledger recovery, fresh IDs, interrupted DELETE, and rejection of unrelated source/seed changes.

| Narrowing control | What it narrows the gate to | Named failing test | Survival bound |
|---|---|---|---|
| Actual accepted `b63d9019` DELETE handler restored in memory | Closed sessions must release native slot binding/files, not only remote RAM sessions | `BenchmarkProxyTests.test_ruler_slot_files_bounded_after_committed_rows` | Killed: exit 1, 2 leaked files / 234 bytes after the first fixture row, `old-behavior-red-02.log`. No survivor claimed. |

Earlier diagnostics are retained honestly: `old-behavior-red-01.log` exited 1 from selecting the test's wrong class and is not mutant evidence. The corrected control selects its actual class and fails the named file assertion. Initial type-check attempts 01-04 exited 1 due to search-path setup, test dictionary inference and one intermediate syntax error; these were corrected before the final checks. Initial RULER/proxy benchmark checks passed, then relevant checks were rerun after changes. An initial documentation check caught a one-byte rounding transcription error (exit 1, `documentation-check-01.log`); corrected before delivery, final check exit 0. No failed/partial check is counted as passing.

No acceptance surface table or distinct landing-property suite was supplied. Self-review applied the skill's scope, security, general and test checklists. No reviewer tests were removed. Runtime changes outside `hybrid-eval` are limited to proxy session release, required to prevent re-parking after deletion; the associated proxy tests and root tool index document that scope. GPU disk high-water behavior, actual node relaunch, production data/bootstrap, GPU inference/quality and deeper-depth size calibration remain outside this CPU-only qualification. Independent review is pending.

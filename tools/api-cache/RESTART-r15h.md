# r15h settlement evidence and restart

Continue accepted signed31c017edb63bd07b4ac46cfd84e763c1c0c36e37 on m3-api-cache-r15g. All previous supervisor/runner/observers have exited, verified from proc before source edits. New source carries the accepted r15g identity gate, independent analysis filter, CLI messages, provenance and usage-error fixes. Sign source, tests and this revision before new paid requests. No push or session links.

## Exact cause and source citations

The failing paid source was03ad1a09c386faa8d1e62b15f25b468d972d1a3f, retained unchanged at .temp/hybrid-d-r15e-261103/worktree. In that source:

- openrouter_recovery.py:55 selects a600s deadline; :57-59 bounds the poll loop; :70-76 tests three consecutive samples through resolve_delta; :79-80 raises at expiry. resolve_delta at:33-43 rejects a stable counter still below expected beyond rounding. That rejection is correct while earlier calls are absent from the cumulative counter; it is not an immediate stop in the polling loop.
- openrouter_restart.py:340 initializes samples=[], :342 assigns it only if settled_samples returns. On an exception, :345 writes the still-empty caller variable to unresolved-0153.json. The function's real40-sample history remains in settlement-0153.json. This is a diagnostic loss, not proof that polling never occurred.
- openrouter_restart.py:468-488 reconciles interrupted requests through a fresh watchdog poll window and stores the receipt. supervise():584-585 checks the child exit code after that reconciliation and retains failed phase custody; financial reconciliation alone does not convert a failed child into a successful phase.

Actual phase evidence: settlement-0153.json starts2026-11-03T21:22:20.944381Z and records40 lagging reads atUSD7.212300721, last21:32:07.769779Z. V1 terminal is21:32:21.010759Z. The following watchdog reads remain lagged at21:32:21/36/51 and21:33:06, then reachUSD7.638341724 at21:33:21/36/51. ExpectedUSD7.6383417282 differs by -USD0.0000000042, within the existing USD0.00000001 rounding tolerance. The watchdog resolved zero extra debit at21:33:51, and workflow ended21:33:51.432003Z. The preserved timestamps, rather than the approximate21:42 notification, establish the order. No premature pre-sample decision or rejected in-window stable terminal triplet occurred: the phase deadline expired before that triplet existed.

Sequence153 already has a settled receipt and retains its complete USD0.0204141 reservation. At inspection the live ledger is knownUSD7.6383417282, committedUSD8.2649027682. USD8.2444886682 was the value before adding that retained reservation. Do not reset the ledger, add the receipt again or replay sequence153. Fixtures/settlement-0153.json records the exact47 phase-plus-watchdog observations without credentials or payloads.

## Prospective recovery rule

Allow1200s of settlement polling at15s cadence. This is bounded, covers the observed approximately690.44s to the third caught-up sample, and changes waiting time rather than the delta bounds. A lagging counter must catch up to expected within rounding, or to a positive debit no greater than maximum, and hold for three equal consecutive readings. Any sample above expected+maximum still stops immediately. A missing, stale or unstable counter that does not satisfy the rule by1200s still stops. Keep the full maximum reservation for every resolved unknown charge, including zero debit.

Before each call the scheduler now reserves its depth deadline plus1200s settlement plus120s for three bounded metadata GETs and output flush. It time-skips a cell that cannot fit the fixed cutoff rather than compressing any pause. The HTTP request depth deadlines, heartbeat600s interval, v1 cumulativeUSD22 cap, combinedUSD45 cap and live-spend checks are unchanged. Record the complete persisted poll history, count, evidence path and refusal reason in unresolved diagnostics instead of losing samples on exception. The supervisor still refuses to continue an unsettled/excessive receipt and never blindly overrides a failed child's exit status.

## Rate-limited availability and identity

After financial settlement, a429 without generation ID, usage, stream, output-guard event or observed provider/model is an availability failure, not a completed-response identity violation. It stays accepted=false, has no cache result, and allows the next unattempted index. This narrow Phase transport-error path supersedes r15g's stronger handling of such a429; classify_call itself still unconditionally refuses missing provider for completed/guard-completed rows. Evidence of any completion or identity prevents the exception. The independent r15g analysis filter remains mandatory on every report.

The earlier supervisor stamped timed_out=true while attaching the watchdog receipt to sequence153. That historical flag is preserved in the original raw row, but availability analysis recognizes the completed429 shape as a rate-limit error, not a transport timeout. No original JSON, cost or receipt is rewritten. Ordinary timeout rows still count as timeouts and availability failures.

## Tests and custody

The exact47-value regression uses15s virtual cadence: it resolves at690s with zero debit; it fails against the old600s default with the intended deadline Refusal. A neighbouring late stable positive delta resolves USD0.01. Named overshoot and never-stable negatives stop at the first excessive reading and at the finite deadline respectively. A real loopback429 has no generation/usage, settles zero debit, retains its reservation, counts against availability and permits a second POST. Neighbouring completed/guarded/identity-bearing shapes cannot take that exception. Existing r15g absent-provider, legacy repair, final-filter and provenance controls remain.

Resume through the r15d settled-receipt mechanism from this signed head, continuing at the next index with no replay. V1 cutoff2026-11-04T09:00Z; amendment cutoff2026-11-05T05:40Z. The same source checkout now supplies both new phases and its canonical live report contains the r15g filter. Use nohup+setsid, independently verify supervisor/child PIDs and workflow source hashes, and keep the600s heartbeat. A new GET-only end observer uses a fresh r15h audit-end directory; previous terminal audits remain historical. Record actual head/start/PIDs/ledger in .temp/hybrid-d-r15h-261103/SUMMARY.md and send the explicitly requested orchestrator notice. Source remains frozen after launch. Muse review is expected alongside the experiment; no independent verdict on this delta is claimed.

## Qualification and self-review

The local code-review skill's scope/security/approach/tests/general checks were applied to this isolated module delta. The source was clean at the accepted base before edits; previous processes were absent. No changes to other worktrees or their source, no dependencies, no downloads, no push. The complete reservation/budget implementation and independent r15g analysis filter are unchanged. No surface table was supplied. Loopback controls qualify the instrument, not providers; elapsed virtual time reproduces the15s cadence without claiming exact network jitter.

Full CPU command: PYTHONDONTWRITEBYTECODE=1 TMPDIR=<r15h>/tmp python3 -m unittest discover -s tools/api-cache -p 'test_openrouter*.py' -v. tests-02.log:107 tests, exit0; tests-01.log:106 tests, exit0, before the historical429 timeout-classification control. AST9 modules and diff checks exit0. Source/fixture hashes are in validation-identity.json. No C++ or accelerator code changed, so a C++ build is inapplicable. All completed-response identity and analysis-filter controls from r15g remain. The producer HTTP429 test is renamed to match the owner's newly required continuation contract; no reviewer-written committed test is removed.

The exact regression was first run against the original600s source: exact-regression-red-01.log, real exit1 with the intended settlement-deadline Refusal. In-memory narrowing controls never edit source or make real provider requests:

| Mutant | Property narrowed to | Named failing test | Result |
|---|---|---|---|
| Restore600s default | A late caught-up counter gets its three stable samples before the configured window expires | test_exact_0153_lag_then_three_stable_samples_resolves_zero | mutant-deadline600-01.log, exit1, intended deadline Refusal at600s |
| Ignore overshoot | Excessive debit stops immediately rather than waiting for stability/deadline | test_named_negative_lag_then_overshoot_stops_immediately | mutant-ignore-overshoot-01.log, exit1, expected excessive-delta refusal replaced by deadline refusal |
| Halt settled429 | A generation-less rate-limit error may continue only after charge settlement | test_rate_limited_no_generation_zero_debit_continues | mutant-halt-settled429-01.log, exit1, phase fails instead of continuing |

All three controls were killed by their named witnesses; no survivor or unrelated setup failure. The first is reported as an intended exception rather than a failing assertion. These expected-red processes are not described as passing tests. Source hashes remained unchanged and the actual107-test candidate suite passed.

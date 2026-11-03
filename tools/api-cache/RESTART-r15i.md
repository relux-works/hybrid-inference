# r15i disk headroom wait and restart

Continue accepted signed0929c1de2428f686dafda023390ee7d52bc095d0 on m3-api-cache-r15g. Prior supervisor4112406, v1 runner4112536 and GET observer4112893 have exited, verified from proc before edits. No signal or source change to another checkout, no downloads, no push or session link. Sign this revision and source before any new paid request.

## Evidence and rule

V1 completed sequence205, then refused headroom at2026-11-03T22:11:14Z. The preserved workflow's terminal timestamp is22:11:14.107216Z rather than the approximate22:17 notice. The failed heartbeat sampled30523002880 free bytes, above28GiB but below28.5GiB, and the old code touched a zero-byte STOP in that same second. Ledger: knownUSD9.6333691129, committedUSD10.4700746929;206 settled receipts. No request was left unrecorded by that stop. Capture: .temp/hybrid-d-r15i-261103/disk-stop-before.json.

Use one shared admission helper at startup, before each paid request and in the GET-only auditor. At free bytes >=28.5GiB, proceed. From28GiB inclusive to28.5GiB exclusive, wait and recheck every60s for up to1200s. If space returns at or before the final check, resume the same unattempted index. If not, log the end and refuse. At any check below28GiB, refuse immediately without waiting out the remaining soft-headroom window. Do not delete other lanes' files or try to manufacture free space.

The helper emits start, each recheck and end, with UTC time, free bytes, reason and elapsed seconds. Paid/startup events append disk-waits.jsonl and the process log; Phase progress records waiting-disk, last completed call, spend and next recheck time. Audit events stay in its own log. Check operator STOP each polling iteration and preserve it. The heartbeat no longer writes STOP for a soft dip; it records waiting-headroom. Hard-floor observations still set STOP. If a request is already in flight, its stream/row/charge evidence is recorded and settled before the next admission check; the monitor does not abandon or kill it merely for a soft dip.

Paid waiting is before any new intent/POST. Check disk again after metadata: if that check waits, refresh the live endpoint prices and key spend before continuing. Recheck fixed phase timing after recovery and before admission; wait-induced delays are actual pauses and never relabelled as exact target TTL samples. A call that cannot fit its depth deadline, r15h settlement allowance and metadata margin is explicitly skipped. Those timing checks also ensure a newly admitted20-minute wait ends before the phase watchdog cutoff. The1200s disk window is per continuous wait, not an extra allowance to spend past a cutoff.

## Restart custody

Preserve all original call JSON, intents and receipts. Before launch, verify206 settled receipts, no excessive or unresolved charges, and no amendment paid data. The old automatic STOP is separately evidenced by its inode/hash/mtime, failed headroom terminal and heartbeat bytes. For this explicitly authorized restart only, archive that exact unchanged zero-byte file into the r15i task directory. Refuse to archive if it changed since capture. No source routine clears STOP, so future operator and hard-floor stops remain effective.

Use nohup+setsid with --resume from this signed head and append supervisor.log. Continue at sequence206 and each series' next index, never replaying a prior index or resetting the ledger. Keep cumulative v1USD22 / combinedUSD45 caps, v1 cutoff2026-11-04T09:00Z, amendment cutoff2026-11-05T05:40Z, registered pauses and600s heartbeat. Both paid phases load this source. Preserve the r15g provider filter and r15h stable-counter settlement/output/identity rules. Independently verify supervisor/runner PIDs against workflow.json and source hashes. A new GET-only end observer uses .temp/hybrid-d-r15i-261103/audit-end, leaving prior terminal audits intact. Record actual launch/head/PIDs/ledger and archive evidence in SUMMARY.md, then send the requested orchestrator notice. Freeze source after launch.

## Qualification and bounds

The local code-review skill's scope, security, approach, tests and general checks apply. Full command: PYTHONDONTWRITEBYTECODE=1 TMPDIR=<r15i>/tmp python3 -m unittest discover -s tools/api-cache -p 'test_openrouter*.py' -v. tests-02.log:115 CPU tests, exit0; tests-01.log:111 tests, exit0, before the added timing/evidence neighbours. AST covers all9 Python modules; no C++ or accelerator code changed. No surface table was supplied and no independent acceptance of this delta is claimed. All prior107 test names remain, including r15g identity/filter and r15h exact settlement controls. Legacy headroom negatives now use virtual time to prove bounded waiting rather than immediate soft refusal.

Controls cover recovered headroom, a full1200s dip, crossing the hard floor, correct logs, preserving the prior call/receipt and next index, preflight refresh, post-wait cutoff refusal and preserved operator STOP. Financial ledger/cost logic is unchanged. CPU virtual time does not claim real host disk recovery or continuous uptime.

| In-memory mutant | Property narrowed to | Named failing test | Result |
|---|---|---|---|
| No headroom wait | A recoverable dip permits the original next call | test_disk_headroom_dip_recovers_and_continues | mutant-no-wait-01.log, exit1 with intended early disk refusal |
| Restore soft sticky STOP | A soft dip cannot permanently block recovery | test_heartbeat_soft_dip_waits_but_hard_floor_sets_stop | mutant-soft-sticky-stop-01.log, exit1 at STOP assertion |
| Skip post-wait cutoff check | Recovery cannot spend after the request/reconciliation slot ceases to fit | test_disk_recovery_rechecks_fixed_phase_cutoff | mutant-skip-cutoff-after-wait-01.log, exit1 at forbidden-call result |

These are expected-red controls, not passing suites. No source edits or real provider calls occur in the controls. Their exact intended failures are inspected; no surviving mutant or unrelated setup failure is claimed. Source/fixture identity and validation evidence are persisted in the r15i task directory before launch.

# r15k bounded settlement and permanent holds

[RESTART-r15k.md](RESTART-r15k.md) fixes the0321 cumulative precision failure and supersedes the older blanket unresolved-charge stop only for failed calls without an observed generation. Three stable samples within the declared lower precision allowance and strict upper reservation bound resolve the debit. After1200s without resolution, the full maximum becomes a permanent unresolved-reserved receipt counted against both22/45USD caps; the fourth lifetime incident halts. Overshoot, known-generation uncertainty, missing evidence and proven identity/instrument failures still halt. No paid request is retried.

The supervisor rechecks permanent receipts before amendment handoff and at the end, without releasing holds or inventing per-call attribution. Resume first reconciles any missing prior receipt with GETs only and starts the next unattempted index. Use the same signed --resume/nohup command below; CPU TMPDIR is .temp/hybrid-d-r15k-261104/tmp, runtime remains .temp/hybrid-d-r15d-261103/, and the new GET-only end observer writes .temp/hybrid-d-r15k-261104/audit-end/. The task SUMMARY.md records cause lines, tests, head, restart time, independently verified PIDs and ledger. Cutoffs, pauses, disk wait, identity filter and600s heartbeat remain unchanged. Source stays frozen after launch.

# r15j unattended continuation

[RESTART-r15j.md](RESTART-r15j.md) supersedes the earlier stop policy. Unknown/absent model, provider evidence failures and depth deviations exclude cache rows without halting the study. HTTP200 with usable complete usage remains successful availability; missing usage fails availability. Concrete wrong models, excessive/unresolved charges, hard disk floor and instrument failures still halt. Preflight failures and1200s soft disk-wait expiry skip cells without paid requests. Late phases skip explicitly; a cutoff hands off only after settled receipts and independent identity/instrument checks. Operator STOP remains authoritative.

Run the CPU suite and the same signed-head --resume/nohup command documented below. Use .temp/hybrid-d-r15j-261103/tmp for CPU work and the canonical r15d/tmp for paid runtime. Runtime log, workflow.json, ledger.json and600s heartbeat.json remain in .temp/hybrid-d-r15d-261103/. New GET-only end analysis goes to .temp/hybrid-d-r15j-261103/audit-end/. SUMMARY.md in the r15j task root records the halt-path audit, signed head, restart time, independent PID verification and ledger. No source edits after launch. Caps22/45, registered pauses and both cutoffs are unchanged.

# r15i disk headroom admission wait

[RESTART-r15i.md](RESTART-r15i.md) registers the bounded disk wait from accepted0929c1de. Below28.5GiB, the active runner, startup and GET-only auditor recheck every60s for up to1200s before refusing. Below28GiB the guard refuses immediately. Waiting starts before an intent or paid POST; previous responses and charges are already recorded. Every wait logs start/rechecks/end with free bytes. disk-waits.jsonl and progress.json disclose paid-worker waits. Heartbeats report soft headroom without writing a sticky STOP; hard-floor observations still block future calls.

If a dip occurs during metadata, wait before POST then refresh live prices/spend; recheck the fixed cutoff after recovery. Operator STOP is always honored. The r15i restart reversibly archives only the evidenced automatic zero-byte STOP from the previous failed soft-headroom heartbeat; no general STOP-clearing mechanism is added. Resume from the signed m3-api-cache-r15g head with the same r15d --resume command below, preserving every original row/receipt, the22/45 caps, cutoffs, r15g filter, r15h settlement and600s heartbeat. Use a fresh .temp/hybrid-d-r15i-261103/audit-end for the new GET-only final observer. Keep source frozen after launch; independently verified PIDs, head, ledger and stop-archive evidence are in the r15i task SUMMARY.md.

# r15h settlement deadline and guarded restart

[RESTART-r15h.md](RESTART-r15h.md) registers the sequence153 diagnosis and continuation from accepted31c017ed. The phase's600s recovery window expired before earlier-call debits reached the key counter; the supervisor then settled the charge at zero debit. Recovery now allows a bounded1200s window and reserves that time plus120s metadata allowance before a paid call. Three stable bounded samples, immediate overshoot refusal, retained reservations, USD22/USD45 caps and both fixed phase cutoffs stay unchanged. Failure diagnostics retain the actual poll history.

A settled HTTP429 with no generation, usage, stream or observed identity is a rate-limit availability failure and permits the next scheduled index. Completed responses still use the r15g identity gate, and every report keeps its independent provider filter. The historical watchdog's timeout flag on such a429 is not counted as a transport timeout. No original receipt or row needs replay or rewriting.

From the signed m3-api-cache-r15g worktree, after verifying stopped custody and settled receipts:

```sh
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15h-261103/tmp
python3 -m unittest discover -s tools/api-cache -p 'test_openrouter*.py' -v
export TMPDIR=/home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15d-261103/tmp
nohup setsid --fork env PYTHONDONTWRITEBYTECODE=1 TMPDIR="$TMPDIR" python3 -u tools/api-cache/openrouter_restart.py --root /home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15d-261103 --resume >> /home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15d-261103/supervisor.log 2>&1 < /dev/null &
```

The fixed head carries all r15g fixes into both new paid phases, the live report, and the separate GET-only end audit under .temp/hybrid-d-r15h-261103/audit-end/. Keep this source frozen while running. Verify supervisor/child PIDs from a separate shell against workflow.json; heartbeat remains600s. The prior r15g audit is retained as a completed audit of the stopped cohort. No pause is shortened; cells that cannot fit the extended settlement allowance are skipped explicitly. The exact47-poll fixture, CPU evidence, cause citations, launch PIDs and ledger are recorded in the r15h task SUMMARY.md and RESTART-r15h.md.

# r15g identity closure and authoritative filtered analysis

Muse r7au2 found a prospective missing-provider bypass on03ad1a09c; no collected row was invalidated. The fixed source is isolated on m3-api-cache-r15g. Never edit the running r15e checkout or signal its supervisor, runner or observer. Amendment continues from that same original source: its supervisor has no alternate-head handoff or phase-boundary hash check. See REVIEW-r15g.md and .temp/hybrid-d-r15g-261103/AMENDMENT-BOUNDARY.md for the exact load chain and custody decision.

classify_call now rejects missing observed provider regardless of preset error, including guard-completed usage and historical output-only repair. Complete over-cap usage with the pinned provider is still accepted. The analysis filter independently rejects any accepted research row lacking provider observations, counts it as an availability failure, and excludes it from cache/price metrics. Raw rows and paid financial receipts stay unchanged; legacy serving_provider observations apply only when observed_providers is absent, never when explicitly empty. Reports disclose the refusal count and rejected row identities. Financial generation-cost checks still cover those charges. Audit source_commit comes from live workflow.json; audit_source_commit and module hashes identify this separate analysis source.

From the signed isolated checkout:

```sh
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15g-261103/tmp
python3 -m unittest discover -s tools/api-cache -p 'test_openrouter*.py' -v
nohup setsid --fork env PYTHONDONTWRITEBYTECODE=1 TMPDIR="$TMPDIR" python3 -u tools/api-cache/openrouter_audit.py --watch-end --output /home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15g-261103/audit-end > /home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15g-261103/end-observer-01.log 2>&1 < /dev/null &
```

The r15g observer is additional read-only analysis custody; the three original processes remain untouched. It reads no key while waiting and performs only allowlisted OpenRouter GETs after terminal state. Its authoritative filtered end report is .temp/hybrid-d-r15g-261103/audit-end/API-CACHE-REPORT.md. The prior live/r15f reports remain runtime diagnostics and do not contain the r15g filter. The task SUMMARY.md records the CPU-only current-data snapshot, observed refusal count and separately verified observer PID. End results append the final identity-refusal count there. No replay, ledger change or new completion is made by this workflow.

# r15f output-length non-compliance and continuation

[RESTART-r15f.md](RESTART-r15f.md) registers the owner-approved output classification and guard. Full HTTP200 usage above the requested16 tokens is accepted, flagged with its actual count, and priced at its real cost. Reservations use1024 output tokens. The client closes a stream above1024 reported completion tokens or1024 exposed UTF-8 output bytes, retaining the partial journal and reconciling unknown charges before another request. The byte limit is conservative and can cancel before1024 actual tokens; SSE chunks are never counted as tokens.

The active source is the signed m3-api-cache-r15e worktree. Run the CPU suite with an on-disk TMPDIR, then migrate the stopped study's settled historical output refusals and resume without replay:

```sh
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15f-261103/tmp
python3 -m unittest discover -s tools/api-cache -p 'test_openrouter*.py' -v
export TMPDIR=/home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15d-261103/tmp
python3 tools/api-cache/openrouter_restart.py --root /home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15d-261103 --reclassify-output
nohup setsid --fork env PYTHONDONTWRITEBYTECODE=1 TMPDIR="$TMPDIR" python3 -u tools/api-cache/openrouter_restart.py --root /home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15d-261103 --resume >> /home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15d-261103/supervisor.log 2>&1 < /dev/null &
```

The migration reads no key, sends no requests, archives original rows under reclassification-r15f/, validates identity/depth/usage/settled receipts, and leaves ledger.json byte-identical. The live report and CSV are rebuilt with per-provider over-cap counts and corrected availability. Supervisor and child PIDs, signed head, cutoffs and source hashes are in workflow.json. The fixed caps remain cumulative USD22 for v1 and USD45 combined, including all prior costs and retained reservations. No pause is shortened. Free disk below28.5GiB stops before paid requests; the floor remains28GiB. Heartbeat cadence stays600s. A new GET-only end observer writes to the r15f task directory so the r15e checks repeat at the extended study end.

# r15e read-only audit and legacy retirement

On m3-api-cache-r15e, both legacy report functions and standalone run routes refuse before reading a key or making requests, and name the live r15d report path. Their older commands below are historical records. During r15e the paid run stayed on95671982b in its original worktree; the owner explicitly authorizes the prospective r15f restart above from this branch. See RESTART-r15d.md for the explicit cost/price acceptance and pooled1M label.

From this isolated worktree:

```sh
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15e-261103/tmp
python3 -m unittest discover -s tools/api-cache -p 'test_openrouter*.py'
python3 tools/api-cache/openrouter_audit.py --output /home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15e-261103/audit-now
nohup setsid --fork env PYTHONDONTWRITEBYTECODE=1 TMPDIR="$TMPDIR" python3 -u tools/api-cache/openrouter_audit.py --watch-end --output /home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15e-261103/audit-end > /home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15e-261103/end-observer.log 2>&1 < /dev/null &
```

Audit output paths must be new and outside the live runtime. The auditor permits only GET/key, GET/generation and GET/models/.../endpoints; it cannot send completions. A fixed call-row snapshot, per-call cost agreement CSV, redacted generation records, current endpoint prices, drift/provenance JSON and a labelled API-CACHE-REPORT.md go into each audit directory. Every accepted call is attempted; persistent429 limits leave an explicitly disclosed ordered sample and unverified remainder. The now audit and final audit fetch records independently. Streamed usage.cost stays authoritative and no reconciliation/ledger is mutated.

The observer records end-observer.pid and end-observer.json in the r15e task root, checks terminal state every30s without opening the key while waiting, and appends final results to SUMMARY.md. It has a2026-11-05 06:10Z wait deadline and refuses below28.5GiB free disk. Verify its actual PID after launch. The live supervisor, child PIDs, source and data paths are read-only. The full dependency evidence is .temp/hybrid-d-r15e-261103/RUNTIME-LOAD-EVIDENCE.md. Old manifests/hashes remain unchanged and the standalone amendment's historical hash is intentionally not regenerated for a retired route.

---

# OpenRouter cache measurement

Standard-library Python 3 tool for run hybrid-d-r15b-261103. Read PROTOCOL.md before paid use. The signed snapshot and protocol precede smoke; no downloads or external dependencies are needed.

From the repository root:

```sh
mkdir -p .temp/hybrid-d-r15b-261103/tmp
export TMPDIR="$PWD/.temp/hybrid-d-r15b-261103/tmp"
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s tools/api-cache -p 'test_*.py' -v
python3 tools/api-cache/openrouter_cache.py --smoke --output .temp/hybrid-d-r15b-261103/smoke
python3 tools/api-cache/openrouter_cache.py --output .temp/hybrid-d-r15b-261103/full
```

The production HTTP process alone reads the mode-600 dedicated key. No credentials are accepted as CLI values. Output directories must be new; automatic resume is refused because a repeated call can be billed twice. Every call records a JSON, consolidated calls.csv, raw usage, timing, cost and routing evidence. summary.json and API-CACHE-REPORT.md are refreshed incrementally. The runner enforces the cumulative USD45 dedicated-key spend limit and checks 28 GB free disk before each paid request. Smoke also enforces USD0.50 incremental cost.

Use `--report-only --output PATH` to rebuild summaries without network access. Keep runtime logs, PID and reports in the task's .temp directory. A nohup launcher command and its actual PID are recorded in the task plan when the full run starts.

A detached launch must survive the invoking terminal/process group. On this host plain nohup did not survive the execution wrapper. The verified task command was:

```sh
nohup setsid --fork env PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.temp/hybrid-d-r15b-261103/tmp" sh -c 'printf "%s\n" "$$" > .temp/hybrid-d-r15b-261103/full.pid; exec python3 -u tools/api-cache/openrouter_cache.py --output .temp/hybrid-d-r15b-261103/full' > .temp/hybrid-d-r15b-261103/full-02.log 2>&1 < /dev/null &
```

Check the recorded PID from a separate shell and verify run.json has that PID before reporting successful startup. Do not repeat the launch into an existing output directory.

For the corrected experiment, use output full-v3/, log full-v3-01.log and full-v3.pid. Include --prior-spend-floor with the reconciled sum of all earlier API-reported costs (smoke, full, full-v2); never launch while a prior intent lacks a response record. The exact launch and actual floor are persisted in .temp/tasks.md and full-v3/run.json. An optional file named STOP in a running output directory stops spending at the next scheduler boundary after the current response is recorded. The prior cohorts are retained with failed terminal records and explanations.

## Amendment 1: deep contexts and log-matched pauses

Read [AMENDMENT-1.md](AMENDMENT-1.md) and its signed provider manifest before new data. The separate runner preserves v1's code, runtime and exclusive key lock, queues until v1 exits, then carries all earlier costs into the same USD45 cap. It uses Qwen Alibaba/Novita/Phala and the >=1M GLM endpoints, 524288/1000000 nominal depths, 8s/35s core pauses, independent cold probes and separate long-pause sessions. Queued means no amendment requests, not completed measurements.

```sh
mkdir -p .temp/hybrid-d-r15c-261103/tmp
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/.temp/hybrid-d-r15c-261103/tmp"
python3 -m unittest discover -s tools/api-cache -p 'test_*.py' -v
python3 tools/api-cache/openrouter_amendment1.py --help
nohup setsid --fork env PYTHONDONTWRITEBYTECODE=1 TMPDIR="$TMPDIR" sh -c 'printf "%s\n" "$$" > .temp/hybrid-d-r15c-261103/run.pid; exec python3 -u tools/api-cache/openrouter_amendment1.py --output .temp/hybrid-d-r15c-261103/run' > .temp/hybrid-d-r15c-261103/run-01.log 2>&1 < /dev/null &
```

Verify the PID in a separate command. Runtime state, calls.csv, per-call JSON/journals, summary.json and API-CACHE-REPORT.md live in that run directory. ranking.json freezes startup priorities; skipped_or_pending lists every missing provider/depth/cohort. terminal.json and a final done/failed log line show termination. The key is read only after prior custody has ended; no API-key value is passed through the CLI or environment. No automatic resume or replay. The amendment's STOP file affects its own runner only.

## r15d: charge resolution and deadline-bounded restart

[RESTART-r15d.md](RESTART-r15d.md) supersedes the earlier admission and timeout logic. The original engines and evidence stay unchanged. `openrouter_recovery.py` adds stable usage-delta reconciliation with retained reservations and flushed stream evidence. `openrouter_restart.py` supervises sequential v1/amendment children, per-depth deadlines, warm-first scheduling, a shared USD45 ledger, common-window analysis and a 10-minute heartbeat. The new disk guard is 28 GiB (30064771072 bytes).

```sh
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15d-261103/tmp
python3 -m unittest discover -s tools/api-cache -p 'test_openrouter*.py'
python3 tools/api-cache/openrouter_restart.py --help
nohup setsid --fork env PYTHONDONTWRITEBYTECODE=1 TMPDIR="$TMPDIR" python3 -u tools/api-cache/openrouter_restart.py --root /home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15d-261103 > /home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15d-261103/supervisor.log 2>&1 < /dev/null &
```

Run that launcher from the committed m3-api-cache-test worktree only once. The initial call34 proof must be at the output root. The supervisor verifies the signed HEAD and refuses uncommitted cache instrumentation or an existing workflow.json. Actual source commit/hashes, PIDs and phase cutoffs are recorded there; inspect them from a separate shell before treating startup as verified. Phase logs, per-call JSON/journals/CSV and skip manifests are in v1/ and amendment/. The root has ledger.json, heartbeat.json, analysis.json, API-CACHE-REPORT.md, terminal.json and SUMMARY.md. The heartbeat samples live usage inside its HTTP process and records the in-flight deadline when the next call time is unknown. This command makes no Matrix or other non-OpenRouter requests. A root STOP file prevents further calls at the next scheduling boundary.

If a failed v1 has settled every prior charge and amendment has no paid data, the same signed launcher may use `--resume` and append to supervisor.log. This preserves all calls and receipts and resumes the next index; it does not replay requests. Stale cumulative key counters get a bounded600s settlement wait; excessive debits stop immediately. Actual pauses after a stop are retained as late and are separated from timing-matched pause rates. Workflow and terminal histories stay beside the current state. Never use this flag to bypass an unresolved receipt.

Cost checks include all accepted dedicated-experiment financial cohorts (earlier smoke/full/full-v2 plus research full-v3/restarted-v1/amendment). Smoke/pilot charges are labelled legacy financial phases and remain outside the research-only call-rows.json; accepted-rows.json is the broader cost cohort. Cost verification never makes a smoke observation a cache result.

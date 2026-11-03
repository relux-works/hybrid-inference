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

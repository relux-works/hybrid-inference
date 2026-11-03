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

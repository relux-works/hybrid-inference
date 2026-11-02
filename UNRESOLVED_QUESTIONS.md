# Unresolved questions

The r36 numerical and depth owner decisions are resolved by hybrid-b-r37-261102: use a same-session local partition band and derive the deep start within native 262144 with 2048 margin. See [runbook](tools/hybrid-bench/GROWING-SESSION.md) and [r37 results](tools/hybrid-bench/RESULTS-hybrid-b-r36-261102.md).

Exact decode catch-up is excluded from r37: the flat replay suffix has no durable prompt/decode evaluation boundaries. A future exact replay protocol needs an explicitly owned evaluation-schedule design. No such protocol change is required for the current measured-band experiment.

GPU/Metal and 100k/deep hardware measurements remain operator-host work; CPU controls do not qualify them.

The fresh CPU sessions still fail the selected numerical criterion at turn 10: hybrid mean KL 0.000667384 > measured partition p95 0.000393690. Turn 12 passes; both starts preserve all journals, top-1 64/64 and free-running 64-token agreement. This is a remaining numerical acceptance gap, not an unresolved definition of the criterion.

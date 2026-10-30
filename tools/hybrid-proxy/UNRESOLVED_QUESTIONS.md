# Qualification limits after review fixes

- Spec v0.3.4 section11 resolves the CPU batch-partition equivalence criterion: exact same-partition greedy64 plus teacher-forced whole-prompt KL/top1 bounds. Free-running whole-prompt text can differ; retain its divergence index in evidence.
- CUDA-to-Metal distribution thresholds still need M1 calibration. This run qualifies only CPU on Qwen3.5-0.8B-Q4_0.
- Performance on real client/server hardware and the larger prompt matrix remain outside this run. The second fix round measures only the three explicitly requested small-model append costs.
- Anthropic Messages and SWA import are not implemented.
- A and the lane orchestrator still need to review this signed fix candidate.

See [fixes-r2-261030.md](fixes-r2-261030.md) for current evidence and exact commands. Rollback-outage state-machine coverage is synthetic; native process-crash atomicity across the decoder and SQLite remains unqualified.

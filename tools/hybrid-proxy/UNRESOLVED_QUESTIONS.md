# Qualification limits after review fixes

- Spec v0.3.4 section11 resolves the CPU batch-partition equivalence criterion: exact same-partition greedy64 plus teacher-forced whole-prompt KL/top1 bounds. Free-running whole-prompt text can differ; retain its divergence index in evidence.
- CUDA-to-Metal distribution thresholds still need M1 calibration. This run qualifies only CPU on Qwen3.5-0.8B-Q4_0.
- Performance on real client/server hardware and the larger prompt matrix remain outside this run. The second fix round measures only the three explicitly requested small-model append costs.
- Anthropic Messages and SWA import are not implemented.
- A and the lane orchestrator still need to review this signed fix candidate.

See [fixes-r2-261030.md](fixes-r2-261030.md) for current evidence and exact commands. Rollback-outage state-machine coverage is synthetic; native process-crash atomicity across the decoder and SQLite remains unqualified.

## M0.1 concurrency qualification

- M0.1 implements per-session FIFO, pinned slot leases and independent replay; A and the orchestrator still need to review the candidate. See [m01-concurrency-261030.md](m01-concurrency-261030.md).
- The supplied Qwen35 GGUF removes earlier reasoning tags when a new user turn is rendered. Explicit reuse requires an append-only token history; native contention uses assistant continuations to retain it. General multi-user-turn explicit reuse is not qualified by this change. Inferred mapping continues to fork when rendered prefixes change.
- A downstream disconnect drains the bounded upstream request before releasing its slot. Immediate cancellation, process-crash transactions across SQLite/native memory, and cross-process ownership of the same slot pool are not added here.
- Session queues and journal retention follow the existing unbounded session table; TTL/metadata eviction remains future store work. CUDA/Metal, large-model/depth, maximum concurrent state-transfer memory and O5 double-failure recovery remain outside this proxy-only scope.

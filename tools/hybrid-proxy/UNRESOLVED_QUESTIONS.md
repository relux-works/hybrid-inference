# Qualification limits after review fixes

## M0.2 scope decision

- M0.2 implementation is paused at the user's explicit server-change boundary. Existing chat requests always use the re-rendered prompt, while raw completion cannot retain the server's chat parser, tool grammar and message-boundary handling. Native controls verify both paths; this is not a claim that a separate proxy implementation of those systems is impossible.
- Proposed authorization: let the chat route accept an authoritative token array while retaining its `common_chat_session`, and retain sampled IDs in the final verbose stream result when `return_tokens` is enabled. No engine or prefill-node edit is proposed. New-message suffix rendering still needs its own exactness gate before implementation can be accepted.
- Native user-boundary checkpoints work when enabled, but the current proxy erases a mismatching slot before the native checkpoint path can run. The documented M0.1 launch disables checkpoints. Remote historical restore/fork remains an absent API; the allowed remote fallback is a fresh session.
- See [m02-append-only-261030.md](m02-append-only-261030.md). The requested M0.2 conversation gates, M0.1 conversation witnesses and prefilled-token totals remain unrun; only API diagnostics and the unchanged 38-test regression suite ran.

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

# R49 session measurement design

R49 is a separate measurement entry point on the accepted r46 successor. The r46 exactness/rehearsal driver and its Qwen equality gates remain intact.

## Transport

Use eight independent HTTPS range connections on a direct node port, with a per-run token read from a private file and a certificate verified against a run-specific CA file delivered through the operator's authenticated SSH channel. Never send tokens in URLs, arguments or logs. A bounded in-memory gateway fronts the loopback-only llama.cpp node. It forwards authenticated control requests, stages an immutable native delta envelope once, returns its length/hash/opaque ID, and serves exact authenticated ranges. Ownership, expiry, byte limits and range lengths are enforced. No redirect or arbitrary manifest URL is accepted. The native import and all native checksums are preserved. Transport integrity checks are production work and remain timed; evidence snapshot/file hashing is post-turn work.

HTTPS avoids deploying and supervising eight SSH tunnels on each Mac, and supports a future node backend that produces the same immutable envelope contract. The existing node stays the inference and state-format authority. Connections are independent rather than multiplexed over one congestion window. Count encrypted TLS record bytes through the standard SSL MemoryBIO API, including handshake and control traffic. These are actual TCP payload bytes, excluding TCP/IP/link headers and retransmissions; report that boundary explicitly rather than calling envelope bytes physical link traffic.

The qualification matrix is 15 and 20 decimal MB/s, RTT 50/100/150 ms, at least 80% payload throughput for each path. Use actual kernel tc netem/tbf in an isolated network namespace. agents1 currently cannot create the required mapped network namespace and lacks CAP_NET_ADMIN. Unshaped TLS tests do not qualify this target; a runnable test and its failed capability probe are retained until an authorized runner is supplied.

## Workload and arms

The input artifact is an owner-log export, not a draw from published quantiles. Preserve the source log hash, session identity, ordered source turn indices and tokenizer/model hash. Select the earliest chronological contiguous 40-turn interval whose starting context is 90k-110k, final context is 190k-210k, at least 12 incoming portions have 2k tokens, and at least three have 10k tokens. Do not reorder, expand, truncate, duplicate or choose by predicted routing or observed performance. If none qualifies, refuse and report the bounds. The precise export is still an external prerequisite; aggregate quantiles cannot reconstruct it.

Every arm receives identical initial and incoming token IDs, observed answer-length budgets, pauses and seeds from that selected interval. Generation remains free-running, so bf16 can change its answer IDs. Preserve each arm's actual journal; do not silently substitute another arm's history. Checkpoint quality uses teacher-forced identical inputs on that arm's own journal against a fresh local reference, and reports free-running divergence. It does not gate equality of independently generated histories or claim task accuracy from these logits.

Arms, in declared order: local; hybrid (bf16, catch-up only when offloading); hybrid-background (bf16, committed-journal catch-up overlapping the tool pause or a later local generation). An explicit f32 control profile is available, without relabelling either bf16 arm. Routing uses measured current HTTPS link rate, PRI31 local rates, an import fit for the selected dtype and matching wire geometry. Include cold start separately and reject empty hybrid routing as a qualification failure.

## Concurrency and timing

A background job reads an immutable durable prefix, never in-progress incoming or sampled IDs. It is coalesced per session and bounded. Remote mutation is serialized per session separately from the decoder. An on-demand offload joins any in-flight remote mutation and validates the exact acknowledged prefix before extending. A worker updates only remote identity/acknowledgement fields, never overwrites a newer token journal, slot binding or benchmark mode. Never assume HTTP 202 proves application; verify GET n and prefix hash. A node ahead of the durable client journal is a named refusal.

The timed turn ends at the committed streamed response, before any evidence callback. A post-turn phase captures the client's checkpoint, hashes files and records its own duration. Checker inference runs in a separate evidence phase, never a timed turn. Correctness journal reconciliation and transport/native integrity checks are real production work and stay inside timing. Report raw session wall time plus cold, timed turns, pauses, evidence and shutdown separately; never present capture-subtracted TTFT as a direct measurement.

Per-call accounting retains local_tail_seconds, meter.local_prompt_ms and reconciliation_seconds. Record monotonic boundaries, TTFT, native decode rate, envelope/control/encrypted traffic, range throughput, critical catch-up, concurrent catch-up, foreground join time and node prefill evaluation wall time. The latter is a synchronized inference wall-time proxy for GPU occupation, not a CUDA kernel utilization measurement; background work is charged even when hidden by local work. Record rejected/fallback routes and quality gates rather than dropping them.

## Required negative controls

Wrong token, untrusted certificate, truncated/wrong range and corrupt envelope must fail without import or successful partial output. Cache ownership/expiry and size limits must refuse. Background catch-up ahead of the durable journal must refuse without dispatch; stale acknowledgements must not replace a new journal. An evidence callback blocked on a test event must be unable to delay completion of a timed turn. Old r46 labels, f32 bytes/imports and exactness gates remain regression controls.

## Review clarifications

The live policy additionally prices the pending durable suffix at the declared node rate and the measured serial control cost; catch-up is never silently free. Cold-prefill requests can produce one unused sampled ID even with n_predict=0. That ID is recorded and discarded before journal commit; slot-info must prove the evaluated prefix equals the unchanged owner base. Per-turn generated IDs and full checkpoint bindings are written privately after the timer, so a deferred-checker interruption does not lose the actual input journal.

# Node state tiers and resident execution

Design proposal, no tiering implementation. The current limits and LRU eviction are independent of this proposal. Capacity estimates use the 27B geometry in the prototype specification and benchmark sizing note; they are not measurements of a downloaded 27B GGUF or a GPU. Verify the actual target GGUF before renting hardware. No CUDA or Metal execution is qualified here.

## Current cost and ownership

The node has one model and one live context, serialized by the request scheduler. Each session retains token IDs and a native sequence snapshot in host RAM. Every extend restores that entire snapshot, evaluates the suffix, optionally exports a bounded delta, then saves the entire snapshot again. Even a one-token shadow incurs work proportional to the retained prefix. LRU eviction bounds the sum of retained snapshot bytes and four-byte token IDs, not process RSS. Model weights, the live context, the empty-state template, allocation overhead, staged new/old snapshots, token vectors, JSON, native delta buffers and HTTP responses are additional allocations. A budget rejection currently occurs after inference and state serialization, so it does not prevent transient allocation pressure.

An extend has exclusive model access. A session referenced by an active response is pinned through completion or disconnect, including a stalled snapshot or delta reader. Queued requests have not acquired a session yet; eviction can make them observe 404 when scheduled. Idle eviction discards the session ID and state, so the client journal must replay into a new session. A request that cannot fit after reclaiming eligible idle state returns 503 without deleting any session. These semantics must survive tiering.

## Memory arithmetic for 27B

Assumptions: 16 full-attention layers, four KV heads, key and value dimensions 256, q8_0 blocks of 34 bytes per 32 values, and 48 recurrent layers with approximately 157,000,000 bytes of f32 state. Then attention KV is `16 * 4 * (256 + 256) * 34 / 32 = 34,816 bytes/token`. Let N be retained tokens and `S(N) = 157,000,000 + 34,816*N` bytes. Serialized metadata is extra; approximately 24 bytes/token for one sequence's attention-cell metadata plus four bytes/token for the journal gives `H(N) ~= S(N) + 28*N`. Check these constants against actual serialization; alignment and fixed metadata are omitted.

| Depth | Attention KV bytes | State S bytes | S GiB | Host store H bytes | Restore + save logical bytes (2S) |
|---:|---:|---:|---:|---:|---:|
| 32k = 32,768 | 1,140,850,688 | 1,297,850,688 | 1.209 | 1,298,768,192 | 2,595,701,376 |
| 128k = 131,072 | 4,563,402,752 | 4,720,402,752 | 4.396 | 4,724,072,768 | 9,440,805,504 |
| 256k = 262,144 | 9,126,805,504 | 9,283,805,504 | 8.646 | 9,291,145,536 | 18,567,611,008 |

At 256k, three idle host snapshots already require about 25.96 GiB, before the model and live context. Illustrative 27B Q4_K_M weights are `27e9*4.85/8 = 16,368,750,000 bytes` (15.245 GiB); Q8_0 weights are 28,687,500,000 bytes (26.717 GiB). Actual file sizes and GPU allocations decide admission. Q4 weights + 256k state + an assumed 2 GiB graph reserve is about 25.89 GiB, so a 24 GiB card cannot hold that combination even before safety headroom. With 20% headroom, 32k is about 23.1 GiB of capacity for this estimate, while 128k and 256k require about 27.1 and 32.4 GiB. These are planning estimates, not throughput or supported-context claims.

The default 512 MiB store cannot fit even the 32k row. Ignoring metadata it fits at most `floor((536,870,912 - 157,000,000)/34,816) = 10,910` tokens for one session, less after metadata. Enlarging a budget does not remove the live-context or transient-copy allocations. For the current round trip at 256k, old and new host snapshots plus an emitted delta and temporary export/encoding buffers can coexist with the live KV. Bound staging separately before admitting concurrent work.

## Time arithmetic

The following bandwidths are explicit assumptions, not measurements: effective host copy rate 20 GiB/s, GPU/host transfer rate 12 GiB/s and sequential disk rate 2 GiB/s. A "logical copy" counts one state-size moved; physical DRAM read plus write traffic is larger. GPU restore/save includes host serialization as well as device transfers, so those terms can add.

| Depth | Whole-state restore + save at 20 GiB/s | Same 2S at 12 GiB/s | One disk restore at 2 GiB/s |
|---:|---:|---:|---:|
| 32k | 0.121 s | 0.201 s | 0.604 s |
| 128k | 0.440 s | 0.733 s | 2.198 s |
| 256k | 0.865 s | 1.441 s | 4.323 s |

For 100 one-token shadows at 256k, the current path moves at least 1.86 TB of logical whole-state restore/save data, about 86.5 seconds at the host-copy assumption before inference and allocation. A resident context removes these particular copies while the session remains resident. It does not remove suffix inference, recurrent snapshot export, wire encoding, hashing or client import. The q8_0 attention growth for a 256-token chunk is 8,912,896 bytes; adding the complete recurrent state gives about 165.9 MB per delta. The unchanged 512 MiB body permits roughly 10.9k new attention tokens per delta on this geometry, with a smaller operational chunk after metadata and safety reserve. The 1 MiB header independently limits a worst-case int32-token chunk to about 95k tokens. At 256k, a full wire handoff therefore requires multiple append deltas and repeated recurrent payloads; increasing context capacity does not change those wire bounds.

## Proposed tiers

1. **Resident live context.** Keep an engine context assigned to each admitted active session, with a bounded pool sized from device free bytes. Reuse one loaded model. While resident, append directly and retain the authoritative token count, prefix hash, context generation and recurrent endpoint beside it. Account actual rounded context capacity, not only occupied tokens; one 256k-capacity context per short session is expensive. Start with one resident context and reuse it for consecutive extends of its owner, then introduce a bounded multi-context pool after measurement. Preserve existing serialized execution until cross-context backend safety and device-memory admission are established.
2. **Host-RAM tier.** Demote an idle resident session by synchronizing and saving one native state, only at a valid recurrent endpoint. Free or reuse its device context after a verified snapshot is committed. Restore once on promotion. Drop the stale host snapshot after successful promotion, or count it explicitly as a recovery checkpoint. Keep model SHA, layout fingerprint, codec/native-state version, token boundary, prefix hash and a checksum with each snapshot. Exact model SHA-256 is required by default. Experimental `--hetero-quant` (`hetero_quant=true` in the proxy) admits a different SHA only after a matching layout fingerprint; the proxy then rewrites the delta envelope SHA, bypassing the native SHA guard. A layout match establishes shapes only, not base-weight identity or numerical equivalence. Quality is unverified and uncalibrated: the r11 CPU Q4_0 node / Q8_0 client measurement gave mean KL 0.0145 nats/token (above the 0.01 reference) and top-1 62/64. The Q8 file was a requantization of Q4 weights, not an independent higher-precision source. Heterogeneous benchmark results remain uncalibrated even when their measured values meet the reference.
3. **Disk tier.** Demote idle host state to a bounded local disk cache through a private temporary file, checksum verification, fsync and atomic rename. Persist a small manifest transactionally with the same model/boundary identity. Encrypt if the deployment threat model requires it; state and tokens are sensitive even without plaintext prompts. Use per-session generations to reject stale files, quotas/TTL, startup cleanup and a validated format before promotion. Pin files and metadata during reads; partial writes must never become the selected generation. Evict from disk only when the client can recover by journal replay. A disk-full failure must preserve the last committed usable tier, or return a documented refusal.

A session has one authoritative generation and an explicit tier/transition state. Pin before selecting a promotion/demotion victim, release on all success, failure and cancellation paths. Eviction, explicit DELETE and disconnect must not free memory or files still read by another request. Maintain per-client isolation and FIFO, and retain critical-client fairness. A queued request must either acquire the expected generation or recover through the existing 404/replay contract.

## Risks and acceptance gates

- Device allocation can exceed the serialized-state estimate because the engine rounds capacity, reserves graph buffers and keeps recurrent rollback cells. Measure resident bytes and high-water RSS/VRAM for each context class; refuse before allocation when a reservation cannot fit. All-pinned pools need a bounded wait/refusal policy.
- Removing per-extend snapshots changes rollback. Inference, delta export, allocation or disconnected-response failure can leave a resident context ahead of the acknowledged journal. Specify the commit boundary and either commit the new boundary with a retrievable 409 resynchronization response or restore a valid checkpoint. Never claim that attention-only truncation restores a recurrent prefix.
- Native snapshots bind to engine ABI, model weights, partition, KV dtype and recurrent layout. Validate every identity on restore; a shared layout fingerprint does not imply equivalent logits across quantizations.
- Full-range recurrent state is not historical state. Forks require replay or a previously saved endpoint; do not invent arbitrary-prefix recurrent rollback.
- Disk stalls, corruption, quota exhaustion and interrupted promotion must not hold the global scheduler indefinitely, expose another client's state or lose the previous valid generation.
- Report resident/host/disk bytes, reservations, pinned counts, transition failures and promotion latency separately. A low serialized-store counter must not be presented as a low RSS/VRAM guarantee.

## Order of work

First instrument load/save/export times, high-water allocations and bytes at 32k/128k/256k on the actual model. Retain the current configurable budget, LRU, in-use protection and fixed wire limits as the baseline. Next implement the single resident-owner fast path with exact same-partition 64/64, teacher-forced KL <=0.01 and top-1 >=95% checks, plus inference/export failure, retry, interleaving and eviction controls. Then add explicit memory reservations and a small resident pool with pinning, transactional demotion/promotion and bounded all-pinned admission. Add the disk tier only after host-tier generation and rollback behavior are stable. Finally qualify CUDA-to-Metal chunked transfer, realistic concurrency, bandwidth and depth ladders on the rented hardware; use fresh calibrated cross-backend thresholds and measure tier misses separately from warm resident extensions.

Open decisions: resident pool capacity classes and scheduling fairness, retry/rollback acknowledgement boundary, disk encryption/key ownership and quota/TTL policy, supported native-state version migration, and calibrated CUDA/Metal numerical thresholds. These need concrete deployment evidence before tiering implementation.

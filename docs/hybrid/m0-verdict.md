# M0 verdict: hybrid prefill/decode prototype

- **Reviewed head:** `prototype-m0` at `23e8ab0f`. It integrates `hybrid-server` `9a029f38` (producer A) and `hybrid-client` `a93a0cde` (producer B) with `main` `1d6624c5`. The only later commit adds this file and one README item.
- **Reviewer:** the hybrid lane orchestrator (Claude Opus 5.5), 2026-10-30.
- **Specification:** `docs/hybrid/prototype-spec.md` v0.3.5.

## Verdict

**Accept M0 into `main`** as a CPU-verified prototype of hybrid prefill/decode transfer for the `qwen35` layout. It is
fit to carry the next milestone: a CUDA server, a Metal client and Qwen3.8-27B at depth.

It is not a production component. Two limits are recorded below and in `HYBRID.md`:
- the proxy decodes one session at a time (O4);
- a double failure, in which the restore after a failed append also fails, leaves a slot that needs a resync (O5).

## What M0 establishes

- **Transfer exactness.** A delta import of [0, n−1) followed by local evaluation of token n−1 decodes exactly like a local run with the same batch partition: 64/64 tokens at 512, 2048 and 8192, and end to end.
- **Numerical closeness to a whole-prompt local prefill.** Teacher-forced over 64 positions: mean KL at most 0.00204 nats per token, top-1 agreement at least 63/64. Free-running greedy text can diverge (index 38 in one conversation); this comes from batch-partition arithmetic, not from transfer (section 11).
- **End to end.** 2 clients × 2 conversations run through the proxy, the prefill node and `llama-server`, including streaming, offline completion and journal replay.
- **Safety of the client state.**
  - Each of 15 malformed deltas is rejected, with both sequences byte-identical.
  - A post-validation capacity failure rolls back only the new positions.
  - Header budgets are enforced, from 16,384 tokens accepted to 16,385 rejected, with the native state unchanged.
- **Cost of the safety.** Append rollback keeps a copy of the recurrent state only, so append time no longer grows with depth: 4.04 / 4.04 / 4.86 ms at n0 = 512 / 2048 / 8192, down from 4.7 / 11.7 / 24.4 ms.
- **Admission on the prefill node.** 2 permits per client and 3 in total, taken before a request waits for the model and held until the response ends. Excess requests get an immediate `429`. Slow readers and queued requests cannot take every worker.

Evidence: `tools/hybrid-proxy/integration-m0-261030.md` lists every gate command and exit code for the integrated
build. Earlier rounds are in `tools/hybrid/*.md` and `tools/hybrid-proxy/*.md`.

## Review history

Two producers on Codex `gpt-6.1-sol` (high) ran three cross-review rounds. The orchestrator read both
implementations and added O1–O5.

| ID | Area | Severity | Status |
|---|---|---|---|
| R1 | Shared core: recurrent record with two cells hit an engine assertion | blocker | fixed, round 1 |
| F1 | Prefill node: slow readers exhausted the HTTP workers | major | fixed, round 1 |
| R2 | Proxy: an incompatible resync changed the decoder before the 409 | major | fixed, round 1 |
| R3 | Proxy: network loss during recovery escaped the offline fallback | major | fixed, round 1 |
| F2 | Codec: header limit of 64 KiB was too small for a 16k-token delta | minor | fixed, round 1 |
| R4 | Tests: the neighbour-sequence test checked metadata only | minor | fixed, round 1 |
| F1-R2 | Prefill node: permits were taken after the model wait, so queued requests held every worker | major | fixed, round 2 (spec v0.3.5) |
| C11 | Client codec had not adopted the shared header budgets | major | fixed, round 2 (byte-identical codec) |
| C12 | Proxy: a failed decoder rollback was swallowed and the backup deleted | minor | fixed, round 2 |
| O1 | Shared core: a full sequence copy before every append (gigabytes at 110k for 27B) | major for measurement | fixed, round 2 |
| O2 | Shared core: internal flags shared the public flag space | minor | fixed, round 2 |
| O3 | `llama-server`: model hashed on the first import, in the main loop | minor | fixed, round 2 |
| T1 | Tests: A's pending-request test cancels two waiters and checks one release | minor | open; B's per-route probe covers it |
| O4 | Proxy: one lock covers each whole request, the decode stream included (`tools/hybrid-proxy/proxy.py`), so one client decodes one session at a time | major for M1, not a correctness defect | fixed in M0.1 (see below) |
| O5 | Shared core and server: if the recurrent restore after a failed append itself fails, the slot keeps its tokens with emptied memory | minor, double failure | open: known limitation |

## Integration check (orchestrator)

- `src/` equals B's tree.
- A-owned files equal A's tree: `prefill.cpp`, `hybrid-scheduler.h`, `test_prefill.py`, `check.cpp`, `hybrid-state.*`.
- B-owned paths equal B's tree: proxy, `tools/server/`, checkers.
- The shared append test is identical on both sides.
- Documentation was copied verbatim.
- All 39 commits after `main` carry good signatures, and `main` is an ancestor of the head.

## Not qualified by M0

- CUDA server with a Metal client.
- Qwen3.8-27B at 32k–128k over the 50-prompt matrix.
- NVMe session store and shadow batching.
- Process memory bounds under maximum simultaneous transfers.
- HTTP/2.

## Next

1. **M0.1, O4.**
   - Per-session locks in the proxy, and a short slot-table lock for acquire, park and evict.
   - Pin a slot while its session is in use.
   - Stream without holding any shared lock.
   - Tests for overlap, contention and eviction.
   - B implements, A reviews.
2. **M1 on a GPU host.** A CUDA server and a Metal client on the owner's Mac.
   - Measure delta size, import time and time to first token at depth.
   - Each delta carries the recurrent state, which the engine keeps in f32: 156.9 MB per delta for 27B, measured on an M3 Max.
3. **Recurrent state as bf16 on the wire.** This halves the fixed part of every delta. Accept it only if the equivalence checks of section 11 still hold.

## Addendum: M0.1, concurrent proxy sessions (accepted)

- **Head:** `m01-proxy-concurrency` at `8e892551`, on top of `prototype-m0` `7d37eab0`. Proxy-only change: commits `1d53c0f4` and `5199481b`, with the report at `tools/hybrid-proxy/m01-concurrency-261030.md`.
- **Specification:** v0.3.6, which adds the proxy-concurrency rules to section 4a.
- **Review:** producer A accepted it in `tools/hybrid/review-m01-of-b-261030.md` (`hybrid-server` `5c8cc7ac`).
  - All 38 tests pass.
  - Two native sessions overlap by 2.27 s, both slots processing, and each matches a solo run 64/64.
  - 16 native journal audits are exact.
  - The M0 witness fails on its serialization assertion.
  - The 503 with every slot pinned arrives at 30.0 s.
  - An out-of-order tokenizer control passes.
- **Orchestrator check of the lock design.**
  - One short shared lock covers the table and the store.
  - Per-session FIFO tickets order a session's requests, and a slot lease pins the slot.
  - While streaming, only the session ticket and the slot lease are held.
- **New finding O6, for M1, not a blocker.** The test model's chat template re-serializes earlier reasoning on a new user turn, so the token prefix changes. An inferred session then forks from position 0, and an explicit one is rejected unchanged. Real agent traffic needs history-preserving templates, or recovery from a checkpoint before the divergence; see `HYBRID.md`.

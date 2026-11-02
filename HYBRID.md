# Hybrid prefill/decode: proof of concept (llama.cpp fork)

Proof of concept of hybrid inference by Relux Works: Swarm Inference, https://inference.swarma.si. The paper that
describes the scheme and its economics is https://github.com/relux-works/hybrid-inference-paper.

The idea: a GPU server prefills an agent session and ships the model state to a unified-memory client (a Mac). The
state is the full-attention KV of the new positions plus either the fixed recurrent state of linear-attention layers or the complete live sliding-window cache. The
client continues decoding as if it had prefilled locally. The specification is `docs/hybrid/prototype-spec.md`;
the M0 review verdict is `docs/hybrid/m0-verdict.md`.

This is research code, not a production service. What has and has not been verified is listed under Status.

## Branches

`main` is the pinned upstream base `e0f142d5` (llama.cpp, 2026-10-29) plus the prototype: M0 and M0.1 from pull
request #1, M0.2, append-only sessions, from pull request #2, and M1, node limits, chunked extends and benchmark integration, from pull request #4. Other branches are historical producer branches.

## Components

| Path | What |
|---|---|
| `src/` (small patch) | `llama_state_seq_get_data_range` exports full-attention KV of positions [p0, p1) plus recurrent state or the complete sliding window. `llama_state_seq_set_data_append` validates the complete delta, then appends full KV and replaces the fixed state; recoverable failures restore the previous state. |
| `tools/hybrid/` | Delta codec (section 6 of the specification), prefill node `llama-hybrid-prefill`, native checkers |
| `tools/server/` (small patch) | `llama-server` endpoints to import a delta into a slot and to read slot state |
| `tools/hybrid-bench/` | Locked offline-capable launchers, seeded three-mode benchmark, fresh and append-chain numerical checks, sanitized result bundles |
| `tools/hybrid-proxy/` | Client proxy: maps conversations to server sessions, decides when to offload, keeps a journal for recovery and replay |

## Status (2026-11-01, M1 prototype, verified on CPU)

Use the [M1 operator guide](tools/hybrid-bench/README.md) for node deployment, token files, SSH tunneling, client setup, benchmark collection and PID-safe shutdown. Builds always disable both UI sources; use `--offline --model LOCAL.gguf` to prevent model downloads as well. Public listeners require `--public --token-file`; `/metrics` and health activity counters require authentication on public nodes.

M1 adds `--max-ctx` (default 16,384), `--ram-budget-mib` (512), and `--max-request-mib` (1). The launcher derives limits from the chosen context and memory; these limits bound retained state and request bodies, not RSS or transient inference buffers. The node evicts globally least-recently-used idle sessions, pins active responses, and returns `503` with need/budget when an append cannot fit. Rejection is after inference/serialization and preserves existing sessions. An evicted ID returns `404`; the proxy recreates and replays its exact token journal.

Critical chunks use `(512 MiB - recurrent_bytes - window_bytes - 1 MiB) // (kv_bytes_per_token + 24)`, also limited by `(1 MiB - 4096) // 11` for the JSON header. The 24 bytes are native per-cell metadata for one sequence; the 1 MiB reserve covers fixed serialization overhead. The boundary includes this reserve: one more token violates the reserved planning bound, although it can still fit the raw body cap. The launcher writes separate `delta_geometry` values from GGUF metadata; policy cost estimates do not control this safety bound. Without launcher geometry the proxy and benchmark use a conservative 27B bound (34,816 KV bytes/token, 157,000,000 recurrent bytes), safe also for 0.8B (6,528 KV bytes/token, 20,201,472 recurrent bytes). Explicit chunk sizes above the planning bound refuse startup. These are the supported 0.8B/27B geometries; other geometry needs verified `delta_geometry` before use.

The benchmark runs identical seeded schedules in local, hybrid and offload modes, records timing and chunk imports, releases sessions and checks fresh plus append-chain state. Native prefill timing is available without `--bench-link`; that flag controls only the optional link probe. The prior CPU 0.8B run near 40k matched 64/64 in both checks, with mean KL 0.00004021 (fresh) and 0.00011765 (append chain), within section 11. See [r11 evidence](tools/hybrid-bench/RESULTS-hybrid-a-r11-261101.md); [r12 revalidation](tools/hybrid-bench/RESULTS-hybrid-a-r12-261101.md) is reported separately. This does not qualify CUDA/Metal or 27B execution.

Exact model SHA-256 is required by default. Experimental `--hetero-quant` (`hetero_quant=true` in the proxy) admits a different SHA only after a matching layout fingerprint; the proxy then rewrites the delta envelope SHA, bypassing the native SHA guard. A layout match establishes shapes only, not base-weight identity or numerical equivalence. Quality is unverified and uncalibrated: the r11 CPU Q4_0 node / Q8_0 client measurement gave mean KL 0.0145 nats/token (above the 0.01 reference) and top-1 62/64. The Q8 file was a requantization of Q4 weights, not an independent higher-precision source. Heterogeneous benchmark results remain uncalibrated even when their measured values meet the reference.

Verified on CPU, same backend, with Qwen3.5-0.8B (arch `qwen35`, the layout of Qwen3.8-27B). Three rounds of
cross-review between the two producers; the reports are in `tools/hybrid/*.md` and `tools/hybrid-proxy/*.md`.

- **Transfer exactness.**
  - A delta import followed by local evaluation of the last prompt token gives the same greedy decode as a local run with the same batch partition: 64/64 tokens at depths 512, 2048 and 8192, and in the end-to-end run.
  - Against a local prefill of the whole prompt in one batch, decoding is numerically close but not bit-identical, because floating-point results depend on the batch partition. Teacher-forced comparison: mean KL at most 0.0021 nats per token; top-1 agreement at least 63 of 64 positions. Specification, section 11.
- **End to end.** 2 clients with 2 conversations each run through the proxy, a real prefill node and a real `llama-server` decode node. Offline completion and replay after reconnect also work.
- **Safety of the client state.**
  - Every malformed delta is rejected, and the existing prefix stays byte-identical.
  - A failed append rolls back only the new positions.
  - Append cost no longer grows with depth: about 4–5 ms at n0 = 512 to 8192 on CPU with the small model, down from 24 ms at 8192.
- **Admission on the prefill node.**
  - At most 2 requests per client and 3 in total hold a permit, from admission until the response is complete. Over either limit the answer is `429` at once.
  - Slow readers and queued requests cannot occupy every worker.
  - A permit is released about 0.1 s after its response completes, or after a queued client disconnects. Right at the global limit a client can therefore get a brief `429`; the proxy treats it as retryable or completes locally.

- **Concurrent sessions in the proxy (M0.1).**
  - Different sessions decode at the same time, one per slot; one session's requests run in arrival order.
  - A slot stays pinned while its session is busy. When every slot is pinned, a new session waits up to 30 s, then gets `503`.
  - 38 proxy tests pass. Two native sessions overlap their decoding and each matches a solo run 64/64.

- **Append-only sessions (M0.2).**
  - The proxy keeps a journal of the exact token IDs in each slot and appends only new messages; a rewrite of the
    history rolls back to the latest checkpoint at a user-message boundary instead of position 0. An explicit
    session whose rewrite cannot be appended or rolled back gets `409`.
  - Prefilled tokens over a three-turn agent conversation drop from 1530 to 524 (thinking on), from 2102 to 536
    (thinking off) and from 3264 to 562 (client omits reasoning).
  - Message identity compares a canonical typed JSON encoding, so `1` and `true` in tool arguments differ.
  - 61 proxy tests pass. Report: `tools/hybrid-proxy/m02-append-only-261030.md`.

Not yet done:
- CUDA server with a Metal client, and Qwen3.8-27B at 32k–128k depth over the 50-prompt matrix;
- the NVMe session store and asynchronous shadow batching;
- a failed recurrent restore after a failed append, a double failure: the slot is then inconsistent and needs a resync.

## Rules

- Commits are signed.
- This repository does not track upstream automatically. GitHub Actions stay disabled: the tree carries upstream
  workflows that are not meant to run here.

## Sliding-window extension

The local `m3-swa` prototype supports `gemma4` and `muse-glimmer` alongside dense `qwen35`, with metadata-derived full-layer and fixed-window geometry. [The state design](tools/hybrid/SWA-STATE-DESIGN.md) describes upstream serialization and exact transactional rollback. [The r16 report](tools/hybrid-bench/RESULTS-hybrid-a-r16-261101.md) records synthetic CPU evidence and rented-GPU commands. Real Gemma 4/Muse Glimmer inference remains unverified here; same-partition logit calibration is the separate PR 9 change.

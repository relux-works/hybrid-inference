# Hybrid prefill/decode prototype (llama.cpp fork)

Private working repository of the hybrid inference prototype by Relux Works. Owner: Ivan Oparin.

The idea: a GPU server prefills an agent session and ships the model state to a unified-memory client (a Mac). The
state is the attention KV of the new positions plus the fixed recurrent state of the linear-attention layers. The
client continues decoding as if it had prefilled locally. The specification is `docs/hybrid/prototype-spec.md`
(v0.3.5); the M0 review verdict is `docs/hybrid/m0-verdict.md`.

## Branches

| Branch | What |
|---|---|
| `main` | Pinned upstream base `e0f142d5` (llama.cpp, 2026-10-29), this README, the specification and the M0 prototype once merged |
| `prototype-m0` | Integration of both producer branches; the pull request into `main` |
| `hybrid-server` | Producer A: range export, native delta codec, prefill node with per-client sessions, admission and a fair scheduler |
| `hybrid-client` | Producer B: append import with range rollback, slot import and info endpoints, client proxy with session mapping and a journal |

Feature branches land in `main` through pull requests after review.

## Components

| Path | What |
|---|---|
| `src/` (small patch) | `llama_state_seq_get_data_range` exports the attention KV of positions [p0, p1) plus the recurrent state. `llama_state_seq_set_data_append` validates a delta completely before changing anything, appends it and, on failure, restores only the recurrent state and removes the new positions. |
| `tools/hybrid/` | Delta codec (section 6 of the specification), prefill node `llama-hybrid-prefill`, native checkers |
| `tools/server/` (small patch) | `llama-server` endpoints to import a delta into a slot and to read slot state |
| `tools/hybrid-proxy/` | Client proxy: maps conversations to server sessions, decides when to offload, keeps a journal for recovery and replay |

## Status (2026-10-30)

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

Not yet done:
- concurrent decoding through one proxy: the proxy holds one lock for a whole request, the decode stream included,
  so the sessions of one client decode one at a time even with several decode slots (next task, M0.1);
- CUDA server with a Metal client, and Qwen3.8-27B at 32k–128k depth over the 50-prompt matrix;
- the NVMe session store and asynchronous shadow batching;
- a failed recurrent restore after a failed append, a double failure: the slot is then inconsistent and needs a resync.

## Rules

- Commits are signed.
- Builds on the shared agent host are CPU only.
- This repository does not track upstream automatically. GitHub Actions stay disabled: the tree carries upstream
  workflows that are not meant to run here.

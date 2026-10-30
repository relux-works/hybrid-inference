# Hybrid prefill/decode prototype (llama.cpp fork)

Private working repository of the hybrid inference prototype by Relux Works. Owner: Ivan Oparin.

The idea: a GPU server prefills an agent session and ships the model state to a unified-memory client (a Mac). The
state is the attention KV of the new positions plus the fixed recurrent state of the linear-attention layers. The
client continues decoding as if it had prefilled locally. The specification is `docs/hybrid/prototype-spec.md`.

## Branches

| Branch | What |
|---|---|
| `main` | Pinned upstream base `e0f142d5` (llama.cpp, 2026-10-29) plus this README and the specification |
| `hybrid-server` | Producer A: range export, native delta codec, prefill node with per-client sessions, fair scheduler |
| `hybrid-client` | Producer B: append import, slot import and info endpoints, client proxy with session mapping and journal |

Feature branches land in `main` through pull requests after review.

## Status (2026-10-30)

- **Verified on CPU, same backend**, with Qwen3.5-0.8B (arch `qwen35`, the layout of Qwen3.8-27B):
  - importing a range delta into an existing prefix gives the same greedy decode as an independent local prefill,
    64/64 tokens at depths 512, 2048 and 8192;
  - the appended native state equals a full export;
  - every malformed delta is rejected and leaves the prefix byte-identical.
- **Cross-review is in progress** and has found open issues on the client side, recorded in the review files.
- **Not yet done:**
  - CUDA server with a Metal client;
  - an end-to-end proxy against real prefill and real decode servers;
  - the NVMe session store;
  - async shadow batching.

## Rules

- Commits are signed.
- Builds on the shared agent host are CPU only.
- This repository does not track upstream automatically.

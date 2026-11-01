# hybrid-a-r22-261101: CPU prototype handoff

Ready for review. Branch `m3-vllm-bridge`, exact parent `a3752b04cbd905bf61ffe6d9a4e7df738f26f4c2`. This is a local private-fork prototype; no push, PR, model download or GPU operation. The signed head is reported to the orchestrator after commit verification.

## Result

Route B uses a source-pinned post-forward hook in the original GPUModelRunner. The hook copies one request's paged attention and align-mode GDN endpoint while it still owns the blocks. Route A is viable: v0.32.0 NIXL explicitly supports Mamba/GDN through HMA, not just attention KV. It requires transfer/peer lifecycle machinery that this dump does not need. An offline LLM instance without the hook cannot safely inspect blocks after generate returns. File/line references and the runnable GPU plan are in [DESIGN.md](DESIGN.md).

`bridge.py` converts the source-derived logical [page,head,token,K+V] layout, BF16 or FP8 E4M3FN/E5M2 with K/V scales, conv SD/DS and BF16/f32 [value_head,V,K] state. It reorders GDN value heads, converts conv history to [channel,time], applies the client's q8_0 Hadamard rotations and emits the existing native sequence format and HYBD envelope with XXH3-128/SHA-256. It recomputes the receiving GGUF's layout fingerprint. Unknown geometry, dtype, state records, token boundaries and oversize deltas refuse; a failed conversion preserves an existing output file. CPU imports require no vLLM/torch/NumPy.

The hook and live CUDA copies have not run. Synthetic native import proves serialization and tensor permutation controls, not cross-engine mathematical or numerical equivalence. State-only import has no boundary logits; the provided checker plan compares at N+1 after a common probe token. The report does not claim logits at N, a speedup, calibrated quality, or CUDA-to-Metal qualification.

## Evidence

Evidence root: `.temp/hybrid-a-r22-261101/`. Final CPU observation: 2026-11-01 19:21 UTC. All commands ran as standalone processes with actual exits recorded; no validation was piped through tee. Python had `PYTHONDONTWRITEBYTECODE=1`, disk TMPDIR was under this task's `.temp/` (ext2/ext3), and CPU compilation/tests were niced. Each compiler invocation used one driver, no parallel build jobs. No full engine rebuild was needed because engine/server sources are unchanged.

| Check | Actual exit | Evidence and bound |
|---|---:|---|
| Native helper build | 0 | `build-native-04.log`; C++17 helper and current `tools/hybrid/hybrid-state.cpp`, linked to existing CPU llama/ggml/vendor-hash libraries |
| CPU suite, 16 tests | 0 | `tests-05.log`, 11.652 s while the independent control process also ran; this is not a benchmark |
| Unchanged-source control, 16 tests | 0 | `control-identity-02.log`; same copied Python source and native helper |
| Mutation driver | 0 | `controls-driver-02.log`, `controls-02.json`; 11 intended assertion failures, each child exit 1, no errors/skips |
| Source hook installation/AST parse | 0 | `patch-source-01.log`; patch applied to the downloaded exact tagged runner and parsed, without importing vLLM |
| Python/ASCII/whitespace and runbook Python syntax | 0 | `syntax-ascii-02.log`; all nine added files plus four modules and four embedded GPU-plan Python blocks checked |
| GPU extraction, actual model imports, logits, latency/RSS/VRAM | Unrun | Orchestrator-owned GPU/Mac experiment, outside agents1 CPU scope |

The full synthetic native test converts BF16 KV/f32 states, then appends FP8 KV with non-unit K/V scales and BF16 states. Both deltas use all 16 attention and 48 recurrent layers. The existing C++ envelope reader verifies hashes/fingerprint; real allocated hybrid caches import and re-export every native body byte, including the append endpoint. A 16-token native body is 157452212 B. Changing one body byte gives child exit 1 with `delta checksum mismatch`, after the valid positive control. No model weights are loaded. The helper's scope is <=1024 tokens on CPU, one sequence; it is not the HTTP slot path or a model inference test.

Other tests independently assert known q8_0 scale/rounding bytes, Hadamard results, distinct KV pages/head values, FP8 dequantization, conv history orientation, GDN head permutation and nonsymmetric within-head sentinels. Separate named tests refuse wrong layer count, block size, head dimension, partial final page, KV dtype, conv dtype and SSM dtype. Tests also cover identity, mode, page/state shape, scales, source config, unsafe paths, file length and atomic refusal. E5M2 has scalar decode/nonfinite controls; a full E5M2 native chain and live GPU dtypes remain unqualified.

## Narrowing controls

Each plant modifies a copied real source module and runs its named test. A setup error is not credited as a kill.

| Mutant | What the gate is narrowed to | Named failing test | Actual exit / survival bound |
|---|---|---|---|
| layer_count | Exact 64 distinct captured layer IDs | `test_wrong_layer_count` | 1, intended assertion |
| block_size | Minimum/multiple-of-16 manager size independent of a consistent page shape | `test_wrong_block_size` | 1, intended assertion |
| head_geometry | 256-dimensional heads independent of an internally consistent fingerprint | `test_wrong_head_geometry` | 1, intended assertion |
| kv_dtype | Explicit KV dtype whitelist | `test_unsupported_dtype` | 1, intended assertion |
| partial_endpoint | Whole-block exporter endpoint | `test_partial_last_block` | 1, intended assertion |
| kv_page_order | Ordered logical pages, not reversed pages | `test_paged_kv_layout_and_scales` | 1, intended assertion |
| fp8_scale | Per-layer dequantization scales are applied | `test_paged_kv_layout_and_scales` | 1, intended assertion |
| gdn_heads | Grouped-to-tiled head permutation | `test_recurrent_head_and_conv_layout` | 1, intended assertion |
| rotation | Actual Hadamard rotation before q8_0 | `test_q8_and_rotation` | 1, intended assertion |
| async_schedule | Synchronous single-request execution contract | `test_source_config` | 1, intended assertion |
| runner_hash | Inspected runner identity, not marker-only patching | `test_patch_location` | 1, intended assertion |
| identity (control) | Unmodified copied code runs the same suite | All 16 tests | 0, survives; proves the control harness admits this source/fixture/environment only |

No planted mutant survived. This finite set does not prove arbitrary dtype/geometry rejection or live GPU extraction correctness. The original brief supplied no formal surface table; this is a declared brief gap, not an independent coverage certification. All existing reviewer tests are retained unchanged. No previous-review rework class applies.

## Corrections and limits

- Initial `tests-01.log` exited 1: the synthetic model fixture omitted `kv_source_impl[i]=i`, leaving zero allocated attention layers. The fixture was corrected. `tests-02.log` then passed 14 tests; `tests-03.log` passed 15; `tests-04.log` passed 16 before adding BF16-state/FP8-KV full append coverage. Final evidence is `tests-05.log` and matching control identity 02. No failed run is represented as passing.
- Initial `git fetch origin main` exited 128 because this local mirror has master and no main ref. Its HEAD is `e0f142d5`; the user's explicit base a3752b04 was used for the new branch. The previous clean branch at c4e0faae was preserved. `base-01.log` records the discrepancy; nothing was reset or rewritten.
- Download: shallow tag-only source clone, 185 MiB, commit `5df63a72`, under 1 GB. No wheel or model download. The original runner hash and 18 source-file hashes were captured before patching; source references remain reproducible from the immutable tag. The downloaded clone is deleted before signed handoff; deletion evidence is `source-cleanup-01.log`.
- Source-derived native constant metadata makes the fixed body term 156894772 B, 12 B below the brief's fixed figure. The exact native reader/re-export passes with this size; there is no invented padding. The 24 B/token metadata is additional to 34816 B/token KV. The fixed f32 tensor payload alone is 156893184 B.
- No changes to native import, server scheduling, model code, protocol limits or existing tests. Diff scope is `tools/hybrid-vllm/` plus a root README tools link. Unresolved GPU gates are listed in [UNRESOLVED_QUESTIONS.md](UNRESOLVED_QUESTIONS.md).
- Task-board is absent (`task-board goal get` exit 1, missing `.task-board`); the existing `.temp/tasks.md` is the tracker. No review, landing, ledger clear or external CI success is claimed. Local self-review covers scope, security, design, general conventions and tests. Signed local commits and the requested team summary are authorized; no push is authorized.

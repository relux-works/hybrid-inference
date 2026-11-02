# hybrid-d-r7-261101: fast converter and vLLM node

Ready for review on `m3-vllm-node`, based on `4a25d528751fd8c43b2ed2bdac62b7c702cb412c`. Local signed commits only; no push or PR. CPU observations span 2026-11-01/02 UTC. This adapter's GPU execution, CUDA byte equivalence, live prefix-cache reuse and heterogeneous numerical qualification remain unrun here. The orchestrator supplied the base bridge's first GPU result during handoff preparation; it is recorded separately below.

## Converter

`fast_codec.py` vectorizes dtype conversion, page/head/token ordering, separate K/V scaling, float32 Hadamard butterflies, q8_0 and recurrent permutations. NumPy is the CPU default. `--backend torch` uses CUDA for rotation and quantization; `auto` selects it when CUDA is available. `--backend reference` preserves the scalar arithmetic oracle in `bridge.py`. Dtype decode and source-scale multiplication retain the reference's rounding order; half ties round away from zero. No matrix multiplication, reduced precision or fused butterfly approximation is used.

Existing independent matrix/layout fixtures remain green. Randomized BF16/E4M3FN/E5M2 pages cover two page counts, separate non-unit/non-f32 K/V scales, rotations 0/64/256 and recurrent SD/DS layouts. A complete 64-layer envelope is compared against the scalar path, including both checksums. Torch CPU arithmetic also agrees. CUDA agreement is a GPU qualification gate, not a CPU result.

Measured on agents1, Python 3.14, NumPy 2.4.2, niced CPU process:

| Synthetic tokens | NumPy wall time | Wire bytes | Delta partition |
|---:|---:|---:|---|
| 4,000 | 2.316 s | 296,270,866 | One delta |
| 40,000 | 20.914 s | 2,178,232,149 | Five 8,000-token deltas |

Times include source decode, rotation, q8_0, all 48 recurrent layers, disk staging/write, canonical XXH3-128, SHA256 and fsync. Fixture generation is excluded. Real 27B geometry is used, with shared synthetic source files across layers; these are warm filesystem synthetic measurements, not model inference or cold-storage throughput. Repeated recurrent endpoints account for the extra 40k wire bytes. The existing 512 MiB body limit remains enforced. Command: `python tools/hybrid-vllm/bench_converter.py --scratch .temp/hybrid-d-r7-261101/tmp`; evidence `converter-bench-02.log`. The earlier converter-step measurement was 2.568/21.857 s at signed `74761bd6b714bd787f588d54bfde601afeb2c7c8`, and was sent to the orchestrator.

## Node and compatibility

`node_adapter.py` implements health/identity, bearer authentication, owner-scoped session create/list/info, extend, snapshot and release, plus metrics/link probes. `node_backend.py` serializes real vLLM completion requests through private atomic hook control. The source-pinned hook captures only the exact final forward of each request. Intermediate chunks are skipped. Failed/interrupted backend completion marks the node unhealthy until restart; a failed conversion leaves the authoritative journal unchanged.

Fresh extends export the largest block-aligned endpoint at or below the requested end. Append exports contain only the new attention range, including starts inside a page, plus recurrent state at the new aligned endpoint. vLLM receives the full journal and owns cache hits/eviction. No attention-only substitute for recurrent state is used. Unsupported geometry, speculative/MTP configuration/cache slots and unaligned historical snapshots refuse. Exact error statuses and boundary semantics are in [NODE-PROTOCOL.md](NODE-PROTOCOL.md).

The old proxy requires exactly its requested endpoint, so an unchanged adapter-only interface could not represent a shorter export. The minimal compatible extension is optional `export_block_size` in health/node info/config. The existing launcher propagates it; the proxy asks for aligned chunks and computes the short tail locally. The checker accepts optional `local_tail_start` and evaluates every tail token. Ordinary nodes default to alignment 1. User CLI and benchmark modes remain unchanged. Refusal fallback is still rejected by forced-offload metering; a planned short alignment tail is explicitly visible.

The adapter's `X-Hybrid-Prefill-Ms` is total node preparation (vLLM request, hook capture, conversion and staging cleanup). Do not interpret it as raw vLLM kernel-prefill tok/s. Overall offload TTFT remains the measured user-facing quantity.

The native envelope/body/checksum format is unchanged. Remote main was freshly fetched at 2026-11-02 00:01 UTC: `528724c96894aa5f65e98b329317de59aa694afc` still uses `llama-native-seq-v1`, canonical XXH3-128 plus SHA256 and the same section shape. Other in-flight formats require separate integration.

The node's identity follows the actual cached FP8 source files, with source hashes persisted separately from the BF16 GGUF layout anchor. The Q4 client uses the existing heterogeneous layout admission. `weight_comparison=heterogeneous`, KL and uncalibrated status are preserved in benchmark results. Matching layout alone does not prove source checkpoint/tokenizer provenance.

`deploy-vllm-node.sh` uses a supervisor with recorded PIDs, dedicated process groups, start ticks, command/log records and guarded runner restoration. Startup probes the live manager block size. Normal stop, startup failure and signals share cleanup. Hook installation is atomic and preserves original permissions. A runner changed externally is preserved with a cleanup error and backup. No processes are stopped by name. [GPU-NODE-RUNBOOK.md](GPU-NODE-RUNBOOK.md) contains exact B300 commands using the supplied venv/cache/GGUF paths, CUDA 12.9, `103a-real`, 8k/32k local/offload smoke and heterogeneous checks.

## Validation

All logs below are under `.temp/hybrid-d-r7-261101/`. Commands ran directly, with real statuses; no gate was piped through tee.

| Gate | Result and real exit | Evidence |
|---|---|---|
| Full bridge/GPU-script/fast-codec/adapter/runtime discovery | 59 tests, no skips, exit 0 | `vllm-tests-05.log` |
| Existing proxy discovery | 69 tests, exit 0 | `proxy-tests-01.log` |
| Existing bench discovery | 42 tests, exit 0 | `bench-tests-01.log` |
| Production page gather with torch CPU | 2 tests, exit 0 | `gather-tests-01.log` |
| Fast converter with torch CPU installed | 4 tests, exit 0 | `fast-tests-02.log` |
| Native helper + checker + prefill build | Niced, at most -j4, exit 0 | `build-01.log`, `native-build-01.log`, `checker-build-01.log`, `prefill-build-01.log` |
| Existing local 0.8B model, native imported prefix + 18-token local tail | Checker exit 0, same-partition exact, full-vocabulary KL 0 over two continuation positions | `native-tail-probe-02.log`, `native-tail-positive.result.json` |
| Wrong/absent local-tail boundary on that shortened chain | Checker exit 2, intended `incomplete check chain`; no result emitted | `native-tail-wrong-boundary.log`, `native-tail-legacy.log` |
| New production/test Python modules type check, Python 3.11 target | exit 0 | `type-check-final-02.log` |
| Shell syntax, Python AST, ASCII/newline/whitespace and diff scope | exit 0 | `style-final-01.log`, `source-files-review.json`, `self-review.md` |

The adapter tests use real HTTP with `FakeBackend` producing synthetic vLLM-layout pages and recurrent tensors. They use the proxy's actual `Node`, proxy handler/state machine, release path and existing decoder fixture. `test_existing_proxy_namespace` reruns the existing node-facing namespace test against the adapter, using configured credentials and the native count/hash fields. The decoder fixture checks actual native checksums/journals; the separate native helper imports and round-trips real allocated 64-layer caches. The proxy HTTP fixture is not a numerical inference oracle. Source identity and HTTP hook control are independently tested without importing vLLM.

The full-directory type-check attempt exited 1: vLLM/Hugging Face GPU-only imports are absent, and older fixture typing also reports diagnostics. It is not reported as green. Scoped new production/test modules are green with the actual installed NumPy/CPU torch environment. The standalone editorconfig-checker is unavailable; scoped syntax/ASCII/whitespace checks are used and reported separately. Existing Python 3.14 sqlite ResourceWarnings in proxy fixtures remain visible; their tests exit 0.

## Narrowing controls

| Mutant | Narrows the gate to | Named failing test | Result / surviving bound |
|---|---|---|---|
| q8-half-ties | Exact half rounds toward zero | `FastCodecTests.test_rounding_and_refusals` | Killed, unittest exit 1 |
| prefix-guard | Caller n0/hash disagreement ignored | `AdapterTests.test_refusals_auth_release` | Killed, exit 1 |
| release-leak | DELETE acknowledges but keeps authority | `AdapterTests.test_refusals_auth_release` | Killed, exit 1 |
| leading-page-rows | Unaligned append emits old leading rows and fails native body sizing | `AdapterTests.test_fresh_append_reference_bytes` | Killed, exit 1 |
| proxy-chunk-alignment | Non-multiple requested chunk treated as exact export | `AdapterTests.test_proxy_fresh_append_tail_and_release` | Killed, exit 1 |
| hook-old-cursor | Endpoint capture uses an old request's range | `RuntimeTests.test_node_hook_endpoints_and_probe` | Killed, exit 1; CPU wiring only |
| fp4-mislabel | NVFP4 weights labelled FP8 because their KV configuration mentions FP8 | `RuntimeTests.test_deploy_source_identity` | Killed, exit 1 |
| native held-token-only checker | Shortened chain cannot evaluate its 18-token tail | `native_tail_probe.py` positive case | Killed, native exit 2 with `incomplete check chain` |

The unchanged copied candidate passes the proxy integration control, exit 0. Seven Python mutants were applied only to isolated source copies (`mutant_controls.py`, `mutant-controls-05.log`, `mutant-results.json`). The native mutant was built from a copied checker source (`mutant-build-01.log`, `control-native-held-token.log`). No changed-behavior mutant survived. These controls do not establish live CUDA ownership, arbitrary concurrent load or numerical equivalence.

## Orchestrator-reported base GPU result

Message `$rNqSRyxrz__UFzAb2BRX3UXNBVVmpTxZR2v2aqchwc8` from `agents1_hybrid_orch` in `#hybrid/prototype-go`, saved in `inbox-03.log`: base `m3-vllm-bridge` head `4a25d528`, B300, vLLM 0.32.0 FP8, `EXPORT_N=784`, `gpu-run all` exit 0. Equivalence at N+1 over 64 positions: BF16 top-1 64/64, mean KL 0.00055 nats; Q4_K_M top-1 64/64, mean KL 0.0027; no free-running divergence in either leg. Both checkers returned 5 (uncalibrated). The initial ambiguous FP8/NVFP4 cache selection refused; explicit `HF_MODEL` corrected it. A 31,360-token run was still in progress when reported.

These observations are attributed to the orchestrator; this agent did not execute or independently inspect the GPU logs. They qualify neither the new node lifecycle nor automatic torch CUDA conversion. The B300 runbook now requires the confirmed FP8 snapshot and a mandatory scalar/CUDA synthetic byte gate before node launch. No in-progress long run is represented as successful.

## Corrections and finite gaps

- Initial NumPy 2.2.6 selection had no Python 3.14 wheel and unintentionally started a source build. Its recorded PID tree was terminated; that attempt had not established the niced/-j4 controls and is not claimed compliant. Subsequent installs used binary-only NumPy 2.4.2, CPU torch 2.10.0 and ty 0.0.88 in the task-local venv. No vLLM install, model download, GPU use or source clone occurred.
- The first adapter test run had a test-only metering assumption (`remote_applied` is absent from the legacy raw-token fixture). It failed exit 1, was corrected to verify actual imported chunk ranges, and then passed. The first native negative probe expected exit 1; the checker correctly returned 2 with its exact refusal. The corrected probe passes and preserves the original log. The first half-tie plant matched two implementations and was not applied; a unique NumPy plant then ran and failed the named assertion.
- The brief has no formal surface table. Coverage is the explicit contract map above; there is no claim of full attack-surface coverage or independent acceptance. Production changes outside `tools/hybrid-vllm/` are limited to proxy/launcher/checker alignment compatibility, its existing namespace fixture and the root guide link. No src/include/ggml/server behavior changed.
- GPU tensor ownership, actual cache reuse/recompute, CUDA arithmetic equivalence, FP8-to-Q4 KL and 8k/32k offload timing are out of the agents1 execution contract (`CPU only`, with an orchestrator GPU runbook requested). Those gates remain open. No passing CPU test is promoted to GPU qualification.
- Signed commits preserve the requested co-author trailer. The converter progress notice was sent to `agents1_hybrid_orch` in `#hybrid`, topic `prototype-go`. Final handoff records the exact signed head after review/cleanup; no publication or merge is requested.

Task-local build trees, native helper/mutant binaries, probe deltas and the validation venv were removed before handoff (`cleanup-01.json`). Logs, source hashes, small control scripts/results and review notes remain under the task evidence directory. Sampled free space stayed above 28 GB; a late pre-cleanup sample was 34,603,261,952 bytes, and cleanup left 35,893,628,928 bytes free. Existing shared models/tools and GPU-host venv/cache are preserved.

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

## r8: Muse r7p rework (hybrid-d-r8-261102)

Ready for review on `m3-vllm-node`, based on reviewed head `7936196cf0a48a452a9a014abaaef4e86f6b9150`. CPU observations on agents1, 2026-11-02 UTC. No GPU, vLLM installation, model downloads, push or PR. All production and test changes remain under `tools/hybrid-vllm/`. The existing private-fork task authorizes the local signed commit and requested co-author trailer; this is not an upstream submission.

**Receiver binding.** `build_hashes` now derives the complete file-backed dependency closure from `ldd` on both actual receivers, including transitive dependencies and future project libraries outside bin. The receipt includes their resolved absolute paths and bytes, plus the existing sorted bin file set. Missing, relative, changed-path and changed-byte dependencies refuse. Whitespace in paths is preserved. The explicit static-project check also includes mtmd. Thus the reviewer ELF reproduction refuses even with a complete new receipt: it cannot execute a different outside-bin libmtmd. A named test confirms that the real receiver would load the outside marker, requires admission refusal, then freshly links static project code and requires admission. Future library names are separately tested with byte, path and missing-file plants and a restored positive control.

This intentionally also pins system/CUDA runtime files. Old bin-only receipts refuse; rebuild once rather than regenerate receipts for an unverified build. A runtime library upgrade requires rebuilding. Fresh receivers still use `BUILD_SHARED_LIBS=OFF` and `GGML_BACKEND_DL=OFF`. The existing exclusive-custody requirement between hash/inspection/execution remains.

**Source precision.** Only ignored `.pyc`/`.pyo` files directly under `__pycache__` are exempt from the ignored/local-exclude scans. The initial full-tree status still refuses tracked edits, including tracked bytecode edits. Ignored `.py`, `.cu`, `.gitignore`, untracked sources, local excludes and index flags remain checked. Each script launch creates a private empty Python cache prefix under disk TMPDIR with bytecode writes disabled. A timestamp-valid cache plant first proves that an ordinary interpreter executes its different cached value, then proves that the launched child reads the reviewed source instead. Admitting ignored caches therefore does not make them executable inputs to the script's children.

**Launch-bound CUDA gate.** Every CUDA `auto`/`torch` adapter startup calls `cuda_codec_check.check()` in the adapter process before opening its listener. An exception or non-passing result refuses startup; deploy cannot become ready without adapter health. Each launch repeats the check and never consumes an old receipt. The selected backend is frozen for this process. `scratch/codec-check.json` records source hashes, torch/CUDA versions, current device index/properties and the byte verdict. CPU auto selects NumPy once. `--codec reference` keeps scalar conversion available without importing torch; explicit NumPy remains available. The B300 runbook now relies on this mandatory launch check rather than a manual prerequisite. Fake-device tests exercise pass, mismatch exception and returned failure for both auto and torch, and assert that failures create neither a listener nor a verdict artifact. They do not execute or qualify CUDA arithmetic.

### r8 regression evidence

Logs and exact command/exit/PID/disk records are in `.temp/hybrid-d-r8-261102/`; commands ran as direct subprocesses, without tee. `validation-commands.txt`, `source-identity-final.json` and `implementation-notes.md` retain reproduction and source/configuration identities. The task-board is absent (`goal-read-02`: exit 1, missing `.task-board`); tracking stays in the existing `.temp/tasks.md`.

| Gate | Result / real exit | Evidence |
|---|---|---|
| Full bridge/script/codec/adapter/runtime discovery | 65 tests, no skips, exit 0 (prior 59 plus six r8 tests) | `vllm-tests-04.log` |
| Proxy regression discovery | 69 tests, exit 0 | `proxy-tests-01.log` |
| Bench regression discovery | 42 tests, exit 0 | `bench-tests-01.log` |
| Production gather, torch CPU | 2 tests, exit 0 | `gather-tests-01.log` |
| Fresh static CPU server, bench checker and prefill build | nice 10, at most -j4, exit 0; no project shared dependency in either receiver | `configure-01.log`, `build-01.log`, `linkage-01.log` |
| Native synthetic fresh/append imports and checksum-negative plant | Executed within the 65-test suite; helper build exit 0 | `native-build-01.log`, `test_native_roundtrip`, `test_fresh_append_reference_bytes` |
| Existing local 0.8B model, shortened native chain + 18-token tail | Positive checker exit 0, same-partition exact; no download | `native-tail-probe-01.log`, `native-tail-positive.result.json` |
| Wrong/absent native tail boundary | Both checker exits 2, exact `incomplete check chain`, no result files | `native-tail-wrong-boundary.log`, `native-tail-legacy.log` |
| Changed production/tests type check, Python 3.11 target | exit 0 | `type-check-04.log` |
| Bash/sh syntax, Python AST, ASCII/newlines/whitespace and retained original test names | exit 0 | `bash-syntax-01.log`, `sh-syntax-01.log`, `style-final-01.log` |

The existing scalar serialization reference now explicitly selects `reference`: its previous default selected NumPy after r7 and let a scalar token-order plant survive that particular witness. The named scalar witness now kills it, while existing fast-codec tests still compare complete NumPy envelopes with the scalar oracle. Positive HTTP operations now fail named assertions on unexpected protocol errors; refusal probes request binary responses so an accidentally accepted delta cannot be mistaken for the intended HTTP rejection through a JSON-size error. No original test name was removed.

### r8 narrowing controls

Each applied mutation lives in an isolated copy. The unchanged copied candidate passes. Each row records its real failing outcome; unittest assertion kills exit 1 and the native refusal kill exits 2. Expected-red is not presented as passing execution.

| Mutant | Narrows the gate to | Named failing test | Result |
|---|---|---|---|
| `fp4-mislabel` | labels NVFP4 weights as FP8 because their cache mentions FP8 | `test_node_runtime.RuntimeTests.test_deploy_source_identity` | Assertion kill, exit 1 |
| `q8-half-ties` | rounds exact halves toward zero | `test_fast_codec.FastCodecTests.test_rounding_and_refusals` | Assertion kill, exit 1 |
| `prefix-guard` | ignores caller n0/hash disagreement | `test_node_adapter.AdapterTests.test_refusals_auth_release` | Assertion kill, exit 1 |
| `release-leak` | acknowledges DELETE while retaining authority | `test_node_adapter.AdapterTests.test_refusals_auth_release` | Assertion kill, exit 1 |
| `leading-page-rows` | exports cached prefix rows before unaligned n0 | `test_node_adapter.AdapterTests.test_fresh_append_reference_bytes` | Assertion kill, exit 1 |
| `proxy-chunk-alignment` | assumes a non-multiple chunk end was exported exactly | `test_node_adapter.AdapterTests.test_proxy_fresh_append_tail_and_release` | Assertion kill, exit 1 |
| `hook-old-cursor` | reuses the offline bridge cursor instead of this request range | `test_node_runtime.RuntimeTests.test_node_hook_endpoints_and_probe` | Assertion kill, exit 1 |
| `r8-cache-read-enabled` | timestamp-valid cached code can replace reviewed Python source | `test_gpu_run.ScriptTests.test_bytecode_read_isolation` | Assertion kill, exit 1 |
| `r8-mtmd-static-gate` | shared mtmd can execute outside bin with a matching complete receipt | `test_gpu_run.ScriptTests.test_real_elf_mtmd_runpath_refusal` | Assertion kill, exit 1 |
| `r8-loader-closure` | future external libraries are excluded from receipt hashes | `test_gpu_run.ScriptTests.test_future_library_receipt_binding` | Assertion kill, exit 1 |
| `r8-cache-overrefusal` | ignored bytecode is treated as dirty source | `test_gpu_run.ScriptTests.test_ignored_bytecode_source_gate` | Assertion kill, exit 1 |
| `r8-cuda-check-skipped` | CUDA launches without executing byte equality | `test_node_runtime.RuntimeTests.test_node_cuda_codec_startup_gate` | Assertion kill, exit 1 |
| `r8-auto-not-frozen` | auto can select a different codec after admission | `test_node_runtime.RuntimeTests.test_node_cuda_codec_startup_gate` | Assertion kill, exit 1 |
| `native held-token-only checker` | Only held-token tail boundaries admitted | `native_tail_probe.py` positive shortened-chain case | Native refusal kill, exit 2 |

The unchanged r7 native held-token-only mutant was rebuilt from its retained source. `native_tail_probe.py`'s positive shortened-chain case passes on the candidate (exit 0) but exits 2 with `incomplete check chain` on the mutant, without writing a result (`native-held-token-01.log`, `native-mutant-verdict.json`). This is an intended native refusal kill, separate from unittest assertion kills.

Previous bridge/export and shell/gather plants were also replayed, with historical anchors updated only where r7 changed the same contract's implementation. Existing green named-test evidence is reused only for unchanged source/test/environment identities; changed scalar serialization witnesses were replayed. The complete per-mutant table is retained in `mutant-evidence.md` with the applied narrowing, failing test and finite survivor bound.

| Control family | Actual results | Evidence |
|---|---|---|
| Scalar codec/exporter | 24/24 named assertion kills; identity exit 0 | `codec-plants.json`, `codec-plants-03.log`, `codec-plants-04.log` |
| Previous shell controls | 36/37 named assertion kills, no ERRORs; identity exit 0 | `shell-mutants.json`, `shell-mutants-01.log` |
| Gather/rotated FP8 | 14/15 named assertion kills; identity exit 0; one ERROR-only detection uncredited | `gather-plants/plants.json`, `gather-plants-01.log` |

| Surviving or uncredited mutant | Narrows the gate to | Named failing assertion | Bound stated by survival |
|---|---|---|---|
| `interrupt_no_child_term` | Removes redundant interrupt-time TERM, keeps cleanup group TERM/KILL | None; exit 0 | The tested cleanup path still stops the recorded owned group within its deadline; it does not prove that this redundant early signal is necessary. |
| `upper_bound_off` | Removes the exporter's upper page bound | None; `GatherTests.test_refusals` errors, exit 1 | The mutant reaches a tensor indexing error rather than the intended early refusal. ERROR-only detection is not credited as a named assertion kill. |

### Corrections and finite scope

- `focused-tests-01` exited 1 because the existing copied-ELF test expected the old linkage refusal while the stronger new hash gate now rejects earlier. Its assertion now requires that exact hash refusal. Initial type checks (`type-check-01`/`03`, exits 1) exposed Optional setup handles in old tests; explicit setup assertions make the full changed-file type check green.
- `mutant-controls-01`/`02` exited 1 on inherited ERROR-only HTTP detections; those attempts are not assertion kills. The corrected positive/refusal assertions produce the named failures in `mutant-controls-03`. `codec-plants-01` stopped on an unapplied historical anchor (exit 1), then the scalar token-order witness survived its old default path in `codec-plants-02`; both are retained. Corrected anchors and explicit scalar selection produced the credited current results. No skipped plant, setup error or surviving witness is counted as a kill.
- No formal surface table was supplied (brief gap). The six new tests map the explicit loader, ignored-cache/read-isolation, CUDA launch and scalar/CPU requirements. No reviewer test was removed. Rework stays inside `tools/hybrid-vllm/`.
- CUDA arithmetic, live vLLM tensor ownership/cache reuse and FP8-to-Q4 numerical/8k/32k qualification remain outside this CPU-only execution contract and unrun. The synthetic CUDA gate still covers block size 16 and a fresh envelope; live captures and other block sizes need GPU evidence. CPU fake-device admission tests do not close those gates. Existing TOCTOU/exclusive custody and unmodified process-ownership limits remain.

Environment: Python 3.14.4, NumPy 2.4.2, torch 2.10.0+cpu and ty 0.0.88 in an isolated validation venv. CMake uses the verified read-only hybrid-a tool binary because it is absent from PATH. Binary-only, no-cache installs are recorded; no source dependency build was started. TMPDIR remained on disk, PYTHONDONTWRITEBYTECODE=1 throughout. Sampled free disk minimum was 32710434816 bytes (above 28 GB), including guarded concurrent validations. Builds were nice 10 and at most -j4; host build processes were sampled before and during work. Task-local build trees, venv, native binaries and large deltas are removed before signed handoff; shared models/tools are preserved.

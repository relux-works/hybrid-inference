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


## r9: current-main merge for landing (2026-11-02)

Continued the owned in-progress merge from the stopped first attempt; no abort or rebase. The reviewed parent is `c2f47472c0e6a5b239c7a00fe70a9a556f5accc1` (bridge/node r6-r8, accepted by r7q). A fresh fetch confirmed `origin/main` and `MERGE_HEAD` at `528724c96894aa5f65e98b329317de59aa694afc`; the merge base is `a3752b04cbd905bf61ffe6d9a4e7df738f26f4c2`. This is local signed delivery only: no push, PR submission or landing.

### Every conflict and resolution

- Textual conflict: `README.md` only. Retain the reviewed bridge/node guides and main's Mirai provenance/tools, local-kept latency and reference-scoring documentation, with a paragraph separator. The original `merge-01.log` and the independent patch replay both identify this one conflict.
- Semantic composition conflict: main's new `tools/hybrid-bench/score.cpp` requires an imported endpoint of N-1, while the reviewed aligned exporter can end earlier. `hybrid_bench.py` now supplies the actual `local_tail_start` to reference scoring. The scorer requires an integer marker equal to the imported endpoint, evaluates the missing local tail, and retains main's final one-token decode and partition probes. The absent-marker default remains N-1. Wrong/missing boundaries do not authorize implicit local fallback.

The automatic merges of `hybrid_bench.py`, `manage.py`, `proxy.py` and `test_proxy.py` retain both parents' changes. All bridge/adapter/runtime Python, C++, shell and CUDA-gate source bytes remain identical to c2f47472; the reviewed native local-tail checker is also byte-identical. No extra src/include/ggml/server changes were introduced by the resolutions.

### Delta reconstruction

Evidence is in `.temp/hybrid-d-r9-261102/`. `reviewed-branch.patch` is the complete binary patch from a3752b04 to c2f47472. Applying it with a separate Git index and `--cached --3way` to 528724c9 reports only the README conflict. Choosing the retained README resolution produces tree `9d18b479ef68a0ce9ca7ed1850efd4725a375872`, exactly equal to the continued staged merge tree before new resolutions.

`resolutions.patch` captures the complete remaining difference: the README separator, scorer/fixture composition, `test_aligned_reference_tail` and this r9 report. `proof.json` records the independent replay, tree equality and byte equality of the reconstructed `git diff 528724c9..HEAD` with `landing.diff`. Thus the final delta consists of the reviewed branch delta composed onto current main, plus these explicit resolutions and report. The final signed head is recorded in `landing-note.md` and `delivery.json` after signing.

### GPU evidence on the reviewed parent

The orchestrator's B300 smoke completed at 2026-11-02 03:37Z on exact head c2f47472, before this merge. Evidence source: `/home/agent/lanes/hybrid/.temp/node-smoke-b300/b300-bridge-node-evidence.tar.gz`. The retained `stage-node2.log` identifies the head, vLLM 0.32.0, successful codec checks at 03:04:16Z, benchmark exit 5 at 03:37:47Z, and node-run exit 0. The mandatory CUDA byte gate passed before readiness, as reported by the orchestrator. This r9 run inspected existing CPU-readable logs and measurements; it ran no GPU code.

The FP8 vLLM node was behind the adapter, with a llama.cpp Q4_K_M client, standard proxy and bench. All imports succeeded. These are heterogeneous, uncalibrated measurements; checker exit 5 is not an exactness pass.

| Context | Fresh KL, nats/token | Fresh top-1 | Append-chain KL | Append top-1 |
|---|---|---|---|---|
| 8k (8196 tokens) | 0.0054964635 | 63/64 | 0.0054964635 | 63/64 |
| 32k (32806 tokens) | 0.0256115405 | 60/64 | 0.0231588934 | 60/64 |

The exported endpoints were 7840 and 32144, with local tails of 356 and 662 tokens. The CUDA gate, tensor ownership and numerical observations belong to c2f47472, not to a new GPU run on the merged head. These results do not establish calibrated quality thresholds or the full acceptance matrix.

### CPU validation and real exit codes

Commands ran directly without tee, with on-disk `TMPDIR`, `PYTHONDONTWRITEBYTECODE=1`, nice 10 and CMake at -j4. CMake 4.4.4 is the verified existing read-only hybrid-a tool; it is absent from PATH. Exact arguments, PIDs, timestamps, resource observations and exits are retained in each `<label>.json` and `validation-ledger.json`.

| Gate / evidence label | Actual result / exit |
|---|---|
| Fresh fetch `fetch-02` | 0; exact requested main |
| CPU static build `build-02`, resolved scorer rebuild `build-03` | 0 / 0; CUDA and UI downloads disabled |
| Native helper `build-helper-02` | 0 |
| Full bridge/script/codec/adapter/runtime discovery `vllm-tests-01` | 1; 58 discovered entries, 54 passed, four NumPy dependency errors; not a green full suite |
| Runtime-only `runtime-tests-01` | 0; all 8 tests |
| Proxy discovery `proxy-tests-02` | 0; all 73 tests |
| Bench discovery after resolution `bench-tests-03` | 0; 68 tests, six native-header skips; explicit header suite below covers those cases |
| Production torch gather `gather-tests-02` | 1 before tests; torch absent |
| Reference native suite `reference-native-green-01` | 0; 42 recorded scorer invocations plus real native-node/checker orchestration, no skips |
| Integrated native bench/proxy `native-integrated-01` | 0; exact imported state, corrupt/checksum controls, timing, replay, budget refusal |
| Main heterogeneous-study native suite `native-hetero-study-01` | 0; real CPU reference/local/hybrid, resume/identity/negative controls |
| Existing 18-token local-tail probe `native-tail-probe-02` | Harness 0; positive checker 0 and same-partition exact; wrong/absent boundary each 2, exact refusal and no result |
| Explicit scalar 64-layer native probe `native-scalar-probe-01` | Harness 0; fresh/append helper 0/0; checksum plant 1 with intended refusal |
| Mirai Python/native identity `native-mirai-identity-02` | 0; all 12 tests using existing local headers, no skips |
| Mirai CPU op `native-mirai-op-02` | 0; CPU backend passes |
| Codec `native-codec-02` | 0; positive and rejection controls |
| Main file hash / SHA-NI `native-hash-02`, `native-hash-ni-02` | 0 / 0; 4 / 8 tests |
| Mirai parity `mirai-parity-tests-02`; eval `eval-tests-02` | 0 / 0; 6 / 11 tests |
| Shell syntax `shell-syntax-01`; diff whitespace `tracked-diff-check-01` | 0 / 0 |
| Workflow model naming `code-style-01` | 0; 166 mappings |

The reference regression uses an actual 16-token native export with a 19-token context, explicit three-token tail, deterministic repeat and an independent same-weight full-local oracle. Before/after/negative/past-context, float/bool/null and absent markers each produce scorer exit 2 with `incomplete score delta chain`, without result or temporary-output publication. Legacy N-1 orchestration still passes. All original test names from both parents remain present; no reviewer test was removed.

| Narrowing control | Narrows the gate to | Named failing test | Real result / bound |
|---|---|---|---|
| Unextended main scorer (retained before resolution) | Only N-1 imports admitted | `test_aligned_reference_tail` in `reference-native-red-02` | Harness 1, native scorer 2, exact incomplete-chain refusal, no output; resolved suite 0 |
| One-byte native-body plant | Reader accepts changed body bytes | Scalar native checksum case (same rejection as `BridgeTests.test_native_roundtrip`) | Native helper 1, exact checksum refusal; fresh/append positive controls 0 |

Corrections remain visible: first helper link `build-helper-01` exited 1 on an incorrect static-library path, then the actual CMake link layout passed. The stopped first build has no completion record; the first reference attempt was interrupted before the new witness, so neither is green evidence. The deliberate unextended-scorer regression above is expected-red, not passing execution. Broad eval discovery exited 1 on the pre-existing `test_native` module-name collision; the documented `test_eval.py` suite then passed. Six first native attempts were not launched because the admission guard exited 1 on another lane's active compiler; the -02 attempts waited and passed. No foreign process was stopped.

Before every native build/suite, admission records require `pgrep -u agent cc1plus` empty and one-minute load below 8. Builds ran one at a time in this lane and never overlapped another observed build at admission. The sampled minimum free disk after restart was 29398687744 bytes, above the 28 GB floor. Task-owned build trees, helpers and large native payloads are removed after hashing; existing shared tools/models and GPU evidence remain intact.

### Remaining validation scope

NumPy and torch are absent from system Python and the located local environments; no local wheels were found. The no-download rule was preserved. Two bridge cases and the fast-codec/adapter modules require NumPy, and gather requires torch: those exact full-suite gates remain open. The separately labeled scalar probe does not qualify NumPy/torch arithmetic or replace adapter tests. Type-checker and EditorConfig executables are also unavailable; syntax, ASCII, newline, whitespace and workflow naming checks are separate evidence, not substitutes for those tools. No formal surface table was supplied (brief gap). GPU execution on the merged head and the large acceptance matrix remain outside this CPU-only run.

### Landing note (draft only)

PR title: `hybrid: add vLLM node adapter with aligned exports`

Body draft:

> Add the reviewed vLLM bridge and HTTP node adapter (r6-r8, accepted by r7q) on current main 528724c9. Keep aligned exports and local decoder tails, and compose them with main's reference scorer through an exact explicit local-tail boundary. Preserve the scorer's final one-token decode and strict incomplete-chain refusals.
>
> CPU builds, runtime, proxy, bench, native reference/integration/study, scalar import, codec, identity and hash checks passed. Full NumPy-dependent bridge/fast-codec/adapter discovery and torch gather remain unverified because those dependencies are unavailable under the no-download rule. The reference regression retains valid legacy behavior and rejects eight invalid or missing boundary variants without output.
>
> Existing B300 evidence is bound to c2f47472: vLLM 0.32.0 FP8 node, Q4_K_M client, all imports succeeded; fresh 8k KL 0.0055 with 63/64 top-1, fresh 32k KL 0.0256 with 60/64. CUDA byte admission passed. These are heterogeneous, uncalibrated results, not GPU qualification of this merged head.
>
> Integration and conflict resolution were assisted by Codex. No publication was performed. Resolve the dependency validation gaps and review the exact signed merged head before landing.

Branch: `m3-vllm-node`. The exact signed head and this draft are in `.temp/hybrid-d-r9-261102/landing-note.md`, written after signing so the commit does not contain a self-referential ID. No session links are included.


## r10: second main merge after PR #14

Run: `hybrid-d-r10-261102`, 2026-11-02 UTC. Branch: `m3-vllm-node`. Merge first parent: `2614402ee9e1463f7766b39b8dbc96c03ed629ff`; pinned main second parent: `683caeed94c64ad69699d5f69a8fb533d4de6737`. The working tree started clean. No push, GPU execution or model download was performed. Explicit user instructions authorize this local signed merge, the requested co-author trailer, and the landing-note draft in this fork.

### Every merge conflict and resolution

| Path | Resolution |
|---|---|
| `tools/hybrid-bench/check.cpp` | Keep main's staged `delta_import` and committed-final endpoint check, then evaluate the branch's exact `local_tail_start` tail on imported, split and repeat contexts. |
| `tools/hybrid-bench/manage.py` | Keep both main's window-state policy/geometry accounting and the branch's validated `export_block_size` capability. |
| `tools/hybrid-proxy/proxy.py` | Keep main's backup/restore/abort transaction and unchanged committed slot during staging. Generate actual aligned chunk ranges; transfer count and final endpoint describe those ranges. Commit the session only after the final chunk, then run the local tail. |

These are the three textual conflicts recorded by `merge-01.log`. Main's native serialization, atomic chunk implementation and physical SWA layout are inherited unchanged; none of their `src/` files required conflict resolution.

Necessary semantic resolutions also update the bridge/adapter to HYBD v2 / `llama-native-chunks-v3`, with bounded, typed transfer metadata. Intermediate native bodies contain KV only; only a standalone or final chunk contains fixed recurrent state. Invalid versions, partial metadata and unaligned explicit transfer endpoints refuse before backend work or journal changes. Transfer IDs use the native reader's UTF-8 byte limit. Health advertises `delta_version=2`.

The benchmark counts the actual rounded chunk boundaries: block16/chunk31/endpoint48 requires three chunks. The reference scorer uses the production staged importer and refuses a chain without final fixed state before evaluating the exact local tail and held token. The allocated-cache adapter helper validates each chunk, checks unchanged committed cache positions during staging, atomically imports the chain and compares each chunk's exported native bytes. Protocol, tools and unresolved-GPU documentation describe the merged contract.

### Validation and limits

Every command, PID, real exit and sampled disk floor is persisted in `.temp/hybrid-d-r10-261102/validation-ledger.json` and the numbered command logs/JSON records. Commands execute directly with disk-backed `TMPDIR`, `PYTHONDONTWRITEBYTECODE=1` and niced CPU work. CPU configure/build/helper gates record `pgrep -u agent cc1plus` and `uptime` admission first; CMake builds use `nice -n 10` and `-j4`, one build at a time. Helper compilation waited for the other lane's compiler to finish. The existing CMake 4.4.4 binary is read from hybrid-a's task tools; no build tool was installed.

| Check | Result | Evidence |
|---|---|---|
| Static CPU named-target build; scorer and server-component incremental build; native helper | exit 0 each | `build-configure-01`, `build-01`, `build-final-01`, `build-helper-01` |
| Runtime | 8 tests, exit 0 | `runtime-tests-02` |
| Full bridge/node/converter/adapter/process discovery | 67 tests, exit 0; optional torch case skipped | `vllm-tests-01` |
| Final adapter tests, including 3-chunk native round-trip, checksummed early/missing fixed state and metadata refusals | 7 tests, exit 0 | `adapter-final-tests-01` |
| Fast codec | 4 tests, exit 0; optional torch case skipped | `codec-tests-01` |
| Proxy with system Jinja oracle | 123 tests, no skips, exit 0 | `proxy-system-tests-01` |
| Bench with SWA allocated-cache identity enabled | 81 tests, exit 0; six opt-in native-header discovery skips | `bench-full-tests-01` |
| Native header fixtures, covering those six discovery skips using existing local headers | 12 tests, no skips, exit 0 | `native-mirai-identity-01` |
| Physical SWA allocated-cache suite; hybrid chunk transaction suite; server components; native codec | exit 0 each | `native-swa-01`, `native-hybrid-chunks-01`, `native-server-component-01`, `native-codec-01` |
| Native node/proxy/checker integration; heterogeneous study | exit 0 each | `native-integrated-01`, `native-hetero-study-01` |
| Reference scoring, including aligned and chunked local tails, independent same-model local oracle, missing-final refusal and all prior boundary cases | exit 0 | `reference-native-01` |
| File hash and SHA-NI/portable hash suites | exit 0 each | `native-hash-01`, `native-hash-ni-01` |
| AST, ASCII, newline/whitespace, retained parent test names, shell syntax, exact workflow model naming and diff whitespace | pass; command gates exit 0 | `source-identity.json`, `retained-test-names.json`, `shell-syntax-01`, `code-style-01`, `diff-check-01` |

The NumPy-only venv was created with `python3 -m venv .temp/hybrid-d-r10-261102/venv`, then `venv/bin/python -m pip install --only-binary=:all: --no-deps numpy==2.4.2`. Its package inventory contains NumPy and bootstrap pip only. System Python and the existing local CPU environment have no torch. Gather and optional torch arithmetic were not run, and no torch was installed. System Python supplies the existing Jinja dependency for proxy/bench oracles; the isolated NumPy environment did not install it.

Expected-red and corrected attempts are retained: `adapter-red-01` exits 1 at the old adapter's intended HTTP400 refusal of the new chunk request. Premature `adapter-tests-01` exits 1 with two missing-native-helper setup errors and is uncredited; both subsequent adapter runs pass. Initial venv proxy/bench discovery skips Jinja controls; the system-Python runs cover them. Type and EditorConfig gates each exit 127 because `ty` and `editorconfig-checker` are unavailable. Their packages are outside this run's NumPy-only install authorization, so those gates remain unrun.

| Mutant | Gate narrowed to | Named failing test | Result / survival bound |
|---|---|---|---|
| Restore unrounded chunk-count ceiling | Count the actual aligned request ranges | `RuntimeTests.test_checker_aligned_tail` | Assertion failure, exit 1; no setup errors |
| Emit fixed state in every chunk | Fixed state only in the final chunk | `AdapterTests.test_chunked_final_fixed_state` | Assertion failure, exit 1; no setup errors |

Both in-memory mutants run in isolated processes; no source file was mutated. There are no survivors among these two controls. Correctly checksummed early/absent fixed-state input plants additionally fail the native chunk validator, while the neighboring complete chain imports and round-trips. Existing reviewer test names are retained. No formal surface table was supplied (brief gap). GPU/live vLLM extraction, merged-head numerical qualification, and downloads are outside the explicit CPU-only scope. Earlier B300 evidence remains bound to `c2f47472`, as recorded in r9; it is not a claim about this merge. No further product surface is treated as out of contract.

The sampled minimum free disk during recorded commands was 29,727,318,016 bytes, above 28 GB; the guard abort threshold was 28.1 GB and never triggered. Task-owned build trees, native helper, test payloads and NumPy venv are removed after hashing. `cleanup-01.json` records exact removals and disk readback; logs, JSON outcomes and hashes remain.

### Exact-diff comparison

`reviewed-branch.patch` is the reviewed `c2f47472` delta from its merge base `a3752b04`; SHA256 `649e5e20cf2ad0a7cd8521905a9a61cd2fb83f9e1d48d92160867dcd957e7132`, identical to the r9 reviewed patch. Replay it on pinned main in a separate index and select the reviewed stage-3 blobs for deterministic conflict baselines. Export all remaining differences as `all-resolutions.patch`, including inherited r9 integration and this run's explicit code/test/report resolutions. Applying that patch reconstructs the exact candidate tree; its `683caeed..tree` binary diff must be byte-identical to `git diff --binary 683caeed..HEAD` after signing.

An independent reconstruction starts from the r9 `528724c9..2614402e` landing patch and exports `r10-resolutions.patch`. Both equality checks, tree IDs, path inventories, patch/diff hashes and the exact signed head are in `proof.json`. The original-reviewed algebraic replay has an additional historical README conflict; it is not an extra conflict in this second merge. No unrelated working-tree change was included.

### Updated landing note

PR title: `hybrid: add aligned vLLM node with atomic chunk deltas`

Body draft:

> Add the reviewed vLLM bridge and HTTP node adapter on main 683caeed after PR #14. Emit HYBD v2 / llama-native-chunks-v3 with KV-only intermediate chunks and recurrent fixed state only in the final chunk. Preserve exact aligned exports and local tails in the proxy, native checker and reference scorer, with atomic client import and incomplete-chain refusal.
>
> CPU builds and runtime, proxy, bench, converter/adapter, native integration/reference/study, physical SWA, codec, identity and hash checks passed. The full bridge/node suite ran 67 tests with only optional torch arithmetic skipped. Gather was not run because torch was absent; NumPy 2.4.2 was the only installed test dependency. Type and EditorConfig checks could not run because their tools were unavailable.
>
> The merge and protocol resolutions were assisted by Codex. No push or PR submission was performed. Existing B300 evidence belongs to c2f47472, and does not qualify this merged head; GPU execution and merged-head numerical qualification remain pending. Review the exact signed head and the persisted reconstruction evidence before landing.


The exact signed head is written after signing to `.temp/hybrid-d-r10-261102/landing-note.md` and `proof.json`; this avoids a self-referential commit ID in a committed report. Both parents, signature verification and exact committed-tree equality are checked there. Ready for independent review; no publication or landing performed.

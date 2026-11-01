# Mirai S hybrid client port, r1

Status: ready for review. CPU evidence collected on agents1 on 2026-11-01. Evidence snapshot: 2026-11-01T16:56:33Z. No push, GPU execution, model download or quality ranking. The orchestrator's GPU greedy-token comparison remains a required qualification gate.

## Source and scope

- Assigned base: `a3752b04cbd905bf61ffe6d9a4e7df738f26f4c2`, branch `m3-mirai-s`, relux-works/hybrid-inference. Origin main was fetched and verified equal to HEAD before branching.
- Mirai remote: `https://github.com/alesha-pro/llama.cpp-mirai-s`. Ported Alexey Fateev's support from `fdb1e815dbbad220c2f41b94535049fdbd20c3ca`; included `d1c5a693444ee2bd13205bc1992e1853b1ef4426`'s `-np 1` documentation correction. No later control-vector, refusal-direction or abliteration changes are included.
- Port includes types 90-93, MIRAI_QUANTIZE/MIRAI_MUL_MAT, CPU reference and NVIDIA CUDA kernels, GGUF constants, row scales, split full-attention Q/gate, grouped SSM output input, converter/decoder/parity tools and backend tests.
- Deliberately omitted from fdb1e815db: `ggml/src/ggml-cuda/fattn-vec.cuh`, `ggml/src/ggml-cuda/gated_delta_net.cu`, `tools/server/server-common.cpp`, `tools/server/server-common.h`, `tools/server/server-context.cpp`. These implement unrelated FA GQA packing, chunked DeltaNet and text-aligned prompt caching. The orchestrator agreed to compare moderate contexts and attribute any divergence against this omission list.
- Conflict resolution retained F8_E4M3, TOP_K_MASK, Hadamard/Bonsai transforms, newer tensor mappings and shared graph fields. Qwen35 retains this base's 2D final linear-attention reshape. Mirai CUDA uses cuBLASLt; its NVIDIA-specific kernels and dispatch are guarded out for HIP/MUSA, and Mirai CUDA admission requires Ampere or newer. PTX instructions are architecture-guarded so older CUDA targets retain their ordinary paths. No CUDA/HIP/MUSA build is claimed here.

`tools/mirai-s/LICENSE-mirai_s-plugin` and `NOTICE` are retained; the Apache license text has LF line endings. NOTICE identifies the plugin and source port. The root MIT license is unchanged. These permissive components can coexist with their respective terms retained: Apache-2.0 section 4 permits distribution of modifications subject to license, change-notice and attribution conditions, and permits additional terms for one's modifications. The plugin-derived files retain Apache-2.0; they are not relabeled as solely MIT. Reference: [Apache-2.0 section 4](https://www.apache.org/licenses/LICENSE-2.0). No model-weight license or redistribution assessment is implied; no weights are included.

## Trunk identity and load contract

Native identity already iterates `hparams.n_layer()`, equal to raw `block_count - nextn_predict_layers`. Native KV and recurrent memory also iterate the effective trunk; no state serializer or envelope change was needed. Python now uses the same count and recurrent pattern for identity and sizing, while validating per-layer arrays against the raw count, including the inactive tail. Fingerprint version 1 and stock Qwen35 canonical JSON are unchanged.

For this file, 65 raw blocks minus one nextn block gives 64 trunk layers: 48 recurrent and 16 full-attention. Full-attention Q/gate splitting and per-row quantization scales do not change state geometry. Both native and Python identity yield:

```
d4fb520322100b1e438d60aa641ee86bd0d22576198a6dc49890c04f181924a8
```

This is equal for stock Qwen3.8-27B BF16, stock Q4_K_M and Mirai S. Q8_0 KV geometry is 34,816 bytes/token; fixed recurrent state is 156,893,184 bytes. MTP contributes neither a cache layer nor recurrent state to these numbers.

With `load_mtp=false`, Qwen35 now requires the trunk instead of accepting the MTP-only fallback. If a tail block is present, its recognized attention/FFN/nextn names and shapes must validate before skipping. The real blk.64 set has 15 tensors, including EH projection [10240,5120], full Q [5120,12288], K/V [5120,1024], output [6144,5120], FFN [5120,17408]/[17408,5120], normalization and nextn tensors. A completely absent tail remains a valid trunk-only export retaining the nextn declaration. A partial tail is refused. Native `done_getting_tensors()` still refuses unaccounted tensors; no partial-load option was introduced.

Python validates the recognized tensor table for nextn files and Mirai files, including required trunk tensors, split projections, row-scale shape/type and shared auxiliary tensors. This header admission is deliberately scoped to the dense Qwen35 variants exercised here; it does not generalize every optional upstream Qwen tensor form. The native loader remains authoritative. Invalid raw/nextn counts produce ordinary loader errors rather than assertion termination. The zero-trunk rejection is restricted to Qwen35 with MTP disabled, preserving other architectures and explicit MTP loading. Model SHA remains required by default; explicit `--hetero-quant` still requires equal nonempty layout fingerprints. The existing documented envelope SHA adaptation is unchanged; this port adds no fingerprint bypass.

The prefill node explicitly sets `load_mtp=false`; the client launcher explicitly supplies `--spec-type none`. The native fixture checker asserts that skipped MTP weights were not retained in the model's tail layers.

## Header fixture provenance

The orchestrator supplied exact header-only files from the GPU host, ending after the tensor table, with no data section. They remain outside git at `/home/agent/lanes/hybrid/.temp/research/bonsai/gguf-dumps/`.

| Header | Bytes | SHA-256 |
|---|---:|---|
| mirai-s.header.gguf | 11018205 | bb3d789a8c3e701c26267f556f5abcfe759f3ed775c3df125b5d012e413c3c24 |
| qwen3.8-27b-bf16.header.gguf | 10993991 | 9dacb72a2cf03ef77e6ad3a32b5765a1ff0d2b85e68e7fdecd82d63f5f8935ef |
| qwen3.8-27b-q4_k_m.header.gguf | 10993991 | 3a1fdf8b30b779852c16b2c04599e763a27252d3619a1dfe36ea9bbf08d9ff2e |

`fixtures/mirai-s-headers.json` retains hparams, vocabulary count/hash and all names/shapes/types from the real tensor tables, with source header hashes. It contains no tokenizer text, templates or weights. Native tests verify the external header hashes before using their exact ordered tokenizer bytes.

`llama-hybrid-check-identity --header-only` is a glibc-only test adapter: a bounded header stream reports the descriptor-derived logical end to the existing file-pointer loader, and fails every attempted weight read. It always sets `no_alloc=true`, `load_mtp=false`, GPU layers zero and mmap off. It creates no sparse multi-GB file. Normal `llama-hybrid-check-identity MODEL.gguf` uses the ordinary native file loader with simulated allocation.

Three exact, unmodified headers pass the native GGUF parser and model loader and equal Python's file-based identity. Separate fixtures rebuild contiguous descriptor offsets to exercise mutations. This validates native header/shape/tensor accounting, not the existence or correctness of weight payloads. Actual full-file hashes and inference remain GPU-host obligations.

## Validation

Evidence root: `.temp/hybrid-c-r1-261101/`. Compiler: GCC/G++ 15.2.0; Python 3.14.4; CMake 3.31.10; x86_64 CPU, two build jobs, nice priority and disk TMPDIR. CMake was initially absent. Only CMake was installed, into a task-local venv, using `python3 -m venv .temp/hybrid-c-r1-261101/venv` and `.temp/hybrid-c-r1-261101/venv/bin/python -m pip install cmake==3.31.10` (exit 0). No system installation or model download.

Commands below use `R=.temp/hybrid-c-r1-261101`, `C=$R/venv/bin/cmake`, `B=$R/build`, `H=/home/agent/lanes/hybrid/.temp/research/bonsai/gguf-dumps`, `TMPDIR=$PWD/$R/tmp`, and `OMP_NUM_THREADS=2`. Build and CPU test processes were run through `nice -n 10`.

| Command / evidence | Real exit | Claim and limit |
|---|---:|---|
| `$C -S . -B $B -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF` (`configure-01.log`) | 0 | CPU configuration, no UI/model download |
| `$C --build $B --target llama-hybrid-check-identity llama-hybrid-prefill llama-server test-backend-ops -j 2` (`build-05.log`); checker adapter rebuild (`build-06.log`), test rebuild (`build-07.log`), check/codec targets (`build-08.log`), scoped zero-trunk guard rebuild (`build-09.log`) | 0 each | Native CPU build, including actual server; no CUDA compilation |
| `$B/bin/test-backend-ops test -b CPU -o MIRAI_MUL_MAT` (`backend-03.log`) | 0 | 39/39: 33 ported random-code cases and six independent zero-tape expected-output cases |
| `python3 tools/hybrid-bench/test_mirai_identity.py --binary $B/bin/llama-hybrid-check-identity --headers $H --artifacts $R/identity-native-07` (`identity-native-07.log`) | 0 | 10/10 unittest methods, 31 native subprocess calls; exact native/Python equality, tail exclusion and 22 intended normal-error refusals |
| `PYTHONPATH=tools/hybrid-proxy:tools/hybrid-bench python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py` (`python-bench-04.log`) | 0 | 42/42, including explicit client `--spec-type none`, existing launch/identity/sizing contracts |
| `$B/bin/test-backend-ops test -b CPU -o GATED_DELTA_NET -p 'n_seq_tokens=1,n_seqs=1,'` (`gdn-regression-01.log`) | 0 | 5/5 selected CPU recurrence cases |
| `$B/bin/test-backend-ops test -b CPU -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=4,nr23=\[1,1\],kv=512,nb=1,'` (`fa-regression-01.log`) | 0 | 8/8 selected CPU attention cases; not the full backend matrix |
| `$B/bin/llama-hybrid-check-codec` (`codec-regression-01.log`) | 0 | Existing envelope positive/negative, SHA/layout/range/cap/checksum guards |
| `$B/bin/llama-hybrid-check /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf $R/qwen-golden 512` (`qwen-regression-01.log`; same test repeated after the scoped guard change in `qwen-regression-02.log`, without writing another golden directory) | 0 | Existing local model, no download; range append/local-prefill 64/64 greedy equality and unchanged-state rejection controls |
| Persisted 0.8B golden comparison (`qwen-golden-compare-01.log`) | 0 | Same model SHA, prompt IDs and all 64 greedy tokens as the pre-existing golden. Native state bytes differ from that older build; no historical byte-equivalence claim |
| Workflow's exact model-naming checker (`code-style-01.log`) | 0 | 166 mappings valid |
| `python3 -m py_compile` on all nine changed Python files (`python-syntax-01.log`) | 0 | Syntax only; converter dependencies and real conversion not exercised |
| `git diff --check HEAD` (`diff-check-03.log`), added-line ASCII scan | 0 | Reviewed delta whitespace and ASCII; pre-existing non-ASCII is retained |

The 22 malformed cases cover unknown tail/trunk tensors; missing first/later trunk tensors; MTP-only content; missing MTP EH/query; malformed MTP EH/query and trunk-gate shapes; missing/malformed/wrong-type row scales; missing/wrong-type rotation tensors; raw count zero/over 512; nextn count equal to raw/negative; short raw recurrent/head arrays; recurrent MTP. Each is paired with valid controls. Native rejection requires exit 1, the intended diagnostic and empty identity stdout; a crash or unrelated failure does not pass.

The 33 random-code cases compare the CPU implementation with the same CPU reference in this environment; they do not independently prove decoder arithmetic. Six additional cases use analytic zero-tape levels, positive/negative/zero constant inputs, identity rotations, signed row scales and an MS_I3 ladder. Their expected values do not call the Mirai decoder. They establish the exercised special cases, not arbitrary-code or CUDA equivalence.

### Bounded mutation controls

| Mutant | What the gate is narrowed to | Named failing test | Outcome / bound |
|---|---|---|---|
| Use raw count, retaining MTP in Python layout | Effective trunk count must remain 64 | `MiraiIdentityTests.test_real_header_layouts` | Killed, one assertion failure |
| Disable Python nextn tensor validation | Metadata geometry alone is insufficient admission | `MiraiIdentityTests.test_refusals` | Killed, 15 expected-rejection assertions fail |
| Ignore unknown-extra check | Every unconsumed tensor must be refused | `MiraiIdentityTests.test_refusals` | Killed, two expected-rejection assertions fail |
| Remove CPU trellis row-scale multiplication | CPU-to-CPU agreement cannot waive analytic output expectations | `MIRAI_MUL_MAT(type=ms_v4t8,k=5120,n=32,t=3,known=1)` and four other trellis known cases | Killed, actual backend command exit 1; MS_I3 remains green because its separate path was not mutated |

No tested mutant survived. Bounds remain finite: these four controls are not an exhaustive mutation campaign. CPU mutation was restored byte-for-byte, rebuilt (exit 0), and all 39 cases rerun green. Evidence: `mutants-python.json`, `mutants-cpu.json`, `mutant-*.log`, `backend-03.log`. Expected-red backend failure is recorded as exit 1, not a pass.

### Corrected failures and skipped checks

- Initial three-way application conflicted with this newer base. `build-01.log` exited 2 for a misplaced graph brace from conflict resolution; `build-02.log` exited 2 before regenerating the new CMake target; `build-03.log` exited 2 because `check_tensor_dims` requires its fourth argument. Corrected builds are green.
- First Python run failed on a metadata-only sizing fixture that lacks non-sizing head/FFN keys; validation now checks present raw arrays without adding stock sizing requirements. A subsequent launcher test caught the initially missing explicit speculative-decoding flag; it is now present and tested.
- First analytic sizing assertion used 157,286,400 bytes; the independently calculated correct recurrent allocation is 156,893,184. Corrected assertion and stock/Mirai equality are green.
- Initial synthetic header experiment with overlapping zero offsets was correctly refused by the GGUF parser. It was discarded. Current fixtures use contiguous offsets, and exact original headers are tested separately. No GGUF parser check was relaxed.
- `identity-native-03.log` exited 1 because two assertions expected the wrong wording for array-length errors. The loader had refused both correctly. The assertions now require its actual `wrong array length` diagnostic; full runs 04-07 pass.
- `ty` and `editorconfig-checker` are unavailable, so the exact hosted Python type check and EditorConfig binary were not run. Syntax, diff whitespace, ASCII and the workflow naming check passed. Full multi-platform CI, CUDA/Metal/HIP/MUSA builds, the converter's torch/numpy/safetensors execution and model numerical/performance suites are unrun.
- During verification, one short checker-only incremental build briefly overlapped the tail of the server build in the same task build tree. Both exited 0; all later validation used the restored, stable libraries. The full model build itself used two jobs. No other session's process was stopped.

## Review and remaining qualification

Self-review covered scope, security, approach, ggml/backend, model integration, tests and general conventions. The CPU codec and CUDA math retain the source implementation; adaptations add backend/architecture guards and preserve newer-base operations and transforms. No server prompt-cache, existing FA/GDN kernel, Gemma4, Muse Glimmer, KV-cache or recurrent-state implementation was edited. Native Qwen BF16/Q4 headers and the local 0.8B inference control are green. Gemma/Glimmer behavior preservation is supported by unchanged architecture/state paths and non-Mirai branch scoping, not real-model execution; this assigned base's Python launcher still refuses those architectures as before.

The brief supplies no normative surface table; this is a brief gap, not a claim of universal coverage. There are no previously supplied reviewer tests or a repository task-board in this checkout. Tracking and logs are in `.temp/tasks.md` and the evidence root. GPU parity and independent review are pending. No rework module boundary was supplied; root docs, ggml registration, model integration, hybrid launch/identity and tests are all required by this port.

GPU-host acceptance must build this signed head with CUDA, compare greedy token IDs against alesha-pro on identical prompts/flags and moderate contexts with MTP disabled, verify the two real model fingerprints, and exercise actual 27B hybrid state import. The source fork's omitted FA/GDN optimizations may alter near-tie numerics at long contexts; record any divergence instead of assigning a quality ranking. Full-file SHA checks and explicit hetero layout checks remain required.

Only after that qualification should Mirai S enter the measurement arms beside Bonsai 2. No KL, accuracy, throughput, memory-performance comparison or quality conclusion is reported by this CPU port.

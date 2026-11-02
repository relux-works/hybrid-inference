# vLLM prefill bridge prototype

CPU codec and a source-pinned post-forward hook for vLLM 0.32.0, one CUDA worker, dense Qwen35 27B, text-only prompts. GPU extraction and numerical equivalence are not qualified. See [DESIGN.md](DESIGN.md) for source evidence, limits and the GPU procedure, and [RESULTS-hybrid-d-r1-261101.md](RESULTS-hybrid-d-r1-261101.md) for the rework's CPU evidence.

The server adapter is specified in [NODE-PROTOCOL.md](NODE-PROTOCOL.md), with exact B300 commands in [GPU-NODE-RUNBOOK.md](GPU-NODE-RUNBOOK.md). The existing client launcher discovers its block alignment capability; the CLI and benchmark modes are unchanged. The proxy and checker have a small compatible extension for a short local tail. CPU measurements and remaining qualification gates are in [RESULTS-hybrid-d-r7-261101.md](RESULTS-hybrid-d-r7-261101.md).

## Tools

| Tool | Purpose | Command | Output |
|---|---|---|---|
| Python 3 + NumPy | Convert an exported capture | `python3 tools/hybrid-vllm/bridge.py CAPTURE/snapshot.json TARGET.json OUTPUT.delta --backend numpy` | Atomic HYBD delta; `reference` keeps the scalar oracle, `torch` requires CUDA, `auto` chooses CUDA when available |
| NumPy; optional torch | Byte equivalence and synthetic converter timing | `python3 -m unittest discover -s tools/hybrid-vllm -p test_fast_codec.py -v`; `python3 tools/hybrid-vllm/bench_converter.py --scratch .temp/TASK/tmp` | Test log and JSON timing rows; scratch tensors/deltas are removed |
| torch CUDA + NumPy, GPU host only | Scalar/CUDA page and full-envelope byte gate | Automatic during adapter `--codec auto`/`torch` startup on CUDA; standalone `python3 tools/hybrid-vllm/cuda_codec_check.py` | Node `scratch/codec-check.json`: source/runtime/device and byte verdict; scratch tensors removed; no model or vLLM import |
| Python 3 + vLLM venv | Supervise vLLM and the HTTP adapter | `bash tools/hybrid-vllm/deploy-vllm-node.sh start --state .temp/TASK/node --token-file PRIVATE.json`; replace `start` with `stop` or `status` | State directory: node identity, source hashes, PID/start-tick/group records, numbered logs, runner backup and cleanup results |
| Python 3 + NumPy | Adapter HTTP and hook/process CPU contracts with synthetic vLLM-layout pages | `python3 -m unittest discover -s tools/hybrid-vllm -p 'test_node_*.py' -v` | Logs in task `.temp/`; native byte checks require `HYBRID_VLLM_NATIVE_CHECK` |
| Python 3 | Install the source-pinned GPU runner hook | `python3 tools/hybrid-vllm/patch_vllm.py /path/to/vllm/v1/worker/gpu_model_runner.py` | One hook in that runner; matching reruns preserve bytes, other hashes or hook locations refuse |
| vLLM 0.32.0, torch 2.13.0 and their NumPy dependency | Copy live request pages and GDN endpoint state | Set `HYBRID_VLLM_CONFIG` and use the launch command in DESIGN | Private `capture-N/` directories published by rename after validation; failed staging directories removed |
| libxxhash | Existing protocol XXH3-128 checksum, through ctypes | Used by `bridge.py`; `python3 -c 'import ctypes.util; print(ctypes.util.find_library("xxhash"))'` | No extra artifact; library required on the converter machine |
| C++17 compiler and existing CPU llama/ggml build | Build native synthetic reader | Command below | `.temp/hybrid-a-r22-261101/native_check` |
| unittest | CPU conversion, negative and native import tests | Command below | Redirect stdout/stderr to numbered logs in the task's `.temp/` directory |
| Bash, CMake and CUDA 12.8 | Logged GPU-host preflight, exact-ref receiver build, capture and both target legs | `EXPECTED_REF=FULL_REVIEWED_R6_SHA bash tools/hybrid-vllm/gpu-run.sh all RUN_DIR` | `RUN_DIR`: command/server logs, source/build refs, bin file hashes, captures, deltas and checker/import evidence; reusable ref/hash receipts in `BUILD_DIR` |
| Git, realpath, find, sort, xargs, sha256sum, ldd and setsid | Source/artifact preflight and complete static-build receipt and loader check | Used by `gpu-run.sh build`, then checked before reuse | `BUILD_DIR/binaries.sha256`: sorted bin file hashes plus both receivers' resolved loader closures, following symlinks |
| torch 2.13.0 and NumPy, CPU only | Independent byte reference for the exporter's actual page gather | `bash tools/hybrid-vllm/gpu-run.sh gather RUN_DIR` | Mandatory first gate; failure stops every GPU subcommand before any build or launch |

From the repository root, with the existing CPU libraries built:

```sh
mkdir -p .temp/hybrid-a-r22-261101/tmp
export TMPDIR="$PWD/.temp/hybrid-a-r22-261101/tmp"
export PYTHONDONTWRITEBYTECODE=1
nice -n 10 c++ -std=c++17 -O1 -DGGML_BACKEND_SHARED -DGGML_SHARED -DLLAMA_SHARED \
  -Itools/hybrid -Isrc -Iinclude -Iggml/include -Ivendor \
  tools/hybrid-vllm/native_check.cpp tools/hybrid/hybrid-state.cpp \
  build/vendor/hash/libvendor-hash.a -Lbuild/bin -lllama -lggml -lggml-base -lggml-cpu \
  -Wl,-rpath,"$PWD/build/bin" -o .temp/hybrid-a-r22-261101/native_check
export HYBRID_VLLM_NATIVE_CHECK="$PWD/.temp/hybrid-a-r22-261101/native_check"
nice -n 10 python3 -m unittest discover -s tools/hybrid-vllm -p 'test_*.py' -v
```

The native test is mandatory: an absent helper fails the test. NumPy is required for the fast converter; torch and vLLM remain optional for CPU discovery. No GPU or model file is needed by this test command. Synthetic data uses the real 64-layer cache geometry; the native helper allocates about 184 MiB of CPU caches. It accepts only chains ending at or before 1024 tokens and is not a production import service.

`test_serialized_layout_reference` checks delta bytes against a separately written explicit Hadamard matrix, reference q8_0 encoder and source-layout recurrent permutation. Its files differ by layer, head, token and K/V. `test_fp8_rotated_bytes` independently checks non-unit K/V scales with rotation. Native round-trip checks serialization through allocated caches; it does not establish the source layout. `test_export_refusal_cleanup` uses CPU doubles for directory cleanup and publication. The separate `gather_test.py` uses real CPU tensors and the production page writer, including non-monotonic block tables and split kernel pages. It is outside stdlib-only discovery and is always run first by `gpu-run.sh`.

The GPU defaults match `/root/kit161/gpu/work/vllm-venv`, `/root/kit161/gpu/work/hf`, `/root/proto/models/` and sm_120a. See DESIGN for environment overrides and the one-block/two-block procedure. Execute the script with Bash; do not source it. All commands return to the caller on failure, with the failing command, exit and log path. No install or download is performed by the run script.

Use a fresh `RUN_DIR` for each capture attempt. Set `BUILD_DIR` to a previous build to verify its exact ref and complete file manifest and reuse it without CMake. The build uses `BUILD_SHARED_LIBS=OFF` and `GGML_BACKEND_DL=OFF`. The receipt binds every bin file and every file resolved by `ldd` on either receiver, including transitive dependencies outside bin, system/CUDA runtime libraries and future project libraries. Both bytes and resolved paths must match; missing or relative dependencies refuse. The known llama/ggml/mtmd libraries must also be static. Loader overrides `LD_PRELOAD` and `LD_LIBRARY_PATH` are unset before any child runs. Legacy receipts covering bin alone refuse; rebuild once rather than regenerate a receipt for an unverified build. Runtime library upgrades also require rebuilding. Only ignored `.pyc`/`.pyo` under `__pycache__` are exempt from ignored-source scans; tracked edits and ignored sources still refuse. Every script launch uses a fresh Python cache prefix with bytecode writes disabled, so admitted ignored caches cannot substitute executable code.

Node startup freezes `auto` to NumPy on CPU or runs the scalar/CUDA byte gate in the adapter process before selecting torch and opening its listener. No earlier verdict file authorizes a launch. A failed CUDA check refuses startup; `--codec reference` keeps scalar conversion available without importing torch, and `--codec numpy` selects CPU vectorization explicitly. See [GPU-NODE-RUNBOOK.md](GPU-NODE-RUNBOOK.md).

Tracked/untracked changes, ignored files under source directories and index flags that hide tracked state refuse. This includes a new CUDA source hidden by a self-ignoring `.gitignore`. Before building, in-repo run/build directories must be ignored. `BUILD_JOBS` defaults to the host's core count, capped at four on agents1. Request/health curl and CMake run in new process groups through `setsid`, with recorded leaders and interruptible waits. Cleanup ignores repeated INT/TERM/HUP/PIPE and restores the runner first. It stops the recorded server PID and active command group with TERM, then SIGKILL after `STOP_TIMEOUT` (120 seconds by default).

Captures contain token IDs and model state. Keep them in a private workbench, outside version control. The converter reads only single-component tensor paths and rejects symlinks, truncated files, unsupported dtypes and geometry. It streams the body to disk, checksums it, then replaces the destination atomically. It does not relabel one target GGUF as another: generate a target identity and a separate envelope for each target.

For isolated CPU validation, create a task-local environment and remove it after testing. This run used Python 3.14, `python3 -m venv .temp/hybrid-d-r7-261101/venv`, then `.temp/hybrid-d-r7-261101/venv/bin/python -m pip install --only-binary=:all: numpy==2.4.2 ty==0.0.88`. To exercise the torch arithmetic without CUDA, install `torch==2.10.0` from `https://download.pytorch.org/whl/cpu` using `--only-binary=:all:`. The GPU host uses its existing vLLM environment instead. These are validation dependencies, not extra runtime services. Set disk-backed `TMPDIR` and `PYTHONDONTWRITEBYTECODE=1` for every command.

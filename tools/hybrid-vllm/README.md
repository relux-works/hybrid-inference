# vLLM prefill bridge prototype

CPU codec and a source-pinned post-forward hook for vLLM 0.32.0, one CUDA worker, dense Qwen35 27B, text-only prompts. GPU extraction and numerical equivalence are not qualified. See [DESIGN.md](DESIGN.md) for source evidence, limits and the GPU procedure, and [RESULTS-hybrid-d-r1-261101.md](RESULTS-hybrid-d-r1-261101.md) for the rework's CPU evidence.

## Tools

| Tool | Purpose | Command | Output |
|---|---|---|---|
| Python 3 (standard library) | Convert an exported capture | `python3 tools/hybrid-vllm/bridge.py CAPTURE/snapshot.json TARGET.json OUTPUT.delta` | Atomic HYBD delta at the explicit output path |
| Python 3 | Install the source-pinned GPU runner hook | `python3 tools/hybrid-vllm/patch_vllm.py /path/to/vllm/v1/worker/gpu_model_runner.py` | One hook in that runner; matching reruns preserve bytes, other hashes or hook locations refuse |
| vLLM 0.32.0, torch 2.13.0 and their NumPy dependency | Copy live request pages and GDN endpoint state | Set `HYBRID_VLLM_CONFIG` and use the launch command in DESIGN | Private `capture-N/` directories published by rename after validation; failed staging directories removed |
| libxxhash | Existing protocol XXH3-128 checksum, through ctypes | Used by `bridge.py`; `python3 -c 'import ctypes.util; print(ctypes.util.find_library("xxhash"))'` | No extra artifact; library required on the converter machine |
| C++17 compiler and existing CPU llama/ggml build | Build native synthetic reader | Command below | `.temp/hybrid-a-r22-261101/native_check` |
| unittest | CPU conversion, negative and native import tests | Command below | Redirect stdout/stderr to numbered logs in the task's `.temp/` directory |
| Bash, CMake and CUDA 12.8 | Logged GPU-host preflight, exact-ref receiver build, capture and both target legs | `EXPECTED_REF=FULL_REVIEWED_R6_SHA bash tools/hybrid-vllm/gpu-run.sh all RUN_DIR` | `RUN_DIR`: command/server logs, source/build refs, bin file hashes, captures, deltas and checker/import evidence; reusable ref/hash receipts in `BUILD_DIR` |
| Git, realpath, find, sort, xargs, sha256sum, ldd and setsid | Source/artifact preflight and complete static-build receipt and loader check | Used by `gpu-run.sh build`, then checked before reuse | `BUILD_DIR/binaries.sha256`: sorted paths and content hashes for all regular files under bin, following symlinks |
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

The native test is mandatory: an absent helper fails the test. No torch, vLLM, NumPy, GPU or model file is needed by this test command. Synthetic data uses the real 64-layer cache geometry; the native helper allocates about 184 MiB of CPU caches. It accepts only chains ending at or before 1024 tokens and is not a production import service.

`test_serialized_layout_reference` checks delta bytes against a separately written explicit Hadamard matrix, reference q8_0 encoder and source-layout recurrent permutation. Its files differ by layer, head, token and K/V. `test_fp8_rotated_bytes` independently checks non-unit K/V scales with rotation. Native round-trip checks serialization through allocated caches; it does not establish the source layout. `test_export_refusal_cleanup` uses CPU doubles for directory cleanup and publication. The separate `gather_test.py` uses real CPU tensors and the production page writer, including non-monotonic block tables and split kernel pages. It is outside stdlib-only discovery and is always run first by `gpu-run.sh`.

The GPU defaults match `/root/kit161/gpu/work/vllm-venv`, `/root/kit161/gpu/work/hf`, `/root/proto/models/` and sm_120a. See DESIGN for environment overrides and the one-block/two-block procedure. Execute the script with Bash; do not source it. All commands return to the caller on failure, with the failing command, exit and log path. No install or download is performed by the run script.

Use a fresh `RUN_DIR` for each capture attempt. Set `BUILD_DIR` to a previous build to verify its exact ref and complete bin file manifest and reuse it without CMake. The build uses `BUILD_SHARED_LIBS=OFF` and `GGML_BACKEND_DL=OFF`. After hashes match, `ldd` must succeed for both receivers with no `libllama` or `libggml` shared dependency and no missing runtime library. Loader overrides `LD_PRELOAD` and `LD_LIBRARY_PATH` are unset before any child runs. Legacy shared builds refuse even with valid receipts; rebuild them. Followed symlink targets and the file set must still match the receipt.

Tracked/untracked changes, ignored files under source directories and index flags that hide tracked state refuse. This includes a new CUDA source hidden by a self-ignoring `.gitignore`. Before building, in-repo run/build directories must be ignored. `BUILD_JOBS` defaults to the host's core count, capped at four on agents1. Request/health curl and CMake run in new process groups through `setsid`, with recorded leaders and interruptible waits. Cleanup ignores repeated INT/TERM/HUP/PIPE and restores the runner first. It stops the recorded server PID and active command group with TERM, then SIGKILL after `STOP_TIMEOUT` (120 seconds by default).

Captures contain token IDs and model state. Keep them in a private workbench, outside version control. The converter reads only single-component tensor paths and rejects symlinks, truncated files, unsupported dtypes and geometry. It streams the body to disk, checksums it, then replaces the destination atomically. It does not relabel one target GGUF as another: generate a target identity and a separate envelope for each target.

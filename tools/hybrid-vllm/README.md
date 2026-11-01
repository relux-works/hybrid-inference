# vLLM prefill bridge prototype

CPU codec and a source-pinned post-forward hook for vLLM 0.32.0, one CUDA worker, dense Qwen35 27B, text-only prompts. GPU extraction and numerical equivalence are not qualified. See [DESIGN.md](DESIGN.md) for source evidence, limits and the GPU procedure, and [RESULTS-hybrid-a-r22-261101.md](RESULTS-hybrid-a-r22-261101.md) for CPU evidence.

## Tools

| Tool | Purpose | Command | Output |
|---|---|---|---|
| Python 3 (standard library) | Convert an exported capture | `python3 tools/hybrid-vllm/bridge.py CAPTURE/snapshot.json TARGET.json OUTPUT.delta` | Atomic HYBD delta at the explicit output path |
| Python 3 | Install the source-pinned GPU runner hook | `python3 tools/hybrid-vllm/patch_vllm.py /path/to/vllm/v1/worker/gpu_model_runner.py` | One hook in that runner; refuses other source hashes or a second installation |
| vLLM 0.32.0, torch 2.13.0 and their NumPy dependency | Copy live request pages and GDN endpoint state | Set `HYBRID_VLLM_CONFIG` and use the launch command in DESIGN | Private `capture-N/` directories under the configured output; manifest published last |
| libxxhash | Existing protocol XXH3-128 checksum, through ctypes | Used by `bridge.py`; `python3 -c 'import ctypes.util; print(ctypes.util.find_library("xxhash"))'` | No extra artifact; library required on the converter machine |
| C++17 compiler and existing CPU llama/ggml build | Build native synthetic reader | Command below | `.temp/hybrid-a-r22-261101/native_check` |
| unittest | CPU conversion, negative and native import tests | Command below | Redirect stdout/stderr to numbered logs in the task's `.temp/` directory |

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

Captures contain token IDs and model state. Keep them in a private workbench, outside version control. The converter reads only single-component tensor paths and rejects symlinks, truncated files, unsupported dtypes and geometry. It streams the body to disk, checksums it, then replaces the destination atomically. It does not relabel one target GGUF as another: generate a target identity and a separate envelope for each target.

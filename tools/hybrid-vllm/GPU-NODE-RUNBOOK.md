# B300 vLLM node + Q4 decoder qualification

Run from the transferred checkout root on the B300 host after the orchestrator transfers the signed `m3-vllm-node` head. This is a GPU runbook, not evidence of an executed GPU run. Set `EXPECTED_REF` to the exact 40-hex head supplied in the signed agents1 handoff. Keep the first bridge qualification's logs and stop its owned processes before starting this run. Do not weaken a runner-hash, geometry or capture refusal.

Prior evidence reported by `agents1_hybrid_orch` in `#hybrid/prototype-go`: base bridge `4a25d528`, vLLM 0.32.0 FP8, `EXPORT_N=784`, `gpu-run all` exit 0 on B300. At N+1 over 64 continuation positions, BF16 top-1 was 64/64 with mean KL 0.00055 nats; Q4_K_M top-1 was 64/64 with KL 0.0027; neither free-running continuation diverged. Both checkers exited 5 (uncalibrated). This is orchestrator-reported base-bridge evidence, not an independent observation or qualification of this adapter. A 31,360-token bridge run was still in progress when reported.

The first attempt correctly refused ambiguous HF cache selection because both FP8 and NVFP4 snapshots matched. Set `HF_MODEL` to the confirmed absolute FP8 snapshot from the successful bridge run's `environment.json`; do not select the first directory or substitute NVFP4. The adapter explicitly rejects FP4/NVFP4 weight configuration even if its KV quantization mentions FP8.

```sh
set -eu
: "${EXPECTED_REF:?set the full signed head from the handoff}"
: "${HF_MODEL:?set the confirmed FP8 snapshot from the passed bridge run}"
test -f "$HF_MODEL/config.json"
test "$(git rev-parse HEAD)" = "$EXPECTED_REF"
test -z "$(git status --porcelain --untracked-files=all)"
# The orchestrator verifies signatures on agents1; this host must have the same full SHA.
export RUN_DIR="$PWD/.temp/hybrid-d-r7-b300"
mkdir -m 700 "$RUN_DIR"
mkdir -m 700 "$RUN_DIR/tmp" "$RUN_DIR/results" "$RUN_DIR/check-scratch"
export TMPDIR="$RUN_DIR/tmp" PYTHONDONTWRITEBYTECODE=1
export VLLM_VENV=/root/kitb300/gpu/work/vllm-venv
export HF_HOME=/root/kitb300/gpu/work/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.9
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$VLLM_VENV/bin:$PATH"
unset LD_PRELOAD LD_LIBRARY_PATH
python3 --version
cmake --version
nvcc --version
nvidia-smi
"$VLLM_VENV/bin/python" -c 'import vllm,torch,numpy; print(vllm.__version__,torch.__version__,numpy.__version__); assert vllm.__version__ == "0.32.0"; assert torch.cuda.is_available()'
"$VLLM_VENV/bin/python" tools/hybrid-vllm/gather_test.py > "$RUN_DIR/gather-01.log" 2>&1
"$VLLM_VENV/bin/python" -m unittest discover -s tools/hybrid-vllm -p test_fast_codec.py -v > "$RUN_DIR/codec-cpu-01.log" 2>&1
"$VLLM_VENV/bin/python" tools/hybrid-vllm/cuda_codec_check.py > "$RUN_DIR/codec-cuda-01.log" 2>&1

nice -n 10 cmake -S . -B "$RUN_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=103a-real \
  -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  > "$RUN_DIR/configure-01.log" 2>&1

python3 tools/hybrid-bench/manage.py node --state "$RUN_DIR/token-setup" \
  --generate-token "$RUN_DIR/client-tokens.json"
bash tools/hybrid-vllm/deploy-vllm-node.sh start \
  --state "$RUN_DIR/node" --token-file "$RUN_DIR/client-tokens.json" \
  --venv "$VLLM_VENV" --hf-cache "$HF_HOME" --hf-model "$HF_MODEL" \
  --layout-model /root/models/Qwen3.8-27B-BF16.gguf \
  --context 65536 --codec auto --gpu-memory-utilization 0.45 \
  --port 8091 --vllm-port 8093 --startup-timeout 1800

nice -n 10 bash tools/hybrid-bench/setup-client.sh \
  --state "$RUN_DIR/client" --offline --cuda \
  --model /root/kitb300/gpu/work/models/Qwen3.8-27B-Q4_K_M.gguf \
  --node-info "$RUN_DIR/node/node-info.json" --token-file "$RUN_DIR/client-tokens.json" \
  --prefill-url http://127.0.0.1:8091 --hetero-quant --ctx 65536 \
  --request-timeout 1800 --startup-timeout 1800 --ref "$EXPECTED_REF" \
  --build-dir "$RUN_DIR/build" --port 8092 --decode-port 8090 \
  > "$RUN_DIR/client-start-01.log" 2>&1

status=0
python3 -c 'import os,pathlib,urllib.request; pathlib.Path(os.environ["RUN_DIR"],"vllm-metrics-before.txt").write_bytes(urllib.request.urlopen("http://127.0.0.1:8093/metrics",timeout=10).read())'
python3 tools/hybrid-bench/hybrid_bench.py \
  --client-state "$RUN_DIR/client" --output "$RUN_DIR/results" \
  --scratch "$RUN_DIR/check-scratch" --label vllm-b300-q4 \
  --smoke --depths 8192 32768 --modes local offload --check \
  --check-depths 8192 32768 --check-tokens 64 --request-timeout 1800 \
  > "$RUN_DIR/bench-01.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$RUN_DIR/bench-01.exit"
printf 'benchmark exit=%s (5 means uncalibrated, not passing exactness)\n' "$status"
python3 -c 'import os,pathlib,urllib.request; pathlib.Path(os.environ["RUN_DIR"],"vllm-metrics-after.txt").write_bytes(urllib.request.urlopen("http://127.0.0.1:8093/metrics",timeout=10).read())'

python3 tools/hybrid-bench/manage.py client --state "$RUN_DIR/client" --stop
bash tools/hybrid-vllm/deploy-vllm-node.sh stop --state "$RUN_DIR/node"
cat "$RUN_DIR/node/stopped.json"
nvidia-smi
```

On failure, execute both stop commands for the states that were created, preserve their logs and report the actual exit code. Do not delete the shared GPU-host venv or cached models. A stop refusal requires inspecting its recorded identity and backup; do not kill by process name.

Inspect the bundle's JSONL equivalence rows for `weight_comparison=heterogeneous`, `hetero_quant=true`, node `FP8` and client Q4 quantization, complete KL/top-1/logit-difference metrics and the actual `exported_endpoint`/`local_tail_tokens`. Require successful native imports and exact token/hash journals. Every offloaded call must have nonzero wire bytes and prefill/import/transfer timing; alignment-only short local tails are separately visible, and no refusal fallback is accepted as an offload measurement. Exit 5 remains an uncalibrated heterogeneous result. Missing metrics, imports, a release failure or an earlier exit are a failed/incomplete run, even if the servers became healthy.

The node hashes cached source weights once during startup. The BF16 GGUF is only its layout anchor; it is never loaded into a GPU receiver by this run. Both runtime memory reservations and the checker's five Q4 contexts must fit alongside the node; the utilization setting is a reservation request, not a measured peak. Record `nvidia-smi` before/after and actual peak memory. Report GPU tensor ownership, cache-hit/recompute counts and CUDA converter byte equivalence separately from CPU serialization results.

The mandatory `cuda_codec_check.py` command above compares actual CUDA FP8/BF16 page conversion, half ties and complete envelopes against the scalar oracle before automatic CUDA conversion is used. It must exit 0. A live capture still needs the same byte comparison after ownership/extraction qualification; preserve one private capture, convert it with `bridge.py ... --backend reference` and `--backend torch` using the same target/session, then `cmp` the files. `test_fast_codec.py` alone checks torch CPU arithmetic.

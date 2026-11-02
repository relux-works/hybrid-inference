# vLLM node + Q4 decoder qualification

## PRO 6000 re-smoke after PR #15 (r11)

Run the following Bash commands from the transferred, clean checkout on the PRO 6000 host. This block was prepared on a CPU-only host and has not been executed on a GPU. Set `EXPECTED_REF` to the full r11 signed merge head in `.temp/hybrid-d-r11-261102/landing-note.md`. The Q4 GGUF is also the layout anchor; no BF16 GGUF or download is required. The source-weight manifest supplies the distinct FP8 node identity. Keep the shared vLLM environment and model caches.

```bash
set -euo pipefail
umask 077
: "${EXPECTED_REF:?set the full signed r11 head from the handoff}"
test "$(git rev-parse HEAD)" = "$EXPECTED_REF"
test -z "$(git status --porcelain --untracked-files=all)"
export RUN_DIR="$PWD/.temp/hybrid-d-r11-pro6000-${EXPECTED_REF:0:12}"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR"/{tmp,results,check-scratch}
export TMPDIR="$RUN_DIR/tmp" PYTHONDONTWRITEBYTECODE=1
export VLLM_VENV=/root/kit161/gpu/work/vllm-venv
export HF_HOME=/root/kit161/gpu/work/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export Q4_GGUF=/root/proto/models/Qwen3.8-27B-Q4_K_M.gguf
test -f "$Q4_GGUF"
export PATH="$VLLM_VENV/bin:$PATH"
export PYTHONPATH="$PWD/tools/hybrid-vllm:$PWD/tools/hybrid-bench"
unset LD_PRELOAD LD_LIBRARY_PATH HYBRID_HASH_PORTABLE HF_MODEL HF_HUB_CACHE
export CUDACXX="$(command -v nvcc)"
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$CUDACXX")")")"
export HF_MODEL="$("$VLLM_VENV/bin/python" - <<'PY'
import json, os
from pathlib import Path
home = Path(os.environ['HF_HOME'])
candidates = set()
for cache in (home / 'hub', home):
    for config in cache.glob('models--*/snapshots/*/config.json'):
        raw = json.loads(config.read_text())
        text = raw.get('text_config', raw)
        quant = json.dumps(raw.get('quantization_config', {})).lower()
        if (raw.get('architectures') == ['Qwen3_5ForConditionalGeneration']
                and text.get('num_hidden_layers') == 64 and text.get('hidden_size') == 5120
                and 'fp8' in quant and 'fp4' not in quant):
            candidates.add(str(config.parent.resolve()))
assert len(candidates) == 1, 'expected one cached FP8-weight model: %r' % sorted(candidates)
print(candidates.pop())
PY
)"
test -f "$HF_MODEL/config.json"
git rev-parse HEAD > "$RUN_DIR/source.ref"
cmake --version > "$RUN_DIR/cmake-version.log"
nvcc --version > "$RUN_DIR/nvcc-version.log"
nvidia-smi > "$RUN_DIR/nvidia-before.log"
"$VLLM_VENV/bin/python" - <<'PY' > "$RUN_DIR/runtime-version.log"
import json, os, torch, vllm
from pathlib import Path
q = json.loads((Path(os.environ['HF_MODEL']) / 'config.json').read_text()).get('quantization_config', {})
assert 'fp8' in json.dumps(q).lower() and 'fp4' not in json.dumps(q).lower()
assert vllm.__version__ == '0.32.0' and torch.cuda.is_available()
assert torch.cuda.get_device_capability()[0] == 12
print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
print(os.environ['HF_MODEL'])
PY

# Gather is CPU-only even on this host. CUDA conversion has its separate gate.
CUDA_VISIBLE_DEVICES='' "$VLLM_VENV/bin/python" tools/hybrid-vllm/gather_test.py \
  > "$RUN_DIR/gather-01.log" 2>&1
"$VLLM_VENV/bin/python" tools/hybrid-vllm/cuda_codec_check.py \
  > "$RUN_DIR/cuda-byte-gate-01.json" 2> "$RUN_DIR/cuda-byte-gate-01.log"

# Do not compile while another lane is compiling.
pgrep -u "$(id -un)" cc1plus > "$RUN_DIR/cc1plus-before.log" && exit 1 || test "$?" -eq 1
uptime > "$RUN_DIR/uptime-before.log"
nice -n 10 cmake -S . -B "$RUN_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  > "$RUN_DIR/configure-01.log" 2>&1
nice -n 10 cmake --build "$RUN_DIR/build" -j4 --target \
  llama-server llama-hybrid-bench-check llama-hybrid-score \
  > "$RUN_DIR/build-01.log" 2>&1

# Exclude all TCP states, including TIME_WAIT, for both endpoints. Bind without
# SO_REUSEADDR as a second check; never restart using a prior run's ports.
"$VLLM_VENV/bin/python" - <<'PY' > "$RUN_DIR/ports.env"
import socket
from pathlib import Path
used = set()
for name in ('tcp', 'tcp6'):
    for line in Path('/proc/net/' + name).read_text().splitlines()[1:]:
        fields = line.split()
        used.update(int(fields[i].rsplit(':', 1)[1], 16) for i in (1, 2))
for base in range(18080, 28080, 4):
    ports = list(range(base, base + 4))
    if used.intersection(ports):
        continue
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(('127.0.0.1', port))
    except OSError:
        continue
    else:
        for key, port in zip(('NODE_PORT', 'VLLM_PORT', 'PROXY_PORT', 'DECODE_PORT'), ports):
            print('export %s=%d' % (key, port))
        break
    finally:
        for sock in sockets:
            sock.close()
else:
    raise SystemExit('no four TIME_WAIT-free ports; do not reuse occupied ports')
PY
source "$RUN_DIR/ports.env"

cleanup() {
  code=$?
  trap - EXIT
  if test -d "$RUN_DIR/client"; then
    python3 tools/hybrid-bench/manage.py client --state "$RUN_DIR/client" --stop \
      > "$RUN_DIR/client-stop-01.log" 2>&1 || code=1
  fi
  if test -d "$RUN_DIR/node"; then
    bash tools/hybrid-vllm/deploy-vllm-node.sh stop --state "$RUN_DIR/node" \
      > "$RUN_DIR/node-stop-01.log" 2>&1 || code=1
  fi
  nvidia-smi > "$RUN_DIR/nvidia-after.log" || code=1
  printf '%s\n' "$code" > "$RUN_DIR/run.exit"
  exit "$code"
}
trap cleanup EXIT
python3 tools/hybrid-bench/manage.py node --state "$RUN_DIR/token-setup" \
  --generate-token "$RUN_DIR/client-tokens.json" > "$RUN_DIR/token-setup-01.log" 2>&1
bash tools/hybrid-vllm/deploy-vllm-node.sh start \
  --state "$RUN_DIR/node" --token-file "$RUN_DIR/client-tokens.json" \
  --venv "$VLLM_VENV" --hf-cache "$HF_HOME" --hf-model "$HF_MODEL" \
  --layout-model "$Q4_GGUF" --context 65536 --codec torch \
  --gpu-memory-utilization 0.40 --port "$NODE_PORT" --vllm-port "$VLLM_PORT" \
  --startup-timeout 1800 > "$RUN_DIR/node-start-01.log" 2>&1

# Startup runs a new CUDA gate inside the adapter before opening its listener.
python3 - <<'PY'
import json, os
from pathlib import Path
r = Path(os.environ['RUN_DIR'])
receipt = json.loads((r / 'node/scratch/codec-check.json').read_text())
assert receipt['codec'] == 'torch' and receipt['cuda_check']['status'] == 'passed', receipt
PY
nice -n 10 bash tools/hybrid-bench/setup-client.sh \
  --state "$RUN_DIR/client" --offline --cuda \
  --model "$Q4_GGUF" --node-info "$RUN_DIR/node/node-info.json" \
  --token-file "$RUN_DIR/client-tokens.json" --prefill-url "http://127.0.0.1:$NODE_PORT" \
  --hetero-quant --ctx 65536 --extend-chunk-tokens 7840 \
  --request-timeout 1800 --startup-timeout 1800 --ref "$EXPECTED_REF" \
  --build-dir "$RUN_DIR/build" --port "$PROXY_PORT" --decode-port "$DECODE_PORT" \
  > "$RUN_DIR/client-start-01.log" 2>&1
python3 - <<'PY'
import os, urllib.request
from pathlib import Path
Path(os.environ['RUN_DIR'], 'vllm-metrics-before.txt').write_bytes(
    urllib.request.urlopen('http://127.0.0.1:' + os.environ['VLLM_PORT'] + '/metrics', timeout=10).read())
PY
bench_code=0
python3 tools/hybrid-bench/hybrid_bench.py \
  --client-state "$RUN_DIR/client" --output "$RUN_DIR/results" \
  --scratch "$RUN_DIR/check-scratch" --label vllm-pro6000-r11-q4 \
  --smoke --depths 8192 32768 --modes local offload --check \
  --check-depths 8192 32768 --check-tokens 64 --request-timeout 1800 \
  > "$RUN_DIR/bench-01.log" 2>&1 || bench_code=$?
printf '%s\n' "$bench_code" > "$RUN_DIR/bench-01.exit"
python3 - <<'PY'
import os, urllib.request
from pathlib import Path
Path(os.environ['RUN_DIR'], 'vllm-metrics-after.txt').write_bytes(
    urllib.request.urlopen('http://127.0.0.1:' + os.environ['VLLM_PORT'] + '/metrics', timeout=10).read())
PY
test "$bench_code" -eq 0 || test "$bench_code" -eq 5
python3 - <<'PY' > "$RUN_DIR/matrix-01.json"
import json, math, os
from pathlib import Path
r = Path(os.environ['RUN_DIR'])
rows = [json.loads(line) for p in (r / 'results').glob('*.jsonl') for line in p.read_text().splitlines()]
checks = [row for row in rows if row.get('stage') == 'equivalence']
assert len(checks) == 4, checks
assert {(row['check_mode'], row['depth'] >= 32768) for row in checks} == {
    (mode, large) for mode in ('fresh', 'append_chain') for large in (False, True)}
for row in checks:
    assert row['depth'] >= 8192 and row['k'] == 64
    assert row['checker_exit_code'] in (0, 5), row
    assert math.isfinite(row['mean_kl_nats']) and 0 <= row['top1_agreed'] <= 64
    assert row['weight_comparison'] == 'heterogeneous' and row['exported_endpoint'] > 0
    if row['depth'] >= 32768:
        assert row['delta_count'] > 1, row
transfers = [row['chunks'] for row in rows if row.get('stage') == 'call' and row.get('mode') == 'offload' and row.get('chunks')]
assert transfers
for chunks in transfers:
    assert chunks[-1]['status'] == 'imported', chunks
    assert all(chunk['status'] == 'staged' for chunk in chunks[:-1]), chunks
    assert all(chunk['wire_bytes'] > 0 for chunk in chunks), chunks
print(json.dumps(checks, indent=2))
PY
```

Require four equivalence rows: both `fresh` and `append_chain` at actual depths near 8k and 32k. The 32k rows must have `delta_count > 1`; `--extend-chunk-tokens 7840` forces this. Preserve `delta_count`, `append_endpoints`, exported endpoint, local-tail length, all 64-position KL/top-1/logit metrics, checker exits, import receipts and wire hashes. The existing checker imports the captured chunk chain through `delta_import`; its fixed state must occur only in each transfer's last chunk. Require intermediate offload chunks to report `status=staged` and the final chunk `status=imported`, all with nonzero wire bytes and without local fallback. Inspect cache hit/recompute metrics and the node's codec receipt alongside the numerical rows.

Benchmark exit 5 records uncalibrated FP8-to-Q4 measurements and is not an exactness pass. Even with that exit, missing rows, import refusals, CUDA-gate failure, missing KL/top-1, incomplete logs or cleanup errors leave this re-smoke incomplete. Preserve the exact source SHA and binary hashes with the bundle. If a port bind races, stop this run's recorded states and choose a new run directory and four new ports. Do not change shared runner ownership or kill processes by name. The 0.40 node memory fraction is a reservation request; record actual memory and refuse allocation failures instead of reducing coverage.

## Historical B300 runbook

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

With `--codec auto` or `--codec torch` on a CUDA device, the adapter now runs `cuda_codec_check.check()` in its own process before opening its listener. Startup cannot become ready unless actual CUDA FP8/BF16 page conversion, half ties and a complete envelope match the scalar oracle. A failed or interrupted check refuses startup. Each launch runs a new check; a manual run or an old receipt never authorizes it. Preserve `$RUN_DIR/node/scratch/codec-check.json`, which records the selected backend, source hashes, torch/CUDA versions and current device properties. The synthetic gate covers block size 16 and a fresh envelope; live capture and other block sizes still require GPU qualification.

Use `--codec reference` for scalar conversion without torch/CUDA imports, or `--codec numpy` for CPU vectorization. Neither mode claims CUDA byte qualification. CPU `auto` resolves once to NumPy. A live capture still needs the same byte comparison after ownership/extraction qualification; preserve one private capture, convert it with `bridge.py ... --backend reference` and `--backend torch` using the same target/session, then `cmp` the files. `test_fast_codec.py` alone checks torch CPU arithmetic.

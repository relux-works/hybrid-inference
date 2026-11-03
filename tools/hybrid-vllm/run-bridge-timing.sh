#!/usr/bin/env bash
# Rental only. All state and stop operations belong to this invocation.
set -euo pipefail
umask 077
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
owned_run=0
client_owned=0
node_owned=0
cleanup() {
  code=$?
  trap '' INT TERM HUP
  trap - EXIT
  if [ "$client_owned" = 1 ]; then
    python3 tools/hybrid-bench/manage.py client --state "$RUN_DIR/client" --stop > "$RUN_DIR/client-stop-01.log" 2>&1 || code=1
  fi
  if [ "$node_owned" = 1 ]; then
    bash tools/hybrid-vllm/deploy-vllm-node.sh stop --state "$RUN_DIR/node" > "$RUN_DIR/node-stop-01.log" 2>&1 || code=1
  fi
  if [ "$owned_run" = 1 ]; then printf '%s\n' "$code" > "$RUN_DIR/run.exit"; fi
  if [ "$code" = 0 ]; then echo 'done bridge-timing'; else echo "failed bridge-timing exit=$code"; fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
: "${EXPECTED_REF:?set the bridge timing signed head}"
: "${RUN_DIR:?set a new private absolute run directory}"
: "${VLLM_VENV:?set the existing pinned venv}"
: "${HF_HOME:?set the existing kit HF cache}"
: "${HF_MODEL:?set the exact FP8 snapshot path}"
: "${Q4_GGUF:?set the Qwen Q4_K_M GGUF}"
: "${CUDA_ARCH:?120a-real on PRO 6000; 103a-real on B300}"
test "$(git rev-parse HEAD)" = "$EXPECTED_REF"
test -z "$(git status --porcelain --untracked-files=all)"
test ! -e "$RUN_DIR"
mkdir -p "$RUN_DIR"/tmp
owned_run=1
export TMPDIR="$RUN_DIR/tmp" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="$VLLM_VENV/bin:$PATH"
unset LD_PRELOAD LD_LIBRARY_PATH
python3 -c 'import torch,vllm; assert vllm.__version__=="0.32.0"; assert torch.__version__.split("+")[0]=="2.13.0"; assert torch.cuda.is_available()' > "$RUN_DIR/readiness-01.log" 2>&1
nvidia-smi > "$RUN_DIR/gpu-before-01.log"
git rev-parse HEAD > "$RUN_DIR/source.ref"
python3 tools/hybrid-vllm/gather_test.py > "$RUN_DIR/gather-01.log" 2>&1
cmake -S . -B "$RUN_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF > "$RUN_DIR/configure-01.log" 2>&1
python3 - <<'PY' > "$RUN_DIR/ports.env"
import socket
from pathlib import Path
used = set()
for name in ('tcp','tcp6'):
    for line in Path('/proc/net/'+name).read_text().splitlines()[1:]:
        fields = line.split()
        used.update(int(fields[i].rsplit(':',1)[1],16) for i in (1,2))
for base in range(18080,28080,4):
    if used.intersection(range(base,base+4)): continue
    sockets=[]
    try:
        for port in range(base,base+4):
            s=socket.socket(); sockets.append(s); s.bind(('127.0.0.1',port))
        for key,port in zip(('NODE_PORT','VLLM_PORT','PROXY_PORT','DECODE_PORT'),range(base,base+4)):
            print('export %s=%d'%(key,port))
        break
    except OSError: pass
    finally:
        for s in sockets: s.close()
else: raise SystemExit('no unused ports')
PY
source "$RUN_DIR/ports.env"
python3 tools/hybrid-bench/manage.py node --state "$RUN_DIR/token-setup" --generate-token "$RUN_DIR/client-tokens.json" > "$RUN_DIR/token-setup-01.log" 2>&1
node_owned=1
bash tools/hybrid-vllm/deploy-vllm-node.sh start --state "$RUN_DIR/node" \
  --token-file "$RUN_DIR/client-tokens.json" --venv "$VLLM_VENV" --hf-cache "$HF_HOME" \
  --hf-model "$HF_MODEL" --layout-model "$Q4_GGUF" --context 147456 --codec torch \
  --gpu-memory-utilization 0.55 --port "$NODE_PORT" --vllm-port "$VLLM_PORT" --startup-timeout 1800 \
  > "$RUN_DIR/node-start-01.log" 2>&1
client_owned=1
bash tools/hybrid-bench/setup-client.sh --state "$RUN_DIR/client" --offline --cuda --model "$Q4_GGUF" \
  --node-info "$RUN_DIR/node/node-info.json" --token-file "$RUN_DIR/client-tokens.json" \
  --prefill-url "http://127.0.0.1:$NODE_PORT" --hetero-quant --ctx 147456 --extend-chunk-tokens 7840 \
  --request-timeout 1800 --startup-timeout 1800 --ref "$EXPECTED_REF" --build-dir "$RUN_DIR/build" \
  --port "$PROXY_PORT" --decode-port "$DECODE_PORT" > "$RUN_DIR/client-start-01.log" 2>&1
sha256sum "$RUN_DIR/build/bin/llama-server" > "$RUN_DIR/binary.sha256"
python3 tools/hybrid-vllm/bridge_timing.py --client-state "$RUN_DIR/client" --out "$RUN_DIR/timing" \
  > "$RUN_DIR/timing-01.log" 2>&1
nvidia-smi > "$RUN_DIR/gpu-after-01.log"

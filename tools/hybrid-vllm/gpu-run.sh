#!/usr/bin/env bash
# Execute with bash; sourcing would change the caller's shell options.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    printf 'Run: bash tools/hybrid-vllm/gpu-run.sh SUBCOMMAND [RUN_DIR]\n' >&2
    return 2
fi
set -Eeuo pipefail
umask 077
trap 'status=$?; printf "FAIL setup exit=%s line=%s command=%q\n" "$status" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR

usage() {
    cat <<'HELP'
Usage: EXPECTED_REF=<reviewed full SHA> bash tools/hybrid-vllm/gpu-run.sh COMMAND [RUN_DIR]
Commands: gather, build, capture, convert, check, import, all
all runs these in order and stops at the first failure. Each GPU command reruns gather.
RUN_DIR defaults to $RUN_DIR or .temp/hybrid-d-r2-261101-gpu under this checkout.
Use a new RUN_DIR for each capture attempt. No install or download is performed.
Set BUILD_DIR to a previous verified build to reuse it without rebuilding.
HELP
}
if [[ ${1:-} == --help || $# == 0 ]]; then usage; exit 0; fi
if (( $# > 2 )); then usage >&2; exit 2; fi
step=$1
case "$step" in gather|build|capture|convert|check|import|all) ;; *) usage >&2; exit 2 ;; esac
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
export RUN_DIR=${2:-${RUN_DIR:-$repo/.temp/hybrid-d-r2-261101-gpu}}
mkdir -p -- "$RUN_DIR"
RUN_DIR=$(cd -- "$RUN_DIR" && pwd)
export TMPDIR=${TMPDIR:-$RUN_DIR/tmp} PYTHONDONTWRITEBYTECODE=1
mkdir -p -- "$TMPDIR"
TMPDIR=$(cd -- "$TMPDIR" && pwd)
export HF_HOME=${HF_HOME:-/root/kit161/gpu/work/hf}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_VENV=${VLLM_VENV:-/root/kit161/gpu/work/vllm-venv}
export PYTHON=${PYTHON:-$VLLM_VENV/bin/python}
export PATH="$VLLM_VENV/bin:$PATH"
export BF16_GGUF=${BF16_GGUF:-/root/proto/models/Qwen3.8-27B-BF16.gguf}
export Q4_K_M_GGUF=${Q4_K_M_GGUF:-/root/proto/models/Qwen3.8-27B-Q4_K_M.gguf}
export TOKEN_FILE=${TOKEN_FILE:-$RUN_DIR/tokens.json}
export EXPORT_N=${EXPORT_N:-784} CONTEXT_SIZE=${CONTEXT_SIZE:-32768} PROBE_TOKEN=${PROBE_TOKEN:-1}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto} PREFILL_CHUNK=${PREFILL_CHUNK:-7840}
export VLLM_PORT=${VLLM_PORT:-8100} LLAMA_PORT=${LLAMA_PORT:-8101}
export HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-1800} REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1800}
export STOP_TIMEOUT=${STOP_TIMEOUT:-120}
export BUILD_DIR=${BUILD_DIR:-$RUN_DIR/build}
if [[ $BUILD_DIR != /* ]]; then BUILD_DIR=$repo/$BUILD_DIR; fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export CUDACXX=${CUDACXX:-$CUDA_HOME/bin/nvcc}
export CUDA_ARCH=${CUDA_ARCH:-120a-real}
export PYTHONPATH="$repo/tools/hybrid-vllm:$repo/tools/hybrid-bench${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V2_MODEL_RUNNER=0
exec 3>&2
log=$(mktemp "$RUN_DIR/$step-XXXXXX.log")
printf 'Log: %s\n' "$log" >&3
exec > "$log" 2>&1
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x
trap 'status=$?; printf "FAIL exit=%s line=%s command=%q log=%s\n" "$status" "$LINENO" "$BASH_COMMAND" "$log" >&3; tail -n 30 "$log" >&3; exit "$status"' ERR
owned_pid=
runner=
backup=
stop_pid() {
    local status=0 deadline=$((SECONDS + STOP_TIMEOUT))
    if kill -0 "$owned_pid" 2>/dev/null; then kill -TERM "$owned_pid" 2>/dev/null || :; fi
    while kill -0 "$owned_pid" 2>/dev/null; do
        if (( SECONDS >= deadline )); then
            printf 'Stop timeout after %s seconds; sending SIGKILL to owned PID %s\n' "$STOP_TIMEOUT" "$owned_pid"
            kill -KILL "$owned_pid" 2>/dev/null || :
            break
        fi
        sleep 1
    done
    wait "$owned_pid" || status=$?
    printf 'PID %s wait exit: %s\n' "$owned_pid" "$status"
    owned_pid=
    case "$status" in 0|137|143) return 0 ;; *) return "$status" ;; esac
}
cleanup() {
    local status=$?
    trap '' INT TERM HUP
    trap - EXIT ERR
    if [[ -n $backup ]]; then
        cp -p -- "$backup" "$runner" || status=1
        cmp -- "$backup" "$runner" || status=1
    fi
    if [[ -n $owned_pid ]]; then stop_pid || status=1; fi
    printf 'EXIT %s\n' "$status"
    printf '%s exit=%s log=%s\n' "$step" "$status" "$log" >&3
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

exact_ref() {
    local changes
    [[ ${EXPECTED_REF:-} =~ ^[0-9a-f]{40}$ ]]
    [[ $(git rev-parse HEAD) == "$EXPECTED_REF" ]]
    changes=$(git status --porcelain --untracked-files=all)
    if [[ -n $changes ]]; then
        printf 'Refusing dirty source tree (tracked or untracked files):\n%s\n' "$changes" >&2
        return 1
    fi
    printf '%s\n' "$EXPECTED_REF" > "$RUN_DIR/source.ref"
}
gather() {
    # This is the first gate, including on hosts without CUDA or vLLM imports.
    CUDA_VISIBLE_DEVICES='' "$PYTHON" "$repo/tools/hybrid-vllm/gather_test.py"
}
build() {
    exact_ref
    if [[ -e $BUILD_DIR ]]; then
        assert_build
        printf 'Reusing verified build: %s\n' "$BUILD_DIR"
        return
    fi
    local host
    host=$(hostname -s)
    if [[ -z ${BUILD_JOBS:-} ]]; then
        BUILD_JOBS=$(nproc)
        if [[ $host == agents1 || $host == relux-agents1 ]] && (( BUILD_JOBS > 4 )); then BUILD_JOBS=4; fi
    fi
    [[ $BUILD_JOBS =~ ^[1-9][0-9]*$ ]]
    if [[ $host == agents1 || $host == relux-agents1 ]]; then (( BUILD_JOBS <= 4 )); fi
    nice -n 10 cmake -S "$repo" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDAToolkit_ROOT="$CUDA_HOME" \
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_CURL=OFF
    nice -n 10 cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS" \
        --target llama-server llama-hybrid-bench-check
    exact_ref
    (cd -- "$BUILD_DIR" && sha256sum bin/llama-server bin/llama-hybrid-bench-check) > "$BUILD_DIR/binaries.sha256"
    printf '%s\n' "$EXPECTED_REF" > "$BUILD_DIR/build.ref"
    assert_build
}
assert_build() {
    local hashes
    exact_ref
    [[ $(cat "$BUILD_DIR/build.ref") == "$EXPECTED_REF" ]]
    hashes=$(cd -- "$BUILD_DIR" && sha256sum bin/llama-server bin/llama-hybrid-bench-check)
    if [[ $hashes != "$(cat "$BUILD_DIR/binaries.sha256")" ]]; then
        printf 'Receiver binary hashes FAILED: %s\n' "$BUILD_DIR" >&2
        return 1
    fi
    cp -- "$BUILD_DIR/build.ref" "$BUILD_DIR/binaries.sha256" "$RUN_DIR/"
}
wait_health() {
    local url=$1 deadline=$((SECONDS + HEALTH_TIMEOUT))
    while kill -0 "$owned_pid" 2>/dev/null; do
        if curl --fail --silent --max-time 5 "$url" > /dev/null; then return 0; fi
        if (( SECONDS >= deadline )); then
            printf 'Health timeout: %s\n' "$url" >&2
            return 1
        fi
        sleep 5
    done
    printf 'Server exited before health: PID %s\n' "$owned_pid" >&2
    return 1
}
capture() {
    assert_build
    [[ $STOP_TIMEOUT =~ ^[1-9][0-9]*$ ]]
    [[ ! -e $RUN_DIR/captures && ! -e $RUN_DIR/gpu_model_runner.py.original ]]
    "$PYTHON" "$repo/tools/hybrid-vllm/gpu_run.py" prepare
    runner=$("$PYTHON" - <<'PY'
import hashlib, importlib.util
from pathlib import Path
from patch_vllm import RUNNER_SHA256
path = Path(importlib.util.find_spec('vllm').origin).parent / 'v1/worker/gpu_model_runner.py'
assert hashlib.sha256(path.read_bytes()).hexdigest() == RUNNER_SHA256, 'require original pinned runner'
print(path)
PY
)
    cp -p -- "$runner" "$RUN_DIR/gpu_model_runner.py.original"
    backup=$RUN_DIR/gpu_model_runner.py.original
    "$PYTHON" "$repo/tools/hybrid-vllm/patch_vllm.py" "$runner"
    export HYBRID_VLLM_CONFIG=$RUN_DIR/bridge.json
    HF_MODEL=$(cat "$RUN_DIR/hf-model.path")
    "$VLLM_VENV/bin/vllm" serve "$HF_MODEL" --served-model-name bridge --host 127.0.0.1 --port "$VLLM_PORT" \
        --tensor-parallel-size 1 --pipeline-parallel-size 1 --max-num-seqs 1 \
        --dtype bfloat16 --kv-cache-dtype "$KV_CACHE_DTYPE" --enable-prefix-caching --mamba-cache-mode align \
        --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype float32 \
        --max-model-len "$CONTEXT_SIZE" --max-num-batched-tokens "$PREFILL_CHUNK" --enable-chunked-prefill \
        --no-async-scheduling --enforce-eager > "$RUN_DIR/vllm-01.log" 2>&1 &
    owned_pid=$!
    printf '%s\n' "$owned_pid" > "$RUN_DIR/vllm.pid"
    wait_health "http://127.0.0.1:$VLLM_PORT/health"
    curl --fail --max-time "$REQUEST_TIMEOUT" -H 'Content-Type: application/json' \
        --data-binary "@$RUN_DIR/request.json" "http://127.0.0.1:$VLLM_PORT/v1/completions" > "$RUN_DIR/response.json"
    cp -p -- "$backup" "$runner"
    cmp -- "$backup" "$runner"
    backup=
    stop_pid
    unset HYBRID_VLLM_CONFIG
    [[ -f $RUN_DIR/captures/capture-$(printf '%08d' "$EXPORT_N")/snapshot.json ]]
    # Require explicit confirmation before any receiver reserves GPU memory.
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$RUN_DIR/gpu-after-vllm.csv"
    [[ $(wc -l < "$RUN_DIR/gpu-after-vllm.csv") == 1 ]]
}
gpu_idle() {
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$RUN_DIR/gpu-before-receiver.csv"
    [[ $(wc -l < "$RUN_DIR/gpu-before-receiver.csv") == 1 ]]
}
convert() {
    exact_ref
    "$PYTHON" "$repo/tools/hybrid-vllm/gpu_run.py" convert bf16 "$BF16_GGUF"
    "$PYTHON" "$repo/tools/hybrid-vllm/gpu_run.py" convert q4_k_m "$Q4_K_M_GGUF"
}
check_target() {
    local quant=$1 model=$2 first status=0
    first=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["deltas"][0])' "$RUN_DIR/check-$quant.json")
    "$BUILD_DIR/bin/llama-hybrid-bench-check" "$model" "$RUN_DIR/check-$quant.json" \
        "$first" "$RUN_DIR/logits-$quant.json" 999 || status=$?
    printf '%s\n' "$status" > "$RUN_DIR/check-$quant.exit"
    printf '%s checker exit=%s (5 is uncalibrated, not passing)\n' "$quant" "$status" >&3
    case "$status" in 0|5) return 0 ;; *) return "$status" ;; esac
}
check() {
    assert_build
    gpu_idle
    check_target bf16 "$BF16_GGUF"
    check_target q4_k_m "$Q4_K_M_GGUF"
}
import_target() {
    local quant=$1 model=$2
    mkdir -- "$RUN_DIR/slots-$quant"
    "$BUILD_DIR/bin/llama-server" -m "$model" -ngl 999 -c "$CONTEXT_SIZE" -np 1 \
        -ctk q8_0 -ctv q8_0 -fa on --host 127.0.0.1 --port "$LLAMA_PORT" \
        --slots --slot-save-path "$RUN_DIR/slots-$quant/" > "$RUN_DIR/client-$quant.log" 2>&1 &
    owned_pid=$!
    printf '%s\n' "$owned_pid" > "$RUN_DIR/client-$quant.pid"
    wait_health "http://127.0.0.1:$LLAMA_PORT/health"
    "$PYTHON" "$repo/tools/hybrid-vllm/gpu_run.py" import "$quant"
    stop_pid
}
import_chains() {
    assert_build
    [[ $STOP_TIMEOUT =~ ^[1-9][0-9]*$ ]]
    gpu_idle
    import_target bf16 "$BF16_GGUF"
    import_target q4_k_m "$Q4_K_M_GGUF"
}

gather
case "$step" in
    gather) ;;
    build) build ;;
    capture) capture ;;
    convert) convert ;;
    check) check ;;
    import) import_chains ;;
    all) build; capture; convert; check; import_chains ;;
esac

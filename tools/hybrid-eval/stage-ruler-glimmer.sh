#!/usr/bin/env bash
# Run only on a fresh, dedicated RTX PRO 6000. See GLIMMER-RULER-RUNBOOK.md.
set -Eeuo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${RUN:-}
EXPECTED_HEAD=${EXPECTED_HEAD:-}
started=()
finish() {
    status=$?
    trap - EXIT INT TERM
    for ((i=${#started[@]}-1; i>=0; i--)); do
        state=${started[i]}
        launcher=setup-client.sh
        if [[ $state == node ]]; then launcher=deploy-prefill-node.sh; fi
        "$ROOT/tools/hybrid-bench/$launcher" --state "$RUN/$state" --stop >> "$RUN/logs/stop-$state.log" 2>&1 || status=1
    done
    if [[ $status == 0 ]]; then
        echo "stage ruler-glimmer done head=$EXPECTED_HEAD run=$RUN"
    else
        echo "stage ruler-glimmer failed exit=$status head=$EXPECTED_HEAD run=$RUN" >&2
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$ROOT"
[[ -n $RUN && -n $EXPECTED_HEAD ]] || { echo "Set RUN and EXPECTED_HEAD; see the runbook" >&2; exit 2; }
[[ $RUN == /* && ! -e $RUN ]]
[[ $(git rev-parse HEAD) == "$EXPECTED_HEAD" ]]
[[ -z $(git status --porcelain --untracked-files=no) ]]
# All GPU/network operations follow these admission checks.
for tool in git nvidia-smi python3 cmake nvcc nice pgrep sha256sum; do
    command -v "$tool" >/dev/null
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | python3 -c '
import sys
rows=sys.stdin.read().splitlines()
if len(rows)!=1 or "RTX PRO 6000" not in rows[0] or int(rows[0].split(",")[1].split()[0]) < 95000:
    raise SystemExit("dedicated 96 GB RTX PRO 6000 required")
'
python3 - "$RUN" <<'PY'
import pathlib, shutil, sys
parent=pathlib.Path(sys.argv[1]).parent
while not parent.exists():
    parent=parent.parent
if shutil.disk_usage(parent).free < 120_000_000_000:
    raise SystemExit('at least 120 GB free disk required')
PY
mkdir -p "$RUN/logs" "$RUN/tmp" "$RUN/models" "$RUN/archives"
export TMPDIR="$RUN/tmp"
python3 tools/hybrid-eval/ruler_glimmer.py > "$RUN/profile.json"
NATIVE_REF=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["native_ref"])' "$RUN/profile.json")
MODEL=$(python3 -c 'import json,sys; print(sys.argv[2]+"/models/"+json.load(open(sys.argv[1]))["model"]["file"])' "$RUN/profile.json" "$RUN")
python3 - "$RUN/profile.json" "$MODEL" <<'PY' > "$RUN/logs/model.log" 2>&1
import json, pathlib, sys
from types import SimpleNamespace
sys.path.insert(0, 'tools/hybrid-bench')
from bench_common import obtain_model, require
pin=json.load(open(sys.argv[1]))['model']
path=pathlib.Path(sys.argv[2])
url='https://huggingface.co/{repo}/resolve/{revision}/{file}'.format(**pin)
model, sha, _=obtain_model(SimpleNamespace(model=None, sha256=pin['sha256'], hf_repo=None, source_url=url), path.parent)
require(model.stat().st_size==pin['bytes'], 'model size mismatch')
model.replace(path)
print(sha, pin['bytes'], path.name)
PY
# The accepted Qwen base includes the fork's SWA restore/export repairs.
git worktree add --detach "$RUN/native-source" "$NATIVE_REF" > "$RUN/logs/source.log" 2>&1
NATIVE="$RUN/native-source"
BUILD="$RUN/build"
if pgrep -u "$(id -un)" cc1plus > "$RUN/logs/compilers.log"; then
    echo 'another native build is active' >&2
    exit 1
else
    probe_status=$?
    [[ $probe_status == 1 ]] || exit "$probe_status"
fi
nice -n 10 cmake -S "$NATIVE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_METAL=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=120a-real -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    > "$RUN/logs/configure.log" 2>&1
nice -n 10 cmake --build "$BUILD" -j4 --target llama-hybrid-prefill llama-server llama-hybrid-bench-check llama-hybrid-score \
    > "$RUN/logs/build.log" 2>&1
python3 -m venv "$RUN/venv" > "$RUN/logs/venv.log" 2>&1
PY="$RUN/venv/bin/python"
"$PY" -m pip install --only-binary=:all: -r tools/hybrid-eval/ruler-requirements.txt > "$RUN/logs/pip.log" 2>&1
"$PY" tools/hybrid-eval/ruler_data.py bootstrap --gpu-host-downloads --data "$RUN/data" \
    > "$RUN/input-lock.sha256" 2> "$RUN/logs/bootstrap.log"
"$PY" tools/hybrid-eval/ruler_data.py verify --data "$RUN/data" > "$RUN/logs/data-verify.log" 2>&1
INPUT_SHA=$(cat "$RUN/input-lock.sha256")
common=(--offline --ref "$NATIVE_REF" --build-dir "$BUILD" --build-lock "$RUN/build.lock" --model "$MODEL" --cuda
        --ctx 131072 --request-timeout 7200 --startup-timeout 1800)
# The standard launchers generate their own build records after an incremental build.
nice -n 10 "$NATIVE/tools/hybrid-bench/deploy-prefill-node.sh" "${common[@]}" --state "$RUN/node" > "$RUN/logs/node.log" 2>&1
started+=(node)
nice -n 10 "$NATIVE/tools/hybrid-bench/setup-client.sh" "${common[@]}" --batch 64 --ubatch 64 \
    --node-info "$RUN/node/node-info.json" --token-file "$RUN/node/client-tokens.json" --state "$RUN/client" > "$RUN/logs/client.log" 2>&1
started+=(client)
nice -n 10 "$NATIVE/tools/hybrid-bench/setup-client.sh" "${common[@]}" --batch 256 --ubatch 256 --decode-port 8093 --port 8094 \
    --node-info "$RUN/node/node-info.json" --token-file "$RUN/node/client-tokens.json" --state "$RUN/partition" > "$RUN/logs/partition.log" 2>&1
started+=(partition)
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RUN/logs/vram-services.log"
for depth in 8192 32768 65536 122880; do
    size=$("$PY" tools/hybrid-eval/ruler_glimmer.py --depth "$depth")
    evaluation=(--model-profile glimmer30 --client-state "$RUN/client" --partition-client-state "$RUN/partition" --data "$RUN/data"
                --input-lock-sha256 "$INPUT_SHA" --depth "$depth" --saved-session-bytes "$size" --seed 261102 --request-timeout 7200)
    for phase in pilot results; do
        extra=(--samples-per-task 30 --margin .02)
        if [[ $phase == pilot ]]; then extra=(--tasks niah_single_1 qa_1 --samples-per-task 1); fi
        nice -n 10 "$PY" tools/hybrid-eval/ruler_eval.py "${evaluation[@]}" "${extra[@]}" --output "$RUN/$phase" \
            > "$RUN/logs/$phase-$depth.log" 2>&1
        (cd "$RUN/$phase/depth-$depth" && sha256sum -c complete.tar.gz.sha256) > "$RUN/logs/archive-$phase-$depth.log" 2>&1
        cp "$RUN/$phase/depth-$depth/complete.tar.gz" "$RUN/archives/$phase-$depth.tar.gz"
        (cd "$RUN/archives" && sha256sum "$phase-$depth.tar.gz" > "$phase-$depth.tar.gz.sha256")
        echo "stage ruler-glimmer $phase depth=$depth done"
    done
done
cat "$RUN/results"/depth-*/rows.jsonl > "$RUN/paired-all-depths.jsonl"
"$PY" tools/hybrid-eval/ruler_analysis.py "$RUN/paired-all-depths.jsonl" --output "$RUN/analysis" --margin .02 \
    > "$RUN/logs/analysis.log" 2>&1

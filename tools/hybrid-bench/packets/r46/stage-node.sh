#!/bin/bash
set -eu
umask 077
model_kind=$1
shift
case "$model_kind" in
    qwen) context=196608; architecture=qwen35; expected_sha=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34 ;;
    glimmer) context=131072; architecture=muse-glimmer; expected_sha=4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e ;;
    *) exit 2 ;;
esac
: "${MODEL:?absolute staged GGUF path required}" "${MODEL_SHA256:?trusted full hash required}" "${R46_HEAD:?signed r46 head required}"
test "$MODEL_SHA256" = "$expected_sha" || { echo "wrong target GGUF hash" >&2; exit 2; }
test "$(git rev-parse HEAD)" = "$R46_HEAD" || { echo "wrong source head" >&2; exit 2; }
export MODEL MODEL_SHA256
PYTHONDONTWRITEBYTECODE=1 python3 - "$architecture" <<'PY'
import os, sys
sys.path.insert(0,'tools/hybrid-bench')
from bench_common import model_identity, sha256, require
p=os.environ['MODEL']
x=model_identity(p)['model_layout']
require(sha256(p)==os.environ['MODEL_SHA256'], 'model hash mismatch')
require(x['architecture']==sys.argv[1], 'wrong node model')
if sys.argv[1]=='muse-glimmer':
    require(x['sliding_window']==2048 and not any(r['recurrent'] for r in x['layers']), 'wrong Glimmer state layout')
PY
exec tools/hybrid-bench/deploy-prefill-node.sh --model "$MODEL" --sha256 "$MODEL_SHA256" --ctx "$context" --cuda --bench-link --ref "$R46_HEAD" --bind 127.0.0.1 --port 8091 --startup-timeout 3600 --offline "$@"

#!/bin/bash
set -eu
: "${BF16_NODE_STATE:?absolute task-owned node state required}"
bash "$(dirname "$0")/stage-qwen-node.sh" --state "$BF16_NODE_STATE" "$@"
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import json,os,urllib.request
with urllib.request.urlopen('http://127.0.0.1:8091/v1/health',timeout=30) as response:
    health=json.load(response)
if health['model_sha256']!=os.environ['MODEL_SHA256'] or not {'f32','bf16'}<=set(health.get('state_dtypes',[])):
    raise SystemExit('node lacks matching f32/bf16 transport capability')
print('matching Qwen node advertises f32/bf16; native compute remains f32')
PY

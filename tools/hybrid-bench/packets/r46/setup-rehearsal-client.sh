#!/bin/bash
set -eu
: "${MODEL:?staged GGUF required}" "${MODEL_SHA256:?trusted hash required}" "${R46_HEAD:?signed head required}"
: "${NODE_ROOT:?node task root required}" "${CLIENT_ROOT:?client task root required}" "${BUILD:?shared build directory required}"
case "$1" in
    qwen) context=196608; state=${BF16_NODE_STATE:?Qwen bf16 node state required} ;;
    glimmer) context=131072; state=$NODE_ROOT/glimmer-node ;;
    *) exit 2 ;;
esac
test "$(git rev-parse HEAD)" = "$R46_HEAD" || { echo 'wrong source head' >&2; exit 2; }
# The node stamp covers prefill only. The client launcher builds and records its own targets under the same lock.
exec tools/hybrid-bench/setup-client.sh --state "$CLIENT_ROOT/client" --model "$MODEL" --sha256 "$MODEL_SHA256" \
    --ctx "$context" --cuda --node-info "$state/node-info.json" --token-file "$state/client-tokens.json" \
    --prefill-url http://127.0.0.1:8091 --request-timeout 3600 --startup-timeout 3600 \
    --ref "$R46_HEAD" --build-dir "$BUILD" --build-lock "$NODE_ROOT/build.lock" --offline

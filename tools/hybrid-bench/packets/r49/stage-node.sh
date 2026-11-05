#!/bin/bash
set -eu
umask 077
: "${R49_HEAD:?exact reviewed source head}" "${MODEL:?staged GGUF path}" "${RUN_ROOT:?new private node run root}"
: "${BUILD:?node build path}" "${NODE_IP:?public IPv4 used by Mac}"
test "$(git rev-parse HEAD)" = "$R49_HEAD"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT/tmp"
export TMPDIR="$RUN_ROOT/tmp" PYTHONDONTWRITEBYTECODE=1
model_sha=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
# The native launcher remains loopback-only. It records its exact-ref prefill target under this build lock.
bash tools/hybrid-bench/deploy-prefill-node.sh --state "$RUN_ROOT/node" --model "$MODEL" --sha256 "$model_sha" \
    --ctx 229376 --cuda --bench-link --ref "$R49_HEAD" --build-dir "$BUILD" --build-lock "$RUN_ROOT/build.lock" \
    --bind 127.0.0.1 --port 8091 --startup-timeout 3600 --offline
openssl req -x509 -newkey rsa:3072 -nodes -days 3 -subj "/CN=$NODE_IP" \
    -addext "subjectAltName=IP:$NODE_IP" -keyout "$RUN_ROOT/tls-key.pem" -out "$RUN_ROOT/tls-cert.pem" > "$RUN_ROOT/tls-setup.log" 2>&1
chmod 600 "$RUN_ROOT/tls-key.pem"
python3 tools/hybrid-proxy/parallel_transport.py --upstream http://127.0.0.1:8091 --port 9443 \
    --token-file "$RUN_ROOT/node/client-tokens.json" --cert "$RUN_ROOT/tls-cert.pem" --key "$RUN_ROOT/tls-key.pem" \
    > "$RUN_ROOT/gateway.log" 2>&1 &
gateway_pid=$!
printf '%s\n' "$gateway_pid" > "$RUN_ROOT/gateway.pid"
# Deliver only the public certificate, node-info.json and private client token file by authenticated SSH.

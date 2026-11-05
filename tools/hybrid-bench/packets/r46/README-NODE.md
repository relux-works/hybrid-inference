# r46 fresh RTX PRO 6000 node stage

Use one dedicated RTX PRO 6000 96GB, existing CUDA12.8+ toolchain and a staged, owner-pinned Q4_K_M GGUF. This packet does not provision hosts, download weights or start a GPU on agents1. Set MODEL (absolute GGUF path), MODEL_SHA256 (trusted full hash), R46_HEAD (delivered signed commit) and NODE_ROOT (new private task directory under .temp). Work in the clean r46 source. The wrappers pin Qwen SHA25631629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34 and Glimmer SHA2564cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e (the kit1.6.5 Glimmer MANIFEST), and verify hash/architecture; Glimmer additionally requires SWA2048 with no recurrent layer. Qwen context is196608; Glimmer context is131072. Native context limits are never extended with YaRN.

With 96GB host RAM and 96GB GPU RAM, allow 120GB free disk for the Glimmer checkpoint rehearsal or 45GB for a timing-only Glimmer node, after weights are staged; Qwen timing node reserves80GB. Fresh builds need another10GB. Actual launcher geometry/memory admission is authoritative: no overrides to force fit. Keep node through client final cleanup. Use a separate state per model, stop only the node owned by that state, and run the models sequentially.

```bash
set -eu
umask 077
mkdir -p "$NODE_ROOT/tmp"
export TMPDIR="$NODE_ROOT/tmp" PYTHONDONTWRITEBYTECODE=1
export LLAMA_HYBRID_TRACE="$NODE_ROOT/native-%p.jsonl"
nvidia-smi > "$NODE_ROOT/gpu-01.log"
bash tools/hybrid-bench/packets/r46/stage-qwen-node.sh --state "$NODE_ROOT/qwen-node" --build-dir "$NODE_ROOT/build" --build-lock "$NODE_ROOT/build.lock" > "$NODE_ROOT/qwen-deploy-01.log" 2>&1
```

For Glimmer, use its MODEL/MODEL_SHA256 and the alternative command, after releasing the owned Qwen node:

```bash
bash tools/hybrid-bench/packets/r46/stage-glimmer-node.sh --state "$NODE_ROOT/glimmer-node" --build-dir "$NODE_ROOT/build" --build-lock "$NODE_ROOT/build.lock" > "$NODE_ROOT/glimmer-deploy-01.log" 2>&1
```

Builds use deploy-prefill-node's existing stamped-ref/build-lock flow. For stamped exact-head node reuse only, add `--skip-build`. The node stamp covers llama-hybrid-prefill, not the client targets. GPU rehearsals next use `bash tools/hybrid-bench/packets/r46/setup-rehearsal-client.sh qwen` (or `glimmer`) with BUILD pointing to that same build tree and NODE_ROOT to that same lock root. The wrapper builds and records all client targets at R46_HEAD under the shared lock; do not skip this build or manually edit the stamp. See each rehearsal's sequential operator commands. The launcher's25 GiB build floor stays active. Bind remains127.0.0.1:8091. q8_0 K/V, flash attention, node batch/ubatch256/256 and2 threads remain unchanged. Record source/tree, model, binaries/libraries, stamp and CUDA/device identity. Start the authenticated r44 helper for the separate SSH link probe and retain its PID:

```bash
python3 "$R44_PACKET/node-link.py" serve --port 8093 > "$NODE_ROOT/link-01.log" 2>&1 &
R46_LINK_PID=$!
printf '%s\n' "$R46_LINK_PID" > "$NODE_ROOT/link.pid"
```

Transfer the chosen state's node-info.json and client-tokens.json to the Mac by the established private SSH channel; chmod600, no token values in output. Forward8091/8093 over SSH Compression=no. Preserve logs/metadata for the same model/run. After client cleanup, authenticate from the credential file to `/v1/sessions` and `/v1/health` and require sessions0/ram_bytes0. Then `tools/hybrid-bench/deploy-prefill-node.sh --state "$NODE_ROOT/glimmer-node" --stop` (or qwen-node) and `kill "$R46_LINK_PID"`. A nonzero remaining-session count is an ownership finding; do not delete foreign sessions.

For r46b calibration/timing, native tracing must be enabled before starting the node. Health reports `native_trace_enabled`; missing tracing refuses the instrumented driver. Keep native JSONL private and transfer it after the run. Authenticated bench-only `/v1/bench/trace-flush` drains outside measured calls; no snapshots are taken. See TIMING-SPANS.md. Use a new trace directory for every node launch, never overwrite a previous trace.

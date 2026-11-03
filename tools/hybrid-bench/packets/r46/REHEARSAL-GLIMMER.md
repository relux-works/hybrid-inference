# Glimmer CUDA-to-CUDA checkpoint rehearsal

Functional qualification on one RTX PRO 6000 96GB over loopback. It is not a timing comparison: node, decoder and checker share the card. Use the signed r46 source and the exact same Glimmer30B Q4_K_M file on both roles, SWA2048, no recurrent state, native context131072. Stage the node with README-NODE.md. Do not run this on agents1. Plan3-6h compute/checker/storage allowance, 120GB initial free disk after staging weights/build, 96GB host RAM, 96GB GPU RAM. Allow roughly 40-60GB loopback capture traffic and 5-10GB selective evidence archive; all-turn capture reservation below is a conservative bound, not a physical counter. Target runtime and SWA exactness are unmeasured; preserve failures.

## Start depth

Use76000, seed1,40 sampled turns and the real rendered-template framing. With framing32, the sampler grows52011 tokens to128011, leaving3061 below131072 (1013 beyond the mandatory2048 margin). At framing16 it ends127371. An80000 start fails with the existing depth-conditioned sampler. Do not truncate turns or outputs. The driver repeats the bound with actual framing before any arm.

## Ten exact controls

A is deterministic local-batch replay, B the node-batch replay, L the live local snapshot, H the live hybrid snapshot, and CHAIN the recorded sequence of real imported envelopes and local gaps. Window means the serialized SWA cache including physical cell placement/head and metadata, not a recurrent tensor. Non-SWA attention rows and SWA window are the two native memory components (`llama_kv_cache_iswa.get_base/get_swa`).

| Control | Glimmer contract and reason |
|---|---|
| NODE-B | Same captured node snapshot/tail vs node-partition replay; unchanged |
| RESTORE-B | Same replay with native roundtrips at node boundaries; exact snapshot bytes and logits; unchanged |
| BSTATE-B | Node-replay state equals captured snapshot before tail; unchanged |
| NODE_RANGE-B | Node snapshot through range export/import retains bytes and logits; unchanged |
| OWN_NATIVE-A | Own native state roundtrip is identity; unchanged |
| OWN_RANGE-A | Own range state roundtrip is identity; unchanged |
| CHAIN-H | Captured import chain reproduces live hybrid continuation; unchanged |
| CHAIN_ATTENTION-B | Replace the chain's full-attention component with node reference; window bytes must already equal node's, and the result equals B. Full-attention provenance is isolated as in Qwen |
| CHAIN_WINDOW-H | Replace the already imported window with the identical node window; require window_equal and equality to H. Replaces CHAIN_RECURRENT-H, which would be vacuous on a model without recurrence |
| A-L | Local deterministic replay equals the live local checkpoint; unchanged |

All10 must be exact (zero KL and logit difference,64/64, no free-running divergence inside each identity), plus journal/hash/raw-logit integrity. L-H teacher-forced top1 must be >=63/64; KL, noise flags and L-H free-running divergence remain report-only. No tolerance is added for GPU/SWA. A missing/failed window or native-save/restore identity is a failed rehearsal, not permission to remove a control. Native slot compaction and physical SWA layout must be exercised beyond2048; no small SWA weight file exists on agents1, so those real inference paths are an explicit qualification gap. CPU Qwen and synthetic window gate tests cannot close it.

## Run

Prepare separate NODE_ROOT and CLIENT_ROOT under this GPU checkout's .temp and staged BUILD/model paths. Use the launcher-generated node metadata/token files. Client setup below builds all required targets on the same signed head. The rehearsal policy deliberately uses an illustrative local100tok/s target, 16MB/s simulated WAN rate and the prior PR9 CUDA import fit1.6186573567s +0.0004953084s/token. These are functional routing parameters, not PRI31 Glimmer Mac data or measured loopback rates. The Mac packet refuses this placeholder and requires its own fixture/calibration.

```bash
set -eu
umask 077
mkdir -p "$CLIENT_ROOT/tmp"
export TMPDIR="$CLIENT_ROOT/tmp" PYTHONDONTWRITEBYTECODE=1
tools/hybrid-bench/setup-client.sh --state "$CLIENT_ROOT/client" --model "$MODEL" --sha256 "$MODEL_SHA256" --ctx 131072 --cuda --node-info "$NODE_ROOT/glimmer-node/node-info.json" --token-file "$NODE_ROOT/glimmer-node/client-tokens.json" --prefill-url http://127.0.0.1:8091 --request-timeout 3600 --startup-timeout 3600 --ref "$R46_HEAD" --build-dir "$BUILD" --offline > "$CLIENT_ROOT/setup-01.log" 2>&1
python3 - "$CLIENT_ROOT" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,'tools/hybrid-bench')
from bench_common import private_json,read_private
r=Path(sys.argv[1]); c=read_private(r/'client/proxy-config.json'); x=c['delta_geometry']
private_json(r/'rehearsal-policy.json',dict(local_rates=[[0,100]],server_rate=2823,link_rate=16000000,fixed=.110,
 import_fixed=1.618657356721653,import_per_token=.0004953084013066794,
 kv_bytes=x['kv_bytes'],state_bytes=x['state_bytes']+x.get('window_bytes',0)))
PY
tools/hybrid-bench/setup-client.sh --state "$CLIENT_ROOT/client" --stop > "$CLIENT_ROOT/stop-launcher-01.log" 2>&1
```

Execute this command through the existing foreground command/PID/exit recorder (r44 `run-session.execute` accepts an arbitrary argument list), or the GPU operator's equivalent recorder. Do not use r44 runner's fixed Mac CLI, which would overwrite these depth/checkpoint parameters.

```bash
python3 tools/hybrid-bench/growing_session.py --client-state "$CLIENT_ROOT/client" --output "$CLIENT_ROOT/rehearsal" --depth 76000 --turns 40 --seed 1 --start pre-staged --disconnect-turn 20 --disconnect-turns 3 --checkpoint-turns 10,20,30,40 --keep-checkpoint-states --state-probes --policy "$CLIENT_ROOT/rehearsal-policy.json" --link-label 'PRO 6000 same card CUDA loopback; illustrative routing policy, not timing evidence' --request-timeout 3600
```

Keep states for offline rechecks. Expected successful exit0/statuspassed, 120 turn calls (hybrid/local/local_partition), checkpoints10/20/30/40 each with10 green controls and top1>=63, connected local and offload routes, local turns20-22, verified resync after23 and final cleanup with no remaining remote sessions. A trace aborted before completion or any failed checkpoint is not accepted. Record actual checkpoint durations, all events and node counters; archive selective trace/results/raw-logits/deltas and model/build/source identities privately, keeping native snapshots on the run volume for rechecks. Actual GPU execution and independent acceptance remain external gates.

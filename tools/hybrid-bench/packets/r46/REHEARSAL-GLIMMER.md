# Glimmer CUDA-to-CUDA checkpoint rehearsal

Functional qualification on one RTX PRO 6000 96GB over loopback. It is not a timing comparison: node, decoder and checker share the card. Use the signed r46 source and the exact same Glimmer30B Q4_K_M file on both roles, SWA2048, no recurrent state, native context131072. Stage the node with README-NODE.md. Do not run this on agents1. Use the explicit [common-journal design](COMMON-JOURNAL.md) and model declaration below. Plan4-7h compute/checker/storage allowance, 120GB initial free disk after staging weights/build, 96GB host RAM, 96GB GPU RAM. Allow roughly 40-60GB loopback capture traffic and 5-10GB selective evidence archive; all-turn capture reservation below is a conservative bound, not a physical counter. Target runtime and SWA exactness are unmeasured; preserve failures.

## Start depth

Use76000, seed1,40 sampled turns and the real rendered-template framing. Node F at7d512238 measured framing7:51011 tokens of growth, final depth127011, leaving4061 below native131072. The policy generator's conservative framing32 plan ends128011, still with3061 margin. The new mode never resamples or truncates turns/outputs. The driver recomputes the actual bound with the current template before any arm. An80000 start is not substituted.

## Declared pairing contract

Pass `--pairing-contract tools/hybrid-bench/fixtures/glimmer-common-journal.json`. It binds the pinned GGUF hash, architecture, SWA2048 and absence of recurrence, and declares the measured property free_running_batch_invariant=false from node F. There is no fallback on a journal mismatch. Omitting the contract keeps the original independent-arm equality gate, which remains required for Qwen. Common mode requires checkpoints and state probes and is refused for recurrent Qwen.

Hybrid alone generates the live40-turn conversation. At each checkpoint, its native H snapshot and exact token IDs define the common history. The checker independently replays L and P from those IDs: L uses256/256, P uses64/32; generated spans are evaluated one token at a time. Both cross a native save/restore boundary before the final held token. Neither reference samples its own past or re-tokenizes hybrid text. These are teacher-forced reference histories, not live free-running timing arms. The checker records the common journal hash,64 teacher token IDs and each64-token independent free continuation. Divergence is measured over those64 positions, not an alternate40-turn conversation.

## Ten exact controls

A is deterministic local-batch replay, B the node-batch replay, L the native-roundtripped teacher-forced local reference, H the live hybrid snapshot, and CHAIN the recorded sequence of real imported envelopes and local gaps. Window means the serialized SWA cache including physical cell placement/head and metadata, not a recurrent tensor. Non-SWA attention rows and SWA window are the two native memory components (`llama_kv_cache_iswa.get_base/get_swa`).

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
| A-L | Independent local replay equals the teacher-forced local reference after its native checkpoint roundtrip; exact bytes/logits remain required |

All10 must be exact (zero KL and logit difference,64/64, no free-running divergence inside each identity), plus journal/hash/raw-logit integrity. L-H teacher-forced top1 must be >=63/64. Actual teacher token IDs must match for H/L/P. KL, noise flags and cross-path H-L/L-P free-running divergence remain report-only; continuation token IDs bind each reported first divergence. Divergence within an exact identity still fails that identity. No tolerance is added for GPU/SWA. A missing/failed window or native-save/restore identity is a failed rehearsal, not permission to remove a control. Native slot compaction and physical SWA layout must be exercised beyond2048; no small SWA weight file exists on agents1, so those real inference paths are an explicit qualification gap. CPU Qwen and synthetic window gate tests cannot close it.

## Run

Prepare separate NODE_ROOT and CLIENT_ROOT under this GPU checkout's .temp and staged BUILD/model paths. Set MODEL, MODEL_SHA256 and R46_HEAD as in README-NODE.md. The command list below stages the node, then builds and records all three client targets at the same exact ref under the same build.lock. No `--skip-build`, manual target build or stamp edit is required; the launcher's 25 GiB build admission/runtime floor remains active. The rehearsal additionally keeps28GB free.

The [policy source and route audit](REHEARSAL-POLICY.md) uses the r46b interim Glimmer100tok/s planning scenario at20 decimal MB/s, server2823tok/s, RTT0.1406s and prior Qwen Metal import1.498s+0.0000487s/token. These are illustrative functional routing parameters, not PRI31 Glimmer Mac data or a loopback calibration. The Mac packet continues to require its own measured fixture/calibration. Real geometry is7072 attention bytes/token, no recurrent bytes, and43514896 window bytes including metadata. Framing32, seed1,76000 start and outage20-22 predict26 connected local and11 offload turns:3,5,6,7,13,16,17,31,32,33,40. The old16MB/s policy already gave the same mixed counts for Glimmer; it did not have Qwen's degeneracy defect. The replacement shares the documented planning source. Actual template framing and live transfer-rate EMA can change routes; preserve the runtime preflight and record actual counts.

```bash
set -eu
umask 077
mkdir -p "$NODE_ROOT/tmp" "$CLIENT_ROOT/tmp"
export TMPDIR="$NODE_ROOT/tmp" PYTHONDONTWRITEBYTECODE=1
bash tools/hybrid-bench/packets/r46/stage-glimmer-node.sh --state "$NODE_ROOT/glimmer-node" --build-dir "$BUILD" --build-lock "$NODE_ROOT/build.lock" > "$NODE_ROOT/deploy-01.log" 2>&1
export TMPDIR="$CLIENT_ROOT/tmp"
bash tools/hybrid-bench/packets/r46/setup-rehearsal-client.sh glimmer > "$CLIENT_ROOT/setup-01.log" 2>&1
python3 tools/hybrid-bench/packets/r46/rehearsal_policy.py --model glimmer --client-state "$CLIENT_ROOT/client" --output "$CLIENT_ROOT/rehearsal-policy.json" > "$CLIENT_ROOT/policy-01.log" 2>&1
tools/hybrid-bench/setup-client.sh --state "$CLIENT_ROOT/client" --stop > "$CLIENT_ROOT/stop-launcher-01.log" 2>&1
```

Execute this command through the existing foreground command/PID/exit recorder (r44 `run-session.execute` accepts an arbitrary argument list), or the GPU operator's equivalent recorder. Do not use r44 runner's fixed Mac CLI, which would overwrite these depth/checkpoint parameters.

```bash
python3 tools/hybrid-bench/growing_session.py --client-state "$CLIENT_ROOT/client" --output "$CLIENT_ROOT/rehearsal" --depth 76000 --turns 40 --seed 1 --start pre-staged --disconnect-turn 20 --disconnect-turns 3 --checkpoint-turns 10,20,30,40 --keep-checkpoint-states --state-probes --pairing-contract tools/hybrid-bench/fixtures/glimmer-common-journal.json --policy "$CLIENT_ROOT/rehearsal-policy.json" --link-label 'PRO 6000 same card CUDA loopback; illustrative routing policy, not timing evidence' --request-timeout 3600
```

Keep states for offline rechecks. Expected successful exit0/statuspassed, 40 live hybrid turn calls plus its forced base, and L/P teacher-forced references at four checkpoints, checkpoints10/20/30/40 each with10 green controls and top1>=63, connected local and offload routes, local turns20-22, verified resync after23 and final cleanup with no remaining remote sessions. A trace aborted before completion or any failed checkpoint is not accepted. Record actual checkpoint durations, all continuation tokens, actual route counts, all events and node counters; archive selective trace/results/raw-logits/deltas and model/build/source identities privately, keeping native snapshots on the run volume for rechecks. Actual GPU execution and independent acceptance remain external gates.


After successful driver cleanup and verified zero remaining owned sessions, stop the owned node:

```bash
tools/hybrid-bench/deploy-prefill-node.sh --state "$NODE_ROOT/glimmer-node" --stop > "$NODE_ROOT/stop-01.log" 2>&1
```

## Full-run time estimate

The node-F hybrid calls took593.19s plus789.44s sampled pauses, about23.0min; use25min for the live hybrid stage. There were27142 generated tokens. This does not include the new four checkpoint programs. Each program creates L/P teacher-forced histories, four factorial histories and nine exact state/transport probes, including sequential generated-token evaluation,64-position teacher/free observations, state read/write and model/context startup. Reserve4-7 GPU hours for a fresh build/setup/full run. This is an engineering allowance, not a measured full result. Measure checkpoint10 and update the remaining estimate from its actual durations. Keep120GB free after weights/build,96GB host/GPU memory and the28GB runtime disk floor; the shared-lock client build retains25GiB admission. The packet author ran CPU tests only and did not execute CUDA or download weights.

# Qwen recurrent bf16 growing-session GPU rehearsal

Question: does recurrent bf16 rounding change distributions increasingly over turns, including after the outage? Use one exclusive RTX PRO6000 96GB, CUDA to CUDA, identical pinned Qwen3.8-27B Q4_K_M GGUF, context196608, q8_0 attention, seed1,100000 start and40 turns. No GPU/download runs on agents1. Accepted codec input is exactly64dc3c699d2885f49146a0536f98c4fb06555692. Native recurrent state, node snapshots and recovery files stay f32; only recurrent wire tensor payloads become bf16.

The primary rehearsal first captures the ordinary f32 hybrid/local/partition program at checkpoints10/20/30/40, outage2-4 and reconnect5. It then replays its full actual import chain from the base through each checkpoint, converting each critical recorded f32 envelope with the accepted bf16 codec, staging and applying through the real native importer, and replaying local gaps at their original boundaries. Node catch-up states remain f32. Common tokens and routing isolate client-side effects from rounding; fresh full-prefix f32 node state at a later import can reset recurrent error, so monotonic accumulation is not assumed. Pauses and outage position remain the original program. This is common-journal quality rehearsal, separate from the three-arm timing packet's independent policy and free-running history. It does not substitute those replay results for measured production timing.

## Checkpoint measurements

At each checkpoint, retain original H (live f32 hybrid), L (local), P (partition), A/B schedules and all ten f32 identity controls. All ten must remain green, including CHAIN-H and recurrent replacement, and f32 L-H>=63/64. The original f32 trace must be passed/homogeneous. A baseline failure is preserved, not repaired by a lossy verdict.

Add BF16_CHAIN: replay every actual critical chunk with state_dtype=bf16, including the base chunks, and local prompt/generated gaps, on the exact same recorded token journal. Score64 teacher-forced positions using the live local continuation. Report per-position/mean/p95 KL(L || BF16_CHAIN), KL(H || BF16_CHAIN), max logit difference, top1 counts against L and H, and each first greedy free-running divergence in0..63 (-1 means none within64). Compare raw H/L logit file hashes to the original f32-arm checkpoint: a changed reference refuses. Native raw logits and hashes bind each result. Plot the four checkpoint values with cumulative import count; report resets or decreases as observed. The top1 gate for L-BF16_CHAIN stays>=63/64 at every checkpoint. KL and free-running divergence are reported, without an invented threshold.

The treatment has `state_transport_exact=false`, `lossy_import=true`, statuslossy, exit5 even if logits happen to match. The exact transport gate never passes. Mathematical logit equality remains separate from provenance. The top-level checker likewise reports lossy/5; `summary.json.quality_gate` separately records whether all four63/64 gates hold. Exit5 alone is neither a quality pass nor a failure to measure. Exit2 means refusal; missing/failed f32 controls or mismatched raw reference hashes refuse. Existing f32 controls are still required, never relaxed into approximate controls.

## Stage and run

Use the signed delivered head as R46_HEAD and the same MODEL/MODEL_SHA256 as README-NODE.md (Qwen SHA31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34). Set private NODE_ROOT, CLIENT_ROOT, BUILD and BF16_NODE_STATE under the GPU checkout's .temp. Stage once; the wrapper verifies both transport capabilities. Use the existing GPU job/PID recorder and GPU lock. No live binaries are replaced under another job.

```bash
set -eu
umask 077
mkdir -p "$NODE_ROOT/tmp" "$CLIENT_ROOT/tmp"
export TMPDIR="$NODE_ROOT/tmp" PYTHONDONTWRITEBYTECODE=1
export BF16_NODE_STATE="$NODE_ROOT/qwen-bf16-node"
bash tools/hybrid-bench/packets/r46/stage-qwen-bf16-node.sh --build-dir "$BUILD" --build-lock "$NODE_ROOT/build.lock" > "$NODE_ROOT/deploy-01.log" 2>&1
export TMPDIR="$CLIENT_ROOT/tmp"
bash tools/hybrid-bench/packets/r46/setup-rehearsal-client.sh qwen > "$CLIENT_ROOT/setup-01.log" 2>&1
python3 tools/hybrid-bench/packets/r46/rehearsal_policy.py --model qwen --client-state "$CLIENT_ROOT/client" --output "$CLIENT_ROOT/rehearsal-policy.json" > "$CLIENT_ROOT/policy-01.log" 2>&1
tools/hybrid-bench/setup-client.sh --state "$CLIENT_ROOT/client" --stop > "$CLIENT_ROOT/stop-launcher-01.log" 2>&1
python3 tools/hybrid-bench/growing_session.py --client-state "$CLIENT_ROOT/client" --output "$CLIENT_ROOT/f32-rehearsal" --depth 100000 --turns 40 --seed 1 --start pre-staged --disconnect-turn 2 --disconnect-turns 3 --checkpoint-turns 10,20,30,40 --keep-checkpoint-states --state-probes --policy "$CLIENT_ROOT/rehearsal-policy.json" --link-label 'PRO6000 CUDA loopback; illustrative routing, common-journal bf16 rehearsal' --request-timeout 3600
```

Client setup deliberately omits `--skip-build`: the node stamp records only llama-hybrid-prefill. The shared wrapper builds and records llama-server, llama-hybrid-bench-check and llama-hybrid-score at R46_HEAD under the node's same build.lock, with the launcher's existing 25 GiB admission/runtime floor. No manual target build or stamp edit is required. Both stages are sequential. Keep at least28GB free throughout the rehearsal as well.

The [policy source and route audit](REHEARSAL-POLICY.md) uses the r46b planning scenario at20 decimal MB/s: measured Qwen PRI31 delta512 floor rates, server1767tok/s, RTT0.1406s, prior Metal import1.498s+0.0000487s/token. This is illustrative routing on loopback, not a calibration or a WAN timing measurement. Geometry comes from the launched client's delta_geometry, including34816 attention bytes/token and156893184 recurrent bytes. With the unchanged seed1 schedule/framing14, f32 preflight predicts30 connected local and7 offload turns (7,13,17,31,32,33,40), plus three outage turns2-4 and the forced base. The independent bf16-byte policy would predict28 local/9 offloads; BF16_CHAIN deliberately follows the actual captured f32 routes instead.

Repeated-rounding coverage needs imports beyond the base. Under the configured-link prediction, BF16_CHAIN gets seven such turns, with cumulative final import counts2/4/4/8 at checkpoints10/20/30/40 including the base. Base chunk count is not a repeated-state import count: only the final chunk installs recurrent state. Live transfer-rate EMA can change route decisions on loopback; record actual turn imports and cumulative counts from the captured trace and replay summaries. A base-only chain cannot qualify repeated-rounding behavior; retain it as insufficient coverage. Preflight still requires both routes. Do not alter samples, seed, turns or context to force a result. After baseline exit0/statuspassed and zero remaining remote sessions, stop the owned node to release its GPU memory. The offline checker then uses the same card sequentially, with the node and decoder stopped:

```bash
tools/hybrid-bench/deploy-prefill-node.sh --state "$BF16_NODE_STATE" --stop > "$NODE_ROOT/stop-01.log" 2>&1
set +e
python3 tools/hybrid-bench/packets/r46/replay_bf16.py --trace "$CLIENT_ROOT/f32-rehearsal/trace.json" --checker "$BUILD/bin/llama-hybrid-bench-check" --model "$MODEL" --output "$CLIENT_ROOT/bf16-replay" > "$CLIENT_ROOT/bf16-replay-01.log" 2>&1
REPLAY_EXIT=$?
set -e
printf '%s\n' "$REPLAY_EXIT" > "$CLIENT_ROOT/bf16-replay.exit"
test "$REPLAY_EXIT" -eq 5
```

Inspect every checkpoint and summary quality_gate; retain failures. Save original trace, deltas, node snapshots, H/L/P checkpoints, all raw logits, source/model/build hashes and process exit records. The native files are private. Selective archives without native snapshots cannot support later snapshot replay; keep working snapshots until replay is verified. An interrupted command has no resume claim; preserve partial results and use a fresh output for another authorized attempt.

## Resource and cost estimate

Plan6-10 GPU hours total including stage, three-arm f32 baseline/four full checkpoints, four complete bf16-chain replays with unchanged reference controls, and storage checks. This is unmeasured at27B and is deliberately above a prefill-only floor: roughly100000-140000 tokens x many deterministic histories per checkpoint, plus two64-token continuations per observation, state I/O and model loads. Time the first checkpoint and revise the remaining estimate from its actual phase spans. Do not interpret loopback timings as WAN TTFT.

Require450GB free on the working volume after weights/build, plus30GB selective archive allowance (actual archive bytes are measured before writing), and keep28GB free throughout. The prior r44 full100k/four-checkpoint bound was342.8GB without retaining all checkpoint native files; retaining12 native states and four additional raw-logit sets warrants this round-up. All-turn-offload capture bound is about198.4GB snapshots plus12.3GB deltas; expected route-dependent traffic is lower. Replay converts in memory and reads existing captures; it does not transfer another100k base over WAN. Node logs/model/build need another10GB if on a separate disk. Allow128GB host RAM and96GB GPU RAM with actual geometry admission and no sustained swap; node/client/checker overlap during baseline is the peak, replay runs alone. Models (~17-19GB), build and existing data are additional.

Using the prior packet's planning rate$1.57/GPU-hour gives$9.42-$15.70 GPU rental for6-10h. This is an arithmetic scenario, not a verified current booking quote; replace with the actual booked rate. Storage, egress, taxes and any host rental are additional. The rehearsal and target quality remain operator execution and independent review gates.

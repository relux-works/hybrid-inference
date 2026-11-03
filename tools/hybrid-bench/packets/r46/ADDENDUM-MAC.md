# r46 Mac timing addendum

This packet measures production wall-clock TTFT directly. Fidelity remains with the r44 main run and r45 recheck. Use the signed `m3-growing-r46` head delivered with this packet, based on 72566bca. Do not modify the original traces. Run Qwen and Glimmer sequentially through the existing PRI-31 supervisor. Its PID, descendant priority, disk, power, RAM and exit monitoring stay active. The payload refuses unless its own priority is 31 and nice is 0; the supervisor must also record actual decoder priority. Launch from the pinned source root, not a background-band tmux descendant.

| Run | Start/context, seed | Arms/checkpoints | Planning wall time | Application traffic per run | Admission after weights/build are staged |
|---|---|---|---|---|---|
| Qwen3.8-27B Q4_K_M | 100000 / 196608, seed 1, 40 turns | hybrid, local; none | 3-4.5 h including 10-30 min calibration and 25-50% operational allowance | 4.1-5.6 GB including base and 403 MB link probe; calibration adds roughly 10-30 GB | At least 80 GB free on run/state/build volumes, and computed RAM <80% of physical RAM; about 73 GiB with a 17 GB GGUF |
| Glimmer 30B Q4_K_M, SWA 2048, no recurrence | 76000 / 131072, seed 1, 40 turns | hybrid, local; none | Hypothetical 2-4.5 h, pending PRI-31 rates and Glimmer import calibration | Hypothetical 1.1-1.8 GB including base/probe; calibration adds roughly 2-8 GB | At least 45 GB free on run/state/build volumes, and computed RAM <80%; about 47 GiB with an 18 GB GGUF |

GB means decimal bytes; GiB means 2^30 bytes. Staged weights are assumptions for the examples; `mac_timing.py` uses actual file size, geometry and physical memory. Formula: `2*weights + 5*S(context) + 8GiB < 0.8*physical`. Disk is `max(80GB Qwen / 45GB Glimmer, 28GB + 6*S(context) + 4GB)`. Run on the qualified external LaCie volume, keep TMPDIR there, and reserve another 10 GB for a fresh build before admission if needed. Existing r44 volume-probe requirements still apply; record observed storage rate. Archive allowance is selective, not a duplicate of state files.

`planning.json` is reproducible with `python3 tools/hybrid-bench/packets/r46/estimate.py`. These are estimates, not upper bounds or measurements. Qwen uses the archive-derived PRI-31 floor curves. Glimmer explores 50/100/150 prefill tok/s and 12/18/24 decode tok/s solely as sensitivity cases; none is a policy fixture. Both use an illustrative prior Qwen import fit, which must be replaced before execution. Times include both arm pauses and base staging, and omit filesystem tails/background waits before the operational allowance. Traffic includes estimated envelope bodies, token uploads and the link probe; HTTP/SSH/TCP overhead, retransmission, setup transfers and calibration are additional. Record physical link as `5G` or `Wi-Fi` independently for each run. Physical network byte counters belong in the operator's existing supervisor evidence; driver `wire_bytes` means delta body bytes, not link-layer traffic.

## Source and prerequisite packet

Receive the signed source bundle and its SHA256 manifest over the established owner channel. Verify the full manifest hash against the separate handover, then `sha256sum -c SHA256SUMS` (or `shasum -a 256 -c SHA256SUMS` on Mac). Verify commit signatures using existing owner trust; never install a packet-supplied key. Fetch the local bundle, create a clean detached worktree at the delivered full head, and verify its commit/tree against the handover. No GitHub push or PR is part of this packet.

Keep the previously authenticated r44 packet available as `R44_PACKET`. r46 reuses its `node-link.py`, `fit-import.py`, `run-session.py` and `packet-ops.py` instead of creating another supervisor, fitting system or archive format. Use the current r44 packet (manifest 52c2c2fa5bed6bfe3291e5562109c3a18525ceb57d13b87ed44f0233a01955d5) with r45 addendum. Its old 325GB capture budget, 25MB/s link gate and checkpoint-10 payload are replaced for these timing runs by the admission and command below. Slow-link all-local routing is a valid timing observation. The fidelity rehearsal retains its route-mix requirement.

## Stage the matching node and client

Have the node operator execute `stage-qwen-node.sh` or `stage-glimmer-node.sh` as described in README-NODE.md. Forward node port 8091 and link-probe port 8093 over the existing SSH transport with compression disabled. Copy only `node-info.json` and `client-tokens.json` over the private channel into the new run's private directory, chmod 600; do not print credentials. Use the same full GGUF SHA on both roles.

For each model, set these absolute paths in the supervisor environment: `R46_HEAD`, `MODEL`, `MODEL_SHA256`, `RUN_ROOT`, `BUILD`, `R44_PACKET`. `RUN_ROOT` is a new task-owned external-volume directory. Set `CTX=196608` for Qwen, `CTX=131072` for Glimmer. Retain the r44 Metal-build recipe with CUDA OFF and Metal ON; build all three client targets on this exact head. A stamped prebuild is allowed only with its exact commit, flags and executable hashes. Remove `--skip-build` below for a new build; never reuse another head's stamp.

```bash
set -eu
umask 077
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$RUN_ROOT/tmp" "$RUN_ROOT/private" "$RUN_ROOT/calibration"
export TMPDIR="$RUN_ROOT/tmp"
tools/hybrid-bench/setup-client.sh --state "$RUN_ROOT/client" --model "$MODEL" --sha256 "$MODEL_SHA256" --ctx "$CTX" --node-info "$RUN_ROOT/private/node-info.json" --token-file "$RUN_ROOT/private/client-tokens.json" --prefill-url http://127.0.0.1:8091 --request-timeout 3600 --startup-timeout 3600 --ref "$R46_HEAD" --build-dir "$BUILD" --skip-build --offline > "$RUN_ROOT/setup-01.log" 2>&1
```

## Import calibration and required rate fixtures

Run calibration through the same PRI-31 supervisor before timing, with the launcher still running. It has no checker or growing-driver node captures. For Qwen use depths `2048 20000 50000 100000`; for Glimmer use `2048 20000 50000 76000`. Each command below is a single repetition; use fresh rep-2/rep-3 directories when more evidence is needed and fit all repetitions. Calibration traffic is outside the timing trace. This fit still lacks a conditioning-depth term; preserve its raw points and that limitation.

```bash
mkdir "$RUN_ROOT/calibration/rep-1"
python3 tools/hybrid-bench/hybrid_bench.py --client-state "$RUN_ROOT/client" --modes offload --depths 2048 20000 50000 "$START_DEPTH" --seed 1 --scratch "$RUN_ROOT/calibration/rep-1" --output "$RUN_ROOT/calibration/rep-1" --label r46-import --request-timeout 3600 > "$RUN_ROOT/calibration-01.log" 2>&1
python3 "$R44_PACKET/fit-import.py" --source-root "$PWD" --output "$RUN_ROOT/import-fit.json" "$RUN_ROOT"/calibration/rep-*/*.jsonl > "$RUN_ROOT/import-fit-01.log" 2>&1
```

Set `START_DEPTH=100000` for Qwen or `76000` for Glimmer. Qwen's ready fixture is `tools/hybrid-bench/fixtures/m3-max-qwen-pri31.json`, derived from c2e83fbd4f59a38181e48b17206783e91d6d5b6734624f46d4cede68d7bcace5. It retains all delta sizes and weighted decode by depth; routing uses delta512 at the conditioning-depth floor. The kit engine is dde879bc, so this is target-client calibration, not a new r46 engine measurement. The driver stores the complete fixture and hash in its trace.

Glimmer's shipped `m3-max-glimmer-pri31.pending.json` is a placeholder and refuses before link traffic or measurement artifacts. When the owner delivers the normal-priority Glimmer archive, set its trusted full SHA and generate a new fixture:

```bash
python3 tools/hybrid-bench/growing_fixture.py --archive "$GLIMMER_ARCHIVE" --sha256 "$GLIMMER_ARCHIVE_SHA256" --output "$RUN_ROOT/glimmer-pri31.json"
```

The builder verifies archive hash, actual server PRI31/nice0 evidence, no priority override, completed step40, all three deltas per depth, no re-reads/truncation, and model binding. Do not hand-fill rates or reuse background fixtures. If the delivered kit format differs, retain it and adjust the reader with tests before running.

## Supervisor timing payloads

Set `LINK_KIND` to the observed `5G` or `Wi-Fi`. Each payload measures 15 pings and 3 x 64MiB in each direction immediately before the run, constructs the policy with the PRI31 fixture and fresh import fit, checks actual template depth/RAM/disk, records admission, and stops its own launcher before the driver starts its decoder. Qwen uses the sourced scalar server rate 1767 tok/s at 100k; Glimmer uses 2823 at 118k (FACTS 1a, conservative for the 76k-129k ladder; the tail remains unmeasured). Link rate then follows the existing live EMA. Do not run the two payloads simultaneously.

```bash
python3 tools/hybrid-bench/packets/r46/mac_timing.py --client-state "$RUN_ROOT/client" --output "$RUN_ROOT/timing-qwen" --fixture tools/hybrid-bench/fixtures/m3-max-qwen-pri31.json --import-fit "$RUN_ROOT/import-fit.json" --r44-packet "$R44_PACKET" --link "$LINK_KIND"
```

After changing to Glimmer's separately staged client/root and completing its calibration:

```bash
python3 tools/hybrid-bench/packets/r46/mac_timing.py --client-state "$RUN_ROOT/client" --output "$RUN_ROOT/timing-glimmer" --fixture "$RUN_ROOT/glimmer-pri31.json" --import-fit "$RUN_ROOT/import-fit.json" --r44-packet "$R44_PACKET" --link "$LINK_KIND"
```

Both fix seed1, 40 sampled turns, pre-staged starts, outage2+3/reconnect5, and `--checkpoint-turns none`. No state probes, node snapshots/delta captures, partition arm, or checker runs. Production rollback saves/import validation and ordinary metering remain part of measured timing. Cheap monotonic protocol durations already exist; the trace reserves versioned `instrumentation.timestamps` for dev-d r13's owner-delivered list. That list is still pending; r46 adds no native profiling or synchronous probes to TTFT.

Exit0 means `status=measured`, not numerical qualification. Heterogeneous CUDA/Metal timing can exit0 without a fidelity verdict. Exit2 means refusal/abort. Preserve real command/PID/exit records, startup costs separately from turn TTFT, admission policy/link samples, zero capture counters, and the per-turn route/decode/transport fields. Slow-link Qwen can choose all-local turns at the corrected PRI31 rates (5.19/9.75 MB/s examples); report that directly. At 16/25MB/s the illustrative plans offload 5/7 turns. Do not force offload to obtain a speedup.

Verify after each run: arms exactly hybrid/local, 40 turn calls each, `capture={enabled:false,node_snapshots:0,node_deltas:0,checker_runs:0}`, checkpoints empty, no node-states/check-* artifacts, status measured, and cleanup event has no remaining remote sessions. Retain the node through final cleanup, then verify authenticated node session counters are zero. Archive with the existing r44 selective helper (`python3 "$R44_PACKET/packet-ops.py" archive --help` for current arguments); include the adjacent admission/policy/command/exit/calibration/supervisor evidence separately and exclude credentials and native bases. Preserve originals before any owned-state cleanup.

## Reserved bf16 addendum

Stable arm IDs are `hybrid`, `local`, and future `hybrid-bf16`. Never relabel existing `hybrid` or change its f32 recurrent transport. Each call records `transport_profile`, with route and attention/recurrent/window wire types; the trace has `arm_definitions` and an unavailable `reserved_arms.hybrid-bf16`. Glimmer has recurrent_transport=null and q8_0 window transport. A future Qwen third arm requires dev-a r38 capability negotiation, exact node/client protocol/build pins, and independently qualified bf16 import. It must record requested and negotiated recurrent dtype, protocol version, actual state bytes and model/layout identity per transfer; no silent fallback to f32. Policy state bytes must use the negotiated dtype. Reuse the same sampled sequence, seed, pauses, outage and paired journal contract, with a separate session and base for the third arm. Refactor arm execution by route/profile when enabling it; existing arm IDs, field units and their transport semantics stay fixed. This packet does not enable bf16.

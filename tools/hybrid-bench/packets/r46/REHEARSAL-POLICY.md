# r46d illustrative rehearsal routing

The rehearsal uses the existing r46b `estimate.py` planning policy at20 decimal MB/s, not the old PR9 CUDA import coefficients. `planning_policy` is shared with the unchanged link table; `rehearsal_policy.py` reads geometry from the launched client's private proxy-config.json, writes rehearsal-policy.json and rehearsal-policy.planning.json, and checks both routes before emitting either. No samples, sampler limits, seed, start depth, turns or outage positions change. The normal driver independently preflights actual rendered framing before any arm. Policy JSON is a functional simulation, never a calibration or measured loopback/WAN timing.

Qwen local rates are the measured PRI31 delta512 floor curve in fixtures/m3-max-qwen-pri31.json (100k-143k uses65.3tok/s). Its provenance pins the step40 archive c2e83fbd4f59a38181e48b17206783e91d6d5b6734624f46d4cede68d7bcace5 and raw/readiness hashes. Server rates1767 Qwen /2823 Glimmer, RTT0.1406s, import1.498s+0.0000487s/token, eta0.85, margin0.1 and hysteresis0.05 come from the r46b planning scenario. The prior Metal import fit is an illustrative routing assumption here; it does not satisfy Mac fit-reuse admission. Glimmer uses r46b's explicitly interim100tok/s case because no measured PRI31 Glimmer fixture was available to this packet. No bf16 import cost or quality claim follows from these parameters.

## Geometry and schedule evidence

Qwen uses34816 attention bytes/token and156893184 recurrent bytes from the actual node-E trace and policy. The copied aborted trace SHA256 is6f4fb3cd69d0e8157228d816e29dfdb51b5b905854e034745d2f37bf58a82040; its40 workload rows equal the current sampler exactly at seed1,100000 start, framing14, output cap4506 and input cap10875. Its input range is6-2329, final depth143439. The old local100/16MB/s/PR9-import policy refuses with37 connected local/0offload, exactly reproducing the recorded error.

Glimmer geometry is from kit1.6.5 glimmer30-gpu-bench-results-20261102T024947Z.tar.gz, archive SHA256 a9c8887e1e5b24347802ca8049439aa04313d0a397f6e0ea1200aaf542f9606c. Its model-geometry.json SHA256 is9b54f4bc66b44fd8aaf42f0aa836c635a37ff8ae7e08617768fa02bc7299460d:13 full attention layers,39 SWA layers,7072 full-attention bytes/token,21216 window bytes/token,2048 window, zero recurrence. The launcher's existing formula adds2304*28+16 metadata bytes to43450368 window tensor bytes, giving43514896 fixed window bytes. Seed1,76000 start, framing32 and40 turns end at128011 within native131072 with the required2048 margin. The Glimmer rehearsal retains outage20-22; the general link table instead uses outage2-4.

## Configured-link route predictions

All counts exclude the forced base and three outage turns. Forty turns contain37 connected policy decisions. These are preflight predictions at the configured20MB/s. The production proxy's live transfer-rate EMA remains active; loopback transfers and actual Glimmer framing can change actual routes. Preserve and report runtime trace counts and BF16_CHAIN native import_count rather than substituting this table for execution.

| Replay / policy | Connected local | Connected offload | Turn offloads |
|---|---:|---:|---|
| Qwen f32; outage2-4 |30|7|7,13,17,31,32,33,40|
| Qwen bf16 independent byte policy, reference only |28|9|5,6,7,13,17,31,32,33,40|
| BF16_CHAIN actual rehearsal, same captured f32 routing |30 predicted|7 predicted|Same f32 captured turns; no independent bf16 routing|
| Glimmer f32; outage20-22 |26|11|3,5,6,7,13,16,17,31,32,33,40|

A base-only chain observes the effect of one rounding followed by local computation; it cannot test repeated import rounding. BF16_CHAIN therefore refuses a captured session with no successful hybrid turn import through the requested checkpoints, before creating replay output. Seven additional imports are predicted here. Final state-import counts including the base are2/4/4/8 at checkpoints10/20/30/40; the last new import before checkpoint30 is still turn17. Nonfinal base chunks contain no recurrent state and do not increase this count. Fresh full-prefix f32 node state at later imports can reset error; these observations do not imply monotonic accumulation. The summary records captured offload turns and cumulative counts alongside native counts, KL, top1 and divergence. F32 controls remain exact; bf16 remains lossy/exit5 with separate >=63/64 quality gates.

Glimmer's old16MB/s policy with its actual window also gives26 local/11offload. A CPU test records that positive fact. Calling it degenerate would be false. Its named degenerate negative uses2MB/s with the new sourced profile and expects37/0 plus the precise preflight refusal. The new20MB/s positive checks26/11. No window or recurrence bytes are omitted to manufacture a route mix.

## Build custody and execution

Node staging records only llama-hybrid-prefill. Both runbooks now call setup-rehearsal-client.sh without --skip-build, at the same exact R46_HEAD/BUILD and with NODE_ROOT/build.lock shared with the node stage. manage.py client builds and records llama-server, llama-hybrid-bench-check and llama-hybrid-score itself. The existing25 GiB configure/compile/runtime disk guard stays active; runbooks also retain their28GB rehearsal floor. Use the sequential stage/setup/policy/stop-launcher/run commands in each runbook. No manual target build, stamp editing, altered workload or resampling is needed.

CPU tests execute the runbooks' actual client command lines with fixture launchers, then use real manage.build/compile_targets stamp and hash logic with only compiler processes and git identity substituted. They reproduce missing binaries and manually built/unrecorded binaries, verify rejection leaves the node stamp untouched, then build/record all targets, accept exact reuse and reject a changed binary. This is build-record contract coverage; CUDA compilation and large-model execution remain GPU-operator validation. No native source or f32 wire/import path changed in this repair. Resource bounds remain Qwen6-10h/450GB work+30GB archive/128GB host/96GB GPU, and Glimmer3-6h/120GB work/96GB host/GPU after weights/build. The existing prior-rate Qwen rental scenario remains$9.42-$15.70, not a current quote. Route counts do not reduce checkpoint storage reservations.

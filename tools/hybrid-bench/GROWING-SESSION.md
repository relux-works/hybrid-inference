# Growing-session experiment

`growing_session.py` runs one hybrid session and two fully local sessions sequentially on the same client. It uses the existing proxy HTTP handler, policy, SQLite journal, replay worker, node protocol and native importer. It creates its own decoder, proxy database and slot directory. Launcher metadata is read only. Stop an operator-owned launcher client before this run to avoid holding two unnecessary model/context allocations; the driver never stops that client's processes.

The same GGUF SHA-256 is required at node and client. Different quantizations are outside this experiment. Temperature is zero, seeds are fixed, EOS is ignored, and generation lengths are checked. Input text uses the benchmark's existing `" x"` fixture. This is a synthetic agent call mix, not a task-success study.

## Sampling and starts

The source is measurement export v0.10, `stats.json`, `agents.all`, SHA-256 `8c5f1f2c44e344b78d32d0b4c216d0ebdf2ecdae3cde0f1a983cb329e488e2a7`. Like kit `agents_loop.py` v1.2, interpolation is linear below the median and log-linear above it. Only input p50/p90 are published per depth bin: <50k (406/6215), 50-100k (197/5344), 100-200k (98/2329), 200k+ (222/1448). The conditional input CDF starts at zero and is flat above p90. It does not invent a conditional p99. Output uses p10/p25/p50/p75/p90/p99 = 54/113/229/530/1135/4506. Gaps use 0.2/3.8/7.8/17.9/34.8/340 seconds. Marginals are sampled independently; no joint-distribution claim is made. Reasoning is already inside output and is not added again.

Before inference, the driver measures incremental ChatML framing with the actual renderer/tokenizer, then samples every turn using that growth. The same fixed sequence, including explicit caps and seeds, is replayed by both local sessions. Every actual committed depth and delta must match the plan; unexpected framing refuses. The second local session restarts the decoder with different batch and ubatch (defaults 64/32 versus 256/256). Both branches use normal user-message appends. `input_tokens` counts fixture text; actual `new_tokens` also includes template framing. Any exhausted context, unexpected generation length, replay of an old prompt, or missing checkpoint refuses rather than silently clamping or skipping.

D0 is the committed depth after a D0-1-token prompt and one sampled/evaluated base token. In `cold`, this start call is measured: node prefill/import for hybrid, local prefill for baseline. In `pre-staged`, both starts happen before either measured turn window. The hybrid base travels through verified protocol imports, then each arm's own base is saved as a native slot file and restored before its window. The fully local base is calculated locally. Pre-stage time and bytes remain in the trace. Preparation is automatic in the invocation; it is not a portable stage-file import command or an authorization to copy an unbound kit file.

The start hybrid call explicitly requests offload. Every subsequent hybrid call uses the real policy with no forced route. `--policy FILE` supplies a single calibrated policy for the whole run. A trace without both local and offloaded turns, or without an offload after reconnection, cannot pass. Do not tune policy per turn to obtain a desired chart.

The driver puts an owned TCP relay between proxy and configured node URL. At the one-based outage turn it terminates that relay by its recorded PID, leaving the node process and retained state intact. After the requested local-only turns it starts a new relay on the same port. Existing journal replay and critical prefix verification recover the suffix. This is an actual client-to-node transport outage, not a change to the policy. It tests reconnect/catch-up, not node RAM loss; the existing PR13 restart regression covers the latter separately. Native/checker/relay PID files and logs stay in the private output directory.

## Checkpoints and verdicts

At turns 10, 20, 30, 40 and the final turn, the proxy saves the actual native state just before final-sampled-token reconciliation. The checker uses the existing server token deserializer, validates snapshot hashes and token journals, restores all three accumulated states, evaluates that actual last token, and compares 64 full-vocabulary next-token distributions at the checkpoint depth. Local greedy IDs teacher-force both contexts. This tests the live growing states rather than rebuilding a fresh remote delta chain. Checkpoint state does not change the running session.

Same-backend acceptance uses the second local session as a partition-noise reference at the same checkpoints. For each of 64 local-greedy teacher-forced positions, report KL(local || hybrid) and KL(local || local_partition). The checkpoint KL limit is max(mean partition KL, nearest-rank p95 partition KL across the 64 positions). Accept only hybrid mean KL <= that limit, top-1 >=63/64, and no differing token in two independently free-running 64-token greedy continuations restored from the hybrid/local snapshots. All three sessions must have exactly matching journal hashes at every turn; snapshot checksums and native token journals are verified before scoring. The trace reports both mean KL values, their ratio, the p95 and the limit. A zero denominator has a null ratio and explicit both_zero/zero_noise_band status. Partition noise is a measured bound for this session, not a universal tolerance.

CUDA-to-Metal remains heterogeneous, exit5, with uncalibrated measurements even when within the observed client partition band. Different histories refuse comparison. The independent exact-state, checksum, journal and capacity controls remain strict; no numerical tolerance applies to those controls. Two-snapshot checker inputs still require exact logits. The driver requires a three-snapshot partition result and verifies its verdict independently.

`--catchup-mode decode` is deliberately absent. `Proxy.sync` sends one flat journal suffix with `return_delta=false`; after an outage it includes both locally prefilled prompts and generated tokens, and restart replay can include the entire history. The node has no record of their evaluation boundaries, batch sizes, or which tokens were generated. Single-token evaluation of that whole suffix would change the prompt path too and cannot guarantee the client's state. Clean exact replay needs a durable evaluation schedule in the journal/protocol, including prompt partitions and reconciliation steps; that protocol extension is outside this revision. No decode-mode exactness or relative timing is claimed. Batch catch-up native evaluation time comes from X-Hybrid-Prefill-Ms and is exported separately from restore/save/HTTP wall time.

Exit codes: 0 accepted same-backend session under the measured criterion, 2 refusal/aborted run, 3 measured criterion failure, 5 heterogeneous measurements without calibrated acceptance.

## Engine trace schema

The output is one atomically replaced, fsynced `trace.json`, `schema="hybrid-growing-session-v2"`. It is updated after every committed call and checker result and on termination. Read `status` before consuming results. `running` and `aborted` are incomplete; `failed` contains measured numerical failures; `heterogeneous` is uncalibrated. Unknown additive fields may be ignored. Missing required fields, unknown versions or incomplete arms must be rejected by a reader. All times are seconds except original `meter.*_ms` fields. Token IDs and conversations are absent from this export; the private SQLite/checker inputs contain them.

| Field | Type and meaning |
|---|---|
| `depth_plan` | D0, total growth including framing and all outputs, final depth, 2048 margin, native/allocated capacity and framing tokens |
| `final_depth` | Actual final depth per arm, required to equal the plan |
| `local_partition` | Alternate decoder batch and ubatch, both required to differ from the client metadata |
| `depth`, `turns`, `seed`, `start` | Integer base depth, turn count, seed; `cold` or `pre-staged` |
| `client`, `node` | Sanitized launcher identities: backend, model SHA, quantization, layout, context and available host/build/binary/toolchain data; no paths/credentials |
| `driver` | Driver and unchanged proxy source SHA-256 |
| `policy` | Full single-run `Policy` configuration |
| `link` | Operator physical-link label; when node `--bench-link` is enabled, three-probe median RTT and 1MiB application-download rate, bytes and duration; not physical/upload bandwidth |
| `provenance` | Source SHA, depth bins, quantiles and interpolation rule |
| `workload[]` | Preflight sampled sequence, verified against actual hybrid depths: `turn`, `conditioning_depth`, `seed`, `input_tokens`, `output_tokens`, `pause_seconds`, and uncapped `sampled_*` values; input material is always `" x" * input_tokens` |
| `disconnect` | One-based first outage turn and duration in turns; turn zero disables |
| `calls[]` | `arm` (`hybrid`/`local`/`local_partition`), `stage` (`start`/`turn`), `turn`, requested sequence row and actual component measurements below |
| `events[]` | Ordered observations with monotonic `at_seconds` relative to process start; protocol/sync, pre-stage, disconnect/reconnect, window and cleanup events |
| `checkpoints[]` | Turn, depth, k=64, exactness, top-1 count, mean KL(local || hybrid), max absolute logit difference, per-position data, native exit code, status, three snapshot hashes, journal hash, local partition mean/p95 KL, hybrid-to-partition mean KL ratio, KL limit and free-running first divergence (-1 means none) |
| `started_utc`, `ended_utc` | Actual UTC observations |

Each call records `depth_before`, `depth_after`, actual `new_tokens`, `generated_tokens`, `decision.inputs` (delta excluding held token, depth, previous offload, current measured link rate, server factor) and `decision.result` (policy choice and estimated local/offload seconds). `path` records actual execution; `fallback_reason` explains a requested offload that continued locally. The unmodified complete proxy `meter` includes chunks, local/remote work and replay acknowledgements.

`ttft_seconds` is wall time to the first nonempty streamed text/reasoning delta. Components are `local_prefill_seconds` for local calls, or `server_prefill_seconds`, body-only `transfer_seconds`, HTTP/native `import_seconds` and `local_tail_seconds` for offload (including any locally processed remainder after a partial-transfer fallback). `ttft_other_seconds` is the signed remainder, including rendering, queueing, node restore/save/export, synchronization and HTTP overhead. It is not clipped. `remote_prepare_seconds` is a containing interval; do not sum it again. Native prompt time covers prefill/tail, not decoder startup. `reconciliation_seconds` covers finish/journal work after streamed generation. `total_seconds` includes that work and any checkpoint save; `checkpoint_save_seconds` identifies that instrumentation cost. Decode rate uses generated tokens after the first divided by native predicted time; single-token starts have null rate.

`wire_bytes` is received delta-envelope body bytes. Protocol events give JSON request-body bytes and successful response-body bytes per method/path and duration, including catch-up uploads. Request bytes on a failed event describe the attempted body, not confirmed delivery. These are application-body counters, not TCP/TLS/SSH/IP/5G byte counts. `catchup_tokens`, `catchup_seconds`, `catchup_upload_bytes` include successful journal extensions since the previous call, including during its pause. `catchup_node_seconds` sums the node evaluation durations from successful catch-up response headers; it excludes native restore/save, serialization and HTTP overhead. They can overlap local work and must not be summed into TTFT. The first offload after reconnect has `resync` with recovered tokens, protocol time (including prefix reads), catch-up time and upload bytes. Raw protocol/sync events permit reconstruction of the acknowledgement sequence. `journal_hash` is SHA-256 of little-endian int32 evaluated token IDs, identical to the protocol prefix hash.

For turn-only comparisons, sum `stage="turn"` call components by arm and include pauses separately. For cold end-to-end comparisons, include each arm's `stage="start"` row. Pre-staged totals exclude start rows and `pre_stage` events. `window.seconds` is observed wall time, including instrumentation, checker work in the baseline window and pauses; it is not a pure inference total. The decoder runs hybrid then local at its original partition, then restarts for local_partition. The partition arm prepares its own base outside its pre-staged window. Three checker contexts coexist with the decoder; free-running checks reuse two after restoring their snapshots. Arm order and cache warmth are limitations. Report this order as a limitation.

## CPU commands (existing models only)

Use an existing CMake and GCC/Clang toolchain. On agents1, set an on-disk TMPDIR and disable bytecode, use the shared build lock, at most two niced builds with at most four jobs each, and keep free disk above 28 GB. Do not download a model. The qualification commands need existing local 0.8B Q4_0 weights; the tests never fetch them.

```sh
mkdir -p .temp/growing/tmp .temp/growing/native
export TMPDIR="$PWD/.temp/growing/tmp"
export PYTHONDONTWRITEBYTECODE=1
nice -n 19 cmake -S . -B .temp/growing/build -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
flock ~/Werkstatt/hybrid-models/.build.lock nice -n 19 cmake --build .temp/growing/build -j4 --target llama-server llama-hybrid-prefill llama-hybrid-bench-check
python3 -m unittest discover -s tools/hybrid-bench -p test_growing_session.py -v
nice -n 19 python3 tools/hybrid-bench/test_growing_session_native.py --build-dir .temp/growing/build --model "$MODEL" --artifacts .temp/growing/native
```

The native command runs positive exactness/checksum/journal instrument controls, then both start modes with 12 turns, a real two-turn outage, local/offload routing, and 64-token checkpoints at 10 and 12. It accepts only when the measured session criterion is green for both starts; the original exact-state controls are unchanged. `--controls-only` runs only the explicitly limited native instrument controls. Remove only this task's build tree after process cleanup, retaining logs/JSON and hashes.

## PRO 6000 GPU rehearsal (operator host only)

Use a dedicated PRO 6000 host, enough host RAM for retained node states and transient copies, and a GGUF already qualified at >=150k context. This run is not authorized on agents1. The example reserves 196608 tokens. An actual 27B Q4 model uses about 16GiB of weights and 6.5GiB of q8_0 state per context at that capacity; node, decoder and the three checker contexts can coexist. Budget approximately 80-96GiB VRAM and >=128GiB host RAM, then verify the actual launcher admission and peak allocations. The node's RAM snapshots do not bound total RSS. A smaller rehearsal model changes the configuration and must have its own trace.

Build/deploy at the signed experiment branch head, using the exact same GGUF on both roles. The existing launchers build the required binaries and record their hashes. Set a long timeout for whole-state saves and hashes. Policy fields must come from measured marginal client prefill by depth, node prefill, link rate, fixed import overhead and actual GGUF geometry. Save them once as `.temp/growing-gpu/policy.json`; do not use `always=true` to force mixed routing. A run that naturally selects only one path is a missing experiment, not a pass.

```sh
mkdir -p .temp/growing-gpu/tmp
export TMPDIR="$PWD/.temp/growing-gpu/tmp"
export PYTHONDONTWRITEBYTECODE=1
tools/hybrid-bench/deploy-prefill-node.sh --state .temp/growing-gpu/node --model "$MODEL" --sha256 "$MODEL_SHA256" --ctx 196608 --cuda --bench-link --request-timeout 3600 --startup-timeout 3600
tools/hybrid-bench/setup-client.sh --state .temp/growing-gpu/client --model "$MODEL" --sha256 "$MODEL_SHA256" --ctx 196608 --cuda --node-info .temp/growing-gpu/node/node-info.json --token-file .temp/growing-gpu/node/client-tokens.json --request-timeout 3600 --startup-timeout 3600
tools/hybrid-bench/setup-client.sh --state .temp/growing-gpu/client --stop
python3 tools/hybrid-bench/growing_session.py --client-state .temp/growing-gpu/client --output .temp/growing-gpu/run-100k --depth 100000 --turns 40 --seed 1 --start pre-staged --disconnect-turn 20 --disconnect-turns 3 --policy .temp/growing-gpu/policy.json --link-label 'single PRO 6000 host, loopback, CUDA to CUDA'
tools/hybrid-bench/deploy-prefill-node.sh --state .temp/growing-gpu/node --stop
```

Repeat with `--start cold` and a new output directory to include initial fill. Use the session-measured partition band; do not replace a failed criterion with an arbitrary tolerance. Budget 2-5 hours as an unmeasured planning allowance for three 40-turn arms, pauses and four native checkpoint comparisons, not a measured GPU duration. Seed 1 has about 27k output tokens and 13.2 minutes of pauses per arm. At 10-30 decode tokens/s that alone is about 28-58 minutes per arm before new-input work and checkpoint overhead.

## Incident Mac run (Vast CUDA node to M3 Max)

The incident lane owns the Vast rental, Mac execution, real link, credentials and model delivery. Provision only there. Use the existing SSH-tunnel pattern in the [operator guide](README.md#2-open-the-ssh-tunnel-from-the-client); record that the traffic traverses the Mac's physical link. Keep a record of host GPU/driver, Mac RAM/chip/OS, GGUF SHA, binary hashes, commit, KV types, policy calibration and link label. Node and client must use the same GGUF; CUDA-to-Metal is heterogeneous solely by backend. Copy `node-info.json` and the mode-0600 token JSON securely using the operator's established channel. Do not put bearer tokens in shell arguments or exported traces.

For D0=100000, provision the node with 196608 context as above on Vast. On the Mac, omit `--cuda`; the launcher selects Metal. A 96/128GB M3 Max needs actual allocation admission for a 27B Q4 model with the live decoder and three additional checker contexts. A 64GB machine needs actual allocation evidence before admission; this driver does not unload the decoder for checks. A deep 262144-context rehearsal with all roles on one 96GiB GPU may exceed its VRAM once three checker contexts and compute buffers coexist; use actual admission and adequate hardware. Keep node RAM large enough for the full snapshots and transient staging; inspect `node-info.json` and authenticated health rather than assuming a 512MiB store is sufficient.

```sh
mkdir -p .temp/growing-mac/tmp
export TMPDIR="$PWD/.temp/growing-mac/tmp"
export PYTHONDONTWRITEBYTECODE=1
tools/hybrid-bench/setup-client.sh --state .temp/growing-mac/client-100k --model "$MODEL" --sha256 "$MODEL_SHA256" --ctx 196608 --node-info /secure/node-info.json --token-file /secure/client-tokens.json --prefill-url http://127.0.0.1:8091 --request-timeout 3600 --startup-timeout 3600
tools/hybrid-bench/setup-client.sh --state .temp/growing-mac/client-100k --stop
python3 tools/hybrid-bench/growing_session.py --client-state .temp/growing-mac/client-100k --output .temp/growing-mac/run-100k --depth 100000 --turns 40 --seed 1 --start pre-staged --disconnect-turn 20 --disconnect-turns 3 --policy .temp/growing-mac/policy.json --link-label 'Vast CUDA to M3 Max Metal, Mac physical 5G link via SSH tunnel'
```

The run prepares each arm's own native base before its measured turns. Start rows and `pre_stage` events show the cost and bytes; archived native base files remain in `slots/`. A cold run uses the same command with `--start cold` and a new output directory. Leave the tunnel and node available until cleanup events confirm no remaining remote session IDs. The driver stops its own decoder/relay/checkers; use the node operator's recorded state to stop the actual node afterward.

**Native depth rule:** keep the 100000 start. For the deep arm, use `--depth deep` and allocate exactly 262144 context on both roles, without YaRN or another context extension. The driver selects the largest multiple of 1024 whose complete depth-conditioned sampled sequence fits with a 2048-token margin: D0 + sum(all input deltas including template framing + all outputs) + 2048 <=262144. It scans candidate starts in descending order, sampling the fixed seed at each candidate's depths, so boundary-dependent inputs are included. First it verifies that the sequence also fits at D0=100000; otherwise it refuses without truncating any turn. The actual D0, growth and final depth are in depth_plan and are verified against committed calls. Small explicit depths remain available for CPU controls.

Deploy a separate node/client at 262144 using the same identity/tunnel flow, then stop the launcher client and run:

```sh
python3 tools/hybrid-bench/growing_session.py --client-state .temp/growing-mac/client-deep --output .temp/growing-mac/run-deep --depth deep --turns 40 --seed 1 --start pre-staged --disconnect-turn 20 --disconnect-turns 3 --policy .temp/growing-mac/policy.json --link-label 'Vast CUDA to M3 Max Metal, Mac physical 5G link via SSH tunnel; native context'
```

Repeat the 100k and deep arms with cold starts and fresh output paths as needed. Do not use the former 250000 start or 128/128 tail caps to stand in for the deep arm. Both native contexts must fit the full sequence and margin before inference. No model download or GPU experiment is authorized on agents1.

Disk admission is `28e9 + (2 * checkpoint_count + 6) * S(context)` free bytes, with `S ~= recurrent_bytes + (kv_bytes_per_token + 32) * context`. For the illustrative 27B geometry, reserve >126GB free at 196608 capacity and >158GB at 262144, in addition to already-installed model/build files. Round to 150GB and 180GB free respectively. The node needs model/build space and RAM, not a disk state store. Snapshots can approach 7GB and 9.3GB each at these capacities. Do not stage them in RAM-backed `/tmp` or archive the private output directory wholesale.

Use 3-9 hours as an unmeasured 100k three-session Mac planning allowance, plus initial fill and transport variability; reserve 4-12 hours for the deep arm until measured. Both include three copies of the same output/pause workload and sequential 64-token teacher-forced/free-running checks. The earlier 39,504-token fresh call (local 271.5s, hybrid 201.7s) does not establish growing-turn costs. Keep checkpoint hashing/loading and node whole-state restore/save observations separate from TTFT.

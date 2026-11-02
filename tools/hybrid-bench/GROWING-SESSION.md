# Growing-session experiment

`growing_session.py` runs one hybrid session and one fully local session sequentially on the same client. It uses the existing proxy HTTP handler, policy, SQLite journal, replay worker, node protocol and native importer. It creates its own decoder, proxy database and slot directory. Launcher metadata is read only. Stop an operator-owned launcher client before this run to avoid holding two unnecessary model/context allocations; the driver never stops that client's processes.

The same GGUF SHA-256 is required at node and client. Different quantizations are outside this experiment. Temperature is zero, seeds are fixed, EOS is ignored, and generation lengths are checked. Input text uses the benchmark's existing `" x"` fixture. This is a synthetic agent call mix, not a task-success study.

## Sampling and starts

The source is measurement export v0.10, `stats.json`, `agents.all`, SHA-256 `8c5f1f2c44e344b78d32d0b4c216d0ebdf2ecdae3cde0f1a983cb329e488e2a7`. Like kit `agents_loop.py` v1.2, interpolation is linear below the median and log-linear above it. Only input p50/p90 are published per depth bin: <50k (406/6215), 50-100k (197/5344), 100-200k (98/2329), 200k+ (222/1448). The conditional input CDF starts at zero and is flat above p90. It does not invent a conditional p99. Output uses p10/p25/p50/p75/p90/p99 = 54/113/229/530/1135/4506. Gaps use 0.2/3.8/7.8/17.9/34.8/340 seconds. Marginals are sampled independently; no joint-distribution claim is made. Reasoning is already inside output and is not added again.

Hybrid input sampling uses the actual previous committed depth. Its sequence, including all explicit caps and seeds, is then replayed unchanged by the local baseline. The initial plan uses estimated framing only for capacity admission; per-call bounds use the actual depth. Both branches use normal user-message appends. `input_tokens` counts fixture text; actual `new_tokens` also includes template framing. Any exhausted context, unexpected generation length, replay of an old prompt, or missing checkpoint refuses rather than silently clamping or skipping.

D0 is the committed depth after a D0-1-token prompt and one sampled/evaluated base token. In `cold`, this start call is measured: node prefill/import for hybrid, local prefill for baseline. In `pre-staged`, both starts happen before either measured turn window. The hybrid base travels through verified protocol imports, then each arm's own base is saved as a native slot file and restored before its window. The fully local base is calculated locally. Pre-stage time and bytes remain in the trace. Preparation is automatic in the invocation; it is not a portable stage-file import command or an authorization to copy an unbound kit file.

The start hybrid call explicitly requests offload. Every subsequent hybrid call uses the real policy with no forced route. `--policy FILE` supplies a single calibrated policy for the whole run. A trace without both local and offloaded turns, or without an offload after reconnection, cannot pass. Do not tune policy per turn to obtain a desired chart.

The driver puts an owned TCP relay between proxy and configured node URL. At the one-based outage turn it terminates that relay by its recorded PID, leaving the node process and retained state intact. After the requested local-only turns it starts a new relay on the same port. Existing journal replay and critical prefix verification recover the suffix. This is an actual client-to-node transport outage, not a change to the policy. It tests reconnect/catch-up, not node RAM loss; the existing PR13 restart regression covers the latter separately. Native/checker/relay PID files and logs stay in the private output directory.

## Checkpoints and verdicts

At turns 10, 20, 30, 40 and the final turn, the proxy saves the actual native state just before final-sampled-token reconciliation. The checker uses the existing server token deserializer, validates snapshot hashes and token journals, restores both accumulated states, evaluates that actual last token, and compares 64 full-vocabulary next-token distributions at the checkpoint depth. Local greedy IDs teacher-force both contexts. This tests the live growing states rather than rebuilding a fresh remote delta chain. Checkpoint state does not change the running session.

Identical CPU/CUDA backends require bit-exact logits at all 64 positions, KL=0 and top-1=64/64. A completed experiment that fails this gate exits 3 with `status="failed"`. CUDA-to-Metal results have `status="heterogeneous"`, exit 5, and uncalibrated numerical measurements; they never become an exactness pass. Differing token histories refuse the comparison and abort: KL over different histories is not reported as equivalence. The driver independently checks checker verdict consistency.

**Current blocker:** the measured CPU mixed-route session is not bit-exact against the natural local baseline. See [r36 results](RESULTS-hybrid-b-r36-261102.md) and [open decisions](../../UNRESOLVED_QUESTIONS.md). Keep the strict gate until the experiment owner resolves the criterion. A passing same-snapshot instrument control does not qualify mixed-route equivalence.

Exit codes: 0 exact accepted run, 2 refusal/aborted run, 3 measured same-backend exactness failure, 5 heterogeneous measurements without calibrated acceptance. The native qualification harness exits nonzero while mixed-route exactness remains red.

## Engine trace schema

The output is one atomically replaced, fsynced `trace.json`, `schema="hybrid-growing-session-v1"`. It is updated after every committed call and checker result and on termination. Read `status` before consuming results. `running` and `aborted` are incomplete; `failed` contains measured numerical failures; `heterogeneous` is uncalibrated. Unknown additive fields may be ignored. Missing required fields, unknown versions or incomplete arms must be rejected by a reader. All times are seconds except original `meter.*_ms` fields. Token IDs and conversations are absent from this export; the private SQLite/checker inputs contain them.

| Field | Type and meaning |
|---|---|
| `depth`, `turns`, `seed`, `start` | Integer base depth, turn count, seed; `cold` or `pre-staged` |
| `client`, `node` | Sanitized launcher identities: backend, model SHA, quantization, layout, context and available host/build/binary/toolchain data; no paths/credentials |
| `driver` | Driver and unchanged proxy source SHA-256 |
| `policy` | Full single-run `Policy` configuration |
| `link` | Operator physical-link label; when node `--bench-link` is enabled, three-probe median RTT and 1MiB application-download rate, bytes and duration; not physical/upload bandwidth |
| `provenance` | Source SHA, depth bins, quantiles and interpolation rule |
| `workload[]` | Actual hybrid-sampled sequence: `turn`, `conditioning_depth`, `seed`, `input_tokens`, `output_tokens`, `pause_seconds`, and uncapped `sampled_*` values; input material is always `" x" * input_tokens` |
| `disconnect` | One-based first outage turn and duration in turns; turn zero disables |
| `calls[]` | `arm` (`hybrid`/`local`), `stage` (`start`/`turn`), `turn`, requested sequence row and actual component measurements below |
| `events[]` | Ordered observations with monotonic `at_seconds` relative to process start; protocol/sync, pre-stage, disconnect/reconnect, window and cleanup events |
| `checkpoints[]` | Turn, depth, k=64, exactness, top-1 count, mean KL(local || hybrid), max absolute logit difference, per-position data, native exit code, status, snapshot hashes |
| `started_utc`, `ended_utc` | Actual UTC observations |

Each call records `depth_before`, `depth_after`, actual `new_tokens`, `generated_tokens`, `decision.inputs` (delta excluding held token, depth, previous offload, current measured link rate, server factor) and `decision.result` (policy choice and estimated local/offload seconds). `path` records actual execution; `fallback_reason` explains a requested offload that continued locally. The unmodified complete proxy `meter` includes chunks, local/remote work and replay acknowledgements.

`ttft_seconds` is wall time to the first nonempty streamed text/reasoning delta. Components are `local_prefill_seconds` for local calls, or `server_prefill_seconds`, body-only `transfer_seconds`, HTTP/native `import_seconds` and `local_tail_seconds` for offload (including any locally processed remainder after a partial-transfer fallback). `ttft_other_seconds` is the signed remainder, including rendering, queueing, node restore/save/export, synchronization and HTTP overhead. It is not clipped. `remote_prepare_seconds` is a containing interval; do not sum it again. Native prompt time covers prefill/tail, not decoder startup. `reconciliation_seconds` covers finish/journal work after streamed generation. `total_seconds` includes that work and any checkpoint save; `checkpoint_save_seconds` identifies that instrumentation cost. Decode rate uses generated tokens after the first divided by native predicted time; single-token starts have null rate.

`wire_bytes` is received delta-envelope body bytes. Protocol events give JSON request-body bytes and successful response-body bytes per method/path and duration, including catch-up uploads. Request bytes on a failed event describe the attempted body, not confirmed delivery. These are application-body counters, not TCP/TLS/SSH/IP/5G byte counts. `catchup_tokens`, `catchup_seconds`, `catchup_upload_bytes` include successful journal extensions since the previous call, including during its pause. They can overlap local work and must not be summed into TTFT. The first offload after reconnect has `resync` with recovered tokens, protocol time (including prefix reads), catch-up time and upload bytes. Raw protocol/sync events permit reconstruction of the acknowledgement sequence. `journal_hash` is SHA-256 of little-endian int32 evaluated token IDs, identical to the protocol prefix hash.

For turn-only comparisons, sum `stage="turn"` call components by arm and include pauses separately. For cold end-to-end comparisons, include each arm's `stage="start"` row. Pre-staged totals exclude start rows and `pre_stage` events. `window.seconds` is observed wall time, including instrumentation, checker work in the baseline window and pauses; it is not a pure inference total. The decoder is loaded once and both arms run sequentially, so startup is excluded and arm order is hybrid then local. Report this order as a limitation.

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

The native command runs positive exactness/checksum/journal instrument controls, then both start modes with 12 turns, a real two-turn outage, local/offload routing, and 64-token checkpoints at 10 and 12. It remains a failing acceptance command while the strict mixed-route gate is unmet. `--controls-only` runs only the explicitly limited native instrument controls. Remove only this task's build tree after process cleanup, retaining logs/JSON and hashes.

## PRO 6000 GPU rehearsal (operator host only)

Use a dedicated PRO 6000 host, enough host RAM for retained node states and transient copies, and a GGUF already qualified at >=150k context. This run is not authorized on agents1. The example reserves 196608 tokens. An actual 27B Q4 model uses about 16GiB of weights and 6.5GiB of q8_0 state per context at that capacity; node, decoder and the two checker contexts can coexist. Budget approximately 80-96GiB VRAM and >=128GiB host RAM, then verify the actual launcher admission and peak allocations. The node's RAM snapshots do not bound total RSS. A smaller rehearsal model changes the configuration and must have its own trace.

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

Repeat with `--start cold` and a new output directory to include initial fill. Expect the strict exactness criterion to be a gate, not a promise of success. CUDA nondeterminism or partition differences are not excused automatically. Budget 1-3 hours as a planning allowance for two 40-turn arms, pauses and four native checkpoint comparisons, not a measured GPU duration. Seed 1 has about 27k output tokens and 13.2 minutes of pauses per arm. At 10-30 decode tokens/s that alone is about 28-58 minutes per arm before new-input work and checkpoint overhead.

## Incident Mac run (Vast CUDA node to M3 Max)

The incident lane owns the Vast rental, Mac execution, real link, credentials and model delivery. Provision only there. Use the existing SSH-tunnel pattern in the [operator guide](README.md#2-open-the-ssh-tunnel-from-the-client); record that the traffic traverses the Mac's physical link. Keep a record of host GPU/driver, Mac RAM/chip/OS, GGUF SHA, binary hashes, commit, KV types, policy calibration and link label. Node and client must use the same GGUF; CUDA-to-Metal is heterogeneous solely by backend. Copy `node-info.json` and the mode-0600 token JSON securely using the operator's established channel. Do not put bearer tokens in shell arguments or exported traces.

For D0=100000, provision the node with 196608 context as above on Vast. On the Mac, omit `--cuda`; the launcher selects Metal. A 96/128GB M3 Max is the conservative target for a 27B Q4 model with the live decoder and two additional checker contexts. A 64GB machine needs actual allocation evidence before admission; this driver does not unload the decoder for checks. Keep node RAM large enough for the full snapshots and transient staging; inspect `node-info.json` and authenticated health rather than assuming a 512MiB store is sufficient.

```sh
mkdir -p .temp/growing-mac/tmp
export TMPDIR="$PWD/.temp/growing-mac/tmp"
export PYTHONDONTWRITEBYTECODE=1
tools/hybrid-bench/setup-client.sh --state .temp/growing-mac/client-100k --model "$MODEL" --sha256 "$MODEL_SHA256" --ctx 196608 --node-info /secure/node-info.json --token-file /secure/client-tokens.json --prefill-url http://127.0.0.1:8091 --request-timeout 3600 --startup-timeout 3600
tools/hybrid-bench/setup-client.sh --state .temp/growing-mac/client-100k --stop
python3 tools/hybrid-bench/growing_session.py --client-state .temp/growing-mac/client-100k --output .temp/growing-mac/run-100k --depth 100000 --turns 40 --seed 1 --start pre-staged --disconnect-turn 20 --disconnect-turns 3 --policy .temp/growing-mac/policy.json --link-label 'Vast CUDA to M3 Max Metal, Mac physical 5G link via SSH tunnel'
```

The run itself prepares both native bases before measured turns. Start rows and `pre_stage` events show the cost and bytes; archived native base files remain in `slots/`. A cold run uses the same command with `--start cold` and a new output directory. Leave the tunnel and node available until cleanup events confirm no remaining remote session IDs. The driver stops its own decoder/relay/checkers; use the node operator's recorded state to stop the actual node afterward.

**D0=250000 capacity decision:** the uncapped default seed-1 workload needs about 293k tokens, while the current launcher rejects a context beyond GGUF training capacity (262144 on the existing fixture). Do not force a larger RoPE/context setting to make it launch. The owner must choose either a model/configuration already qualified beyond 300k, or the following explicitly capped 40-turn workload. For the capped variant, deploy a separate node/client at 262144 context with the same identity/tunnel flow, stop its launcher client, then run:

```sh
python3 tools/hybrid-bench/growing_session.py --client-state .temp/growing-mac/client-250k --output .temp/growing-mac/run-250k-capped --depth 250000 --turns 40 --seed 1 --start pre-staged --disconnect-turn 20 --disconnect-turns 3 --max-new 128 --max-output 128 --policy .temp/growing-mac/policy.json --link-label 'Vast CUDA to M3 Max Metal, Mac physical 5G link via SSH tunnel; explicit 128/128 caps'
```

This variant fits a 262144-token context by conservative admission. It truncates the measured input/output tails; label it separately and preserve every sampled/capped value. It may not exercise both routes under an honest calibrated policy, in which case it refuses acceptance. It does not fulfill an uncapped 250k experiment. The larger-context alternative is blocked until its model/context is qualified; no run command here silently authorizes extrapolation.

Disk admission is `28e9 + (checkpoint_count + 4) * S(context)` free bytes, with `S ~= recurrent_bytes + (kv_bytes_per_token + 32) * context`. For the illustrative 27B geometry, reserve >84GB free at 196608 capacity and >103GB at 262144, in addition to already-installed model/build files. Round to 100GB and 120GB free respectively. The node needs model/build space and RAM, not a disk state store. Snapshots can approach 7GB and 9.3GB each at these capacities. Do not stage them in RAM-backed `/tmp` or archive the private output directory wholesale.

Use 2-6 hours as a 100k paired Mac planning allowance, plus initial fill, retries and transport variability. The capped 250k run has fewer decode tokens, but deeper-state serialization and checks; reserve 1-4 hours until measured. An uncapped 250k run needs its own estimate after capacity qualification. These are scheduling bounds, not measured results. The earlier 39,504-token fresh call (local 271.5s, hybrid 201.7s) does not establish growing-turn costs. Checkpoint full-state hashing/loading and the node's whole-state restore/save can dominate; keep their observations separate from TTFT.

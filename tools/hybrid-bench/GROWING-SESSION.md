# Growing-session experiment

`growing_session.py` runs one hybrid session and two fully local sessions sequentially on the same client. It uses the existing proxy HTTP handler, policy, SQLite journal, replay worker, node protocol and native importer. It creates its own decoder, proxy database and slot directory. Launcher metadata is read only. Stop an operator-owned launcher client before this run to avoid holding two unnecessary model/context allocations; the driver never stops that client's processes.

The same GGUF SHA-256 is required at node and client. Different quantizations are outside this experiment. Temperature is zero, seeds are fixed, EOS is ignored, and generation lengths are checked. Input text uses the benchmark's existing `" x"` fixture. This is a synthetic agent call mix, not a task-success study.

## Sampling and starts

The source is measurement export v0.10, `stats.json`, `agents.all`, SHA-256 `8c5f1f2c44e344b78d32d0b4c216d0ebdf2ecdae3cde0f1a983cb329e488e2a7`. Like kit `agents_loop.py` v1.2, interpolation is linear below the median and log-linear above it. Only input p50/p90 are published per depth bin: <50k (406/6215), 50-100k (197/5344), 100-200k (98/2329), 200k+ (222/1448). The conditional input CDF starts at zero and is flat above p90. Its lower half puts p25 at median/2; the measured global p25/median is 0.20, so this construction over-sizes small conditional deltas. It is a bounded synthetic marginal, not an inferred measured conditional CDF. Growing output/pause sampling adds (0,0); hybrid_bench.quantile_sample instead floors at p10. The audited source rows and selectors are retained in fixtures/growing-provenance.json and its SHA is exported in each trace. It does not invent a conditional p99. Output uses p10/p25/p50/p75/p90/p99 = 54/113/229/530/1135/4506. Gaps use 0.2/3.8/7.8/17.9/34.8/340 seconds. Marginals are sampled independently; no joint-distribution claim is made. Reasoning is already inside output and is not added again.

Before inference, the driver measures incremental ChatML framing with the actual renderer/tokenizer, then samples every turn using that growth. The same fixed sequence, including explicit caps and seeds, is replayed by both local sessions. Every actual committed depth and delta must match the plan; unexpected framing refuses. The second local session restarts the decoder with different batch and ubatch (defaults 64/32 versus 256/256). Both branches use normal user-message appends. `input_tokens` counts fixture text; actual `new_tokens` also includes template framing. Any exhausted context, unexpected generation length, replay of an old prompt, or missing checkpoint refuses rather than silently clamping or skipping.

D0 is the committed depth after a D0-1-token prompt and one sampled/evaluated base token. In `cold`, this start call is measured: node prefill/import for hybrid, local prefill for baseline. In `pre-staged`, both starts happen before either measured turn window. The hybrid base travels through verified protocol imports, then each arm's own base is saved as a native slot file and restored before its window. The fully local base is calculated locally. Pre-stage time and bytes remain in the trace. Preparation is automatic in the invocation; it is not a portable stage-file import command or an authorization to copy an unbound kit file.

The start hybrid call explicitly requests offload. Every subsequent hybrid call uses the real policy with no forced route. `--policy FILE` supplies a single calibrated policy for the whole run. Before any base or measured arm, a constant-link simulation of all sampled turns must predict both connected policy-local and policy-offload turns. Degenerate configurations refuse before inference. The live gate separately requires connected policy-local turns without fallback, successful policy-offloads, connected catch-up of a policy-local prompt, and an offload after reconnection. Fallbacks during an outage never count as policy-local evidence. Link EMA and outage hysteresis can change live decisions; prediction is recorded alongside actual route counts. Do not tune policy per turn to obtain a desired chart.

The driver puts an owned TCP relay between proxy and configured node URL. At the one-based outage turn it terminates that relay by its recorded PID, leaving the node process and retained state intact. After the requested local-only turns it starts a new relay on the same port. Existing journal replay and critical prefix verification recover the suffix. This is an actual client-to-node transport outage, not a change to the policy. It tests reconnect/catch-up, not node RAM loss; the existing PR13 restart regression covers the latter separately. Native/checker/relay PID files and logs stay in the private output directory.

## Checkpoints and verdicts

At turns 10, 20, 30, 40 and the final turn, the proxy saves the actual native state just before final-sampled-token reconciliation. The checker uses the existing server token deserializer, validates snapshot hashes and token journals, restores all three accumulated states, evaluates that actual last token, and compares 64 full-vocabulary next-token distributions at the checkpoint depth. Local greedy IDs teacher-force both contexts. This tests the live growing states rather than rebuilding a fresh remote delta chain. Checkpoint state does not change the running session.

R41 gates the explained append-import mechanism with `status_rule=session_append_replay_exact`. The client retains historical attention KV, imports only the new node KV suffix, and replaces recurrent state. The node computed that suffix over its own catch-up history. Batch/token arithmetic can make historical KV different, so this state is deliberately a mixture of two histories. The r39 append-chain replay reproduces live H exactly; replacing attention alone gives the coherent node replay B. Mean KL(local || hybrid)=0 is therefore not the correctness contract for append import.

Every checkpoint requires bit-exact CHAIN-H over all 64 full-vocabulary positions (max logit difference 0), NODE-B, RESTORE-B, BSTATE-B, NODE_RANGE-B, OWN_NATIVE-A, OWN_RANGE-A, CHAIN_ATTENTION-B, CHAIN_RECURRENT-H and A-L. Save/load and own/range roundtrips must preserve bytes, BSTATE must match the live node bytes, and both component replacements must confirm equal recurrent bytes before replacement. Attention bytes may already be equal on another backend; a difference is reported, not required. Exact control comparisons also require 64/64 top-1 and equal 64-token independent greedy continuations. Live journals and every snapshot/delta/raw-logit SHA must validate. L-H teacher-forced top-1>=63/64 is the sole L-H sanity gate. Free-running first divergence is reported at every checkpoint and never gates L-H. Missing controls abort; a measured mismatch fails exit3. Two-snapshot exact controls and legacy zero-band fixtures keep their original rules.

Report L-H mean KL, nearest-rank p95 (sorted element61), maximum absolute logit difference, all positions and free-running first divergence at every checkpoint as the measured cost of append import. None of these numerical costs is suppressed or used as a hidden positive tolerance. `kl_limit` is null for this rule. Retain partition mean/p95 and the hybrid-to-partition mean ratio (null with both_zero or zero_noise_band if the partition mean is 0). `gate_controls` records each required identity; Python independently checks it against the raw logits and byte controls. The r41 CPU smoke passes both checkpoints, including turn 12 with free divergence 0 and top-1 63/64. The native checker observes free continuations; teacher-forced raw logits cannot independently re-derive them.

`fidelity_flag` in each exported checkpoint compares L-H against all three noise cells A-B1, A-B2 and A-B, per statistic and like-for-like only. Its keys are `mean_kl_nats`, `p95_kl_nats`, `max_abs_logit_diff`, `top1_agreed` and `free_running_first_divergence`. A flag is true only if L-H is strictly worse than every cell: greater mean/p95/max, lower top-1, or earlier free divergence, with -1 ranked as positive infinity. Equality with any cell leaves that flag false. Flags are derived after validation, are report-only and never change an exit code. CPU turn 10 flags mean and p95; turn 12 flags nothing.

Four fresh replay contexts cross prompt and generated evaluation paths. A uses local prompt batches with logits only at prompt ends and one-token generated decode. B1 changes generated spans to intersections with exact node catch-up batches. B2 changes prompt spans to intersections with the exact node schedule and chunk-end logits, retaining generated decode. B preserves the full flat node batches up to the last imported boundary, including mixed prompt/generated outage batches, then follows the client tail. Splitting a mixed batch in either one-factor cell is explicit; only B retains that interaction. Calls provide prompt/generated boundaries and successful per-session protocol extends provide the exact flat node ranges. The retained input contains tokens and all four schedules. The checker reports A-A, A-B1, A-B2, A-B, A-L and B-H, all over the same 64 local-greedy teacher-forced IDs, plus independent greedy continuations. A-L is the local replay control; a nonzero A-L limits attribution. Source replay currently requires the primary local decoder at 256/256, two threads, q8_0 KV and flash attention, as in these traces; unsupported partitions refuse.

The checker writes hashed little-endian float32 full-vocabulary logits for every position in L/H/P/A/B1/B2/B. Python independently recomputes each position number, KL, top-1, exactness and max logit error, then re-derives all numerical summaries and checks the rule. Independent greedy continuations are exercised by the native checker; the Python raw-logit audit covers the teacher-forced distributions. Snapshots are removed only after successful scoring and validation; inputs, logits, hashes, results and PID/command logs remain private artifacts. Two-snapshot controls retain strict bit-exactness. CUDA-to-Metal remains heterogeneous, exit 5 and uncalibrated. On the Mac, CHAIN-H, A-L, OWN_NATIVE-A and OWN_RANGE-A (OWN_*-A) are client-backend identities expected exact; a mismatch is a finding even though the run remains exit 5. NODE-B, RESTORE-B, BSTATE-B, NODE_RANGE-B and CHAIN_ATTENTION-B compare across backends and are report-only there. Different histories refuse.

`--catchup-mode decode` is deliberately absent. `Proxy.sync` sends one flat journal suffix with `return_delta=false`; after an outage it includes both locally prefilled prompts and generated tokens, and restart replay can include the entire history. The node has no record of their evaluation boundaries, batch sizes, or which tokens were generated. Single-token evaluation of that whole suffix would change the prompt path too and cannot guarantee the client's state. Clean exact replay needs a durable evaluation schedule in the journal/protocol, including prompt partitions and reconciliation steps; that protocol extension is outside this revision. No decode-mode exactness or relative timing is claimed. Batch catch-up native evaluation time comes from X-Hybrid-Prefill-Ms and is exported separately from restore/save/HTTP wall time.

Exit codes: 0 accepted same-backend session under the measured criterion, 2 refusal/aborted run, 3 measured criterion failure, 5 heterogeneous measurements without calibrated acceptance.

## Engine trace schema

The output is one atomically replaced, fsynced `trace.json`, `schema="hybrid-growing-session-v3"`. It is updated after every committed call and checker result and on termination. Read `status` before consuming results. `running` and `aborted` are incomplete; `failed` contains measured numerical failures; `heterogeneous` is uncalibrated. Unknown additive fields may be ignored. Missing required fields, unknown versions or incomplete arms must be rejected by a reader. All times are seconds except original `meter.*_ms` fields. Token IDs and conversations are absent from this export; the private SQLite/checker inputs contain them.

| Field | Type and meaning |
|---|---|
| `depth_plan` | D0, total growth including framing and all outputs, final depth, 2048 margin, native/allocated capacity and framing tokens |
| `final_depth` | Actual final depth per arm, required to equal the plan |
| `local_partition` | Alternate decoder batch and ubatch, both required to differ from the client metadata |
| `depth`, `turns`, `seed`, `start` | Integer base depth, turn count, seed; `cold` or `pre-staged` |
| `client`, `node` | Sanitized launcher identities: backend, model SHA, quantization, layout, context and available host/build/binary/toolchain data; no paths/credentials |
| `driver` | Driver and unchanged proxy source SHA-256 |
| `policy` | Full single-run `Policy` configuration, including import intercept and seconds/token |
| `route_preflight`, `route_counts` | Predicted connected decision mix before arms; actual connected policy-local/offload, fallback reasons, local-prompt catch-up turns and background catch-up count |
| `engine` | extend_chunk_tokens, delta_geometry, node/checker 256/256 and two threads, temperature=0, ignore_eos and thinking setting |
| `link` | Operator physical-link label; when node `--bench-link` is enabled, three-probe median RTT and 1MiB application-download rate, bytes and duration; not physical/upload bandwidth |
| `provenance` | Source SHA, depth bins, quantiles and interpolation rule |
| `workload[]` | Preflight sampled sequence, verified against actual hybrid depths: `turn`, `conditioning_depth`, `seed`, `input_tokens`, `output_tokens`, `pause_seconds`, and uncapped `sampled_*` values; input material is always `" x" * input_tokens` |
| `disconnect` | One-based first outage turn and duration in turns; turn zero disables |
| `calls[]` | `arm` (`hybrid`/`local`/`local_partition`), `stage` (`start`/`turn`), `turn`, requested sequence row and actual component measurements below |
| `events[]` | Ordered observations with monotonic `at_seconds` relative to process start; protocol/sync, pre-stage, disconnect/reconnect, window and cleanup events |
| `checkpoints[]` | Turn, depth, k=64, exactness, top-1 count, mean KL(local || hybrid), max absolute logit difference, per-position data, native exit code, status, three snapshot hashes, journal hash, local partition mean/p95 KL, hybrid-to-partition mean KL ratio, KL limit, free-running first divergence (-1 means none), factorial noise cells and report-only per-statistic fidelity_flag |
| `started_utc`, `ended_utc` | Actual UTC observations |

Each call records `depth_before`, `depth_after`, actual `new_tokens`, `generated_tokens`, `decision.inputs` (delta excluding held token, depth, previous offload, current measured link rate, server factor) and `decision.result` (policy suggestion before benchmark-mode overrides and estimated local/offload seconds). Local/local_partition force local regardless of that suggestion; engine readers must pair it with meter.mode and path. `path` records actual execution; `fallback_reason` explains a requested offload that continued locally. The unmodified complete proxy `meter` includes chunks, local/remote work and replay acknowledgements.

`ttft_seconds` is wall time to the first nonempty streamed text/reasoning delta. Components are `local_prefill_seconds` for local calls, or `server_prefill_seconds`, body-only `transfer_seconds`, HTTP/native `import_seconds` and `local_tail_seconds` for offload (including any locally processed remainder after a partial-transfer fallback). `ttft_other_seconds` is the signed remainder, including rendering, queueing, node restore/save/export, synchronization and HTTP overhead. It is not clipped. `remote_prepare_seconds` is a containing interval; do not sum it again. Native prompt time covers prefill/tail, not decoder startup. `reconciliation_seconds` covers finish/journal work after streamed generation. `total_seconds` includes that work and any checkpoint save; `checkpoint_save_seconds` identifies that instrumentation cost. Decode rate uses generated tokens after the first divided by native predicted time; single-token starts have null rate.

`wire_bytes` is received delta-envelope body bytes. Protocol events give JSON request-body bytes and successful response-body bytes per method/path and duration, including catch-up uploads. Request bytes on a failed event describe the attempted body, not confirmed delivery. These are application-body counters, not TCP/TLS/SSH/IP/5G byte counts. `catchup_tokens`, `catchup_seconds`, `catchup_upload_bytes` include successful journal extensions since the previous call, including during its pause. `catchup_node_seconds` sums the node evaluation durations from successful catch-up response headers; it excludes native restore/save, serialization and HTTP overhead. They can overlap local work and must not be summed into TTFT. The first offload after reconnect has `resync` with recovered tokens, protocol time (including prefix reads), catch-up time and upload bytes. Raw protocol/sync events permit reconstruction of the acknowledgement sequence. `journal_hash` is SHA-256 of little-endian int32 evaluated token IDs, identical to the protocol prefix hash.

For turn-only comparisons, sum `stage="turn"` call components by arm and include pauses separately. For cold end-to-end comparisons, include each arm's `stage="start"` row. Pre-staged totals exclude start rows and `pre_stage` events. `window.seconds` is observed wall time, including instrumentation, checker work in the baseline window and pauses; it is not a pure inference total. The decoder runs hybrid then local at its original partition, then restarts for local_partition. The partition arm prepares its own base outside its pre-staged window. One checker context normally coexists with the decoder; component replacement briefly needs a second checker context. Live snapshots are scored sequentially, then each replay cell gets a fresh context; independent greedy checks restore that context state. Arm order and cache warmth are limitations. Report this order as a limitation.

## CPU commands (existing models only)

Use an existing CMake and GCC/Clang toolchain. On agents1, set an on-disk TMPDIR and disable bytecode, use the shared build lock, one lane build at a time, niced with at most four jobs after checking `pgrep -u agent cc1plus` and `uptime`, and keep free disk above 28 GB. Do not download a model. The qualification commands need existing local 0.8B Q4_0 weights; the tests never fetch them.

```sh
mkdir -p .temp/growing/tmp .temp/growing/native
export TMPDIR="$PWD/.temp/growing/tmp"
export PYTHONDONTWRITEBYTECODE=1
pgrep -u agent cc1plus
uptime
# Continue only after other lane builds have stopped.
nice -n 19 cmake -S . -B .temp/growing/build -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
flock ~/Werkstatt/hybrid-models/.build.lock nice -n 19 cmake --build .temp/growing/build -j4 --target llama-server llama-hybrid-prefill llama-hybrid-bench-check
python3 -m unittest discover -s tools/hybrid-bench -p test_growing_session.py -v
nice -n 19 python3 tools/hybrid-bench/test_growing_session_native.py --build-dir .temp/growing/build --model "$MODEL" --artifacts .temp/growing/native
```

The native command runs positive exactness/checksum/journal instrument controls and degenerate-route preflights, then both start modes with 12 turns, seed 281, input cap 48/output cap 128, pauses capped 1.1 s, a real two-turn outage, connected policy-local catch-up/background replay, and 64-token checkpoints at 10 and 12. Generated tokens are 1223/1915 growth tokens (63.9%); requested answer lengths range30..128. Before the arms, all-local and all-remote native controls must each refuse exit 2 with zero calls. It accepts only when the measured session criterion is green for both starts; the original exact-state controls are unchanged. `--controls-only` runs only the explicitly limited native instrument controls. Remove only this task's build tree after process cleanup, retaining logs/JSON and hashes.

## PRO 6000 GPU rehearsal (operator host only)

Use a dedicated PRO 6000 host, enough host RAM for retained node states and transient copies, and a GGUF already qualified at >=150k context. This run is not authorized on agents1. The example reserves 196608 tokens. An actual 27B Q4 model uses about 16 GiB of weights and 6.5 GiB of q8_0 state per context at that capacity; node, decoder and up to two checker contexts can coexist. Budget approximately 80-96 GiB VRAM and >=128 GiB host RAM, then verify the actual launcher admission and peak allocations. The node's RAM snapshots do not bound total RSS. A smaller rehearsal model changes the configuration and must have its own trace.

Build/deploy at the signed experiment branch head, using the exact same GGUF on both roles. The existing launchers build required binaries and record hashes. Hardware qualification requires this replay/component/token gate on the actual target configuration; the r41 CPU admission smoke is green, with free divergence retained as a reported cost. Once qualified, the single-GPU rehearsal must route using the **target M3 Max canonical delta-prefill rates**, not the rehearsal GPU's local marginal rates. Equal same-host node/client rates make every offload strictly more expensive, so ordinary same-host calibration is expected to produce an all-local plan.

Use the measured canonical 512-token delta points: depth 16k/50k/100k/150k/200k/249k ->171.4/96.9/54.9/27.9/21.9/18.8 tok/s. The fixture also contains all 2k/10k points and the source JSONL hash. Kit1.6.0, c21284cd, checkpoints0, M3 Max128 GB; these supersede the earlier segment-prefill curve. Use the measured PR9 CUDA import OLS intercept 1.618657356721653 s and slope 0.0004953084013066794 s/token for the CUDA rehearsal client. These pre-import-speed-v2 measurements are a labelled surrogate, not a fresh calibration or M3 import fit. The only M3 import observation is 42.4 s/39504tokens: it cannot identify both intercept and slope; a measured M3 fit is required before Mac economics can be qualified. Never describe an assumed zero intercept as measured.

Prepare the measured node/link/fixed-transport/geometry fields separately, then generate one policy before the run:

```sh
python3 - <<'POLICY'
import json
from pathlib import Path
source = json.loads(Path('tools/hybrid-bench/fixtures/growing-provenance.json').read_text())
policy = json.loads(Path('.temp/growing-gpu/measured-node-link.json').read_text())
assert {'server_rate', 'link_rate', 'fixed', 'kv_bytes', 'state_bytes'} <= policy.keys()
points = [(r['level'], r['prefill_tok_s']) for r in source['target_client']['delta_turns'] if r['delta'] == 512]
fit = source['import_cost']['cuda_pr9']
policy.update(local_rates=[[0, points[0][1]]] + points, import_fixed=fit['fixed_s'], import_per_token=fit['seconds_token'])
Path('.temp/growing-gpu/policy.json').write_text(json.dumps(policy, indent=2) + '\n')
POLICY
```

Do not use always=true or per-turn overrides. Automatic preflight refuses a degenerate sampled route plan before any arm. Label the target-client policy provenance in --link-label and retain the calibration file. On a fast local GPU, target-M3 decision estimates will differ from measured local timings: this rehearsal qualifies routing, transport, checkpoints, connected catch-up and outage plumbing; it does not measure M3 speedup or economics. The rehearsal must pass the replay/component/token gate; report L-H KL/p95/max at every checkpoint.

```sh
mkdir -p .temp/growing-gpu/tmp
export TMPDIR="$PWD/.temp/growing-gpu/tmp"
export PYTHONDONTWRITEBYTECODE=1
tools/hybrid-bench/deploy-prefill-node.sh --state .temp/growing-gpu/node --model "$MODEL" --sha256 "$MODEL_SHA256" --ctx 196608 --cuda --bench-link --request-timeout 3600 --startup-timeout 3600
tools/hybrid-bench/setup-client.sh --state .temp/growing-gpu/client --model "$MODEL" --sha256 "$MODEL_SHA256" --ctx 196608 --cuda --node-info .temp/growing-gpu/node/node-info.json --token-file .temp/growing-gpu/node/client-tokens.json --request-timeout 3600 --startup-timeout 3600
tools/hybrid-bench/setup-client.sh --state .temp/growing-gpu/client --stop
python3 tools/hybrid-bench/growing_session.py --client-state .temp/growing-gpu/client --output .temp/growing-gpu/run-100k --depth 100000 --turns 40 --seed 1 --start pre-staged --disconnect-turn 20 --disconnect-turns 3 --policy .temp/growing-gpu/policy.json --link-label 'PRO6000 loopback CUDA; target M3 canonical 512 delta rates; PR9 CUDA import OLS; plumbing rehearsal'
tools/hybrid-bench/deploy-prefill-node.sh --state .temp/growing-gpu/node --stop
```

Repeat with `--start cold` and a new output directory to include initial fill. Use the r40 replay/component/token gate and retain all measured L-H costs. Budget 2-5 hours as an unmeasured planning allowance for three 40-turn arms, pauses and four native checkpoint comparisons, not a measured GPU duration. Seed 1 has about 27k output tokens and 13.2 minutes of pauses per arm. At 10-30 decode tokens/s that alone is about 28-58 minutes per arm before new-input work and checkpoint overhead.

## Incident Mac run (Vast CUDA node to M3 Max)

### R43 merged-head admission

Use the signed `m3-growing-session` r43 merge of reviewed `57e3a23229` with main `9348ecd6bcae02b99dca938b84b6ff312306ba5e`, including PRs #14, #15 and #16. The exact tested source merge is `e0573bb2a3d0812d1eaa0b55cfa34a7b5f01fef5`; its qualification successor changes documentation only. The [r43 results and landing note](RESULTS-hybrid-b-r36-261102.md#r43-current-main-merge-before-the-mac-run) identify the exact tested merge head; `.temp/r43/delivery.json` and `.temp/r43/landing-note.md` bind the final local handoff head. The operator supplies that full SHA as `R43_HEAD` on both hosts. Rebuild node, proxy, decoder and checker together; old prefill/client build receipts and policies do not establish qualification of the merged code.

```sh
# On both operator hosts, after importing the signed local branch through the operator's SSH channel:
test "$(git rev-parse HEAD)" = "$R43_HEAD"
git merge-base --is-ancestor 9348ecd6bcae02b99dca938b84b6ff312306ba5e HEAD
git merge-base --is-ancestor e0573bb2a3d0812d1eaa0b55cfa34a7b5f01fef5 HEAD
git log -1 --format='%H %G?'
```

Verify signatures against the configured machine signing identity. A missing Mac trust entry is an operator trust setup issue, not permission to skip source identity. Retain the full source SHA, executable/shared-library hashes and actual launcher build command on each host. Compare node/client GGUF SHA and layout before admission. Authenticated health must advertise `delta_version: 2`; critical envelopes must use `body_format=llama-native-chunks-v3` and physical SWA record v2 when applicable. Keep the r41 rule and all ten controls above. PR #16 is included in source; this runbook continues to use the native CUDA prefill node. A vLLM node is a separate qualified configuration, not a silent backend substitution.

Non-final chunks contain attention ranges and stage host bytes without changing the live decoder, durable prefix or fixed state. Only the final chunk carries recurrent/SWA fixed state and commits the entire transfer. The native checker replays the same transaction through the borrowed-reader importer; wire storage remains owned through apply. Snapshot capture still streams after every successful critical extend, including intermediate chunks, and its native journal must end at that chunk's n1. Recovery downloads a complete snapshot before changing the decoder. Never require intermediate chunks to expose n1 as the live decoder position or reinterpret their `staged` meter status as a completed import. Preserve catch-up, outage and resync evidence.

Run the model-free codec/SWA/hash gates on the new build, then the existing-model CPU multi-chunk regression and the cold/pre-staged harness before the long Mac run. On agents1 the exact fresh commands and exits are in the r43 report; the production CUDA-to-Metal run must repeat the replay/component/token observations on the actual configuration. A CPU pass does not qualify Metal or production SWA. The snapshot transport test above the old 513 MiB cap and missing-final/checksum/interruption controls remain required when changing those paths.

Re-measure M3 import costs with several transfer sizes on this head before fitting the policy. PR #15 removes import copies and accelerates hashing; the old 42.4 s observation, PR9 CUDA fit and preceding GPU rehearsal's reported 4.3 s average are historical inputs, not the merged-head Mac calibration. Record staged bytes/chunk count, final-commit import time, full snapshot capture and catch-up/resync separately from body transfer and TTFT. Include the initial/pre-stage work and all 40 measured turns when reporting route totals.

Tokens move only over the operator's established SSH channel into private mode-0600 files. Never send token values in chat, Matrix, reports, command arguments, environment exports or bundles. Commands below pass credential file paths only. Keep SSH tunnel ownership and cleanup with the operator; retain only sanitized traces outside the private run directory. Use on-disk `TMPDIR`, `PYTHONDONTWRITEBYTECODE=1` and the documented disk reservation; remove owned build trees and consumed large states after preserving identities and results.


The incident lane owns the Vast rental, Mac execution, real link, credentials and model delivery. Provision only there. Use the existing SSH-tunnel pattern in the [operator guide](README.md#2-open-the-ssh-tunnel-from-the-client); record that the traffic traverses the Mac's physical link. Keep a record of host GPU/driver, Mac RAM/chip/OS, GGUF SHA, binary hashes, commit, KV types, policy calibration and link label. Node and client must use the same GGUF; CUDA-to-Metal is heterogeneous solely by backend. Copy `node-info.json` and the mode-0600 token JSON only over the operator's established SSH channel. Do not put bearer tokens in shell arguments or exported traces.

Mac routing uses its measured target-client delta rates from the pinned fixture and a measured M3 import fit. Create `.temp/growing-mac/policy.json` from a measured node/link JSON and a multi-point import fit JSON with `fixed_s` and `seconds_token`; retain commands, raw points and source hashes. The single 42.4 s/39504-token observation is insufficient. Do not copy the CUDA import fit into the Mac policy. Run the same automatic route preflight; an all-local/all-offload plan refuses before any arm. This calibration step is a pending operator measurement, not a measurement made on agents1.

```sh
python3 - <<'MAC_POLICY'
import json
from pathlib import Path
source = json.loads(Path('tools/hybrid-bench/fixtures/growing-provenance.json').read_text())
policy = json.loads(Path('.temp/growing-mac/measured-node-link.json').read_text())
fit = json.loads(Path('.temp/growing-mac/measured-m3-import-fit.json').read_text())
assert {'server_rate', 'link_rate', 'fixed', 'kv_bytes', 'state_bytes'} <= policy.keys()
points = [(r['level'], r['prefill_tok_s']) for r in source['target_client']['delta_turns'] if r['delta'] == 512]
policy.update(local_rates=[[0, points[0][1]]] + points, import_fixed=fit['fixed_s'], import_per_token=fit['seconds_token'])
Path('.temp/growing-mac/policy.json').write_text(json.dumps(policy, indent=2) + '\n')
MAC_POLICY
```

For D0=100000, provision the node with 196608 context as above on Vast. On the Mac, omit `--cuda`; the launcher selects Metal. A 96/128 GB M3 Max needs actual allocation admission for a 27B Q4 model with the live decoder and up to two additional checker contexts. A 64 GB machine needs actual allocation evidence before admission; this driver does not unload the decoder for checks. A deep 262144-context rehearsal with all roles on one 96 GiB GPU may exceed its VRAM once two checker contexts and compute buffers coexist; use actual admission and adequate hardware. Keep node RAM large enough for the full snapshots and transient staging; inspect the launcher-derived ram_budget_mib/node_limits in node-info.json and authenticated health; require admission for the full retained state and transient copies.

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

Disk admission reserves `28e9 + (2 * checkpoint_count + 6) * S(context) + checkpoint_count * 16 * 64 * vocab_size * 4`, with `S ~= recurrent_bytes + (kv_bytes_per_token + 32) * context`. In addition, every successful critical extend retains its delta plus a full live node snapshot for deterministic replay. These grow with the actual number/depth of offloads, not just checkpoint count. Before the run, reserve `sum(S(depth_i) + recurrent_bytes + kv_bytes_per_token * imported_tokens_i)` for those captures, with the possible offload set conservatively including every turn and start. Capture refuses before writing if free space would fall below 28 GB. For illustrative 27B/vocab 248320, four checkpoints alone need about131 GB free at 196608 capacity or163 GB at 262144, before capture reserve and model/build files; round up after adding that reserve. Do not stage snapshots in RAM-backed /tmp or export the whole private directory.

An alternative state contract refreshes the full attention KV after every catch-up, instead of retaining the old client rows. For the27B q8_0 geometry, `16 attention layers * 2 K/V * 4 KV heads * 256 head width * 34/32 bytes = 34816 bytes/token` (34,816 B/token). A full refresh at 100000 tokens costs 3481600000 bytes (3.4816 GB, 3.2425 GiB) per offload; at the seed 1/40-turn deep start 216064 it costs 7522484224 bytes (7.5225 GB rounded; 7.522484224 GB exact, 7.0059 GiB). Use actual depth *34816 for each later offload. These are attention tensor payloads only: add recurrent state (about 157000000 bytes), row/layout metadata, framing/checksums and transport overhead. Incremental append already sends the new-range KV, so the extra attention cost is `(depth - imported_tokens) *34816`. Refresh removes mixed provenance and restores equality with a same-partition local node replay B. This restores equality to B, not L; equality to differently batched L also requires matching evaluation arithmetic. This alternative is described, not implemented.


Use 3-9 hours as an unmeasured 100k three-session Mac planning allowance, plus initial fill and transport variability; reserve 4-12 hours for the deep arm until measured. Both include three copies of the same output/pause workload and sequential 64-token teacher-forced/free-running checks. The earlier 39,504-token fresh call (local 271.5 s, hybrid 201.7 s) does not establish growing-turn costs. Keep checkpoint hashing/loading and node whole-state restore/save observations separate from TTFT.

## Replay retained checkpoints

This runs the four fresh controls without snapshots. It cannot recompute B-H after the live snapshots have been deleted; use a fresh full driver run for that comparison. Outputs are private raw logits, input schedules, result JSON, checker log and command/PID/hash record.

```sh
nice -n 19 python3 tools/hybrid-bench/growing_session.py replay --trace "$TRACE" --input "$CHECK_INPUT" --turn 10 --output .temp/growing/replay10 --model "$MODEL" --checker .temp/growing/build/bin/llama-hybrid-bench-check
```

## Isolate state paths

The full driver always captures the node's own native snapshot immediately after each successful critical extend, through the existing authenticated GET snapshot endpoint. Full snapshots stream to private files in 64 KiB blocks under the disk budget, independently of the unchanged 513 MiB delta-envelope cap. The snapshot journal must end at that extend's n1; a concurrent boundary change refuses and removes the capture files. Capture must return before the client stages each chunk and verifies the unchanged committed prefix, or verifies the complete token/hash boundary after the final commit, before requesting another chunk. The matching delta response is retained too. Files under `OUTPUT/node-states/` contain private tokens/state and must not be exported with a public trace. Protocol events link their paths and SHA-256 hashes. Capture adds HTTP and disk overhead, so its wall time is diagnostic rather than a performance measurement.

Each checkpoint runs A/B1/B2/B and the exact-control plus L-H top-1 gate defined above. Its mandatory `probes` object supplies the following controls, all with the same 64 local teacher IDs and independently restored greedy continuations. Probe scores use KL(reference || probe); references are A, B and H. Raw logits receive the same size/hash, per-position and aggregate validation as the original cells.

| Probe | State path before the unchanged local tail |
|---|---|
| NODE | Load the live node snapshot at the last imported boundary |
| BSTATE | Compute B locally and compare native bytes with that snapshot |
| RESTORE | Add native save/load at every observed node extend boundary |
| NODE_RANGE | Load node state, export its complete range and import it back in one process |
| OWN_NATIVE / OWN_RANGE | Round-trip A's own state immediately before its final token; compare bytes and continuation with A |
| CHAIN | Replay local evaluation between the original critical imports, importing the actual live delta files |
| CHAIN_ATTENTION | Replace only CHAIN's attention KV with node KV at the final import boundary |
| CHAIN_RECURRENT | Replace only CHAIN's recurrent state at that boundary |

Component controls compare attention and recurrent bytes separately before replacement. The native state writer determines the attention boundary; no model-specific byte offset is assumed. Every replacement checks the model, journal and component layout. The tail, logits schedule, KV type and backend remain fixed. A probe must reach the complete journal and evaluate final logits; empty, incomplete, duplicate-name, corrupt or mismatched evidence refuses. `roundtrips.bytes_equal` is a byte comparison, separate from numerical continuation exactness.

On agents1, use the existing CPU build and model only:

```sh
mkdir -p .temp/growing/state-native
taskset -c 0,1 nice -n 19 python3 tools/hybrid-bench/test_growing_session_native.py --build-dir .temp/growing/build --model "$MODEL" --artifacts .temp/growing/state-native --state-probes --start cold
```

Choose two allowed CPUs on another Linux host. All three processes inherit that affinity and use two compute/two batch threads; `engine.cpu_affinity` records the driver's inherited mask. Inspect the live process masks when collecting evidence. State capture and the extra logits increase disk use; the initial reservation includes all 16 observation files per checkpoint and capture checks the 28 GB floor before writing. Keep private evidence or deduplicate identical artifacts, then delete the task-owned build trees.

Both the CUDA rehearsal and the Mac run must repeat NODE-B, CHAIN-H, RESTORE-B, BSTATE-B, NODE_RANGE-B, OWN_NATIVE-A, OWN_RANGE-A and both component replacements, alongside L-H and A-L, at every checkpoint. State capture is mandatory; --state-probes remains an accepted compatibility flag. An exact CPU node/replay match does not predict exact CUDA batching or eliminate mixed KV provenance. If batch and token evaluation produce different KV entries on CUDA, retaining old local entries while replacing node recurrent state can preserve the same mechanism. No CUDA result or positive numerical tolerance follows from CPU diagnostics.

Checkpoint manifests include the full replay schedules and all probe plans. The checker admits at most 64 MiB of JSON with nesting still bounded to 32; the original 8 MiB bound is too small for complete 40-turn plans (the CPU planning fixture serializes about 22-23 MB). This is an input-size reservation, not a long-context inference result. Full native snapshot comparison streams 64 KiB blocks and does not use the 513 MiB delta-file read bound.

### Multi-chunk and snapshot CPU regressions

`python3 tools/hybrid-bench/test_snapshot_stream.py -v` streams a 537,985,076-byte synthetic HTTP snapshot above the old binary cap, checks its SHA-256, and removes it. It uses disk under `TMPDIR` and requires the 28 GB floor plus that state size. Delta bodies retain their existing limit.

`python3 tools/hybrid-bench/test_growing_session_native.py --build-dir BUILD --model LOCAL_GGUF --artifacts EXISTING_PRIVATE_DIR --multi-chunk-only` runs a 256-token pre-staged base with 64-token chunks, saves/restores it, offloads one appended turn, and forces ahead-state snapshot recovery. Every critical request must match the transfer's unchanged committed prefix until the final chunk; only that chunk commits the complete token journal. The regression also runs a complete four-chunk native CHAIN replay and refuses a missing final chunk, decode during staging and checksum corruption. The full harness runs this regression before the unchanged cold/pre-staged r41 smoke. Logs, meters, events and state hashes go under `EXISTING_PRIVATE_DIR/multi-chunk/`; the regression removes its large states after verifying them. No model download is performed.

The authenticated node snapshot route is available with or without `--bench`/`--bench-link`; those flags control timing/probe facilities. Same-model snapshot recovery downloads to a temporary private file, saves the current decoder slot, restores and verifies the exact ahead prompt prefix, then commits the journal. Truncation and disk-budget refusal remove the partial file before decoder changes. A failed restore or journal commit uses the existing rollback. `snapshot_disk_reserve_bytes` defaults to zero for the general proxy; the growing-session driver sets it to `--disk-floor-gb` in bytes. Recovery reserves space for both the incoming snapshot and the older local backup. Full snapshots still consume node RAM and network/disk time; streaming removes the client buffering cap, not those capacity preconditions.

# Preregistered end-to-end log replay: r49b

Read PREREG-E2E.md at signed commit dfffda35763edb17b832f758f586367999ac8978 and SESSION-REPLAY-FORMAT.md. The earlier r49 large-output selector and free-running journals are superseded for this study. Native CPU controls are available; the owner cohort and exact accepted r47 kit binding are still missing. Kernel-shaped throughput remains unmeasured on agents1. Do not spend target GPU time until those prerequisites and review are resolved.

## Source and node

The delivery has source-base.bundle (complete history through r49) and source-update.bundle (this signed successor and preregistration). Verify the signed SHA256SUMS against the independently trusted machine key. Set R49_HEAD to the delivery receipt's exact head, not a moving branch:

```sh
git clone source-base.bundle r49-source
cd r49-source
git fetch ../source-update.bundle m3-growing-r46:refs/remotes/r49-update/m3-growing-r46
git switch --detach "$R49_HEAD"
git verify-commit "$R49_HEAD"
git verify-commit dfffda35763edb17b832f758f586367999ac8978
```

On a fresh RTX PRO 6000 set absolute MODEL, RUN_ROOT (new node directory), BUILD, R49_HEAD and public IPv4 NODE_IP. No model is fetched:

```sh
bash tools/hybrid-bench/packets/r49/stage-node.sh
```

The stage builds exact-ref native prefill under one lock, preserves the 25 GiB node floor, and launches only this run's node/gateway. Native binds loopback; expose authenticated HTTPS 9443 only. Context is 229376, within native 262144, no YaRN. Use the pinned Qwen GGUF SHA256 31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34. Budget 96 GB node RAM minimum, 128 GB preferred; obey actual launcher VRAM/RAM admission. Eight TLS workers serve bounded immutable native envelopes. Per-run token files and the TLS private key stay private; never put tokens in URLs, argv, environment values or logs.

Deliver only node-info.json, private client-tokens.json (0600) and the public tls-cert.pem over authenticated SSH. Deliver the certificate SHA256 independently. Never deliver tls-key.pem. A vLLM seam remains the loopback upstream contract: model/layout identity, dtype, delta-v2 envelope, acknowledged token prefix and compute clocks must all qualify before it can replace llama.cpp. No vLLM arm is silently enabled.

## Freeze the population before timing

Use the full retained owner-log export described by SESSION-REPLAY-FORMAT.md, with complete-population/date-range and extraction/tokenization receipts. The v2 format needs actual request-start and response-end timestamps. Existing aggregate call gaps cannot reconstruct it. Set OWNER_EXPORT, OWNER_EXPORT_SHA256, MODEL_SHA256, VOCAB_SIZE=248320 and a new private SET_ROOT. Commit/circulate the freeze receipt before starting inference; preserve every exclusion. Do not reroll an inventory with a known seed.

```sh
umask 077
mkdir -p "$SET_ROOT"
export PREREG=dfffda35763edb17b832f758f586367999ac8978
python3 tools/hybrid-bench/packets/r49/prepare_session_set.py freeze --input "$OWNER_EXPORT" --sha256 "$OWNER_EXPORT_SHA256" --model-sha256 "$MODEL_SHA256" --vocab-size "$VOCAB_SIZE" --prereg-commit "$PREREG" --output "$SET_ROOT/population.json" > "$SET_ROOT/freeze.log"
export POPULATION_SHA256=$(shasum -a 256 "$SET_ROOT/population.json" | cut -d ' ' -f 1)
python3 tools/hybrid-bench/packets/r49/prepare_session_set.py select --input "$SET_ROOT/population.json" --sha256 "$POPULATION_SHA256" --prereg-commit "$PREREG" --output "$SET_ROOT/sessions.json" > "$SET_ROOT/select.log"
export SESSION_SET="$SET_ROOT/sessions.json"
export SESSION_SET_SHA256=$(shasum -a 256 "$SESSION_SET" | cut -d ' ' -f 1)
```

Five distinct unfiltered main histories are sampled by the committed SHA256 rank. Their natural number of turns runs from the first prior-context boundary in [100k,200k) through the first response reaching 200k. Main portions are never enlarged or filtered. Three disjoint large-output histories form the separately labelled favourable case. A missing count blocks that collection; it never triggers replacement from the main set. Evidence records actual counts, depths, portion bins, and wait distributions. The measured selected owner tool/wait distribution is currently unavailable. Historical Claude call gaps 8.1/60.6 s and all-agent gaps 7.8/34.8 s are not response-end-to-next-request tool measurements.

The same frozen private set/hash and SESSION-REPLAY-FORMAT.md are the handoff to dev-d's API arm. Preserve complete recorded messages or the public base_text_fragments plus ordered input_text/output_text; the next request always uses recorded context. Record API length noncompliance/provider tokenization separately. This packet sends no provider calls or private transcripts to others.

## Mac setup and bounded calibration

M3 Max128 uses its delivered PRI31 fixture. M5 Max64 is conditional on the colleague's availability and actual RAM/wired-memory admission, using its own fixture/fit. Use the incident lane's PRI31/power/idle supervisor; do not substitute an ordinary background launch. Set absolute RUN_ROOT (new external run volume), MODEL, MODEL_SHA256, BUILD, NODE_INFO, NODE_TOKENS, CA_FILE, CA_SHA256 and NODE_IP. The internal HOME filesystem keeps the kit's 40 GiB reserve as well as run-volume snapshot reserves.

```sh
umask 077
mkdir -p "$RUN_ROOT/tmp"
export TMPDIR="$RUN_ROOT/tmp" PYTHONDONTWRITEBYTECODE=1 SSL_CERT_FILE="$CA_FILE"
test "$(shasum -a 256 "$CA_FILE" | cut -d ' ' -f 1)" = "$CA_SHA256"
bash tools/hybrid-bench/setup-client.sh --state "$RUN_ROOT/client" --model "$MODEL" --sha256 "$MODEL_SHA256" --ctx 229376 --ref "$R49_HEAD" --build-dir "$BUILD" --build-lock "$RUN_ROOT/build.lock" --token-file "$NODE_TOKENS" --node-info "$NODE_INFO" --prefill-url "https://$NODE_IP:9443" --request-timeout 3600 --startup-timeout 3600 --offline
python3 tools/hybrid-bench/packets/r46/calibrate_import.py --client-state "$RUN_ROOT/client" --output "$RUN_ROOT/calibration" --r44-packet tools/hybrid-bench/packets/r49/calibration-helpers --depth 200000 --with-bf16 --budget-bytes 2000000000 > "$RUN_ROOT/calibration-01.log" 2>&1
```

This is one bounded six-size pass per dtype, sharing <=2,000,000,000 body/framing bytes (about 1.91 GB planned). Native/protocol integrity work stays measured. A refusal publishes no usable fit and gets no automatic retry. Keep failed observations and physical interface counters; the budget does not bound retransmission. Source/binary/model/dtype identities must match the reviewed head. Mac setup builds/records its own client targets; --skip-build is not used.

For DEVICE=m3-max-128gb choose fixtures/m3-max-qwen-pri31.json; for m5-max-64gb choose its distinct measured fixture. Set LINK_LABEL to the actual interface/path. Make one initial policy-planning probe, then choose the policy from PRI31 rates and the matching bf16 import fit:

```sh
python3 tools/hybrid-bench/packets/r49/probe.py --client-state "$RUN_ROOT/client" --ca-file "$CA_FILE" --link-label "$LINK_LABEL" --series-id planning --output "$RUN_ROOT/planning-probe.json"
python3 tools/hybrid-bench/packets/r49/make_policy.py --client-state "$RUN_ROOT/client" --fixture "$FIXTURE" --import-fit "$RUN_ROOT/calibration/bf16-import-fit.json" --probe "$RUN_ROOT/planning-probe.json" --server-rate 1097 --server-rate-source 'FACTS 1a Qwen Q4 CUDA delta2000 at200k, prior measured surrogate' --output "$RUN_ROOT/policy.json"
python3 tools/hybrid-bench/packets/r49/estimate.py --session-set "$SESSION_SET" --group main --fixture "$FIXTURE" --policy "$RUN_ROOT/policy.live.json" > "$RUN_ROOT/planning-main.json"
bash tools/hybrid-bench/setup-client.sh --state "$RUN_ROOT/client" --stop
```

1097 tok/s is a declared previous-node rate, not a fresh calibration; actual current node service time is recorded. No empty-route refusal or policy adjustment is used to force gain. The cold base also follows the measured policy; it is not forced to offload on a slow link. Refreshing the series link rate changes link accounting, not samples or arm labels.

## Exact r47 guard and arm schedule

Supply KIT_ROOT as the accepted r47 kit source repository and KIT_REF as its full signed commit. The adapter verifies exact source bytes for bench/mac/scripts/kitlib.py, tools/memory_guard.py, tools/slot_handoff.py, scripts/55-run-scaling.sh and VERSION, and calls their existing interfaces. It runs the native decoder under the actual owned memory supervisor, closes startup, takes before/after cooldown VM/priority boundaries, waits 300 s and re-admits the decoder. Its periodic watcher checks the correct child throughout timing; all raw counters, process birth, disk reserve, deltas and final acceptance remain in the run. This is a source-bound adapter, not a hand-authored ready=true receipt. No diagnostic priority override or CPU fallback is allowed. The exact requested r47 binding remains an owner/incident-lane prerequisite; the current candidate kit is not silently substituted.

Write measurement-config.json with exactly these keys/absolute path values: client-state, ca-file, fixture, policy, import-fit, device, link-label, kit-root, kit-ref. import-fit names bf16-import-fit.json. Values contain paths/settings only; no credential values. Every series is a new directory and immutable series ID:

```sh
mkdir -p "$RUN_ROOT/home-main"
python3 tools/hybrid-bench/packets/r49/session_series.py --session-set "$SESSION_SET" --sha256 "$SESSION_SET_SHA256" --prereg-commit "$PREREG" --group main --link-series home --series-id "$HOME_SERIES_ID" --output "$RUN_ROOT/home-main/plan.json" --execute --config "$RUN_ROOT/measurement-config.json" > "$RUN_ROOT/home-main/operator.log" 2>&1
python3 tools/hybrid-bench/packets/r49/analyze_series.py --plan "$RUN_ROOT/home-main/plan.json" --attempts "$RUN_ROOT/home-main/ATTEMPTS.json" > "$RUN_ROOT/home-main/analysis.json"
```

The first block alternates local/hybrid over five distinct sessions; the second alternates local/hybrid-background over those same five, with independently measured local baselines. Twenty arm attempts per main series; local baselines are not ten independent session clusters. Each arm gets its own cooldown and exact guard. A failed attempt stops the series and records all remaining attempts as not attempted; it is never replaced or retried automatically.

The first arm in each block performs the fresh eight-stream node HTTPS probe after cooldown, immediately before cold/timed work. Subsequent arms use that block's immutable probe. Probe again after each block and retain drift. Budget four 384 MiB block probes (1.5 GiB total) per collection/link/device, plus the one initial planning probe and calibration. Record the exact interface and supervisor physical counters. Home Wi-Fi public-mirror observations of 30-48 Mbit/s are context only; actual node throughput determines this series. Fast-link series uses --link-series fast and a new ID/root, reversing the two pair blocks. Use the same sets and source/model identities. The transport/runbook accepts a gigabit-class path without rate caps, but no gigabit throughput is presumed.

Repeat separately with --group favourable and a new home/fast ID, clearly labelled favourable case. Its three sessions give twelve arm attempts and descriptive paired results; it does not satisfy the five-cluster main success test. A later f32 exploratory control uses a fresh single-arm invocation with --state-dtype f32 and its matching import fit; emitted hybrid-f32/hybrid-background-f32 labels keep it out of the main comparisons.

## Public reproducibility session

public/public-source.json.gz contains complete pinned source text and MIT license for https://github.com/ggml-org/llama.cpp/commit/e0f142d5ddc1461ce721644d92af33db23545105. The deterministic builder uses sorted documentation/server/example files in 200-line portions. Its fixed acknowledgements and waits (8 s, every tenth 60 s) are constructed public-workload data, not owner tool measurements or agent performance.

public/cpu-reference-session.json.gz is one fully rendered reference with the existing 0.8B tokenizer: 68 turns, 101711 -> 203809. It is tokenization/format evidence, not a 27B timing result. Retokenize with the actual pinned target decoder before its inference, while the initial setup server is running:

```sh
python3 tools/hybrid-bench/packets/r49/build_public_session.py --source tools/hybrid-bench/packets/r49/public/public-source.json.gz --source-sha256 959fc89a5597c5bfcb2b8dfa7259450a00be06ae1dd994773ffa7a4360f68b50 --client-state "$RUN_ROOT/client" --prereg-commit "$PREREG" --output "$SET_ROOT/public-target.json"
```

Freeze/deliver that artifact's hash before timing. Run it with session_series.py --group public, its own session-set hash and series ID. Same arm, waiting, evidence and cost contracts apply. Public API replay reconstructs the full recorded history from base_text_fragments plus every ordered recorded input/output, never just the current tool portion.

## Timing, quality, cost and storage

Zero-length recorded responses are preserved; the native unused sample is discarded. A checkpoint after a fully evaluated zero-output call appends fixed probe token0 in both quality references, records that extension separately, and leaves the live journal untouched. This zero-output quality binding has CPU fixture coverage, not a new target-model result.

Recorded output IDs are emitted one at a time by the native recorded-token replay mode, including recorded EOS without early stopping. The native sampler still executes; selected IDs are accepted into the sampler/cache. All arms use identical next-turn context. Record system time, not task success.

Anchor the next tool-wait deadline at measured turn completion. Capture the checkpoint only inside that existing wait, then sleep its remainder. No extra full wait follows evidence. An evidence overrun refuses timing qualification and is retained; never subtract it from TTFT or give background extra time. Ordinary dispatch lateness is recorded. Stop background work at the final timed boundary; final snapshot, file hashing, journal/trace persistence and all checkers run afterward. Checkpoints are the four rounded-up turn quartiles, with 64 common teacher positions; require >=63/64, report KL/free divergence, and keep bf16 lossy/exit 5. No equality of free-running answers is a main-workload gate.

Primary warm system time is the direct post-cold-to-final-turn wall interval, including recorded waits and normal dispatch work. Cold load/base, cooldown/probe, checker/evidence and shutdown are separately recorded. Physical raw wall stays available. Keep local_tail_seconds, meter.local_prompt_ms and reconciliation_seconds. Per-arm TLS bytes and node evaluation/compute wall clocks include cold/background costs. The clocks are GPU-service proxies, not kernel-active seconds. Add --gpu-price-per-hour and --traffic-price-per-GB to analysis only with a named price/tariff receipt; rented wall-time billing remains a separate measured bill. Probe/calibration costs are series overhead, not free arm traffic.

Portion bins are 0-511, 512-1999, 2000-9999 and >=10000 incoming tokens. analysis.json retains every attempted/failed row, per-arm/bin time/cost inputs, paired bin gains and the smallest tested bin with a gain. Main inference uses five session clusters, 10000 bootstrap draws, and the preregistered >=10% median reduction with lower 95% bound >0; no-offload sessions are non-gain outcomes. Favourable/public/API results remain separate.

Keep the inherited run-volume guard: 40 GiB + 15*S(context) + 2 GiB, S=state_bytes+(kv_bytes+32)*229376+1MiB, about 165 GB decimal for Qwen. Twelve inherited checkpoint reservations are conservative for the new single-arm executor (four snapshots/arm). Node launch keeps 25 GiB. One decoder/checker context runs at a time. Time/traffic estimates require actual frozen owner sessions; reserve several days per device for twenty long arm attempts plus cooldowns/checkers until that planning exists. Do not reuse the old synthetic forty-turn estimates.

Retain prereg/set/plan hashes, source/binary identities, raw guard/cooldown/probe evidence, ATTEMPTS.json, traces, quality arrays and tariffs. Keep tokens, TLS keys, SQLite journals and full states out of shared archives without authorization/sanitization. Check native sessions are empty and stop only this run's recorded gateway/node PIDs.

## CPU qualification before target execution

On an authorized isolated Linux runner execute shaped_transport.py as in DESIGN.md. Require exit 0 for every 15/20 MB/s x 50/100/150 ms path at >=80% payload throughput. agents1's uid_map refusal remains blocked evidence, not a measured rate. Do not alter its shared qdisc or use an unshaped substitute.

# Staged preregistered end-to-end log replay: r49c

Read PREREG-E2E.md at dfffda35763edb17b832f758f586367999ac8978 and PREREG-E2E-AMENDMENT.md at 7ab1deddcac1ebe77c13d2919092db8ed70be014 (SHA256 6475c8549e9a6dc0e009ef7b12a5bfd0867b6da93a6f1ce4a6f1abafc816a602). The original remains unchanged. Every artifact binds both registrations. SESSION-REPLAY-FORMAT.md defines the native/API replay. No target inference has run under this protocol. Mandatory public rehearsal, owner v2 export and accepted r47 kit ref remain prerequisites. See BUDGET.md/JSON for reproducible FACTS-rate planning.

## Verify source and node

Verify the signed delivery SHA256SUMS using the independently trusted machine key. Obtain R49_HEAD from DELIVERY.json. The two earlier bundles supply history; the third supplies this exact successor. Never use a moving branch:

```sh
git clone source-base.bundle r49-source
cd r49-source
git fetch ../source-update.bundle m3-growing-r46:refs/remotes/r49b/m3-growing-r46
git fetch ../source-r49c.bundle m3-growing-r46:refs/remotes/r49c/m3-growing-r46
git switch --detach "$R49_HEAD"
git verify-commit "$R49_HEAD"
git verify-commit dfffda35763edb17b832f758f586367999ac8978
git verify-commit 7ab1deddcac1ebe77c13d2919092db8ed70be014
export PREREG=dfffda35763edb17b832f758f586367999ac8978
export AMENDMENT=7ab1deddcac1ebe77c13d2919092db8ed70be014
```

Fresh rented RTX PRO 6000: set absolute MODEL (already staged, no model download), RUN_ROOT (new private node directory), BUILD, R49_HEAD and public IPv4 NODE_IP:

```sh
bash tools/hybrid-bench/packets/r49/stage-node.sh
```

It builds exact-ref prefill under one lock and preserves the 25 GiB node floor. Context 229376, native limit262144, no YaRN. Pinned Qwen GGUF SHA256 31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34. Node RAM96 GB minimum/128 GB preferred; obey actual VRAM/RAM admission. Native binds loopback; authenticated TLS gateway9443 serves eight streams. Deliver only node-info.json, private client-tokens.json0600 and public tls-cert.pem over authenticated SSH. Deliver the certificate hash independently. Never copy tls-key.pem or expose token values in argv, URL, logs or environment. Certificate validity is three days; stage/qualify/run within validity. For longer staged campaigns renew the certificate explicitly and re-probe/requalify the new identity; do not disable TLS verification.

The vLLM seam remains the loopback upstream contract (identity/layout, dtype, delta envelope, committed journal ACK and compute clocks); no vLLM arm is silently enabled.

## Choose transport qualification before inference

The owner selects QUALIFICATION=tc or QUALIFICATION=real-path explicitly before the public run. The immutable plan locks it. No fallback after observing throughput. Both paths remain prepared; neither was qualified on agents1.

(a) Recommended unchanged six-grid test: use a full Linux VM with root/CAP_NET_ADMIN/network namespaces, Python3.10+, OpenSSL, git and iproute2 (tc/ip/unshare), 4vCPU/8GiB RAM/10GiB disk minimum. No GPU/models. Containers lacking capabilities cannot run it. Clone/verify the exact same R49_HEAD above, then as root run:

```sh
mkdir -p .temp/r49-tc/tmp
export TMPDIR="$PWD/.temp/r49-tc/tmp" PYTHONDONTWRITEBYTECODE=1
python3 tools/hybrid-bench/packets/r49/shaped_transport.py --output "$PWD/.temp/r49-tc/results" > .temp/r49-tc/runner-01.log 2>&1
```

Require exit0 and six RESULTS.json rows, each >=80% at15/20MB/s x50/100/150ms. Keep all raw qdisc/probe output, log, RESULTS.json hash and exact source head. It refuses host-network shaping and unavailable capabilities. Deliver RESULTS.json to the Mac as TC_RESULTS; measurement-config qualification is {"tc-results":"/absolute/RESULTS.json"}. Allow about one hour including provisioning. This changes no transport protocol requirement.

(b) Independent real-path reference: node must have nginx installed by its operator. Use a separate unprivileged9444 HTTPS static server, independent of the Python gateway. On the node (RUN_ROOT is its node root):

```sh
python3 tools/hybrid-bench/packets/r49/stage_reference.py --root "$RUN_ROOT/reference" --token-file "$RUN_ROOT/node/client-tokens.json" --cert "$RUN_ROOT/tls-cert.pem" --key "$RUN_ROOT/tls-key.pem" --port 9444
```

This adds384MiB on the node while retaining25GiB free. It records its owned PID; stop that PID after the campaign. Deliver ONLY reference/auth.hdr by authenticated SSH to a private0600 Mac path REFERENCE_HEADER; never deliver nginx.conf or any TLS private key. The header is read from file, never passed as a value in argv. Configure qualification as {"reference-url":"https://NODE_IP:9444/payload.bin","reference-header":"/private/auth.hdr","interface":"en7"}, substituting actual values. Both servers verify the run token and TLS. curl must support --header @file, --cacert, --interface and byte ranges. The OS route to the node must use that same interface. No redirects or certificate bypass.

The series runner executes T/R/T before and after EACH comparison block. T aggregates three128MiB transfers (384MiB/trial), R uses eight parallel curl ranges of the independent384MiB static file; actual trial wall time includes coordination. Both T rates must reach80% of adjacent R, pre and post. Preserve raw outputs, route/interface identity and physical netstat counters per trial, alongside r47 supervisor boundary counters. Failure disqualifies the series, no reroll. Fast/iPhone and gigabit paths must qualify on their own path; a home pass cannot qualify them. This option gives up the six absolute rate/RTT points and known capacity denominator. Qualification alone uses2.25GiB per block; ordinary pre/post policy probes add0.75GiB. Never run these on a metered path without accounting for that budget.

For a standalone qualification, set BLOCK=C, SIDE=pre, LINK_SERIES=fast and a new QUAL_OUT, then use one command with the selected option's arguments:

```sh
python3 tools/hybrid-bench/packets/r49/qualify_transport.py --prereg-commit "$PREREG" --amendment-commit "$AMENDMENT" --method tc --tc-results "$TC_RESULTS" --client-state "$RUN_ROOT/client" --ca-file "$CA_FILE" --series-id "$SERIES_ID" --block "$BLOCK" --side "$SIDE" --link-series "$LINK_SERIES" --link-label "$LINK_LABEL" --output "$QUAL_OUT"
python3 tools/hybrid-bench/packets/r49/qualify_transport.py --prereg-commit "$PREREG" --amendment-commit "$AMENDMENT" --method real-path --reference-url "https://$NODE_IP:9444/payload.bin" --reference-header "$REFERENCE_HEADER" --interface "$INTERFACE" --client-state "$RUN_ROOT/client" --ca-file "$CA_FILE" --series-id "$SERIES_ID" --block "$BLOCK" --side "$SIDE" --link-series "$LINK_SERIES" --link-label "$LINK_LABEL" --output "$QUAL_OUT"
```

Use separate output paths if performing preparatory diagnostics; these do not replace the runner's block-specific pre/post qualification.

## Mac setup and bounded calibration

M3 Max128 uses fixtures/m3-max-qwen-pri31.json. M5 Max64 is a separate conditional device with its own measured fixture/fit, accepted RAM admission and its own public rehearsal. Set absolute SET_ROOT (new private session directory) and Mac RUN_ROOT (new external volume), MODEL, MODEL_SHA256, BUILD, NODE_INFO, NODE_TOKENS, CA_FILE, CA_SHA256, NODE_IP, FIXTURE, DEVICE and LINK_LABEL. Fast initial series is iPhone USB (owner observed150-223Mbit/s to European hubs); actual node link is probed. Home Wi-Fi30-48Mbit/s is optional later. Use the existing incident-lane PRI31/power/idle launch supervisor.

```sh
umask 077
mkdir -p "$RUN_ROOT/tmp" "$SET_ROOT"
export TMPDIR="$RUN_ROOT/tmp" PYTHONDONTWRITEBYTECODE=1 SSL_CERT_FILE="$CA_FILE"
test "$(shasum -a 256 "$CA_FILE" | cut -d ' ' -f 1)" = "$CA_SHA256"
bash tools/hybrid-bench/setup-client.sh --state "$RUN_ROOT/client" --model "$MODEL" --sha256 "$MODEL_SHA256" --ctx 229376 --ref "$R49_HEAD" --build-dir "$BUILD" --build-lock "$RUN_ROOT/build.lock" --token-file "$NODE_TOKENS" --node-info "$NODE_INFO" --prefill-url "https://$NODE_IP:9443" --request-timeout 3600 --startup-timeout 3600 --offline
python3 tools/hybrid-bench/packets/r46/calibrate_import.py --client-state "$RUN_ROOT/client" --output "$RUN_ROOT/calibration" --r44-packet tools/hybrid-bench/packets/r49/calibration-helpers --depth 200000 --with-bf16 --budget-bytes 2000000000 > "$RUN_ROOT/calibration-01.log" 2>&1
python3 tools/hybrid-bench/packets/r49/probe.py --client-state "$RUN_ROOT/client" --ca-file "$CA_FILE" --link-label "$LINK_LABEL" --series-id planning --output "$RUN_ROOT/planning-probe.json"
python3 tools/hybrid-bench/packets/r49/make_policy.py --client-state "$RUN_ROOT/client" --fixture "$FIXTURE" --import-fit "$RUN_ROOT/calibration/bf16-import-fit.json" --probe "$RUN_ROOT/planning-probe.json" --server-rate 1097 --server-rate-source 'FACTS 1a Qwen Q4 CUDA delta2000 at200k, prior measured surrogate' --output "$RUN_ROOT/policy.json"
```

Calibration is one six-size pass per dtype sharing <=2,000,000,000 body/framing bytes (~1.91GB planned), not a physical retransmission bound. A failure gets no automatic retry. Record physical counters. Server1097tok/s is a previous-node surrogate, not calibration; live service time is retained. Mac setup builds/records client targets under the same lock; no --skip-build workaround.

## Stage 1: mandatory public full rehearsal

The delivered pinned public repository artifact contains complete source/license from llama.cpp e0f142d5ddc1461ce721644d92af33db23545105. The CPU reference (68turns/101711->203809) is format/tokenizer evidence only. Its fixed acknowledgements and8/60s waits are constructed data, never owner measurements. Retokenize with the pinned target while the initial setup decoder runs:

```sh
python3 tools/hybrid-bench/packets/r49/build_public_session.py --source tools/hybrid-bench/packets/r49/public/public-source.json.gz --source-sha256 959fc89a5597c5bfcb2b8dfa7259450a00be06ae1dd994773ffa7a4360f68b50 --client-state "$RUN_ROOT/client" --prereg-commit "$PREREG" --amendment-commit "$AMENDMENT" --output "$SET_ROOT/public-target.json"
bash tools/hybrid-bench/setup-client.sh --state "$RUN_ROOT/client" --stop
export PUBLIC_SET_SHA256=$(shasum -a 256 "$SET_ROOT/public-target.json" | cut -d ' ' -f 1)
```

Deliver its hash before timing. Provide KIT_ROOT and KIT_REF: the accepted r47 kit source and full signed ref. The adapter verifies kitlib.py, memory_guard.py, slot_handoff.py,55-run-scaling.sh, VERSION against that commit; actual owned-child memory/priority admission plus300s cooldown runs before every arm, with raw evidence. No hand-authored ready receipt, priority override or CPU fallback. The exact accepted kit binding remains a prerequisite.

Write measurement-config.json with the original nine keys (client-state, ca-file, fixture, policy, import-fit, device, link-label, kit-root, kit-ref), all absolute paths/settings, PLUS qualification as defined under option(a)/(b). policy points at policy.json (f32 baseline), import-fit at bf16-import-fit.json. Live driver derives bf16 bytes. No credential values in this file.

```sh
mkdir -p "$RUN_ROOT/public-fast"
python3 tools/hybrid-bench/packets/r49/session_series.py --session-set "$SET_ROOT/public-target.json" --sha256 "$PUBLIC_SET_SHA256" --prereg-commit "$PREREG" --amendment-commit "$AMENDMENT" --group public --comparison both --link-series fast --qualification-method "$QUALIFICATION" --device "$DEVICE" --model-sha256 "$MODEL_SHA256" --link-label "$LINK_LABEL" --series-id "$PUBLIC_SERIES_ID" --output "$RUN_ROOT/public-fast/plan.json" --execute --config "$RUN_ROOT/measurement-config.json" > "$RUN_ROOT/public-fast/operator-01.log" 2>&1
python3 tools/hybrid-bench/packets/r49/analyze_series.py --plan "$RUN_ROOT/public-fast/plan.json" --attempts "$RUN_ROOT/public-fast/ATTEMPTS.json" > "$RUN_ROOT/public-fast/analysis.json"
export PUBLIC_RECEIPT="$RUN_ROOT/public-fast/SERIES.json"
export PUBLIC_RECEIPT_SHA256=$(shasum -a 256 "$PUBLIC_RECEIPT" | cut -d ' ' -f 1)
```

All four arms must pass, with both pre/post blocks qualified, identical replay, admitted supervisor, node accounting and every native quality gate. Main admission reopens and verifies all raw files and hashes; a boolean cannot substitute. The receipt is a mechanics qualification, not a positive-gain requirement. Public n=1 and favourable n=3 retain pairs/median/min/max without a claim. Publish the public evidence and measured save/checker times to the orchestrator; reassess the time budget BEFORE main. A failed public run retains its ID/evidence and grants no main admission.

## Freeze, publish, then select (control host with a2a)

After the public rehearsal, transfer the complete owner v2 export/trusted hash to the authorized control host, using SESSION-REPLAY-FORMAT.md. It requires response-end and next-request timestamps, complete-population/date/extraction declaration. Aggregate request gaps cannot reconstruct tool time. No owner v2 export exists in the current packet. The cited8s/60s tool distribution is unverified; historical8.1/60.6s values are call gaps.

Step 1 freezes without selection. Set OWNER_EXPORT, OWNER_EXPORT_SHA256, VOCAB_SIZE=248320 and a unique fixed STUDY_ID:

```sh
python3 tools/hybrid-bench/packets/r49/prepare_session_set.py freeze --input "$OWNER_EXPORT" --sha256 "$OWNER_EXPORT_SHA256" --model-sha256 "$MODEL_SHA256" --vocab-size "$VOCAB_SIZE" --prereg-commit "$PREREG" --amendment-commit "$AMENDMENT" --study-id "$STUDY_ID" --output "$SET_ROOT/population.json" > "$SET_ROOT/freeze-01.log"
export POPULATION_SHA256=$(shasum -a 256 "$SET_ROOT/population.json" | cut -d ' ' -f 1)
```

Step 2 publishes observed/eligible counts, every exclusion/message, inventory/source hashes, dates and registrations to agents1_hybrid_orch/#hybrid/prototype-go. The configured a2a identity is agents1_hybrid_dev-b on agents1; do not copy its credentials to a Mac. Run publication AND selection there. Stop if publication/readback fails:

```sh
python3 tools/hybrid-bench/packets/r49/publish_freeze.py --population "$SET_ROOT/population.json" --sha256 "$POPULATION_SHA256" --prereg-commit "$PREREG" --amendment-commit "$AMENDMENT" --study-id "$STUDY_ID" --output "$SET_ROOT/publication.json" > "$SET_ROOT/publish-01.log"
export FREEZE_EVENT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["event_id"])' "$SET_ROOT/publication.json")
```

Step 3 can run only after that event exists; it fetches the actual external event and verifies exact content/channel/sender/destination before selecting:

```sh
python3 tools/hybrid-bench/packets/r49/prepare_session_set.py select --input "$SET_ROOT/population.json" --sha256 "$POPULATION_SHA256" --prereg-commit "$PREREG" --amendment-commit "$AMENDMENT" --study-id "$STUDY_ID" --published-event "$FREEZE_EVENT" --output "$SET_ROOT/sessions.json" > "$SET_ROOT/select-01.log"
export SESSION_SET="$SET_ROOT/sessions.json"
export SESSION_SET_SHA256=$(shasum -a 256 "$SESSION_SET" | cut -d ' ' -f 1)
```

Never change inventory after seeing the fixed-seed selection. Five unfiltered distinct main histories, three disjoint favourable histories if available; no portion, route or wait filtering of the main. Publish actual count/depth/portion/wait distributions. Preserve full exception messages. Missing request_messages is declared per session as has_request_messages, so API feasibility is visible without changing eligibility. Transfer the frozen private set/hash to the Mac and expose the same authorized set/format to dev-d. No provider calls or private transcripts are sent by this packet.

## Stage 2: main C/A on fast link

Budget about14.1-14.4h public-like short-response work (reserve15-18h before the public pilot), or roughly25-30h with512-token responses. Owner actual counts/waits/portions control the estimate; do not promise the proxy hours. Reprice the frozen set without modifying it:

```sh
python3 tools/hybrid-bench/packets/r49/estimate.py --session-set "$SESSION_SET" --group main --fixture "$FIXTURE" --policy "$RUN_ROOT/policy.live.json" > "$RUN_ROOT/planning-main.json"
mkdir -p "$RUN_ROOT/fast-main-CA"
python3 tools/hybrid-bench/packets/r49/session_series.py --session-set "$SESSION_SET" --sha256 "$SESSION_SET_SHA256" --prereg-commit "$PREREG" --amendment-commit "$AMENDMENT" --group main --comparison C/A --link-series fast --qualification-method "$QUALIFICATION" --device "$DEVICE" --model-sha256 "$MODEL_SHA256" --link-label "$LINK_LABEL" --public-rehearsal "$PUBLIC_RECEIPT" --public-rehearsal-sha256 "$PUBLIC_RECEIPT_SHA256" --series-id "$MAIN_SERIES_ID" --output "$RUN_ROOT/fast-main-CA/plan.json" --execute --config "$RUN_ROOT/measurement-config.json" > "$RUN_ROOT/fast-main-CA/operator-01.log" 2>&1
python3 tools/hybrid-bench/packets/r49/analyze_series.py --plan "$RUN_ROOT/fast-main-CA/plan.json" --attempts "$RUN_ROOT/fast-main-CA/ATTEMPTS.json" > "$RUN_ROOT/fast-main-CA/analysis.json"
```

Ten attempts, A-C-A-C across five sessions, each separately cooled/admitted. Failure or overrun disqualifies the whole series and preserves remaining not-attempted rows; no reroll. A missing/failed post qualification prevents a qualified analysis even if every timed arm completed. The primary claim still requires all gates, median>=10% and bootstrap lower>0; n5 CI=[min,max]. A no-offload/zero-gain result is retained.

## Stage 3: explicit budget decision

Only if time permits, authorize a new complete comparison before starting: main B/A fast, main B/A home and/or C/A home, or separately labelled favourable n3. Use a new root/ID, --comparison B/A or C/A and --link-series home or fast; keep the same selected sessions and fresh A baselines. Change actual LINK_LABEL/config/interface for Wi-Fi and qualify that path, with new before/after probes. Retain the verified fast public rehearsal for the same model/device/source; it does not qualify the home transport, which must pass its own block gates. Do not infer a gigabit pass from any slower path. Home non-gain is expected and explained through bins, never removed. No optional partial block contributes a main claim.

## Timing, quality and resources

Replay emits exactly the recorded token IDs/lengths through the normal single-token decode loop (including EOS); all later context is recorded, never free-running. Temp0, speculation and backend sampling disabled for replay; system time/resource use, not task success. Quality alone records free-running divergence at64 positions as a diagnostic.

Only B/C save slots. Freeze midpoint/end placement from recorded following waits: nearest midpoint with wait>=budget, earlier tie; final-only if none; final always after active_end. Public budget30s. Owner budget=max(30,ceil(2*largest actual public capture seconds)); bound in the plan and trace. Snapshot must fit BOTH budget and the single actual wait. Hash/checker work follows timed session; checker builds fresh local references. Local A has no snapshots/checkers. Quality coverage falls from four depths per arm to two per hybrid, with the same>=63/64 gate. Overrun disqualifies the series with no reroll. Zero-output snapshot probe extension is explicit and leaves the live journal unchanged.

Cold base is separate; weights can be OS-cached, no purge. Warm primary includes all recorded waits and actual dispatch lateness; per-bin gains use active timed turns only. TTFT, local_tail_seconds, meter.local_prompt_ms, reconciliation_seconds, TLS encrypted payload bytes, envelope bytes, throughput, critical/background catch-up and native synchronized service wall seconds remain measured. Native service time is not kernel-active time. Background final join is separate and queued unstarted work is dropped. Physical counters include traffic outside TLS payload and remain separate. Supply actual GPU/hour and traffic/GB to analyze_series.py; rental wall cost needs the rental receipt.

External volume: retain run snapshots/raw logits and public receipt evidence through main admission, provision at least120GiB spare beyond the kit's40GiB filesystem reserve. At229376 a slot is about8.15GB; each hybrid needs at most two saved slots plus rollback/working reserve, A needs none. The driver checks this reduced reserve per arm. RAM admission remains the actual r47 guard, M3 Max128 preferred; do not infer M5 Max64 admission. Runtime paths are private and immutable IDs prevent silent retries. No model download, current GPU price or throughput claim is bundled.

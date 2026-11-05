# R49 proper end-to-end session measurement

Status: CPU implementation prepared; the owner-log export and the kernel-shaped throughput qualification are prerequisites. Do not spend target GPU/Mac time before both are available and this head is reviewed. CPU mechanics on the existing 0.8B model are not a 27B quality or WAN result. The previous 40-turn run remains a measured slowdown; this packet makes no speedup claim.

Read DESIGN.md. The transport is eight independent authenticated HTTPS range connections, with verified certificates and per-run credentials. Only the TLS gateway is public. Native llama.cpp remains loopback-only. Its unchanged delta-v2 envelope carries bf16 recurrent tensors by default; the policy charges the corresponding bytes. f32 is an explicit control profile with distinct hybrid-f32 and hybrid-background-f32 labels.

## Prerequisites

- Exact reviewed source head `R49_HEAD`, matching Qwen GGUF SHA256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`, and no dirty tracked source. Verify the commit and packet SHA256SUMS before use. No model is fetched by these commands.
- Dedicated rented RTX PRO 6000, CUDA toolchain, at least 96 GB host RAM (128 GB preferred), node context 229376, native context 262144, no YaRN. The standard launcher performs actual VRAM/RAM admission; do not bypass a refusal. Keep its 25 GiB node disk floor. The gateway adds at most two maximum-envelope reservations, plus TLS/socket buffers; monitor RSS.
- Mac M3 Max 128 GB, or M5 Max 64 GB with its own delivered PRI31 fixture and import fit. No M3 fixture substitution on M5. Source/model already staged. Use the incident lane's existing PRI31 supervisor, including decoder descendant priority, power, wired memory, pageout/swapout, disk and actual interface counters. The payload itself refuses the wrong SoC, priority or backend. Stop only this run's recorded launcher PIDs.
- Budget free run-volume disk `40 GiB + 15*S(context) + 2 GiB`, where `S=state_bytes+(kv_bytes+32)*229376+1MiB`; for real Qwen this is about 165 GB decimal. This reserves twelve full checkpoint states and three rollback/transient states. The live guard preserves the floor. A single decoder/checker context runs at a time on the Mac; checker work is deferred until every timed arm has stopped its decoder. M5 is conditional on launcher and supervisor RAM admission, not assumed to fit from the model name.
- A trusted owner export and its full delivery hash. Aggregate stats, sampled quantiles, modified sizes, reordered turns or a synthetic replacement are not accepted.

## Fresh source checkout

The delivery directory contains source.bundle and SHA256SUMS. Verify SHA256SUMS and its machine signature against the independently trusted machine signing key, then create the exact reviewed checkout:

```sh
git clone source.bundle r49-source
cd r49-source
git switch --detach "$R49_HEAD"
git verify-commit "$R49_HEAD"
test "$(git rev-parse HEAD)" = "$R49_HEAD"
```

The packet's delivery receipt names R49_HEAD and the source bundle's full-history status. Keep model files and runtime roots outside this checkout. The same authenticated source artifact is used on the node and Mac.

## CPU transport qualification before target time

On an authorized isolated Linux runner with namespace creation and CAP_NET_ADMIN, from the reviewed source:

```sh
mkdir -p .temp/r49-qualification/tmp
export TMPDIR="$PWD/.temp/r49-qualification/tmp" PYTHONDONTWRITEBYTECODE=1
python3 tools/hybrid-bench/packets/r49/shaped_transport.py --output .temp/r49-qualification/shaped > .temp/r49-qualification/shaped.log 2>&1
```

Require exit 0 and all six RESULTS.json rows passing: 15/20 decimal MB/s x 50/100/150 ms RTT, eight connections, median of three 128 MiB transfers, payload throughput >=80% of the configured capacity. Preserve qdisc configuration/counters. The tool refuses the host network namespace and has no unshaped fallback. agents1's attempted invocation exited 1 at uid_map setup; there is no shaped throughput number from agents1.

## Node command list

Set absolute `MODEL`, new `RUN_ROOT`, `BUILD`, trusted `R49_HEAD`, and public IPv4 `NODE_IP`. The IP is the exact certificate SAN and Mac HTTPS host. Run on the rental host, not agents1:

```sh
git verify-commit "$R49_HEAD"
test "$(git rev-parse HEAD)" = "$R49_HEAD"
bash tools/hybrid-bench/packets/r49/stage-node.sh
```

The stage builds and records native prefill at the exact ref under one lock and creates a private per-run node token file. It writes the gateway PID to RUN_ROOT/gateway.pid. It does not claim that a prefill-only build record contains Mac client binaries. The Mac setup below builds its own server/checker/scorer targets without --skip-build. The node serves direct HTTPS on TCP 9443; expose that port only for this run. Do not expose native port 8091.

Deliver RUN_ROOT/tls-cert.pem, RUN_ROOT/node/node-info.json and RUN_ROOT/node/client-tokens.json to the incident lane over its authenticated SSH channel. Never deliver tls-key.pem. Record the public certificate SHA256 out of band; the Mac checks that pin before trusting the certificate. Do not put any token in a URL, environment value, command argument or log.

For a future vLLM node, replace the loopback upstream with a qualified adapter implementing the same identity, health/dtype, acknowledged-prefix, delta-v2 and node-compute header contract. Merely switching URLs does not qualify different weights, layouts or checkpoint quality. No vLLM arm is enabled by this packet.

## Owner export and selection

The private JSON schema is hybrid-owner-session-export-v1, with model_sha256, source_log_sha256 and sessions. Each session has id, started_utc with timezone, base_ids and ordered turns. Each turn has source_turn, input_ids, output_ids and pause_seconds. Token IDs must come from the pinned Qwen tokenizer; input_ids include the actual inter-turn protocol framing. For another provider's original log, the incident lane must render and retokenize its messages with this model before export, retaining the original log hash and transformation receipt.

The selector chooses the earliest chronological contiguous 40 turns starting at 90k-110k and ending at 190k-210k, with at least twelve incoming portions >=2k and at least three >=10k. It never consults policy or runtime speed. If no interval qualifies, stop and report. The original output lengths determine the generation budget; answers remain free-running per arm. Every arm gets identical base/input IDs, output budgets, pauses and seeds. Preserve the different generated IDs and journal hashes; do not replace them with another arm's history. This measures a fixed external call workload, not answer/task equivalence.

```sh
python3 tools/hybrid-bench/packets/r49/prepare_workload.py --owner-export "$OWNER_EXPORT" --sha256 "$OWNER_EXPORT_SHA256" --model-sha256 "$MODEL_SHA256" --vocab-size 248320 --output "$RUN_ROOT/selected-workload.json" > "$RUN_ROOT/selection.json"
```

The exact export and trusted delivery hash are currently missing. Do not fabricate either to get past admission.

## Mac command list

Set absolute paths: R49_HEAD, MODEL, MODEL_SHA256, RUN_ROOT (new external-volume root), BUILD (Mac build), NODE_INFO, NODE_TOKENS (0600 id-to-token JSON), CA_FILE, CA_SHA256, OWNER_EXPORT, OWNER_EXPORT_SHA256. Set NODE_IP, LINK_LABEL to the observed physical connection, and DEVICE to m3-max-128gb or m5-max-64gb. Execute these payloads through the existing PRI31 supervisor; its launch interface remains owned by the incident lane. All payload arguments below are complete.

```sh
umask 077
mkdir -p "$RUN_ROOT/tmp"
export TMPDIR="$RUN_ROOT/tmp" PYTHONDONTWRITEBYTECODE=1 SSL_CERT_FILE="$CA_FILE"
test "$(shasum -a 256 "$CA_FILE" | cut -d ' ' -f 1)" = "$CA_SHA256"
git verify-commit "$R49_HEAD"
bash tools/hybrid-bench/setup-client.sh --state "$RUN_ROOT/client" --model "$MODEL" --sha256 "$MODEL_SHA256" --ctx 229376 --ref "$R49_HEAD" --build-dir "$BUILD" --build-lock "$RUN_ROOT/build.lock" --token-file "$NODE_TOKENS" --node-info "$NODE_INFO" --prefill-url "https://$NODE_IP:9443" --request-timeout 3600 --startup-timeout 3600 --offline
```

The global SSL_CERT_FILE pins this run's certificate for the existing setup/calibration helpers; the measurement transport also loads the exact CA file explicitly. There is no insecure-certificate switch. The default launcher stays f32 for calibration and policy provenance; the r49 driver negotiates bf16 on the live wire.

Calibrate once at 200k, conservatively near the upper session depth, using the existing bounded paired calibration and exact fitter delivered in calibration-helpers. This is one six-size pass per dtype, sharing the existing 2,000,000,000-byte body/framing budget. Initial 200k bases are local; catch-up uploads tokens, not a downloaded base state. Keep import fits, calibration depth and all failed observations. No automatic retry or f32-cost substitution for bf16:

```sh
python3 tools/hybrid-bench/packets/r46/calibrate_import.py --client-state "$RUN_ROOT/client" --output "$RUN_ROOT/calibration" --r44-packet tools/hybrid-bench/packets/r49/calibration-helpers --depth 200000 --with-bf16 --budget-bytes 2000000000 > "$RUN_ROOT/calibration-01.log" 2>&1
export IMPORT_FIT="$RUN_ROOT/calibration/bf16-import-fit.json"
```

The budget does not bound TCP retransmission; retain physical interface counters. Estimated paired envelope bodies remain about 1.91 GB. A refused pass publishes no fit and is not retried automatically. Source/binary identity must match this reviewed head; old fits with changed import identities refuse.

Choose the device fixture, probe the actual direct HTTPS link, and write the policy. The example server rate 1097 tok/s is the measured conservative Q4 delta-2000 point at 200k from FACTS section 1a, not a new node calibration. Replace it only with a separately recorded measured source. The fixed control cost and link rate come from this HTTPS probe; pending catch-up is charged in the live r49 policy.

```sh
case "$DEVICE" in
  m3-max-128gb) FIXTURE=tools/hybrid-bench/packets/r49/fixtures/m3-max-qwen-pri31.json ;;
  m5-max-64gb) FIXTURE=tools/hybrid-bench/packets/r49/fixtures/m5-max-qwen-pri31.json ;;
  *) exit 2 ;;
esac
python3 tools/hybrid-bench/packets/r49/probe.py --client-state "$RUN_ROOT/client" --ca-file "$CA_FILE" --link-label "$LINK_LABEL" --output "$RUN_ROOT/link-probe.json"
python3 tools/hybrid-bench/packets/r49/make_policy.py --client-state "$RUN_ROOT/client" --fixture "$FIXTURE" --import-fit "$IMPORT_FIT" --probe "$RUN_ROOT/link-probe.json" --server-rate 1097 --server-rate-source 'FACTS 1a Qwen Q4 CUDA delta2000 at200k, prior measured surrogate' --output "$RUN_ROOT/policy.json"
python3 tools/hybrid-bench/packets/r49/estimate.py --workload "$RUN_ROOT/selected-workload.json" --fixture "$FIXTURE" --policy "$RUN_ROOT/policy.live.json" > "$RUN_ROOT/planning.json"
bash tools/hybrid-bench/setup-client.sh --state "$RUN_ROOT/client" --stop
python3 tools/hybrid-bench/session_measurement.py --client-state "$RUN_ROOT/client" --owner-export "$OWNER_EXPORT" --owner-export-sha256 "$OWNER_EXPORT_SHA256" --output "$RUN_ROOT/measurement" --ca-file "$CA_FILE" --fixture "$FIXTURE" --policy "$RUN_ROOT/policy.json" --import-fit "$IMPORT_FIT" --device "$DEVICE" --link-label "$LINK_LABEL" > "$RUN_ROOT/measurement-01.log" 2>&1
```

Order is local, hybrid (catch-up at offload), hybrid-background. Each arm loads one decoder and measures its cold base separately, then runs forty turns. The driver repeats the HTTPS probe immediately before inference and refuses a preflight or observed empty hybrid arm. It does not impose the old outage or alter the owner workload to force mixed routes. For a separate f32 control invocation, use a new output root, --state-dtype f32 and the matching import-fit.json; its hybrid labels are distinct.

## Timing, quality and retained outputs

Use trace.json calls[*].timing for direct timed turns, ttft_seconds for first returned token, first_text_seconds separately, and native decode_tokens_per_second. Cold base rows are turn 0; decoder_load_seconds is also separate. A native cold-prefill request can emit one unused sampled ID; it is recorded as cold_unused_sample_tokens and discarded before journal commit because the native slot contains exactly the owner base. It is never inserted into the forty turns.

Snapshots and file hashes run in an explicit post-turn phase. Quality checkers run after all three timed arms, one context at a time. Raw session/run wall includes all pauses/evidence/teardown; timed-turn sums exclude the separately recorded post-turn phase. Do not subtract captures from TTFT. Integrity checksums, transactional rollback saves and durable-journal reconciliation are production work and remain measured. Keep local_tail_seconds, meter.local_prompt_ms and reconciliation_seconds.

Checkpoints 10/20/30/40 compare each saved arm state with a fresh local replay of that arm's actual journal, with the same 64 teacher-forced tokens. Require >=63/64 top-1; report KL and first free-running divergence. bf16 reports lossy exit 5 and exact=false even when the quality gate passes. The production quality rule is separate from the retained r46 ten exact controls. Both raw L/H arrays are hashed and independently recomputed by the Python verifier. Delete only the consumed full snapshots after a checker result is verified; retain raw logits, fixtures and result hashes.

Transport counters are actual encrypted TLS records at the TCP-payload boundary, including handshake/control/range traffic; envelope bytes are reported separately. TCP/IP headers and retransmissions require supervisor interface counters and must not be silently conflated. node_gpu_eval_seconds is native synchronized prefill wall time; node_compute_seconds also includes load/export/save. They are explicit GPU-service wall proxies, not CUDA kernel-active seconds. Background work is attributed to its source committed turn even when it overlaps later local generation. Compare its extra node time and traffic, not just client TTFT.

Budget 384 MiB for each probe (one explicit preflight plus one driver-start probe), up to 2 GB bounded calibration, plus measured arm traffic. The two cold hybrid envelopes are each about 3.56 GB at a 100k base. Forty-turn traffic depends on the selected inputs/routes; planning.json reports it before inference. Twelve checkpoint states stay on the Mac, not the wire, until deferred checking. No 40-turn time/traffic estimate is promoted from the old synthetic schedule. Reserve a day per Mac until planning on the actual owner artifact exists; fresh local checkpoint replays can take several hours beyond the timed session.

After the run, retain trace.json, quality results/raw arrays, selection/planning, probe, calibration/fit identities, build records, supervisor/command/PID/exit evidence and public certificate pin. Exclude token/key files, SQLite journals and full states from any shared evidence archive unless separately authorized and sanitized. Verify node session counters are empty before stopping only the recorded gateway PID and this run's native node via its launcher. Preserve failed runs and real exit codes.

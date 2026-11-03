# RULER: one bf16 state arm paired to committed f32 rows

This runbook is GPU-host work on one RTX PRO 6000. No GPU run was performed on agents1. The accepted baseline is the existing 18d3f114 RULER evaluator and its frozen depth directories. Only `offload_bf16` is executed. Local, f32 offload and local_partition results are reused byte-for-byte in the new four-arm rows. The baseline directory is never edited.

Stop the baseline evaluator before pairing: its depth lock must be free. A complete depth's archive/manifest seal is verified. A partial stopped depth is allowed; the new run freezes exactly its already committed row/spec pairs. A later change in that set refuses resume. No new examples are generated or substituted. Partial subsets can be unbalanced across tasks; publish their counts and do not claim the complete 13-task mixture's inference. Copy the private specs/rows alongside the existing manifest if the shareable tar alone lacks them.

Identity has an explicit treatment boundary. `pairing_manifest` moves only evaluator `source_hashes` and each service's `commit`/`binaries` into `declared_arm.implementation`; both baseline and candidate implementation identities remain available, with the baseline manifest hash bound into the new arm. Those hashes necessarily change when implementing bf16. Every other manifest field compares exactly: tasks/order/count, seed, depth, margin, generator, Python/packages, input lock, data/download provenance, model hashes, backend, context, batch/ubatch, threads and chat-template hash. Missing or extra common keys also refuse. Examples are independently hash-checked against every existing row's prompt/spec. This is not permission to change model, data, service settings, generation protocol or scoring. Historical manifests did not record endpoint URLs/credentials; they bind the listed service settings rather than a network address. Different source/binary treatment identities are declared, never silently discarded.

Use the [original RULER runbook](RULER-RUNBOOK.md) for the locked venv/data and three owned services. Build the candidate in a separate immutable build directory. Stop only these owned services using their launchers, preserve/move the stopped state directories, and relaunch with the original model, context, partition settings and ports. Never overwrite libraries of a live service. The candidate ordinary client is batch/ubatch64 with `--state-dtype bf16`; the partition client remains256/f32; node is256 with q8_0 KV. The original node/client build records remain in the frozen baseline manifest. New service state directories keep the candidate records.

Example launch arguments (set absolute paths as in the original runbook; use new state directories):

```bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/disk/tmp-bf16
mkdir -p "$TMPDIR"
export RUN=/disk/ruler-bf16
mkdir -p "$RUN"
# MODEL, BUILD, DATA, PY and INPUT_SHA refer to the existing locked GPU setup.
tools/hybrid-bench/deploy-prefill-node.sh --offline --skip-build --cuda --build-dir "$BUILD" \
  --state "$RUN/node" --model "$MODEL" --ctx 132096 --ram-budget-mib 16000 --port 8091
tools/hybrid-bench/setup-client.sh --offline --skip-build --cuda --build-dir "$BUILD" \
  --state "$RUN/client" --model "$MODEL" --ctx 132096 --batch 64 --ubatch 64 --state-dtype bf16 \
  --prefill-url http://127.0.0.1:8091 --port 8092 --decode-port 8090 \
  --node-info "$RUN/node/node-info.json" --token-file "$RUN/node/client-tokens.json" --request-timeout 7200
tools/hybrid-bench/setup-client.sh --offline --skip-build --cuda --build-dir "$BUILD" \
  --state "$RUN/partition" --model "$MODEL" --ctx 132096 --batch 256 --ubatch 256 \
  --prefill-url http://127.0.0.1:8091 --port 8094 --decode-port 8093 \
  --node-info "$RUN/node/node-info.json" --token-file "$RUN/node/client-tokens.json" --request-timeout 7200
```

The context and all service settings above are examples: they MUST equal the actual baseline manifest. If it used a different context, RAM cap or partition, repeat those settings. The evaluator fails closed on a common manifest mismatch. The manager only admits `--skip-build` when its exact build record/source contract holds; keep the recorded candidate build intact.

```bash
export BASELINE=/disk/ruler-existing/results
for depth in 8192 32768; do
  nice -n 10 "$PY" tools/hybrid-eval/ruler_eval.py \
    --arm offload_bf16 --paired-depth "$BASELINE/depth-$depth" \
    --client-state "$RUN/client" --partition-client-state "$RUN/partition" --data "$DATA" \
    --input-lock-sha256 "$INPUT_SHA" --output "$RUN/results" --depth "$depth" \
    --samples-per-task 30 --seed 261102 --margin .02 --request-timeout 7200 \
    --saved-session-bytes "$(( (442685280 * (depth + 130) + 8191) / 8192 ))"
  (cd "$RUN/results/depth-$depth" && sha256sum -c complete.tar.gz.sha256)
done
```

Match baseline task ordering, sample count, seed and margin exactly; pass its `--tasks` if non-default. The single-arm run needs only the ordinary client's slot disk admission, although both service identities are checked. Every new result requires nonzero offload, no fallback, the frozen prompt hash, `state_dtype=bf16`, `lossy_import=true`. Native equality gates are not used to label this lossy quality experiment exact. Session recovery records only the new arm and reaps it before resume. Already committed new-arm rows are skipped after checking hashes and frozen baseline results.

Outputs in each new depth: `rows.jsonl`, `per-example.csv`, `discordant-pairs.jsonl`, `analysis.json`, `analysis.md`, manifest, sealed `complete.tar.gz` and sidecar. Analysis counts answer changes against local, fractional score disagreements against local/f32, mean RULER scores, and two-sided exact McNemar against the existing f32 offload arm for answer-change indicators and full-credit accuracy (score==1). Primary results are ALL per depth; apply Holm if running all four depths. Nonsignificance is not equivalence. Inspect budget-hit rates, old/new service implementation hashes and wall times alongside quality. Baseline rows were observed earlier, so temporal/device drift is not controlled by rerunning them.

For 390 examples/depth, measured three-arm rates imply the following one-third planning estimates; actual bf16 import/encoding and the arm imbalance are unmeasured:

| Depth | Measured three-arm seconds/example | Existing three-arm total | One new arm estimate | Reserve |
|---:|---:|---:|---:|---:|
| 8192 | 15.4 | 100.1min | 33.4min | 45-60min |
| 32768 | about59 | 6.39h | 2.13h | 3-4h |

Do one example first and extrapolate its observed time with 25% reserve. 64k/128k budgets are unmeasured; linear-depth projections from32k give4.3h/8.5h for390 examples before reserve, not commitments. Do not rerun local/partition to make a mismatched manifest fit.

Disk uses native f32 slot backups even for bf16 wire. Before each example, the client volume must have at least `2*saved_session_bytes + 1GiB` free. The inherited conservative sizing formula above gives roughly1.97GB at8k,4.63GB at32k,15.25GB at128k; exact integer checks are authoritative. Keep the host's normal free-space floor in addition. Allow2GiB for new rows/archives/logs, plus the retained baseline and locked data; model files/builds are additional. Successful sessions are detached and their own files removed; a failed release leaves the recovery ledger and blocks the next example. Stop only the PIDs recorded by these owned launcher state directories.

## Vast 58864154: stage after a completed depth

Use `stage_ruler_bf16.py` on the RULER node (RTX PRO 6000, Bulgaria). It runs no remote commands from agents1. Keep `/root/proto/src-ruler` at its existing 18d3f114 source and `build-ruler` unchanged. Install the signed candidate branch in a **separate checkout**, for example `/root/proto/src-state-bf16`, using the operator's normal source-transfer procedure. No fetch, checkout, rebuild or edit occurs in the original checkout. Transfer a local git bundle if the branch has not been published. The stage requires a clean candidate checkout and the full reviewed commit SHA; it does not resolve a moving branch name.

An idle scheduling window is required. The script inspects `/proc` and refuses if an evaluator from `src-ruler` is active, even if a different depth has completed. It never signals an evaluator or controller. It also holds shared locks on the selected completed depths. Do not launch the later 64k/128k process concurrently with this operator window. If an existing controller immediately starts the next depth, wait for an actual idle window; do not interrupt its running three-arm process to make room for this stage.

Run with the **existing** RULER venv so Python and every package version match the frozen manifest:

```bash
export PYTHONDONTWRITEBYTECODE=1
export OLD=/root/proto/src-ruler
export RUN="$OLD/.temp/ruler-r35"
export CAND=/root/proto/src-state-bf16
export CANDIDATE_SHA='REPLACE_WITH_REVIEWED_40_HEX_SHA'
export PY="$RUN/venv/bin/python"
# If the existing venv has another path, use that exact environment.
export STAGE="$RUN/bf16-stage-8192-32768"
export BF16_BUILD="$CAND/build-state-bf16"
"$PY" "$CAND/tools/hybrid-eval/stage_ruler_bf16.py" \
  --ref "$CANDIDATE_SHA" --build "$BF16_BUILD" --stage "$STAGE" --depths 8192 32768
# The command above checks completed archives/locks/disk and changes no services.
"$PY" "$CAND/tools/hybrid-eval/stage_ruler_bf16.py" \
  --ref "$CANDIDATE_SHA" --build "$BF16_BUILD" --stage "$STAGE" --depths 8192 32768 --execute
```

For separate idle windows, first use `--depths 8192 --stage "$RUN/bf16-stage-8192"`; after 32k completes use `--depths 32768 --stage "$RUN/bf16-stage-32768"`. The candidate build is reused only after verifying its ref, flags and every binary/library hash. The stage requires baseline `results-r36/depth-<d>/complete.json`, archive/sidecar, private `rows/` and `specs/`. It rejects partial depths. It verifies all archive members, the outer seal and the manifest.

The execute phase stops the ordinary client, partition client and node using the original launcher's recorded-PID controls, preserves their state under `STAGE/original-services`, and builds the candidate with CUDA into `build-state-bf16`. It checks `pgrep`/`uptime` and uses `nice -n 10`, `-j4` and the shared build lock. It launches three candidate services using the original model/context/batch/ports and node RAM cap; only the ordinary client's state dtype becomes bf16. Before any example at either depth, the **real evaluator** runs `--preflight-only` for **every selected depth**: source/service implementation identities are declared as the treatment; every common manifest field, corpus/package identity and frozen example must match. A mismatch exits before inference and triggers restoration. The ordinary evaluator repeats that gate when running each arm.

Each bf16 result is archived under `STAGE/results/depth-<d>` and verified against the outer SHA, sidecar, internal `SHA256SUMS` and archived manifest. Whether the arm succeeds or fails, `finally` stops candidate services and relaunches originals from the original checkout/build. The original directories remain preserved; the original service paths receive fresh services/credentials. Restore checks health via launcher status and requires original commit, binary hashes, model, context, partitions and threads. `restored.json` records success. A restoration failure produces `failed`, never `done`; inspect `restore.log` before starting 64k/128k.

A SIGTERM/Control-C reaches restoration. A SIGKILL, host crash or power loss cannot run `finally`. Recover explicitly using the persisted service plan:

```bash
"$PY" "$CAND/tools/hybrid-eval/stage_ruler_bf16.py" \
  --ref "$CANDIDATE_SHA" --build "$BF16_BUILD" --stage "$STAGE" --execute --restore-only
```

The recovery command also refuses an active original evaluator and never kills by pattern. Keep `service-plan.json`, `services.log`, `build.log`, `preflight-*.log`, `arm-*.log`, `restore.log`, `restored.json` and all original/candidate service directories private. Do not restart later depths until restoration succeeds. Normal exits have a terminal `done ruler-bf16-stage seconds=...` line; any refusal/error has `failed ruler-bf16-stage: ...` and exit 1.

Admission needs at least 28 GB plus 8 GiB free before staging; allow 6 GiB for the candidate build, 2 GiB for logs/archives, and the per-example slot gate (1.97 GB at 8k, 4.63 GB at 32k). Both gates are checked independently. Models, original build and frozen results are additional retained disk. Planning: new arm 33.4 min at 8k and 2.13 h at 32k from the measured three-arm rates. Reserve 45-60 min and 3-4 h, plus 10-25 min for a first CUDA build and 5-15 min for service transitions. Both depths in one window: reserve 4.5-6 h. These are estimates, not bf16 GPU measurements. After all services are restored and no candidate process remains, the operator may remove only `build-state-bf16`; retain its build record and logs first. Never remove or rebuild `build-ruler`.

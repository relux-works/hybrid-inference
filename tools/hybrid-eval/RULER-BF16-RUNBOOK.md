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

Disk uses native f32 slot backups even for bf16 wire. Before each example, the client volume must have at least `2*saved_session_bytes + 1GiB` free. The inherited conservative sizing formula above gives roughly1.90GB at8k,4.63GB at32k,15.25GB at128k; exact integer checks are authoritative. Keep the host's normal free-space floor in addition. Allow2GiB for new rows/archives/logs, plus the retained baseline and locked data; model files/builds are additional. Successful sessions are detached and their own files removed; a failed release leaves the recovery ledger and blocks the next example. Stop only the PIDs recorded by these owned launcher state directories.

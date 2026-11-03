# Paired RULER on one RTX PRO 6000

This is the hybrid-a-r35-261103 review revision of the r34 integration for the private hybrid-inference prototype, based on main fca0fe31d. It requires one homogeneous CUDA node, a CUDA client decoder at batch/ubatch 64, and a second CUDA client decoder at 256. All use the same GGUF, native binaries, chat template, context and threads. No GPU experiment or benchmark-data download was performed on agents1.

## Protocol and scope

The 13 configurations in NVIDIA's `synthetic.yaml` are retained: `niah_single_1/2/3`, `niah_multikey_1/2/3`, `niah_multivalue`, `niah_multiquery`, `vt`, `cwe`, `fwe`, `qa_1` (SQuAD v2 answerable questions), and `qa_2` (HotpotQA distractor validation). The unchanged generators, templates, configurations, metrics, licence and source hashes are vendored under `vendor/ruler/`. Source: https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a, Apache-2.0.

The adapter calls upstream `generate_input_output` with a bounded, deterministic haystack-size search instead of upstream's batch sizing. The initial upper bound is three times depth divided by estimated tokens per size unit, measured on a minimum-sized rendered candidate, following upstream's sizing rule. It never starts at depth documents for QA. Each candidate resets Python/NumPy RNG from seed, task and example index. QA selects distinct questions using a seeded permutation. Configured complexities, few-shot CWE examples, needle sentences, query text and answer prefixes come from upstream. VT builds the upstream 500-token worked example through `sys_vartrack_w_noise_random`, with three-letter variables and initial value 12345, then inserts `randomize_icl`'s worked input plus answer before the main template, exactly as upstream does. VT fitting includes that example. Unlike the old synthetic tool-conversation experiment, each RULER trial is one user prompt and the official assistant answer-prefix continuation. There is no ingest acknowledgement or extra question. The decoder applies its actual Jinja template with thinking disabled. The same complete prompt is frozen and supplied to all three arms; the sampled continuation is scored.

Depth means exactly **8192, 32768, 65536 or 131072 rendered input tokens**, including BOS/chat framing and the answer prefix, measured by that GGUF decoder's `/apply-template` and `/tokenize`. These are input depths, rather than upstream's input-plus-output `max_seq_length`. Generation reserves upstream's task-specific output budgets (128/30/120/50/32) beyond depth. No task text is truncated. Small whitespace padding after the first newline reaches the exact input depth; padding count, input hash, full prompt-token hash and record/document character offsets are retained. This is an explicitly documented length-control adapter, not byte-identical reproduction of upstream's sampled example sequence. Native tests use small local fixtures; production evaluation refuses missing or modified input locks.

The offload arm imports prefill state, then samples locally; it must report nonzero imports, no fallback, no replay and a fresh boundary. Both local arms import zero tokens. Every actual evaluated prompt must have the frozen token hash, and every session is released. Arm order is deterministically shuffled per example. The proxy family probe disables final-message continuation only for its synthetic complete-turn probe, preserving the real request continuation setting. Answer identity compares the full sampled text after outer whitespace only; sampled-token hashes and budget hits are also retained. RULER quality uses upstream's case-insensitive substring scoring: fraction of references found for retrieval/tracking/extraction; any alias found for QA. Extra text and substrings can therefore score; partial list recall is distinct from full-credit binary accuracy. McNemar on accuracy uses score == 1, independently of answer-change McNemar.

## Inputs and downloads

Run the bootstrap **only on the GPU host**. The evaluator itself is offline with respect to corpora: it never fetches missing inputs. Bootstrap saves every downloaded raw response, requested URL, resolved public URL without signed query parameters, byte count and SHA-256 in `downloads/` and `downloads.json`. It rejects a failed request or changed pinned digest; it does not silently omit essays or use alternate data. The SHA-256 of `inputs.lock.json` is a required evaluator argument and must be shared across all depths and restarts.

| Input | Download / expected SHA-256 |
|---|---|
| SQuAD | `https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json`; `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8` |
| HotpotQA | `https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json`; `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8` |
| CWE fallback vocabulary | RULER `english_words.json` LFS media at the pinned generator commit; `affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca` |
| NLTK Punkt tables | `nltk/nltk_data` commit `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`, `packages/tokenizers/punkt_tab.zip`; `e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106` (digest from that repository's index.xml). Only English tables are extracted. |
| Paul Graham haystack | Exactly the URL list vendored from NVIDIA. HTTP blog URLs use HTTPS; GitHub essay URLs use `gkamradt/LLMTest_NeedleInAHaystack` commit `021385d68d3202e37893e9d3cd29011c569abe30`. The NVIDIA extraction policy joins sorted repository text files, then sorted HTML essays converted using pinned html2text/BeautifulSoup. Every raw essay and the resulting JSON receive SHA-256 entries at GPU bootstrap. |

The blog essays are not an immutable published dataset. Their aggregate hash is deliberately **not invented on agents1**: bootstrap freezes the actual bytes, and evaluation requires that frozen lock hash. Preserve and reuse this data directory/lock for independent replication; rerunning bootstrap against the live blog can yield another corpus. A failed bootstrap leaves its partial download manifest for diagnosis and cannot pass `verify`. GPU corpus download/digest verification remains unrun until that bootstrap. Do not copy any corpus or model into the source repository or shareable result archive. Respect the original datasets' licences and essay copyrights.

## Sample size and inference

Predeclare **390 distinct examples at each depth: 30 per each of 13 configurations**. Across four depths this is 1560 pairs of comparisons and 4680 completions. The same task/index seed is used at each depth, so depths are correlated; do not treat them as 1560 independent examples of a single length. The primary estimand is the equally weighted 13-task mixture at each depth. N=30 per task is descriptive, not a well-powered task-specific equivalence study.

At n=390, the 95% Wilson interval has worst-case half-width 4.94 percentage points; 20/390 changes gives [3.34%, 7.79%]; zero gives an upper bound of 0.975%. For two-sided exact McNemar at alpha .05, discordant probabilities offload-only=.15 and partition-only=.05 give power .9940. Assuming partition-only=.05, an excess of about 5.78 percentage points is detectable with 80% power. A 5 point excess has only .6906 power. Use Holm across the four primary depth McNemar tests; the conservative first threshold .0125 detects about 7.12 points at 80% power (.9749 power for .15 vs .05). These are planning assumptions, not experimental observations. `ruler_analysis.exact_power` computes the conditional exact rejection probabilities averaged over binomial discordant counts.

For the requested framing, let O=offload answer differs from local and P=partition answer differs from local. Persist all four cells, O-only/P-only example rows, exact two-sided McNemar p, each changed-answer rate with a 95% Wilson interval, and O-P. The difference interval is a conservative 95% two-sided bound from four one-sided Clopper-Pearson tails at .0125 (Bonferroni); it remains valid at zero discordance. A literal "offload changes answers no more often" claim requires the upper bound <=0. Also predeclare a practical margin of **2 percentage points** and report noninferiority (upper <.02) and two-sided equivalence (interval inside [-.02,.02]) separately. That margin is a planning choice, not the literal zero-margin claim. With zero discordances at n=390 the conservative upper excess is 1.12 points: practical equivalence can pass while the literal claim remains unestablished. A nonsignificant McNemar p never establishes equivalence. Per-depth intervals are marginal; no pooled-depth or universal model-quality conclusion follows.

Practical 2 pp equivalence needs near-zero total discordance under this conservative interval. At n=390, b=c=0, 1, 2, 5 give upper excess bounds 1.12, 1.62, 2.02, 2.89 pp respectively. Thus b=c=2 already fails despite exactly equal change rates. If disagreement persists, report "no significant difference; equivalence at 2 pp not established" when supported by the tests.

Predeclare the 128k budget option before collecting that depth: set `SAMPLES_128K=20` for n=260, leaving the first three depths at 30/task. At partition-only probability .05, detectable excess at 80% power is 7.45 pp (alpha .05) or 9.21 pp (Holm-first .0125), versus 5.78/7.12 pp at n=390. Zero discordance gives an upper excess bound of 1.67 pp, so 2 pp equivalence remains reachable, but b=c=1 already fails. n=195 (15/task) gives 2.22 pp even at zero discordance and makes 2 pp equivalence impossible. Choose 30 or 20 before the depth starts; changing its denominator after partial collection requires a new output directory and a declared protocol change.

## GPU setup and ordered execution

Use Python 3.11+ with binary wheels available for the pinned dependencies. Use the owner's already qualified CUDA build and the same local GGUF as the existing pilot. No model download is part of these commands. The native build record must match the binaries; do not change a stamp to bypass admission. The model file must be visible to the evaluator, because live decoder identity hashes it. This runbook does not start a remote deployment from agents1.

```bash
export PYTHONDONTWRITEBYTECODE=1
export RUN="$PWD/.temp/ruler-r35"
export DATA="$RUN/data"
export MODEL=/absolute/path/to/qualified-model.gguf
export BUILD=/absolute/path/to/qualified-cuda-build
export LOCK=/absolute/path/to/shared/build.lock
export TMPDIR="$RUN/tmp"
mkdir -p "$TMPDIR"
python3 -m venv "$RUN/venv"
"$RUN/venv/bin/python" -m pip install --only-binary=:all: -r tools/hybrid-eval/ruler-requirements.txt
export PY="$RUN/venv/bin/python"
"$PY" tools/hybrid-eval/ruler_data.py bootstrap --gpu-host-downloads --data "$DATA" > "$RUN/input-lock.sha256"
export INPUT_SHA=$(cat "$RUN/input-lock.sha256")
"$PY" tools/hybrid-eval/ruler_data.py verify --data "$DATA"
export NATIVE_REF=$(python3 -c 'import json,os; print(json.load(open(os.environ["BUILD"]+"/hybrid-bench-build.json"))["commit"])')
```

Keep the bootstrap output and input lock. On restart, **reuse** the existing venv, data and `input-lock.sha256`; do not run the new-directory bootstrap again. The evaluator verifies the vendored source, all locked corpus/NLTK files, dependencies and all client identities before measuring a depth.

If bootstrap failed before creating a valid lock, preserve the partial directory with `mv "$DATA" "$RUN/data-failed-$(date -u +%Y%m%dT%H%M%SZ)"`, then rerun bootstrap into the now-absent `$DATA`. This retry is only for failed bootstrap, never for an input corpus already used by an evaluation.

Launch all three services on the same node, sequentially. Use these state directories only for this experiment:

```bash
nice -n 10 tools/hybrid-bench/deploy-prefill-node.sh \
  --offline --skip-build --ref "$NATIVE_REF" --build-dir "$BUILD" --build-lock "$LOCK" \
  --model "$MODEL" --cuda --ctx 135168 --request-timeout 7200 --state "$RUN/node"
nice -n 10 tools/hybrid-bench/setup-client.sh \
  --offline --skip-build --ref "$NATIVE_REF" --build-dir "$BUILD" --build-lock "$LOCK" \
  --model "$MODEL" --cuda --ctx 135168 --request-timeout 7200 --batch 64 --ubatch 64 \
  --node-info "$RUN/node/node-info.json" --token-file "$RUN/node/client-tokens.json" --state "$RUN/client"
nice -n 10 tools/hybrid-bench/setup-client.sh \
  --offline --skip-build --ref "$NATIVE_REF" --build-dir "$BUILD" --build-lock "$LOCK" \
  --model "$MODEL" --cuda --ctx 135168 --request-timeout 7200 --batch 256 --ubatch 256 \
  --decode-port 8093 --port 8094 --node-info "$RUN/node/node-info.json" \
  --token-file "$RUN/node/client-tokens.json" --state "$RUN/partition"
```

Require aggregate VRAM admission for three model/context allocations on the 96 GB card. Inspect service startup commands and model contexts; all requests are sequential. If the actual model cannot fit, stop and choose an explicitly documented smaller model or change the service-lifecycle design before collecting data. CPU fallback or silently changing contexts/partitions does not satisfy this CUDA-to-CUDA protocol. Disk state size also depends on that model's KV/recurrent geometry.

Before each full depth run a separate pilot with one QA and one noise example. At minimum run 8k and 32k before the reservation decision; each pilot has its own output directory and cannot inflate the paper denominator. Repeat at 64k/128k to estimate the remaining budget at the actual depth:

```bash
for depth in 8192 32768; do
  nice -n 10 "$PY" tools/hybrid-eval/ruler_eval.py \
    --client-state "$RUN/client" --partition-client-state "$RUN/partition" --data "$DATA" \
    --input-lock-sha256 "$INPUT_SHA" --output "$RUN/pilot" --depth "$depth" \
    --saved-session-bytes "$(( (442685280 * (depth + 130) + 8191) / 8192 ))" \
    --tasks niah_single_1 qa_1 --samples-per-task 1 --seed 261102 || break
done
```

Then run complete depths in the requested order. A command failure must stop the loop. It is safe to repeat it with the same options:

```bash
export SAMPLES_128K=30 # Set to 20 before starting 128k for the n=260 budget option.
for depth in 8192 32768 65536 131072; do
  samples=30
  if [ "$depth" = 131072 ]; then samples="$SAMPLES_128K"; fi
  nice -n 10 "$PY" tools/hybrid-eval/ruler_eval.py \
    --client-state "$RUN/client" --partition-client-state "$RUN/partition" --data "$DATA" \
    --input-lock-sha256 "$INPUT_SHA" --output "$RUN/results" --depth "$depth" \
    --saved-session-bytes "$(( (442685280 * (depth + 130) + 8191) / 8192 ))" \
    --samples-per-task "$samples" --seed 261102 --margin .02 --request-timeout 7200 || break
  (cd "$RUN/results/depth-$depth" && sha256sum -c complete.tar.gz.sha256) || break
  # Copy each completed depth directory's complete.tar.gz and sidecar to durable storage here.
done
```

A hard stop loses at most the current three-arm example. Before any arm starts, `pending-sessions.json` durably records fresh session IDs for all three arms. Normal completion, errors, Ctrl-C and SIGTERM release them; SIGKILL/power loss defers release to the next evaluator launch. Recovery must succeed before another example starts. The next attempt always uses new UUIDs. Each arm is independent: no saved native file is a base for another arm. Release detaches the idle decoder slot before deleting only that session's parked, resync and rollback files, preventing a later lease from saving it again. Remote-release failures leave the recovery ledger and stop execution; they do not retain a successfully detached local slot.

Fully committed rows are skipped. Complete depths have `complete.json`, `complete.tar.gz`, an external archive hash and per-member `SHA256SUMS`. A depth is marked complete only after all expected pairs and analysis are present. An interrupted depth retains partial `specs/` and `rows/`. Never combine subsets from changed evaluation protocols. Restarts refuse model, template, partition, seed, source, data or margin changes, except the explicit, audited slot-lifetime repair below. A partial live session after power loss belongs to the private service state: stop these owned services, preserve their logs, and restart before resuming. Do not delete other sessions' state.

`--state` must not exist at launch. After a crash or stop, use this sequence only for these three owned state directories. Moving them preserves logs and removes the old paths; all stop commands must succeed before moving anything. Then repeat the three launch commands above with identical build/model/settings (the node launch creates fresh credentials for the new clients). Keep `$RUN/results`, `$RUN/pilot`, `$DATA`, the lock and venv in place. Evaluator identity excludes state paths/credentials, so this service relaunch can resume the same depth.

```bash
tools/hybrid-bench/setup-client.sh --state "$RUN/client" --stop && \
tools/hybrid-bench/setup-client.sh --state "$RUN/partition" --stop && \
tools/hybrid-bench/deploy-prefill-node.sh --state "$RUN/node" --stop && \
  backup="$RUN/services-stopped-$(date -u +%Y%m%dT%H%M%SZ)" && \
  mkdir -m 700 "$backup" && \
  mv "$RUN/client" "$RUN/partition" "$RUN/node" "$backup/"
# Relaunch only after the entire sequence succeeds and the three old state paths are absent.
```

### Recovery of the 2026-11-03 disk-full stop (17 rows)

The operator already stopped all three services cleanly and removed only the leaking `ruler-*.bin` files. Keep the original `RUN` pointing to this incident's existing directory. The stop commands above are the prerequisite and have already been performed for this incident. Move the three stopped state directories, preserving logs and journals, before relaunch:

```bash
backup="$RUN/services-stopped-r36-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -m 700 "$backup" && mv "$RUN/client" "$RUN/partition" "$RUN/node" "$backup/"
```

Leave `results/`, `pilot/`, frozen `specs/`, `data/`, the input lock and venv in place. Install the reviewed R36 Python source, then repeat the three launch commands above with the original native build/model/settings: node first, batch-64 client second, batch-256 partition client third. Use the existing `NATIVE_REF` from the build record; this Python-only fix needs no native rebuild. The launchers refuse an existing `--state` directory, hence the move is required. Do not copy the old slots or SQLite journals into the new services.

Resume the same 8k depth with the same 30/task denominator and seed, adding the one-time migration flag:

```bash
nice -n 10 "$PY" tools/hybrid-eval/ruler_eval.py \
  --client-state "$RUN/client" --partition-client-state "$RUN/partition" --data "$DATA" \
  --input-lock-sha256 "$INPUT_SHA" --output "$RUN/results" --depth 8192 \
  --samples-per-task 30 --seed 261102 --margin .02 --request-timeout 7200 \
  --saved-session-bytes 449710315 --resume-slot-cleanup
```

`--resume-slot-cleanup` accepts only an incomplete depth whose changed Python files have the exact accepted `b63d9019` predecessor hashes in the repair allowlist. All other manifest fields and source hashes must match. It saves `manifest-before-slot-cleanup.json` and `slot-cleanup-migration.json` with both identities and the retained row-file hashes; both audit files enter the completed archive. The 17 existing rows and their checksums/spec hashes are verified and reused byte-for-byte. No inference is repeated for them; the next inference is `niah_single_1` index 17 (example 18). A CPU regression recreates this exact 17-row boundary. Frozen specs are still rendered/tokenized for integrity checks, which is not inference.

After migration, normal restarts need no migration flag. An already completed pre-R36 pilot/archive remains historical evidence; leave it intact and use a fresh pilot output directory for new pilots, rather than migrating completed bundles. A source identity refusal beyond this allowlisted repair must be investigated; do not edit hashes or create a fresh output directory that loses the 17 committed rows.

On an incomplete-depth resume, existing frozen specs are re-rendered/tokenized even for saved rows to verify integrity. At late 128k this can mean hundreds of full prompt renders before new inference; budget restart time separately.

`specs/` contains long private prompts; `rows/` stores one atomic JSON per completed pair. Keep both for reproduction. Shareable archives include only example answers/scores and hashes, traces without credentials, analysis tables, allowlisted model metadata, source identity and input/download locks. They exclude long corpus text, tokens, native state, endpoints, model paths and credentials. Include budget-hit counts when interpreting equal truncated continuations.

For a joint report, concatenate only completed `rows.jsonl` files into a new analysis input and run `ruler_analysis.py` with all four depths. Its JSON reports `mcnemar_holm_four_depths` only when all four primary depths are present. Per-depth/subset reports instead expose `mcnemar_bonferroni_four_depths` (4 x p, capped at 1), not a joint Holm result; task rows remain descriptive.

```bash
cat "$RUN/results"/depth-*/rows.jsonl > "$RUN/paired-all-depths.jsonl"
"$PY" tools/hybrid-eval/ruler_analysis.py "$RUN/paired-all-depths.jsonl" --output "$RUN/analysis" --margin .02
tools/hybrid-bench/setup-client.sh --state "$RUN/client" --stop
tools/hybrid-bench/setup-client.sh --state "$RUN/partition" --stop
tools/hybrid-bench/deploy-prefill-node.sh --state "$RUN/node" --stop
```

## Time and disk budget

These are scenario estimates, not measured PRO 6000 results. A depth executes 1170 completions, at least 1170 x depth input-token work, output generation and remote shadow/state I/O. Separate prompt generation/tokenization from inference: each frozen spec and paired row records `generation_seconds` (including generator initialization on the first example of each task), and each arm records `seconds`. For m pilot examples at depth D, compute mean G=sum(generation_seconds)/m, mean A=sum(all arm seconds)/m, and inference R=3*D/A. Budget H=1.5*n*(G+A)/3600 per depth, or equivalently 1.5*(3*n*D/R+n*G)/3600. Include output decode, prefill and imports in A; external setup, restart verification and archive/copy time remain additional. Measure QA and noise separately; two examples do not bound all 13 tasks. Remeasure at each deeper length; attention and state I/O can grow faster than linearly.

The table is an inference scenario at R=2500..10000 tokens/s plus 50% contingency. Add `1.5*sum(n_depth*G_depth)/3600` hours for measured prompt preparation; the CPU fixture pilot in the revision report is a preparation-only observation, not a GPU throughput measurement. The upper end of the table is not a worst-case bound for 128k.

| Input depth | Cases / completions | Input work | GPU hours with 50% contingency | Private prompt storage estimate |
|---:|---:|---:|---:|---:|
| 8192 | 390 / 1170 | 9.58M tokens | 0.40..1.60 | 0.03 GB |
| 32768 | 390 / 1170 | 38.34M tokens | 1.60..6.39 | 0.10 GB |
| 65536 | 390 / 1170 | 76.68M tokens | 3.19..12.78 | 0.20 GB |
| 131072 | 390 / 1170 | 153.35M tokens | 6.39..25.56 | 0.41 GB |
| Total | 1560 / 4680 | 277.95M tokens | **11.6..46.3 hours** | 0.74 GB |

With n=260 at 128k, that depth is 260 cases / 780 completions, 102.24M input tokens and 4.26..17.04 inference GPU-hours. The four-depth total is 1430 cases / 4290 completions, 226.84M input tokens and **9.45..37.81 inference GPU-hours**, plus measured preparation and setup. This saves one third of 128k inference work (18.4% overall).

Prompt storage assumes roughly 4 UTF-8 bytes/token and allows JSON overhead; QA/UUIDs can differ. Reserve 2 GB for immutable data/raw downloads, Python wheels and analysis, and another 2 GB for private prompts/results. Model/build storage is additional.

The old "4 x a native saved-session size" statement was wrong for `b63d9019`: benchmark release left native slot bindings alive, so later arms parked closed sessions into permanent files. At 390 examples this grows with the number of examples. R36 instead detaches/releases each arm and removes its files before the next arm; ordinary row boundaries retain zero native files. Resync can transiently hold an export and rollback backup, at most one example's native state at a time. No cross-arm saved-session base is required.

Before every uncommitted example the evaluator checks both slot volumes for **free bytes >= 2 x S + margin**, where required `--saved-session-bytes S` is an upper bound for one native saved session at that depth, including generated tokens and serialization overhead. The default `--slot-margin-bytes` is 1073741824 (1 GiB). A refusal reports free/required/S/margin bytes and starts no sessions or inference; previous committed rows stay usable. Disk admission is per volume, even if both paths share a filesystem. The check is an admission snapshot; concurrent unrelated writers can still consume its margin.

For this incident the observed 8k file is 442685280 bytes. The commands above use `ceil(442685280 * (depth+130) / 8192)`, conservatively scaling the whole file, including fixed state, and allowing the 128-token output budget plus two token positions. At 8k this is S=449710315 bytes and the default floor is 1973162454 bytes. This is a model/build-specific planning bound, not a universal session-size estimator: verify it against native geometry/pilot high-water sizes at each deeper depth and increase S if needed. An underestimated S weakens admission. Do not reuse this size for another model/layout. Operational disk-budget options are recorded in the invocation/log; they do not change prompt/scoring identity.

Preserved service backups consume disk until the operator archives/removes them. Moving directories on the same filesystem does not free their bytes. The incident's leaking native files have already been removed; the state move preserves the remaining logs/journals. Monitor disk usage during pilots. The request timeout is per request, not a total time budget. A budget stop after a completed depth preserves its archive; a partial 128k run does not invalidate completed lower depths.


## CPU qualification entry points

All tests avoid external corpus/model downloads and GPUs. Use disk TMPDIR and one niced native process group at a time. Build only if compatible native binaries are unavailable; check `pgrep -u agent cc1plus` and `uptime`, then use `nice -n 10 cmake --build BUILD -j4 --target llama-server llama-hybrid-prefill`. Keep agents1 free disk above 28 GB.

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest discover -s tools/hybrid-eval -p test_eval.py -v
PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest discover -s tools/hybrid-eval -p test_ruler.py -v
nice -n 10 "$PY" tools/hybrid-eval/test_native.py \
  --node-binary /absolute/cpu-build/bin/llama-hybrid-prefill \
  --decode-binary /absolute/cpu-build/bin/llama-server \
  --model "$HOME/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf" \
  --artifacts .temp/ruler-cpu-native --depths 2048 --tasks single_needle --ruler-checks
```

The native extension uses all 13 local-fixture generators with the real GGUF tokenizer at 2k (4k CWE), plus a three-arm RULER needle execution and the existing five-arm synthetic execution. Local abbreviation-free fixtures use untrained Punkt to avoid a corpus/tokenizer-data download on agents1. This verifies determinism, token fitting, scoring and routing; it does not measure long-context quality or verify GPU data availability. Unit tests include wrong-answer rejection, known McNemar cells/intervals, invalid/missing pair rejection, source/data integrity and incomplete-depth refusal. No surface table was supplied in the brief; no exhaustive platform or statistical coverage is claimed.

R35's real-tokenizer CPU preparation pilot (four fixture tasks, 8k/32k) measured mean G=1.034/1.259 seconds including initialization. An illustrative linear deeper-depth extrapolation gives preparation additions of 1.60 hours full or 1.33 hours with n=260 at 128k, including contingency: total **13.2..47.9 hours** or **10.8..39.1 hours**, respectively. These are planning scenarios, not GPU measurements or upper bounds. See RESULTS-hybrid-a-r34-261102.md for per-task observations and limitations; replace both G and inference R with GPU-host measurements before each deeper depth.

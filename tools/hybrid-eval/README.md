# Patchwork cache task evaluation

Python 3.10+ standard library only. This is an offline synthetic quality experiment for the hybrid prototype, not an upstream submission. No datasets, model downloads, packages or native builds are performed by the evaluator or its tests. Reuse the existing [node/client launchers](../hybrid-bench/README.md) and their private state. Run from the repository root.

## Tasks and depths

`hybrid_eval.py` generates five seeded, independently checkable task types:

| Task | Records | Exact answer |
|---|---|---|
| `single_needle` | One key/value fact | Six-digit value |
| `multi_key` | Six distinct keys | Value of the requested key |
| `multi_value` | Four additions to one key | Every value in record order |
| `variable_tracking` | Assignments, copies and overwrites of shuffled names | All names holding the target value, alphabetically |
| `aggregation` | Six additions spread through the material | Sum of all six amounts |

Filler contains only neutral landscape prose, no digits, task keys or variable names. The generator does not read any outside text. Gold answers follow the records, never substring matches against the context. Copies capture the value at assignment time. Seeds and repeat indices select facts; changing position or depth preserves their answers. Use `--positions 0.1 0.5 0.9` to sweep the needle location. Multi-record tasks spread records across the context while preserving their required order. Every record's actual decoder-token offset is saved.

`--depths` specifies **tool-material tokens**, measured with the client's `/tokenize`, `add_special=false`, fitted to `[depth-8, depth]`. It does not mean total chat prompt length. JSONL records requested/material depth, record offsets and every actual prompt depth from the proxy. Reserve at least 2048 additional context tokens for framing, acknowledgements and answers; 4096 is convenient for the default 128k ceiling. Oversized tasks abort explicitly and produce no success claim for that task. Material hashes and generator version bind the inputs without bundling the long prompts.

## Conversation and scoring

Each fresh session starts with a user request, a synthetic assistant archive tool call and the long tool result. Its first completion is grammar-constrained to `OK` so all modes have the same intermediate message. A later user turn asks the question. That answer is greedy (`temperature=0`), unrestrained by grammar, with the same `--seed` and `--max-tokens` (default 48) for every treatment. Thinking is disabled through the existing chat-template option. No answer is supplied in the prompt except the task records and, for tracking, the queried value.

The ingest request uses `X-Hybrid-Bench-Mode: local`, `hybrid` or `offload`. **The answer request always uses `local` in the same explicit session**, keeping the previously imported cache. Thus `offload` here names the tool-ingest treatment, not offloading the later question. The evaluator checks nonzero imported tokens for both hybrid treatments, zero imports on the question, no rollback/replayed prefix, no fallback and an unchanged prior evaluated boundary. It also requires identical actual answer-prompt token hashes between local and each hybrid treatment. Hybrid's policy must select offload for the long material; an all-local fallback aborts with a calibration message. The normal launcher defaults select long ingests and local short turns; operator policy calibration remains separate from this quality tool.

The mandatory `local_partition` control uses a **second local decoder and proxy**, with the node's `-b/-ub` settings. The conversation, model, decoder binaries, backend, context and thread counts match the primary client; only batch/ubatch changes. Both turns stay local. Supply its state with `--partition-client-state`; the evaluator refuses equal partitions, different client identities/settings, a shared decoder or proxy endpoint, and material too short to exercise the batch boundary. Batch/ubatch is bound to launcher metadata and recorded startup commands; the current `/props` endpoint does not expose live batch settings. Audit those commands when supplying custom client state. This control replaces `local_split`; no framing probe contributes to noise counts.

`local_repeat` runs the ordinary local conversation twice on the primary decoder with identical settings, fresh sessions and the same seed. It reports the run-to-run determinism floor separately from partition noise. Every nonreference treatment, including both controls, must have an identical actual answer-prompt token hash; any mismatch aborts, even when its generated answer is identical. No mode is exempt. A different ingest `OK` tokenization therefore refuses as well.

Accuracy compares the entire answer after outer whitespace and whitespace beside commas are removed. Order, case, digits and duplicates remain significant. Partial lists, extra prose, Markdown wrappers and filler-only answers fail. Final-answer agreement uses the same normalization. `first_differing_token` is the zero-based first difference between **actual sampled decoder IDs**, including EOS; `null` means identical, and a strict prefix differs at the shorter length. It is not re-tokenized answer text. Raw IDs remain in memory only; bundles contain their hashes and the comparison index. Budget exhaustion is recorded separately and is not an automatic accuracy failure if the exact answer is present.

The markdown table groups accuracy, answer agreement, token disagreements, budget hits and request time by depth, task and mode. It labels model hashes, quantization and heterogeneous results. Hybrid and offload are explicitly labeled as the same treatment when both offload the whole ingest (all but its final token); their matching outcomes are not independent evidence. Budget-hit comparisons describe only the observed prefix, including when both outputs match.

For each depth/type and hybrid treatment, JSONL `paired` rows and the markdown count the four paired outcomes relative to the same local reference: both agree, only hybrid disagrees, only partition disagrees, both disagree. Separate tables use normalized full-answer disagreement and sampled-token disagreement; accuracy is a separate endpoint. The two-sided exact McNemar p-value is `min(1, 2 * sum(comb(b+c,k), k=0..min(b,c)) / 2**(b+c))`, where `b` and `c` are the two discordant directions. With no discordants, p=1. This is the [exact binomial version of McNemar's test](https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html); no statistical packages are required. Values are unadjusted for multiple comparisons. Failure to reject is not an equivalence bound or proof that hybrid is within noise.

For a paper comparison, plan **300 distinct tasks per depth and type**, using `--repeats 100 --positions 0.1 0.5 0.9`. Repeat/position pairs now receive different task indices, so the three positions do not repeat the same facts; every treatment still shares each exact task. At four depths and five types this is 6000 cases, 30000 conversations and 60000 completion requests. There is no universal sufficient N without an effect size. For planning, if hybrid-only discordance is 0.15 and partition-only discordance is 0.05, conditioning on discordant cases and summing the exact rejection probabilities gives power 0.9736 at N=300, two-sided alpha=0.05 (N=100: 0.5444; N=200: 0.8753). These are assumed rates, not observations; smaller effects or multiplicity correction need more cases. Predeclare primary endpoints and a multiplicity policy before paper claims. Even six discordants all in one direction can give p=0.03125, while five total cases cannot reach p<0.05. Duplicate depth, task or position arguments refuse before state admission, preventing repeated cases from inflating the paired denominator. The default one-repeat sweep is only a pilot.

Execution exit 0 means the requested conversations and comparisons ran; it does not assert high accuracy, calibrated noise equivalence or paper acceptance. Exit 2 is an aborted/refused run, 130 an interruption. Aborted/interrupted runs after metadata admission still write bundles; pre-admission errors only report a sanitized refusal. Compare homogeneous, cross-backend and heterogeneous experiments separately.

## One GPU host: exact commands

Use the owner's already qualified CUDA build and local model. `--skip-build` requires its matching build record and binary hashes; a mismatch is a refusal, not permission to rebuild. Any separately authorized build must follow the existing node guide and common build lock. Do not change the build record to make a refusal pass.

```sh
export MODEL=/absolute/path/to/model.gguf
export BUILD="$PWD/build"
export RUN="$PWD/.temp/hybrid-eval-gpu"
export LOCK=/absolute/path/to/shared/build.lock
mkdir -p "$RUN"
NATIVE_REF=$(python3 -c 'import json,os; print(json.load(open(os.environ["BUILD"]+"/hybrid-bench-build.json"))["commit"])')
nice -n 10 tools/hybrid-bench/deploy-prefill-node.sh \
  --offline --skip-build --ref "$NATIVE_REF" --build-dir "$BUILD" --build-lock "$LOCK" \
  --model "$MODEL" --cuda --ctx 135168 --state "$RUN/node"
nice -n 10 tools/hybrid-bench/setup-client.sh \
  --offline --skip-build --ref "$NATIVE_REF" --build-dir "$BUILD" --build-lock "$LOCK" \
  --model "$MODEL" --cuda --ctx 135168 --request-timeout 7200 --batch 64 --ubatch 64 \
  --node-info "$RUN/node/node-info.json" --token-file "$RUN/node/client-tokens.json" \
  --state "$RUN/client"
nice -n 10 tools/hybrid-bench/setup-client.sh \
  --offline --skip-build --ref "$NATIVE_REF" --build-dir "$BUILD" --build-lock "$LOCK" \
  --model "$MODEL" --cuda --ctx 135168 --request-timeout 7200 \
  --batch 256 --ubatch 256 --decode-port 8093 --port 8094 \
  --node-info "$RUN/node/node-info.json" --token-file "$RUN/node/client-tokens.json" --state "$RUN/partition"
nice -n 10 python3 tools/hybrid-eval/hybrid_eval.py \
  --client-state "$RUN/client" --partition-client-state "$RUN/partition" --output "$RUN" --label cuda-task-quality --seed 17 \
  --depths 8192 32768 65536 131072 --positions 0.1 0.5 0.9 --repeats 1 \
  --request-timeout 7200
```

The default five tasks x three positions x four depths is 60 cases, each with five treatments and ten completion requests. The paper-size command changes `--repeats 1` to `--repeats 100`; measure the pilot first. Start with `--depths 2048 --positions 0.5 --tasks single_needle` to check routing before the long run. Supply another `--label` for a repeat. If all services are already running, only the evaluator command is needed. Ports default to node 8091, decoder 8090 and proxy 8092; choose unused ports using the launcher options when needed. The partition decoder/proxy use ports 8093/8094 above. One GPU must fit three model allocations and contexts together; each launcher's individual admission is not an aggregate reservation.

Stop only these owned services after the experiment:

```sh
tools/hybrid-bench/setup-client.sh --state "$RUN/client" --stop
tools/hybrid-bench/setup-client.sh --state "$RUN/partition" --stop
tools/hybrid-bench/deploy-prefill-node.sh --state "$RUN/node" --stop
```

## Mac client through an SSH tunnel

Run the node command above on the GPU host. On the Mac, copy the small node metadata and private token file using the owner's normal SSH alias; keep the token file private. Open the tunnel in a separate terminal:

```sh
mkdir -p .temp/hybrid-eval-mac
scp gpu-host:/absolute/node-state/node-info.json .temp/hybrid-eval-mac/node-info.json
scp gpu-host:/absolute/node-state/client-tokens.json .temp/hybrid-eval-mac/client-tokens.json
chmod 600 .temp/hybrid-eval-mac/client-tokens.json
ssh -N -L 8091:127.0.0.1:8091 gpu-host
```

In the Mac checkout, reuse the existing Metal build record and local model (no `--cuda`):

```sh
export MODEL=/absolute/path/to/client-model.gguf
export BUILD="$PWD/build"
export RUN="$PWD/.temp/hybrid-eval-mac"
export LOCK=/absolute/path/to/shared/build.lock
NATIVE_REF=$(python3 -c 'import json,os; print(json.load(open(os.environ["BUILD"]+"/hybrid-bench-build.json"))["commit"])')
nice -n 10 tools/hybrid-bench/setup-client.sh \
  --offline --skip-build --ref "$NATIVE_REF" --build-dir "$BUILD" --build-lock "$LOCK" \
  --model "$MODEL" --ctx 135168 --request-timeout 7200 --batch 64 --ubatch 64 --prefill-url http://127.0.0.1:8091 \
  --node-info "$RUN/node-info.json" --token-file "$RUN/client-tokens.json" --state "$RUN/client"
nice -n 10 tools/hybrid-bench/setup-client.sh \
  --offline --skip-build --ref "$NATIVE_REF" --build-dir "$BUILD" --build-lock "$LOCK" \
  --model "$MODEL" --ctx 135168 --request-timeout 7200 \
  --batch 256 --ubatch 256 --decode-port 8093 --port 8094 \
  --node-info "$RUN/node-info.json" --token-file "$RUN/client-tokens.json" --state "$RUN/partition"
nice -n 10 python3 tools/hybrid-eval/hybrid_eval.py \
  --client-state "$RUN/client" --partition-client-state "$RUN/partition" --output "$RUN" --label metal-task-quality --seed 17 \
  --depths 8192 32768 65536 131072 --positions 0.1 0.5 0.9 --request-timeout 7200
```

For a different node quantization, add **`--hetero-quant` to both client setups** with the actual node-info file. Both local decoders still use the same client model. The evaluator command stays unchanged. Setup enforces layout compatibility; the evaluator records both model SHA-256 values, quantization types and the hetero flag in every JSONL row. Live node health binds the node-info quantization to the loaded model SHA/layout; this health API does not itself expose the quantization label. Both decoder health endpoints are checked, then `/props` supplies its actual model path and ftype; the file is hashed and its GGUF quantization is checked against client-info. Hashing a large local file adds startup time. Matching layout does not prove equal source weights or quality. Heterogeneous disagreement includes model/quantization effects and cannot be interpreted as numerical noise alone. All results retain `comparison_calibrated=false`.

## Tools, outputs and tests

| Tool / command | Purpose | Artifacts |
|---|---|---|
| `python3 tools/hybrid-eval/hybrid_eval.py --help` | Evaluator options | Help only |
| Evaluator command above | Synthetic quality run | `hybrid-eval-LABEL-TIME.jsonl`, `.md`, `.tar.gz`, `.tar.gz.sha256` under `--output` |
| `python3 -m unittest discover -s tools/hybrid-eval -p test_eval.py -v` | Seeded oracle, scoring, fitting, routing and archive controls | Redirect output to `.temp/<run>/unit-01.log` |
| Native test below | Real CPU node, two decoders/proxies, all modes plus repeat/partition controls, tokenizer and routing claims | Private native logs, commands, PID records, JSON metadata, result bundle and cleanup record under `--artifacts` |
| Existing `deploy-prefill-node.sh` / `setup-client.sh` | Service lifecycle | Private launcher state under `--state` |

```sh
mkdir -p .temp/hybrid-eval-cpu
nice -n 10 python3 tools/hybrid-eval/test_native.py \
  --node-binary /absolute/existing/build/bin/llama-hybrid-prefill \
  --decode-binary /absolute/existing/build/bin/llama-server \
  --model "$HOME/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf" \
  --artifacts .temp/hybrid-eval-cpu/native --depths 2048
```

`--depths 2048 8192` tests both bounds, but takes longer. Each test states its claim. The native harness uses two CPU threads per native service, `nice -n 10`, a single decode slot and one task at a time. It records and stops its own PIDs in `finally`, including failures. Existing native binaries must already support the current hybrid API. The harness intentionally does not use launchers to rebuild or change build stamps.

The shareable archive has only JSONL, markdown, allowlisted client/node/partition-client info and a per-member `SHA256SUMS`; the sidecar hashes the archive itself. It includes synthetic questions, expected answers and generated answers, but no long tool contents, token-ID arrays, native state, credentials, paths, addresses, configs or databases. Keep runtime directories private. After stopping services, hash native dumps if needed for audit, then remove only this run's private state. The evaluator's DELETE releases remote state; the proxy may retain local parked slot files until that cleanup.

## Runtime planning

GPU/Metal timings must be measured by the owner; no hardware throughput is assumed. At each depth the default 15 cases execute roughly `75 * depth` input tokens across the five treatments, plus fixed chat turns, answer generation, state serialization and shadow synchronization. Use a small pilot to estimate an effective aggregate throughput `R` tokens/s across the whole experiment, then reserve at least `75 * depth / R` seconds plus startup and transfer overhead. The paper-size sweep is 100 times that work. Deep recurrent/attention behavior and native whole-session save/load can make linear extrapolation optimistic.

| Tool depth | Approximate input work per default depth sweep | Illustrative lower budget at measured R=1000 tokens/s |
|---:|---:|---:|
| 8192 | 614400 tokens | 10.2 minutes plus overhead |
| 32768 | 2457600 tokens | 41.0 minutes plus overhead |
| 65536 | 4915200 tokens | 81.9 minutes plus overhead |
| 131072 | 9830400 tokens | 163.8 minutes plus overhead |

These are planning calculations, not observed GPU runtimes. Historically, with the rejected framing control, the five 2k cases at one position took 160.263 seconds of requests across all treatments; one 8k retrieval case took 231.130 seconds. Startup, identity hashing, generator checks and cleanup add time. Request timeouts apply to individual requests, not the entire sweep. Those historical timings do not price the new control. The [CPU evidence report](RESULTS-hybrid-b-r21-261101.md) records failures, limits and the exact GPU invocation.

## Reference-logit study

`hybrid-eval` does not yet accept `--reference-logits`. Its current endpoint samples free-running task answers and scores those answers. Different arms can produce different answers, so reusing each answer as its continuation would compare different token positions. The evaluator also does not export its answer-prefix state as a native delta chain. A valid integration needs a per-item fixed continuation and the exact framed answer-prefix token/state identity shared across arms, plus a mapping from each item to its D reference. Adding a file flag alone would not establish those contracts.

Use the fixed-token scorer in [tools/hybrid-bench/README.md](../hybrid-bench/README.md#reference-model-teacher-forced-scoring) for this prototype's KL(D || arm) study. Task accuracy, local partition controls and free-running answer disagreement remain the existing evaluator measures. This is a declared integration gap, not a substitute for the GPU reference-model study.

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

The mandatory `local_split` noise reference divides the same material at a newline into two tool results with separate `OK` completions, then asks the same question locally. This changes turn splitting **and chat framing**. It is an empirical local-to-local sensitivity control, not a clean measurement of only floating-point partition noise and not a calibrated acceptance threshold. The final prompt hash explicitly records that difference. The current decoder offers no per-request batch/ubatch setting. The normal local/hybrid pair keeps turn boundaries and actual prompt tokens identical.

Accuracy compares the entire answer after outer whitespace and whitespace beside commas are removed. Order, case, digits and duplicates remain significant. Partial lists, extra prose, Markdown wrappers and filler-only answers fail. Final-answer agreement uses the same normalization. `first_differing_token` is the zero-based first difference between **actual sampled decoder IDs**, including EOS; `null` means identical, and a strict prefix differs at the shorter length. It is not re-tokenized answer text. Raw IDs remain in memory only; bundles contain their hashes and the comparison index. Budget exhaustion is recorded separately and is not an automatic accuracy failure if the exact answer is present.

The markdown table groups accuracy, agreement, token disagreements and request time by depth, task and mode. Compare the hybrid rows with the `local_split` row at the same depth/task. Execution exit 0 means the requested conversations and comparisons ran; it does not assert high accuracy, calibrated noise equivalence or paper acceptance. Exit 2 is an aborted/refused run, 130 an interruption. Aborted/interrupted runs after metadata admission still write bundles; pre-admission errors only report a sanitized refusal. Compare homogeneous, cross-backend and heterogeneous experiments separately.

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
  --model "$MODEL" --cuda --ctx 135168 --request-timeout 7200 \
  --node-info "$RUN/node/node-info.json" --token-file "$RUN/node/client-tokens.json" \
  --state "$RUN/client"
nice -n 10 python3 tools/hybrid-eval/hybrid_eval.py \
  --client-state "$RUN/client" --output "$RUN" --label cuda-task-quality --seed 17 \
  --depths 8192 32768 65536 131072 --positions 0.1 0.5 0.9 --repeats 1 \
  --request-timeout 7200
```

The default five tasks x three positions x four depths is 60 cases, each with four treatments and nine completion requests. Start with `--depths 2048 --positions 0.5 --tasks single_needle` to check routing before the long run. Supply another `--label` for a repeat. If both services are already running, only the evaluator command is needed. Ports default to node 8091, decoder 8090 and proxy 8092; choose unused ports using the launcher options when needed. One GPU must fit both model allocations and both contexts under the launcher's conservative budget.

Stop only these owned services after the experiment:

```sh
tools/hybrid-bench/setup-client.sh --state "$RUN/client" --stop
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
  --model "$MODEL" --ctx 135168 --request-timeout 7200 --prefill-url http://127.0.0.1:8091 \
  --node-info "$RUN/node-info.json" --token-file "$RUN/client-tokens.json" --state "$RUN/client"
nice -n 10 python3 tools/hybrid-eval/hybrid_eval.py \
  --client-state "$RUN/client" --output "$RUN" --label metal-task-quality --seed 17 \
  --depths 8192 32768 65536 131072 --positions 0.1 0.5 0.9 --request-timeout 7200
```

For a different node quantization, add **`--hetero-quant` to client setup** with the actual node-info file. The evaluator command stays unchanged. Setup enforces layout compatibility; the evaluator records both model SHA-256 values, quantization types and the hetero flag in every JSONL row. Live node health binds the node-info quantization to the loaded model SHA/layout; this health API does not itself expose the quantization label. Decoder health is checked, then `/props` supplies its actual model path and ftype; the file is hashed and its GGUF quantization is checked against client-info. Hashing a large local file adds startup time. Matching layout does not prove equal source weights or quality. Heterogeneous results always retain `comparison_calibrated=false`.

## Tools, outputs and tests

| Tool / command | Purpose | Artifacts |
|---|---|---|
| `python3 tools/hybrid-eval/hybrid_eval.py --help` | Evaluator options | Help only |
| Evaluator command above | Synthetic quality run | `hybrid-eval-LABEL-TIME.jsonl`, `.md`, `.tar.gz`, `.tar.gz.sha256` under `--output` |
| `python3 -m unittest discover -s tools/hybrid-eval -p test_eval.py -v` | Seeded oracle, scoring, fitting, routing and archive controls | Redirect output to `.temp/<run>/unit-01.log` |
| Native test below | Real CPU node/proxy/decoder, all modes plus noise control, tokenizer and routing claims | Private native logs, commands, PID records, JSON metadata, result bundle and cleanup record under `--artifacts` |
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

The shareable archive has only JSONL, markdown, allowlisted client/node info and a per-member `SHA256SUMS`; the sidecar hashes the archive itself. It includes synthetic questions, expected answers and generated answers, but no long tool contents, token-ID arrays, native state, credentials, paths, addresses, configs or databases. Keep runtime directories private. After stopping services, hash native dumps if needed for audit, then remove only this run's private state. The evaluator's DELETE releases remote state; the proxy may retain local parked slot files until that cleanup.

## Runtime planning

GPU/Metal timings must be measured by the owner; no hardware throughput is assumed. At each depth the default 15 cases execute roughly `60 * depth` input tokens across the four treatments, plus fixed chat turns, answer generation, state serialization and shadow synchronization. Use a small pilot to estimate an effective aggregate throughput `R` tokens/s across the whole experiment, then reserve at least `60 * depth / R` seconds plus startup and transfer overhead. Deep recurrent/attention behavior and native whole-session save/load can make linear extrapolation optimistic.

| Tool depth | Approximate input work per default depth sweep | Illustrative lower budget at measured R=1000 tokens/s |
|---:|---:|---:|
| 8192 | 491520 tokens | 8.2 minutes plus overhead |
| 32768 | 1966080 tokens | 32.8 minutes plus overhead |
| 65536 | 3932160 tokens | 65.5 minutes plus overhead |
| 131072 | 7864320 tokens | 131.1 minutes plus overhead |

These are planning calculations, not observed GPU runtimes. On this shared CPU host the five 2k cases at one position took 160.263 seconds of requests across all treatments; one 8k retrieval case took 231.130 seconds. Startup, identity hashing, generator checks and cleanup add time. Request timeouts apply to individual requests, not the entire sweep. The [CPU evidence report](RESULTS-hybrid-b-r21-261101.md) records failures, limits and the exact GPU invocation.

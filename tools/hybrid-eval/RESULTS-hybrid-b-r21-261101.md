# Task-level patchwork cache quality: hybrid-b-r21-261101

Ready for review. The evaluator and CPU execution evidence are local; no push, GPU evaluation, external review or paper-level quality acceptance is claimed. All changes are in `tools/hybrid-eval/`.

## Source and implementation

Fetched `/home/agent/Werkstatt/hybrid-prototype.git main` and created `m2-task-eval` from exactly `0124e89a933872b54bd445921b9ac2e1c9c43a09`, which was the fetched main. The existing checkout was clean. No task-board is configured: the initial goal read failed because `.task-board` does not exist. Tracking, readiness logs and self-review are in `.temp/tasks.md` and `.temp/hybrid-b-r21-261101/`.

`hybrid_eval.py` adds five seeded task types with finite record-derived gold answers, decoder-measured tool depth and actual record offsets. It uses existing benchmark HTTP, streaming, metering and session-release helpers. Each normal treatment ingests one long tool result, emits the fixed `OK` acknowledgement, then answers a later question locally in the same session. Hybrid/offload require actual imports. Paired actual answer-prompt token hashes must match local; every question must preserve the evaluated boundary and import zero new remote tokens. The local split control ingests identical material in two tool turns and records its framing difference. Scores require the entire answer, including every ordered multi-value item. First divergence uses actual sampled IDs including EOS, with zero-based indices and null for identity.

Bundles contain synthetic questions, expected/generated answers, token hashes, routing metrics and allowlisted metadata. They exclude material text, token-ID arrays, native state, credentials, configs, databases, paths and endpoints. Both model SHA-256 values, quantization labels and the hetero flag are recorded in every row. Node-info quantization is bound to live node-health SHA/layout; decoder health and props are checked and its actual model file is hashed and parsed for quantization. No claim is made that either health API supplies a quantization label it does not expose.

The evaluated evaluator source SHA-256 is `e0e314b1792fa8b34c2acd01e224880736315c8d425cf60e8f4f573e5fcaa4ff`. Self-review applied the repository code-review skill: scope, security, design, tests and general conventions. No surface table was supplied; that is a brief gap. No native/proxy/API changes or existing reviewer-test edits were made. No additional external landing gate was assigned. The owner explicitly requests the Codex co-author trailer for this private local task; there is no upstream submission.

## CPU environment and exact commands

Qwen3.5 0.8B Q4_0, 507154688 bytes, SHA-256 `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`. Both sides use this exact file. Native KV is q8_0, flash attention enabled, batch/ubatch 256, two CPU threads, one decoder slot and disabled context shift. The harness/evaluator runs at nice 10; its additional `nice -n 10` on native/proxy children yields effective nice 19, verified live. Each fixture starts isolated node/decoder/proxy processes and stops those recorded PIDs in `finally`. No model downloads, package installs or builds ran.

Existing native binaries were read-only under `/home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/`. Observed decoder `--version` reports `0.8.0-dev`, build 12081, commit `0124e89a93`, GNU 15.2.0, Linux x86_64. The installed build stamp names an older build and has hash mismatches against current executables/libraries. It is preserved as historical data, not treated as exact provenance or modified to permit `--skip-build`. The runtime source directories in that workbench have no diff against the pinned base, but that alone is not proof of how an existing binary was built. Actual observed binary/library hashes are persisted in `native-build-identity.json`; the fixture also hashes the two executables at startup. This limits build attestation, not the observed real-node execution claims.

| Installed artifact | Observed SHA-256 |
|---|---|
| `llama-server` | `04aeb70d4aaf510c20d9184583fab2dc0c87336543f3656fca4d6472313e79c4` |
| `llama-hybrid-prefill` | `0b7541a9378243efeebd18353a1cb6e20a27ce9ca50ddb84d4c498f3027e8764` |
| `libllama.so` | `84128f68790f1cf122c786a1a55ca140d9ac1a97415f3d3d523b437e43d2b63e` |
| `libllama-server-impl.so` | `bb4bde137f86ca2768a07726f64c43fa39166f9807e13fc98001909121edd208` |

With `RUN=.temp/hybrid-b-r21-261101`, the final 2k command was:

```sh
nice -n 10 python3 tools/hybrid-eval/test_native.py \
  --node-binary /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-hybrid-prefill \
  --decode-binary /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-server \
  --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf \
  --artifacts .temp/hybrid-b-r21-261101/native-03 --depths 2048
```

The bounded 8k command changes the last line to:

```sh
  --artifacts .temp/hybrid-b-r21-261101/native-04 --depths 8192 --tasks single_needle
```

The fixtures run the real evaluator CLI, all three normal modes plus `local_split`, seed 17, position 0.1 and answer budget 48. Their private `commands.json`, `evaluation-exit.json`, PID files, logs, `generator-checks.json`, `result.json` and `cleanup.json` preserve actual execution and release evidence. Context was 4096 for the 2k run and 10240 for 8k. The real decoder tokenizer was called twice per generated case to verify deterministic generation, in addition to all run-time depth measurements.

## Measured 2k results

One case per type, not a statistical quality evaluation. Material depths were exactly 2048 tokens except tracking, 2047. For the single needle, the record starts at material token 210. Multi-record offsets span early through late context. Local/hybrid/offload answer-prompt depths match exactly; maximum prompt depth is 2190 (split-local maximum 2243). Each normal hybrid treatment imported a total of 10744 tokens across the five ingests; every answer imported zero, with no lost prefix or fallback.

| Task | Gold | Local | Hybrid / offload | Split-local | Hybrid first differing token | Split-local first differing token |
|---|---|---|---|---|---:|---:|
| Single needle | `807569` | `807569` | `807569` | `807569` | null | null |
| Multi-key | `848382` | `848382` | `848382` | `848382` | null | null |
| Multi-value | `259093,382780,115067,759638` | exact | exact | `115067,759638` | null | 0 |
| Variable tracking | `name_a,name_d` | `name_b` | `name_b` | prose ending in `name_b` | null | 0 |
| Aggregation | `286` | `100.0` | `110` | `100` | 1 | 3 |

| Treatment | Accuracy | Final-answer agreement with local | Actual sampled-sequence agreement | Sum request time |
|---|---:|---:|---:|---:|
| local | 3/5 | reference | reference | 37.494 s |
| hybrid | 3/5 | 4/5 | 4/5 | 41.565 s |
| offload ingest / local question | 3/5 | 4/5 | 4/5 | 41.643 s |
| local split | 2/5 | 2/5 | 2/5 | 39.561 s |

Total request time: 160.263 s, excluding startup, hashing, generator checks and cleanup. These are sequential shared-host measurements, not a speedup claim. The final 2k run reproduced all earlier corrected-run answers. Both aggregation paths are wrong; agreement on other wrong answers is not accuracy. Hybrid disagreed on one case while the split-local reference disagreed on three. With N=5 and a framing-confounded control, this does not establish numerical equivalence or a calibrated noise bound.

## Measured 8k result

The single-needle case at requested depth 8192 has 8191 actual material tokens, with the fact at token 819. Every treatment returned the correct `807569`. Hybrid, offload and split-local sampled sequences all exactly match local (`first_differing_token=null`). Each hybrid ingest imported 8292 tokens; every later question imported zero. Normal answer prompts contain 8327 tokens, split-local 8380. There were no budget hits, fallbacks or prefix replays. Execution and the fixture both exited 0, and the node ended with zero sessions/retained bytes.

| Treatment | Correct | Request time |
|---|---:|---:|
| local | 1/1 | 57.047 s |
| hybrid | 1/1 | 58.066 s |
| offload ingest / local question | 1/1 | 58.659 s |
| local split | 1/1 | 57.357 s |

Total request time: 231.130 s, excluding setup and generation checks. Only retrieval was run at 8k; the other four types were exercised natively at 2k and in generator/scorer unit controls. This bounded CPU evidence does not establish multi-task 8k or long-context GPU quality.

## Tests, controls and exits

| Command / evidence under `.temp/hybrid-b-r21-261101/` | Exit | Claim / limit |
|---|---:|---|
| `python3 -m unittest discover -s tools/hybrid-eval -p test_eval.py -v`, `unit-01.log`, `unit-02.log` | 0, 0 | Six tests; 12 seeds x five independently interpreted oracles, 2k/4k/8k fitting with a character-tokenizer stand-in, actual positive/negative scorer cases, token divergence, routing controls, sanitized archive hashes. Native tokenizer evidence is separate. |
| `test_native.py`, `native-01.log` / `native-01/evaluation-exit.json` | 1 / 2 | Failed harness/evaluator, not passing evidence: incorrect full-rerender equality assumption after the answer. Its aborted bundle and release/cleanup records are retained. |
| `test_native.py`, `native-02.log` | 0 | Corrected 2k five-task/four-treatment E2E, real imports/local questions, equal paired actual prompts, empty remote store afterward. |
| `test_native.py`, `native-03.log` | 0 | Same matrix after adding stronger live decoder model identity admission; all prior observed answers reproduced. |
| `test_native.py`, `native-04.log` | 0 | Bounded 8k single-needle follow-up, four correct and identical sampled answers, real imports/local questions, empty remote store. |
| `python3 -m py_compile tools/hybrid-eval/hybrid_eval.py tools/hybrid-eval/test_eval.py tools/hybrid-eval/test_native.py`, `syntax-01.log` | 0 | Python syntax/import compilation, not type checking. |
| `python3 tools/hybrid-eval/hybrid_eval.py --help`, `eval-help.log` | 0 | Evaluator CLI readiness. |
| `ty --version`, `ty-readiness.log` | 127 | Missing; full type check unrun. |
| `editorconfig-checker --version`, `editorconfig-readiness.log` | 127 | Missing; full external EditorConfig check unrun. |
| `python3 .temp/hybrid-b-r21-261101/audit.py`, `audit-01.log` | 0 | Four real archive manifests/sidecars, actual credential exclusion, final evaluator identity, stable observed native binaries, 16 stopped PIDs and scoped native-dump cleanup. |
| `git diff --cached --check`, `diff-check-01.log`; focused format audit, `scoped-format-01.log` | 0, 0 | Staged whitespace check and ASCII/LF/final-newline/no-trailing-space checks on all five scoped files; Python has no tabs. |

Each test has a nearby claim comment. Negative scorer controls reject a wrong scalar, a partial multi-value response, reversed/duplicated/extra values, prose wrappers, empty output and filler-only output beside valid full-answer controls. Routing controls reject zero imports under a hybrid label, explicit fallback, lost evaluated boundary, replayed prefix and inflated prefill accounting. The real native run supplies the positive imported-cache control. Archive controls plant secret/path/address canaries in private metadata and verify that the allowlist excludes them, then independently verify each member and archive checksum. No source-mutant campaign or universal coverage is claimed.

## Shareable bundles and cleanup

Archives remain under their corresponding private run directories in `.temp/hybrid-b-r21-261101/`; only these sanitized archives are shareable. All four passed independent member-manifest/archive-sidecar checks and actual credential/path/address exclusion checks.

| Run / archive | SHA-256 |
|---|---|
| `native-01/hybrid-eval-cpu-20261101T113831157651Z.tar.gz` (aborted) | `6b18a5b6adb42e52f8c8ac4e5e498ce902d16c53811c5981eb8a81b6885a61d8` |
| `native-02/hybrid-eval-cpu-20261101T113921562600Z.tar.gz` (corrected 2k) | `e7a0f8383ec9c4cea6195655651a1b41169b3a0cffc34df0777a3a78fd713e3b` |
| `native-03/hybrid-eval-cpu-20261101T114244385972Z.tar.gz` (final 2k) | `c6b4c77b7ac3d1132824a22fee3a24ea3332a08717be7a7a021b53da75e30312` |
| `native-04/hybrid-eval-cpu-20261101T114537107831Z.tar.gz` (8k retrieval) | `86848f3347853c3ee950012594b064809df1cfac91dc07705b2305f90d7fd80f` |

All 16 recorded service/evaluator PIDs were inactive at audit. All observed native executable/library hashes remained unchanged, and both final runs bind the same evaluator source hash. Cleanup hashed and removed 61 owned native slot/database/credential/config files, 1554883728 bytes, retaining public metrics, logs, commands, checksums, process/cleanup records and result bundles. `native-dump-hashes.json` SHA-256: `450a8911014eb0be0703c037b1fb6d9a044b02ba403dc29b16ad948ad27759eb`. Earlier runs and model files were preserved. Private configs were removed only after the real-credential exclusion audit; full recreation uses the recorded harness command rather than the stopped runtime state.

## Exact GPU commands for the owner

Use an existing verified CUDA build record that actually matches its binaries. `--skip-build` refuses mismatches; do not substitute the stale CPU stamp above. `MODEL`, `BUILD`, `LOCK` and `RUN` are operator-owned absolute paths. No evaluator edits are needed for a different quantization; add `--hetero-quant` to client setup and supply the actual node-info.

```sh
export MODEL=/absolute/path/to/model.gguf
export BUILD="$PWD/build"
export LOCK=/absolute/path/to/shared/build.lock
export RUN="$PWD/.temp/hybrid-eval-gpu"
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
tools/hybrid-bench/setup-client.sh --state "$RUN/client" --stop
tools/hybrid-bench/deploy-prefill-node.sh --state "$RUN/node" --stop
```

For an already-running qualified node/client, run only the evaluator command with the actual client state. Pilot at 2048 tokens and one task/position first if policy calibration or memory fit is uncertain. The [README](README.md) gives the Mac SSH tunnel, local Metal client setup, tools/output table and per-depth runtime budget formula. The default GPU sweep has 60 cases and 540 completion requests; no GPU/Metal runtime estimate is presented as an observation.

Remaining qualification: GPU/Metal, 27B, long depths through 128k, heterogeneous task accuracy, more seeds/positions and an actual noise acceptance bound. The split control changes framing, so a later backend exposing an identical-prompt batch partition control would narrow the numerical interpretation. The small 0.8B tracking/aggregation failures remain visible. Required upstream/hosted checks, independent review, publication and landing were not requested for this no-push local handoff.

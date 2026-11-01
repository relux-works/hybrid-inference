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


## r22 - clean same-token partition control (2026-11-01)

Review r4q requested changes on `46c967334e902f2d6ab1f3c607c2a1c705c5102b`. The old `local_split` results above remain historical **framing sensitivity**, not numerical noise evidence. They must not support a paper claim that hybrid is within local noise.

The single task commit was rebased cleanly and re-signed onto the owner's exact `7e1a477b2e8dfc8146744f4f81d73abca25e7d4a`, producing `fa79daa472` before rework. The prescribed fetch of local prototype `main` actually returned `0124e89a`; the requested hash was available on its `m1-test-host-aware` branch and was fetched by exact object ID. No remote/default ref was changed. All subsequent work stays local on `m2-task-eval`, with no push.

### Mechanism and paired design

The simpler clean control is a **second local decoder/proxy with the node's batch/ubatch**, not a split conversation. The primary local decoder uses `-b 64 -ub 64`; the control and native node use 256/256. The same decoder binary, model, backend, context, threads, KV types, attention and checkpoint options are used for both local decoders. Both consume the exact same conversation. `comparison()` now refuses any unequal actual answer-prompt token hash, including the partition control and local repeat. `local_repeat` runs the primary local mode again in a fresh session with the same seed/settings. No framing control is included in any noise count.

JSONL and markdown report paired answer and sampled-token disagreement for each depth/type. Cells are both agree, H-only disagreement, P-only disagreement and both disagree relative to local, with H the named hybrid/offload treatment and P the partition control. The exact two-sided McNemar tail is conditional Binomial(H-only + P-only, 0.5), capped at one; no discordants gives p=1. The [official statistical reference](https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html) describes this exact binomial form. P-values are unadjusted and do not prove equivalence.

Duplicate depth/task/position arguments now refuse before state admission so repeated cases cannot inflate paired counts. Plan **300 distinct tasks per depth/type**, balanced over three positions with `--repeats 100 --positions 0.1 0.5 0.9`. Repeat/position pairs have distinct task indices rather than reusing facts at each position. Four depths x five types therefore require 6000 paired cases, five conversations each. There is no effect-independent sufficient sample size: assuming H-only probability .15 and P-only .05, exact two-sided alpha .05 power is .9736022 at N=300, versus .5444454 at N=100 and .8753005 at N=200. These are planning assumptions, not measured rates. Smaller effects and multiplicity correction require more cases; declare primary endpoints and multiplicity handling before paper claims. The reproducible calculation is `.temp/hybrid-b-r22-261101/planning.py`, output `planning-02.log` (exit 0).

Minimal scope extension outside `tools/hybrid-eval/`: `tools/hybrid-bench/manage.py`, its existing `test_bench.py`, and its README now accept, validate, test and document client `--batch/--ubatch`. Native node partition remains fixed at 256/256; defaults are unchanged. This makes the second decoder provisionable through the established launcher instead of an ad-hoc runtime script. No core/native code or public API changed.

### Short CPU pilot

Command, run from this checkout:

```sh
nice -n 10 python3 tools/hybrid-eval/test_native.py \
  --node-binary /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-hybrid-prefill \
  --decode-binary /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-server \
  --model "$HOME/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf" \
  --artifacts .temp/hybrid-b-r22-261101/native-02 --depths 2048
```

Exit **0**: five task types x five treatments, position .1, seed 17, max output 48, actual material 2048 tokens, context 4096, CPU only and two threads per service. Native children inherited effective nice 19 (harness nice 10 plus child adjustment). Existing A-workbench binaries were read-only. Native-01 had equal before/after executable/library hashes. The shared A build changed between pilots at 12:10 UTC, before native-02 launched at 12:13 UTC. Native-02 startup executable hashes match the post-run files; library hashes were captured afterward and all their mtimes precede launch, but its full before-run library snapshot is unavailable. The two pilots are not identical-environment evidence. Their build stamp is historically stale, so this is not an exact-source native rebuild claim. No configure, build, install or download ran; the shared build lock was not needed. No model or golden was modified.

| Treatment | Exact correct | Answer disagreements/local | Token disagreements/local | Budget hits | Request seconds |
|---|---:|---:|---:|---:|---:|
| local | 3/5 | - | - | 0 | 38.144 |
| local_repeat | 3/5 | 0 | 0 | 0 | 39.077 |
| hybrid | 3/5 | 1 | 1 | 0 | 49.161 |
| offload | 3/5 | 1 | 1 | 0 | 46.842 |
| local_partition | 3/5 | 0 | 0 | 0 | 42.764 |

All 20 nonreference prompt comparisons are token-identical. The aggregate descriptive hybrid/partition table is (both agree=4, H-only=1, P-only=0, both disagree=0) for both answer and token endpoints. The sole hybrid/offload disagreement is aggregation; the other four tasks agree with local. Each persisted per-type paired table has N=1 and exact p=1. This pilot is too small for a noise/equivalence conclusion. Accuracy remains 3/5, with aggregation and tracking failures visible in every treatment.

The generated summary says **hybrid equals offload on whole-ingest routing: 5/5**. Both modes offload the whole ingest except its final local token and answer the later question locally. Their agreement is not independent evidence. It now also includes per-mode budget-hit counts and both model/quantization identities plus a heterogeneous label; heterogeneous disagreements explicitly include quantization/model effects, not just numerical noise.

### Tests and controls

All test commands ran directly with redirected logs; no tee pipeline or fabricated status. Logs below are relative to `.temp/hybrid-b-r22-261101/`.

| Check | Exit | Evidence and bound |
|---|---:|---|
| Evaluator unit suite | 0 | `unit-05.log`, 11 tests, including prompt rejection/acceptance, partition identity, local-repeat reporting, constructed paired cells/tails, endpoint separation and archive exclusion |
| Existing benchmark suite plus partition tests | 0 | `bench-unit-02.log`, 27 tests; selected partition reaches real startup-command construction and metadata, invalid settings refuse before preflight/state writes |
| Existing proxy suite | 0 | `proxy-unit-01.log`, 68 tests, unchanged proxy source |
| Native integration | 0 | `native-01.log` and final-source `native-02.log`, each five real task types x all five modes; import, identical prompt and local-answer checks; zero retained remote sessions/RAM |
| Python syntax, whitespace and evaluator CLI help | 0 | `syntax-02.log`, `whitespace-01.log`, `evaluator-help-01.log` |
| Archive/private-data/process/binary audit | 0 | `audit-02.log`/`audit-02.json` (pilot 1), `audit-native02-02.log`/`audit-native-02.json` (pilot 2); member hashes/sidecars, actual credential/address/path exclusion, prompt hashes, all 12 recorded PIDs gone and ten ports closed; native provenance bounds stated above |
| Narrowing mutation runner, corrected controls | 0 | `mutants-04.log` (eleven controls); each named mutant subprocess actually exits 1 with its named assertion failure |
| Type/EditorConfig checker readiness | 127 / 127 | `ty-readiness.log`, `editorconfig-readiness.log`; both unavailable, so their actual checks are **unrun**, no packages installed |

The preimplementation unit run `unit-red-01.log` exited 1 because the new functions were not yet defined; that is not mutation/control evidence. The native node readiness invocation intentionally omitted arguments and returned its expected usage with exit 2, not a passing gate. The first audit exited 1 because its port index selected the token-file path; the corrected task-local audit uses the actual node port and exits 0. No product/source change was needed for that audit correction. The first native-02 audit (`audit-native02-01.log`, exit 1) incorrectly reused pilot 1's binary snapshot and exposed the between-pilot rebuild; the corrected audit verifies pilot 2's startup executable hashes and records the missing library-before snapshot instead of claiming identical environments. Its immediately dependent report read also exited 1 because no successful audit artifact yet existed. The corrected audit exits 0. A final self-review added duplicate-argument refusal, then reran the 11 evaluator unit tests and all five native task types on the final source; both pilots have identical prompt/answer hashes, exact scores and budget flags (`repeat-check-01.log`).

| Mutant | Narrows the gate to | Named failing test | Result / survivor bound |
|---|---|---|---|
| Remove prompt identity refusal | Rejecting the r4q changed-framing witness even with identical output | `test_noise_control_changes_input_regression` | killed, exit 1 |
| Remove different-partition refusal, first control | Original negative combined same partition with node mismatch | none | **survived**, exit 0; original test proved only another identity/partition rejection, not this guard |
| Remove different-partition refusal, isolated control | Same client/control/node partition alone must refuse | `test_partition_identity_controls` | killed after adding isolated regression, exit 1 |
| One-sided tail | Two-sided exact probability, not half the intended p | `test_paired_summary_constructed_table` | killed, exit 1 |
| Swap discordant cells | Direction and denominator of each paired cell | `test_paired_summary_constructed_table` | killed, exit 1 |
| Use text for token endpoint | Token disagreement is independent of normalized text agreement | `test_paired_summary_constructed_table` | killed, exit 1 |
| Hide budget column | Budget censoring remains visible in markdown | `test_summary_baselines_and_labels` | killed, exit 1 |
| Substring scoring | Extra prose must not pass exact scoring | `test_exact_scoring_negative_controls` | killed, exit 1 |
| Allow missing import | Hybrid routing requires an actual nonzero import | `test_patchwork_route_controls` | killed, exit 1 |
| Allow wrong control model | Model identity must match between the local decoders | `test_partition_identity_controls` | killed, exit 1 |
| Archive credential | Shareable metadata must exclude a credential canary | `test_sanitized_bundle` | killed, exit 1 |
| Allow duplicate sweep cases | Repeated arguments must not inflate paired counts or reach state admission | `test_duplicate_cases_refuse_before_admission` | killed, exit 1 |

Initial mutation runner exit 1 and all initial results remain in `mutants-01.log`/`mutants-01.json`; there are no survivors in the corrected six-mutant, extended ten-mutant or final eleven-mutant runs. These controls are finite instrument checks, not exhaustive mutation coverage.

### Artifacts, review and limits

Final-source shareable bundle: `.temp/hybrid-b-r22-261101/native-02/hybrid-eval-cpu-20261101T121346660615Z.tar.gz`, SHA-256 `53f44a23f68ef0cc2d8aaea876710c73685597fe283c417e29392d22d10a3864`. Evaluator SHA-256 `60be16484a23b5669271f0ef39d2308e75addb28ef20d0afab52d5d969f07b69`. Public JSONL contains 20 per-type paired rows and the determinism/partition results. `audit-native-02.json` observed cleanup and the bounded native provenance at `2026-11-01T12:19:04.727273+00:00`. All shareable member manifests and the sidecar verified. The first pilot's retained archive is `native-01/hybrid-eval-cpu-20261101T120301427504Z.tar.gz`, SHA-256 `7d1bb92ce19b5581308dfddd0849c265d693b9529660fcc4d02fc365f54dd7a2`, evaluator SHA-256 `5d6168ba33a9912925208c584491ade575911bc725a12b8565fedac96550da18`; its full before/after binary snapshot is in `native-identity-before.json` and `audit-02.json`.

Owned parked slots, SQLite state and credentials were hashed then removed: 32 files, 800984469 bytes, `cleanup-manifest.json` SHA-256 `e51910d46811f7af739d8f05b4b559adf90961490c6f575ea1a2610d8eec6619`. Final-source pilot 2 cleanup: 32 more files, 800988589 bytes, `cleanup-manifest-02.json` SHA-256 `dad336c3053d68775a61cf7bbc4f9182e1b605b63c11cd4ab0578caf4d03adc5`. Across both pilots 64 owned files / 1601973058 bytes were removed. No other task's state was removed.

Self-review uses the code-review skill's scope, security, approach, tests and general checklists. The named prompt regression addresses r4q's finding; review said `repeat-of: none`. The owner rework brief has no acceptance surface table or designated landing-property suite, an explicit brief gap. The six observational rows in r4q are mapped in [coverage-map.md](coverage-map.md), with named tests, killed narrowing mutants and finite limits; this does not claim exhaustive backend coverage. Previously committed evaluator/benchmark/proxy tests remain present and green. The only out-of-module paths are the three launcher paths justified above. Unchanged native/backend behavior, cross-backend distribution/KL calibration, EOG inclusion details, heterogeneous native accuracy, 27B/GPU/Metal and long-depth/paper-size measurements remain outside this CPU-only task. Batch metadata is checked but `/props` does not expose its live batch settings; custom launches need their startup-command audit. No calibrated noise-equivalence acceptance is claimed.

This is a local ready-for-review handoff. Independent acceptance, hosted checks, publication and landing were not requested and no push was made. The signed rework head is supplied in the orchestrator notice and human handoff; no session links are included.

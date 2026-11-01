# R26: pinned reference-model logits

Ready for review on local branch `m3-reference-logits`, based on fork main `a3752b04cbd905bf61ffe6d9a4e7df738f26f4c2`, fetched from `https://github.com/relux-works/hybrid-inference`. Local signed delivery only; no push or PR. The owner-specified co-author trailer applies.

Evidence directory: `.temp/hybrid-b-r26-261101/` (`RUN` below). Final native evidence is `native-06/`; code/test/build-input identity is `source-identity.json`, SHA-256 `bccd2bc8931160c6bb98b458d37c69215667d2fb627ceaffcec8be950792c397`. Audit observed at 2026-11-01T16:32:03.699955+00:00.

## Behavior and scope

`llama-hybrid-score` is a small sibling of `llama-hybrid-bench-check`, linked to the existing state/delta/hash library. It locally prefills N context tokens or imports the same verified delta chain ending at N-1 and computes the last context token locally. Position N+i predicts continuation token i after precisely the preceding i tokens. The initial context supplies row 0; batches decode continuation tokens 0 through M-2 with all logits requested. No sampling occurs. Explicit position, token, hash and independent prefix witnesses bind the indexing.

The same binary produces D references from local prefill and compares a client against a separately pinned D model hash. Reference and client vocabulary size, vocabulary SHA-256 and full layout fingerprint must match. Client model SHA-256 is verified against its GGUF. Heterogeneous delta import requires the source model hash, `hetero_quant` and a matching node/client layout; the scorer reads original source envelopes without relabeling them. The existing checker continues to report its own same-partition, whole-prompt and alternate-partition controls.

`hybrid_bench.py reference` produces D first. The existing `--check` flow accepts `--reference-logits PATH --reference-model-sha256 SHA`; it scores the exact continuation from that file and journals the native exit and metrics before cleanup. M is independent of the older checker K (which remains capped at 256). JSONL and Markdown report both arms and their spread, with status `measured` and no quality threshold or ranking. A reference is for one exact context, N and M; mismatched scheduled checks refuse instead of aligning or regenerating it. Client setup builds/hashes the new target.

`hybrid-eval` integration is explicitly deferred in its README: its sampled task answers differ between arms, and it does not export the framed answer-prefix state as an importable native chain. It needs per-item fixed continuations and exact prompt/state identities before reference scoring can be meaningful. The direct scorer supports arbitrary fixed-token items now.

## Reference file format

All integers and float16 values use little endian. The file starts with 8-byte magic `HYBLOG01`, a uint32 JSON-header length (bounded to 8 MiB), then that UTF-8/ASCII header. Header format is `f16-logsoftmax-le-v1`; it records model SHA-256, SHA-256 of all N+M token IDs encoded as int32 LE, N/M, context/continuation IDs, vocabulary size/hash, layout fingerprint, GPU layers, continuation batch, threads and KV type.

Each row contains uint32 absolute position, uint32 target token, uint32 original D argmax token, V float16 log-softmax values, and a 32-byte SHA-256 over that row metadata and those float16 bytes. Total: **2*V + 44 bytes per position**, plus the shared header. At V=248320 this is **496684 bytes/position** (about 485 KiB), about half float32 storage. M=2048 would use 1017208832 payload bytes (about 970 MiB), plus the header. Reading is streaming: row length, exact file length, row checksum, finite/nonpositive values and near-unit mass are validated before any score result is opened, then rows are reread during scoring. The per-row probability vectors are normalized in double precision before KL.

Original argmax IDs preserve top-1 agreement even if float16 rounding introduces a tie. KL is explicitly **D || arm**, in nats; `local_to_hybrid` is **local client || imported client**. Quantiles interpolate at `(M-1)*p`. Context+continuation IDs and full distributions remain in the private reference; result identities contain hashes/settings and per-position numbers. Existing outputs refuse overwrite; new files publish from private temporary files. Invalid reference inputs leave no result or partial result. This does not claim atomic publication of the reference/result pair across disk errors or process termination.

Final nine-row reference: `ff05355c920b3b3c5f7e578f36c4216931e16cbfcd703a9b667062fcd333fccd`, **4470751 bytes** including header. The production Python CLI generated a byte-identical file and removed its temporary token fixture.

## CPU measurements

No downloads or GPU calls. Short `nice -n 10` runs used two CPU threads, q8_0 KV, f32 recurrent state, batch/ubatch 256, continuation batch 3, N=16 and M=9 synthetic token IDs. The token fixture is persisted under `native-06/reference-write.json`; token SHA-256 is `0dafe25875a9584582ddf3ff7d1f7257cd32d8087baa7b718699b528c4245542`. These are protocol/numerical witnesses, not a natural-language quality sample.

- Client Q4_0: `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`.
- Reference/local node `Qwen3.5-0.8B-Q8_0-requant.gguf`: `f543fad263339779c3a38a21a531c6adbffcfdecdf5f749dd542de924b1e7302`.
- Shared layout: `33d9f73396507b1f8402bf1e439936f42caee2c8875fb46e127b06e0e656d2fa`.

This local Q8 artifact is not the BF16 27B reference D planned for the GPU study. No claim is made that requantization restores original high-precision weights.

| Comparison | Mean KL | Median | p90 | p99 | Max | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Stored Q8 || local Q8 | 6.338408327e-07 | 8.095402196e-07 | 9.480821731e-07 | 1.006014253e-06 | 1.012451151e-06 | 9/9 |
| Stored Q8 || local Q4 | 0.005895910388 | 0.0041638235 | 0.01034454679 | 0.01964576069 | 0.0206792289 | 9/9 |
| Stored Q8 || Q8-state/Q4 decoder | 0.005826162669 | 0.003638569604 | 0.01220887507 | 0.01716350176 | 0.01771401584 | 9/9 |
| Local Q4 || Q8-state/Q4 decoder | 0.002525589087 | 0.00187312621 | 0.004755845606 | 0.005354256037 | 0.005420746085 | 9/9 |

Per-position KL (nats), with identical teacher-forced tokens in all three columns:

| Position | D || local Q4 | D || imported Q4 | Local Q4 || imported Q4 |
|---:|---:|---:|---:|
| 16 | 0.007760876265 | 0.002597946943 | 0.005420746085 |
| 17 | 0.0206792289 | 0.01771401584 | 0.00187312621 |
| 18 | 0.0041638235 | 0.006294783896 | 0.004475476491 |
| 19 | 0.002690779341 | 0.002424754516 | 0.001399294695 |
| 20 | 0.005440269076 | 0.005641998299 | 0.001554170864 |
| 21 | 0.003473220577 | 0.01083258988 | 0.004589620486 |
| 22 | 0.003682809168 | 0.002313881102 | 0.001953815987 |
| 23 | 0.004403568028 | 0.003638569604 | 0.0008631373004 |
| 24 | 0.0007686186412 | 0.0009769239443 | 0.0006009136677 |

The separately requested reverse-direction control, KL(Q4 || Q8), was also positive (mean 0.005970495401, max 0.021474202422 nats) and exactly stable on rerun. The study endpoint remains D || arm. A genuine Q4 reference file paired with the Q8 pin rejects as `reference model hash mismatch`.

Q8 and Q4 reruns had identical per-position results; the imported arm was also exactly stable on its repeated partition. Q8 continuation batches of one and three produced identical rows on this fixture. Separate local prefills ending at continuation indices 3, 4 and 8 agree with their stored rows within the witness bound of 1e-4 nats, covering adjacent batch boundaries and the last row. A three-delta append chain (ends 5, 10, 15) is additionally exercised; it is reported separately from the repeated fresh partition.

Float16 encoding of the original Q8 distributions measured mean KL **6.338485908e-7**, max **1.012431874e-6** (original || normalized encoded), maximum absolute log-probability error **0.007813447647**, and 9/9 argmax agreement. Stored Q8 || local Q8 has max **1.012451151e-6**, within the measured encoding floor (test margin 10%). The unencoded self-comparison is exactly zero. The Q4 encoding max KL is **1.094791994e-6**. These bounds justify float16 for this fixture; every production reference writes its own encoding-error measurements and must be assessed on that corpus. No universal error bound or 27B result is inferred.

## Validation and controls

All commands ran as standalone processes, without tee/pipeline masking. TMPDIR was `$PWD/.temp/hybrid-b-r26-261101/tmp`, on disk. Exact native commands, PIDs, expected and real exits are in `native-06/commands.json` (29 processes); node/checker processes have separate logs/PID records. Outer CPU harness exits 0 while preserving expected nonzero native and witness exits.

| Command / evidence under RUN | Real exit | Claim |
|---|---:|---|
| Existing CMake `-S . -B build`, then shared-lock `cmake --build build -j2 --target llama-hybrid-score llama-hybrid-bench-check llama-hybrid-prefill`; `configure-06.log`, `build-06.log`, `build-record-06.json`, driver-06 | 0 | CPU compilation. CUDA/Metal/UI downloads disabled in the existing cache; no install. Minimum sampled free bytes 33777303552, above 25 GiB. |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v`; `bench-tests-02.log` | 0 | All 42 existing benchmark/cleanup/identity tests. |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_reference.py -v`; `reference-tests-02.log` | 0 | Two metadata framing/binding tests, with valid controls and wrong pin/context/N/M/length cases. |
| `python3 -m unittest discover -s tools/hybrid-eval -p test_eval.py -v`; `eval-tests-01.log` | 0 | Existing 11 evaluator tests; evaluator behavior is unchanged. |
| `python3 tools/hybrid-bench/test_reference_native.py --client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --reference /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --artifacts $RUN/native-06`; `native-06.log` | 0 | Analytic asymmetric KL/spread self-test; same-model encoding floor, positive Q4 distance, deterministic reruns, scalar batches, independent prefixes, all file refusals, real Q8 node and both Q4 arms, three-delta chain and sanitized bundle. |
| Native malformed-reference subcommands in `native-06/commands.json` | **2** each | Wrong expected reference model, context token, continuation token, N, M, truncation, trailing bytes, vocabulary token hash, vocabulary size, NaN, +inf, -inf and row checksum. Each asserts its specific message, no result/temp result, and unchanged original reference. |
| Existing native equivalence checker in three real-node runs | **5** each | Existing heterogeneous status is `uncalibrated`, not green/accepted. All reference scorer invocations return 0 and preserve the original equivalence rows. |
| `hybrid_bench.py reference ... --score-batch 3 --max-ctx 512`; `producer-cli-02.log` | 0 | Production CLI creates the same reference bytes and encoding result as the native path; fixture removed. |
| Python compile and `git diff --check`; `gates-01.json` | 0 each | Changed Python syntax and whitespace. Added source/test text is ASCII. |
| `ty`, `editorconfig-checker` readiness; `gates-01.json` | UNRUN | Executables absent; type and editorconfig checks were not run, no dependencies installed. Model-naming style gate is outside this diff (no model files changed). |
| `$RUN/audit.py`; `audit-03.log`, `audit-03.json` | 0 | Source identity, reference equality/private mode, no producer fixture, all 187 recorded owned PIDs inactive, five archive manifests valid, no reference/delta/token-input files in public bundles. |

| Plant / mutant | Gate narrowed to | Named failing test / real exit | Survivor bound |
|---|---|---|---|
| Checksum-valid row marker shifted from N to N+1 | Exact absolute position, rather than checksum-only integrity | `shifted-position-control`: native 2, `reference position mismatch`, no result | Killed; metadata identity only |
| Correct position/target/checksum, but log-probabilities copied from the next token position | Actual predictive distribution at the independently prefilled prefix | `test_prefix_position_witness`: real subprocess **1**, intended assertion; native scorer returns **0** with measured KL **0.200981206653** | Killed by numerical witness; scorer adds no quality gate. Positive prefix controls pass. |

No surviving planted controls are claimed. The numerical control is a separate expected-red witness, not a native file-format refusal; its output and traceback are retained in `wrong-row-control.result.json`, `wrong-row-witness.log` and `position-control.json`.

Earlier failures remain explicit: build-driver-01 exited 1 because CMake had not generated the new target (build command 2); regeneration corrected it. Native-01 exited 1 due to a harness session-list wrapper assumption after the first native scorer had succeeded; the route returns a bare list, and native-02/03/04/05/06 passed after correcting the assertion. Native-03 caught the numerical plant as an in-process AssertionError; its `assertion_exit_code` field was an inference, not a process exit, and is not credited as exit evidence. Native-04 runs the actual witness subprocess and records its real exit 1. Audit-01 exited 1 because some PID records are JSON instead of bare integers; audit-02 corrected this, and audit-03 passed on final source. The last review added the writer header-size guard; build06/native05/native06 reran that changed identity. No failed gate is reported green.

## Review and remaining qualification

Self-review applied the code-review skill scope/security/general/tests/design checklists: no core/model/backend/API change; existing hash/layout/context/delta infrastructure reused; bounded header/row/fixture sizes; finite checks before metrics; original source hashes retained; correct D || arm direction independently checked against an analytic pair; explicit teacher-forcing indexing and scalar/prefix witnesses; private full distributions excluded from public bundles. Prior benchmark 42, integration 6 and evaluator 11 test names remain unchanged (`prior-tests-kept.json`).

The brief supplies no surface table, recorded as a coverage-map brief gap. No rows were treated as out of contract to remove a requested test. The existing evaluator reference flag is deferred under the task's explicit documentation alternative. No upstream review/check/landing or independent reviewer acceptance is claimed; this is a locally signed prototype handoff.

Only the GPU run can measure the actual BF16 Qwen3.8-27B D against Bonsai/Mirai arms over long held-out contexts, 2k continuations, realistic distributions and their document-level spread; it must measure its own repeat/encoding floor. CUDA/Metal differences, GPU memory/throughput, large-context import, true low-bit quantization effects, resync B2, natural-language task accuracy and bootstrap confidence intervals are unqualified here. The tiny CPU fixture provides no model-quality ordering or paper-level inference.

# Sequential depth quality: r47 CPU handoff

Review correction r47c: historical factor-two 512k fixtures are superseded. The current study uses static YaRN factor 4 at both depths above original 262,144 for kit/vLLM74/1M comparability; regenerate old inputs. The handoff verifier now refuses absent/null engine revisions, which previously skipped the engine check. Licence declarations and GPU-side hashed evidence are recorded in the corpus pins and README. Historical CPU outcomes below do not qualify this corrected profile at 512k/1M. See `RESULTS-hybrid-b-r47c-261103.md` for revalidation.


Ready for review. Base: `m3-state-bf16` at `8d876c71b798651a8c5df768dc374468ced77544`; local branch: `m3-depth-quality`. No push or GPU/model/corpus download on agents1. This is implementation and CPU mechanics evidence, not a 512k/1M quality result.

## Delivered experiment

`stage_depth_quality.py` fetches and verifies the existing immutable Qwen3.8-27B Q4_K_M pin, freezes two Gutenberg sources before seeded token construction, builds the new `llama-hybrid-depth` target, and runs 524,288 and 1,040,000 token cases sequentially. Seeds are 42/43; each source token stream repeats cyclically with deterministic rotation. Needles replace tokens at known spread positions, preserving document length. Default batches are 2048 for offload/local, 1024 for partition noise, and 32 for the common full-vocabulary scoring schedule. Model-card YaRN is factor 2/4, original context 262144, theta 10000000, with identical recorded native settings in each arm.

The node encodes bounded f32 delta-v2 chunks in one disk container and exits before client launch. The client uses the existing `hybrid::delta_import` stage/apply path. The transfer is removed before the native reusable snapshot is saved; no second large state file is created. The stage requires an idle 96 GB RTX PRO 6000, at least 120 GB host RAM, enough available RAM for transactional delta staging plus weights/margin, and approximately 142 GB initially free disk including the 28 GB floor. Raw logits total approximately 48.8 GB for the default four cases. Builds use `nice`, `-j4`, and an active-compiler check. Owned PID/exit ledgers, checked archives, terminal done/failed lines and build cleanup are included.

The default `--window final` checks the final N document positions: export at depth minus N, import, teacher-force the N suffix tokens on the client, then save the full post-context state for needles. This short local scoring suffix is explicit; the node does not prefill those N tokens. Optional `--window after` exports the full depth and checks a held-out continuation, with needle restoration to the pre-tail snapshot. This preserves the alternative discussed during implementation without conflating it with the requested in-document window. Reported positions are the input positions after which logits are observed.

The report includes all three arm pairs, both KL directions, full-vocabulary top-1 agreement, maximum absolute raw-logit difference, every scored position, exact needle scores and raw answer agreement. Offload || local is shown next to partition || local, with numeric differences in JSON. There is no equivalence threshold, population confidence claim or upper bound on partition noise. HTTP/network behavior, bf16 recurrent transport, cross-backend portability and GPU performance are outside this CPU handoff.

## Validation and observed limits

Evidence is task-local under `.temp/hybrid-b-r47-261103/worktree/.temp/` in the original workbench. `source-binary-identity.json` binds the reviewed code/docs, tests and native binary; `build-cleanup.json` retains the binary hash after build removal. Runtime/native evidence is `native-08/`, including per-child commands/PIDs/real exit codes, raw logits and comparison reports. No brief surface table was supplied; that remains a brief gap rather than a claim of exhaustive coverage.

| Command or control | Actual exit | Evidence / claim |
|---|---:|---|
| CPU CMake configure, Release/no CUDA/no UI/no downloads | 0 | `configure-01.log`; existing local CMake 4.4.4 |
| Initial native build | 2 | `build-01.log`; missing extension-header declaration, corrected |
| Corrected native build, final source | 0 | `build-06.log`, `build-driver-06.log`; `nice -n 19`, `-j4` |
| Initial native suite | 1 | `tests-native-01.log`; ordered-JSON key-order mismatch on the valid positive control, corrected to semantic JSON equality |
| Final depth suite, native enabled | 0 | `tests-native-08.log`; 12 tests, 29.144 s, no skips |
| Existing shared stage regression | 0 | `tests-stages-regression-02.log`; 19 tests |
| Earlier regression discovery at wrong directory | 5 | `tests-stages-regression-01.log`; zero tests collected, corrected directory above |
| Stage plan, CPU/read-only | 0 | `stage-plan-05.log`; terminal done line, no GPU/download/build |
| Stage invalid batch | 1 | `stage-invalid-batch-01.log`; intended failed line before any GPU action |
| Python grammar, ASCII, whitespace and scoped diff checks | 0 | Python 3.10 grammar parsed locally; `git diff --check` |
| `ty --version`, `editorconfig-checker -version` | 127 each | Tools absent; full type/editorconfig checks unrun, no installation/download attempted |

The native end-to-end test uses the existing `Qwen3.5-0.8B-Q4_0.gguf` as the small-model CPU exception, 512 document tokens, final positions 504-511, two transfer chunks, two needles and prefill batches 8/7. All three snapshots have depth 512. Offload || local KL and max logit difference are zero over these eight full-vocabulary rows; top-1 is 8/8. Partition || local mean KL is 0.000150417994 nats, max absolute logit difference 0.588937283, top-1 8/8. All arms score 2/2 needles and return identical raw answers. A second native case exercises `after` positions 128-135 and verifies all needle snapshots remain at depth 128. Neither case bounds deep CUDA behavior.

Named negative controls use a valid exported state first, then change only the document or YaRN binding. Both native imports exit 2 with `state document or YaRN mismatch`, leave state and sidecar bytes unchanged, and create neither logits nor result JSON. Unit tests independently plant those violations in the arm-report gate with a valid positive control. Truncated and nonfinite logits also exit 2 for their intended reasons and publish no comparison. A planted wrong archive member hash raises the intended checksum refusal and publishes no archive seal. Exact scoring rejects incorrect values, substrings and explanatory text alongside a valid whitespace-only control. These are bounded input/control plants, not a claim of exhaustive mutation coverage.

Self-review used the repository code-review skill: scope, security, design, general and tests. Shared runtime/importer/core behavior is unchanged. The only changes outside `tools/hybrid-bench` are the native target registration in `tools/hybrid/CMakeLists.txt` and the root README tool entry. Existing reviewer tests were retained; there is no separate new landing-gate property suite defined for this task. Independent review remains pending.

## GPU execution and estimates

Use the [operator commands](README.md#depth-quality-at-512k-and-about-1m) on the signed candidate checkout. The entry point is `tools/hybrid-bench/stage_depth_quality.py`; planning requires no GPU. Execution requires `--execute --root /data/depth-r47 --ref FULL_COMMIT_SHA`. Optional pre-existing Gutenberg pins can be supplied with `--document-pins`; otherwise the first fetch is frozen as a new study, not historical reconstruction.

For two documents per depth, the planner budgets six document prefills, 24,576 scoring positions, up to 3,072 generated needle tokens, snapshot writes/checksums and repeated reads. Assumed prefill rates are 1,000-3,000 tokens/s, tail scoring 40-120 tokens/s, effective state I/O 2-5 GB/s and greedy generation 20 tokens/s. At the illustrative, unverified rental assumption of $1.57/h:

| Depth, two documents | GPU hours | Cost assumption |
|---|---:|---:|
| 524,288 | 0.51-1.39 | $0.80-$2.18 |
| 1,040,000 | 0.91-2.53 | $1.43-$3.98 |

Setup, download, build, query prefill, full-logit comparisons/hashes, archive and billing granularity add time. The under-one-hour expectation is not established. Actual CUDA/27B capacity, throughput, YaRN behavior and quality at both depths remain GPU-host qualification work.

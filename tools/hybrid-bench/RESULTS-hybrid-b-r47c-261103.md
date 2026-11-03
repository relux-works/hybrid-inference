# Depth quality review corrections: r47c

Ready for review. Parent: `fc317d82ca9d61206a78510e89c151384f48c3bf` on `m3-depth-quality`. No push, GPU execution or downloads on agents1.

F1: every depth above 262,144 uses static YaRN factor 4, including 512k. Original context 262,144, theta 10,000,000, beta 32/1 and ext 1 stay fixed. The README distinguishes the model card's factor-two 512k-only recommendation from this study's shared kit/vLLM74/1M profile. Old factor-two long fixtures are superseded; CPU fixture regeneration is recorded below.

F2: a missing or null engine ref now refuses before fixture access or native replay. The recorded full revision must match a clean checkout. Named missing/null negatives retain a real correct-ref saved-slot positive control in the native pipeline. That positive runs from the signed code candidate without mocking Git identity or cleanliness.

N1: each public source records its declared identifier and location: MIT in the pinned llama.cpp root LICENSE, apache-2.0 in oasst1 README YAML, cdla-sharing-1.0 in TinyStories README YAML. GPU retrieval saves LICENSE or license-metadata.md beside the document, verifies the declaration and records its actual SHA256/size in license.json and the derived document identity. The MIT expected hash comes from the existing pinned local Git object. Dataset README hashes are first recorded at GPU retrieval. No offline fixture establishes remote availability or legal suitability.

## Validation

Tested signed code candidate: `28c8a117b6023f88bebef6957d62705297b4fe6f` (signature G, clean checkout). The subsequent evidence-only commit changes this report; source and test hashes remain those in `.temp/r47c/source-binary-identity.json`. Evidence lives under `.temp/hybrid-b-r47-261103/worktree/.temp/r47c/` in the workbench.

| Validation | Actual exit | Evidence and scope |
|---|---:|---|
| CPU configure / native build | 0 / 0 | `configure-01.log`, `build-01.log`, `build-record.json`; compiler-idle check, existing CMake 4.4.4, nice 19, -j4, no CUDA/UI/download |
| Full depth suite | 0 | `tests-native-01.log`; 21 tests, 39.699 seconds, no skips; all 17 prior names retained |
| Shared-stage suite | 0 | `tests-stages-02.log`; 19 tests, 0.555 seconds |
| Fixture regeneration | 0 | `fixture-regeneration-01.log`, `fixtures/regeneration.json`; fresh tiny native pipeline plus CPU vocab-only 524,288/1,040,000 fixtures with factor 4 |
| Read-only stage plan | 0 | `stage-plan-01.log`; both depths explicitly record factor 4 and the remaining YaRN fields |
| Scoped static checks | 0 | Python 3.10 grammar, ASCII, JSON and whitespace checks; no replacement claim for the unavailable full type/editorconfig checks |
| Type/editorconfig readiness | 127 / 127 | `ty-readiness-01.log`, `editorconfig-readiness-01.log`; tools absent, full checks unrun, no installation/download attempted |
| Mutation-control wrapper | 0 | `mutation-controls.json`; the individual planted-regression tests fail as expected, as listed below |

The preliminary contract-only run exits 0 with four native skips; it does not substitute for the full suite. An initial wrong discovery directory selected zero shared-stage tests (`tests-stages-01.log`); it is excluded from passing coverage. The corrected standalone command above ran all 19.

The long regenerated fixtures use the pre-existing Qwen3.5-0.8B Q4_0 model's vocabulary and synthetic CPU prose. Their 5,215,841 / 10,325,747-byte JSON files prove token construction/settings binding without deep inference. They are not production Qwen3.8-27B Q4_K_M inputs. Production document fixtures must be generated afresh on the GPU host with the pinned weights and public source bytes; no old factor-two 512k fixture is reused.

At CPU D512/final8/K2, local needle accuracy is 2/2, and offload/local logits are bit-identical. The correct-ref native saved-slot64 replay has KL 0, maximum absolute difference 0 and top-1 64/64. It runs from the actual clean candidate ref with no Git mocking. Both named missing/null tests and corresponding mutations of that real handoff refuse before native replay; the state bytes remain unchanged. These are CPU mechanics observations, not long-context or Metal quality results.

Generated corpus fixtures exercise positive hashed licence receipts beside each document and negative missing/wrong declaration controls. Remote corpus/licence retrieval remains unrun. The actual code LICENSE hash was computed from the pinned existing local Git object, not downloaded.

| Mutant | Gate narrowed to | Named failing test(s) | Survivor bound |
|---|---|---|---|
| Restore factor 2 through 524,288 | Static factor 4 starts above 262,144 | `test_yarn_factor_512k_static_four` | Killed; only the stated boundary/depth samples are covered |
| Restore the old optional engine gate | Missing/null engine identities must refuse before reading the fixture | `test_handoff_missing_engine_ref_refuses`, `test_handoff_null_engine_ref_refuses` | Killed; no exhaustive malformed-manifest claim |

The owned build tree was deleted after recording its binary hash. Raw native arrays were hashed before deleting regenerable extras (`native-01-raw-identity.json`); the decisive reference/candidate pair and three final-window arm logits remain. No state file remains and the retained evidence is not an intact transferable slot bundle. Cleanup free space was 28,873,416,704 bytes, above the floor; the pre-build admission required at least 28.2 GB.

Planning remains 512k three documents 1.58-2.41 GPU hours / $2.49-$3.79, 1M two documents 2-3 hours / $3.14-$4.71 at assumed $1.57/hour, excluding setup and other documented overhead.

All 17 prior depth tests are retained under their original names; four added tests cover the factor boundary, missing/null engine refs and licence provenance/declaration refusal. Changed review surfaces: 1b (F1), 4c (F2), 3b (N1). No new surface table or additional landing property suite was supplied; that is a brief gap. Other accepted surfaces are unchanged. Changes stay inside tools/hybrid-bench; no core/backend or kit changes. Target CUDA/Metal/27B/deep quality, real public retrieval and timing remain unrun. Independent review is pending.

# Depth quality review corrections: r47c

Ready for review after the evidence below is filled. Parent: `fc317d82ca9d61206a78510e89c151384f48c3bf` on `m3-depth-quality`. No push, GPU execution or downloads on agents1.

F1: every depth above 262,144 uses static YaRN factor 4, including 512k. Original context 262,144, theta 10,000,000, beta 32/1 and ext 1 stay fixed. The README distinguishes the model card's factor-two 512k-only recommendation from this study's shared kit/vLLM74/1M profile. Old factor-two long fixtures are superseded; CPU fixture regeneration is recorded below.

F2: a missing or null engine ref now refuses before fixture access or native replay. The recorded full revision must match a clean checkout. Named missing/null negatives retain a real correct-ref saved-slot positive control in the native pipeline. That positive runs from the signed code candidate without mocking Git identity or cleanliness.

N1: each public source records its declared identifier and location: MIT in the pinned llama.cpp root LICENSE, apache-2.0 in oasst1 README YAML, cdla-sharing-1.0 in TinyStories README YAML. GPU retrieval saves LICENSE or license-metadata.md beside the document, verifies the declaration and records its actual SHA256/size in license.json and the derived document identity. The MIT expected hash comes from the existing pinned local Git object. Dataset README hashes are first recorded at GPU retrieval. No offline fixture establishes remote availability or legal suitability.

## Validation

Pending native rebuild and full suite from the clean signed code candidate. Preliminary contract run: 21 tests, exit 0, four native skips; this is not the full-depth gate. Shared-stage suite: 19 tests, exit 0 (`.temp/r47c/tests-stages-02.log`). An initial wrong discovery directory ran zero tests (`tests-stages-01.log`); it is not passing coverage. Rebuilt read-only default plan: exit 0 (`stage-plan-01.log`), factor 4 at both depths. Planning remains 512k three documents 1.58-2.41 GPU hours / $2.49-$3.79, 1M two documents 2-3 hours / $3.14-$4.71 at assumed $1.57/hour, excluding setup and other documented overhead.

All 17 prior depth tests are retained under their original names; four added tests cover the factor boundary, missing/null engine refs and licence provenance/declaration refusal. Changed review surfaces: 1b (F1), 4c (F2), 3b (N1). No new surface table or additional landing property suite was supplied; that is a brief gap. Other accepted surfaces are unchanged. Changes stay inside tools/hybrid-bench; no core/backend or kit changes. Target CUDA/Metal/27B/deep quality, real public retrieval and timing remain unrun. Independent review is pending.

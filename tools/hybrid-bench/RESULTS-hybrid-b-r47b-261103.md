# Depth quality refinements: r47b CPU handoff

Ready for review. Continue `m3-depth-quality` from signed r47 head `773faa4cb19fe677433ca6877c79853d4dd840d1`. No push, GPU work, model download or corpus download on agents1. Target RTX/M3 execution remains unrun.

## Changed behavior

The GPU stage now uses three public input types at 512k: source from a large public project, a public chat dataset and narrative prose. At 1M it defaults to code/chat, with `--million-documents 3` adding prose. `depth_corpus_pins.json` pins every raw input by immutable URL/revision, exact size and SHA256. Code pins were computed from existing local public upstream Git objects at `e0f142d5ddc1461ce721644d92af33db23545105`, whose public commit was verified. OpenAssistant and TinyStories pins came from public distributor file/LFS metadata, without fetching corpus content. The OpenAssistant revision is `fdf72ae0827c1cda404aff25b6603abec9e3399b`; TinyStories revision is `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. TinyStories is synthetic prose, not human-authored literary evidence. No owner/private transcripts are inputs.

`depth_corpus.py` verifies all raw bytes on the GPU host before constructing contexts. It records derived text hashes and transformation identity, selects deterministic English conversation paths and quotes special-token markers. The source/type/transform identity is part of the state binding. A type is one constructed document per depth; correlated positions, needles, cyclic repetition and reused source across depths are explicitly not extra independent observations. `document-observations.json` retains document-level rows without pooled-position confidence intervals.

Every report now has per-arm needle count/denominator/accuracy, explicit `local_needle_accuracy`, and an all-miss flag. If all arms miss every needle, the report says agreement provides no retrieval-quality evidence and that the comparison rests on KL/other logits. Default sequential needle chat restores once per arm, then carries each previous answer into the next turn. The independent protocol remains available and separately labelled. Context allocation reserves the query chain and refuses an insufficient budget before arm work. Fixture v2 binds the protocol and fixed 64-token reference continuation; old v1 inputs must be regenerated.

Every arm also writes a scalar64 post-context reference in dev-a's raw layout: contiguous row-major full-vocabulary float32 little-endian rows after forced inputs `D..D+63`, predicting `D+1..D+64`. The new `depth-slot-handoff-1` envelope binds state/sidecar/fixture/reference byte counts and SHA256, exact settings/continuation/positions, and producer engine/binary identity. It is not misrepresented as dev-a's rigid two-slot, fixed-engine/factor-four `slot-handoff-1` manifest. A synchronous `--handoff-command` receives the exact saved state before deletion, so transport can finish without keeping a second large state on the GPU disk. Without that callback, the state is ephemeral and the archived manifest alone cannot restore it.

The same native helper's probe mode and `depth_quality.py --handoff ...` verify a transferred slot on the Mac using the recorded settings. Engine/source, model, state, sidecar, fixture, continuation, reference shape/hash and position conventions are checked before replay. CUDA-vs-Metal acceptance is not inferred: output is descriptive KL/top-1/max-difference. Dev-a kit source was inspected read-only at the head recorded in `.temp/r47b/kit-head.txt`; no kit files were modified. See the [shared format and exact commands](README.md#mac-slot-verification-hook).

## Time and resource plan

At 1M, the analyst's planning envelope is now three 15-20 minute prefills plus 15-30 minutes of checks/state I/O per document: 1-1.5 GPU hours per document. The old under-one-hour-per-pair target is retired. These are assumptions, not measured RTX rates. At the illustrative, unverified $1.57/hour:

| Depth | Types/documents | GPU hours | Cost assumption |
|---|---:|---:|---:|
| 512k | 3 | 1.58-2.41 | $2.49-$3.79 |
| 1.04M default | 2 | 2-3 | $3.14-$4.71 |
| 1.04M optional | 3 | 3-4.5 | $4.71-$7.07 |

512k scales the prefill term by depth and retains a 60% floor on check overhead. Setup, downloads, build, remote handoff, archive and billing granularity add time. More than 4096 scoring positions/16 needles or slow storage needs recalibration. Independent probes add repeated snapshot reads; after-window mode also requires one extra reference restore. The default five cases need approximately 154 GB initially free disk, including the 28 GB floor, approximately 62 GB raw logits/references, the model and one state file. Host RAM remains at least 120 GB with an available-memory admission gate; GPU is one 96 GB RTX PRO 6000.

## CPU evidence

Evidence root: `.temp/hybrid-b-r47-261103/worktree/.temp/r47b/` in the workbench. Source/binary hashes are in `source-binary-identity.json`; native commands, owned PIDs and real exits are under `native-02/`. The decisive local reference/Mac-style candidate pair and all final-window arm logits remain available. Regenerable extra CPU raw arrays were hashed before deletion to protect the disk floor (`native-01-raw-hashes.json`, `native-02-raw-identity.json`); logs and computed results remain. Missing raw files are not claimed as an intact transferable slot bundle. The owned build was removed; `build-cleanup.json` retains its binary SHA and final free space.

| Validation | Actual exit | Bounded claim |
|---|---:|---|
| CPU configure and native build | 0 / 0 | `build-driver-01.log`, `configure-01.log`, `build-01.log`; existing CMake 4.4.4, no CUDA/UI/download, `nice -n 19`, `-j4`, compiler-idle check |
| Initial depth suite | 0 | `tests-native-01.log`; 16 tests before additional protocol controls |
| Final depth suite, native enabled | 0 | `tests-native-02.log`; 17 tests, 35.733 s, no skips |
| Existing shared-stage regression | 0 | `tests-stages-01.log`; 19 tests |
| Read-only default/three-type plans | 0 / 0 | `stage-plan-02.log`, `stage-plan-three-01.log`; no GPU or downloads |
| Scoped whitespace/ASCII/JSON/Python3.10 grammar checks | 0 | Local static checks and `git diff --check` |

The small local Qwen3.5-0.8B Q4_0 fixture is the CPU-only model exception. At D512/final8/K2, all arms retrieve 2/2 needles; local accuracy is 1.0. Offload/local logits are bit-identical in those eight rows (KL0, max0, top1 8/8). Partition/local KL is 0.000150417994 nats, max difference 0.588937283, top1 8/8. Sequential native receipts record exactly one restore per arm. The separate independent after-window case records three restores per arm (reference plus two questions).

A real saved-slot scalar64 replay produces identical reference/candidate logits: KL0, max0, top1 64/64. The original slot bytes remain unchanged. An actual owned callback executable sees the state present for every arm before deletion; final state paths are absent afterward. This is CPU mechanics evidence, not Metal or deep-quality qualification.

Named negative controls retain valid positive controls: native document/YaRN mismatches exit2 with the intended reason, unchanged state/sidecar and no logits/result; handoff document/YaRN/engine/state/reference/continuation/position mutations raise the intended refusal before any native launch; private/local corpus URLs, mutable revisions and absent hashes refuse; a generated retrieval fixture with wrong raw bytes refuses its SHA/size gate without a derived corpus manifest; the all-zero needle fixture emits the KL-only caveat; an undersized sequential context refuses. Known-distribution KL/top-1 tests, archive checksum control and negative scoring cases remain green. Corpus tests mock retrieval of generated fixtures; real public data conversion remains GPU-host execution, not a claimed agents1 download.

Self-review followed the repository code-review skill (scope, security, design, general, tests). No core importer/backend or dev-a kit changes. The root README tool entry is the only change outside `tools/hybrid-bench`. No surface table or additional landing property suite was supplied; that is an explicit brief gap. Current type/editorconfig readiness commands both exit127 (tools absent); no installation was attempted and those full checks remain unrun. Independent review, target-GPU corpus/model checks, CUDA/Metal state portability, and 512k/1M quality remain open.

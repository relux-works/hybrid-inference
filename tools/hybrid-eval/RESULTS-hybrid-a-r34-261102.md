# R34 paired RULER CPU integration

Branch: `m3-ruler-eval`, base `fca0fe31d`. No push, PR or GPU execution is part of this delivery. The protocol and operator commands are in [RULER-RUNBOOK.md](RULER-RUNBOOK.md).

The integration vendors NVIDIA/RULER c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a, its licence, exact task configurations/templates and scoring rules. The adapter executes upstream candidate generators with deterministic bounded length fitting against the model's own rendered-prompt tokenizer. All 13 configurations are present. Full input depth is exact, with no context/query truncation. There are three arms, shared frozen examples, shuffled arm order, homogeneous model admission, actual prompt-token hash checks and per-arm routing/release checks.

Default design: 390 examples/depth, 30 per configuration, at 8192/32768/65536/131072 input tokens. Exact McNemar planning gives 80% power for about 5.78 percentage points excess when partition-only discordance is .05, or 7.12 points at the conservative .0125 first threshold for four-depth Holm. Changed-answer rates receive Wilson intervals; the paired excess has a conservative exact-cell 95% interval. Literal upper <=0, practical .02 noninferiority and two-sided .02 equivalence are separate outputs. The runbook prices 11.6..46.3 GPU-hours using declared effective throughput assumptions and 50% contingency; this is not a measured GPU estimate.

Bootstrap runs only on the GPU host with explicit opt-in. Every raw response gets its requested URL, resolved public URL, byte count and SHA-256; static QA/vocabulary/Punkt digests are checked. The live essay corpus is frozen by aggregate SHA-256 at bootstrap and the evaluator requires that exact input-lock hash. Neither real benchmark data nor another model was downloaded on agents1. Independent replication must reuse the frozen input directory/lock, rather than scrape the blog again.

Each depth journals frozen specifications and complete paired rows atomically, rejects changed resume identity, locks against concurrent writers and creates a complete archive only at the planned denominator. CPU tests exercise interruption before a pair commit, recovery without regeneration, completed-depth skipping, changed-identity rejection, archive-member hashes and incomplete-depth refusal.

## CPU evidence

Artifacts are in the worktree's `.temp/`; services record their owned PIDs and cleanup. The C++ source is unchanged from fca0fe31d. A fresh CPU build was configured with CUDA/Metal off and compiled using `nice -n 10`, `-j4`, targets `llama-server` and `llama-hybrid-prefill`. Configure/build exit 0. `.temp/build-01.json` records PID, command and minimum sampled free disk, 32184549376 bytes. The model is the existing Qwen3.5-0.8B-Q4_0 GGUF, SHA-256 `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`.

| Check | Exit | Evidence |
|---|---:|---|
| Existing eval unit suite, 11 tests | 0 | `.temp/eval-unit-02.log` |
| RULER unit suite, 8 tests | 0 | `.temp/ruler-unit-06.log` |
| Data-lock/bootstrap controls after mandatory English Punkt lock | 0 | `.temp/data-unit-02.log` |
| Resume identity control after Python/source binding update | 0 | `.temp/resume-unit-03.log` |
| Existing proxy journal suite plus continuation regression, 18 tests | 0 | `.temp/journal-unit-01.log` |
| Template suite, 43 tests, 5 optional external/native oracle skips | 0 | `.temp/template-unit-01.log` |
| New RULER module type check, ty 0.0.88 | 0 | `.temp/type-check-06.log` |
| Python bytecode build into task scratch | 0 | `.temp/python-build-02.log` |
| Native five-task eval plus RULER extension, 13 generators and three-arm needle | 0 | `.temp/native-07.log` |

The native extension checks all 13 generators twice using the actual GGUF tokenizer at 2048 tokens (4096 CWE), named wrong-answer rejection with valid controls, and one real RULER needle across all three arms. Local corpus fixtures use untrained Punkt only on abbreviation-free text. This is integration evidence, not production-corpus or long-context quality evidence.

The only runtime scope extension outside hybrid-eval is a one-line `Renderer.family` correction in `tools/hybrid-proxy/journal.py`. A synthetic complete-turn family probe must disable the real request's final-message continuation flag. The actual request's settings remain intact. The regression is in the existing `test_journal.py`; it was run red before the correction and green afterwards.

| Narrowing control | Gate narrowed to | Named failing test | Survival bound |
|---|---|---|---|
| Original family probe inherits continuation, equivalent to removing the one-line fix | Complete-turn template probes cannot inherit assistant-prefix continuation | `JournalTests.test_family_probe_disables_final_message_continuation` | Killed, expected exit 1 in `.temp/continuation-red-01.log`; no surviving mutant is claimed |

## Corrected controls and limits

Native attempts are preserved rather than counted as passing: native-01 used incompatible older binaries; native-02/03 exposed BPE whitespace padding gaps; native-04 exposed the old evaluator replacing its public metadata snapshot over endpoint metadata; native-05 exposed continuation-probe rejection; native-06 passed routing and 11 generators but rejected an oversized QA fixture minimum. The final adapter searches whitespace with actual tokenizer feedback, the native extension retains original endpoint metadata, and QA's minimum is its supporting documents, allowing upstream distractor selection to fit the remaining space.

The first dependency install attempted a NumPy source build on Python 3.14; its own PID tree was stopped. Corrected installs use binary wheels only, pinned in `ruler-requirements.txt`. No project native build ran concurrently with another observed cc1plus process. CMake was absent from PATH; the lane's existing CMake was verified and reused. All test processes used disk TMPDIR and PYTHONDONTWRITEBYTECODE=1. An accidental trailing shell assignment made the first bytecode-validation shell exit 127; the standalone validation was rerun with exit 0 as recorded above.

Wildcard eval test discovery encounters the repository's duplicate `test_native` module names in hybrid-bench and hybrid-eval and fails before executing the suite. Documented per-file discovery commands run green. Including the existing journal module in ty reports its unchanged `MutableMapping.setdefault` diagnostic at line 303; the same diagnostic reproduces on the untouched fca0fe31d source in `.temp/type-baseline-01.log`. The new modules type-check; repository-wide type-check success is not claimed.

No surface table was supplied in the brief. Coverage is limited to the requested generator/scorer/analysis/routing/resume contracts; no exhaustive attack-surface claim or independent reviewer verdict is made. GPU corpus download/hash verification, long-context model quality, effective throughput, aggregate VRAM admission and hardware-specific numerical behavior remain unrun operator-host work. Production data, GPU behavior and venue acceptance are outside this CPU evidence. The practical equivalence margin is explicitly .02; it must not be relabelled as proof of the literal zero-margin statement.

The admitted native run is `.temp/native-07`: five existing tasks x five treatments; 13 deterministic real-tokenizer generator checks; one three-arm 2048-token RULER needle with 2047 imported offload tokens, identical prompt-token hashes and zero local imports. Cleanup readback reports both decoders/proxies and the prefill node stopped, with zero retained remote sessions before stop. Final observed agents1 free disk was 29848883200 bytes, above 28 GB.

Full `git diff --check fca0fe31d..HEAD` exits 2 on 14 trailing-whitespace lines in the unchanged NVIDIA vendor copy (`.temp/full-whitespace-01.log`). The same check excluding `tools/hybrid-eval/vendor/ruler/**` exits 0 (`.temp/owned-whitespace-01.log`). Vendor source hashes verify against SOURCE.json. Those bytes are preserved to keep the upstream source pin exact; full whitespace-check success is not claimed.

## R35 review revision (hybrid-a-r35-261103)

Reworks Muse r7ad3's CHANGES REQUESTED verdict on 330eb251. Ready for review; no push or GPU execution. F1 now calls the pinned upstream 500-token VT ICL builder, appends its worked answer, randomizes it and inserts it before the main task exactly as upstream does. The example is included in rendered length fitting. F2 estimates tokens per size unit from a small rendered candidate and uses upstream's 3x upper-bound slack instead of depth documents/repetitions. Vendor bytes remain unchanged.

`generation_seconds` is frozen with each spec and retained in the paired row, including initialization for the first example of a task. Timing is measured by the evaluator, so direct generator determinism remains testable. The runbook now gives per-depth QA/noise pilots and an inference-plus-preparation budget formula, exact owned-service stop/preserve/remove-state-path/relaunch commands, failed-bootstrap recovery, restart verification costs, near-zero-discordance equivalence limits, and the predeclared 20/task (n=260) option at 128k. Partial-depth analysis labels 4xp as Bonferroni; joint Holm is emitted only for all four depths.

R35 evidence is in `.temp/hybrid-a-r35-261103/` in this worktree. All commands use disk TMPDIR, disabled bytecode and niced CPU execution. No native rebuild or downloads. Free disk stayed above 28 decimal GB (28.826 GB at start, 28.825 GB after pilot); this is not 28 GiB.

| Check | Exit | Evidence |
|---|---:|---|
| RULER, 11 tests | 0 | `ruler-unit-04.log` |
| Existing eval, 11 tests | 0 | `eval-unit-01.log` |
| Journal, 18 tests | 0 | `journal-unit-01.log` |
| Templates, 43 tests, 5 optional skips | 0 | `templates-unit-01.log` |
| Changed Python modules, ty 0.0.88 | 0 | `type-check-01.log` |
| Real GGUF tokenizer preparation pilot, 8k and 32k | 0 | `pilot-01.log`, `pilot.json`, `pilot-cleanup.json` |
| Task diff whitespace | 0 | `git diff --check` |

Earlier unit attempts `ruler-unit-01/02.log` were interrupted/truncated, not passing evidence. Diagnostic attempt 03 was terminated by its recorded PID after showing repeated construction/sorting of upstream's adjective/noun Cartesian product. Tests now use a bounded fixture vocabulary (300 words) for the fixture-only generation tests; the production generator and real-tokenizer pilot retain the actual wordlists. The admitted standalone requested unittest command is attempt 04, 11 tests in 4.258s. Native inference was not rerun for this Python-only rework; r34/reviewer native evidence remains historical, not candidate inference qualification.

| Mutant | Gate narrowed to | Named failing test | Result / surviving bound |
|---|---|---|---|
| Remove VT worked-example insertion | Upstream worked input, answer and placement | `test_vt_worked_example_matches_upstream` | exit 1, assertion differs from upstream; `mutant-F1-01.log`; killed |
| Restore initial hi=depth | Bounded tokenizer request size | `test_all_generators_bounded_render_work` | exit 1, 311411 tokens exceeds 32768 bound at 8k; `mutant-F2-01.log`; killed |

The bound test exercises all 13 configurations at 8192 and 32768 with local corpus/vocabulary and word/whitespace tokens: largest render <=4x depth, total rendered <=30x depth, exact final depth and valid/wrong-answer controls. These are fixture bounds, not a universal bound for arbitrary corpora/tokenizers. Both narrowing mutants run in memory, leaving source unchanged. No surviving mutant is claimed killed.

The CPU pilot used the existing Qwen3.5-0.8B-Q4_0 and fca0fe31d server with actual `/apply-template` and `/tokenize`, local fixture corpora, actual wordlists and no inference. It recorded per-example preparation including initialization:

| Task | 8192 seconds | 32768 seconds |
|---|---:|---:|
| niah_single_1 | 3.735 | 3.978 |
| qa_1 | 0.105 | 0.362 |
| qa_2 | 0.158 | 0.338 |
| vt | 0.136 | 0.358 |

Largest real-tokenizer render was 2.20x depth; maximum total rendered was 17.57x depth. Every final prompt reached exact depth; the owned server exited 0 after termination and no session state was created. Production corpus QA at 32k+, GPU prefill/decode, model quality, aggregate VRAM and deep throughput remain unverified.

Updated budget scenarios: inference at flat R=2500..10000 tok/s with 50% contingency is 11.6..46.3 hours for n=390 at every depth, or 9.45..37.81 hours with n=260 only at 128k. To include preparation explicitly, one illustrative extrapolation uses mean CPU pilot G=1.034 s at 8k and 1.259 s at 32k, then scales G linearly to 2.518/5.035 s at 64k/128k. This adds 1.60 hours (full) or 1.33 hours (budget), including 50% contingency: **13.2..47.9 hours full; 10.8..39.1 hours with n=260 at 128k**. Those totals are planning scenarios, not GPU measurements or upper bounds; setup/restart/archive costs are additional. Replace G and arm times with the GPU-host pilots before spending on each deeper depth.

At n=390, practical 2 pp equivalence already fails at b=c=2 (upper 2.02 pp). At n=260, zero discordance gives 1.67 pp and b=c=1 fails; detectable excess is 7.45/9.21 pp (alpha .05/.0125). n=195 cannot establish 2 pp equivalence even at zero discordance (2.22 pp). No post-data margin/denominator changes are authorized.

Coverage map for review rows: row 1 -> upstream VT comparison, all-generator determinism/scoring and bounded-render tests, killed F1/F2 controls; row 2/3 -> paired-table/interval and partial/joint adjustment tests (full statistical requalification is outside this focused delta); row 4 -> measured CPU preparation pilot plus inspected operator commands, GPU throughput explicitly unverified; row 5 -> suites above. No prior reviewer tests were removed. Changes stay inside tools/hybrid-eval. GPU hardware/data/VRAM, upstream byte-identical sample sequences and universal model quality are out of contract under the brief's CPU-only/no-download rules and the disclosed adapter protocol.

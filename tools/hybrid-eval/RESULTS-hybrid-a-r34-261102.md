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

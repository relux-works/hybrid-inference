# hybrid-d-r1-261101: r6x bridge rework

Ready for review on `m3-vllm-bridge`, parent `876c9019ad1fa092d543cca334d71020df7e424f`, native base `a3752b04cbd905bf61ffe6d9a4e7df738f26f4c2`. All seven r6x findings have CPU-scope fixes. GPU qualification and acceptance by the independent reviewer remain pending. No push, PR, model/source download or GPU operation was performed.

## Findings

| r6x finding | Status and change | Named test or check |
|---|---|---|
| 1: layout-blind fixtures | Fixed. Each attention and recurrent layer has separate tensor files. Seeded BF16 KV varies by layer/head/token/KV; f32 recurrent sentinels vary by layer/head/V/K and conv channel/time. Serialized bytes are checked against an explicit Sylvester Hadamard matrix and a separately written f32/q8_0 reference, plus an independently enumerated grouped-to-tiled recurrent mapping. The converter math is unchanged. | `test_serialized_layout_reference`; `test_paged_kv_layout_and_scales`; all four layout survivors now fail assertions |
| 2: startup and GPU ownership | Fixed runbook. Both servers have bounded health polling with PID liveness checks. vLLM's recorded PID is stopped and waited before any conversion, checker or client; client cleanup also stops its PID. | `runbook-check-02.log`: retry, timeout, exited-server and owned shutdown probes |
| 3: missing Q4_K_M leg | Fixed runbook. Exact BF16 and Q4_K_M invocations recompute each target's identity/hash, convert into separate delta directories and write separate checker fixtures; disk estimates include both retained delta sets. Neither leg takes its target from the capture config. | `targets-check-01.log`: both real codec envelopes carry distinct synthetic target hashes; a mismatched layout exits 1 with the intended refusal |
| 4: permanent runner patch | Fixed. Capture runs in a subshell with an EXIT trap that stops vLLM, restores and byte-compares the runner backup and unsets the hook config. Matching patch reruns preserve bytes; altered source, changed line endings or moved hooks refuse. | `test_patch_location`; success/failure restore probes in `runbook-check-02.log` |
| 5: partial capture on refusal | Fixed. A private TemporaryDirectory stages tensors and manifest; validation precedes rename to the final capture name. Exceptions remove staged files. Existing captures refuse replacement. | `test_export_refusal_cleanup`: late ownership refusal after files were written, final validation refusal, retry, successful publication and existing-output preservation |
| 6: claim overstatements | Fixed. Synthetic config, missing-layer and minimum-block claims now match their assertions. Native permutation claim removed; the historical report now attributes permutation evidence to Python assertions and links this rework. The expanded FP8 test independently checks every page/token/head/KV row. | Test comment self-review; `test_paged_kv_layout_and_scales`; `test_native_roundtrip` claims serialization only |
| 7: conversion estimate omitted | Fixed runbook. States the reviewer's measured 38.5 ms/token and per-target ladder estimates: 10 min at 15680, 42 min at 65072, 160 min (2.67 h) at 249312, excluding fixed work and I/O. Describes compiled/vectorized kernels, exact rounding and bounded parallelism needed for a faster converter. | Documentation/arithmetic check; no new throughput claim or faster converter |

The metadata notes are also covered: the byte-reference test asserts every cell's position, x/y, token ID and reserved field. All 16 original test names remain, with two new tests. Source layout is checked by independent Python expectations; native acceptance alone remains a serialization check.

## CPU evidence

Evidence root: `.temp/hybrid-d-r1-261101/`. Evidence snapshot on 2026-11-01 at 20:00 UTC, Linux x86_64, Python 3.14.4, GCC 15.2.0. Exact source/native-helper hashes, environment and exits are in `validation.json`; per-plant source hashes and edits are in `controls-final.json`. Tests ran niced with disk TMPDIR and `PYTHONDONTWRITEBYTECODE=1`. No command was piped through tee.

| Command/check | Actual exit | Evidence and limit |
|---|---:|---|
| First native helper link | 0 | `build-native-01.log`; preliminary link used the neighbor's vendor hash archive |
| Compile this branch's SHA-256 C source, then relink helper | 0 / 0 | `build-sha256-01.log`, `build-native-02.log`; removes dependence on the neighbor's newer SHA-256 implementation |
| First 18-test CPU run | 0 | `tests-01.log`, 32.846 s; preliminary helper and earlier fixture revision, superseded by the next run |
| Current 18-test CPU run | 0 | `tests-03.log`, 23.607 s; includes mandatory native BF16/f32 import, FP8/BF16 append and checksum rejection control |
| Unmodified copied-source control, all 18 tests | 0 | `control-identity-02.log`, 23.661 s; same final Python modules and relinked helper |
| r6x plant driver: original 11 plus reviewer 11 | 0 | `controls-driver-02.log`, `controls-final.json`; all 22 child exits 1, named assertion failures, zero errors or survivors |
| Pre-fix exporter plant | 0 | `export-control-02.log`, `export-control-final.json`; child exit 1 at the no-partial-output assertion |
| Runbook Bash/Python syntax, ASCII and shell probes | 0 | `runbook-check-02.log`; all three Bash blocks parse, all six embedded Python blocks parse; retry, timeout, exited server and restoration on exits 0/7 exercised with CPU doubles and owned sleep PIDs |
| Target parameterization probe | 0 | `targets-check-01.log`; real delta conversion with synthetic GGUF identity reader, two target file hashes; layout mismatch child exit 1, no checker fixture published |
| Whitespace check | 0 | `diff-check-03.log`; `git diff --check` |
| Python type checker | 127 | `typecheck-01.log`: `ty` is unavailable. No type-check pass is claimed; no tools installed. Syntax/runtime tests are separate evidence. |
| Live vLLM capture, real BF16/Q4_K_M GGUF, CUDA/Metal, logits, throughput/RSS/VRAM | Unrun | Requires the orchestrator's GPU/Mac experiment |

Native helper provenance: linked this checkout's `native_check.cpp`, `hybrid-state.cpp` and freshly compiled `vendor/hash/sha256/sha256.c` against existing CPU libraries under `/home/agent/Werkstatt/hybrid-a/llama.cpp/build`. `reused-source-diff-01.log` is empty for src/ggml/include/common and vendor XXHash relative to the reviewed head. Library hashes are in `native-library-hashes-01.log` (`libllama.so.0.8.0`: `84128f68790f1cf122c786a1a55ca140d9ac1a97415f3d3d523b437e43d2b63e`). No engine build tree was created. Only one niced compiler ran at a time. Exact build commands and exits are in `native-build-commands.json`. Test fixtures and temporary deltas were removed by their contexts; scratch compilation objects/helper are removed after validation. Observed free disk stayed above 25 GB.

Reproduce the CPU suite with the README helper-build command and `python3 -m unittest discover -s tools/hybrid-vllm -p 'test_*.py' -v`. The task-local `controls.py` reuses all literal r6x plants and selects the named test per plant; its only anchor adaptation is the runner hash expression changing from `path.read_bytes()` to the unpatched `original.encode()`. Each plant edits a separate copy, requires exactly one matching anchor, and only a named AssertionError with child exit 1 and no unittest errors qualifies. The unmodified copy runs the complete suite. No setup error is credited as a kill.

## Plant results

All exits below are the actual unittest child exits. Driver exit 0 means the expected control outcomes were observed, not that the mutants passed.

| Mutant | What it narrows the gate to | Named failing test | Actual exit / result |
|---|---|---|---|
| p_layer_count | Exactly 64 distinct captured layer IDs | `test_wrong_layer_count` | 1, killed |
| p_block_size | Minimum/multiple-of-16 size despite consistent page shape | `test_wrong_block_size` | 1, killed |
| p_head_geometry | 256-dimensional heads despite matching fingerprint | `test_wrong_head_geometry` | 1, killed |
| p_kv_dtype | Explicit KV dtype whitelist | `test_unsupported_dtype` | 1, killed |
| p_partial_endpoint | Whole-block exporter endpoint | `test_partial_last_block` | 1, killed |
| p_kv_page_order | Logical page order | `test_paged_kv_layout_and_scales` | 1, killed |
| p_fp8_scale | Actual per-K/V dequantization scales | `test_paged_kv_layout_and_scales` | 1, killed |
| p_gdn_heads | Grouped-to-tiled state head order | `test_recurrent_head_and_conv_layout` | 1, killed |
| p_rotation | Actual Hadamard operation | `test_q8_and_rotation` | 1, killed |
| p_async_schedule | Synchronous scheduling requirement | `test_source_config` | 1, killed |
| p_runner_hash | Exact original runner hash | `test_patch_location` | 1, killed |
| r_kv_swap | K and V source halves | `test_paged_kv_layout_and_scales` | 1, killed |
| r_conv_time_reversal | Oldest-to-newest conv history | `test_recurrent_head_and_conv_layout` | 1, killed |
| r_state_transpose | Preserve [V,K] within each state head | `test_recurrent_head_and_conv_layout` | 1, killed |
| r_kv_head_order | Attention head order | `test_paged_kv_layout_and_scales` | 1, killed |
| r_fp8_scale_swap | K scale for K, V scale for V | `test_paged_kv_layout_and_scales` | 1, killed |
| r_rotation_kv_swap | Serialized K uses 256, V uses 64 | `test_serialized_layout_reference` | 1, killed |
| r_attention_layer_order | Serialized attention layer identity/order | `test_serialized_layout_reference` | 1, killed |
| r_recurrent_layer_order | Serialized conv/state layer identity/order | `test_serialized_layout_reference` | 1, killed |
| r_token_order_in_page | Serialized token order within each page | `test_serialized_layout_reference` | 1, killed |
| r_cell_ext_xy | Text x/y positions follow the cell position | `test_serialized_layout_reference` | 1, killed |
| r_cell_ext_token | Cell token ID follows the journal | `test_serialized_layout_reference` | 1, killed |
| e_export_refusal_cleanup | Refusal after writes leaves no partial capture | `test_export_refusal_cleanup` | 1, killed by restoring the pre-fix exporter |
| identity | Admit the unchanged source/fixture/environment | All 18 tests | 0, survives as required; bounded to this candidate and environment |

This finite set establishes detection of these plants only. The matrix reference uses exactly representable dyadic BF16 inputs, so its agreement does not establish arbitrary floating-point input equivalence. The cleanup test's two-layer CPU doubles intentionally bypass final validation only for the publication positive control; real validation is exercised by the preceding refusal. They do not validate live cache extraction. No formal surface table was supplied; this remains a brief gap. There are no out-of-contract review findings in the rework. No `repeat-of:` class was assigned by r6x.

## Remaining GPU gates and scope

- Actual vLLM 0.32.0 wheel hash, CUDA views, manager/kernel page gather, align endpoint ownership, SD/DS state chronology and live FP8 scales remain unverified. Source provenance from r6x is reused, not re-established by downloading vLLM.
- Real HF/GGUF vocabulary equality, tokenizer/checkpoint provenance and both receiving GGUF hashes must be recorded on the GPU host. The capture copies target layout from config; source admission relies on `check_config`, not on that copied-layout equality.
- Both BF16 and Q4_K_M must run the checker and actual HTTP slot import, then the intended Mac/Metal path. Quality bounds are uncalibrated; checker exit 5 remains failing/uncalibrated evidence, not a pass. The comparison begins at N+1 after a common probe; logits at N are not claimed.
- Full E5M2 conversion, warm-prefix ownership, long-depth memory capacity and end-to-end latency still need qualification. About 2.7 hours per target at 249k is the reviewer's CPU extrapolation, not a new performance result.
- Self-review caught potential line-ending normalization in the patch hash check. Reading raw bytes and the CRLF refusal regression preserve the original exact-byte guard; the full suite and all plants were rerun after that fix.
- The runner restore trap cannot survive SIGKILL or host failure; DESIGN retains the original backup and gives the recovery requirement.

The diff is confined to `tools/hybrid-vllm/`: fixtures/reference tests, capture publication, patch idempotence, runbook, README and result/qualification notes. No native readers, model code, server scheduling, converter math or protocol limits changed. Self-review covered scope, security, tests, claim accuracy and general conventions. Task-board is unavailable in this workbench (`board-goal-01.log`, exit 1, missing `.task-board`); the existing `.temp/tasks.md` tracks this rework. No independent acceptance, landing, hosted checks or goal clear is claimed. Signed local commits and the requested team summary are the authorized handoff; no push is authorized.

## r2: hybrid-d-r2-261101, GPU runbook preflight

Ready for review on `m3-vllm-bridge`, parent `6ed84b9c034c9d31b866d894230fba2c393d16aa`. Claude r7b accepted that r1 head; that acceptance does not cover this new diff. Scope remains `tools/hybrid-vllm/`. CPU only, no GPU operations or model downloads, no push.

| Request | Change and evidence |
|---|---|
| Finding 2: interactive errexit and hidden failures | `gpu-run.sh` runs as a child Bash process, refuses sourcing before changing shell options, and has gather/build/capture/convert/check/import/all subcommands. Logs contain commands and actual exits; failures print the command, exit, log path and tail. `test_hidden_command_exit` observes child exit 9 and a surviving caller. `test_capture_cleanup` observes HTTP exit 22 and TERM exit 143, restored runner bytes and a stopped owned dummy server. |
| Finding 3: fresh clone and hardcoded RUN_DIR | Creates parents; takes the second argument before RUN_DIR environment, with a task-scoped default. `test_build_and_paths` exercises absent nested parents, spaces and argument precedence. Disk TMPDIR and disabled bytecode are retained. |
| Finding 4: missing exact-ref CUDA build | Requires a full reviewed EXPECTED_REF, equal HEAD and clean tracked files before/after a fresh CUDA build. Targets are `llama-server llama-hybrid-bench-check`; saves source/build refs and SHA-256 hashes, verified before capture/check/import. Defaults: CUDA 12.8, `120a-real`, niced build with <=4 jobs. `test_build_refusals` and `test_binary_tamper` cover stale refs, dirty source, invalid jobs, existing builds and changed binaries. Actual CUDA build remains unrun on agents1. |
| N1: exporter page gather | Extracts the existing writer without changing its mapping. `gather_test.py` calls that production writer with real torch CPU BF16/uint8 tensors, contiguous/strided views, ratios 1/2/3, non-monotonic block tables and full/append intervals. Independently enumerated scalar coordinates define every expected byte. This is the script's first gate, repeated before every subcommand; failure stops before any build or live process. `test_page_bytes` kills all four ordering plants below. |
| Finding 1, optional: rotated FP8 | `test_fp8_rotated_bytes` checks every q8_0 row for independent E4M3FN decode, separate non-unit K/V scales, explicit Hadamard matrices (256/64) and reference rounding. It kills the review's scale-under-rotation plant. |
| Host defaults | Uses `/root/kit161/gpu/work/vllm-venv`, HF cache under `/root/kit161/gpu/work/hf`, and both named GGUFs under `/root/proto/models`. Cached model auto-selection checks both HF_HOME/hub and HF_HOME; ambiguity refuses. Environment overrides, offline resolution, default token journal creation and installed-version recording are documented. |

Evidence root: `.temp/hybrid-d-r2-261101/`; commands, environment, source hashes and actual exits are recorded in `validation.json`, with plant edits and hashes in `controls-final.json`. Linux x86_64, Python 3.14.4, GCC 15.2.0. All test runs use disk TMPDIR and disabled bytecode; suite and plant commands run at nice 10. No gate command is piped through tee.

| Validation | Actual exit | Evidence and limit |
|---|---:|---|
| Compile own SHA-256 object; link native helper against pinned CPU libraries | 0 / 0 | `build-sha256-01.log`, `build-native-01.log`, `native-libraries.json`. No engine rebuild. Copies match r1 library hashes; `git diff 41eb6ca4 HEAD -- src ggml include common` is empty. Native helper sources and archive provenance are recorded. |
| Current stdlib/native suite, 29 tests | 0 | `tests-02.log`, 27.734 s. Includes all original 18 names, new FP8 bytes and ten script/data tests. Shell command doubles exercise orchestration; they do not qualify CUDA, vLLM or real GGUFs. |
| Torch-CPU gather, two tests; actual script gather subcommand | 0 / 0 | `control-identity-02.log`, `script-gather-02.log` and its linked gather log. Torch 2.13.0+cpu, NumPy 2.5.4; no CUDA build or initialization. Same torch release as the host, CPU wheel suffix differs. |
| Five bounded plants plus unchanged gather control | 0 driver | `controls-02.log`, `controls-final.json`: control exit 0; five child exits 1 at named assertions, no errors. |
| Both target envelopes with the real codec and synthetic GGUF identities | 0 driver | `targets-check-01.log`: distinct BF16/Q4_K_M hashes in fixtures and envelopes. Mismatched layout child exit 1 with the intended refusal; no checker fixture published. |
| Real all-subcommand invocation without torch | 1, expected failure | `no-torch-01.log`: ModuleNotFoundError at the first gather gate; failing command/exit visible, no CUDA build or server launch. |
| Bash syntax, Python/heredoc AST, ASCII and whitespace | 0 | `bash-syntax-02.log`, `source-check-01.log`, `diff-check-02.log`. |
| Python type checker | 127 | `typecheck-01.log`: ty unavailable; no type-check pass claimed. |
| Live capture, exact CUDA build, BF16/Q4_K_M model runs, HTTP import and logits | Unrun | Reserved for the orchestrator's GPU host. CPU command doubles are not live qualification. |

| Mutant | What it narrows the gate to | Named failing test | Result |
|---|---|---|---|
| kernel_interleave | Restore head/kernel-page axes before joining a manager page | `GatherTests.test_page_bytes` | Exit 1, assertion killed |
| manager_page_swap | Follow the logical block table in its given order | `GatherTests.test_page_bytes` | Exit 1, assertion killed |
| joint_token_reverse | Keep token positions even when K and V move together | `GatherTests.test_page_bytes` | Exit 1, assertion killed |
| kv_swap | Preserve K/V halves and channel coordinates | `GatherTests.test_page_bytes` | Exit 1, assertion killed |
| fp8_scale_rotated | Apply separate FP8 scales with rotation enabled | `BridgeTests.test_fp8_rotated_bytes` | Exit 1, assertion killed |
| identity | Admit the unchanged CPU gather implementation | Both `GatherTests` tests | Exit 0, required survivor; limited to the listed CPU views and geometry |

CPU torch and NumPy were installed only into this task's scratch directory, using the official CPU wheel index and binary-only packages, to execute N1 without CUDA, vLLM or model installation. Exact command and rationale: `tool-install.md`; install exit 0 in `torch-install-01.log`. Scratch dependencies and native compilation artifacts are removed after validation. Observed free disk stayed above 25 GiB; one niced compiler ran at a time, alongside at most one observed neighbor build. No engine build was started.

Self-review covered scope, security, design, tests and general conventions. No formal surface table was supplied (brief gap); no requested r2 row is treated as out of contract. No reviewer tests were removed. Rework stays inside the named module. GPU runtime compatibility, actual CUDA views, recurrent ownership/chronology, source checkpoint/tokenizer provenance, calibrated quality bounds and long-depth memory remain open. The script requires an idle compute-process list after stopping vLLM and before receiver subcommands; it never stops foreign processes. Checker exit 5 is retained as uncalibrated evidence, with a dedicated positive/error policy test, not reported as passing quality. The earlier r7b finding 5 (three untested converter refusals) is outside this requested runbook follow-up and remains unchanged.

Task-board remains absent (`goal-read-01.log`, exit 1); `.temp/tasks.md` tracks this run. Signed local delivery and the requested #hybrid / prototype-go handoff are authorized; publication, landing and GPU acceptance are not claimed.

## r3: hybrid-d-r3-261101, bounded stop and clean-tree gate

Ready for review on `m3-vllm-bridge`, parent `09557d6d3e3e961883d858fcb251656be4dbfd53`, accepted by Claude r7d. This new diff is confined to `tools/hybrid-vllm/`. CPU only; no GPU operation, CUDA build, vLLM installation or model download. Signed local delivery is requested; no push is authorized.

| Request | Change and named evidence |
|---|---|
| Low 1: stop hangs and second signal skips restore | Cleanup ignores INT/TERM/HUP, restores and compares the runner first, then stops the recorded PID. TERM polls to `STOP_TIMEOUT` (default 120 s), logs PID-only SIGKILL escalation and reaps the child. Normal capture also restores before stopping. `test_capture_term_ignoring_cleanup` covers HTTP failure, one TERM and repeated INT/TERM/HUP during cleanup, preserving exits 22/143 and observing child wait exit 137. The reviewer's original TERM-ignoring Bash loop passes the same assertions in `reviewer-hang-probe-01.log`. |
| Low 2: untracked CUDA sources | `exact_ref` requires empty `git status --porcelain --untracked-files=all`, with a dirty-tree message listing offending paths. `test_untracked_cuda_refusal` uses real Git porcelain on an untracked `ggml/src/ggml-cuda/zz-untracked-probe.cu`, requires exit 1 before CMake, then removes the plant and admits the same clean fixture. |
| Fresh capture retries reuse the build | Ref and relative binary hash receipts now live in `BUILD_DIR`; they are verified and copied to each `RUN_DIR`. `all`/`build` reuse an existing matching build without CMake; incomplete, stale, empty-receipt or changed-binary builds refuse without rebuilding. `test_build_reuse`, `test_build_reuse_refusals`, `test_binary_tamper` and `test_capture_reuses_build` cover both receiving binaries, fresh directories and successful teardown before the idle check. |
| GPU host build concurrency | Default jobs use `nproc`. Only hostnames `agents1` and `relux-agents1` default to at most four and refuse explicit jobs >4. Other hosts accept positive job counts. `test_build_jobs_by_host` covers a 16-core GPU-host double and explicit 12 jobs; `test_build_refusals` covers the agents1 excess. Actual native compilation used one niced compiler at a time. |

Retry example, after r3 acceptance: `BUILD_DIR=/root/proto/runs/first/build EXPECTED_REF=FULL_REVIEWED_R3_SHA bash tools/hybrid-vllm/gpu-run.sh all /root/proto/runs/retry`. Use a new capture directory. Builds whose r2 receipts exist only in RUN_DIR require one fresh build.

Evidence root: `.temp/hybrid-d-r3-261101/`. `validation.json` records command argv, actual exits, environment, timestamps and hashes. `controls-final.json` records each edit, copied-source hashes, selected tests and failures; child PIDs and full logs are under `plant-logs/`. TMPDIR is on disk, `PYTHONDONTWRITEBYTECODE=1`, and commands run directly with redirected logs, never through tee. Platform: Linux x86_64, Python 3.14.4, GCC 15.2.0. CPU gather used torch 2.13.0+cpu and NumPy 2.5.4; CUDA build is None and initialization is false.

| Validation | Actual exit | Evidence and limit |
|---|---:|---|
| Own SHA-256 object and native helper link | 0 / 0 | `build-sha256-01.log`, `build-native-01.log`, `native-libraries.json`. Pinned CPU library hashes match r2/r7d; own SHA-256 source/object and archive hash recorded. No engine build. |
| Full CPU/native suite on the settled code, 35/35 | 0 | `tests-05.log`, 28.753 s unittest time; `tests-05.json` binds the exact executable/test hashes. Includes all prior 29 names and six new script tests. Minimum observed free disk 29,223,968,768 bytes. |
| Torch CPU gather control, 2/2; actual script gather | 0 / 0 | `plant-logs/r7d_identity.log`, `script-gather-01.log`; no GPU or live vLLM. |
| Reviewer's TERM-ignoring Bash-loop double, three exit/signal cases | 0 | `reviewer-hang-probe-01.log`. Readiness FIFO removes server-start races. Repeated signals are injected from the stop polling command, after cleanup has begun. |
| r1/r6x 22 plants plus pre-fix export cleanup | 1 each | Named assertion kills, no unittest errors; `controls-other.json` and `plant-logs/`. Targeted named tests, not a full suite per mutant. |
| r2 five plants plus r7d 15 mutants | 1 each | 19 assertion kills; upper-bound mutant remains an unstructured RuntimeError and is not credited as an assertion kill. `controls-torch.json`. |
| Eight new shell mutants | 1 each | All named assertion kills with no unittest errors on the settled tests; `controls-shell.json`, `controls-shell-01.log`. Driver exits 0. |
| Bash syntax, Python/heredoc AST, ASCII, whitespace and prior test-name retention | 0 | `bash-syntax-01.log`, `source-check-03.log`, `diff-check-02.log`; original `test_bridge.py` is unchanged and all 10 prior `test_gpu_run.py` names remain. |
| Python type checker | 127 | `typecheck-02.log`: ty unavailable. An earlier direct launcher attempt failed with FileNotFoundError (launcher exit 1); no type-check pass is claimed. |
| CUDA build, live capture, real BF16/Q4_K_M, logits, HTTP import and Metal | Unrun | Orchestrator's GPU/Mac qualification gates. Command doubles do not qualify these paths. |

Disk control failed in two early batches: `tests-02.json` recorded 27,145,482,240 free bytes and `controls-torch-01.json` recorded 27,223,601,152, below the required 28,000,000,000-byte floor. This run did not maintain the floor throughout. The floor-only admission check was insufficient. After removing the 951 MiB task-local CPU dependencies, the monitor gained an abort margin: `tests-03` was interrupted by SIGTERM (child return -15) at 28,138,360,832 bytes; `controls-other-01` exited 143 at 28,196,454,400. Their partial evidence is not a full-suite/driver pass. Completed named plant records were retained, the remaining controls resumed, and the settled full suite passed with the minimum above. No foreign files or processes were removed. `disk-control-failure.md` preserves the failed control and correction; final free space is in validation.json.

Two test-harness corrections are explicit: the initial script-only run exited 1 on three set/iterator errors in log assertions (`script-tests-01.log`); fixed before the full suite. Early ref/cap mutants exposed cross-case fixture contamination, producing extra errors (`controls-rest-01.log`, `controls-last-01.log`); isolated directories and finally-restored fixtures removed the cascades, and the final shell controls are clean. No setup error is counted as a kill.

| Mutant | Gate narrowed to | Named failing test | Exit / verdict and survivor bound |
|---|---|---|---|
| p_layer_count | Exactly 64 distinct captured layer IDs | `test_wrong_layer_count` | 1, assertion killed |
| p_block_size | Minimum/multiple-of-16 size despite consistent page shape | `test_wrong_block_size` | 1, assertion killed |
| p_head_geometry | 256-dimensional heads despite matching fingerprint | `test_wrong_head_geometry` | 1, assertion killed |
| p_kv_dtype | Explicit KV dtype whitelist | `test_unsupported_dtype` | 1, assertion killed |
| p_partial_endpoint | Whole-block exporter endpoint | `test_partial_last_block` | 1, assertion killed |
| p_kv_page_order | Logical page order | `test_paged_kv_layout_and_scales` | 1, assertion killed |
| p_fp8_scale | Actual per-K/V dequantization scales | `test_paged_kv_layout_and_scales` | 1, assertion killed |
| p_gdn_heads | Grouped-to-tiled state head order | `test_recurrent_head_and_conv_layout` | 1, assertion killed |
| p_rotation | Actual Hadamard operation | `test_q8_and_rotation` | 1, assertion killed |
| p_async_schedule | Synchronous scheduling requirement | `test_source_config` | 1, assertion killed |
| p_runner_hash | Exact original runner hash | `test_patch_location` | 1, assertion killed |
| r_kv_swap | K and V source halves | `test_paged_kv_layout_and_scales` | 1, assertion killed |
| r_conv_time_reversal | Oldest-to-newest conv history | `test_recurrent_head_and_conv_layout` | 1, assertion killed |
| r_state_transpose | Preserve [V,K] within each state head | `test_recurrent_head_and_conv_layout` | 1, assertion killed |
| r_kv_head_order | Attention head order | `test_paged_kv_layout_and_scales` | 1, assertion killed |
| r_fp8_scale_swap | K scale for K, V scale for V | `test_paged_kv_layout_and_scales` | 1, assertion killed |
| r_rotation_kv_swap | Serialized K uses 256, V uses 64 | `test_serialized_layout_reference` | 1, assertion killed |
| r_cell_ext_xy | Text x/y positions follow the cell position | `test_serialized_layout_reference` | 1, assertion killed |
| r_cell_ext_token | Cell token ID follows the journal | `test_serialized_layout_reference` | 1, assertion killed |
| r_attention_layer_order | Serialized attention layer identity/order | `test_serialized_layout_reference` | 1, assertion killed |
| r_token_order_in_page | Serialized token order within each page | `test_serialized_layout_reference` | 1, assertion killed |
| r_recurrent_layer_order | Serialized conv/state layer identity/order | `test_serialized_layout_reference` | 1, assertion killed |
| e_export_refusal_cleanup | Late export refusal leaves no partial capture | `test_export_refusal_cleanup` | 1, assertion killed |
| s_cleanup_signals_armed | Repeated cleanup INT/TERM/HUP cannot interrupt restoration or stop | `test_capture_term_ignoring_cleanup` | 1, assertion killed |
| s_restore_after_stop | Runner bytes are restored before the server stop | `test_capture_term_ignoring_cleanup` | 1, assertion killed |
| r7d_identity | Unmodified CPU gather admits the listed layouts only | `GatherTests.test_page_bytes, test_refusals` | 0, required survivor; assumed CPU layouts/geometry only |
| r7d_tok_swap01 | First token pair retains position, K and V together | `test_page_bytes` | 1, assertion killed |
| r7d_tok_roll_in_kernel | Token order within kernel pages | `test_page_bytes` | 1, assertion killed |
| r7d_tok_swap_last_pair | Last token pair retains position | `test_page_bytes` | 1, assertion killed |
| r7d_page_order_reversed | Logical manager page order, K and V together | `test_page_bytes` | 1, assertion killed |
| r7d_page_swap_first_two | First logical page pair retains order | `test_page_bytes` | 1, assertion killed |
| r7d_kernel_subpage_flip | Kernel subpage order within manager pages | `test_page_bytes` | 1, assertion killed |
| r7d_kernel_interleave_r7b | Head/kernel-page axes before joining manager pages | `test_page_bytes` | 1, assertion killed |
| r7d_ratio_ignored | Manager/kernel ratio in physical page addressing | `test_page_bytes` | 1, assertion killed |
| r7d_append_from_zero | Append interval starts at its actual logical offset | `test_page_bytes` | 1, assertion killed |
| r7d_head_flip | KV-head identity | `test_page_bytes` | 1, assertion killed |
| r7d_kv_half_swap | K/V half identity | `test_page_bytes` | 1, assertion killed |
| r7d_null_page_admitted | Null physical page refuses | `test_refusals` | 1, assertion killed |
| r7d_upper_bound_off | Structured upper-bound refusal; not an assertion kill | `GatherTests.test_refusals (ERROR, not an assertion)` | 1, survivor of assertion criterion; fails closed via torch RuntimeError, structured refusal unproved |
| r7d_fp8_scale_rotated | Separate FP8 scales apply under rotation | `test_fp8_rotated_bytes` | 1, assertion killed |
| r7d_fp8_scale_kv_swap_rotated | Rotated K and V use their own scales | `test_fp8_rotated_bytes` | 1, assertion killed |
| r2_kernel_interleave | Head/kernel axes before joining manager pages | `test_page_bytes` | 1, assertion killed |
| r2_manager_page_swap | Logical manager page table order | `test_page_bytes` | 1, assertion killed |
| r2_joint_token_reverse | Token order even when K and V move together | `test_page_bytes` | 1, assertion killed |
| r2_kv_swap | K/V halves and channel coordinates | `test_page_bytes` | 1, assertion killed |
| r2_fp8_scale_rotated | FP8 scale applies under rotation | `test_fp8_rotated_bytes` | 1, assertion killed |
| s_no_kill | Ignored TERM reaches KILL and a bounded exit | `test_capture_term_ignoring_cleanup` | 1, assertion killed |
| s_untracked_ignored | Untracked CUDA input refuses before CMake | `test_untracked_cuda_refusal` | 1, assertion killed |
| s_build_ref_ignored | Reused build receipt matches the exact ref | `test_build_reuse_refusals` | 1, assertion killed |
| s_binary_hash_ignored | Both binaries match complete hash receipts | `test_build_reuse_refusals` | 1, assertion killed |
| s_reuse_disabled | Fresh RUN_DIR reuses verified binaries without CMake | `test_build_reuse` | 1, assertion killed |
| s_agents1_cap_removed | Explicit agents1 jobs above four refuse | `test_build_refusals` | 1, assertion killed |

The finite set has 50 assertion kills, one uncredited error mutant and one required identity survivor. No formal surface table was supplied (brief gap). No requested r3 row is treated as out of contract. The earlier r7b finding 5 (three converter refusal tests) and r7d N2's structured upper-bound refusal are outside this brief's two Low findings/operator notes; both remain unchanged and explicit. No reviewer tests were removed. Self-review covered scope, security, design, tests, process ownership and claim accuracy. Runtime CUDA views, recurrent ownership/chronology, checkpoint/tokenizer provenance, calibrated quality bounds and long-depth capacity remain GPU gates.

Task-board is absent (`goal-read-01.log`, exit 1); the existing `.temp/tasks.md` tracks this rework. No primary-goal mutation/clear, independent r3 acceptance, hosted check or landing is claimed. Signed local handoff and the authorized #hybrid / prototype-go summary are the delivery scope.

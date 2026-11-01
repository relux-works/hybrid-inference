# Mirai S follow-ups, hybrid-c-r4-261101

Ready for review. CPU-only on agents1, 2026-11-01; no push, GPU execution, model/book downloads, or new quality claim.

## Changes and branch evidence

- F2 is a dedicated signed commit 9bb5ba989fe1220fc4b60176d21eccf99b2a7dae on m3-mirai-s, following eed20cac. Header identifies the source fork fdb1e815, sm_80/HIP/MUSA guards and companion Ampere gate.
- Copied-file audit compares every file added by fdb1e815 against the port. The CPU kernel, CUDA header, converter, decoder and check_decoder are unchanged. LICENSE differs only by CRLF normalization; license text preserved. All four substantively modified copied files (CUDA kernel, compare_vllm, README and NOTICE) now carry notices. Integration hunks retain existing upstream licenses.
- F1/F3 signed follow-up e8449484dfc1589fea39148b956c129575d672ac adds parity.py and six CPU contract tests, measured-result provenance, and the updated r2 commands on the port branch. The capability query rejects <8.0, mixed old/new GPUs, empty/unknown capability and query failures before either CUDA build. Sequential fork/port parity precedes study arms; collection or comparison failure explicitly exits 1. PID cleanup is limited to each started server.
- The parity collector retains the orchestrator's exact eight short prompts, 4096/12288-token book prefixes, greedy budgets, and token reuse. Compare refuses missing/empty/malformed case sets and completions. The finite pass criterion is identical prompt and generated token IDs in every case; early EOS can produce shorter nonempty equal completions.
- Signed merge: `f46c03a3abb3f6d95eff0d79af0efa3bde6265ff` on `m3-study-mirai`, parents `5464a4fa1b79c23b1cec7177299f4c6cceae6616` and port head `e8449484dfc1589fea39148b956c129575d672ac`. Merge-tree and actual merge each returned expected exit 1 for two documentation conflicts: keep the updated r2 report and preserve both unresolved-question entries. Automatic README merge retains both tool sections. All three new commits have signature `G` and `git verify-commit` exit 0. This report is a separate signed documentation follow-up; the tested code/test/build-source tree remains the merge tree.

## CPU evidence

Evidence root: .temp/hybrid-c-r4-261101/. Commands and real exits live in port/commands.json and combined/commands.json, with standalone logs and PID files. Build/runtime validation commands use `nice -n 10`, disk `TMPDIR`, `PYTHONDONTWRITEBYTECODE=1` and `OMP_NUM_THREADS=2`. Configure logs are separate; configuration also uses disk temporary storage. No pipeline masks a gate's exit code. Compiler GNU 15.2.0; reused local CMake 3.31.10, no installation. Builds are sequential -j4, with no more than one other observed host build.

The port deliberately has no study/scorer and only ten original identity methods. To execute the requested r2 twelve-method/study set without backporting the study, r2 sources are copied byte-for-byte from 5464a4fa into an external compatibility harness. Its scorer is compiled against the port's actual headers and freshly built native libraries. Python harness modules use port bench_common and native binaries. The reference integration harness also uses the unchanged r2 hybrid_bench adapter. `scope-and-harness-audit.json` records every source blob. This establishes compatibility of that r2 harness with the port; it does not claim a native study target exists on m3-mirai-s. The combined head uses its own native scorer, driver and tests.

Validated port head: `e8449484dfc1589fea39148b956c129575d672ac`. Validated combined head: `f46c03a3abb3f6d95eff0d79af0efa3bde6265ff`. The following results apply to **each** head, with the external-harness distinction above. `B=.temp/hybrid-c-r4-261101/build`; headers and Qwen3.5-0.8B Q4_0/Q8_0 model paths are the same existing local inputs as r2. Exact argument vectors, environments, observation times and binary/source SHA-256 values are retained in each branch's `commands.json` and `identities.json`.

| Check / command | Exit on port / combined | Result per head |
|---|---|---|
| CMake CPU configure: CUDA OFF, NATIVE OFF, UI/CURL OFF, tests ON | 0 / 0 | `configure-{port,combined}-01.log` |
| CMake `--build $B --target ... -j4` | 0 / 0 | server, prefill, identity, bench-check, codec, hybrid-check, backend-ops; combined also builds native score |
| Port external r2 scorer compile against port headers/libraries | 0 / n/a | `port/external-r2-scorer-build-01.log`; combined builds its own target |
| `python3 -m unittest discover -s tools/mirai-s -p test_parity.py -v` | 0 / 0 | 6/6 methods, including exact operator guard blocks with CPU fixtures |
| unittest `test_hetero_study.py` | 0 / 0 | 12/12 |
| unittest `test_reference.py` | 0 / 0 | 2/2 |
| unittest `test_bench.py` | 0 / 0 | 42/42 |
| unittest `tools/hybrid-eval/test_eval.py` | 0 / 0 | 11/11 |
| `test_mirai_identity.py --binary $B/bin/llama-hybrid-check-identity --headers ... --artifacts ... -v` | 0 / 0 | 12/12, no skips; 34 logged native calls: 11 at 0, 23 intended refusals at 1 |
| `test_reference_native.py --client ... --reference ... --build-dir $B --artifacts ...` | 0 / 0 | 31 recorded commands: 14 at 0, 16 intended refusals at 2, one intended witness failure at 1 |
| `test_hetero_study_native.py --client ... --reference ... --build-dir $B --artifacts ...` | 0 / 0 | 25 CLI commands: 11 at 0, 14 intended refusals at 2; includes real local/imported arms and live proxy control |
| `$B/bin/test-backend-ops test -b CPU -o MIRAI_MUL_MAT` | 0 / 0 | 39/39 |
| Backend r2 GATED_DELTA_NET / FLASH_ATTN_EXT filters | 0 / 0, each | 5/5 and 8/8 |
| `$B/bin/llama-hybrid-check-codec` | 0 / 0 | Existing positive/negative controls |
| `$B/bin/llama-hybrid-check Qwen3.5-0.8B-Q4_0 ... 512` | 0 / 0 | Greedy 64/64, append/refusal controls; exact token JSON equals r2 |
| r1 workflow model-naming checker | 0 / 0 | 166 mappings |
| `bash -n .temp/hybrid-c-r4-261101/gpu-commands.sh` | 0 | Extracted commands equal the committed r2 report |
| Static preservation/ASCII audit; `git diff --check` | 0, each | Original r2 identity tests byte-identical; CUDA delta is header comment only |

Reference-suite refusals keep their real nonzero exits; the harness itself exits 0 after checking the intended reasons and state. The legacy equivalence checker still reports uncalibrated exit 5; this is not promoted to equivalence acceptance. Header tests prove static loader/layout admission, not real 27B scoring or weight validity.

The first external port study attempt exited 1: `engine()` needed a sibling `bench_common.py` for its helper fingerprint. A symlink now points to the byte-identical port/r2 helper, retaining the port's actual repository root. The first resume wrapper also exited 1 before launching tests on an undefined local variable. Both setup errors were corrected without changing product or test sources; the affected native study suite then passed. Failed logs and `commands-through-setup-failure.json` are retained. Existing compiler warnings remain in build logs.

Disk observations are explicit: the lowest manual sample was 25,055,973,376 bytes during concurrent host activity. Owned build/run PIDs were paused, then resumed after space recovered. The first background watcher exited with its shell; the retained-session restart was verified and sampled throughout the remaining runs. Watcher logs preserve later minima. No other lane's process or files were touched. Builds were one at a time in this run with one other niced host build observed. At the final audit, all **222 recorded owned PIDs were inactive**; expired test `clients.json` files were removed only after that check. `audit.json` records observation time `2026-11-01T20:25:26.290182+00:00`.

## Bounded controls

| Mutant | Narrows the gate to | Named failing test | Result / limit |
|---|---|---|---|
| M1: return 0 after divergent token comparison | Generated token equality beyond evidence completeness | test_compare_tokens; test_gpu_parity_guard | Exit 1; killed on CPU JSON fixtures and exact documented compare block |
| M2: accept a case set with fewer than ten rows | Complete fixed workload, beyond agreement of present rows | test_incomplete_evidence | Exit 1; killed; no GPU inference claim |
| M3: lower the documented capability threshold to 7.0 | Explicit Ampere admission before build | test_gpu_capability_guard | Exit 1; killed with 7.5 and mixed-device stubs |

No surviving mutant. Mutants were isolated under .temp/ and did not modify tested product sources. Existing r2 reference-position witness controls remain in the requested native suite; earlier mutation campaigns are not relabeled as current evidence.

## Limits and review

The measured 2026-11-01 GPU parity is supplied by the orchestrator and bound to fork fdb1e815 versus combined 5464a4fa: same GGUF/token IDs, -np 1, q8_0 KV, flash attention, MTP off, three unrelated fork features disabled; eight short x48 and two book prefixes x32, all 10 identical. README distinguishes that measured result from this CPU-only run. Raw GPU-host artifacts were not fetched. No new CUDA/HIP/MUSA/Metal build, GPU inference, 27B imported-state equivalence, KL quality/performance, or hosted checks are claimed.

Self-review applies scope, security, approach, general, backend/comment and test checks. No normative surface table was supplied (brief gap); no requested CPU case is declared out of contract. Original reviewer tests are retained. Independent review remains with the orchestrator. ty/editorconfig-checker readiness exits 1 (unavailable); these checks were not run. Rework is bounded to the named Mirai and GPU-command documentation scope plus root tool documentation and the existing unresolved-questions record.

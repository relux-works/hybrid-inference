# Glimmer paired RULER: CPU qualification and GPU handoff

Run `hybrid-d-r14-261103`, branch `m3-ruler-glimmer`, base `18d3f114e46c4f801c4a09e2c6cf8ba2303ae42d`. Ready for review. Local signed delivery only; no push or PR. Scope: evaluator profile, stage, tests and documentation. No C++ changes, model/data downloads or GPU run on agents1. The Qwen producer checkout was read only.

## Delivered behavior

- `ruler_glimmer.py` pins the kit registry's Glimmer Q4_K_M repository/revision/file/size/SHA and native prototype base, plus 131072 context, 2048 SWA, no recurrent state and four depths.
- `ruler_eval.py --model-profile glimmer30` checks every service's model/native/context/backend/batch identity and the minimum native-slot bound before examples. The profile becomes restart-manifest identity. Generic context admission also includes the partition client's context.
- Depths: 8192/32768/65536/122880. The last is the deepest whole 8192-token step within native context plus 128 output and two guard positions; 8062 positions remain. Same 13 tasks, seed 261102, 30/task at every depth, unchanged scoring/statistics.
- Bound `S=10240*(depth+130)+67108864`, per-volume floor `2*S+1073741824`. S is 152326144/403984384/739528704/1326731264 bytes; floors are 1378394112/1881710592/2552799232/3727204352 bytes. This replaces Qwen's model-specific formula.
- `stage-ruler-glimmer.sh` gates a fresh dedicated 96 GB RTX PRO 6000 and 120 GB free disk, downloads/verifies the pinned model using the existing atomic benchmark downloader, builds the pinned native source niced at -j4, bootstraps the existing RULER inputs, launches 64/256 client partitions, runs two-example pilots then full depths, verifies and copies archives, computes joint analysis, stops its services, and emits terminal done/failed lines. It does not publish source or provision a host.

The [Glimmer runbook](GLIMMER-RULER-RUNBOOK.md) documents exact commands, flags, pins, recovery, disk math, rate inputs and interpretation limits. The native pin is the accepted prototype base, which contains the fork's SWA fixes; stock kit `dde879bc` is the restore-behavior reference, not a substitute hybrid binary.

## Validation

All commands below ran in this task worktree. `E=.temp/hybrid-d-r14-261103`; logs are private, disk-backed and untracked. `PY` is the already installed offline RULER environment at `/home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r34-261102/prototype/.temp/ruler-venv/bin/python`. All test runs used `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=$PWD/$E/tmp`. No dependency was installed.

| Command | Result / real exit | Log |
|---|---|---|
| `$PY -m unittest discover -s tools/hybrid-eval -p test_ruler.py -v` | 21 tests, exit 0 | `$E/ruler-tests-04.log` |
| `$PY -m unittest discover -s tools/hybrid-eval -p test_eval.py -v` | 11 tests, exit 0 | `$E/eval-tests-01.log` |
| `$PY -m unittest discover -s tools/hybrid-proxy -p test_journal.py -v` | 18 tests, exit 0 | `$E/journal-tests-01.log` |
| `python3 -m unittest discover -s tools/hybrid-proxy -p test_templates.py -v` | 43 tests, exit 0, no skips; existing system Jinja2 3.1.6 | `$E/template-tests-02.log` |
| `$PY -m unittest discover -s tools/hybrid-proxy -p test_benchmark.py -v` | 22 tests, exit 0 | `$E/benchmark-tests-01.log` |
| `PYTHONPATH=tools/hybrid-bench python3 -m unittest test_bench.BenchmarkTests.test_model_hash_atomicity -v` | 1 test, exit 0; local file fixture, no network | `$E/model-hash-01.log` |
| `nice -n 10 $PY tools/hybrid-eval/test_native.py --node-binary <existing-build>/bin/llama-hybrid-prefill --decode-binary <existing-build>/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts $E/cpu-native --depths 2048 --tasks single_needle --ruler-checks` | exit 0; five-arm synthetic needle, all 13 real-tokenizer fixture generators (2k, CWE 4k), three-arm RULER needle | `$E/native-01.log`; `$E/cpu-native/` |
| `bash -n tools/hybrid-eval/stage-ruler-glimmer.sh` | exit 0 | `$E/stage-syntax-03.log` |
| Python AST, embedded stage-Python AST, new-file ASCII/newline/whitespace audit | exit 0 | `$E/syntax-style-01.log` |
| `git diff --check` | exit 0 | `$E/diff-check-02.log` |
| Retained reviewer test names, unchanged native source comparison, source hashes and owned PID absence audit | exit 0; 17 inherited RULER tests retained, four added | `$E/source-audit-02.json`, `$E/source-audit-02.log` |

`<existing-build>` above is `/home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r34-261102/prototype/.temp/build`. Its executable version is `fca0fe31d`, CPU Release, built with GCC 15.2. Native source paths (CMake/cmake/common/ggml/include/src/server/hybrid/component tests) have zero diff between that native revision and base 18d3f114; the test harness uses this candidate's Python sources. This is existing-binary reuse, not a new build or a claim that the executable embeds the Python delivery revision. The native harness records binary hashes, commands, health, generated fixture checks, the three-arm row and cleanup. All five owned service PIDs and its evaluation child were absent at the final audit. No build tree was created; no other lane's build tree was removed. Disk stayed above the 28 GB floor; task-native artifacts are about 2 MB.

The four new named tests are `test_glimmer_pins_services_and_negative_controls`, `test_glimmer_depth_and_slot_arithmetic`, `test_glimmer_cli_admission_and_manifest_identity`, and `test_glimmer_stage_wrong_head_has_terminal_failure_and_no_writes`. Wrong hash/native/context/backend/batch/ubatch and one-byte-under-bound controls require the intended refusal, paired with valid admission. CLI rejection leaves no output directory or sessions; valid CLI admission writes the production profile and 13-task/30-example/seed identity. Missing stage settings and wrong head exit nonzero with terminal failed and no run directory. Existing one-byte-below-disk-floor, source/data corruption, wrong-answer, interrupted-pair, 17-row resume, recovery and archive member-hash controls remain green. The reused downloader test checks a wrong digest leaves no partial cache and correct bytes are accepted. These are bounded controls, not exhaustive fault injection or a full simulated GPU stage.

Development failures are retained: RULER runs 01 and 02 exited 1 on errors in new test expectations (1 KiB arithmetic typo, then wrongly demanding Holm fields on descriptive task rows). Production formula and statistical analysis were unchanged by those corrections. Run 03 passed 20 tests; run 04 adds the CLI admission test and passes 21. The initial venv template run exited 0 with five Jinja2-related skips; the system interpreter rerun removes all five skips. The initial PID audit assumed scalar PID files and failed parsing JSON records; the corrected audit reads both formats. Details: `$E/diagnostics-01.md`.

No independent reviewer acceptance, hosted CI or landing is claimed. `ty`, `editorconfig-checker` and `shellcheck` are unavailable and were not run; no tool download was authorized. Native build validation was satisfied by reusing unchanged native sources/binaries for the actual CPU execution; C++ was not rebuilt. The complete fresh-GPU stage, CUDA compilation and real external model/data bootstrap remain unrun here.

## Research anchors and budget

Read-only evidence used:

| Input | Location / identity |
|---|---|
| Kit model pin | `/home/agent/Werkstatt/hybrid-a/paper/bench/gpu/scripts/kitlib.py`, SHA-256 `e7d4ec8357fc33a649b1dce1cb30c015fa0b621f0153d0be75cc71d92e294c4f` |
| Glimmer geometry and native slot saves | `/home/agent/lanes/hybrid/.temp/kit165-host2/x/glimmer30-gpu-bench-results-20261102T024947Z/results/`; `model-geometry.json` SHA-256 `9b54f4bc66b44fd8aaf42f0aa836c635a37ff8ae7e08617768fa02bc7299460d` |
| PRO 6000 kit rates, section 1a | `/home/agent/Werkstatt/hybrid-c/paper/data/measurements/run16-2026-11-03/FACTS-261103.md`, SHA-256 `dee4432762372bd007d323899b6c7312281754d8044d60bd999632f3bcdd0d92` |
| Qwen paired-RULER wall time | Task brief: 15.4 s/example at 8192, about 59 s/example at 32768; these include the three arms |
| Price assumption | Orchestrator `HANDOFF.md` historical $1.35/h; conditional estimate, not a checked live rental quote |

Scale Qwen wall time by the interpolated Qwen/Glimmer segment-prefill ratio; beyond 32k extrapolate Qwen time linearly, and hold Glimmer's last measured 118k rate at 122880. Per-depth central estimates are **1.60/5.70/9.84/15.84 hours**. With 50% contingency: **2.41/8.56/14.76/23.75 hours**, **$3.25/$11.55/$19.93/$32.07** at the declared $1.35/h assumption. Total **32.98 hours central, 49.47 hours / $66.79 with contingency**, plus setup, eight pilot examples, transfer and restart overhead. These are rate-derived scenarios, not Glimmer RULER measurements or upper bounds.

## Review bounds and remaining GPU gap

Self-review covered scope, security, design, general and tests. Reused the existing launchers, hash-checked downloader, generator/scorer, disk gate, cleanup/recovery, archives and statistics; no new service subsystem or dependency. All changed paths are evaluator/tests/docs except the root README tool entry and unresolved-question record, which document the same scope. This is initial implementation, not a rework.

The brief supplied no formal surface table or additional landing-property suite; that is a declared brief gap. The applicable existing suites listed above ran, and every inherited RULER reviewer test remains under its original name. No exhaustive coverage claim is made.

Out of the CPU acceptance contract under the explicit no-GPU/no-download rules: actual 30B GGUF tokenizer/template continuation, SWA import/save/restore and no-re-read behavior beyond the window, CUDA batch numerical behavior, aggregate three-service VRAM, native-slot transient high-water bounds, long-context paired quality and end-to-end timing. No small SWA model is on this host. Only the pinned-model GPU pilots and full run close those gaps. The prior kit restore observations do not qualify this exact evaluator run. Model template redistribution/license decisions remain under the existing unresolved-question entry; this change copies no template source.

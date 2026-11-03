# r38b: review fixes and GPU operator stages

Base: accepted `64dc3c699d2885f49146a0536f98c4fb06555692`, Muse r7ah ACCEPT in `/home/agent/Werkstatt/hybrid-review-proto/llama.cpp/.temp/REVIEW-state-bf16-muse.md`. Branch `m3-state-bf16`, same isolated prototype worktree under the workbench's `.temp/hybrid-a-r38-261103/prototype`. No push, GPU execution, downloads or changes to the running RULER process.

## Result and remaining decision

The legacy client/distribution/guard harnesses, plus append's legacy golden-fixture mode, explicitly admit f32 before expansion. The labelled bf16 benchmark/scorer and append's `--bf16` mode retain their intended lossy behavior. Empty HYBD transfers were already outside the f32 contract: the writer now rejects zero-token ranges and empty bodies consistently before dtype-specific parsing. No support for a zero-token transfer is claimed. Valid default f32 bytes and exact import remain covered. The 8k disk text is corrected to 1.97 GB.

`tools/hybrid-eval/stage_ruler_bf16.py` stages only completed frozen 8k/32k depths in an idle operator window, preserves `build-ruler` and original service state, builds in a separate directory, preflights **all** selected manifests/examples before any inference, verifies archives and restores original services on success/failure. `--restore-only` uses the persisted service plan after an uncatchable interruption. Source, binary, model, partition, template and live decoder/proxy identity are checked on restoration. The evaluator's `--preflight-only` performs the same real manifest/input gate as execution and writes no candidate output.

`tools/hybrid-bench/stage_kl_bf16.py` audits the surviving archive and refuses unsupported exact reconstruction. The reports contain no document/token hashes and no homogeneous Q4-to-Q4 imported row. Historical Q4-node imports used Bonsai/Mirai clients. Model pins were recovered separately from the retained `host-setup.sh`, and stored in `state_bf16_model_pins.json`. The explicit `new-study` alternative bootstraps a new pinned corpus/token lock, regenerates D, runs local/f32/bf16 Q4 arms, produces paired/bootstrap reports and records its non-reproduction status. It cannot satisfy the original request for historical document/token identity or a missing same-partition historical f32 comparator. That owner decision remains in `UNRESOLVED_QUESTIONS.md`; no historical hashes or metrics were fabricated.

Runbooks: `tools/hybrid-eval/RULER-BF16-RUNBOOK.md` (Vast 58864154 stage/restoration) and `tools/hybrid-bench/STATE-BF16-RUNBOOK.md` (fresh-node KL, pins, constraints and alternative). RULER reserve: 4.5-6 h for both depths including first build/restoration. KL reserve: 4-8 h on a fast link, 72.78 GB models, volume >=160 GB with about136.2 GB initially free. These are unmeasured planning estimates.

## CPU evidence

All logs are private under `.temp/r38b/`. `tested-identity.json` binds source and binary hashes; `CMakeCache-tested.txt` preserves configuration. `configure.sh`, `build.sh` and `build-consumers.sh` retain exact native commands. System Python runs the standard-library checks; RULER uses the existing r34 `ruler-venv/bin/python`, with no package installation. `TMPDIR` was on disk and `PYTHONDONTWRITEBYTECODE=1` was set.

| Standalone command / gate | Real exit | Evidence / claim |
|---|---:|---|
| `bash .temp/r38b/configure.sh` | 0 | CPU configuration, UI/curl/CUDA disabled; `configure-01.log`. |
| `bash .temp/r38b/build.sh` | 0 | Codec, legacy client/distributions/guards, append; `build-01.log`. |
| `bash .temp/r38b/build-consumers.sh` | 0 | Append, main native check, benchmark checker, scorer, prefill and server compile/link against the changed internal C++ function; `build-02.log`. |
| `python -m unittest discover -s tools/hybrid-eval -p test_ruler.py` | 0 | 22 tests; `ruler-unit-02.log`. Existing named tests preserved; new preflight read-only/mismatch control. |
| `python -m unittest discover -s tools/hybrid-eval -p test_gpu_stages.py` | 0 | 14 tests; `stages-unit-08.log`. CPU orchestration, archive/hash/pin/identity/refusal controls. |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_hetero_study.py` | 0 | 13 tests; `hetero-unit-01.log`. Existing report/receipt/arm controls retained. |
| `.temp/r38b/build/bin/llama-hybrid-check-codec` | 0 | `codec-02.log`: f32 frozen wire, shared empty-transfer refusal, dtype admission, rounding/checksum controls. |
| `.temp/r38b/build/bin/llama-hybrid-check-append MODEL --bf16` | 0 | `native-02.log`: actual Qwen3.5-0.8B state, f32 exactness, bf16 error/size/KV/chunk and rejection controls. Maximum normal finite relative error .00389105 <=2^-8. |
| `nice -n 10 python3 tools/hybrid/test_legacy_dtype.py --build-dir .temp/r38b/build --model MODEL --artifacts .temp/r38b/legacy` | 0 | `legacy-01.log` and `legacy/results.json`: native 512-token fixtures; all four actual legacy binaries accept f32 (each exit0), refuse a bf16 header label before expansion (each exit1, exactly `FAIL state dtype mismatch`). The label-only plant deliberately leaves body bytes unchanged; valid bf16 payloads are covered by the preceding native append test. |
| `ty check gpu_stage.py stage_kl_bf16.py stage_ruler_bf16.py ruler_eval.py test_legacy_dtype.py` with explicit existing venv and tool search paths | 0 | `typing-05.log`; full command retained in `validation.json`. |
| Both stage `--help` entry points | 0 | `kl-help-01.log`, `ruler-help-01.log`; no work begins on import/help. |
| `python3 tools/hybrid-bench/stage_kl_bf16.py --historical-archive /home/agent/lanes/hybrid/.temp/kl-study/kl-report-stage2.tar.gz --audit-only` | **1** | Expected-red `audit-stage2-02.log`: real archive lacks historical provenance/comparator; terminal `failed`, no download/build/GPU. |
| `git diff --check` | 0 | Reviewed scoped diff has no whitespace errors. |

`MODEL` above is `/home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf`, the existing local CPU model. There are 49 unit tests in the three selected suites, plus codec/native/legacy binary checks. Earlier type check `typing-01.log` exited1 for a missing `sys` import, Python3.10-incompatible `hashlib.file_digest`, nullable stream narrowing and tool search-path configuration; corrected before the passing checks. No failing attempt is counted as green.

Known-violation controls are executable product/gate negatives, not unreported code mutants:

| Plant | Named test / intended observation |
|---|---|
| Every nondeclared RULER manifest field, including scalar type | `test_manifest_every_nondeclared_field_refuses`: common identity refuses. |
| Later selected depth preflight fails | `test_all_depths_preflight_before_examples_and_restore_on_failure`: zero examples and restoration attempted; paired all-depth positive control runs after both preflights. |
| Active evaluator | `test_active_evaluator_refuses_with_idle_control`: refuse without signalling; idle admits. |
| Archive member changed with newly valid outer seal | `test_archive_internal_corruption_refuses_with_valid_control`: internal member checksum refuses. |
| Changed/missing cached build target | `test_cached_build_record_admission`: refuse without compiling; valid cached record admits. |
| Wrong restored service commit | `test_restore_identity_with_positive_control`: no `restored.json`; original identity admits. |
| Mutable model revision, missing token hash or reordered document set | `test_missing_hash_revision_comparator_refusals_with_new_study_control`: intended provenance refusal. |
| Heterogeneous f32 row relabelled as homogeneous | `test_reproduction_comparator_identity_with_positive_control`: refuse; synthetic truly homogeneous comparator admits. No such historical row is claimed. |
| Downloaded bytes have wrong SHA256 | `test_model_download_digest_with_positive_control`: final model filename is absent; partial retained. Network boundary is mocked, hashes/file publication are real. |
| One prepared continuation token changes | `test_prepared_token_hash_mismatch_with_control`: named document/token refusal. |
| New mean outside saved interval | `test_historical_interval_gate`: explicit failed check; endpoints admit. |
| Non-target GPU or low disk | `test_gpu_identity_admission`, `test_disk_gate_boundary`: refuse; target/boundary controls admit. |

Self-review used the code-review skill's scope, security, approach, general and test checklists. No explicit surface table was supplied in the brief; coverage is bounded to the named tests above. No code mutant run is claimed. GPU downloads, CUDA build on the target card, real service transition/recovery, reference regeneration, full KL/RULER measurements and timing estimates are **unverified** here by the CPU-only rule. SIGKILL/power-loss recovery is an operator step, not an automatic finally guarantee. Original proxy behavior and unrelated suites are unchanged and were not rerun.

The native builds checked `pgrep -u agent cc1plus` and `uptime`, then ran niced at `-j4`, serially. Owned build trees and generated binary fixture directories were removed after verification. `cleanup.json` records 232570880B freed; observed free space before/after cleanup was 32295768064/32528338944B, above the 28GB floor. All nine recorded legacy child PIDs are inactive. Logs, process exit records and small JSON evidence remain. Delivery is a signed local review handoff; no publication or GPU result is claimed.

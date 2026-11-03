# Paired RULER: Glimmer 30B on RTX PRO 6000

Run hybrid-d-r14-261103 extends the accepted Qwen evaluator at `18d3f114e46c4f801c4a09e2c6cf8ba2303ae42d`. This is a private-fork experiment. No Glimmer weights, external data or GPU execution were used on agents1. Do not install this revision into the running Qwen experiment: Python source hashes are part of restart identity.

## Frozen experiment

Use `meta-models/Muse-Glimmer-30B-GGUF@b44f1072cf75e918a1b82a3a17b45c4389b076e7`, file `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`, 16,756,683,904 bytes, SHA-256 `4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e`. This is the GPU kit registry's `glimmer30` entry. It has SWA 2048, 13 full-attention and 39 window layers, no recurrent state, native context 131072. No RoPE extension, HF tokenizer, replacement chat template or Qwen template is used.

All three services use native revision `18d3f114e46c4f801c4a09e2c6cf8ba2303ae42d`, the accepted prototype base. Build that revision, not stock upstream `dde879bc`: the latter is kit 1.6.5's restore-threshold reference and lacks this fork's hybrid endpoints. The prototype already contains its own SWA restore/export repairs, including the threshold clamp to one. Native sources are unchanged by this evaluator extension. The stage creates a separate native-source worktree and uses the standard launchers to generate real build records; do not fabricate a stamp or relax binary verification.

The arms are `local` (batch/ubatch 64), `offload` (node prefill 256, import into that same batch-64 decoder), and `local_partition` (a separate local decoder at 256/256). All use CUDA, q8_0 K/V, flash attention on, one slot, context 131072, two generation/prefill threads, `--spec-type none`, `--jinja`, `--no-context-shift`, `--cache-ram 0`, `--no-cache-idle-slots`, `--ctx-checkpoints 0`, `--checkpoint-min-step 0`, and private slot-save paths. The prefill binary configures its own matching q8_0 KV/flash-attention layout. Do not enable `--swa-full`, which changes window allocation. The client launchers supply these flags without a model-specific override.

The existing GGUF decoder's `/apply-template` and `/tokenize` render and count the actual prompt, including BOS, framing and assistant answer-prefix continuation. Thinking is disabled through the existing template kwargs. Each arm must have the frozen prompt-token hash, no fallback/replay/re-prefill, and the offload arm must import state. The output budget remains task-specific (maximum 128 tokens). The profile additionally refuses an incorrect model SHA, native revision, service context, backend, batch/ubatch or undersized slot budget before any example. The generic evaluator remains available for small CPU fixtures.

Use the same 13 NVIDIA RULER configurations as Qwen: `niah_single_1/2/3`, `niah_multikey_1/2/3`, `niah_multivalue`, `niah_multiquery`, `vt`, `cwe`, `fwe`, `qa_1`, `qa_2`. Seed 261102, 30 examples/task/depth: 390 paired examples, 1170 completions/depth; 1560 examples and 4680 completions overall. Keep the inherited paired statistics, exact McNemar, four-depth Holm adjustment and predeclared .02 equivalence margin. Equal scores alone do not establish answer identity or equivalence. Pilots never enter the full-run denominator.

## Depth and disk admission

Input depths are **8192, 32768, 65536, 122880**, counted after rendering. The last is 120 Ki tokens: the deepest whole 8 Ki-token step satisfying `depth + 128 + 2 <= 131072`. A 128 Ki-token input fails that constraint. The mathematical input ceiling is 130942; 122880 deliberately retains 8062 positions after the maximum answer and two-position guard, keeps a round binary step, and stays near the kit's measured 118000-token point. Every depth is well beyond the 2048-token window. Do not substitute 131072 or reduce the denominator at the last depth.

The kit 1.6.5 Glimmer q8_0 geometry has 7072 full-attention bytes/token and 43450368 window bytes (conservative allocation 54312960). Its saved native slots at 16000/50000/100000/118000 tokens were 156884268/397876268/752276268/879860268 bytes, exactly `7088 * depth + 43476268`. These are measured planning anchors, not a measurement of this evaluator's transient peak.

Use `S(D) = 10240 * (D + 128 + 2) + 67108864` for `--saved-session-bytes`: 10 KiB per position plus 64 MiB for window/headers/padding. It exceeds the kit's observed slope and intercept. The existing R36 disk gate checks each client slot volume before an uncommitted example: `free >= 2*S(D) + 1073741824`. It accounts for a transient native file plus rollback/export, not 390 retained slots. Release/recovery removes session files; do not reuse Qwen's 442685280-at-8k formula.

| Input depth | S: saved-session bytes | Slot-volume free floor (bytes) | Floor (GiB) |
|---:|---:|---:|---:|
| 8192 | 152326144 | 1378394112 | 1.284 |
| 32768 | 403984384 | 1881710592 | 1.752 |
| 65536 | 739528704 | 2552799232 | 2.377 |
| 122880 | 1326731264 | 3727204352 | 3.471 |

Require **120 GB free disk before fetching**, on the run filesystem; reserve the 16.76 GB model, CUDA build, wheels/data, all frozen prompts/archives and logs. This exceeds the slot-only floor. Both client directories normally share one filesystem: the arms run sequentially, so do not sum two simultaneous inference peaks, but each volume must pass admission. Moving stopped service state does not free its bytes. Concurrent writers can defeat a snapshot floor.

## Fresh GPU host

Provision one dedicated 96 GB RTX PRO 6000 with CUDA >=12.8 (`nvcc`, CMake supporting `120a-real`), a C++ toolchain, Python 3.11+ with venv/pip, git, Bash, `nice`, `pgrep`, and `sha256sum`. No installation or host provisioning is performed by the stage. The pinned Python requirements need compatible binary wheels. Supply the reviewed Glimmer branch through the owner's existing source/bundle transfer process; the stage never pushes or changes the active source branch.

```bash
cd /absolute/path/to/reviewed/glimmer-checkout
export EXPECTED_HEAD=<full-reviewed-glimmer-commit>
export RUN="$PWD/.temp/ruler-glimmer-$(date -u +%Y%m%dT%H%M%SZ)"
bash tools/hybrid-eval/stage-ruler-glimmer.sh
```

Use a new absolute `RUN` path. The stage refuses a wrong/dirty source head, wrong GPU or insufficient disk before any download. It uses the existing benchmark downloader for the immutable HF revision, verifies SHA-256 atomically and checks byte size before selecting the model filename, builds the accepted native revision niced at `-j4`, and bootstraps exactly the same pinned RULER source/corpus/NLTK inputs and requirements as Qwen. One model is shared by all services. Incremental standard-launcher builds create/verify their own records; no parallel native builds are started.

The launchers estimate resource admission separately. Three model/context allocations should consume roughly 53-60 GB before runtime scratch, below the 96 GB device. This is a planning estimate, not aggregate VRAM qualification. Service startup and `$RUN/logs/vram-services.log` must confirm the actual allocations. Any admission/startup failure stops the experiment; changing backend/context/partitions would be a different protocol.

At each depth the stage first runs a two-example pilot (`niah_single_1` and `qa_1`, one each), verifies its archive, then runs all 390 examples and verifies/copies that archive. It stops on the first error. Each phase writes a separate log and `complete.tar.gz` with a hash sidecar, internal member hashes, frozen input/download identities, source hashes, metadata and analysis. Copies are placed in `$RUN/archives/{pilot,results}-DEPTH.tar.gz` with matching named SHA-256 sidecars. Copy those files to durable storage through the owner's transfer channel after each depth; a second local copy is not durable external storage. Private prompts/specs, service credentials and native state remain outside shareable archives.

A successful phase prints `stage ruler-glimmer PHASE depth=DEPTH done`. After four full depths and joint analysis, the exit trap stops only services recorded as launched by this stage and prints terminal `stage ruler-glimmer done head=... run=...`. Any failure, including service cleanup failure, prints terminal `stage ruler-glimmer failed exit=... head=... run=...` and exits nonzero. INT/TERM also take that cleanup path; SIGKILL/power loss cannot. Preserve failed logs and partial evaluator files; the benchmark downloader removes a rejected temporary model. A model/hash/bootstrap failure never advances to inference. Credentials are never printed.

Inspect pilots for exact token depth, successful imports, no fallback/re-prefill, scores, output-budget hits, generation time, per-arm time, native-slot high-water bytes and VRAM. Equality between arms is an outcome to measure, not a mandatory pass. If a saved file exceeds S, increase `--saved-session-bytes` before resuming; that operational bound does not change prompt/scoring identity. The current stage is for fresh runs only. For an interrupted evaluator, preserve the data/input lock, venv, specs, rows and source revision; use the [Qwen recovery procedure](RULER-RUNBOOK.md) for these owned service directories. Do not rerun the fresh-stage entry point with an existing RUN, delete committed rows, or invoke `--resume-slot-cleanup` on a Glimmer run.

After relaunching identical services, resume each depth with the same command (pilot uses its own output and two-task selection):

```bash
PY="$RUN/venv/bin/python"
INPUT_SHA=$(cat "$RUN/input-lock.sha256")
for depth in 8192 32768 65536 122880; do
    size=$("$PY" tools/hybrid-eval/ruler_glimmer.py --depth "$depth")
    nice -n 10 "$PY" tools/hybrid-eval/ruler_eval.py \
        --model-profile glimmer30 --client-state "$RUN/client" --partition-client-state "$RUN/partition" \
        --data "$RUN/data" --input-lock-sha256 "$INPUT_SHA" --output "$RUN/results" --depth "$depth" \
        --saved-session-bytes "$size" --samples-per-task 30 --seed 261102 --margin .02 --request-timeout 7200 || break
done
```

Completed rows are reused with checksums, incomplete examples use new session IDs, and pending sessions must be released before any new example. Complete depths are verified and returned without inference. Archive copying and joint analysis after a manual resume follow the stage's commands. Keep the native build until services have stopped and the experiment no longer needs restart; then remove only this run's build tree and registered native worktree.

## Time and cost scenario

The Qwen observations supplied for this run are **15.4 seconds per paired example at 8192** and **about 59 seconds at 32768**, including the three arms; do not multiply those times by three again. Beyond 32k extrapolate Qwen seconds as `59 * depth / 32768`. Scale that wall-time scenario by `Qwen segment-prefill rate / Glimmer segment-prefill rate` from FACTS section 1a (kit 1.6.5, PRO 6000). Interpolate the 16k/50k/100k/150k Qwen anchors (4295/3241/2255/1699 tok/s) and 16k/50k/100k/118k Glimmer anchors (4468/3986/3428/3028); use the 16k ratio below 16k and hold Glimmer's 118k rate at 122880. This models prefill-dominated work; prompt preparation, decoding, imports and fixed overhead need not scale with that ratio.

| Depth | Scaled seconds / paired example | Hours for 390 | Hours +50% | Cost +50% at assumed $1.35/h |
|---:|---:|---:|---:|---:|
| 8192 | 14.8 | 1.60 | 2.41 | $3.25 |
| 32768 | 52.7 | 5.70 | 8.56 | $11.55 |
| 65536 | 90.8 | 9.84 | 14.76 | $19.93 |
| 122880 | 146.2 | 15.84 | 23.75 | $32.07 |
| Total | | 32.98 | 49.47 | $66.79 |

$1.35/h is a declared historical rental-price assumption from the orchestrator's handoff, not a current offer. For another price P, cost is hours x P. Budget roughly **50 GPU-hours / $67 at that assumed rate**, plus setup/model/data fetch/build, pilots, archive transfer and restarts. The 50% allowance is a planning contingency, not an upper bound. The four two-example pilots add approximately 10 minutes before contingency under this scenario. The 64k/120k estimates extrapolate Qwen wall time; they are unmeasured. FACTS also shows Glimmer decode faster (73.4/68.1/61.6/59.4 tok/s at 16k/50k/100k/118k, versus Qwen 63.8/57.3/49.6 at the first three). Scaling every wall-time component as prefill is still only an estimate. Replace it with pilot `generation_seconds` plus summed arm `seconds` at each depth; QA/noise pilots do not bound all 13 tasks.

## CPU evidence and GPU gap

Use the existing offline Python environment, disk TMPDIR, and `PYTHONDONTWRITEBYTECODE=1`. The commands and exit codes are in [the run report](RESULTS-hybrid-d-r14-261103.md). Unit tests exercise pins, context/output arithmetic, batch controls, disk floors, archived analysis and restart guards. Existing eval, journal, template and RULER suites remain required. The real local Qwen3.5-0.8B harness exercises model-agnostic generation through the actual tokenizer and the three-arm routing/scoring path; it does not exercise SWA.

**Only a GPU run with the pinned 30B GGUF closes Glimmer's actual template/continuation and tokenizer execution, SWA save/restore/import with no re-reads beyond the window, CUDA batch numerical behavior, three-service VRAM admission, transient slot high-water sizing and long-context paired quality/timing.** Kit restore observations are supporting history, not this evaluator's GPU qualification. No small SWA model is available on agents1, and downloading one is forbidden. No formal surface table was supplied; exhaustive SWA/platform coverage is not claimed.

# Client partition noise: hybrid-a-r13-261101

Ready for review. Branch `m2-noise-floor` starts exactly at public M1 `0124e89a933872b54bd445921b9ac2e1c9c43a09`, fetched from `/home/agent/Werkstatt/hybrid-prototype.git main`. Implementation commit: `02607164371664a3db437bc6398b65133e292e7b` (Relux Nexus, verified signature G). No push, PR, downloads or GPU runs were performed. The requested co-author trailer is present; no session links are included.

## Change and status contract

The checker records the original imported and independent whole-prompt greedy continuations without changing the original decode order. After releasing the original four contexts, it measures each alternate local partition sequentially with three contexts, the same model/depth/input/backend/context settings and the original hybrid teacher-forcing IDs. Two default candidates use 128 and 255 tokens per batch. Effective sizes are reduced when needed to ensure decode boundaries differ from the whole prompt, from the actual same-partition import chain, and from other noise partitions. The whole prompt itself is internally chunked at 256. `--noise-chunks N ...` accepts one to four sizes in 1..256; one partition lowers cost. The selected partitions are observations, not an upper bound over all possible partitions/prompts.

Each `noise_partitions` entry reports requested/effective chunk size, batch count, alternate-to-whole `noise_kl_nats`, noise top-1 count/fraction and noise free-running first divergence. All KL uses the full vocabulary; all teacher-forced comparisons use identical continuation IDs. Free-running noise is the independently greedy alternate versus independently greedy whole prompt. Divergence indexes start at zero; -1 means no divergence within K steps. Top-level `noise_kl_nats` and `noise_kl_nats_max` report maximum KL; aggregate agreement is the minimum and divergence is the earliest across measured partitions. `kl_to_noise_ratio` is hybrid-to-whole KL / maximum noise KL. Zero noise gives JSON null with `both_zero` or `zero_noise_floor`, rather than an infinite ratio.

| Case | Applied `status_rule` | Decision |
|---|---|---|
| Homogeneous CPU | `cpu_legacy_reference` | Original same-partition K/K, KL <=0.01 and top-1 >=95%; noise does not alter status |
| Homogeneous same-backend GPU | `same_backend_same_partition_exact` | Same-partition K/K decides; whole-prompt KL, noise, ratio, top-1 and free-running divergence are diagnostics |
| Cross-backend or heterogeneous quantization | `uncalibrated_client_noise` | Always `uncalibrated`, exit 5; no numeric acceptance threshold is invented |

The GPU rule is not strictly equivalent to the CPU rule, so CPU retains the legacy rule and original metric computations. Every original CPU field and exit code matched the retained M1 binary bit-for-bit for the 132-token/8-continuation control below. This is bounded evidence, not proof for every CPU/prompt. The checker now requires at least three prompt tokens: two tokens admit only two partitions and cannot provide a third distinct reference. Supported benchmark depths are unaffected. Same-partition exactness retains the existing meaning of K/K greedy token IDs, not bitwise equality of every logit.

Legacy `kl_limit`, `top1_minimum` and `numerical_within_bounds` remain CPU-reference diagnostics on GPU/cross-backend rows. Cross-backend's old `uncalibrated_outside_reference` label becomes `uncalibrated`, still exit 5, with all values preserved. Same-backend failure remains exit 3; malformed/corrupt input remains exit 2. Neither a low KL nor a favorable noise ratio can rescue a same-backend same-partition mismatch.

The benchmark passes the noise plan and calibration identity to the checker. JSONL, the Markdown comparison table and the bundle carry each partition, maximum, ratio and applied rule. Each equivalence status line names its rule. The existing bundle whitelist and credential exclusion are preserved. No transfer format, backend kernel, model or public API was changed. README and qualification questions describe these rules.

Two noise partitions add six local prompt-prefill passes and six continuation streams; peak context count remains four. The checker still loads an extra full model while decoder/node models may remain resident. Use `--noise-chunks 128` to reduce runtime at expensive depths. GPU allocation must be checked on the target host.

## CPU evidence

Host-only CPU, existing Qwen3.5-0.8B Q4_0, no downloads. All native runtime/check invocations inherited `nice -n 10`; two compute/two batch threads, batch/ubatch 256, q8_0 KV, f32 recurrent state. Builds were incremental in the existing tree, `-j2` under `/home/agent/Werkstatt/hybrid-models/.build.lock`, using retained CMake 4.4.4. The build runner required at least 26,843,545,600 free bytes; recorded free space remained above this floor. Build tree: 131,076,572 -> 131,537,134 bytes (+460,562). CUDA/Metal/UI/prebuilt UI/CURL remained OFF. No tools were installed.

Standalone real-node fixture: 132 synthetic prompt IDs, three delta boundaries 48/96/131 plus the held final token, K=8.

| Comparison | KL nats/token | Top-1 | Free-running first divergence |
|---|---:|---:|---:|
| Hybrid / whole prompt | 1.0139874675422596e-06 | 8/8 | -1 |
| Local chunks 128 / whole prompt | 0.00055141042817293884 | 8/8 | -1 |
| Local chunks 131 / whole prompt | 1.0139874675422596e-06 | 8/8 | -1 |

Same-partition 8/8; maximum noise 0.00055141042817293884; hybrid/noise ratio 0.0018388978802995049; CPU status passed, native exit 0. Noise is finite and nonzero under forced distinct partitions. Retained M1 executable and new executable produced identical original fields and exit code. Its checker source/native core match pinned M1; the shared native library hashes below remain unchanged from the pre-task build.

The negative control XORs one byte at native body offset 1024 in the first imported delta. The same checker exits **2**, explicitly `delta checksum mismatch`, with no result JSON. This proves rejection at the checksum gate before numerical evaluation; it does not claim to exercise greedy detection of a deliberately rechecksummed corrupt native body. The nearby unmodified delta passes. `test_gpu_noise_classification` separately requires failure for same-partition mismatch regardless of the numerical-reference result, and verifies cross-backend cannot pass on either side of that reference.

Real benchmark checks, each at depth 132 and K=8, with four forced-offload calls and zero retained sessions/bytes on release:

| Link endpoint | Check mode | Same partition | Hybrid KL | Max noise KL | Hybrid/noise | Rule/status |
|---|---|---:|---:|---:|---:|---|
| enabled | fresh | 8/8 | 0.0091130111649018516 | 0.0099025743618485207 | 0.92026687524926798 | cpu_legacy_reference / passed |
| enabled | append_chain | 8/8 | 0.0091130111649018516 | 0.0099025743618485207 | 0.92026687524926798 | cpu_legacy_reference / passed |
| disabled | fresh | 8/8 | 0.0091130111649018516 | 0.0099025743618485207 | 0.92026687524926798 | cpu_legacy_reference / passed |
| disabled | append_chain | 8/8 | 0.0091130111649018516 | 0.0099025743618485207 | 0.92026687524926798 | cpu_legacy_reference / passed |

These benchmark noise partitions use effective chunk sizes 128 and 130: the 131-token candidate equals the fresh import boundary and is reduced. The standalone append fixture can use 131 because its additional 48/96 boundaries differ. This exercises selection against actual boundaries rather than assuming a fixed partition is always different.

Validation results:

- Benchmark unit suite: **25/25**, exit 0 (all existing test names retained). New tests cover GPU/CPU/cross-backend rules, invalid noise plans with a valid positive control, and JSONL/Markdown/archive preservation.
- Existing proxy suite: **68/68**, exit 0.
- Integrated all-cases run: exit 0, including native noise/corruption, original defaults, short affine chunk-size control, real timing with both link settings, LRU replay with unchanged decoder tokens, and actual 503 budget refusal/release. After extending the timing test to request checks, its focused rerun passed both real fresh/append paths with the new fields (exit 0).
- Native codec and core codec: exit 0; append/guards at existing 512-token golden fixture: exit 0. Append gives 64/64 and rejects malformed layout/recurrent endpoints without changing the slot.
- Existing distribution suite at 512 tokens: exit 0, same-partition 64/64, top-1 64/64, mean KL 3.2963e-16; its KL-only and top-1-only negative controls still reject.
- Audit: exit 0; four CPU equivalence rows, ratios/maxima, two real bundle sidecars/manifests, exact JSONL archive bytes, absence of actual fixture credentials, zero release counters, original test names and no src/include/ggml delta. Copied-record plants for false same-partition equality and nonfinite noise failed the intended named assertions.
- Repository code-style workflow's actual Python payload, Python syntax and whitespace validation: exit 0. `ty` and `editorconfig-checker` are absent (readiness exit 127); their checks are **unrun**, no download attempted.

Tests carry concise claim comments. No normative surface table or separate landing-property suite was supplied: this is a brief gap, not a claim of exhaustive coverage. No out-of-contract brief rows are silently claimed covered. GPU/CUDA/Metal and cross-backend numerics, 27B, deep-context runs, cross-prompt calibration, and valid-checksum adversarial native-state corruption remain unverified here. Independent review is pending. Changes stay within tools/hybrid-bench; there is no rework verdict or out-of-module delta.

The first unit invocation exited 1 because the new valid-plan test used a missing scratch directory; corrected setup passed. That first invocation inherited the host TMPDIR for automatically removed unit fixtures. All subsequent runs explicitly set TMPDIR to this task's `.temp/.../tmp`. The first closure audit exited 1 because it counted its own active PID; corrected audit excludes only itself and passed. These failed attempts are not passing evidence. The embedded budget-refusal benchmark exits 2 intentionally; its enclosing test passes only after checking the exact refusal and zero retained state.

## Exact local commands and logs

Run from the checkout root. Scratch is `.temp/hybrid-a-r13-261101/`. `run.py` redirects each command directly to NAME.log, records PID, UTC start/end, real exit code and free bytes in commands.jsonl; no gate is piped through tee. It sets TMPDIR to the task tmp directory and refuses builds below the stated free-byte floor. Commands below are the actual child commands, including diagnostic repeats and unavailable-tool probes.

`bench-unit-02.log` - exit 0, 2026-11-01T11:37:44.519724+00:00 to 2026-11-01T11:37:44.620933+00:00:

```sh
nice -n 10 python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v
```

`build-01.log` - exit 0, 2026-11-01T11:37:44.648240+00:00 to 2026-11-01T11:37:52.759286+00:00:

```sh
nice -n 10 flock /home/agent/Werkstatt/hybrid-models/.build.lock .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-hybrid-prefill llama-hybrid-check llama-hybrid-check-codec llama-hybrid-check-append llama-hybrid-check-guards llama-hybrid-check-distributions llama-hybrid-check-client llama-hybrid-bench-check llama-server
```

`build-02.log` - exit 0, 2026-11-01T11:38:50.487854+00:00 to 2026-11-01T11:38:54.450600+00:00:

```sh
nice -n 10 flock /home/agent/Werkstatt/hybrid-models/.build.lock .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-hybrid-bench-check
```

`noise-native-01.log` - exit 0, 2026-11-01T11:38:54.479359+00:00 to 2026-11-01T11:39:07.575356+00:00:

```sh
nice -n 10 python3 tools/hybrid-bench/test_integrated.py --case noise --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --checker-binary build/bin/llama-hybrid-bench-check --baseline-checker .temp/hybrid-a-r13-261101/check-baseline --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r13-261101/noise-native-01
```

`integrated-01.log` - exit 0, 2026-11-01T11:39:15.691319+00:00 to 2026-11-01T11:39:54.689761+00:00:

```sh
nice -n 10 python3 tools/hybrid-bench/test_integrated.py --case all --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --checker-binary build/bin/llama-hybrid-bench-check --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r13-261101/integrated-01
```

`bench-unit-03.log` - exit 0, 2026-11-01T11:40:05.318370+00:00 to 2026-11-01T11:40:05.443054+00:00:

```sh
nice -n 10 python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v
```

`codec-01.log` - exit 0, 2026-11-01T11:40:05.469501+00:00 to 2026-11-01T11:40:05.485052+00:00:

```sh
nice -n 10 build/bin/llama-hybrid-check-codec
```

`core-codec-01.log` - exit 0, 2026-11-01T11:40:05.512087+00:00 to 2026-11-01T11:40:05.524632+00:00:

```sh
nice -n 10 build/bin/llama-hybrid-check --codec
```

`append-512-01.log` - exit 0, 2026-11-01T11:40:05.548583+00:00 to 2026-11-01T11:40:12.251035+00:00:

```sh
nice -n 10 build/bin/llama-hybrid-check-append /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden/prompt-512
```

`guards-512-01.log` - exit 0, 2026-11-01T11:40:12.277874+00:00 to 2026-11-01T11:40:13.919800+00:00:

```sh
nice -n 10 build/bin/llama-hybrid-check-guards /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden/prompt-512
```

`proxy-unit-01.log` - exit 0, 2026-11-01T11:38:56.754222+00:00 to 2026-11-01T11:40:25.172945+00:00:

```sh
nice -n 10 python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v
```

`style-01.log` - exit 0, 2026-11-01T11:41:04.470244+00:00 to 2026-11-01T11:41:04.498811+00:00:

```sh
nice -n 10 python3 .temp/hybrid-a-r13-261101/code_style.py
```

`whitespace-01.log` - exit 0, 2026-11-01T11:41:04.533195+00:00 to 2026-11-01T11:41:04.539443+00:00:

```sh
git diff --check
```

`distributions-512-01.log` - exit 0, 2026-11-01T11:41:04.576738+00:00 to 2026-11-01T11:41:17.041090+00:00:

```sh
nice -n 10 build/bin/llama-hybrid-check-distributions /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden/prompt-512.tokens.json
```

`timing-checks-02.log` - exit 0, 2026-11-01T11:40:39.140890+00:00 to 2026-11-01T11:41:24.373464+00:00:

```sh
nice -n 10 python3 tools/hybrid-bench/test_integrated.py --case timing --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --checker-binary build/bin/llama-hybrid-bench-check --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r13-261101/timing-checks-02
```

`python-syntax-01.log` - exit 0, 2026-11-01T11:41:50.782991+00:00 to 2026-11-01T11:41:50.814659+00:00:

```sh
nice -n 10 python3 -c 'import ast; from pathlib import Path; files=["hybrid_bench.py","test_bench.py","test_integrated.py"]; [ast.parse((Path("tools/hybrid-bench")/f).read_text(),filename=f) for f in files]; print("PASS Python syntax:", ", ".join(files))'
```

`ty-readiness-02.log` - exit 127, 2026-11-01T11:41:50.840585+00:00 to 2026-11-01T11:41:50.849401+00:00:

```sh
bash -c 'ty --version'
```

`editorconfig-readiness-02.log` - exit 127, 2026-11-01T11:41:50.877743+00:00 to 2026-11-01T11:41:50.879358+00:00:

```sh
bash -c 'editorconfig-checker --version'
```

`audit-01.log` - exit 0, 2026-11-01T11:43:10.576556+00:00 to 2026-11-01T11:43:10.626730+00:00:

```sh
nice -n 10 python3 .temp/hybrid-a-r13-261101/audit.py
```

`whitespace-02.log` - exit 0, 2026-11-01T11:44:13.846600+00:00 to 2026-11-01T11:44:13.856463+00:00:

```sh
git diff --check
```

`closure-01.log` - exit 1, 2026-11-01T11:45:06.979275+00:00 to 2026-11-01T11:45:07.081955+00:00:

```sh
nice -n 10 python3 .temp/hybrid-a-r13-261101/closure.py
```

`closure-02.log` - exit 0, 2026-11-01T11:45:14.516546+00:00 to 2026-11-01T11:45:14.537552+00:00:

```sh
nice -n 10 python3 .temp/hybrid-a-r13-261101/closure.py
```

The initial unledgered unit command was `nice -n 10 python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v` (bench-unit-01.log, exit 1). Scratch scripts and private fixtures are retained for local review and must not be shared wholesale; send only sanitized benchmark bundles and reports.

Measured artifact identities:

| Artifact | SHA-256 |
|---|---|
| build/bin/llama-hybrid-bench-check | `51f2f8a15356601d94b70996af8b3ae7b31ed9146820bfe4040cb16334288ad7` |
| build/bin/libllama.so | `84128f68790f1cf122c786a1a55ca140d9ac1a97415f3d3d523b437e43d2b63e` |
| build/bin/libggml-cpu.so | `2fd4283f9615f258b4a3c3e74e93e69961d545e2314f70dad674846e5e8b1fdb` |
| /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf | `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c` |

Runtime closure at 2026-11-01T11:45:14.534254+00:00: all 43 previously recorded task PIDs were inactive; the audit's own current PID was excluded. Foreign runtimes using shared binaries were observed and left untouched. No shared lock, model, cache or other agent process was removed/stopped.

## GPU commands for the operator

Use Bash on the rented Linux CUDA host with this branch already transferred/checked out, an installed CUDA build toolchain, enough free disk/VRAM and the existing local GGUF. No model/tool download is required. Node and client must use the same model path for this homogeneous test. Run the 0.8B block first; stop it before the 27B run. The launchers build under the same named advisory lock and record actual model/binary/backend identities. Do not use --skip-build with a record from M1.

```bash
git switch m2-noise-floor
git merge-base --is-ancestor 0124e89a933872b54bd445921b9ac2e1c9c43a09 HEAD
git merge-base --is-ancestor 02607164371664a3db437bc6398b65133e292e7b HEAD
git verify-commit HEAD

# Set this to the existing 0.8B file on the GPU host.
MODEL=/absolute/path/Qwen3.5-0.8B-Q4_0.gguf
LABEL=r13-08b-cuda
CTX=65536
DEPTHS=(128 39500)
R13_GPU_ROOT="$PWD/.temp/$LABEL"
mkdir -p "$R13_GPU_ROOT/output"

nice -n 10 tools/hybrid-bench/deploy-prefill-node.sh \
  --state "$R13_GPU_ROOT/node" --build-dir build \
  --build-lock "$R13_GPU_ROOT/build.lock" --offline --cuda \
  --model "$MODEL" --ctx "$CTX" --port 8091 --bench-link

nice -n 10 tools/hybrid-bench/setup-client.sh \
  --state "$R13_GPU_ROOT/client" --build-dir build \
  --build-lock "$R13_GPU_ROOT/build.lock" --offline --cuda \
  --model "$MODEL" --ctx "$CTX" \
  --node-info "$R13_GPU_ROOT/node/node-info.json" \
  --token-file "$R13_GPU_ROOT/node/client-tokens.json" \
  --prefill-url http://127.0.0.1:8091 --decode-port 8090 --port 8092

R13_BENCH_RC=0
nice -n 10 python3 tools/hybrid-bench/hybrid_bench.py \
  --client-state "$R13_GPU_ROOT/client" --scratch "$R13_GPU_ROOT" \
  --output "$R13_GPU_ROOT/output" --label "$LABEL" \
  --smoke --depths "${DEPTHS[@]}" --check-depths "${DEPTHS[@]}" \
  --modes local hybrid offload --check --check-tokens 64 \
  --noise-chunks 128 255 || R13_BENCH_RC=$?

tools/hybrid-bench/setup-client.sh --state "$R13_GPU_ROOT/client" --stop
tools/hybrid-bench/deploy-prefill-node.sh --state "$R13_GPU_ROOT/node" --stop
printf 'benchmark exit: %s\n' "$R13_BENCH_RC"
test "$R13_BENCH_RC" -eq 0
```

For 27B, rerun the deployment/setup/benchmark/stop block after replacing only these settings (choose the actual existing filename):

```bash
MODEL=/absolute/path/Qwen3.5-27B-Q4_K_M.gguf
LABEL=r13-27b-cuda
CTX=132096
DEPTHS=(128 39500 127500)
R13_GPU_ROOT="$PWD/.temp/$LABEL"
mkdir -p "$R13_GPU_ROOT/output"
```

The requested rungs target the original 132/39504/127504 prompt depths with the current smoke chat framing; confirm actual JSONL depth instead of inferring it from CLI input. At each reached depth both fresh and append_chain checks should show same-partition 64/64 and rule same_backend_same_partition_exact. Whole-prompt KL may exceed 0.01 without failure if K/K holds. Inspect each actual noise partition and maximum/ratio, including free-running divergence. A GPU allocation failure is an aborted run, not numerical equivalence failure. The checker adds a full model copy and up to four contexts; node and decoder remain resident on the shared GPU.

For CUDA-node/Metal-client verification, start/keep the CUDA node using the node command above and securely copy its node-info/token files to the Mac (token file mode 0600). Use the identical GGUF there. Keep the node running until after the Mac check. With the branch/toolchain already on the Mac, run:

```bash
# Mac: existing local GGUF, securely copied node-info/token, normal SSH host alias.
MODEL=/absolute/path/Qwen3.5-0.8B-Q4_0.gguf
R13_METAL_ROOT="$PWD/.temp/r13-metal"
mkdir -p "$R13_METAL_ROOT/output"
chmod 600 /secure/copied/client-tokens.json
# Keep this tunnel open in a separate terminal:
# ssh -N -L 8091:127.0.0.1:8091 rented-host

tools/hybrid-bench/setup-client.sh \
  --state "$R13_METAL_ROOT/client" --build-dir build \
  --build-lock "$R13_METAL_ROOT/build.lock" --offline \
  --model "$MODEL" --ctx 65536 \
  --node-info /secure/copied/node-info.json \
  --token-file /secure/copied/client-tokens.json \
  --prefill-url http://127.0.0.1:8091

R13_METAL_RC=0
python3 tools/hybrid-bench/hybrid_bench.py \
  --client-state "$R13_METAL_ROOT/client" --scratch "$R13_METAL_ROOT" \
  --output "$R13_METAL_ROOT/output" --label r13-cuda-metal \
  --smoke --depths 128 39500 --check-depths 128 39500 \
  --check --check-tokens 64 --noise-chunks 128 255 || R13_METAL_RC=$?
tools/hybrid-bench/setup-client.sh --state "$R13_METAL_ROOT/client" --stop
printf 'cross-backend exit: %s\n' "$R13_METAL_RC"
test "$R13_METAL_RC" -eq 5
```

Stop the CUDA node on its host afterwards using its state path. Cross-backend checks must say uncalibrated_client_noise / uncalibrated, exit 5, while reporting hybrid-to-client-whole KL, client local noise and ratio. Do not interpret a favorable ratio as calibrated acceptance. Return only each output tar.gz and sha256 sidecar; never return private runtime state or credential files. Verify SHA256SUMS members after extraction.

## Report verification

`nice -n 10 python3 .temp/hybrid-a-r13-261101/report_check.py` exited 0 at 2026-11-01T11:48Z (report-check-02.log). It checked the exact persisted numerical strings, absence of private fixture credential values, syntax of every shell block via `bash -n`, and absence of session links. Attempt 01 exited 1 on an overly literal text assertion (`exit` versus the report's `exits`); the assertion was corrected without changing numerical evidence. Final whitespace check exited 0. Report-only changes followed the signed implementation; no measured source or binary changed.

## r15 - logit-level same-partition rework (hybrid-a-r15-261101)

Ready for review. This section supersedes the r13 top-1-only acceptance rule and its checksum-only corruption coverage. r14 was finished and signed on its separate `m2-fast-hash` branch at `41eb6ca445d77d907e21a981b8f82758f006c784` before r15 began. The clean `m2-noise-floor` branch continued from `c3d5c7fae4ca4565b30742e0909cc1cc0c72b41c` and rebased cleanly with `git rebase --gpg-sign` onto requested public main `7e1a477b2e8dfc8146744f4f81d73abca25e7d4a`. Rebased r13 commits are `15b33beb3e` and `2ea459cf7e`, both signature G; the r15 signed commit follows them. All carry the requested Codex co-author trailer. No push, PR, session link, model download, package install or GPU execution.

The supplied r6f review was read from `/home/agent/Werkstatt/hybrid-review-proto/llama.cpp/.temp/REVIEW-noise-floor-claude.md`. Scope is the existing `tools/hybrid-bench` module. There is no active task-board in this checkout (`goal get` exit 1, `.task-board` missing); the existing `.temp/tasks.md` tracks the work. No board goal was changed or cleared.

### F1: decisive logit rule

For each of K teacher-forced continuation positions, the checker now compares every imported logit with the independent local computation of the identical delta-boundary partition. It reports mean `same_partition_kl_nats` (imported || local), `same_partition_max_abs_logit_diff` across all checked positions/vocabulary entries, and the individual position measurements. `same_partition_exact` is an actual byte comparison of finite float arrays. It no longer means argmax agreement. Equal finite bit patterns report KL zero directly, avoiding tiny cancellation residuals in the KL expression.

A second fresh local context independently repeats that same partition and the same teacher-forced continuation. Its KL, maximum difference and exactness are reported, including per-position values. When the repeat is bit-identical, the decisive rule is `same_backend_same_partition_exact`. When it differs, the rule is `same_backend_same_partition_measured_floor`: at each position, imported/local KL AND maximum absolute difference must not exceed that position's observed repeat/local values. An exact repeat at an individual position still requires byte equality there. There is no epsilon, multiplier, alternate-partition tolerance or pooled maximum across positions. K/K argmax agreement remains an additional condition. CPU also retains the whole-prompt KL <=0.01 and top-1 >=0.95 reference bounds. CPU acceptance is intentionally strengthened: an argmax-preserving lossy delta that passed the old CPU rule now fails.

On this CPU fixture, all eight independent-repeat logit arrays and imported/local arrays are bit-identical, with KL=0 and max difference=0. Exactness is therefore measured here, rather than inferred from greedy output. The nonzero-repeat branch has synthetic positive/negative startup controls for inclusive bounds, KL excess, max-difference excess and exact-reference rejection; Python tests cover propagation of a nonexact-but-within-floor verdict. No GPU nonreproducibility observation is claimed. One independent repeat is a bound for this run, not a statistical calibration of a backend or a proof about unobserved logits/native state.

### F2 and N1-N3: rejection evidence

Both mutations start after body byte 4096 and recompute canonical XXH3-128 and SHA-256 trailers using the existing proxy `stub_server.xxh128` test helper and hashlib. The original trailer is verified before mutation. The existing system libxxhash dependency is reused; no dependency was added. Native metrics/status exist for both cases, so neither a checksum failure nor import rejection accounts for their exit 3.

| Delta / control | Same-partition argmax | Same-partition KL | Maximum absolute logit difference | Exit and reason |
| --- | ---: | ---: | ---: | --- |
| Clean | 8/8 | 0 | 0 | 0: passed, exact logits |
| Checksum-valid corrupt: stride 1031, xor 0x40 | 0/8 | 7.18231430791 | 20.1812210083 | 3: failed, same-partition logit criterion |
| Checksum-valid lossy: stride 4099, xor 0x01 | 8/8 | 0.00058444492864 | 0.418376684189 | 3: failed, same-partition logit criterion |
| One body byte changed, old trailer | not evaluated | not evaluated | not evaluated | 2: delta checksum mismatch, no JSON metrics |
| CPU build requested GPU layers=1 | not evaluated | not evaluated | not evaluated | 2: no GPU offload device, no JSON metrics |
| CPU computation labelled CUDA | not evaluated | not evaluated | not evaluated | 2: backend label/device mismatch, no JSON metrics |

The lossy row retains all eight argmax tokens and remains inside the old whole-prompt numerical bounds, yet fails. This directly narrows the F1 hole. GPU_LAYERS >0 on this CPU-only build now refuses early rather than simulating a GPU acceptance rule. Actual positive-layer GPU corruption runs remain unrun under the CPU-only instruction.

N1: the checker verifies `llama_supports_gpu_offload()`, loaded model devices and their actual backend registration, refuses mixed device backends, and compares that backend with the client label. CPU contexts disable both KQV and operation offload. The Python classifier also refuses mismatches, including cross-backend measurements. N2: missing logit/noise fields give an explicit incompatible-checker/rebuild refusal immediately after reading metrics. The bundle entry point independently checks before writing JSONL. `test_checker_version_before_classification` proves refusal before classification and remote-session release; `test_old_checker_fields` proves no partial output. N3: noise partitions must have a different last boundary before the endpoint from the hybrid. At 132 tokens, requested chunks 128/255 select 128/130 with final prefill boundaries 128/130, rather than hybrid 131; the original 131 candidate is refused.

Cross-backend and hetero-quant measurements remain uncalibrated, exit 5, with no acceptance threshold inferred from the measured noise.

### N4: measured cost and snapshot variant

Final short CPU fixture: 132 prompt tokens, K=8, two partitions. `noise_reference_seconds` was **2.744668 s**; independent-repeat prefill/decode calls added **0.591282 s**. The noise timer includes context construction, prefills, snapshot/restore, comparisons and all noise decode steps; the repeat timer excludes context construction and numerical comparison overhead. Timings vary with the shared host workload and are diagnostics, not test thresholds.

The alternate free-running stream now loads a native state snapshot instead of prefilling the same prompt again. Its first greedy token is cached separately because sequence snapshots do not restore logits. A task-local comparison executable restoring the duplicate-prefill path produced exactly the same partition KL/top-1/divergence measurements (audit exit 0); its noise loop took 4.416972 s. This single, non-interleaved comparison is not a calibrated speedup measurement.

Each noise partition now adds two prompt-prefill passes instead of three, with 3K decode steps. The new independent repeat adds one prompt-prefill pass and K decode steps. Two default partitions add four noise prefills plus the repeat; the original r13 noise added six. Peak comparison context count increases from four to five. During the later sequential noise loop there are three contexts plus one temporary native snapshot. The whole-prompt reference is still independently prefilled per partition. No 27B/128k GPU wall-clock estimate is presented as measured; that remains target-host qualification. `--noise-chunks 128` bounds expensive runs to one alternate partition.

### Verification and narrowing controls

31 benchmark unit tests and 68 proxy unit tests pass. The final full integrated suite passes with clean/valid-corrupt/valid-lossy/checksum/device controls, native defaults and bounded chunk slope, fresh and append-chain checked offloads with link disabled/enabled, exact LRU replay, and native budget refusal. Four actual equivalence rows preserve the new fields, per-position arrays and rules through JSONL, Markdown and sanitized archives. Three bundle manifests and sidecars verify, archive bytes match local public artifacts, and release rows report zero sessions/bytes. The retained M1 executable gives identical original numerical metrics and exit on the clean fixture; this does not claim legacy acceptance for corrupt inputs.

| Mutant | Property narrowed | Named failing test | Result / surviving bound |
| --- | --- | --- | --- |
| Native `passed` drops `within_floor` | Lossy import cannot pass on 8/8 argmax and small whole-prompt KL | `test_equivalence_noise`, `valid-lossy` assertion | Killed: suite exit 1; mutant falsely exits 0 with same_partition_within_floor=false |
| Python classifier drops `same_partition_within_floor` | GPU-label classification must retain native logit verdict | `test_gpu_noise_classification` | Killed: assertion failure, intended test exit 1 |
| Cross-backend returns passed/0 | Cross-backend cannot inherit calibrated acceptance | `test_cross_backend_classification` | Killed: assertion failure, intended test exit 1 |
| Ignore backend label | Label mismatch cannot select another device's rule | `test_checker_backend_mismatch` | Killed: expected Refusal absent, intended test exit 1 |

No surviving code mutant is claimed in this bounded set. The duplicate-prefill executable is a positive equivalence/performance control, not a mutant of a forbidden behavior. These controls do not measure a nonreproducible GPU backend. The mutation compilation and native build use the shared `/home/agent/Werkstatt/hybrid-models/.build.lock`, nice 19, at most -j2 and an inside-lock >=26843545600-byte guard. The initial build also rebuilt the existing node/server. UI download and local UI build flags were OFF; the existing stale-assets warning is retained and no retrieval ran. Final checker build exit 0. The pre-existing signed/unsigned depth-comparison warning is retained.

Self-review used the code-review skill's scope, security, general and tests checklists. Changed paths stay within `tools/hybrid-bench`; no src/include/ggml change or previous test removal/rename. Python syntax, ASCII/newline/trailing-space audit, the repository code-style workflow payload and whitespace checks pass. `ty` and `editorconfig-checker` are absent (readiness exit 1); type/editorconfig checks are unrun, with no install attempted. There is no separate external landing-property suite or normative surface table in this brief; the review's observational rows are mapped below. Independent acceptance/review and landing are pending; no hosted check or landing is claimed for a no-push run.

| Review row / request | Evidence and finite limit |
| --- | --- |
| 1, N3: distinct partitions, same positions | `test_equivalence_noise`: 128/130 vs hybrid 131; per-position teacher forcing; short fixture only |
| 2a, F1: decisive same-backend logits | Native lossy negative plus argmax-only mutant; real GPU execution unrun |
| 2b: CPU behavior | Clean import/repeat bit-identical; retained M1 original numerical metrics match; corrupt acceptance deliberately strengthened |
| 2c: uncalibrated | `test_cross_backend_classification` and calibrated-cross-backend mutant; actual cross-device run unrun |
| 3, N1/N2: reporting/device/version | Bundle roundtrip/audit, device mismatch mutant, actual CPU refusals, old-checker tests; GPU-specific device registration execution unrun |
| 4, F2: valid-checksum negatives | `test_equivalence_noise`: clean 0, both imported mutations 3, lossy still 8/8; native mutant killed |
| 5, N4: extra references | Measured short CPU times and duplicate-prefill positive control; 27B/128k GPU timing is outside this CPU-only execution |

No other requested row is treated as out of contract. Long-context, 27B, CUDA/Metal compilation/execution, cross-device numerics and nondeterministic-backend empirical calibration were not run. The native measured-floor startup controls cover its scalar decision rule, not GPU reproducibility. Testing is bounded to local 0.8B CPU fixtures.

Failed/corrected attempts are preserved: `noise-01` exit 1 asserted exact-zero KL before the identical-array roundoff shortcut; `integrated-01` exit 1 still expected the old CPU Markdown rule name (corrected, then integrated-02/03 exit 0); `python-mutants-01` exit 1 detected two mutants but its device control used an incomplete fixture and hit KeyError. The fixture was completed and all three intended assertion failures are verified in python-mutants-02 (wrapper exit 0). Native argmax-mutant suites intentionally exit 1 and are never described as green product runs. Embedded budget benchmarks intentionally exit 2; their enclosing test asserts the precise budget reason and zero retained sessions.

### Reproduction and private evidence

All local evidence is under `.temp/hybrid-a-r15-261101/`. `commands.jsonl` records exact argv, PID, UTC start/end, free bytes and real exit code; each command ran directly with a separate numbered log, without tee. `tested-source.json` binds reviewed source, binaries, libllama and CMake configuration. `compile_variants.py` records compiler argv and PIDs for the task-only controls. `audit.py` validates the real negative exits, snapshot equivalence, bundle manifests, original test names and process closure; audit exit 0. At its observation time no recorded task PID was alive. Native delta fixtures remain private for reproduction; credentials are not public bundle members.

Core reproduction commands (existing binaries, no build or download):

```sh
nice -n 19 python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v
nice -n 19 python3 -m unittest discover -s tools/hybrid-proxy -p 'test*.py' -v
nice -n 19 python3 tools/hybrid-bench/test_integrated.py --case all --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --checker-binary build/bin/llama-hybrid-bench-check --model "$HOME/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/r15-rerun
```

Command ledger through the review audit (logs contain complete output; compiler variants record nested PIDs/argv inside their build log):

| Log | Exit | UTC start | UTC end |
| --- | ---: | --- | --- |
| unit-01.log | 0 | 2026-11-01T12:10:19.006003+00:00 | 2026-11-01T12:10:19.158465+00:00 |
| build-01.log | 0 | 2026-11-01T12:10:19.193412+00:00 | 2026-11-01T12:10:31.265305+00:00 |
| noise-01.log | 1 | 2026-11-01T12:10:43.890297+00:00 | 2026-11-01T12:10:52.099266+00:00 |
| build-02.log | 0 | 2026-11-01T12:11:20.101238+00:00 | 2026-11-01T12:11:24.724733+00:00 |
| noise-02.log | 0 | 2026-11-01T12:11:43.631366+00:00 | 2026-11-01T12:12:10.444810+00:00 |
| unit-02.log | 0 | 2026-11-01T12:12:15.581033+00:00 | 2026-11-01T12:12:15.820646+00:00 |
| integrated-01.log | 1 | 2026-11-01T12:12:21.953320+00:00 | 2026-11-01T12:13:11.232794+00:00 |
| build-controls-01.log | 0 | 2026-11-01T12:13:22.815393+00:00 | 2026-11-01T12:13:32.094947+00:00 |
| proxy-unit-01.log | 0 | 2026-11-01T12:12:21.957598+00:00 | 2026-11-01T12:13:48.149198+00:00 |
| build-03.log | 0 | 2026-11-01T12:13:54.491885+00:00 | 2026-11-01T12:14:01.302141+00:00 |
| argmax-mutant-01.log | 1 | 2026-11-01T12:13:55.635597+00:00 | 2026-11-01T12:14:20.484389+00:00 |
| python-mutants-01.log | 1 | 2026-11-01T12:14:45.136308+00:00 | 2026-11-01T12:14:45.270733+00:00 |
| snapshot-reference-01.log | 0 | 2026-11-01T12:14:45.312359+00:00 | 2026-11-01T12:14:53.591645+00:00 |
| python-mutants-02.log | 0 | 2026-11-01T12:14:59.088078+00:00 | 2026-11-01T12:14:59.200598+00:00 |
| unit-03.log | 0 | 2026-11-01T12:14:59.229761+00:00 | 2026-11-01T12:14:59.512893+00:00 |
| integrated-02.log | 0 | 2026-11-01T12:14:46.476498+00:00 | 2026-11-01T12:16:12.303218+00:00 |
| ty-readiness-01.log | 1 | 2026-11-01T12:16:15.403403+00:00 | 2026-11-01T12:16:15.406055+00:00 |
| editorconfig-readiness-01.log | 1 | 2026-11-01T12:16:15.440469+00:00 | 2026-11-01T12:16:15.442857+00:00 |
| build-04.log | 0 | 2026-11-01T12:16:14.250475+00:00 | 2026-11-01T12:16:20.222963+00:00 |
| unit-04.log | 0 | 2026-11-01T12:16:53.050879+00:00 | 2026-11-01T12:16:53.335109+00:00 |
| style-01.log | 0 | 2026-11-01T12:16:53.361292+00:00 | 2026-11-01T12:16:53.387065+00:00 |
| build-controls-02.log | 0 | 2026-11-01T12:16:53.413404+00:00 | 2026-11-01T12:17:06.332450+00:00 |
| snapshot-reference-02.log | 0 | 2026-11-01T12:17:39.230876+00:00 | 2026-11-01T12:17:48.448927+00:00 |
| argmax-mutant-02.log | 1 | 2026-11-01T12:17:38.036677+00:00 | 2026-11-01T12:18:04.701212+00:00 |
| integrated-03.log | 0 | 2026-11-01T12:16:54.201594+00:00 | 2026-11-01T12:18:19.971120+00:00 |
| audit-01.log | 0 | 2026-11-01T12:19:00.715291+00:00 | 2026-11-01T12:19:00.807657+00:00 |

After self-review, the clean M1 control was strengthened to compare every original M1 result field, not just the selected numerical metrics; its repeat in `noise-03` passes with exit 0. The final benchmark unit rerun has 31 passing tests. The final audit refreshes tested-source hashes and confirms no recorded PID remains alive. No runtime behavior changed during this assertion/comment cleanup.

| Additional log | Exit | UTC start | UTC end |
| --- | ---: | --- | --- |
| whitespace-01.log | 0 | 2026-11-01T12:20:53.201394+00:00 | 2026-11-01T12:20:53.206352+00:00 |
| unit-05.log | 0 | 2026-11-01T12:22:16.742866+00:00 | 2026-11-01T12:22:16.982214+00:00 |
| noise-03.log | 0 | 2026-11-01T12:22:17.004517+00:00 | 2026-11-01T12:22:44.733059+00:00 |
| audit-02.log | 0 | 2026-11-01T12:23:12.979822+00:00 | 2026-11-01T12:23:13.050937+00:00 |

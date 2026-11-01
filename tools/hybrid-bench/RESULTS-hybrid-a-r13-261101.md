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

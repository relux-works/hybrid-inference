# M1 node limits qualification

Run: hybrid-a-r10-261031. Branch: `m1-node-limits`, based on `b/m1-hybrid-bench` at `20964cee820951d904606354e4049b64bcf4104c`. Local signed delivery only; no push or upstream submission. CPU node-limit qualification is ready for review within the finite coverage below. No GPU or hosted acceptance is claimed.

## Scope and resulting behavior

Changes are confined to `tools/hybrid/`. No changes to the engine, proxy, benchmark or launcher. Implements `--max-ctx` (ceiling default 16384), `--ram-budget-mib` (512 total across sessions), `--max-request-mib` (1), global idle LRU eviction with response-lifetime pinning, atomic impossible-budget refusal, and model identity plus counters on health/metrics. Existing positional allocation default remains 8448; with no positional context, an explicit `--max-ctx` selects that allocation. Explicit positional context cannot exceed the ceiling.

An evicted ID returns 404. Reads and accepted extends refresh logical LRU order. The scheduler serializes context access, and streaming responses retain a session pin until completion/disconnect. A queued request has not acquired its session and can observe 404 after eviction. Admission finds enough unpinned victims before erasing any. If it cannot fit, it returns `503 {"error":"ram budget","need":BYTES,"budget":BYTES}` without committing any session mutation or eviction. `need` includes the new session state and retained pinned states after every eligible idle state is hypothetically removed.

The retained-state budget counts serialized state and four-byte token IDs. Model weights, live context, empty-state template, allocations and temporary old/new state, token, JSON and wire buffers remain outside that counter. Rejection is after inference/serialization and does not bound transient RSS. Wire limits remain exactly 1 MiB header and 512 MiB body. Resident contexts and host/disk tiers are design-only in [NODE-STATE-DESIGN.md](NODE-STATE-DESIGN.md).

`GET /v1/health` contains JSON fields `max_ctx`, `n_ctx`, `ram_budget_bytes`, `ram_bytes`, `sessions`, `evictions`, `max_request_bytes`, `model_sha256`, `layout_fingerprint`, `model_layout`. Prometheus `/metrics` has corresponding `hybrid_` numeric fields and a `hybrid_model_info` label row. Counters are individually atomic, not one transactionally consistent multi-counter observation.

The layout SHA uses compact JSON with recursively sorted object keys. The actual model projection includes architecture, layer count, embedding size, per-layer query/KV head counts and dimensions, recurrent-layer bitmap and SSM dimensions, vocabulary count and a SHA-256 of vocabulary bytes in token-ID order with little-endian uint64 length framing. Quantization/file name are excluded. The independent benchmark GGUF reader agrees on every layout field and the canonical hash. Exact GGUF SHA remains mandatory for native state; matching layout alone does not establish numerical equivalence across quantizations.

## Environment and reproduction

CPU-only Linux x86_64, GNU C++ 15.2.0, Release, CMake 4.4.4, Python 3.14. No downloads or installations. Reused the existing configured CPU build. Every compile/configure is protected by the shared build lock, checks exact available bytes against 26,843,545,600 first, and compiles at `-j2`. The isolated negative-control executable compiles only the node translation unit and links the already-built libraries; it does not rebuild the engine.

Set these paths for reproduction; private machine paths and credentials are deliberately excluded from this public report:

```sh
MODEL_CACHE=/path/to/existing/model-cache
MODEL="$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf"
REQUANTIZED="$MODEL_CACHE/Qwen3.5-0.8B-Q8_0-requant.gguf"
GOLDEN="$MODEL_CACHE/golden"
CMAKE=.temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake
RUN=.temp/hybrid-a-r10-261031
# All actual argv, times, PIDs and exits are in the private $RUN/commands.jsonl.
flock "$MODEL_CACHE/.build.lock" bash -c 'available=$(df -B1 --output=avail . | tail -n1); echo "available_bytes=$available"; test "$available" -ge 26843545600 || exit 75; exec .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-hybrid-prefill llama-hybrid-check llama-hybrid-check-codec llama-hybrid-check-append llama-hybrid-check-guards llama-hybrid-check-distributions llama-hybrid-check-client'
build/bin/llama-quantize --allow-requantize "$MODEL" "$REQUANTIZED" Q8_0 2
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL" --artifacts "$RUN/limits"
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL" --requantized "$REQUANTIZED" --artifacts "$RUN/identity-02" --case identity
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL" --artifacts "$RUN/deep" --case deep
build/bin/llama-hybrid-check-distributions "$MODEL" "$RUN/deep/deep.tokens.json" --max-ctx 65536
```

The actual lock commands also execute the byte-floor guard inside the lock; see the command appendix and private logs for every guard and exit. The model cache retains the explicitly authorized local re-quantization. Native state fixtures are task-local, hashed and deleted after the gates.

## Results

| Gate / named claim | Evidence | Actual exit and result |
|---|---|---|
| Three sessions / small budget, LRU refresh, 404 and replay | `limits-final-01.log`, `test_lru_replay_and_atomic_refusal` | exit 0; replay tokens and native snapshot bytes identical |
| Impossible extend preserves every session | same final gate, `test_lru_replay_and_atomic_refusal` | Repeated 503 with need 59,538,100, budget 50,331,648; target and both neighbors byte-identical; later smaller append accepted |
| Oldest pinned snapshot and extend survive competing eviction | `limits-final-01.log`, `test_pinned_read_and_extend` | exit 0; active oldest response survives, idle other-client neighbor evicted |
| All-pinned refusal / disconnect release | same gate | exit 0; exact 503, no mutation/eviction, then disconnect permits a valid append |
| Inclusive request cap, one-over and larger configured cap | `limits-final-01.log`, `test_request_caps_and_metrics` | exit 0; 1 MiB and 2 MiB accepted exactly; cap+1 is 413 and unchanged snapshot |
| CLI limits | same gate, `test_invalid_limits` | parent exit 0; seven invalid startups each exit 1 as expected |
| Metrics | same gate | default values 16384 / 536870912 / 1048576 and exact numeric/identity agreement between health and Prometheus |
| Header valid maximum and one-over, default and long configured contexts | `scheduler-default-01.log`, `codec-65536-02.log`, `codec-canonical-01.log` | all exit 0; exact budgets 184320, 724992 and 1048576 bytes accepted; writer and reader reject one byte over |
| Local Q4_0 to Q8_0 re-quantization | `quantize-01.log` | exit 0, two threads; retained in local cache |
| Quantization-independent identity | `identity-02.log`, `identity-reference-01.log` | both exit 0; actual GGUF SHAs differ, every layout field and fingerprint match |
| Different geometry identity | `native-01.log` and codec controls | exit 0; native projected embedding change and synthetic field changes differ; independent second base model unavailable |
| 40000-token HTTP transfer and numerical references | `deep-http-01.log`, `deep-distributions-01.log` | both exit 0; exact results below |
| Existing HTTP, native, malformed and distribution regressions | command ledger and appendix | all final gates exit 0; earlier fixture failures remain recorded |

Canonical small-model layout fingerprint: `33d9f73396507b1f8402bf1e439936f42caee2c8875fb46e127b06e0e656d2fa`. Original GGUF SHA-256: `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`. Re-quantized SHA-256: `f543fad263339779c3a38a21a531c6adbffcfdecdf5f749dd542de924b1e7302`; file size 811,843,840 bytes.

## Controls and corrections

| Mutant / control | Property narrowed by it | Named failing test | Survival bound |
|---|---|---|---|
| Eligibility ignores active response pins, while still excluding the current extender | LRU must skip a pinned oldest session, not merely happen to select another older victim | `test_pinned_read_and_extend`, snapshot leg: active session GET returns 404 instead of 200 | Killed: isolated `unpinned-control-01`, actual exit 1 on the intended assertion. No exhaustive mutation claim |
| Reversed token-distribution / KL-only / top1-only controls | Distribution checker rejects both metric violations, with an identical positive control and exact threshold boundaries | `controls()` in the distribution checker | Existing executable controls; final numerical gate result below |

First build exited 2 because the old CMake build graph lacked the benchmark-check target. No compiler failure; explicit configure of the existing tree exited 0 and subsequent incremental builds passed. The benchmark checker was not edited or used for the node's deep test.

The first two limits-suite attempts exited 1 due to test setup errors: a 2000-token session plus its neighbor exceeded 48 MiB during fixture creation. The first attempt lost the empty future target to LRU and observed 404; the second created that target later but its append could not fit beside the pinned 2000-token state, correctly returning 503. Corrected the active fixture to 1000 tokens and made the pinned session explicitly older than its idle neighbor; the corrected integrated suite exits 0. Both earlier logs retain their already-passing LRU and byte-preserving refusal legs.

The first default HTTP regression exited 1 on an allowed transient 429 during final DELETE cleanup, after its concurrent work had succeeded. Cleanup now uses the harness's existing overload retry; the corrected default run exits 0. Admission limits and product behavior were not weakened.

The first client-fixture preparation exited 1 because its output directory did not exist. Created the task-local directory and reran; subsequent results are listed in the appendix. The first identity implementation used insertion-order JSON hashing; before handoff it was corrected to sorted-key canonical JSON, with insertion-order controls and a fresh actual-quantization plus independent-reader check. Earlier fingerprint observations are superseded.

## Finite coverage and untested steps

No supplied surface table or external landing property suite. No prior reviewer tests removed; the RAM-error fixture retains its original test name and is updated for the new eviction contract. No rework module expansion. Self-review covers scope, security, lifetime, numeric bounds and tests under the local code-review skill; no hosted review, check or landing is claimed or requested.

No GPU, Metal, 27B, 128k/256k inference, rented hardware, transport shaping, restart recovery, disk tier or resident-context implementation. The deep test uses one deterministic synthetic 40000-token prompt on the existing 0.8B CPU model, not the specification's full 50-prompt matrix. Re-quantization verifies file/layout identity, not heterogeneous numerical acceptance. No other base-model GGUF is available without a forbidden download; actual different-base-model identity remains unrun, supplemented by synthetic structural and model-projection controls.

The installed environment lacks `ty` and `editorconfig-checker`; their hosted workflows were not run. Narrow Python compilation, C++ builds, whitespace/scope/public-data checks and the requested functional gates are recorded separately. The code-style workflow checks model source naming, which is outside this delta; no hosted CI is claimed.

The signed implementation is `eb17f195269d6e8026a42ae82835eef49ca797c2` (signature G, `git verify-commit` exit 0). This results update is a separate signed documentation commit; no source behavior changed after that implementation checkpoint.

## Deep numerical result and evidence custody

The prompt has exactly 40000 deterministic synthetic token IDs. The fresh node extend covers [0,39999); the checker evaluates the held final token locally. The HTTP delta is 282,436,009 bytes, and retained session accounting is 282,435,544 bytes. `max_ctx` and allocated `n_ctx` are both 65536; the RAM budget is 536,870,912 and request maximum 1,048,576 bytes. One session, zero evictions. An over-context request returns 400 and leaves n=39999. This is a fresh full-range transfer, not a 40k multi-turn append-chain qualification.

```text
METRICS deep.tokens.json same_partition=64/64 mean_kl=2.32432e-15 free_running_first_divergence=-1 top1=64/64 agreement=1
```

Observed wall times, including setup and teardown: `deep-http-01` 1462.727 s, `deep-distributions-01` 2329.771 s. These are diagnostic CPU observations, not rented-hardware timing estimates.

Both references are independent local inference computations from the same prompt token IDs: same-partition [0,N-1) then held token, and whole-prompt [0,N). Neither reuses the imported state. Only the informational free-running branch clones the independently computed whole-prompt state, using that context's first logits, to avoid a redundant third full prefill. The checker's KL/top-1 controls include known violations and exact threshold boundaries.

The long-lived node started before the sorted-key fingerprint correction and signedness-warning cleanup. Its executable SHA-256 is `9f55c6fb76e7e91f95abb0312395f639fda99f3696b845e9196804df4ca746fa`. The source for node prefill, admission, wire codec and numerical state operations is unchanged by those identity-only corrections; its old fingerprint observation is superseded by the fresh final-node identity gates. The numerical checker uses the signed source. The final binary manifest and implementation source commit bind this distinction rather than labeling the older process as a final-binary run.

Existing distribution fixtures at 512 / 2048 / 8192 tokens each pass same-partition 64/64 and whole-prompt top-1 64/64; mean KL is respectively 3.2963e-16 / 7.69261e-7 / 2.01502e-7 nats. Free-running divergence is -1 for all three (none in the 64 positions). Client 128-token append measurements are 7.954225 / 7.628295 / 7.564696 ms at n0=512 / 2048 / 8192 with exact state equality; these are local diagnostic observations, not rented-hardware performance claims.

Minimum exact available bytes observed inside the build lock: 38726303744. 11 task-owned state/token fixtures (610381158 bytes) were hashed and removed. 115 recorded owned PIDs were absent at 2026-11-01T00:41:20.116717+00:00. Private credentials used by these fixtures were removed after process shutdown. The authorized 811,843,840-byte Q8_0 model remains in the local model cache. Original model/golden files were only read.

Private audit artifacts: `$RUN/validation-commands.jsonl`, `$RUN/evidence-manifest.json`, `$RUN/state-dump-hashes.json`, `$RUN/process-audit.json`, and the complete numbered logs. Validation-ledger SHA-256: `bbf3e8c9ae703b77f492fc5a1fe2de30a8e6158d04da2c89332d820d910f7f6c`. State-hash-manifest SHA-256: `861610b8bf2ad06babb6b8f1555b79a10f6f7e552c5d8adf2340c343dbac4fcd`. No logs or required commands are missing or truncated on disk.

## Validation commands and real exits

The public argv below replace only private absolute model-cache paths with the variables above; the private ledger preserves exact original argv, UTC times and PIDs. Helper probes and the isolated control live under `$RUN`. Communication commands are excluded from this public appendix. Every gate ran as a standalone child process with its actual exit preserved, without a tee pipeline. Nonzero historical setup/control results are intentionally retained.

```sh
# build-01; exit 2; log build-01.log
flock "$MODEL_CACHE/.build.lock" bash -c 'available=$(df -B1 --output=avail . | tail -n1); echo "available_bytes=$available"; test "$available" -ge 26843545600 || exit 75; exec .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-hybrid-prefill llama-hybrid-check llama-hybrid-check-codec llama-hybrid-check-append llama-hybrid-check-guards llama-hybrid-check-distributions llama-hybrid-check-client llama-hybrid-bench-check llama-quantize'
# codec-65536-01; exit 0; log codec-65536-01.log
build/bin/llama-hybrid-check-codec --max-ctx 65536
# codec-262144-01; exit 0; log codec-262144-01.log
build/bin/llama-hybrid-check-codec --max-ctx 262144
# syntax-01; exit 0; log syntax-01.log
python3 -m py_compile tools/hybrid/test_limits.py tools/hybrid/test_prefill.py
# build-02; exit 0; log build-02.log
flock "$MODEL_CACHE/.build.lock" bash -c 'available=$(df -B1 --output=avail . | tail -n1); echo "available_bytes=$available"; test "$available" -ge 26843545600 || exit 75; exec .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-hybrid-prefill llama-hybrid-check llama-hybrid-check-codec llama-hybrid-check-distributions'
# configure-01; exit 0; log configure-01.log
flock "$MODEL_CACHE/.build.lock" bash -c 'available=$(df -B1 --output=avail . | tail -n1); echo "available_bytes=$available"; test "$available" -ge 26843545600 || exit 75; exec .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake -S . -B build'
# limits-01; exit 1; log limits-01.log
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/limits
# build-03; exit 0; log build-03.log
flock "$MODEL_CACHE/.build.lock" bash -c 'available=$(df -B1 --output=avail . | tail -n1); echo "available_bytes=$available"; test "$available" -ge 26843545600 || exit 75; exec .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-hybrid-prefill llama-hybrid-check llama-hybrid-check-codec llama-hybrid-check-distributions llama-quantize'
# quantize-01; exit 0; log quantize-01.log
build/bin/llama-quantize --allow-requantize "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" "$MODEL_CACHE/Qwen3.5-0.8B-Q8_0-requant.gguf" Q8_0 2
# codec-default-01; exit 0; log codec-default-01.log
build/bin/llama-hybrid-check-codec
# codec-absolute-01; exit 0; log codec-absolute-01.log
build/bin/llama-hybrid-check-codec --max-ctx 262144
# scheduler-codec-01; exit 0; log scheduler-codec-01.log
build/bin/llama-hybrid-check --codec --max-ctx 65536
# syntax-02; exit 0; log syntax-02.log
python3 -m py_compile tools/hybrid/test_limits.py tools/hybrid/test_prefill.py
# pinned-03; exit 0; log pinned-03.log
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/pinned-03 --case pinned
# requests-01; exit 0; log requests-01.log
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/requests --case requests
# identity-01; exit 0; log identity-01.log
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --requantized "$MODEL_CACHE/Qwen3.5-0.8B-Q8_0-requant.gguf" --artifacts .temp/hybrid-a-r10-261031/identity --case identity
# limits-02; exit 1; log limits-02.log
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/limits-02
# http-default-01; exit 1; log http-default-01.log
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/http-default --port 18092
# http-slow-readers-01; exit 0; log http-slow-readers-01.log
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/http-slow-readers --port 18092 --regression slow-readers
# http-pending-01; exit 0; log http-pending-01.log
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/http-pending --port 18092 --regression pending
# http-default-02; exit 0; log http-default-02.log
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/http-default-02 --port 18093
# http-errors-01; exit 0; log http-errors-01.log
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/http-errors --port 18092 --regression errors
# build-04; exit 0; log build-04.log
flock "$MODEL_CACHE/.build.lock" bash -c 'available=$(df -B1 --output=avail . | tail -n1); echo "available_bytes=$available"; test "$available" -ge 26843545600 || exit 75; exec .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-hybrid-prefill llama-hybrid-check llama-hybrid-check-codec llama-hybrid-check-distributions'
# identity-02; exit 0; log identity-02.log
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --requantized "$MODEL_CACHE/Qwen3.5-0.8B-Q8_0-requant.gguf" --artifacts .temp/hybrid-a-r10-261031/identity-02 --case identity
# codec-canonical-01; exit 0; log codec-canonical-01.log
build/bin/llama-hybrid-check-codec --max-ctx 262144
# control-configure-01; exit 0; log control-configure-01.log
flock "$MODEL_CACHE/.build.lock" bash -c 'available=$(df -B1 --output=avail . | tail -n1); echo "available_bytes=$available"; test "$available" -ge 26843545600 || exit 75; exec .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake -S .temp/hybrid-a-r10-261031/unpinned-control -B .temp/hybrid-a-r10-261031/unpinned-control/build -DCMAKE_BUILD_TYPE=Release'
# control-build-01; exit 0; log control-build-01.log
flock "$MODEL_CACHE/.build.lock" bash -c 'available=$(df -B1 --output=avail . | tail -n1); echo "available_bytes=$available"; test "$available" -ge 26843545600 || exit 75; exec .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build .temp/hybrid-a-r10-261031/unpinned-control/build -j2'
# pinned-04; exit 0; log pinned-04.log
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/pinned-04 --case pinned
# unpinned-control-01; exit 1; log unpinned-control-01.log
python3 tools/hybrid/test_limits.py --binary .temp/hybrid-a-r10-261031/unpinned-control/build/prefill-unpinned --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/unpinned-control/run --case pinned
# identity-reference-01; exit 0; log identity-reference-01.log
python3 .temp/hybrid-a-r10-261031/check_identity.py
# scheduler-default-01; exit 0; log scheduler-default-01.log
build/bin/llama-hybrid-check --codec
# codec-65536-02; exit 0; log codec-65536-02.log
build/bin/llama-hybrid-check-codec --max-ctx 65536
# client-prepare-01; exit 1; log client-prepare-01.log
build/bin/llama-hybrid-check-client "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --prepare "$MODEL_CACHE/golden" .temp/hybrid-a-r10-261031/client-fixtures
# client-prepare-02; exit 0; log client-prepare-02.log
build/bin/llama-hybrid-check-client "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --prepare "$MODEL_CACHE/golden" .temp/hybrid-a-r10-261031/client-fixtures
# client-measure-512-01; exit 0; log client-measure-512-01.log
build/bin/llama-hybrid-check-client "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --measure .temp/hybrid-a-r10-261031/client-fixtures/cost-512 512
# client-measure-2048-01; exit 0; log client-measure-2048-01.log
build/bin/llama-hybrid-check-client "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --measure .temp/hybrid-a-r10-261031/client-fixtures/cost-2048 2048
# client-measure-8192-01; exit 0; log client-measure-8192-01.log
build/bin/llama-hybrid-check-client "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --measure .temp/hybrid-a-r10-261031/client-fixtures/cost-8192 8192
# client-capacity-01; exit 0; log client-capacity-01.log
build/bin/llama-hybrid-check-client "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --capacity "$MODEL_CACHE/golden/prompt-512"
# build-final-01; exit 0; log build-final-01.log
flock "$MODEL_CACHE/.build.lock" bash -c 'available=$(df -B1 --output=avail . | tail -n1); echo "available_bytes=$available"; test "$available" -ge 26843545600 || exit 75; exec .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-hybrid-prefill llama-hybrid-check llama-hybrid-check-codec llama-hybrid-check-append llama-hybrid-check-guards llama-hybrid-check-distributions llama-hybrid-check-client'
# http-max-delta-01; exit 0; log http-max-delta-01.log
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/http-max-delta --port 18092 --regression max-delta
# recurrent-01; exit 0; log recurrent-01.log
python3 tools/hybrid/test_recurrent_append.py --library build/bin/libllama.so --malformed
# build-distributions-final-01; exit 0; log build-distributions-final-01.log
flock "$MODEL_CACHE/.build.lock" bash -c 'available=$(df -B1 --output=avail . | tail -n1); echo "available_bytes=$available"; test "$available" -ge 26843545600 || exit 75; exec .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-hybrid-check-distributions'
# scope-01; exit 0; log scope-01.log
python3 .temp/hybrid-a-r10-261031/check_scope.py
# syntax-final-01; exit 0; log syntax-final-01.log
python3 -m py_compile tools/hybrid/test_limits.py tools/hybrid/test_prefill.py
# native-01; exit 0; log native-01.log
build/bin/llama-hybrid-check "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" -
# limits-final-01; exit 0; log limits-final-01.log
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/limits-final
# identity-final-01; exit 0; log identity-final-01.log
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --requantized "$MODEL_CACHE/Qwen3.5-0.8B-Q8_0-requant.gguf" --artifacts .temp/hybrid-a-r10-261031/identity-final --case identity
# codec-final-16384; exit 0; log codec-final-16384.log
build/bin/llama-hybrid-check-codec
# codec-final-65536; exit 0; log codec-final-65536.log
build/bin/llama-hybrid-check-codec --max-ctx 65536
# codec-final-262144; exit 0; log codec-final-262144.log
build/bin/llama-hybrid-check-codec --max-ctx 262144
# http-default-final-01; exit 0; log http-default-final-01.log
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/http-default-final --port 18093
# http-slow-readers-final-01; exit 0; log http-slow-readers-final-01.log
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/http-slow-readers-final --port 18093 --regression slow-readers
# append-01; exit 0; log append-01.log
build/bin/llama-hybrid-check-append "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" "$MODEL_CACHE/golden/prompt-512" "$MODEL_CACHE/golden/prompt-2048" "$MODEL_CACHE/golden/prompt-8192"
# guards-01; exit 0; log guards-01.log
build/bin/llama-hybrid-check-guards "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" "$MODEL_CACHE/golden/prompt-512"
# http-pending-final-01; exit 0; log http-pending-final-01.log
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/http-pending-final --port 18093 --regression pending
# http-errors-final-01; exit 0; log http-errors-final-01.log
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/http-errors-final --port 18093 --regression errors
# distributions-01; exit 0; log distributions-01.log
build/bin/llama-hybrid-check-distributions "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" "$MODEL_CACHE/golden/prompt-512.tokens.json" "$MODEL_CACHE/golden/prompt-2048.tokens.json" "$MODEL_CACHE/golden/prompt-8192.tokens.json"
# http-max-delta-final-01; exit 0; log http-max-delta-final-01.log
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/http-max-delta-final --port 18093 --regression max-delta
# deep-http-01; exit 0; log deep-http-01.log
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" --artifacts .temp/hybrid-a-r10-261031/deep --case deep
# deep-distributions-01; exit 0; log deep-distributions-01.log
build/bin/llama-hybrid-check-distributions "$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf" .temp/hybrid-a-r10-261031/deep/deep.tokens.json --max-ctx 65536
```

Final documentation/scope checks: `python3 .temp/hybrid-a-r10-261031/check_scope.py` exited 0 in `scope-final-01.log` and `scope-final-02.log`; `git diff --check` exited 0. These closing documentation checks follow the frozen numerical-validation ledger and remain in the full private command ledger.

# Integrated prototype r6a rework, CPU 0.8B

Run `hybrid-a-r12-261101`. Candidate implementation: signed `70778be6299b4e834a2098356ce7ff0fd3366f92` on `m1-integrated`, based on `7c51e071ebdf442e20e1e0529e77e41caaa4decc`. Local review handoff only; no push, PR or landing. Ready for review: the six fixes and homogeneous CPU 0.8B gates passed. Independent review is pending.

## Fixes and contract

1. Every authenticated native extend reports prefill-only time, independently of the optional link endpoint. This preserves honest measured timing instead of inventing or omitting it. Skipping the optional link probe is recorded and no longer makes a valid workload partial.
2. Proxy and benchmark share a chunk planner with q8_0 KV plus 24 B/token of native cell metadata and a 1 MiB fixed reserve. Separate GGUF-derived `delta_geometry` controls safety, independent of policy cost estimates. A proxy without launcher geometry defaults conservatively to 27B (34,816 KV B/token, 157,000,000 recurrent bytes), also safe for 0.8B. The largest planned chunk is maximal **including the reserve**; one more violates that reserved bound, not necessarily the raw cap. Native body/header caps stay unchanged.
3. Node, design, benchmark, proxy, public M1 docs and the specification identity note now agree: exact model SHA by default; experimental hetero opt-in only after matching layout fingerprint; envelope SHA rewritten, native SHA guard bypassed; quality unverified/uncalibrated. R11 measurement KL 0.0145 > 0.01 and top-1 62/64 is explicitly historical and used requantized Q4 weights.
4. All prototype builds disable both UI sources, with or without `--offline`. No web UI is used, so disabling it avoids the unpinned bucket/latest fallback. Offline additionally forbids model downloads. Upstream UI fetching code is unchanged and unused.
5. Public nodes require a bearer token for metrics and omit health activity counters without valid authentication. Explicit `--public` and raw non-loopback binds are covered. Authenticated health still supplies benchmark release counters; loopback defaults stay unchanged.
6. HYBRID.md describes M1 node limits, LRU, 503, chunked extends, deployment, benchmarks and hetero opt-in.

## Validation and limits

All builds are incremental CPU builds under `/home/agent/Werkstatt/hybrid-models/.build.lock`, `-j2`, using the retained CMake 4.4.4 tool. No tools or weights were installed/downloaded. Both UI flags are OFF. Runtime uses two compute threads, q8_0 KV, f32 recurrent state and batch/ubatch 256. No `src/` changes. Existing reviewer test names remain. No project board is configured (`task-board goal get`: exit1, missing .task-board); the existing `.temp/tasks.md` tracks the run. The user-specified branch/base overrides default-main branching.

Checks prove only the exercised CPU 0.8B behavior. The 27B planner boundary is arithmetic/codec-size evidence, not 27B execution. No CUDA/Metal, 50-prompt matrix, production throughput or heterogeneous quality acceptance is claimed. No second base-model GGUF is available; changed fingerprint metadata and Q4/Q8 admission do not substitute for that unrun case. All reported timings include shared-host contention.

## Numerical checks

| Label / check | Depth | Deltas | Same partition | Mean KL nats | Top-1 | Status / exit |
|---|---:|---:|---:|---:|---:|---|
| integrated-short / fresh | 132 | 3 | 64/64 | 0.0012770034970001 | 64/64 | passed / 0 |
| integrated-short / append_chain | 132 | 3 | 64/64 | 0.0012770034970001 | 64/64 | passed / 0 |
| integrated-hetero / fresh | 132 | 1 | 62/64 | 0.015340486164613 | 62/64 | uncalibrated_outside_reference / 5 |
| integrated-hetero / append_chain | 132 | 1 | 62/64 | 0.015340486164613 | 62/64 | uncalibrated_outside_reference / 5 |
| integrated-40k / fresh | 40038 | 5 | 64/64 | 5.6201898505964e-05 | 64/64 | passed / 0 |
| integrated-40k / append_chain | 40038 | 15 | 64/64 | 6.1083364920693e-05 | 64/64 | passed / 0 |

Section 11 reference: mean KL <=0.01, top-1 >=95%, homogeneous same-backend same-partition 64/64. Both 40k checks report free-running divergence -1. Heterogeneous measurements always remain uncalibrated. R12 hetero uses one delta; R11 used three, so its measured KL need not be identical.

## Test claims

| Gate | Claim and exercised bound | Exit |
|---|---|---|
| bench-unit-03; proxy-unit-02 | 22 benchmark and 68 proxy unit tests exercise admission, routing, chunk bounds, release and existing positive/negative contracts. | 0; 0 |
| integrated-final-01; review-timing-repro-01 | Native defaults, both timing flag paths, exact native size slope, LRU replay of 25 actual IDs, budget503 fields and benchmark classification. | 0; 0 |
| launcher-audit-02 | Native argv, node-info, live health/metrics and loaded fingerprint agree on the actual operator limits. | 0 |
| hetero-refusal-01; launch-hetero-client-01; fingerprint-refusal-01 | Real Q4/Q8 refuses without opt-in, admits matching layout with opt-in, rejects corrupted layout metadata even with opt-in. | 2 expected; 0; 2 expected |
| limits-01; public-final-01; five http-*; header-http-01; deep-http-01 | Retained-budget/LRU/pinning and request/context bounds preserve state; public metrics authenticate; real ~40k HTTP import succeeds. | all 0 |
| codec-*; recurrent-01; native-01; append-01; guards-01; distributions-01; deep-distributions-01 | Inclusive size gates, malformed rollback, native append, same-partition 64/64 and stated KL bounds at exercised depths. | all 0 |
| client-*; remote-append-01; append-api-01; concurrency-native-01; decode-proxy-01 | Native capacity, chunked append, replay/fork, slot rollback and concurrent independent sessions retain their contracts. | all 0 |
| four conversation-*; typed-identity-01; typed-no-checkpoint-01; identity-01 | Thinking variants, omitted/no-checkpoint compatibility, strict typed identity and both rejection directions. | all 0 |
| long-benchmark-01; long-audit-01 | All three modes execute every scheduled call, release to zero, import contiguous chunks and pass fresh/15-delta append checks at40038. | 0; 0 |
| final-source-audit-02 | Measured runtime/binaries unchanged, old reviewer test names retained, allowed scope, signatures and whitespace valid, no src delta. | 0 |

Independent native distribution regressions:

| Fixture depth | Same partition | Mean KL nats | Top-1 | Exit |
|---:|---:|---:|---:|---:|
| 512 | 64/64 | 3.2963e-16 | 64/64 | 0 |
| 2048 | 64/64 | 7.69261e-07 | 64/64 | 0 |
| 8192 | 64/64 | 2.01502e-07 | 64/64 | 0 |
| 40000 | 64/64 | 2.32432e-15 | 64/64 | 0 |

## Routing and release audit

| Workload / mode | Calls | Maximum prompt tokens | Maximum chunks per call | Total seconds | Mean TTFT seconds | Mean decode tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| integrated-short / local | 4 | 340 | 0 | 2.369 | 0.420 | 37.463 |
| integrated-short / hybrid | 4 | 340 | 0 | 3.494 | 0.711 | 39.417 |
| integrated-short / offload | 4 | 340 | 3 | 6.370 | 1.388 | 31.733 |
| integrated-40k / local | 8 | 40198 | 0 | 1774.687 | 221.318 | 12.027 |
| integrated-40k / hybrid | 8 | 40198 | 5 | 2380.743 | 296.864 | 8.748 |
| integrated-40k / offload | 8 | 40198 | 5 | 2460.487 | 306.951 | 10.661 |

These are measured synthetic calls on a shared CPU host, with correctness work overlapping in time; they are not a comparative hardware performance claim. Every mode released the node to zero sessions and zero retained bytes. The long invocation used a node deployed without `--bench-link`. Both runs have zero aborted or skipped workload calls, all forced-offload calls include measured prefill time, and critical imports have contiguous boundaries. Copied-record negative controls reject a false 63/64 agreement, KL 0.02 and a leaked session for their named assertions. The audit verifies each returned archive manifest and archive SHA-256 sidecar, expected member scope, absent private keys and absent actual credential values.

| Returned content-free bundle | SHA-256 |
|---|---|
| hybrid-bench-integrated-40k-20261101T051206080925Z.tar.gz | `258f819b1495db8db1983c162b2dc444b6eca7e6711d0fc9e123afbfbd42a9ca` |
| hybrid-bench-integrated-hetero-20261101T051427396298Z.tar.gz | `ce887f0d1bc043868a15f81e6083133b2586a6d9768ba9c7171c34a3482106b2` |
| hybrid-bench-integrated-short-20261101T051723450321Z.tar.gz | `28e48a7247b0655f2b5c7564b15d7720a1c25949a902755a51eb746566b00746` |
| hybrid-bench-killed-mid-run-20261101T051319767906Z.tar.gz | `28b035801cd9126d5382dde5f7f9202e60d23203fabd8ee6e2836d475dad6ecf` |
| hybrid-bench-reprefill-control-20261101T051317789696Z.tar.gz | `a05fa40c4ccb02a67b43ad1f379324acb9a29ed4573f038c30c3ce042476fd19` |

## Chunk boundary evidence

| Geometry | KV B/token | Native B/token | Recurrent bytes | Planned tokens | Reserved size B | Reserved size with one more B |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B | 6528 | 6552 | 20201472 | 78696 | 536866240 | 536872792 |
| 27B / direct-proxy default | 34816 | 34840 | 157000000 | 10873 | 536863896 | 536898736 |

The last two columns include the 1 MiB serialization reserve and straddle the unchanged 536,870,912-byte body limit. They do not assert that one token above the chosen plan overflows the raw native body.

Real short deltas: slope 6552 B/token, fixed 20202100 B. New 0.8B chunk 78696 predicts body 535818292 B <= 536870912 B; old chunk 78986 predicts 537718372 B > cap. Header and 48-byte checksum trailer are excluded from body size. The review slope script included the 48-byte trailer in its fixed intercept; this explains its +48 B body estimate.

Real maximum-chunk HTTP export: `{"body_bytes": 535818292, "cap": 536870912, "context": 78848, "header_bytes": 315197, "release_bytes": 0, "release_sessions": 0, "seconds": 8468.226244447986, "status": 200, "tokens": 78696, "wire_bytes": 536133541}`.

0.8B Q4_0 GGUF: `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c` (507,154,688 bytes). Q8_0 requantization: `f543fad263339779c3a38a21a531c6adbffcfdecdf5f749dd542de924b1e7302` (811,843,840 bytes). Matching layout fingerprint: `33d9f73396507b1f8402bf1e439936f42caee2c8875fb46e127b06e0e656d2fa`. The model SHA values were checked in this run; models were reused without download.

Native codec checks admit the inclusive 512 MiB size, reject cap+1 byte for the intended size reason, and admit/reject the reserved maximum/one-token-over plans for both supported geometries. Unit proxy startup accepts the planned maximum, rejects one token more and ignores deliberately tiny policy cost estimates. The old 78,986-token full prefill was not repeated: its failing cap predicate is established by the real affine native sizes and the size-gate controls, avoiding another multi-hour CPU prefill that can only reach the same size rejection. The corrected maximum was actually prefilled/exported end to end, not inferred from the formula alone.

## Negative controls and instrument bounds

| Mutant/control | Narrows gate to | Named failing test | Survival bound |
|---|---|---|---|
| r11 timing node | real offload requires measured native timing | test_prefill_timing / timing-no-link (benchmark2, harness1) | Killed for the stated assertion; no survivor claimed |
| omit-cell-metadata | 24 B native metadata per token | BenchmarkTests.test_native_chunk_boundaries | Killed for the stated assertion; no survivor claimed |
| underestimate-27B-default | safe direct-proxy default geometry | BenchmarkTests.test_native_chunk_boundaries | Killed for the stated assertion; no survivor claimed |
| ui-flags-removed | both UI sources disabled | BenchmarkTests.test_offline_build_flags | Killed for the stated assertion; no survivor claimed |
| ui-flags-offline-only | ordinary builds cannot retain the original UI download path; offline positive control passes first | BenchmarkTests.test_offline_build_flags | Killed for the stated assertion; no survivor claimed |
| r11 public node | public metrics require authentication | old_public.py / Node.request(metrics expected401), actual200 | Killed for the stated assertion; no survivor claimed |
| planted re-prefill telemetry | benchmark detects old-prefix work after valid call | test_native.py / reprefill-control, benchmark2 | Killed for the stated assertion; no survivor claimed |
| killed owned node | forced offload cannot silently become local | test_native.py / killed-mid-run, benchmark2 | Killed for the stated assertion; no survivor claimed |
| ui-flags-removed / attempt01 | No UI property exercised | None: unrelated dirty-source Refusal before flag evaluation | Survivor / invalid setup; no detection claim from attempt01. Corrected attempt02 has the named failure above. |

Mutant01 UI setup failed with an unrelated Refusal because copied globals escaped mocks; it is invalid control evidence. Corrected mutant02 runs the copied function against the real module globals and the named test fails for missing UI flags. The maximum-chunk attempt01 was deliberately interrupted (SIGINT return -2; owned node stopped -15) before its one-hour read deadline could waste a full long prefill; attempt02 uses the identical native request with a three-hour read timeout. Two early archive-audit probes failed on a substring scanner that matched decode_token inside decode_tokens_per_second; exact-key plus actual-credential-value scanning passes, with archives unchanged. Launcher-audit01 used an incorrect PID filename and failed; audit02 uses the actual private `.pid` and passes. Mapped-library audit01 expected the unversioned symlink name and failed on the actual versioned SONAME; corrected audit02 confirms all active native gates map the retained project libraries, with no deleted mappings. The final integration/public/codec runs bind the new properties to the clean70778 build used by the long benchmark; their first runs preceded the checkpoint commit. Initial missing-tool wrapper attempts raised FileNotFoundError (wrapper1, tools unrun); shell readiness records the true missing-command127. `ty` and `editorconfig-checker` remain unavailable/unrun; no network install attempted. Syntax/scope, whitespace and repository naming checks are separate green evidence.

## Coverage and retained scope

No normative surface table was supplied in the user brief; the review report contains an observational surface table. This is a brief gap, not a claim of universal coverage. Rework touches only hybrid tools and required entry-point docs. README/HYBRID, NODE-STATE-DESIGN and docs/hybrid/prototype-spec.md are outside the runtime modules for the explicitly requested documentation fixes (items3 and6); the specification retains its homogeneous acceptance scope and adds a dated experimental implementation note. Native model/codec identity gates and numerics are unchanged. Independent review is pending.

| Review surface | Evidence / limit |
|---|---|
| Node limits, LRU, flags, 503, defaults | test_limits, test_native_defaults, test_eviction_replay, test_budget_benchmark, native launcher audit |
| Chunked critical extends | test_native_chunk_boundaries, test_native_chunk_startup_bound, test_native_chunk_slope, native codec boundary checks, real maximum export, 40k append chain |
| Benchmark feature flags | test_prefill_timing both paths and unchanged reviewer reproduction; old node named red |
| Remote-host safety | test_public_metrics explicit-public and non-loopback paths; existing private file/PID/refusal controls |
| Three routing modes / meter integrity | same-seed short/long benchmarks and test_native telemetry/fallback controls |
| Hetero default / fingerprint / documentation | expected SHA/fingerprint refusals, real Q8 client opt-in, uncalibrated numerical report, documentation contract check; no distinct-base native model |
| Build download | ordinary/offline UI flag test with killed mutant; actual locked builds OFF/OFF |
| CUDA readiness | unchanged design/readiness notes; GPU execution unrun, outside requested CPU acceptance |
| Documentation | manual review and cross-document contract fixture; r11 measurement labeled historical |

## Command and exit ledger

Each gate runs as a standalone child with its real exit status; no tee pipelines. Full logs were task-local and hashed before requested scratch deletion. Expected-red nonzero codes remain failures, not passing product evidence. Readiness and scratch-control failures are retained.

Initial direct unit calls inherited host `TMPDIR=/home/agent/Werkstatt/hybrid-a/.tmp`; their TemporaryDirectory fixtures were auto-deleted by the tests without a retained hash manifest. This is a scratch-placement/process deviation. Later commands set TMPDIR to this task root; benchmark-unit03 and proxy-unit02 are the corrected-path evidence. All retained logs and native measurement artifacts are task-local. Final hash/delete covers the remaining task root, not already auto-deleted library fixtures.

Direct pre-ledger proxy suite: `python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v`, exit0, 68 tests (85.216s), log `proxy-unit-01.log`. Initial benchmark suite exit0, 22 tests; repeated ledger runs use unchanged test/source identities except the documented offline-help text edit.

- `configure-01`: exit 0, 2026-11-01T05:08:55.265180+00:00 to 2026-11-01T05:08:55.766011+00:00; minimum free 40069943296 B.

```sh
flock /home/agent/Werkstatt/hybrid-models/.build.lock .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
```

- `build-01`: exit 0, 2026-11-01T05:09:02.187988+00:00 to 2026-11-01T05:09:13.693544+00:00; minimum free 40060796928 B.

```sh
flock /home/agent/Werkstatt/hybrid-models/.build.lock .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-hybrid-prefill llama-hybrid-check llama-hybrid-check-codec llama-hybrid-check-append llama-hybrid-check-guards llama-hybrid-check-distributions llama-hybrid-check-client llama-hybrid-bench-check llama-server
```

- `old-timing-01`: exit 1, 2026-11-01T05:09:03.347024+00:00 to 2026-11-01T05:09:08.850157+00:00; minimum free 40060710912 B.

```sh
python3 tools/hybrid-bench/test_integrated.py --server-binary build/bin/llama-server --prefill-binary .temp/hybrid-a-r12-261101/prefill-r11 --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r12-261101/old-timing --case timing
```

- `integrated-01`: exit 0, 2026-11-01T05:09:37.010143+00:00 to 2026-11-01T05:10:04.524590+00:00; minimum free 40063275008 B.

```sh
python3 tools/hybrid-bench/test_integrated.py --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r12-261101/integrated
```

- `limits-01`: exit 0, 2026-11-01T05:09:38.170195+00:00 to 2026-11-01T05:11:01.203424+00:00; minimum free 40061509632 B.

```sh
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r12-261101/limits
```

- `codec-01`: exit 0, 2026-11-01T05:09:39.331214+00:00 to 2026-11-01T05:09:39.831975+00:00; minimum free 40064794624 B.

```sh
build/bin/llama-hybrid-check-codec --max-ctx 262144
```

- `bench-unit-02`: exit 0, 2026-11-01T05:09:39.879911+00:00 to 2026-11-01T05:09:40.380975+00:00; minimum free 40064471040 B.

```sh
python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v
```

- `build-codec-02`: exit 0, 2026-11-01T05:10:05.606527+00:00 to 2026-11-01T05:10:08.608085+00:00; minimum free 40062701568 B.

```sh
flock /home/agent/Werkstatt/hybrid-models/.build.lock .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-hybrid-check-codec
```

- `codec-02`: exit 0, 2026-11-01T05:10:29.670667+00:00 to 2026-11-01T05:10:30.171486+00:00; minimum free 40063041536 B.

```sh
build/bin/llama-hybrid-check-codec --max-ctx 262144
```

- `scope-01`: exit 0, 2026-11-01T05:10:30.200613+00:00 to 2026-11-01T05:10:30.701670+00:00; minimum free 40063033344 B.

```sh
python3 -c 'import ast,subprocess; from pathlib import Path; base="7c51e071"; paths=subprocess.check_output(["git","diff","--name-only",base],text=True).splitlines(); assert all(p.startswith("tools/hybrid") or p in ("README.md","HYBRID.md") for p in paths); assert not subprocess.check_output(["git","diff",base,"--","src"]); retained=0
for p in paths:
 if p.endswith(".py"):
  ast.parse(Path(p).read_text()); old=subprocess.check_output(["git","show",base+":"+p],text=True); names=lambda s:{n.name for n in ast.walk(ast.parse(s)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith("test_")}; a,b=names(old),names(Path(p).read_text()); assert a<=b,(p,a-b); retained+=len(a)
print("scope/no-src/syntax/original changed-file test names retained",retained)'
```

- `mutants-01`: exit 1, 2026-11-01T05:10:58.649114+00:00 to 2026-11-01T05:10:59.149863+00:00; minimum free 40061898752 B.

```sh
python3 .temp/hybrid-a-r12-261101/mutants.py
```

- `mutants-02`: exit 0, 2026-11-01T05:11:14.546796+00:00 to 2026-11-01T05:11:15.047672+00:00; minimum free 40061480960 B.

```sh
python3 .temp/hybrid-a-r12-261101/mutants.py
```

- `build-codec-03`: exit 0, 2026-11-01T05:11:15.070969+00:00 to 2026-11-01T05:11:17.572727+00:00; minimum free 40060817408 B.

```sh
flock /home/agent/Werkstatt/hybrid-models/.build.lock .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-hybrid-check-codec
```

- `old-public-01`: exit 1, 2026-11-01T05:11:15.694163+00:00 to 2026-11-01T05:11:17.196170+00:00; minimum free 40061132800 B.

```sh
python3 .temp/hybrid-a-r12-261101/old_public.py
```

- `codec-03`: exit 0, 2026-11-01T05:11:33.519929+00:00 to 2026-11-01T05:11:34.020674+00:00; minimum free 40061206528 B.

```sh
build/bin/llama-hybrid-check-codec --max-ctx 262144
```

- `bench-unit-03`: exit 0, 2026-11-01T05:11:34.044161+00:00 to 2026-11-01T05:11:34.544908+00:00; minimum free 40061198336 B.

```sh
python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v
```

- `launch-node-01`: exit 0, 2026-11-01T05:11:41.975489+00:00 to 2026-11-01T05:11:47.478473+00:00; minimum free 40060993536 B.

```sh
tools/hybrid-bench/deploy-prefill-node.sh --state .temp/hybrid-a-r12-261101/long-node --build-dir build --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --offline --ctx 65536 --ram-budget-mib 560 --port 19391
```

- `launch-client-01`: exit 0, 2026-11-01T05:11:49.120996+00:00 to 2026-11-01T05:11:53.123034+00:00; minimum free 40060891136 B.

```sh
tools/hybrid-bench/setup-client.sh --state .temp/hybrid-a-r12-261101/long-client --build-dir build --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --offline --ctx 65536 --node-info .temp/hybrid-a-r12-261101/long-node/node-info.json --token-file .temp/hybrid-a-r12-261101/long-node/client-tokens.json --prefill-url http://127.0.0.1:19391 --decode-port 19390 --port 19392 --request-timeout 7200 --extend-chunk-tokens 8192
```

- `long-benchmark-01`: exit 0, 2026-11-01T05:12:06.035168+00:00 to 2026-11-01T09:56:02.274375+00:00; minimum free 34001223680 B.

```sh
python3 tools/hybrid-bench/hybrid_bench.py --client-state .temp/hybrid-a-r12-261101/long-client --scratch .temp/hybrid-a-r12-261101 --output .temp/hybrid-a-r12-261101 --label integrated-40k --smoke --depths 39500 40000 --check-depths 40000 --request-timeout 7200 --check
```

- `regression-driver-01`: exit 0, 2026-11-01T05:12:07.206253+00:00 to 2026-11-01T05:36:26.380566+00:00; minimum free 36530470912 B.

```sh
python3 .temp/hybrid-a-r12-261101/regressions.py
```

- `codec-core-01`: exit 0, 2026-11-01T05:12:07.251537+00:00 to 2026-11-01T05:12:07.752423+00:00; minimum free 40060567552 B.

```sh
build/bin/llama-hybrid-check --codec
```

- `http-default-01`: exit 0, 2026-11-01T05:12:07.785343+00:00 to 2026-11-01T05:12:11.288670+00:00; minimum free 40060399616 B.

```sh
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/http-default-01 --port 19493
```

- `http-slow-01`: exit 0, 2026-11-01T05:12:11.322209+00:00 to 2026-11-01T05:12:13.823819+00:00; minimum free 40060219392 B.

```sh
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/http-slow-01 --port 19493 --regression slow-readers
```

- `http-pending-01`: exit 0, 2026-11-01T05:12:13.848435+00:00 to 2026-11-01T05:13:17.879347+00:00; minimum free 39552294912 B.

```sh
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/http-pending-01 --port 19493 --regression pending
```

- `monitor-01`: exit 0, 2026-11-01T05:12:41.567822+00:00 to 2026-11-01T09:57:04.217024+00:00; minimum free 34001223680 B.

```sh
python3 .temp/hybrid-a-r12-261101/monitor.py
```

- `launch-control-node-01`: exit 0, 2026-11-01T05:12:42.721257+00:00 to 2026-11-01T05:12:46.223196+00:00; minimum free 40059998208 B.

```sh
tools/hybrid-bench/deploy-prefill-node.sh --state .temp/hybrid-a-r12-261101/control-node --build-dir build --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --offline --ctx 2048 --bench-link --port 19491
```

- `launch-control-client-01`: exit 0, 2026-11-01T05:12:56.194254+00:00 to 2026-11-01T05:13:00.696591+00:00; minimum free 40059678720 B.

```sh
tools/hybrid-bench/setup-client.sh --state .temp/hybrid-a-r12-261101/control-client --build-dir build --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --offline --ctx 2048 --node-info .temp/hybrid-a-r12-261101/control-node/node-info.json --token-file .temp/hybrid-a-r12-261101/control-node/client-tokens.json --prefill-url http://127.0.0.1:19491 --decode-port 19490 --port 19492
```

- `launcher-audit-01`: exit 1, 2026-11-01T05:12:57.352363+00:00 to 2026-11-01T05:12:57.853479+00:00; minimum free 40063963136 B.

```sh
python3 .temp/hybrid-a-r12-261101/launcher_audit.py
```

- `launcher-audit-02`: exit 0, 2026-11-01T05:13:10.742719+00:00 to 2026-11-01T05:13:11.243754+00:00; minimum free 40059555840 B.

```sh
python3 .temp/hybrid-a-r12-261101/launcher_audit.py
```

- `native-controls-01`: exit 0, 2026-11-01T05:13:11.282743+00:00 to 2026-11-01T05:13:22.289340+00:00; minimum free 39530479616 B.

```sh
python3 tools/hybrid-bench/test_native.py --client-state .temp/hybrid-a-r12-261101/control-client --node-state .temp/hybrid-a-r12-261101/control-node --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r12-261101 --output .temp/hybrid-a-r12-261101
```

- `http-errors-01`: exit 0, 2026-11-01T05:13:17.907444+00:00 to 2026-11-01T05:13:33.916794+00:00; minimum free 39530438656 B.

```sh
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/http-errors-01 --port 19493 --regression errors
```

- `http-max-01`: exit 0, 2026-11-01T05:13:33.946351+00:00 to 2026-11-01T05:18:27.649599+00:00; minimum free 38885396480 B.

```sh
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/http-max-01 --port 19493 --regression max-delta
```

- `stop-control-client-01`: exit 0, 2026-11-01T05:13:47.027496+00:00 to 2026-11-01T05:13:47.528481+00:00; minimum free 39530242048 B.

```sh
tools/hybrid-bench/setup-client.sh --state .temp/hybrid-a-r12-261101/control-client --stop
```

- `hetero-refusal-01`: exit 2, 2026-11-01T05:13:47.564650+00:00 to 2026-11-01T05:13:49.066245+00:00; minimum free 39530217472 B.

```sh
tools/hybrid-bench/setup-client.sh --state .temp/hybrid-a-r12-261101/hetero-refused --build-dir build --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --offline --ctx 2048 --node-info .temp/hybrid-a-r12-261101/long-node/node-info.json --token-file .temp/hybrid-a-r12-261101/long-node/client-tokens.json --prefill-url http://127.0.0.1:19391 --decode-port 19590 --port 19592
```

- `more-regression-driver-01`: exit 0, 2026-11-01T05:13:48.201587+00:00 to 2026-11-01T07:33:55.027318+00:00; minimum free 34001223680 B.

```sh
python3 .temp/hybrid-a-r12-261101/more_regressions.py
```

- `fingerprint-refusal-01`: exit 2, 2026-11-01T05:14:07.218945+00:00 to 2026-11-01T05:14:08.720574+00:00; minimum free 39530180608 B.

```sh
tools/hybrid-bench/setup-client.sh --state .temp/hybrid-a-r12-261101/fingerprint-refused --build-dir build --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --offline --ctx 2048 --hetero-quant --node-info .temp/hybrid-a-r12-261101/wrong-fingerprint.json --token-file .temp/hybrid-a-r12-261101/long-node/client-tokens.json --prefill-url http://127.0.0.1:19391 --decode-port 19590 --port 19592
```

- `launch-hetero-client-01`: exit 0, 2026-11-01T05:14:08.373807+00:00 to 2026-11-01T05:14:14.378021+00:00; minimum free 39530033152 B.

```sh
tools/hybrid-bench/setup-client.sh --state .temp/hybrid-a-r12-261101/hetero-client --build-dir build --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --offline --ctx 2048 --hetero-quant --node-info .temp/hybrid-a-r12-261101/long-node/node-info.json --token-file .temp/hybrid-a-r12-261101/long-node/client-tokens.json --prefill-url http://127.0.0.1:19391 --decode-port 19590 --port 19592
```

- `hetero-benchmark-01`: exit 5, 2026-11-01T05:14:27.301881+00:00 to 2026-11-01T05:15:01.832775+00:00; minimum free 39486078976 B.

```sh
python3 tools/hybrid-bench/hybrid_bench.py --client-state .temp/hybrid-a-r12-261101/hetero-client --scratch .temp/hybrid-a-r12-261101 --output .temp/hybrid-a-r12-261101 --label integrated-hetero --smoke --depths 128 --check --modes offload
```

- `review-timing-repro-01`: exit 0, 2026-11-01T05:14:28.490833+00:00 to 2026-11-01T05:14:49.022533+00:00; minimum free 39486083072 B.

```sh
python3 /home/agent/Werkstatt/hybrid-review-proto/llama.cpp/.temp/rev-r6a/scripts/repro_no_bench_link.py /home/agent/Werkstatt/hybrid-a/llama.cpp /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-hybrid-prefill /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-server /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/review-timing
```

- `type-tool-readiness-02`: exit 127, 2026-11-01T05:15:50.758912+00:00 to 2026-11-01T05:15:51.262279+00:00; minimum free 39528923136 B.

```sh
bash -c 'exec ty --version'
```

- `editorconfig-readiness-02`: exit 127, 2026-11-01T05:15:51.307592+00:00 to 2026-11-01T05:15:51.809773+00:00; minimum free 39528820736 B.

```sh
bash -c 'exec editorconfig-checker --version'
```

- `stop-hetero-client-01`: exit 0, 2026-11-01T05:15:51.864819+00:00 to 2026-11-01T05:15:52.367022+00:00; minimum free 39528902656 B.

```sh
tools/hybrid-bench/setup-client.sh --state .temp/hybrid-a-r12-261101/hetero-client --stop
```

- `naming-style-01`: exit 0, 2026-11-01T05:15:56.291007+00:00 to 2026-11-01T05:15:56.793524+00:00; minimum free 39528853504 B.

```sh
python3 -c 'import pathlib,re; s=pathlib.Path(".github/workflows/code-style.yml").read_text(); code=s.split("python3 - << '"'"'EOF'"'"'\n",1)[1].split("          EOF",1)[0]; import textwrap; exec(textwrap.dedent(code))'
```

- `launch-short-client-01`: exit 0, 2026-11-01T05:16:39.222787+00:00 to 2026-11-01T05:16:44.727306+00:00; minimum free 39528390656 B.

```sh
tools/hybrid-bench/setup-client.sh --state .temp/hybrid-a-r12-261101/short-client --build-dir build --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --offline --ctx 2048 --node-info .temp/hybrid-a-r12-261101/long-node/node-info.json --token-file .temp/hybrid-a-r12-261101/long-node/client-tokens.json --prefill-url http://127.0.0.1:19391 --decode-port 19690 --port 19692 --extend-chunk-tokens 48
```

- `actual-boundary-01`: exit -2, 2026-11-01T05:17:22.186281+00:00 to 2026-11-01T05:27:27.187328+00:00; minimum free 38836371456 B.

```sh
python3 .temp/hybrid-a-r12-261101/actual_boundary.py
```

- `short-benchmark-01`: exit 0, 2026-11-01T05:17:23.337865+00:00 to 2026-11-01T05:18:14.893634+00:00; minimum free 39358849024 B.

```sh
python3 tools/hybrid-bench/hybrid_bench.py --client-state .temp/hybrid-a-r12-261101/short-client --scratch .temp/hybrid-a-r12-261101 --output .temp/hybrid-a-r12-261101 --label integrated-short --smoke --depths 128 --check
```

- `documentation-contract-01`: exit 0, 2026-11-01T05:17:46.252840+00:00 to 2026-11-01T05:17:46.761056+00:00; minimum free 39420932096 B.

```sh
python3 -c 'from pathlib import Path
for f in ["HYBRID.md","tools/hybrid/README.md","tools/hybrid/NODE-STATE-DESIGN.md","tools/hybrid-bench/README.md","tools/hybrid-proxy/README.md"]:
 s=Path(f).read_text();assert all(x in s for x in ["required by default", "experimental" if f=="HYBRID.md" else "Experimental", "0.0145", "62/64", "uncalibrated"] ) if False else all(x in s.lower() for x in ["required by default","experimental","0.0145","62/64","uncalibrated","bypassing the native sha guard"]);assert "establishes native state compatibility" not in s;print("contract consistent",f)
assert all(x in Path("README.md").read_text().split("# llama.cpp")[0] for x in ["LLAMA_BUILD_UI=OFF","LLAMA_USE_PREBUILT_UI=OFF"] )'
```

- `recurrent-01`: exit 0, 2026-11-01T05:18:27.689335+00:00 to 2026-11-01T05:18:29.192317+00:00; minimum free 39482298368 B.

```sh
python3 tools/hybrid/test_recurrent_append.py --library build/bin/libllama.so --malformed
```

- `native-01`: exit 0, 2026-11-01T05:18:29.256298+00:00 to 2026-11-01T05:21:45.917855+00:00; minimum free 39480680448 B.

```sh
build/bin/llama-hybrid-check /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf -
```

- `stop-short-client-01`: exit 0, 2026-11-01T05:19:36.203474+00:00 to 2026-11-01T05:19:36.706957+00:00; minimum free 39481131008 B.

```sh
tools/hybrid-bench/setup-client.sh --state .temp/hybrid-a-r12-261101/short-client --stop
```

- `source-identity-01`: exit 0, 2026-11-01T05:20:30.554829+00:00 to 2026-11-01T05:20:31.056655+00:00; minimum free 39480827904 B.

```sh
python3 -c 'import hashlib,json,subprocess;from pathlib import Path
r=Path(".temp/hybrid-a-r12-261101"); areas=["src","common","ggml","tools/server","tools/hybrid","tools/hybrid-proxy","tools/hybrid-bench"]; diff=subprocess.check_output(["git","diff","70778be629","--",*areas],text=True);code=[x for x in diff.split("diff --git ")[1:] if not x.splitlines()[0].endswith(".md")];assert not code
files=[Path("build/bin")/n for n in ["llama-hybrid-prefill","llama-server","llama-hybrid-check","llama-hybrid-check-codec","llama-hybrid-check-append","llama-hybrid-check-guards","llama-hybrid-check-distributions","llama-hybrid-check-client","llama-hybrid-bench-check","libllama.so","libggml-cpu.so"]];d={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(r/"binary-sha256.json").write_text(json.dumps(d,indent=2));print(json.dumps(d))'
```

- `append-01`: exit 0, 2026-11-01T05:21:45.968684+00:00 to 2026-11-01T05:23:35.055240+00:00; minimum free 38890934272 B.

```sh
build/bin/llama-hybrid-check-append /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden/prompt-512 /home/agent/Werkstatt/hybrid-models/golden/prompt-2048 /home/agent/Werkstatt/hybrid-models/golden/prompt-8192
```

- `short-audit-01`: exit 0, 2026-11-01T05:22:07.772522+00:00 to 2026-11-01T05:22:08.275693+00:00; minimum free 39479984128 B.

```sh
python3 .temp/hybrid-a-r12-261101/audit_short.py
```

- `guards-01`: exit 0, 2026-11-01T05:23:35.116895+00:00 to 2026-11-01T05:23:37.118959+00:00; minimum free 39479570432 B.

```sh
build/bin/llama-hybrid-check-guards /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden/prompt-512
```

- `distributions-01`: exit 0, 2026-11-01T05:23:37.168389+00:00 to 2026-11-01T05:27:10.333868+00:00; minimum free 39478157312 B.

```sh
build/bin/llama-hybrid-check-distributions /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden/prompt-512.tokens.json /home/agent/Werkstatt/hybrid-models/golden/prompt-2048.tokens.json /home/agent/Werkstatt/hybrid-models/golden/prompt-8192.tokens.json
```

- `client-prepare-01`: exit 0, 2026-11-01T05:27:10.385061+00:00 to 2026-11-01T05:27:16.892208+00:00; minimum free 39176597504 B.

```sh
build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --prepare /home/agent/Werkstatt/hybrid-models/golden/ /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/client-fixtures
```

- `client-512-01`: exit 0, 2026-11-01T05:27:16.970505+00:00 to 2026-11-01T05:27:17.973479+00:00; minimum free 39150051328 B.

```sh
build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/client-fixtures/cost-512 512
```

- `client-2048-01`: exit 0, 2026-11-01T05:27:18.027762+00:00 to 2026-11-01T05:27:19.029992+00:00; minimum free 39149883392 B.

```sh
build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/client-fixtures/cost-2048 2048
```

- `client-8192-01`: exit 0, 2026-11-01T05:27:19.084252+00:00 to 2026-11-01T05:27:20.586719+00:00; minimum free 39149625344 B.

```sh
build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/client-fixtures/cost-8192 8192
```

- `client-capacity-01`: exit 0, 2026-11-01T05:27:20.641707+00:00 to 2026-11-01T05:27:21.643370+00:00; minimum free 39149559808 B.

```sh
build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --capacity /home/agent/Werkstatt/hybrid-models/golden/prompt-512
```

- `remote-append-01`: exit 0, 2026-11-01T05:27:21.702315+00:00 to 2026-11-01T05:27:40.221265+00:00; minimum free 38962917376 B.

```sh
python3 tools/hybrid-proxy/test_remote_append.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/remote-append-01 --prefill-binary build/bin/llama-hybrid-prefill
```

- `actual-boundary-02`: exit 0, 2026-11-01T05:27:27.080866+00:00 to 2026-11-01T07:48:38.303275+00:00; minimum free 34001260544 B.

```sh
python3 .temp/hybrid-a-r12-261101/actual_boundary.py
```

- `append-api-01`: exit 0, 2026-11-01T05:27:40.261251+00:00 to 2026-11-01T05:28:22.789143+00:00; minimum free 38779105280 B.

```sh
python3 tools/hybrid-proxy/test_append_api.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/append-api-01
```

- `concurrency-native-01`: exit 0, 2026-11-01T05:28:22.834426+00:00 to 2026-11-01T05:28:43.351210+00:00; minimum free 37719093248 B.

```sh
python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/concurrency-native-01
```

- `decode-proxy-01`: exit 0, 2026-11-01T05:28:43.419987+00:00 to 2026-11-01T05:30:34.011790+00:00; minimum free 36656562176 B.

```sh
python3 tools/hybrid-proxy/test_decode.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/decode-proxy-01 --prefill-binary build/bin/llama-hybrid-prefill --golden /home/agent/Werkstatt/hybrid-models/golden/prompt-512
```

- `header-http-01`: exit 0, 2026-11-01T05:30:34.052143+00:00 to 2026-11-01T05:36:26.337017+00:00; minimum free 36530470912 B.

```sh
python3 tools/hybrid-proxy/test_header_budget.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/header-http-01 --prefill-binary build/bin/llama-hybrid-prefill
```

- `model-hashes-01`: exit 0, 2026-11-01T05:32:10.549889+00:00 to 2026-11-01T05:32:12.053792+00:00; minimum free 37299159040 B.

```sh
sha256sum /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf
```

- `conversation-thinking-on-01`: exit 0, 2026-11-01T05:36:28.396262+00:00 to 2026-11-01T05:37:18.443223+00:00; minimum free 35892678656 B.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/conversation-thinking-on-01
```

- `conversation-thinking-off-01`: exit 0, 2026-11-01T05:37:18.500002+00:00 to 2026-11-01T05:38:06.039283+00:00; minimum free 34946379776 B.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/conversation-thinking-off-01 --thinking off
```

- `conversation-inferred-omit-01`: exit 0, 2026-11-01T05:38:06.114431+00:00 to 2026-11-01T05:38:55.665274+00:00; minimum free 34001223680 B.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/conversation-inferred-omit-01 --inferred --omit-reasoning
```

- `conversation-no-checkpoint-01`: exit 0, 2026-11-01T05:38:55.766258+00:00 to 2026-11-01T05:39:08.776452+00:00; minimum free 34208989184 B.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/conversation-no-checkpoint-01 --no-checkpoint-control
```

- `prune-closed-01`: exit 0, 2026-11-01T05:38:59.800556+00:00 to 2026-11-01T05:39:04.807723+00:00; minimum free 34208985088 B.

```sh
python3 .temp/hybrid-a-r12-261101/prune_closed.py
```

- `typed-identity-01`: exit 0, 2026-11-01T05:39:08.829225+00:00 to 2026-11-01T05:39:43.854808+00:00; minimum free 39808200704 B.

```sh
python3 tools/hybrid-proxy/test_message_identity.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/typed-identity-01
```

- `typed-no-checkpoint-01`: exit 0, 2026-11-01T05:39:43.905154+00:00 to 2026-11-01T05:40:05.923140+00:00; minimum free 39847071744 B.

```sh
python3 tools/hybrid-proxy/test_message_identity.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/typed-no-checkpoint-01 --no-checkpoint-control
```

- `identity-01`: exit 0, 2026-11-01T05:40:06.007016+00:00 to 2026-11-01T05:40:11.014254+00:00; minimum free 39867097088 B.

```sh
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --requantized /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/identity --case identity
```

- `deep-http-01`: exit 0, 2026-11-01T05:40:11.082941+00:00 to 2026-11-01T06:17:07.526873+00:00; minimum free 38550536192 B.

```sh
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/deep --case deep
```

- `ui-original-mutant-01`: exit 0, 2026-11-01T06:03:38.949746+00:00 to 2026-11-01T06:03:39.452714+00:00; minimum free 39573585920 B.

```sh
python3 .temp/hybrid-a-r12-261101/ui_original_mutant.py
```

- `deep-distributions-01`: exit 0, 2026-11-01T06:17:07.582260+00:00 to 2026-11-01T07:33:54.579268+00:00; minimum free 35460886528 B.

```sh
build/bin/llama-hybrid-check-distributions /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r12-261101/deep/deep.tokens.json --max-ctx 65536
```

- `proxy-unit-02`: exit 0, 2026-11-01T06:30:41.194818+00:00 to 2026-11-01T06:32:22.284461+00:00; minimum free 38208618496 B.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v
```

- `documentation-contract-02`: exit 0, 2026-11-01T06:37:59.378252+00:00 to 2026-11-01T06:37:59.881917+00:00; minimum free 38868172800 B.

```sh
python3 -c 'from pathlib import Path
for f in ["HYBRID.md","docs/hybrid/prototype-spec.md","tools/hybrid/README.md","tools/hybrid/NODE-STATE-DESIGN.md","tools/hybrid-bench/README.md","tools/hybrid-proxy/README.md"]:
 s=Path(f).read_text(); assert all(x in s.lower() for x in ["required by default","experimental","0.0145","62/64","uncalibrated","bypassing the native sha guard"]); assert "establishes native state compatibility" not in s; print("contract consistent",f)
assert all(x in Path("README.md").read_text().split("# llama.cpp")[0] for x in ["LLAMA_BUILD_UI=OFF","LLAMA_USE_PREBUILT_UI=OFF"])'
```

- `mapped-libraries-01`: exit 1, 2026-11-01T07:24:27.768535+00:00 to 2026-11-01T07:24:28.272218+00:00; minimum free 38629380096 B.

```sh
python3 .temp/hybrid-a-r12-261101/check_mapped_libraries.py
```

- `mapped-libraries-02`: exit 0, 2026-11-01T07:24:41.394195+00:00 to 2026-11-01T07:24:41.897970+00:00; minimum free 38629351424 B.

```sh
python3 .temp/hybrid-a-r12-261101/check_mapped_libraries.py
```

- `integrated-final-01`: exit 0, 2026-11-01T07:28:14.906033+00:00 to 2026-11-01T07:28:58.949013+00:00; minimum free 38553174016 B.

```sh
python3 tools/hybrid-bench/test_integrated.py --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r12-261101/integrated-final
```

- `public-final-01`: exit 0, 2026-11-01T07:28:16.163679+00:00 to 2026-11-01T07:28:21.676243+00:00; minimum free 38554443776 B.

```sh
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r12-261101/public-final --case public
```

- `codec-final-01`: exit 0, 2026-11-01T07:28:21.834297+00:00 to 2026-11-01T07:28:22.337045+00:00; minimum free 38554361856 B.

```sh
build/bin/llama-hybrid-check-codec --max-ctx 262144
```

- `final-source-audit-01`: exit 1, 2026-11-01T08:46:56.501198+00:00 to 2026-11-01T08:46:57.002454+00:00; minimum free 38256553984 B.

```sh
python3 .temp/hybrid-a-r12-261101/final_source_audit.py
```

- `final-source-audit-02`: exit 0, 2026-11-01T08:47:02.981982+00:00 to 2026-11-01T08:47:03.482894+00:00; minimum free 38256525312 B.

```sh
python3 .temp/hybrid-a-r12-261101/final_source_audit.py
```

- `report-draft-03`: exit 0, 2026-11-01T08:52:13.306126+00:00 to 2026-11-01T08:52:13.807681+00:00; minimum free 38181691392 B.

```sh
python3 .temp/hybrid-a-r12-261101/report_draft.py
```

- `a2a-post-help-01`: exit 2, 2026-11-01T09:39:42.693775+00:00 to 2026-11-01T09:39:43.194509+00:00; minimum free 37907972096 B.

```sh
a2a post --help
```

- `a2a-help-01`: exit 0, 2026-11-01T09:39:54.943788+00:00 to 2026-11-01T09:39:55.444482+00:00; minimum free 37907943424 B.

```sh
a2a --help
```

- `a2a-skill-search-01`: exit 0, 2026-11-01T09:40:11.121420+00:00 to 2026-11-01T09:40:11.622157+00:00; minimum free 37907730432 B.

```sh
python3 -c 'from pathlib import Path; print("\n".join(str(p) for r in (Path.home()/".agents/skills",Path.home()/".codex/skills") if r.exists() for p in r.rglob("SKILL.md") if "a2a" in str(p).lower()))'
```

- `a2a-whoami-final-01`: exit 0, 2026-11-01T09:40:32.687275+00:00 to 2026-11-01T09:40:33.188125+00:00; minimum free 37907124224 B.

```sh
a2a whoami
```

- `long-audit-01`: exit 0, 2026-11-01T09:56:39.836352+00:00 to 2026-11-01T09:56:40.337139+00:00; minimum free 38834343936 B.

```sh
python3 .temp/hybrid-a-r12-261101/audit.py
```

- `close-ready-01`: exit 0, 2026-11-01T09:56:44.635904+00:00 to 2026-11-01T09:56:45.136624+00:00; minimum free 38834278400 B.

```sh
python3 .temp/hybrid-a-r12-261101/runtime_cleanup.py --check-ready
```

- `runtime-cleanup-01`: exit 0, 2026-11-01T09:56:45.175030+00:00 to 2026-11-01T09:56:46.175991+00:00; minimum free 38834266112 B.

```sh
python3 .temp/hybrid-a-r12-261101/runtime_cleanup.py
```

- `report-final-01`: exit 0, 2026-11-01T09:57:07.077006+00:00 to 2026-11-01T09:57:07.577844+00:00; minimum free 38833725440 B.

```sh
python3 .temp/hybrid-a-r12-261101/report_draft.py
```

## Log hashes before scratch deletion

| Log | SHA-256 |
|---|---|
| a2a-help-01.log | `50b07bd0fc8c311ccc00df607726408a6174a91d26f65a332fabbbb4f4670ce2` |
| a2a-post-help-01.log | `6a6d0474a459b8dad7f588f59d2e77eb1f6c6c83352b62fdedc4e87ffc6aa9f7` |
| a2a-readiness-01.log | `50b07bd0fc8c311ccc00df607726408a6174a91d26f65a332fabbbb4f4670ce2` |
| a2a-skill-search-01.log | `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b` |
| a2a-whoami-01.log | `0cad04bc31e87e9c4f65dbd740ae1043419a941961b376b9c2e581e3ba3bb446` |
| a2a-whoami-final-01.log | `0cad04bc31e87e9c4f65dbd740ae1043419a941961b376b9c2e581e3ba3bb446` |
| actual-boundary-01.log | `326bddd2bdf9d705f44122f6f78bdf86f91393aa0e468a87cfbd4856b6f49933` |
| actual-boundary-02.log | `60581768f913796f6de02054d1252e5c8dbe57e9d22d51929e6277f1f25c5379` |
| append-01.log | `afac589af21ad62d736296d51e09d90cbd5f1a4d442c20e912b01103020f3686` |
| append-api-01.log | `0a5239b791283423f53e2e33a20dbc68eb94927e3bf6f458e9f23ce5b3ee062f` |
| bad-hash-client.log | `31f95e2ff9b648e04513cc29ef2e07b3956c0141911e4a529cbf2bafc5d5c522` |
| bad-hash-node.log | `31f95e2ff9b648e04513cc29ef2e07b3956c0141911e4a529cbf2bafc5d5c522` |
| bench-unit-01.log | `c909db6b7a003408e0cb17654e17229ac6d9c013849bca793ab5685b2b2f6d43` |
| bench-unit-02.log | `4b9e572d526d91ab82d185d53414e217f534ee276745c07fdf146ca956bc9f1d` |
| bench-unit-03.log | `53a6f9431a9f2ae4a37addc29ea651a59dd5034ce49f31245aed887fbac1f437` |
| board-goal-01.log | `abeb0d9a6acedfa50538bc4309f177b39817911a6dda8632273227b777362f5b` |
| build-01.log | `b3ab137958dab33ef990f58fe3d3eb8f6c6dc0fb4680e4b066eef5e65fd82037` |
| build-codec-02.log | `4ce1aa799e0f127cf8a12b80408dd3f8c09ddfe5602694b74fb7c9d4cd712845` |
| build-codec-03.log | `4ce1aa799e0f127cf8a12b80408dd3f8c09ddfe5602694b74fb7c9d4cd712845` |
| client-2048-01.log | `8d6d41bc53f8b993f95f4f43a149a6137fb9412874f7c939a0c2e5b174539ea7` |
| client-512-01.log | `57dbee904355bbcad3af92a45fea74b13e96a592c94eb911c789bd573e845003` |
| client-8192-01.log | `afd179a4dd80ced82b2a8ce06da42714a8cda85598a9c92df632caa288431eaf` |
| client-capacity-01.log | `6b72188d26a341ac1fdb764b750ef893ecb3a299bf02fd116ef5c36b978ac142` |
| client-prepare-01.log | `0b770ebd907a93870b01c06f2a73efa125c5edab14b201f27c85233081d6bb4a` |
| close-ready-01.log | `68c7dd26a1b34f2121163bd7eef66afb983b05a5309adf44256393efc2607878` |
| cmake-readiness-02.log | `4d92ce18ddf9889d463ac875608423312823ad89e7fd5178b02a21ea26a2bd3d` |
| codec-01.log | `11c0afa23f304d27199d0281709573911946040e2576fcf240d31aa4936a10e6` |
| codec-02.log | `5efa8a3689f7f61dd72d210e144b58c22f61ecb1e27f0f66c0a1050d27f7a101` |
| codec-03.log | `5efa8a3689f7f61dd72d210e144b58c22f61ecb1e27f0f66c0a1050d27f7a101` |
| codec-core-01.log | `dd7e593d25f1c2a9085d05b5bc1f12626ad15854a50f2e2e97fb545879846624` |
| codec-final-01.log | `5efa8a3689f7f61dd72d210e144b58c22f61ecb1e27f0f66c0a1050d27f7a101` |
| concurrency-native-01.log | `7af680806fdf2e1d1428ec1efd2f23cd919ba3d68159c1db7e57e627a2cbd1ba` |
| configure-01.log | `6bb6e90047c108d3d6919cbd028e7562c72951026bcf524c34d9e439ce76dedc` |
| context-overflow.log | `7921e3f9e4ea5f909842c86948d38480f1dc507436593681f6c9cb06ee7e62c4` |
| conversation-inferred-omit-01.log | `2f10b0b15dd7e8578429c48a67087b3f7d638b56e17e03f628aaa29304450f17` |
| conversation-no-checkpoint-01.log | `d5dccbc84cd090ecfd8d06614b3281260ec8670619c04d3fbfd90b9a25246f14` |
| conversation-thinking-off-01.log | `f2412f94c02fa54a599751c5e4e4d752e3394175029352096514d0a37380969d` |
| conversation-thinking-on-01.log | `2f10b0b15dd7e8578429c48a67087b3f7d638b56e17e03f628aaa29304450f17` |
| cpu-observation-01.log | `7cb7108f618eb91fef670b54644901ac4fd3209b03ae70d6b9430ac0f935e46c` |
| decode-proxy-01.log | `d2970732a4e592b49d18472aea677fe515933045af5bb16b03aabeda3215c400` |
| deep-distributions-01.log | `99f2e3baed358670f896e4a139c6cad0c81c8706f6ac444e323c61fd98b94d37` |
| deep-http-01.log | `e152533fe44f8a0a2241c983c75774358708b7209fb614969333937b28f8f972` |
| different-model.log | `8022741b12a118da90640d7ac792b101f042c20cd6081109530d8793142d5e5f` |
| distributions-01.log | `e3f41a5c4071c21db9bc66a45144f004e4c0958a622adf580f470debe48a8481` |
| documentation-contract-01.log | `06441ba0b8ffa2a452671137196e9772c430e578b29545027f85f760194b6bac` |
| documentation-contract-02.log | `8c292ffa6053eadc6d07662e1b5d4b409f516c822e9760546b45b136a20e2f08` |
| download-mismatch.log | `1a3ca8703da85e54c09eb9118f2a69550cb483461bf6f3ba1f22e1d0ec9f4632` |
| editorconfig-readiness-01.log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| editorconfig-readiness-02.log | `3f7bb069a98a6563076937c5d01852c0ca593b5860ccca34cc75a9b80de86e98` |
| final-source-audit-01.log | `8a26197ed9477c3e0cbc2a5979b4fd5c5860ab088dc6b722b9fd902a3001a2fc` |
| final-source-audit-02.log | `6a326263ed8c96e28f8137d30c7b17ee321b7325ba621aa4acb721d9679b200e` |
| fingerprint-refusal-01.log | `2d1b2edcaf5910840f7f363dfcbc9dceff3297a815d5267a96b520a8c248a67d` |
| geometry-01.log | `df1aa66eb2ce14cb26339ad13509473c2485dc64c3d464373b29a8ecf11d9bbe` |
| guards-01.log | `b91302ad758be163e027473a20ce122070705a83f4d1379331f2cc158257a0f0` |
| header-http-01.log | `1377f8e031ac8448141ae22536cb83c4135c0860bbe5f16b25484614ca72bbad` |
| hetero-benchmark-01.log | `ba755d82b3b1aadc77d6f2074c269b00ccfe3fb0b94151512d37b21d9bfcbded` |
| hetero-refusal-01.log | `8022741b12a118da90640d7ac792b101f042c20cd6081109530d8793142d5e5f` |
| http-default-01.log | `29922a5c22d322460ff794487a81ee189919836bcd2b9310b80eae5d3f4166c2` |
| http-errors-01.log | `1ac0e68243f9e46dc96096ac7c2e596566fd4abea0d86efc804c3c0cf920b7e2` |
| http-max-01.log | `4a9792ca66db0bbe09e40549289eab413e10665e34dc2e8cee47e87734266236` |
| http-pending-01.log | `73fc7e86b92612010f27f3194f775f34708695e2a97f0274bc89beeefbcd1e75` |
| http-slow-01.log | `0991889190613d14a12ba8c1f3ba5c408d97bce15341033d05d8bef67d24ead9` |
| identity-01.log | `0482e27981c9a576b9aea4049b31b2fd0dbf5904d1aff8da65b891041d315ad3` |
| integrated-01.log | `e7df450b583b5c9ba7d8dd9c4623dca0b943dd3a409ff4609bca14ccf4780a57` |
| integrated-final-01.log | `e7df450b583b5c9ba7d8dd9c4623dca0b943dd3a409ff4609bca14ccf4780a57` |
| killed-mid-run.log | `c3394ba997fa5326b4564b78621d493345796bac7e1d6af4dea02f106250c234` |
| launch-client-01.log | `e9e3886cf3bbef23a4053672e17b8a188670ad37e8031e5cba15776cb19a330e` |
| launch-control-client-01.log | `30530886bc22df3196358d0315a761f793f7aa4a21f64643cd73e17887855b80` |
| launch-control-node-01.log | `4ac0252cd2f88e6aeca55b70e6574416635f63a8d6b714ea161c34b3459bd982` |
| launch-hetero-client-01.log | `34a5a31139afd7b71562d99e32cc6121b52c3e43cef0e11a5acc8280ff36c620` |
| launch-node-01.log | `df327a1b79ab41db4eb9e818cec425e5b552d6055462cb7783c9f7db6f88026a` |
| launch-short-client-01.log | `1c8b20eff35861f79d6ee9b59fa58bd549ab182c0d9219f13b08eaa010a9fda7` |
| launcher-audit-01.log | `8e11956a291ac8a6b0025c7a9a8ec6fb93d0aea1c489c5854d479b098055622f` |
| launcher-audit-02.log | `649d9c6d13e7515ac6b02cbdef9b9c590375f09305e690ffe65a5b83c24e6203` |
| limits-01.log | `59a452c17162868fd5fce4c50196b0a9b9e12d5031b0b20bfd6ab01d4a0a1cbc` |
| long-audit-01.log | `c9f3a8e02d6f5425aeda6a83169003d26b57a43a25b779064a0cd27c13340c93` |
| long-benchmark-01.log | `e889f945dd59ecdd556e48ee42d6ab8545496d1e40606379d04b78476c337de4` |
| mapped-libraries-01.log | `ff32186f3d417a81af31de49044663b90ae5fa066f287ebc861d056393be0203` |
| mapped-libraries-02.log | `3c779b206a311eec8805ac81b26d0d3333f189f75a2c94ef456b4494feff09ca` |
| missing-client-token.log | `99102849d86e7de6dc4c2a7811aceee60b6edb2096a82b0c655c33492669f54a` |
| model-hashes-01.log | `42f776469c88c9f3d16cda0f4c26f65080015bb7298631d763d7c8b7e74ca443` |
| monitor-01.log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| more-regression-driver-01.log | `b481cf150be21d6ffe1b4ad1445ffdd639269d63667ef49cd25a442f8a52984f` |
| mutants-01.log | `7dd197120ba1094a19894b834769d59b557825c6062d5b1c63bc7a67cf322f6c` |
| mutants-02.log | `b0a6cfd5babed9cc3a15081f2c3aa12badacbe0afb679e85e78cf5ef3d434e37` |
| naming-style-01.log | `19aa7200204dbbfdf1c375e50050a0b6490871a8c1986ed63331ae682d9c145f` |
| native-01.log | `53e9f53b3f6991ac8c62a1b8587f6386328feadc4f5db0c658f1d5e49a38e91b` |
| native-controls-01.log | `f7d31663414b905441263c25e2c1a2f766acd0012a41f12555b86ac579c38b71` |
| old-public-01.log | `0bb98ffba20c142d89ccd82efd35efa36e6c287d8eb45406a97b0e92e63016df` |
| old-timing-01.log | `13cff0d8f5da893ec2ceef07aa63d1f3544c854e123a95b84fd5568aa5c9e35a` |
| omit-cell-metadata.log | `e45372c236bc9491dda5b140943683d28a4a005a36665a81d94910f667caa0de` |
| proxy-unit-01.log | `76840051f7ce63779e7e77ede27358e2361e7b5ae6003e435958151428534ab5` |
| proxy-unit-02.log | `746e7b98e8eaf8c817f9a086ed72c3ce5f9e1231466bf35b4acd551ad071e78b` |
| prune-closed-01.log | `836eef680682a3ccd1197d6091696deb6a7a98508a57593a0b964d6760e85e10` |
| public-final-01.log | `e267a2b1a5cf0a59afe62c2e8833ad4c6df2740cef25fa9ce848bd43cc376242` |
| public-no-token.log | `bbcfc51710372a81154a7646674f7ec0a101eb8bdf1f450bc20e66548f04a903` |
| recurrent-01.log | `904b69583ae23873ff8c9ad91b977db5cd8e243b99ce010da1f98d9c75f78691` |
| regression-driver-01.log | `b3f7b610dd395bfe6c7c7c16015f6caf8aceec6e4da9a7ae24e9644b93a7a9bb` |
| remote-append-01.log | `b4f46296b65f19400f052ba7aeb884a1f82ac860c888299ec1d7743d35e50034` |
| report-draft-03.log | `c09403eea62701eeec5abbd5565e9b9d07f1b313ec64d449c4d06e365f7be610` |
| report-final-01.log | `c09403eea62701eeec5abbd5565e9b9d07f1b313ec64d449c4d06e365f7be610` |
| report-final-02.log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| reprefill-control.log | `b19be70cd8a506ee8d94e1bda6174944a0e3570494cb02c90066e35a84a8bdc9` |
| review-timing-repro-01.log | `2eadfe0f6ab5115413e66ae58a83a5fc1264dc92e41d25453100d65b7cceb4d8` |
| runtime-cleanup-01.log | `0db6018557eb8b855e665da8b3220203d4e1f092c63516fdd845fea0c69716c9` |
| scope-01.log | `ad6c09083dfa63cc9631cefae44cbc6d291adef09f2c50a358ea7838196e076a` |
| short-audit-01.log | `4d7b2b6bad9de20fe2bce98321956e8dd7c51f1c56c6ecc18a33373998497d67` |
| short-benchmark-01.log | `775379b8745efd3ded36f707ae81a8c4ac9fed529cb72aebb41bc7ec68a058da` |
| source-identity-01.log | `3daceaafc0d2a6f1ed1ec2289d43f63819ba5b08b049aff1a6af38f7d5b9cfff` |
| stop-control-client-01.log | `5198400ce9cea55da6f59c873c52563e16dde69d4ad255a557a3f689c5f8ebad` |
| stop-hetero-client-01.log | `5198400ce9cea55da6f59c873c52563e16dde69d4ad255a557a3f689c5f8ebad` |
| stop-short-client-01.log | `5198400ce9cea55da6f59c873c52563e16dde69d4ad255a557a3f689c5f8ebad` |
| task-board-readiness-01.log | `78fc4e202d9d9b8f1147d41f15021d6f7a649894269546896deef9671229f13b` |
| tool-readiness-01.log | `11ed4d70fcfb6746066064a301e6a933298fe873b2ed65d5a80a0c61f0da9ad6` |
| type-tool-readiness-01.log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| type-tool-readiness-02.log | `ab3f46866dd650c9cc8497453795600842c2225cce49d42bbe6b6673d88ec88e` |
| typed-identity-01.log | `508ae2d8e4fa302c58477e3c92e8a672fd272954e3dd79d42778929704f78bce` |
| typed-no-checkpoint-01.log | `d6e1dd6120d52a8be2ddbee5200e27e96c898c0b5847e94fc97dbcea604894ce` |
| ui-flags-removed.log | `15f31e362fbb12fdc66f1f37932d3adfd0a45721057b640ea4cc1b9139092ce0` |
| ui-original-mutant-01.log | `70f4e3bd714804d51acb0514899a1002fe2449e7d38cc234cb126ebb43c35a9e` |
| ui-original-mutant.log | `5c270b5e2eb09ff1a3d5784ae3320f27cf1fbd7eb72cc917855f51fe99fe47d1` |
| underestimate-27B-default.log | `7e2f58e3d62e7b984e0c33ee62511ae09c73e6ffe1f421367eafe02f6f80aa61` |
| unreachable.log | `5d0133238236da9d23b339da70dd836a6034fdcff41f28c93d432bb0437e1ca8` |
| wrong-client-token.log | `1da857d531c0e863466b63282e8809f6cba4ff9a20aac1687d0d503788210d16` |

## Reproduce the full maximum-chunk export

Create `.temp/hybrid-a-r12-261101/actual-boundary/`, save the following as `.temp/hybrid-a-r12-261101/actual_boundary.py`, and run the `actual-boundary-02` ledger command from the checkout root. This is the exact measurement script; its three-hour read timeout changes only the probe, not the native service or benchmark.

```python
# A real node exports the largest planned 0.8B chunk below the 512 MiB body cap; reserve-bound one-more startup refusal is covered by test_native_chunk_startup_bound.
import argparse,json,struct,sys,time,urllib.request
from pathlib import Path
ROOT=Path.cwd();sys.path[:0]=[str(ROOT/'tools/hybrid-bench'),str(ROOT/'tools/hybrid'),str(ROOT/'tools/hybrid-proxy')]
from test_limits import Node
from bench_common import context_plan,host_info
from policy import extend_chunk_limit
R=Path(__file__).resolve().parent
model='/home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf'
plan=context_plan(model,host_info(),65536,True);count=extend_chunk_limit(plan['kv_bytes_per_token'],plan['recurrent_state_bytes']);ctx=((count+255)//256)*256
args=argparse.Namespace(binary=str((ROOT/'build/bin/llama-hybrid-prefill').resolve()),model=model,artifacts=str(R/'actual-boundary-02'))
with Node(args,'node',('--max-ctx',str(ctx)),context=str(ctx)) as node:
 p=node.create();started=time.monotonic()
 body=dict(n0=0,prefix_hash_n0=__import__('hashlib').sha256(b'').hexdigest(),tokens=[100]*count,return_delta=True)
 request=urllib.request.Request(node.base+p+'/extend',json.dumps(body).encode(),headers={'Authorization':'Bearer '+node.clients['a'],'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=10800) as response:
  assert response.status==200
  wire=response.read()
 seconds=time.monotonic()-started
 header_size,=struct.unpack_from('<I',wire);header=json.loads(wire[4:4+header_size]);body_size=len(wire)-header_size-52
 assert header['n1']==count and len(header['tokens'])==count and body_size<=512*1024**2
 assert body_size==20202100+6552*count
 state=node.request('GET',p);assert state['n']==count
 node.request('DELETE',p,expected=204);health=node.request('GET','/v1/health');assert health['sessions']==health['ram_bytes']==0
 result=dict(tokens=count,context=ctx,wire_bytes=len(wire),body_bytes=body_size,header_bytes=header_size,cap=512*1024**2,seconds=seconds,status=200,release_sessions=0,release_bytes=0)
 (R/'actual-boundary/result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
```

## Identity and scratch accounting

Final source audit verifies the runtime C++/Python and test files are unchanged from signed `70778be629`, every introduced commit is signed, original reviewer test names remain, all measured binary/library SHA-256 values are unchanged, and `src/` has no delta from r11. It admits the specifically requested public specification note in the documentation scope. The subcommand `a2a post --help` returned2 (help requested, no message sent); root `a2a --help` returned0 and supplies the post syntax. Audit attempt01 found one trailing blank line at README EOF (`git diff --check` exit2, audit1); that line was removed and attempt02 exits0.

| Measured binary/library | SHA-256 |
|---|---|
| build/bin/libggml-cpu.so | `2fd4283f9615f258b4a3c3e74e93e69961d545e2314f70dad674846e5e8b1fdb` |
| build/bin/libllama.so | `84128f68790f1cf122c786a1a55ca140d9ac1a97415f3d3d523b437e43d2b63e` |
| build/bin/llama-hybrid-bench-check | `294293b5d612260ad626666decddc45f80cf1d0a9b0ec34ad93c5a13cf94f033` |
| build/bin/llama-hybrid-check | `0d6766f44fa94cca58da7fcde555ae68355084851682da7d4da1891453318398` |
| build/bin/llama-hybrid-check-append | `56139e5f1f3657d4400cfa840f453cfc22313cd4e8306a38fc1230e07342fcba` |
| build/bin/llama-hybrid-check-client | `f0d4b1667182d8bddaa6a8cfe5cdb699c43bfd3568faea86151ab9275b1c39d8` |
| build/bin/llama-hybrid-check-codec | `44947787cca4477bae465e340cae56a23805ad2d50032d183bf3b9be03572e04` |
| build/bin/llama-hybrid-check-distributions | `7a19bcb52a6bbbe0c44bc5bcd5795ab3256ee04f3615194d6206420c96ff50ed` |
| build/bin/llama-hybrid-check-guards | `ebaa9473c668e87f7c6faf358d8d84a89d910c81fa1aab70deb4b2555e5407ef` |
| build/bin/llama-hybrid-prefill | `0b7541a9378243efeebd18353a1cb6e20a27ce9ca50ddb84d4c498f3027e8764` |
| build/bin/llama-server | `961a47be04fbf12a02a0f53ecb8d04d3259b25f2f6041d83bdaa56bfd64ff242` |

`pruned-state-hashes.jsonl`: 56 already-hashed/deleted files, 5769262038 bytes, manifest SHA-256 `4957310181661c4638dc092a3d2db0fd317a09adf95756761d68d62cd1527c7d`. Raw files are not part of the final remaining-root count below.
`state-dump-hashes.jsonl`: 52 already-hashed/deleted files, 1097462551 bytes, manifest SHA-256 `1c363b8a64c75b20516e5606dcb0ee9b9d1235fe3ed955985f6bec0d3c6ef32a`. Raw files are not part of the final remaining-root count below.

Monitor observations 2026-11-01T05:12:41.612572+00:00 through 2026-11-01T09:56:43.941967+00:00: minimum free 34069901312 bytes, peak retained node state 565402124 bytes, maximum 2 sessions, maximum eviction counter 0. These are sampled observations, not an allocator or device-memory bound. The node budget is 587,202,560 bytes; 2 serialized 40k states fit, while 3 would not. The model/context, request/response buffers and checker copies are additional host allocations.

Runtime closure 2026-11-01T09:56:45.175030+00:00 through 2026-11-01T09:56:46.175991+00:00, exit 0: authenticated node health was zero sessions/zero retained bytes before stopping the launcher-managed runtimes. Only recorded PID/start identities were stopped; all 80 recorded runtime PIDs were inactive afterwards. The monitor stopped through its task sentinel and exited 0 before final root deletion. Shared models, lock and incremental build cache were preserved.


## Final report verification

`python3 .temp/hybrid-a-r12-261101/final_report_check.py` exited 0 at 2026-11-01T09:58:05.181181+00:00. It verified the exact persisted 40k numerical rows and declared limits, scanned 149 private runtime credential values for absence, and rejected an in-memory public sentinel credential plant for the intended assertion. No real credential was planted or written. The subsequent scratch deletion appends only content-free hashes, counts and timestamps.

The final repeated source audit, `python3 .temp/hybrid-a-r12-261101/final_source_audit.py`, exited0 at 2026-11-01T09:58:32.437974+00:00; only reviewed documentation/report changes followed the measured runtime commit.

| Copied-record/public-sentinel plant | Gate narrowed | Named failing assertion | Survival bound |
|---|---|---|---|
| Agreement63 with a green status | Actual64 agreements | validate/same partition | Killed; no native mutation claim |
| KL0.02 with a green status | Reference KL bound | validate/KL | Killed |
| Released session count1 | Empty retained store after release | validate/retained state leak | Killed |
| Public sentinel credential | Credential-value scanner applies | no_credentials/credential value in report | Killed; no real credential written |

Closing log SHA-256 values:

- final-report-check-01.log: `fb3207a6ab13e6d4d3d93e950432ce655d7c179929f6ec78d13238c7ecfcf56e`
- final-source-audit-03.log: `6a326263ed8c96e28f8137d30c7b17ee321b7325ba621aa4acb721d9679b200e`

## Requested scratch deletion

At 2026-11-01T09:59:00.302601+00:00, every remaining regular file in `.temp/hybrid-a-r12-261101/` was streamed through SHA-256, then that exact task root was deleted and absence verified: 809 files, 1055303740 bytes. Canonical sorted JSONL manifest SHA-256: `79143773171ddb3ceeb92c97bad3eead75607c1c762685192c8f78be1d7c6b4b`. The manifest and raw scratch were discarded; only this content-free digest and summary remain. Final free bytes: 39891288064. Shared models, the shared lock, the existing task tracker and incremental build cache were preserved. Runtime cleanup and monitor had exited 0 before hashing; no active task writer remained.

Post-deletion staged whitespace validation: the first `git diff --cached --check` returned2 for one trailing space in the added closing-log caption; the space was removed and the repeated command returned0. Runtime sources and numerical evidence were unchanged.

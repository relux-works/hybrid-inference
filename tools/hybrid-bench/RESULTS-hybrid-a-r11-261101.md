# M1 A/B integration, CPU 0.8B

Run `hybrid-a-r11-261101`. Ready for review: the requested local merge is signed, and the homogeneous CPU integration and native regression gates passed. The unavailable different-base-model case, heterogeneous numerical limit and no-download process deviation remain explicit below. This is local evidence, not independent acceptance or publication.

## Source and boundaries

Local branch `m1-integrated` merges A `2521c1bafe435deec9c091cdf6a9ed68f247132c` and B `3c4ddcee41e508693f495e50a9123def322650fc`. Signed merge `d04dcba0d8dca537e3f46c4e122f7b960c1701be` has those exact parents; no conflicts. Integration additions are in `tools/hybrid-bench/`; no `src/` edits, pushes, PR or landing. The specified feature bases override default-main branching. No task-board is configured (`task-board goal get`: exit 1, missing `.task-board`); `.temp/tasks.md` is the existing task record.

Signed implementation commits: `817aad9ab5` (depth ladder, long CPU timeouts, lower RAM cap and release metrics), `67c27790dc` (offline launchers), `a6847676b0` (real native budget/replay regressions), `b94b612670` (separate fixed wire caps from the node context default), `d3eca794c9` (direct no-flag native defaults), `675ba551dc` (keep uncapped derived-budget coverage alongside operator caps). All signatures verified locally. B's incoming commits are preserved; the integration deltas are listed above. A's node source and the merged native numerical checker remain unchanged after the merge.

The benchmark now accepts explicit `--depths` and `--request-timeout`; the launcher accepts a lower `--ram-budget-mib` and `--request-timeout`. Default ladders, HTTP timeouts, node 16384/512 MiB/1 MiB defaults and 8448-token no-flag allocation stay unchanged. `--offline --model LOCAL` disables model downloads and CMake UI build/retrieval. Release traces record actual node store counters after DELETE. The planner no longer labels the old node default as an active protocol context cap.

Self-review uses `skills/code-review/SKILL.md`: Scope, Security, Approach, General and Tests. The scope checker preserves all previous test function names (82 from A, 88 from B). No surface table was supplied; that is a brief gap. No unlisted out-of-contract acceptance row or external landing gate is claimed. Independent review remains pending.

## Runtime identity and launcher contract

The native launcher used `--ctx 65536 --ram-budget-mib 560 --bench-link`. The actual `/proc/PID/cmdline` contains `--max-ctx 65536 --ram-budget-mib 560 --max-request-mib 1`. `node-info.json.node_limits`, live health and Prometheus agree on 65536 / 587202560 / 1048576 bytes. Layout fingerprint from the loaded node and B's independent GGUF reader: `33d9f73396507b1f8402bf1e439936f42caee2c8875fb46e127b06e0e656d2fa`. No health bridge/stub was used.

| Model | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3.5-0.8B-Q4_0.gguf` | 507154688 | `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c` |
| `Qwen3.5-0.8B-Q8_0-requant.gguf` | 811843840 | `f543fad263339779c3a38a21a531c6adbffcfdecdf5f749dd542de924b1e7302` |
| `Qwen3.5-0.8B-Q8_0-r16.gguf` | 811843840 | `f543fad263339779c3a38a21a531c6adbffcfdecdf5f749dd542de924b1e7302` |

The Q8 files pre-existed and have the same hash. They are requantizations of Q4 weights, not recovered source precision. No other base-model GGUF exists in the inspected non-scratch home-directory GGUF inventory, so the renamed-different-base-model case is unrun. A corrupted fingerprint supplied with real native node metadata refuses with hetero opt-in before state/process creation; that is a metadata negative, not a substitute for the unavailable second-base-model test.

## Numerical results

| Run/check | Depth | Deltas | Same partition | Mean KL nats/token | Whole top-1 | Free divergence | Status / exit |
|---|---:|---:|---:|---:|---:|---:|---|
| integrated-short / fresh | 132 | 3 | 64/64 | 0.00174820973243 | 62/64 | 7 | passed / 0 |
| integrated-short / append_chain | 132 | 3 | 64/64 | 0.00174820973243 | 62/64 | 7 | passed / 0 |
| integrated-hetero / fresh | 132 | 3 | 62/64 | 0.0145353030525 | 62/64 | 7 | uncalibrated_outside_reference / 5 |
| integrated-hetero / append_chain | 132 | 3 | 62/64 | 0.0145353030525 | 62/64 | 7 | uncalibrated_outside_reference / 5 |
| integrated-40k / fresh | 40038 | 5 | 64/64 | 4.02083065588e-05 | 64/64 | -1 | passed / 0 |
| integrated-40k / append_chain | 40038 | 15 | 64/64 | 0.000117652473519 | 64/64 | -1 | passed / 0 |

The append check replays the measured run's successful critical endpoints and intervening shadow spans into an isolated real-node session. It checks the deltas returned by that replay, rather than capturing the measured mode session's native bytes.

Section 11 reference bounds are mean KL <=0.01 and top-1 >=95%; homogeneous same-backend same-partition requires 64/64. The Q4-node/Q8-client run is accepted for explicit layout compatibility, but its KL is outside the homogeneous reference. Checker exits 3 are preserved; the benchmark deliberately returns 5 (`uncalibrated_outside_reference` per check), with no claim of heterogeneous accuracy acceptance. The run summary uses the broader `uncalibrated` status.

## Real negative controls and regressions

- `test_eviction_replay`: global LRU returns 404 for the proxy's old session; actual proxy replay restores 25 exact token IDs, verified in the native snapshot as well as the journal/hash. Decoder slot info remains unchanged. Two evictions observed, including the replay's replacement.
- `test_budget_benchmark`: native one-token append returns `503 {"error":"ram budget","need":20208656,"budget":1048576}` without committing the append. A real forced-offload benchmark against that node exits 2, reports `ram budget refusal (HTTP 503)`, and releases to zero sessions/bytes.
- Native heterogeneous admission: Q4 node/Q8 client setup refuses without opt-in (2), accepts with it (0), and refuses corrupted fingerprint metadata with opt-in (2). Live native fingerprints are used throughout.
- Budget/LRU/pinning/request/CLI tests preserve 503 fields, impossible-admission atomicity, default limits, inclusive body caps and over-context rejection. Existing malformed native/codec paths preserve both destination and neighbor bytes. The checker retains known-changed-distribution KL controls, including changed probabilities with the same top-1.
- The benchmark's native telemetry plant exits 2 for named full re-prefill and killed-node fallback after valid preceding calls; neither aborted bundle is passing numerical evidence.

The first new native integration attempts exited 1: rounds 01/02 raced non-streaming response delivery before durable commit and replay acknowledgement; round 03 reached the positive replay check but had a missing fixture `decode_token`, producing a KeyError bundle. Round 04 fixed the fixture and passed. Round 05 repeated against the final metadata-only helper change and passed. These harness failures are retained as failures. Initial control-client setup exited 2 on a recently stopped port; `/proc/net/tcp` showed TIME_WAIT (06), and fresh ports succeeded. The regression driver initially exited 1 at `client-prepare-01` because its output directory did not exist; the queued follow-up driver correctly stopped without running its gates. Creating that directory and resuming from the failed command passed, without replaying the already green commands. The first background monitor never produced samples; the foreground replacement is the actual monitor evidence.

## Build and process custody

All builds are incremental CPU builds, `-j2`, with the shared `~/Werkstatt/hybrid-models/.build.lock`. CMake comes from the pre-existing `.temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/`; no installation. Launchers record exact SHA, flags, binary/library hashes, build PID and minimum free bytes. The 25 GiB floor is 26843545600 bytes. Long-run native engines started at clean `817aad9ab5`; later offline builds include clean `a6847676b0` and `b94b612670`. Subsequent source changes only affect offline configuration, tests and a returned sizing metadata field; native prefill, codec, proxy state computation, checker and partition settings are unchanged. The long-run bundle retains the earlier descriptive `sizing.protocol_limit=16384` field; the actual operational node limits are 65536. Final source renames that obsolete descriptor and the fresh native launcher check verifies it. Historical bundles are not edited. Thread/batch settings are 2/2/256/256. Concurrent correctness work on the shared host means these timings are observations, not performance qualification.

No-download deviation: the initial manual server build inherited CMake's prebuilt-UI behavior and downloaded the UI archive plus checksum data (`build-01.log`). The first client launcher made another checksum retrieval and reused that archive (`launch-client-01.log`). My first progress note understated the event as checksums only; the full log audit corrected it. No model weights were downloaded. This violated the requested no-download boundary. The archive size/hash is recorded below; it is a generated build-cache asset outside task scratch. Explicit offline flags now disable both UI routes, and subsequent offline builds show no retrieval. This run must not be described as having had zero downloads.

`ty --version` and `editorconfig-checker --version` both exit 127; full external type/EditorConfig checks are unrun. No installation attempted. Python syntax and whitespace/scope checks are separate evidence. No GPU/Metal, 27B, 128k/256k inference, target-machine checker allocation, 50-prompt section-11 matrix, production throughput, external review or publication is claimed.


## Independent native distribution regressions

| Fixture depth | Same partition | Mean KL nats/token | Whole top-1 | Free divergence | Exit |
|---:|---:|---:|---:|---:|---:|
| 512 | 64/64 | 3.2963e-16 | 64/64 | -1 | 0 |
| 2048 | 64/64 | 7.69261e-07 | 64/64 | -1 | 0 |
| 8192 | 64/64 | 2.01502e-07 | 64/64 | -1 | 0 |
| 40000 | 64/64 | 2.32432e-15 | 64/64 | -1 | 0 |

Downloaded UI build-cache archive: 3242579 bytes, SHA-256 `124fdceb817e4c6cb5b4c3f31d643cd46959f0228646a87a792a6c260de356bb`. It is outside task scratch in `build/tools/ui/dist.tar.gz`.

## Acceptance claims and observed scope

| Gate | Specific claim | Outcome |
|---|---|---|
| `launcher-final-01` | Native argv, node-info, live health and Prometheus expose the same three operator limits and actual loaded fingerprint; download endpoint requires opt-in. | 0 |
| `hetero-setup-01`, `hetero-default-refusal-01`, `fingerprint-refusal-01` | Real Q4/Q8 identity is admitted only with explicit hetero opt-in; mismatched layout metadata refuses even with opt-in. | 0 / expected 2 / expected 2 |
| `long-benchmark-01`, `long-audit-01` | One invocation executes every scheduled call in all three modes, deletes each mode session, imports contiguous nonzero append chunks and checks fresh plus recorded append partitions near 40k. | 0 / 0 |
| `integrated-negatives-06` | Native LRU replay restores 25 actual snapshot token IDs, native budget 503 retains exact fields and benchmark classification, no-flag node defaults remain unchanged. | 0 |
| `limits-01`, five `http-*` gates, `header-http-01`, `deep-http-01` | Budget atomicity/pinning, inclusive HTTP request and context bounds, and malformed native HTTP requests preserve state; real 40k HTTP import works. | all 0 |
| `codec-*`, `recurrent-01`, `native-01`, `append-01`, `guards-01`, `distributions-01`, `deep-distributions-01` | Codec/native append boundaries and malformed rollback, 64/64 partition agreement and stated KL bounds at the exercised depths. | all 0 |
| `client-*`, `remote-append-01`, `append-api-01`, `concurrency-native-01`, `decode-proxy-01` | Native client capacity, replay/fork/chunked append, slot rollback and concurrent independent sessions preserve their existing contracts. | all corrected/final commands 0 |
| Four `conversation-*` and two `typed-*` gates | Thinking on/off/inferred/omitted and no-checkpoint compatibility, strict typed identity, both rejection directions. | all 0 |
| `proxy-unit-01`, `bench-unit-06`, `source-scope-02` | 67 proxy tests and 21 benchmark tests pass; all old A/B test names are retained and edits stay in the allowed tool directories. | 0 / 0 / 0 |

The renamed different-base-model case is UNRUN because no such local weights exist. The metadata-fingerprint rejection above does not establish that case. The two populated-session bound below counts retained serialized state plus token IDs, not model weights, transient allocations or process RSS; it is not a two-session cap for empty or short sessions.

## One-run 40k measurements

| Mode | Calls | Maximum prompt tokens | Maximum chunks | Mean TTFT seconds | Sum request seconds | Mean decode tokens/second | Mode-end sessions / bytes |
|---|---:|---:|---:|---:|---:|---:|---|
| local | 8 | 40198 | 0 | 185.174858 | 1483.412540 | 40.015489 | 0 / 0 |
| hybrid | 8 | 40198 | 5 | 188.274235 | 1509.119357 | 34.710421 | 0 / 0 |
| offload | 8 | 40198 | 5 | 190.981771 | 1530.118162 | 26.734943 | 0 / 0 |

No skipped/aborted calls; summary `passed`, exit 0. Observed retained peak 565402124 bytes, maximum 2 sessions, evictions 0. Two measured ~40k states fit the 587202560-byte cap; three states of the independently observed 282435544-byte size require 847306632 bytes and do not fit.

The measured critical append uses 5 contiguous chunks, 344 -> 8536 -> 16728 -> 24920 -> 33112 -> 39989, for a 39990-token prompt with one held decoder token. The check triggers on the following 40038-token prompt. Fresh replay uses 5 deltas; append replay uses 15 deltas, including intervening successful shadow prefixes. Both checks use K=64, max_ctx=65536 and the original B numerical checker. The table above preserves exact numerical outcomes. The bounded smoke schedule has eight calls per mode and short outputs; it is not the full performance or 50-prompt accuracy campaign.

## Audit controls

| Record plant | Property narrowed | Named failing assertion | Result / surviving bound |
|---|---|---|---|
| `false_exactness` | A green same-partition verdict requires 64 actual agreements. | `validate_results.validate:same_partition_64` | Killed; no claim about unexercised native mutations. |
| `hidden_kl_failure` | KL 0.02 cannot be hidden by a green status string. | `validate_results.validate:kl_limit` | Killed. |
| `leaked_session` | Every mode release must report zero sessions and bytes. | `validate_results.validate:mode_cleanup` | Killed. |

These are copied-record plants, not source mutants; the real JSONL/archive bytes were not edited. The native distribution checker also retains independent known-violation controls. `premature-close-control-01` exited 2 as expected before any signals because the long benchmark had no successful result. The later ready check and actual runtime cleanup run only after all required numerical evidence exists.

## Build identities and artifact integrity

| Launcher state | Source commit | Minimum free bytes during build | Relevant binaries (SHA-256) |
|---|---|---:|---|
| node-q4 | `817aad9ab5ca3f564e6b7c3241c2f781a21e2f58` | 40235720704 | llama-hybrid-prefill: `6e4cf8a2c9c4dbacded0e56987c30c6f5fa4f8d182ef4ba84ffb374a443ed037` |
| client-q4 | `817aad9ab5ca3f564e6b7c3241c2f781a21e2f58` | 40227024896 | llama-server: `53ccca048e622043c26bfa8e792b94c58ef1b9dc59bea96482862a1c8ade30d4`; llama-hybrid-bench-check: `294293b5d612260ad626666decddc45f80cf1d0a9b0ec34ad93c5a13cf94f033` |
| node-final | `b94b61267021ba0bce119f16e61b3f95708d4408` | 36647452672 | llama-hybrid-prefill: `6e4cf8a2c9c4dbacded0e56987c30c6f5fa4f8d182ef4ba84ffb374a443ed037` |

Environment: Linux x86_64 CPU; Python 3.14.4, C++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0, CMake 4.4.4, flock/util-linux 2.41.3. No GPU backend was used.

Initial manual build admission: 39910936576 available bytes; runtime monitor minimum: 36179546112 bytes. All recorded samples exceed 26843545600 bytes. This is sampled observation plus per-build guards, not continuous filesystem reservation. Embedded build-version strings can change binary hashes after a commit even when numerical sources are unchanged.

Unchanged Git tree/blob identities used to bind reused numerical evidence:

- `src`: `2e3ba9a1624f588035787d7ed6ba76d39ba99ad1`.
- `ggml`: `eba81271000a84fdbbf155a1c447a205d95daf03`.
- `tools/server`: `8accbdd3184197a4220a87add3cdfda5710de0fc`.
- `tools/hybrid`: `36ffe1519cc74d14363f363fe7fca6e64777561f`.
- `tools/hybrid-bench/check.cpp`: `74a73b598473263342282d796283a1a9c8ae1e14`.
- `tools/hybrid-proxy/proxy.py`: `804f154534864e25e5f2b2775fd16ebf7ffbfae7`.
- `tools/hybrid-bench/hybrid_bench.py`: `a3f678b20b292a4ac7273361a0ea9b96f55049e6`.

Final archive audit checks every internal SHA-256 and sidecar digest, safe file members and credential exclusion. Archives and raw logs are task scratch and are deleted at closing as requested; the following hashes identify their observed pre-deletion bytes.

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `hybrid-bench-integrated-40k-20261101T004919691425Z.tar.gz` | 6858 | `6264133d8739a61be8217efc55618c5d53e6c60177d015a88aaafd75b03a0a27` |
| `hybrid-bench-integrated-hetero-20261101T005501439841Z.tar.gz` | 6677 | `4cbbd7f667d414b9a736ca4916e6298b43d2554a2945180f39e4224457891dd0` |
| `hybrid-bench-killed-mid-run-20261101T005814108927Z.tar.gz` | 3485 | `aef3af4a5f52966636a28b0b9ebc05d013e0b08a6a8d774e8e37822f72797995` |
| `hybrid-bench-reprefill-control-20261101T005811138673Z.tar.gz` | 3493 | `eaa0928a58a8bc967c4ad2c3e1b0ae16dc54a552ee4f155782afc8c52b2d388b` |
| `hybrid-bench-integrated-short-20261101T005701127275Z.tar.gz` | 6568 | `527c823a33829622d392be70afbfec484a31707ab4ccc76f41b8fb7b61042fb1` |
| `native-negatives-06/budget-proxy/hybrid-bench-native-budget-20261101T012055701337Z.tar.gz` | 1397 | `0dddbf8e9dc2764b6e14b5ecc31a320f90d30a127a953d33592d5900a315f776` |
| `native-negatives-03/budget-proxy/hybrid-bench-native-budget-20261101T005321671709Z.tar.gz` | 1383 | `d3d09bc3a0657c58fef474c5b9ced7aba6b17d68f3a46e10b1b6e008ec9b2e4b` |
| `native-negatives-04/budget-proxy/hybrid-bench-native-budget-20261101T005338129122Z.tar.gz` | 1397 | `9a6b4ef078d3848de4619fb99c525a2fb43c10f000fa9de5500bc69e69d62f0c` |
| `native-negatives-05/budget-proxy/hybrid-bench-native-budget-20261101T005931067950Z.tar.gz` | 1396 | `bd80a296d78d5a2440cf09adb0f02bd29fb5b7680d05fbfb7a4eb6febdf0032f` |

## Runtime closure

At 2026-11-01T04:48:23.896716+00:00, the final live node health was `{"evictions": 0, "max_ctx": 65536, "max_request_bytes": 1048576, "ram_budget_bytes": 587202560, "ram_bytes": 0, "sessions": 0}`. Only recorded owned PIDs were stopped. All 148 recorded PIDs were inactive after cleanup; no active task process remained. Shared model files and the incremental build cache were preserved.

## Recorded commands and exits

`RUN=.temp/hybrid-a-r11-261101`, `MODEL_CACHE=/home/agent/Werkstatt/hybrid-models`. Each gate ran as its own process, without tee. Each test's claim is stated beside its function or in the evidence table above. Full expanded commands below contain paths to private credentials, never credential values.

- `bench-unit-01`: exit 0, 2026-11-01T00:47:32.676053+00:00 to 2026-11-01T00:47:32.793075+00:00.

```sh
python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v
```

- `bench-unit-02`: exit 0, 2026-11-01T00:47:57.332490+00:00 to 2026-11-01T00:47:57.436922+00:00.

```sh
python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v
```

- `bench-unit-03`: exit 0, 2026-11-01T00:48:20.617314+00:00 to 2026-11-01T00:48:20.732122+00:00.

```sh
python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v
```

- `launch-node-01`: exit 0, 2026-11-01T00:48:27.237225+00:00 to 2026-11-01T00:48:33.144159+00:00.

```sh
tools/hybrid-bench/deploy-prefill-node.sh --state .temp/hybrid-a-r11-261101/node-q4 --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --ctx 65536 --ram-budget-mib 560 --port 18191 --bench-link --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock
```

- `launch-client-01`: exit 0, 2026-11-01T00:48:50.678012+00:00 to 2026-11-01T00:48:54.883175+00:00.

```sh
tools/hybrid-bench/setup-client.sh --state .temp/hybrid-a-r11-261101/client-q4 --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --ctx 65536 --request-timeout 7200 --extend-chunk-tokens 8192 --prefill-url http://127.0.0.1:18191 --port 18192 --decode-port 18190 --node-info .temp/hybrid-a-r11-261101/node-q4/node-info.json --token-file .temp/hybrid-a-r11-261101/node-q4/client-tokens.json --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock
```

- `proxy-unit-01`: exit 0, 2026-11-01T00:48:28.400905+00:00 to 2026-11-01T00:49:53.386441+00:00.

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v
```

- `bench-unit-04`: exit 0, 2026-11-01T00:49:58.893255+00:00 to 2026-11-01T00:49:59.016897+00:00.

```sh
python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v
```

- `limits-01`: exit 0, 2026-11-01T00:50:19.886799+00:00 to 2026-11-01T00:51:51.112273+00:00.

```sh
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/limits
```

- `identity-01`: exit 0, 2026-11-01T00:51:51.138113+00:00 to 2026-11-01T00:51:55.715697+00:00.

```sh
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --requantized /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/identity --case identity
```

- `codec-core-01`: exit 0, 2026-11-01T00:51:55.747382+00:00 to 2026-11-01T00:51:55.760697+00:00.

```sh
build/bin/llama-hybrid-check --codec
```

- `codec-16384-01`: exit 0, 2026-11-01T00:51:55.789990+00:00 to 2026-11-01T00:51:55.802473+00:00.

```sh
build/bin/llama-hybrid-check-codec
```

- `codec-65536-01`: exit 0, 2026-11-01T00:51:55.829167+00:00 to 2026-11-01T00:51:55.868538+00:00.

```sh
build/bin/llama-hybrid-check-codec --max-ctx 65536
```

- `codec-262144-01`: exit 0, 2026-11-01T00:51:55.894248+00:00 to 2026-11-01T00:51:55.941017+00:00.

```sh
build/bin/llama-hybrid-check-codec --max-ctx 262144
```

- `http-default-01`: exit 0, 2026-11-01T00:51:55.966728+00:00 to 2026-11-01T00:51:59.641864+00:00.

```sh
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/http-default --port 18193
```

- `http-slow-readers-01`: exit 0, 2026-11-01T00:51:59.667943+00:00 to 2026-11-01T00:52:02.089458+00:00.

```sh
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/http-slow-readers --port 18193 --regression slow-readers
```

- `integrated-negatives-01`: exit 1, 2026-11-01T00:52:02.709847+00:00 to 2026-11-01T00:52:08.476217+00:00.

```sh
python3 tools/hybrid-bench/test_integrated.py --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r11-261101/native-negatives
```

- `integrated-negatives-02`: exit 1, 2026-11-01T00:52:25.769680+00:00 to 2026-11-01T00:52:32.138579+00:00.

```sh
python3 tools/hybrid-bench/test_integrated.py --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r11-261101/native-negatives-02
```

- `http-pending-01`: exit 0, 2026-11-01T00:52:02.115865+00:00 to 2026-11-01T00:53:12.891467+00:00.

```sh
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/http-pending --port 18193 --regression pending
```

- `integrated-negatives-03`: exit 1, 2026-11-01T00:53:12.345305+00:00 to 2026-11-01T00:53:22.420190+00:00.

```sh
python3 tools/hybrid-bench/test_integrated.py --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r11-261101/native-negatives-03
```

- `http-errors-01`: exit 0, 2026-11-01T00:53:12.922698+00:00 to 2026-11-01T00:53:30.120362+00:00.

```sh
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/http-errors --port 18193 --regression errors
```

- `integrated-negatives-04`: exit 0, 2026-11-01T00:53:28.557389+00:00 to 2026-11-01T00:53:40.136023+00:00.

```sh
python3 tools/hybrid-bench/test_integrated.py --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r11-261101/native-negatives-04
```

- `launch-hetero-node-01`: exit 0, 2026-11-01T00:54:06.047778+00:00 to 2026-11-01T00:54:14.214039+00:00.

```sh
tools/hybrid-bench/deploy-prefill-node.sh --offline --state .temp/hybrid-a-r11-261101/hetero-node --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --ctx 2048 --ram-budget-mib 48 --port 18196 --bench-link --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock
```

- `hetero-default-refusal-01`: exit 2, 2026-11-01T00:54:51.767404+00:00 to 2026-11-01T00:54:53.196475+00:00.

```sh
tools/hybrid-bench/setup-client.sh --offline --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --ctx 2048 --extend-chunk-tokens 64 --prefill-url http://127.0.0.1:18196 --port 18198 --decode-port 18197 --token-file /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/hetero-node/client-tokens.json --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock --state /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/hetero-refused --node-info /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/hetero-node/node-info.json
```

- `fingerprint-refusal-01`: exit 2, 2026-11-01T00:54:53.245770+00:00 to 2026-11-01T00:54:54.611498+00:00.

```sh
tools/hybrid-bench/setup-client.sh --offline --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --ctx 2048 --extend-chunk-tokens 64 --prefill-url http://127.0.0.1:18196 --port 18198 --decode-port 18197 --token-file /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/hetero-node/client-tokens.json --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock --hetero-quant --state /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/fingerprint-refused --node-info /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/wrong-fingerprint.json
```

- `hetero-setup-01`: exit 0, 2026-11-01T00:54:54.646767+00:00 to 2026-11-01T00:55:01.306652+00:00.

```sh
tools/hybrid-bench/setup-client.sh --offline --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --ctx 2048 --extend-chunk-tokens 64 --prefill-url http://127.0.0.1:18196 --port 18198 --decode-port 18197 --token-file /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/hetero-node/client-tokens.json --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock --hetero-quant --state /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/hetero-client --node-info /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/hetero-node/node-info.json
```

- `hetero-benchmark-01`: exit 5, 2026-11-01T00:55:01.345925+00:00 to 2026-11-01T00:56:02.499254+00:00.

```sh
python3 tools/hybrid-bench/hybrid_bench.py --client-state /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/hetero-client --scratch /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101 --output /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101 --label integrated-hetero --smoke --check
```

- `hetero-driver-01`: exit 0, 2026-11-01T00:54:51.668295+00:00 to 2026-11-01T00:56:02.847211+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/hetero.py
```

- `launch-control-client-01`: exit 2, 2026-11-01T00:56:44.918222+00:00 to 2026-11-01T00:56:45.060992+00:00.

```sh
tools/hybrid-bench/setup-client.sh --offline --skip-build --state .temp/hybrid-a-r11-261101/control-client --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --ctx 2048 --extend-chunk-tokens 64 --prefill-url http://127.0.0.1:18196 --port 18198 --decode-port 18197 --node-info .temp/hybrid-a-r11-261101/hetero-node/node-info.json --token-file .temp/hybrid-a-r11-261101/hetero-node/client-tokens.json --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock
```

- `launch-control-client-02`: exit 0, 2026-11-01T00:56:56.669441+00:00 to 2026-11-01T00:57:00.813126+00:00.

```sh
tools/hybrid-bench/setup-client.sh --offline --skip-build --state .temp/hybrid-a-r11-261101/control-client --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --ctx 2048 --extend-chunk-tokens 64 --prefill-url http://127.0.0.1:18196 --port 18200 --decode-port 18201 --node-info .temp/hybrid-a-r11-261101/hetero-node/node-info.json --token-file .temp/hybrid-a-r11-261101/hetero-node/client-tokens.json --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock
```

- `bench-unit-05`: exit 0, 2026-11-01T00:57:29.510983+00:00 to 2026-11-01T00:57:29.865504+00:00.

```sh
python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v
```

- `source-scope-01`: exit 0, 2026-11-01T00:57:44.016370+00:00 to 2026-11-01T00:57:44.859263+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/check_source.py
```

- `homogeneous-smoke-01`: exit 0, 2026-11-01T00:57:01.019290+00:00 to 2026-11-01T00:57:48.061262+00:00.

```sh
python3 tools/hybrid-bench/hybrid_bench.py --client-state .temp/hybrid-a-r11-261101/control-client --scratch .temp/hybrid-a-r11-261101 --output .temp/hybrid-a-r11-261101 --label integrated-short --smoke --check
```

- `benchmark-native-controls-01`: exit 0, 2026-11-01T00:58:03.461111+00:00 to 2026-11-01T00:58:16.951967+00:00.

```sh
python3 tools/hybrid-bench/test_native.py --client-state .temp/hybrid-a-r11-261101/control-client --node-state .temp/hybrid-a-r11-261101/hetero-node --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r11-261101/r15-controls --output .temp/hybrid-a-r11-261101
```

- `http-max-delta-01`: exit 0, 2026-11-01T00:53:30.160226+00:00 to 2026-11-01T00:58:22.505360+00:00.

```sh
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/http-max-delta --port 18193 --regression max-delta
```

- `recurrent-01`: exit 0, 2026-11-01T00:58:22.533542+00:00 to 2026-11-01T00:58:23.540497+00:00.

```sh
python3 tools/hybrid/test_recurrent_append.py --library build/bin/libllama.so --malformed
```

- `bundle-audit-01`: exit 0, 2026-11-01T00:59:01.248582+00:00 to 2026-11-01T00:59:01.304194+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/audit.py
```

- `integrated-negatives-05`: exit 0, 2026-11-01T00:59:21.724241+00:00 to 2026-11-01T00:59:33.056426+00:00.

```sh
python3 tools/hybrid-bench/test_integrated.py --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r11-261101/native-negatives-05
```

- `native-01`: exit 0, 2026-11-01T00:58:23.586803+00:00 to 2026-11-01T01:01:15.888232+00:00.

```sh
build/bin/llama-hybrid-check /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf -
```

- `append-01`: exit 0, 2026-11-01T01:01:15.920601+00:00 to 2026-11-01T01:02:48.787485+00:00.

```sh
build/bin/llama-hybrid-check-append /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden/prompt-512 /home/agent/Werkstatt/hybrid-models/golden/prompt-2048 /home/agent/Werkstatt/hybrid-models/golden/prompt-8192
```

- `guards-01`: exit 0, 2026-11-01T01:02:48.819544+00:00 to 2026-11-01T01:02:50.544941+00:00.

```sh
build/bin/llama-hybrid-check-guards /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden/prompt-512
```

- `distributions-01`: exit 0, 2026-11-01T01:02:50.577292+00:00 to 2026-11-01T01:05:53.132676+00:00.

```sh
build/bin/llama-hybrid-check-distributions /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden/prompt-512.tokens.json /home/agent/Werkstatt/hybrid-models/golden/prompt-2048.tokens.json /home/agent/Werkstatt/hybrid-models/golden/prompt-8192.tokens.json
```

- `client-prepare-01`: exit 1, 2026-11-01T01:05:53.174634+00:00 to 2026-11-01T01:05:53.858648+00:00.

```sh
build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --prepare /home/agent/Werkstatt/hybrid-models/golden/ /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/client-fixtures
```

- `regression-driver-01`: exit 1, 2026-11-01T00:50:19.846061+00:00 to 2026-11-01T01:05:53.871612+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/regressions.py
```

- `more-regression-driver-01`: exit 1, 2026-11-01T01:00:13.480760+00:00 to 2026-11-01T01:05:58.689481+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/more_regressions.py
```

- `client-prepare-02`: exit 0, 2026-11-01T01:06:50.954531+00:00 to 2026-11-01T01:06:56.726450+00:00.

```sh
build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --prepare /home/agent/Werkstatt/hybrid-models/golden/ /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/client-fixtures
```

- `client-512-01`: exit 0, 2026-11-01T01:06:56.771608+00:00 to 2026-11-01T01:06:57.597642+00:00.

```sh
build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/client-fixtures/cost-512 512
```

- `client-2048-01`: exit 0, 2026-11-01T01:06:57.654994+00:00 to 2026-11-01T01:06:58.481548+00:00.

```sh
build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/client-fixtures/cost-2048 2048
```

- `client-8192-01`: exit 0, 2026-11-01T01:06:58.516883+00:00 to 2026-11-01T01:06:59.486771+00:00.

```sh
build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/client-fixtures/cost-8192 8192
```

- `client-capacity-01`: exit 0, 2026-11-01T01:06:59.545558+00:00 to 2026-11-01T01:07:00.350728+00:00.

```sh
build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --capacity /home/agent/Werkstatt/hybrid-models/golden/prompt-512
```

- `remote-append-01`: exit 0, 2026-11-01T01:07:00.410036+00:00 to 2026-11-01T01:07:19.317173+00:00.

```sh
python3 tools/hybrid-proxy/test_remote_append.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/remote-append
```

- `append-api-01`: exit 0, 2026-11-01T01:07:19.355854+00:00 to 2026-11-01T01:08:01.209667+00:00.

```sh
python3 tools/hybrid-proxy/test_append_api.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/append-api
```

- `concurrency-native-01`: exit 0, 2026-11-01T01:08:01.241227+00:00 to 2026-11-01T01:08:19.815203+00:00.

```sh
python3 tools/hybrid-proxy/test_concurrency_native.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/concurrency-native
```

- `decode-proxy-01`: exit 0, 2026-11-01T01:08:19.845003+00:00 to 2026-11-01T01:09:55.013554+00:00.

```sh
python3 tools/hybrid-proxy/test_decode.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --golden /home/agent/Werkstatt/hybrid-models/golden/prompt-512 --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/decode-proxy
```

- `regression-driver-02`: exit 0, 2026-11-01T01:06:50.906217+00:00 to 2026-11-01T01:09:55.029390+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/regressions.py
```

- `conversation-thinking-on-01`: exit 0, 2026-11-01T01:09:57.179012+00:00 to 2026-11-01T01:10:38.023806+00:00.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/conversation-thinking-on
```

- `prune-closed-01`: exit 0, 2026-11-01T01:10:56.422274+00:00 to 2026-11-01T01:10:58.388885+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/prune_closed.py
```

- `conversation-thinking-off-01`: exit 0, 2026-11-01T01:10:38.055458+00:00 to 2026-11-01T01:11:18.122464+00:00.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/conversation-thinking-off --thinking off
```

- `conversation-inferred-omit-01`: exit 0, 2026-11-01T01:11:18.161872+00:00 to 2026-11-01T01:11:59.067229+00:00.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/conversation-inferred-omit --inferred --omit-reasoning
```

- `conversation-no-checkpoint-01`: exit 0, 2026-11-01T01:11:59.121831+00:00 to 2026-11-01T01:12:08.877285+00:00.

```sh
python3 tools/hybrid-proxy/test_append_sessions.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/conversation-no-checkpoint --no-checkpoint-control
```

- `typed-identity-01`: exit 0, 2026-11-01T01:12:08.907174+00:00 to 2026-11-01T01:12:36.947199+00:00.

```sh
python3 tools/hybrid-proxy/test_message_identity.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/typed-identity
```

- `typed-no-checkpoint-01`: exit 0, 2026-11-01T01:12:36.981068+00:00 to 2026-11-01T01:12:54.609458+00:00.

```sh
python3 tools/hybrid-proxy/test_message_identity.py --binary build/bin/llama-server --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/typed-no-checkpoint --no-checkpoint-control
```

- `header-http-01`: exit 0, 2026-11-01T01:12:54.641092+00:00 to 2026-11-01T01:16:57.834590+00:00.

```sh
python3 tools/hybrid-proxy/test_header_budget.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/header-http
```

- `launch-final-node-01`: exit 0, 2026-11-01T01:16:51.121170+00:00 to 2026-11-01T01:16:59.195202+00:00.

```sh
tools/hybrid-bench/deploy-prefill-node.sh --offline --state .temp/hybrid-a-r11-261101/node-final --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --ctx 65536 --ram-budget-mib 560 --port 18210 --build-lock /home/agent/Werkstatt/hybrid-models/.build.lock
```

- `launcher-final-01`: exit 0, 2026-11-01T01:17:36.837742+00:00 to 2026-11-01T01:17:36.924249+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/check_launcher.py
```

- `integrated-negatives-06`: exit 0, 2026-11-01T01:20:44.330041+00:00 to 2026-11-01T01:20:57.800150+00:00.

```sh
python3 tools/hybrid-bench/test_integrated.py --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r11-261101/native-negatives-06
```

- `evidence-snapshot-01`: exit 0, 2026-11-01T01:25:54.255139+00:00 to 2026-11-01T01:25:54.427663+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/snapshot_evidence.py
```

- `prune-closed-02`: exit 0, 2026-11-01T01:30:58.442668+00:00 to 2026-11-01T01:31:00.733784+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/prune_closed.py
```

- `deep-http-01`: exit 0, 2026-11-01T01:16:57.860526+00:00 to 2026-11-01T01:40:45.414506+00:00.

```sh
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/deep --case deep
```

- `more-regression-driver-02`: exit 0, 2026-11-01T01:06:52.068011+00:00 to 2026-11-01T01:40:45.427069+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/more_regressions.py
```

- `bench-unit-06`: exit 0, 2026-11-01T02:16:09.009671+00:00 to 2026-11-01T02:16:09.336438+00:00.

```sh
python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v
```

- `audit-record-controls-01`: exit 0, 2026-11-01T02:24:46.072226+00:00 to 2026-11-01T02:24:46.127028+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/validate_results.py --controls
```

- `source-scope-02`: exit 0, 2026-11-01T02:26:57.648704+00:00 to 2026-11-01T02:26:58.406766+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/check_source.py
```

- `deep-distributions-01`: exit 0, 2026-11-01T01:40:49.242438+00:00 to 2026-11-01T02:29:16.665248+00:00.

```sh
build/bin/llama-hybrid-check-distributions /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r11-261101/deep/deep.tokens.json --max-ctx 65536
```

- `deep-distribution-driver-01`: exit 0, 2026-11-01T01:40:04.168443+00:00 to 2026-11-01T02:29:16.679015+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/deep_distribution.py
```

- `premature-close-control-01`: exit 2, 2026-11-01T04:15:14.511532+00:00 to 2026-11-01T04:15:14.605009+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/close_runtime.py --check-ready
```

- `models-final-01`: exit 0, 2026-11-01T04:41:44.339706+00:00 to 2026-11-01T04:41:46.595612+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/check_models.py
```

- `long-benchmark-01`: exit 0, 2026-11-01T00:49:19.644909+00:00 to 2026-11-01T04:47:58.178240+00:00.

```sh
python3 tools/hybrid-bench/hybrid_bench.py --client-state .temp/hybrid-a-r11-261101/client-q4 --scratch .temp/hybrid-a-r11-261101 --output .temp/hybrid-a-r11-261101 --label integrated-40k --smoke --depths 128 40000 --request-timeout 7200 --check --check-depths 40000
```

- `long-audit-01`: exit 0, 2026-11-01T04:48:10.944644+00:00 to 2026-11-01T04:48:10.963566+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/validate_results.py
```

- `bundle-audit-02`: exit 0, 2026-11-01T04:48:21.483592+00:00 to 2026-11-01T04:48:21.538275+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/audit.py
```

- `close-ready-01`: exit 0, 2026-11-01T04:48:21.595021+00:00 to 2026-11-01T04:48:21.652485+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/close_runtime.py --check-ready
```

- `runtime-cleanup-01`: exit 0, 2026-11-01T04:48:23.523387+00:00 to 2026-11-01T04:48:23.907453+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/close_runtime.py
```

- `evidence-snapshot-02`: exit 0, 2026-11-01T04:48:26.017687+00:00 to 2026-11-01T04:48:26.080890+00:00.

```sh
python3 .temp/hybrid-a-r11-261101/snapshot_evidence.py
```


## Initial manual build

Exit 0, initial available bytes 39910936576; this is the build that fetched the UI archive. It preceded the command ledger wrapper. The later launcher commands in the ledger use the same shared lock and -j2.

```sh
flock /home/agent/Werkstatt/hybrid-models/.build.lock bash -c 'available=$(df -B1 --output=avail . | tail -n1); echo "available_bytes=$available"; test "$available" -ge 26843545600 || exit 75; exec .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-hybrid-prefill llama-hybrid-check llama-hybrid-check-codec llama-hybrid-check-append llama-hybrid-check-guards llama-hybrid-check-distributions llama-hybrid-check-client llama-hybrid-bench-check llama-server'
```

Initial `build-01.log` SHA-256: `01619da5846af12c8be0ed570e46d1e0f8ef13826d5711a765d60df08c35fa45`.

## Gate log digests

These hashes bind the observed logs to the command/exit ledger above before requested scratch deletion.

| Gate | SHA-256 |
|---|---|
| `bench-unit-01` | `76908d98d1438d11596c63730cbc5ef6fc00b09fde0e9113d0ac4bad8e4efa1b` |
| `bench-unit-02` | `43d270416666d32ba0c10807221976aaef060da32cfedbcb4715a9b4d9a1efdb` |
| `bench-unit-03` | `b738b3d25f6a3762f1cd8734288a9613d72fde62d9a2486933e17403adc90449` |
| `launch-node-01` | `0cd414a34970aaf37474f0a2b68ecd9a6ae22bf4c867e1ef9c43bbc2b8564ded` |
| `launch-client-01` | `2bb985259c9600bad32da2fed07c5093158a96e350f255e2609d8792a999fbd9` |
| `proxy-unit-01` | `bb9b0ac81da8cbbfc24e89c29082b4f4f03efcd4409daeddb5f5a6e15b17f476` |
| `bench-unit-04` | `4623d9bb6a78b70a4cfebe0042c509df8af194167d2583b86f283ccd8f7a0cb1` |
| `limits-01` | `fdb1baf5c5cbebd33a3be82adfdd8a35c8124e843b69ce5e28a06e9056abd138` |
| `identity-01` | `0482e27981c9a576b9aea4049b31b2fd0dbf5904d1aff8da65b891041d315ad3` |
| `codec-core-01` | `dd7e593d25f1c2a9085d05b5bc1f12626ad15854a50f2e2e97fb545879846624` |
| `codec-16384-01` | `82c3f9c92db9d6c9b4fdc6b2853b959f2371a6057620a259dd8d9140c6ef32b7` |
| `codec-65536-01` | `3563518124fba11eafa8fabd1ef6e0a051f49ad3e4275d7750059f848c5935f8` |
| `codec-262144-01` | `c653f4250258b0d495934f5a08808b908df0ab5039d0d273c5f19166f7f630f4` |
| `http-default-01` | `29922a5c22d322460ff794487a81ee189919836bcd2b9310b80eae5d3f4166c2` |
| `http-slow-readers-01` | `0991889190613d14a12ba8c1f3ba5c408d97bce15341033d05d8bef67d24ead9` |
| `integrated-negatives-01` | `5f77fb537a634443f987872f19f1ed27853ebfcb361afa71ce9dbd04d44cad38` |
| `integrated-negatives-02` | `d0ee09a055b05d498c3b7cd1d8c015fb98f9491014da57a6804b89b54381bd96` |
| `http-pending-01` | `73fc7e86b92612010f27f3194f775f34708695e2a97f0274bc89beeefbcd1e75` |
| `integrated-negatives-03` | `9ffc19e75100c0489d5ecf9022e35704f1803d0d88b62284c78db31f260e02d3` |
| `http-errors-01` | `1ac0e68243f9e46dc96096ac7c2e596566fd4abea0d86efc804c3c0cf920b7e2` |
| `integrated-negatives-04` | `f2a2e3eaa97091847ec08fea2b6bfb219a823e13b7f9498fa5c1a0635074d064` |
| `launch-hetero-node-01` | `071aaa7efe90ffa2760ffb9d9c4a05084154c9b46545e810fb1e26b706a7b85c` |
| `hetero-default-refusal-01` | `8022741b12a118da90640d7ac792b101f042c20cd6081109530d8793142d5e5f` |
| `fingerprint-refusal-01` | `2d1b2edcaf5910840f7f363dfcbc9dceff3297a815d5267a96b520a8c248a67d` |
| `hetero-setup-01` | `3a5a829f87f89f12f9932479220b1dddd0b25c8d5efb5855ef824991b28ab75d` |
| `hetero-benchmark-01` | `7caf37b859fe62fa78af0094190088e3f944887203706bcf6a49a49647019d74` |
| `hetero-driver-01` | `5628796495bb60a25517765165303cbb08eb7541992401b0765341f8cce6d059` |
| `launch-control-client-01` | `c5ff5be7e52c7a36286b71a2665b760e46e5d003ebf25404a490bf2959660e5e` |
| `launch-control-client-02` | `13e0f2bcec3048fbd14f0b3a897493eabedee15e47f696246438b4e4d59b36e9` |
| `bench-unit-05` | `04deaac9650e0ed2e9c56d1fb2de32d56a27c5b2a34f2c2e3886bc04639a3432` |
| `source-scope-01` | `3162728820348abf340ccbd7fe175ee06f974152c34d76e18e5d2437d23881c0` |
| `homogeneous-smoke-01` | `b442419048cc559f328f0f0063542aa67c641611df46fb6026de24aacea5970a` |
| `benchmark-native-controls-01` | `f7d31663414b905441263c25e2c1a2f766acd0012a41f12555b86ac579c38b71` |
| `http-max-delta-01` | `4a9792ca66db0bbe09e40549289eab413e10665e34dc2e8cee47e87734266236` |
| `recurrent-01` | `c83ff12d5ea1d5387d724321a5c0a7302af6d670e9ef02917720c00d946b69fd` |
| `bundle-audit-01` | `c788135d2835ae01ec90ba15177e5ee353ae8f683b688f6d7b9b9c4a3c93daea` |
| `integrated-negatives-05` | `f2a2e3eaa97091847ec08fea2b6bfb219a823e13b7f9498fa5c1a0635074d064` |
| `native-01` | `887edc5904d0589daacfb9ef22a5d4ce0486d8928b76c8e0a8126d14ab47a915` |
| `append-01` | `bccaaccd6e536a8bf712340d0938c0cdff0839fda451c9d1af90d5bea42d0e6f` |
| `guards-01` | `b91302ad758be163e027473a20ce122070705a83f4d1379331f2cc158257a0f0` |
| `distributions-01` | `ca76dbc5f079d622c34964cf4da38bd7bde9477689352d101c215d750344c4a4` |
| `client-prepare-01` | `057789cf7ec34301d3ab6744180516e1af851d39df8cfbd939c88003e188d658` |
| `regression-driver-01` | `d3c0d0f5cf11f4a0ac8824b4d1dda3d03d8d1ac6a44cf7434286c2591ff9654a` |
| `more-regression-driver-01` | `f8ba3f719ae9df6e0744be81172b4f8078055d398daca03cedcbc173413f5a99` |
| `client-prepare-02` | `66e2b38099dce6e3c362a6669a6d2c748e153eedb3fcfdf96f5fcadb4c06fb15` |
| `client-512-01` | `3117f47155706a0f2c62b0e4697717e597f4590768a8036122d4ea97dca03302` |
| `client-2048-01` | `3107ce2cb32bbb97e9148f38a8e3e4b02745bb2603561084d728409bc8423ec9` |
| `client-8192-01` | `d6d24f0ad9320f8f3cbcd5c4ccc8c930fe6e5ccfc2237af304553a3e4a814a9f` |
| `client-capacity-01` | `669a0b5579239d5f6920d53d2256623e7df33123724b9ad80b1dacb42c34cf70` |
| `remote-append-01` | `b4f46296b65f19400f052ba7aeb884a1f82ac860c888299ec1d7743d35e50034` |
| `append-api-01` | `0a5239b791283423f53e2e33a20dbc68eb94927e3bf6f458e9f23ce5b3ee062f` |
| `concurrency-native-01` | `0216e9323dd9531fbe6bcdbc8949ef9fbfd8a071c3db85e54c2aceb8587d5812` |
| `decode-proxy-01` | `d2970732a4e592b49d18472aea677fe515933045af5bb16b03aabeda3215c400` |
| `regression-driver-02` | `a43e1cffee4564982ab06fcee94b2cc528b89f30d5ef200fbceaba1b5cf887d4` |
| `conversation-thinking-on-01` | `2f10b0b15dd7e8578429c48a67087b3f7d638b56e17e03f628aaa29304450f17` |
| `prune-closed-01` | `b5cab8a0d033762966523c26798e2062f7ad39fa9b4b98bbd9ad33ff8558044b` |
| `conversation-thinking-off-01` | `f2412f94c02fa54a599751c5e4e4d752e3394175029352096514d0a37380969d` |
| `conversation-inferred-omit-01` | `2f10b0b15dd7e8578429c48a67087b3f7d638b56e17e03f628aaa29304450f17` |
| `conversation-no-checkpoint-01` | `d5dccbc84cd090ecfd8d06614b3281260ec8670619c04d3fbfd90b9a25246f14` |
| `typed-identity-01` | `508ae2d8e4fa302c58477e3c92e8a672fd272954e3dd79d42778929704f78bce` |
| `typed-no-checkpoint-01` | `d6e1dd6120d52a8be2ddbee5200e27e96c898c0b5847e94fc97dbcea604894ce` |
| `header-http-01` | `1377f8e031ac8448141ae22536cb83c4135c0860bbe5f16b25484614ca72bbad` |
| `launch-final-node-01` | `bee49f3de25cf9243e399184e64da25c207996fc7c9dab584f78988a13de4dab` |
| `launcher-final-01` | `e9787fb06aaab1bf52e9dcad46c9b70133dcc06ee189731df23016295be51469` |
| `integrated-negatives-06` | `807786f2fe5cf8f4fdbf6505a9dd28d638a7645fa99b11c8c9bce25a08ec7ca1` |
| `evidence-snapshot-01` | `66d20cfb576280e42306b3478232c6acde3a0733bf64444d20f66de1797e8021` |
| `prune-closed-02` | `60611e9cb7c802b961aa76d0db8cdd47b8ac592e02c55c7027596aebcc397167` |
| `deep-http-01` | `e152533fe44f8a0a2241c983c75774358708b7209fb614969333937b28f8f972` |
| `more-regression-driver-02` | `5f1589d34394b3d647046b5ac19fb6df6423a76be80831c1beee1d31565d4ca6` |
| `bench-unit-06` | `b167c14f6706022fb262160d7616e3be168a21084aaa7c3742f0055f3e45b7bd` |
| `audit-record-controls-01` | `12f41f0afeac6af054efab06720879e8dd3cf31a4559d530ed756987bb6b5955` |
| `source-scope-02` | `3162728820348abf340ccbd7fe175ee06f974152c34d76e18e5d2437d23881c0` |
| `deep-distributions-01` | `c3dce168f4e197ebb4790a80342342d661389273bfe9f63674cd49963aae88a2` |
| `deep-distribution-driver-01` | `e0d3a3f4c3cfd199f61ef8031d9172bf6ffa5193ab9230e6d829c29a03ff14b9` |
| `premature-close-control-01` | `900a2c20e8470f70bec48aa8373ec9c678c428874b46050b4cde0516093d28b4` |
| `models-final-01` | `b608e51f3b87cc7a2bfd820cfabb34a2b3be84b52f03f74ed2793240c1dfff57` |
| `long-benchmark-01` | `5d6f93c0ec5fbd083b924a5ed18ba91d77c75060c878e3c785d832f4b5da004f` |
| `long-audit-01` | `02c245846bb3cba2a8984972f8ae50a9454a43770c5233453f9b0de678c96e7c` |
| `bundle-audit-02` | `251ae43cee1644c24677a4805e05eb0e6f3995e39733e7c51c2e0ba3ba7e1b38` |
| `close-ready-01` | `0cc26ec1f47910694bd4f0e8c56ff85a38658e50a53c9ba62f7d1e23d3c67fdf` |
| `runtime-cleanup-01` | `b947d2b9181f6244ecf09a38a9820fd535df9e05bc2c1db54a930bbf474689ff` |
| `evidence-snapshot-02` | `44f4335df29efb284b5da769c5f332dc2b5b51c92543bd4f9c4eedeca2b8040f` |

## Requested scratch deletion

At 2026-11-01T04:49:15.369973+00:00, every remaining regular file under `.temp/hybrid-a-r11-261101/` was SHA-256 hashed, then that exact task root was deleted and absence verified: 555 files, 1019352901 bytes. Canonical sorted JSONL manifest SHA-256: `bcf2b2bf2c3c430ea8267c5c8bba2d72ce76577cbcc5f53fa08786200b9c1698`. The manifest itself and all raw task scratch were discarded; only this content-free digest/summary remains in the report. Final free space: 40080838656 bytes.

Earlier hash-before-delete manifests (also discarded with the task root):

| Manifest | Entries | Bytes hashed | Manifest SHA-256 |
|---|---:|---:|---|
| `pruned-state-hashes.jsonl` | 116 | 5879785078 | `7b5d73de1d27341d539182b667c6fa7d957ba7a0e7efe0d9aa41c8d3745db249` |
| `state-dump-hashes.jsonl` | 56 | 1178273476 | `20ba84fc393e96b3596c8c35184e94337e560578d72a564cf06a27daf9cc02a4` |

The shared original model files, shared build lock and incremental build cache were preserved. No owned runtime PID remained active. The deleted paths are evidence locations from this run, not retained downloadable artifacts. Report construction and this final exact-root cleanup each exited 0; the report and signed Git history are the retained delivery artifacts.

# M1 benchmark preparation: hybrid-b-r15-261031

Ready for review. Local signed delivery only; no push, pull request, remote review or landing is claimed.

## Source and scope

- Base: `proto/main` = `231d58ad0821bbd5f5f208bf2ba9f4b1ee09d6c1`; branch `m1-hybrid-bench`. Other branches were preserved.
- Implementation: signed `e61fa75cb1dcec51121b6a27ce872fef05db031c`; exact-ref source guard: signed `d493b9ab2abf862a9fb9af3dfb13594174950ec1`. Both signatures verified `G`.
- Native/proxy/benchmark workload evidence below is bound to clean-source `e61fa75cb1`. The later source guard changes only deployment source admission and its named unit test; the native/proxy/workload paths are byte-unchanged. Their green evidence is reused with that finite identity boundary.
- Added Linux prefill and Linux/macOS client launchers, bounded GGUF sizing/download verification, private PID/credential lifecycle, seeded chat workloads, a real-node distribution checker, opt-in trace/probe/mode support, bundles and the operator guide.
- No `src/` or server source changes, no writes to another workbench, no new CLI/Python dependencies and no model downloads. A pre-existing CMake installation was added to this task process PATH; it was executed read-only.
- Self-review applied the repository code-review skill: Scope, Security, Approach, General and Tests. There is no surface table in the brief; that is a brief gap, not a claim of exhaustive surface coverage. No prior reviewer tests were removed: all 61 prior names are present in the 63-test proxy suite.

## Build identity and resource admission

Only incremental CPU builds ran, under the shared model-directory `.build.lock` using `-j2`. Initial observed free bytes: 42,247,426,048. Final controlled build minimum: 42,008,666,112 bytes, above the 26,843,545,600-byte (25 GiB) floor. No low-disk termination was needed.

Exact native commands, invoked by each launcher under the lock:

```sh
cmake -S "$CHECKOUT" -B "$CHECKOUT/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF
cmake --build "$CHECKOUT/build" -j2 --target llama-hybrid-prefill
# Client configure uses the same flags.
cmake --build "$CHECKOUT/build" -j2 --target llama-server llama-hybrid-bench-check
```

Both final configure/build pairs exited 0. `source_dirty=false`, commit `e61fa75cb1`, context 2,048 on both CPU nodes. Model: `Qwen3.5-0.8B-Q4_0.gguf`, 507,154,688 bytes, SHA-256 `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`. GGUF sizing gives 6,528 KV bytes/token and 20,201,472 recurrent bytes, with a 20% margin and 2 GiB compute reserve.

Final controlled executable hashes (before -> after):

| Target | Before SHA-256 | After SHA-256 |
|---|---|---|
| llama-hybrid-prefill | `e5e618628fdf5609028eb25fa1166a64bcc3e372be6aa387c4d9eac6da370a84` | `e5e618628fdf5609028eb25fa1166a64bcc3e372be6aa387c4d9eac6da370a84` |
| llama-server | `3c64e08cccbd9953574f543923ff914f35531a2f9868b51532645c5b2b1a9ee7` | `c8dd2816ddc37a4303ed7a404ab17dd6fad8ada44ea9cd687ac0f627c2977840` |
| llama-hybrid-bench-check | `e54ce1ebc402a8aedf5559839a0ed61e5680c82737f45e823bc22cc15096b106` | `e54ce1ebc402a8aedf5559839a0ed61e5680c82737f45e823bc22cc15096b106` |

Executable and shared-library hashes, before/after, are retained in `$RUN/build-records-final.json` (SHA-256 `73acc1b5cfbff5cb69f73858e3ec8882af87619039bf6a5f5d5a26f7fd66eafe`). The first diagnostic build record was superseded before all pre-build library hashes were retained; the controlled final builds above close that evidence gap for the shipped runtime.

## Positive CPU results

Signed-runtime smoke: `hybrid-bench-cpu-signed-20261031T230327391335Z`, 2026-10-31 23:03:27 UTC start, exit 0, 24.352 seconds including the check. Each mode made eight calls over the same seeded sizes and authoritative depths; each processed 682 new prompt tokens and generated 32 tokens, ending at depth 714. Raw tool results were 16/32/64 tokens at each smoke rung. Actual prompt depths and each call's timing/counters are in JSONL.

| Mode | Actual paths | Mean TTFT (s) | Call total (s) | Mean native decode (tokens/s) |
|---|---|---:|---:|---:|
| local | local | 0.266586 | 2.962524 | 63.440 |
| hybrid | local | 0.616520 | 5.883238 | 55.441 |
| offload | offload | 0.542733 | 5.282664 | 59.516 |

The policy selected local for every small hybrid-mode turn. Forced offload transferred 166,039,522 delta-envelope body bytes; critical server prefill 2232.554 ms, body transfer 93.712 ms, import HTTP 774.305 ms across eight calls. Awaited shadow time is reported separately. These CPU/loopback measurements show overhead for this small workload, not a GPU/Mac speedup.

Link probe: median application RTT 0.398555 ms; 1,048,576-byte download 1212.176 MB/s including request overhead. This is neither physical-link nor upload bandwidth.

Equivalence at exact observed prompt depth 132:

| Claim | Observation | Section-11 reference |
|---|---|---|
| Same-partition greedy exactness | 64/64 | 64/64 on the same backend |
| Teacher-forced full-vocabulary top-1 | 64/64 (100%) | >=95% |
| Mean KL | 0.001277003497 nats/token | <=0.01 |
| First whole-prompt free-running divergence | none in 64 tokens | informational |

The checker imports a freshly computed remote [0,N-1) delta of the exact observed prompt, evaluates the held token locally, then teacher-forces the same continuation through imported/split/whole references. This one short synthetic prompt does not establish the 50-prompt 4k/32k/128k matrix, accumulated multi-turn distribution equivalence or cross-backend portability.

Initial diagnostic smoke also exited 0: local/hybrid/offload mean TTFT 0.267165/0.518300/0.546530 s; total 2.980116/4.986069/5.257858 s; native decode 64.516/64.593/59.782 tokens/s. Its equivalence numbers match the signed-runtime check. The signed-runtime results are the delivery evidence.

Additional stage-accounting controls:

- Full fixed ladder at the deliberately small 2,048 context: exit 4, partial, one fitting call per mode and 72 explicit skipped workload/check stages. It does not claim 16k-256k execution.
- Measured smoke, seed 0: exit 4, partial, four fitting calls per mode and 12 context-capped skips. Quantile unit tests establish repeatability and tail clamping; no source conversations are read at runtime.

## Negative and regression claims

Each executable test has a concise claim comment beside it. Final native controls exited 0 as a harness; the intended refusal commands exited 2, and unauthorized HTTP requests returned 401. Failed setup left no state/staging directories, source model bytes stayed unchanged and the mismatched download left an empty cache.

| Case | Intended observation | Evidence |
|---|---|---|
| Node/client local model hash mismatch | CLI 2 before build/start | `bad-hash-node`, `bad-hash-client` |
| Different model files | CLI 2, client/node model identity refusal | `different-model`; second full GGUF copy with a synthetic trailing-byte change |
| Download hash mismatch | CLI 2; no final or temporary cache object | `download-mismatch`, small local `file://` fixture |
| Missing/wrong node token | HTTP 401; session list unchanged | `missing-token`, `wrong-token` |
| Missing/wrong client token at setup | CLI 2; no local process/config | `missing-client-token`, `wrong-client-token` |
| Public mode without token file | CLI 2 | `public-no-token` |
| Context cannot fit | CLI 2 with maximum/bound arithmetic | `context-overflow` |
| Node unreachable | CLI 2, explicit unreachable diagnostic | `unreachable` |
| Node killed after the second call preflight | Benchmark 2, aborted; fallback cannot count as forced offload | `killed-mid-run` |
| Full re-prefill trace violation | Benchmark 2 after a positive first call | `reprefill-control` |
| Reused PID identity | no signal sent | `test_pid_reuse` |
| Dirty exact-ref sources | no CMake/worktree mutation | `test_build_refuses_dirty_source` |

The re-prefill control changes only the real run's telemetry (`prefilled += depth_before`), not native engine behavior. `test_full_reprefill_control` and the native `reprefill-control` reject that named violation; this narrows the claim to counter/boundary detection. No code mutation campaign is claimed. On a killed-node fallback the proxy may commit a valid local continuation, per its existing recovery contract; the benchmark aborts the comparison instead of discarding that valid state.

Regression: 63 proxy tests passed, all 61 prior named tests retained. Real remote-append test passed: local 1 + remote 28 = new 29; checkpoint rewrite, retired-session deletion and explicit 409/native-byte preservation controls stayed green.

## Exact command ledger and exits

Commands below run from the checkout root. `$RUN=.temp/hybrid-b-r15-261031`; `$MODEL` is the local 0.8B GGUF specified above. `$CHECKOUT` is this checkout. Logs use the ledger name plus `.log` under `$RUN`. Each command was a standalone process; no test/build was piped through `tee`.

| Ledger name | Exact command (path variables substituted only) | Exit |
|---|---|---:|
| deploy-node-01 | `tools/hybrid-bench/deploy-prefill-node.sh --state $RUN/dumps/node --model $MODEL --ctx 2048 --port 18911` | 0 |
| benchmark-proxy-tests-01 | `python3 -m unittest discover -s tools/hybrid-proxy -p test_benchmark.py -v` | 0 |
| setup-client-01 | `tools/hybrid-bench/setup-client.sh --state $RUN/dumps/client --model $MODEL --ctx 2048 --node-info $RUN/dumps/node/node-info.json --token-file $RUN/dumps/node/client-tokens.json --prefill-url http://127.0.0.1:18911 --decode-port 18910 --port 18912` | 0 |
| smoke-01 | `python3 tools/hybrid-bench/hybrid_bench.py --client-state $RUN/dumps/client --output $RUN --scratch $RUN/dumps --label cpu --smoke --check` | 0 |
| native-negatives-01 | `python3 tools/hybrid-bench/test_native.py --client-state $RUN/dumps/client --node-state $RUN/dumps/node --model $MODEL --artifacts $RUN/dumps --output $RUN` | 1 |
| native-negatives-02 | `python3 tools/hybrid-bench/test_native.py --client-state $RUN/dumps/client --node-state $RUN/dumps/node --model $MODEL --artifacts $RUN/dumps/negatives-02 --output $RUN` | 0 |
| bench-unit-02 | `python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v` | 0 |
| stop-client-01 | `tools/hybrid-bench/setup-client.sh --state $RUN/dumps/client --stop` | 0 |
| remote-append-01 | `python3 tools/hybrid-proxy/test_remote_append.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model $MODEL --artifacts $RUN/dumps/remote-append` | 0 |
| deploy-node-02 | `tools/hybrid-bench/deploy-prefill-node.sh --state $RUN/dumps/final/node --model $MODEL --sha256 444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c --ctx 2048 --port 18911` | 0 |
| bench-unit-03 | `python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v` | 0 |
| setup-client-02 | `tools/hybrid-bench/setup-client.sh --state $RUN/dumps/final/client --model $MODEL --sha256 444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c --ctx 2048 --node-info $RUN/dumps/final/node/node-info.json --token-file $RUN/dumps/final/node/client-tokens.json --prefill-url http://127.0.0.1:18911 --decode-port 18910 --port 18912` | 0 |
| node-status-01 | `tools/hybrid-bench/deploy-prefill-node.sh --state $RUN/dumps/final/node --status` | 0 |
| client-status-01 | `tools/hybrid-bench/setup-client.sh --state $RUN/dumps/final/client --status` | 0 |
| shell-node-01 | `sh -n tools/hybrid-bench/deploy-prefill-node.sh` | 0 |
| shell-client-01 | `sh -n tools/hybrid-bench/setup-client.sh` | 0 |
| syntax-02 | `python3 -m py_compile tools/hybrid-bench/bench_common.py tools/hybrid-bench/manage.py tools/hybrid-bench/hybrid_bench.py tools/hybrid-bench/test_bench.py tools/hybrid-bench/test_native.py tools/hybrid-proxy/proxy.py tools/hybrid-proxy/test_benchmark.py` | 0 |
| diff-check-01 | `git diff --check` | 0 |
| proxy-tests-02 | `python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v` | 0 |
| ty-readiness-02 | `sh -c 'ty --version'` | 127 |
| editorconfig-readiness-02 | `sh -c 'editorconfig-checker --version'` | 127 |
| smoke-02 | `python3 tools/hybrid-bench/hybrid_bench.py --client-state $RUN/dumps/final/client --output $RUN --scratch $RUN/dumps --label cpu-signed --smoke --check` | 0 |
| full-capped-01 | `python3 tools/hybrid-bench/hybrid_bench.py --client-state $RUN/dumps/final/client --output $RUN --scratch $RUN/dumps --label cpu-cap --check` | 4 |
| measured-smoke-01 | `python3 tools/hybrid-bench/hybrid_bench.py --client-state $RUN/dumps/final/client --output $RUN --scratch $RUN/dumps --label cpu-measured --smoke --schedule measured` | 4 |
| bench-unit-04 | `python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v` | 0 |
| native-negatives-03 | `python3 tools/hybrid-bench/test_native.py --client-state $RUN/dumps/final/client --node-state $RUN/dumps/final/node --model $MODEL --artifacts $RUN/dumps/negatives-03 --output $RUN` | 0 |
| node-status-stopped-01 | `tools/hybrid-bench/deploy-prefill-node.sh --state $RUN/dumps/final/node --status` | 3 |
| stop-client-02 | `tools/hybrid-bench/setup-client.sh --state $RUN/dumps/final/client --stop` | 0 |
| stop-node-02 | `tools/hybrid-bench/deploy-prefill-node.sh --state $RUN/dumps/final/node --stop` | 0 |
| cleanup-audit-01 | `python3 $RUN/audit_cleanup.py` | 0 |

Earlier standalone checks: `python3 -m unittest discover -s tools/hybrid-proxy -p test_*.py -v` exited 0 (61 tests; `proxy-tests-01.log`); `python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v` exited 0 (initial seven tests; `bench-unit-01.log`); initial Python compilation and shell syntax checks exited 0. `cmake --version` initially failed 127 because it was absent from PATH; the existing CMake binary then returned 0 (`cmake-readiness-02.log`).

Final native harness subcommands (arguments expanded by the harness; all are expected refusals unless stated):

| Case | Command | Exit |
|---|---|---:|
| bad-hash-node | `tools/hybrid-bench/deploy-prefill-node.sh --model $MODEL --port 18921 --state $RUN/dumps/negatives-03/bad-hash-node --sha256 0000000000000000000000000000000000000000000000000000000000000000` | 2 |
| bad-hash-client | `tools/hybrid-bench/setup-client.sh --model $MODEL --prefill-url http://127.0.0.1:18911 --token-file $RUN/dumps/final/node/client-tokens.json --node-sha256 444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c --decode-port 18920 --port 18922 --state $RUN/dumps/negatives-03/bad-hash-client --sha256 0000000000000000000000000000000000000000000000000000000000000000` | 2 |
| different-model | `tools/hybrid-bench/setup-client.sh --model $RUN/dumps/negatives-03/different-model.gguf --prefill-url http://127.0.0.1:18911 --token-file $RUN/dumps/final/node/client-tokens.json --node-sha256 444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c --decode-port 18920 --port 18922 --state $RUN/dumps/negatives-03/different-model` | 2 |
| public-no-token | `tools/hybrid-bench/deploy-prefill-node.sh --model $MODEL --port 18921 --state $RUN/dumps/negatives-03/public-no-token --public` | 2 |
| context-overflow | `tools/hybrid-bench/deploy-prefill-node.sh --model $MODEL --port 18921 --state $RUN/dumps/negatives-03/context-overflow --ctx 1000000000` | 2 |
| download-mismatch | `tools/hybrid-bench/deploy-prefill-node.sh --state $RUN/dumps/negatives-03/download-mismatch --source-url file://$RUN/dumps/negatives-03/download-fixture --sha256 0000000000000000000000000000000000000000000000000000000000000000 --model-cache $RUN/dumps/negatives-03/cache` | 2 |
| wrong-client-token | `tools/hybrid-bench/setup-client.sh --model $MODEL --prefill-url http://127.0.0.1:18911 --token-file $RUN/dumps/negatives-03/wrong-token.json --node-sha256 444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c --decode-port 18920 --port 18922 --state $RUN/dumps/negatives-03/wrong-client-token` | 2 |
| missing-client-token | `tools/hybrid-bench/setup-client.sh --model $MODEL --prefill-url http://127.0.0.1:18911 --node-sha256 444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c --decode-port 18920 --port 18922 --state $RUN/dumps/negatives-03/missing-client-token` | 2 |
| unreachable | `tools/hybrid-bench/setup-client.sh --model $MODEL --prefill-url http://127.0.0.1:1 --token-file $RUN/dumps/final/node/client-tokens.json --node-sha256 444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c --decode-port 18920 --port 18922 --state $RUN/dumps/negatives-03/unreachable` | 2 |
| reprefill-control | `/usr/bin/python3 tools/hybrid-bench/hybrid_bench.py --client-state $RUN/dumps/negatives-03/reprefill-control --output $RUN --scratch $RUN/dumps/negatives-03 --label reprefill-control --modes offload --smoke` | 2 |
| killed-mid-run | `/usr/bin/python3 tools/hybrid-bench/hybrid_bench.py --client-state $RUN/dumps/negatives-03/killed-mid-run --output $RUN --scratch $RUN/dumps/negatives-03 --label killed-mid-run --modes offload --smoke` | 2 |

The two token HTTP401 checks run inside the native harness, not as separate shell commands. Initial diagnostic subcommand ledgers are retained under `$RUN/negative-evidence/01` and `/02`; their refusal commands also exited 2.

Failures and corrected controls are preserved:

- `native-negatives-01` really exited 1: the harness imported its HTTP helper over the `http` module name before constructing the control bridge. That was a harness setup error, not successful negative evidence. The alias was corrected; rounds 02 and 03 exited 0, with both planted stream controls reaching their intended benchmark exit 2.
- Initial `ty --version` and `editorconfig-checker --version` attempts through the evidence wrapper exited 1 when the wrapper raised `FileNotFoundError`. The corrected standalone shell probes each exited 127. Type checking and the external EditorConfig checker were not run. The final audit checks only ASCII, LF/final newline and trailing whitespace in changed files; it is not reported as the absent external checks.
- Expected nonzero benchmark/status exits remain nonzero: capped schedules 4, deliberately aborted negative runs 2, stopped node status 3. They are not relabelled passing runs.

## Bundle and cleanup

Primary returned bundle: `hybrid-bench-cpu-signed-20261031T230327391335Z.tar.gz`, SHA-256 `3f0f54f911f4689f9d8e937ee3973c442aae03611c30283a2bfd208e189ab438`. Its `SHA256SUMS` is:

```text
1029b65228445018efaecb1c1b8282f903ab0f91e6439339a2cb00c1166477b1  client-info.json
55c84995eb34432fa8bfd884b54e673425cd73057e6afbb802dcae7e5732a543  hybrid-bench-cpu-signed-20261031T230327391335Z.jsonl
54913c1193145d10c0b8ddf8f5f9b29a09373e911dd22c8801414f68ff9d74f9  hybrid-bench-cpu-signed-20261031T230327391335Z.md
42633653f298fea1602f0f952991160873d51b5cfb0c38933159c63be79c16ed  node-info.json
```

All 8 produced bundles (including partial/aborted controls) passed member-manifest and outer-sidecar verification, and were checked against the active credentials for accidental inclusion. Only content-free result files and sanitized node/client info are included. Audit: `$RUN/bundle-audit.json`.

Cleanup audit: 112 closed scratch files / 1,027,185,523 bytes hashed, plus 8 earlier-deleted checker inputs/results / 43,018,984 bytes. Manifest `$RUN/state-dump-manifest.json`, SHA-256 `a4ce72d7f4d2dda9bce9abddb8ebad156ec4ddab37427ef4334f521998999d18`. The entire owned `dumps/` directory was deleted. 39 recorded PIDs checked absent, and all three owned ports closed. Engines were stopped only by recorded PID/start identity. The initial pure unit run used automatically removed synthetic temporary fixtures; native state and every live process fixture were confined to the task dumps directory.

## Untested steps and open questions

- CUDA configure/build/execution, GPU preflight queries and VRAM fitting: unrun because this round forbids GPU code paths and has no assigned GPU host.
- macOS preflight, Metal compile/runtime, SSH tunnelling between machines, non-loopback public binding and CUDA-to-Metal equivalence: unrun because no Mac/GPU/remote pair is assigned. Loopback CPU HTTP exercised actual auth, state transfer and import.
- HF HTTPS model download and large-model download interruption: unrun on this host by instruction. The same standard-library download function was exercised with a small local file, including successful publication and mismatch cleanup.
- The disk guard never hit its 25 GiB abort floor; target-platform OOM and low-disk fault injection were not performed.
- Current inherited 16,384-token and 512 MiB session/delta bounds prevent long-depth 27B acceptance even when card memory is sufficient. The guide shows memory-only arithmetic for 24/32/48/80/96/180 GB classes and distinguishes it from operational support. Store/transport expansion needs its own design and review.
- Section-11 50-prompt coverage, realistic links, multi-client load and calibrated cross-backend thresholds remain open. Cross-backend results explicitly report uncalibrated reference thresholds.
- No surface table was supplied. Out-of-contract qualification here is the GPU/Mac/27B hardware matrix, as expressly deferred by the CPU-only limits; context-capped stages are reported partial. No production security or whole-project correctness claim is made.
- Remote review/acceptance is pending. Local signed commits and producer tests alone do not constitute accepted delivery. See `UNRESOLVED_QUESTIONS.md` for the finite remaining scope.

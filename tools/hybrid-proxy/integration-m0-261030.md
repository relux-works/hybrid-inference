# M0 integration evidence

Run `hybrid-b-r8-261030`, 2026-10-30 UTC. Integration branch `prototype-m0`; no publication or pull request was attempted. The orchestrator owns publication and review. **Verdict: ready for orchestrator review.** The requested integrated CPU gate passes after the setup issues documented below; this does not grant broader section-11 matrix or publication approval.

## Inputs and signed history

| Commit | Purpose |
|---|---|
| `a93a0cde136f5d260a8f8a534a056f0c819557c3` | Accepted client parent |
| `9a029f385df357c8c796280473e2017265bb941c` | Accepted prefill parent |
| `1d6624c50967b117f06e2e10911b0c514a38cc78` | Prototype main documentation baseline |
| `42dd9e3e50a59042ee5fb0693134451eae82e8d3` | Signed no-fast-forward merge of A into B |
| `4878c9bc79d07d927849a8a47267a6fc87786f62` | Signed no-fast-forward merge of main |
| `328cfb23cf25e8e6d202438d45a4bca55ccdc7e8` | Verbatim owner README/specification update; built source identity |

All six objects verify with the configured signing identity. New merge commits retain the accepted signed parent objects. The workbench branch started at the exact accepted B head; upstream and main were not rewritten.

## Conflict resolutions and retention

| Path | Resolution and reason |
|---|---|
| `src/llama-context.cpp` | B verbatim: recurrent-only rollback, deferred writes and attention suffix removal retain the prefix without copying the full sequence. |
| `src/llama-kv-cache.cpp` | B verbatim: portable one-sequence restore, SWA guard and append failure cleanup preserve the existing prefix. |
| `src/llama-memory-hybrid.cpp` | B verbatim: append errors delegate rollback to the transaction and do not clear attention. |
| `src/llama-memory.h` | B verbatim: internal flag bits 30/31 and public-flag overlap assertion. |

These four content conflicts were the only merge conflicts. The complete `src/` tree equals B at `a93a0cde`. A range export remains intact: `llama_context::state_seq_get_data_range`, `llama_state_seq_get_data_range`, `llama_kv_cache::state_write_range`, position filtering and hybrid recurrent export are retained and exercised by A's independent checker. The main merge had no conflict. The A merge command was `git merge -S --no-ff --no-commit 9a029f385df357c8c796280473e2017265bb941c` (exit 1 for the four expected content conflicts); the signed resolving commit exited 0. The main merge was `git merge -S --no-ff refs/remotes/prototype/main` with its explicit message and Assisted-by trailer (exit 0).

`prefill.cpp`, `hybrid-scheduler.h`, `test_prefill.py` and `check.cpp` equal accepted A byte for byte. Both codec files were already identical and merged without a conflict; the explicit no-index comparisons below are empty, exit 0. `test_recurrent_append.py` is also byte-identical in both accepted heads, so retaining it preserves every assertion on both sides.

`tools/hybrid/CMakeLists.txt` retains A's prefill/check targets plus B's PIC setting and append, guards, distributions, client and codec checkers. `tools/hybrid/README.md` combines A's updated admission/budget commands and reports with commands and claims for all B-owned native checkers. The shared-golden-writing example was removed from the test command list; `GOLDEN_DIR=-` runs A's full checker without rewriting fixtures. All A and B review/fix/result reports remain byte-identical to their accepted originals (`R/retention-audit.json`).

The two owner documents were copied without editing in the requested single signed commit:

- `/home/agent/lanes/hybrid/.temp/integration/HYBRID.md` -> `HYBRID.md`; SHA-256 `fe0e5d21294b24cad54ff4bfee8160ab4b3a80d07c827c368bf9e88e067cf722`.
- `/home/agent/lanes/hybrid/docs/prototype-spec.md` -> `docs/hybrid/prototype-spec.md`; SHA-256 `3b648e79bea9363be6b1e7f739f62825d0b429fc5dcc2d23312f69f6da4974d0`.

## Build identity and resource admission

`R` below means `.temp/hybrid-b-r8-261030` in this workbench. Every validation command runs as its own process through `python3 R/run.py LABEL COMMAND...`; the wrapper redirects stdout/stderr directly to `R/LABEL.log`, records the exact argument vector, actual UTC start, elapsed time and real process exit in `R/gates.jsonl`, and returns that exit. No pipeline or `tee` masks a result.

One configure and exactly one build ran under the shared `flock`, with `-j2`, CPU-native release configuration, curl/OpenSSL/UI disabled. The guard runs `df -B1 --output=avail /` before starting and every two seconds while running, refusing or terminating its own recorded process tree below 26,843,545,600 available bytes. No dependency installation was needed; CMake 4.4.4 was read from A's existing task-local installation.

| Binary/library | SHA-256 |
|---|---|
| `build/bin/llama-server` | `b88d7a41a493de8fb1089218954c4fd0c953d6ee0de66f228c8b7d9837340a9e` |
| `build/bin/llama-hybrid-prefill` | `5af5b1901d918b9db7eb6d5b94fb79d238f625ee3cb51b660f4c64f40de80240` |
| `build/bin/llama-hybrid-check` | `749d5d1d96b05295f829cb4e968f9664e7eb46d53b73e9686702d94b81e8830b` |
| `build/bin/llama-hybrid-check-append` | `22bdc0caec1f879feb56fff7f1fd2246d8aa52d6155866c22ce5ceefa17c982e` |
| `build/bin/llama-hybrid-check-guards` | `e56e14889b5a57abb899f0815a75acd8438035cfbca479db4c8de70f91a181e7` |
| `build/bin/llama-hybrid-check-distributions` | `ce6de8ea0b6d2fd37511f903adea0b66d9037c9fccb27878a6e1326f61fa4814` |
| `build/bin/llama-hybrid-check-client` | `9e04a51070f0aa1460fc6817d896d0e06712dd84f014b550377006c790196473` |
| `build/bin/llama-hybrid-check-codec` | `969bedf6aa7eca8e1a60de772bf0e81ef7021b22c9721e4742379d173ef501cf` |
| `build/bin/libllama.so` | `1ad5d26ba8388d05ca87d8872dd6dcbc104694c24b38dff93ee7ae44844c8865` |
| `build/bin/libggml.so` | `dd38e8f2f84f9cd2e0587a14fac04f09e2a6c31856d2d8cf5bd85b417630a0f8` |
| `build/bin/libggml-base.so` | `33a4834d279ec246e9d0f4429069c30d1cbb41af12413ebbe5df71a4b61b968d` |
| `build/bin/libggml-cpu.so` | `a3df2ed3dd3eb095803de4b893c549e2b0868769814a0dc5aaa29d0e79ccb42c` |

Build tree immediately after build: **86,412,492 bytes** (`du -sb build`). `R/source-identity.json` records source HEAD/tree and hashes for source, headers, server, codec, checkers, proxy and tests; `R/binary-identity.json` records these hashes and post-build free bytes. All native gates use this build, including both servers in the end-to-end run. `llama-server --version` identifies build 12044, commit `328cfb23cf`, GNU 15.2.0, Linux x86_64. A later documentation-only correction adds `mkdir -p .temp/append-cost` before the README fixture command; every compiled source, test, configuration and binary remains unchanged. The identity verifier permits only that exact one-line documentation delta and the new report.

## Gate commands and actual exits

All commands below ran from `/home/agent/Werkstatt/hybrid-b/llama.cpp`. Labels name the corresponding `R/LABEL.log`; `R/gates.jsonl` stores exact argument vectors, UTC observation times and durations.

| Label | Exact executed command | Exit |
|---|---|---|
| `readiness-wrapper-01` | `python3 --version` | **0** |
| `configure-01` | `flock /home/agent/Werkstatt/hybrid-models/.build.lock python3 .temp/hybrid-b-r1-261030/build_guard.py --minimum-start-gib 25 --abort-gib 25 --log-prefix .temp/hybrid-b-r8-261030/configure-guard-01 -- /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF` | **0** |
| `build-01` | `flock /home/agent/Werkstatt/hybrid-models/.build.lock python3 .temp/hybrid-b-r1-261030/build_guard.py --minimum-start-gib 25 --abort-gib 25 --log-prefix .temp/hybrid-b-r8-261030/build-guard-01 -- /home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target llama-server llama-hybrid-prefill llama-hybrid-check llama-hybrid-check-append llama-hybrid-check-guards llama-hybrid-check-distributions llama-hybrid-check-client llama-hybrid-check-codec` | **0** |
| `proxy-01` | `python3 -m unittest discover -s tools/hybrid-proxy -p test_proxy.py -v` | **0** |
| `a-codec-scheduler-01` | `build/bin/llama-hybrid-check --codec` | **0** |
| `b-codec-budgets-01` | `build/bin/llama-hybrid-check-codec` | **0** |
| `http-isolation-01` | `python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r8-261030/http-isolation-01 --port 18128` | **0** |
| `admission-f1-01` | `python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r8-261030/admission-f1-01 --port 18128 --regression slow-readers` | **0** |
| `codec-identical-h-01` | `git diff --no-index .temp/hybrid-b-r8-261030/accepted-a-codec/hybrid-state.h tools/hybrid/hybrid-state.h` | **0** |
| `codec-identical-cpp-01` | `git diff --no-index .temp/hybrid-b-r8-261030/accepted-a-codec/hybrid-state.cpp tools/hybrid/hybrid-state.cpp` | **0** |
| `admission-pending-01` | `python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r8-261030/admission-pending-01 --port 18128 --regression pending` | **0** |
| `admission-errors-01` | `python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r8-261030/admission-errors-01 --port 18128 --regression errors` | **0** |
| `a-append-equivalence-01` | `build/bin/llama-hybrid-check /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf -` | **0** |
| `malformed-01` | `python3 tools/hybrid/test_recurrent_append.py --library build/bin/libllama.so --malformed` | **0** |
| `server-version-01` | `build/bin/llama-server --version` | **0** |
| `b-append-01` | `build/bin/llama-hybrid-check-append /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden/prompt-512 /home/agent/Werkstatt/hybrid-models/golden/prompt-2048 /home/agent/Werkstatt/hybrid-models/golden/prompt-8192` | **0** |
| `b-guards-01` | `build/bin/llama-hybrid-check-guards /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden/prompt-512` | **0** |
| `b-capacity-rollback-01` | `build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --capacity /home/agent/Werkstatt/hybrid-models/golden/prompt-512` | **0** |
| `source-binary-inputs-01` | `python3 .temp/hybrid-b-r8-261030/verify-inputs.py` | **0** |
| `f2-max-delta-01` | `python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r8-261030/f2-max-delta-01 --port 18128 --regression max-delta` | **0** |
| `admission-desired-order-01` | `python3 .temp/hybrid-b-r8-261030/adapted-paths.py .temp/hybrid-b-r8-261030/desired-order-01` | **1** |
| `admission-independent-01` | `python3 .temp/hybrid-b-r8-261030/review_admission.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r8-261030/independent-01 --port 18128 --regression pending` | **0** |
| `cost-prepare-01` | `build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --prepare /home/agent/Werkstatt/hybrid-models/golden .temp/hybrid-b-r8-261030/cost` | **1** |
| `b-distributions-01` | `build/bin/llama-hybrid-check-distributions /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden/prompt-512.tokens.json /home/agent/Werkstatt/hybrid-models/golden/prompt-2048.tokens.json /home/agent/Werkstatt/hybrid-models/golden/prompt-8192.tokens.json` | **0** |
| `cost-prepare-02` | `build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --prepare /home/agent/Werkstatt/hybrid-models/golden .temp/hybrid-b-r8-261030/cost` | **0** |
| `native-header-http-01` | `python3 tools/hybrid-proxy/test_header_budget.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r8-261030/native-header-http-01` | **0** |
| `admission-desired-release-control-01` | `python3 .temp/hybrid-b-r8-261030/desired-order-release-control.py` | **0** |
| `admission-desired-order-02` | `python3 .temp/hybrid-b-r8-261030/adapted-paths.py .temp/hybrid-b-r8-261030/desired-order-02` | **0** |
| `real-e2e-01` | `python3 tools/hybrid-proxy/test_decode.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --golden /home/agent/Werkstatt/hybrid-models/golden/prompt-512 --artifacts .temp/hybrid-b-r8-261030/real-e2e-01` | **0** |
| `continuation-control-expected-red-01` | `build/bin/llama-hybrid-check-distributions /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf .temp/hybrid-b-r8-261030/wrong-continuation.tokens.json` | **1** |
| `cost-512-01` | `build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure .temp/hybrid-b-r8-261030/cost/cost-512 512` | **0** |
| `cost-512-02` | `build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure .temp/hybrid-b-r8-261030/cost/cost-512 512` | **0** |
| `cost-512-03` | `build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure .temp/hybrid-b-r8-261030/cost/cost-512 512` | **0** |
| `cost-2048-01` | `build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure .temp/hybrid-b-r8-261030/cost/cost-2048 2048` | **0** |
| `cost-2048-02` | `build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure .temp/hybrid-b-r8-261030/cost/cost-2048 2048` | **0** |
| `cost-2048-03` | `build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure .temp/hybrid-b-r8-261030/cost/cost-2048 2048` | **0** |
| `cost-8192-01` | `build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure .temp/hybrid-b-r8-261030/cost/cost-8192 8192` | **0** |
| `cost-8192-02` | `build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure .temp/hybrid-b-r8-261030/cost/cost-8192 8192` | **0** |
| `cost-8192-03` | `build/bin/llama-hybrid-check-client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure .temp/hybrid-b-r8-261030/cost/cost-8192 8192` | **0** |
| `source-binary-inputs-02` | `python3 .temp/hybrid-b-r8-261030/verify-inputs.py` | **0** |
| `gate-summary-01` | `python3 .temp/hybrid-b-r8-261030/final-checks.py` | **0** |

The real end-to-end harness also directly ran the following native checker as a standalone subprocess, asserted its actual return code 0, and saved `R/real-e2e-01/distribution-check.log`:

```sh
build/bin/llama-hybrid-check-distributions /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r8-261030/real-e2e-01/conversation-a-0.tokens.json /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r8-261030/real-e2e-01/conversation-a-1.tokens.json /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r8-261030/real-e2e-01/conversation-b-0.tokens.json /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r8-261030/real-e2e-01/conversation-b-1.tokens.json
```

Suite-driver aggregate exits, with every child result retained above:

| Command | Exit |
|---|---|
| `python3 .temp/hybrid-b-r8-261030/suite.py .temp/hybrid-b-r8-261030/native-plan.json` | 0 |
| `python3 .temp/hybrid-b-r8-261030/suite.py .temp/hybrid-b-r8-261030/http-plan.json` | **1**: setup-only desired-order failure retained |
| `python3 .temp/hybrid-b-r8-261030/suite.py .temp/hybrid-b-r8-261030/e2e-plan.json` | 0 |
| `python3 .temp/hybrid-b-r8-261030/suite.py .temp/hybrid-b-r8-261030/cost-plan.json` | 0 |

The guarded configure/build subprocesses each exited 0, recorded in `configure-guard-01.monitor.log` and `build-guard-01.monitor.log`. Full compiler output is in `configure-guard-01.log` and `build-guard-01.log`.

## Observations and finite limits

| Requested surface | Actual result and assertion bounds |
|---|---|
| Codec and scheduler | A FIFO/session order, critical priority and client rotation controls pass. Both readers accept 16,384 INT32_MAX tokens (180,611-byte metadata header); writer/reader range and absolute budget violations reject. Both codec no-index diffs are empty. |
| A independent append | 64/64 at depths 512, 2048 and 8192 against independent plain local prefills. Stale, negative, empty and unsupported-flag range requests preserve source bytes. Malformed appends preserve the destination before valid append. No goldens rewritten (`GOLDEN_DIR=-`). |
| Client append and guards | 64/64 at all three golden depths; late endpoint/recurrent failures and invalid envelopes preserve bytes. Portable single-sequence restore passes. Non-SWA append succeeds; STANDARD, CHUNKED and SYMMETRIC SWA appends reject, with both KV sequences byte-identical. |
| Recurrent regression | All 15 malformed records, including named two-cell/no-abort reproducer, reject with both serialized sequences byte-identical; the same valid append succeeds afterward. |
| Post-validation rollback | Actual insufficient KV capacity in a 512-token context rejects after validation with both prefix and neighbor byte-identical. Removing the neighbor allows exactly the same append. |
| HTTP isolation | Three clients x four sessions interleave delta/shadow appends with distinct IDs and exact final hashes. Foreign read/extend/snapshot/delete return 404 without state changes; deletion returns retained RAM accounting to zero. |
| F1 and F1-R2 | Slow readers retain permits; excess X requests get immediate 429; admitted Y completes after one X prefill. The unchanged desired-order helper passes both 2-X and 4-X cases on rerun, retaining all order/body assertions. The first run failed during setup, detailed below. |
| Queued permits and errors | A pending/error suites pass. The independent probe individually cancels queued list/info/create/delete/snapshot/critical requests under global saturation without state changes; shadow and snapshot cancellation also checked individually. Nine release upper bounds: 94.55-107.56 ms, including a 75 ms no-response observation. Authentication precedes permits; health/metrics bypass leases. 400/404/409/JSON errors and repeated actual 503 return permits; eight alternating-client appends succeed. Both A permits are reusable simultaneously after errors. |
| Response lifetime | Saturated admission rejects while slow streams remain open. Draining one stream releases admission (23.84 ms observed); early disconnect releases it (0.77 ms observed). Finite observations, not universal latency guarantees. |
| F2 and native envelope | 16,385 rejects before inference with unchanged snapshot; 16,384 succeeds with exact tokens/hash/native snapshot. Integrated importer accepts the maximum with a 115,101-byte header and 127,550,068 native bytes. Header range/absolute overflows return intended 400 and preserve saved native bytes. Listener cap+1 returns 413 with unchanged metadata. |
| Native decode routes | Portable snapshot restore, import/info, one-token 0->1 and 1->2 boundaries, neighbor KV/recurrent preservation, valid append and held-token decode pass. Sixteen HTTP malformed cases return 400 and preserve both native slot files. |
| Proxy | All 21 tests pass: explicit/prefix mapping, shared-system divergence and persistence, foreign 404, policy, 409 resync, incompatible recovery atomicity, recovery-GET outage, rollback backup/invalidation, delayed-202 draining, offline/lost-ack/restart replay and 3-client x 4-session interleaving. |
| Real end to end | Two clients x two sessions, streaming fork, foreign 404, offline completion and real prefill-node restart/replay pass through integrated servers. Four independent token-only same-partition references match all 64 observed generated tokens; teacher-forced whole-prompt KL/top1 gates pass below. |

### Numerical results

| Fixture | Same-partition | Whole-prompt mean KL | Whole-prompt top-1 |
|---|---|---|---|
| `prompt-512.tokens.json` | 64/64 | 3.2963e-16 | 64/64 |
| `prompt-2048.tokens.json` | 64/64 | 7.69261e-07 | 64/64 |
| `prompt-8192.tokens.json` | 64/64 | 2.01502e-07 | 64/64 |
| `conversation-a-0.tokens.json` | 64/64 | 8.67437e-15 | 64/64 |
| `conversation-a-1.tokens.json` | 64/64 | 0.00203662 | 63/64 |
| `conversation-b-0.tokens.json` | 64/64 | 8.67437e-15 | 64/64 |
| `conversation-b-1.tokens.json` | 64/64 | 0.00203662 | 63/64 |

The gate is mean KL <= 0.01 nats/token and top-1 >= 95%. Free-running whole-prompt decode matches all 64 tokens in both counting conversations. Both colors conversations first diverge at index 38 (zero-based); this result remains informational under section 11 and does not replace the independent exact or teacher-forced gates.

### Requested bounded append measurements

Fixed context 9216 and fixed 128-token suffix; three fresh-process samples per n0, after all owned servers and other inference gates stopped. Exact native-state comparison with the prepared source runs outside the timed interval.

| n0 | Three samples (ms) | Median (ms) | Full prefix bytes | Recurrent backup bytes | Peak RSS range (bytes) |
|---|---|---|---|---|---|
| 512 | 4.080786, 3.925425, 4.039378 | 4.039378 | 23,556,724 | 20,201,932 | 973,344,768-973,389,824 |
| 2048 | 3.998883, 4.035571, 4.043316 | 4.035571 | 33,620,596 | 20,201,932 | 973,365,248-973,533,184 |
| 8192 | 4.463342, 4.864351, 5.507562 | 4.864351 | 73,876,084 | 20,201,932 | 974,393,344-974,491,648 |

The recurrent backup stays 20,201,932 bytes as the prefix grows from 23,556,724 to 73,876,084 bytes. All nine samples report zero additional peak RSS above the pre-append resident set after resetting the process high-water counter. This does not claim zero allocation: allocator reuse and RSS resolution limit that observation. These bounded correctness measurements on a shared host are not a general latency benchmark. Prior before/after measurements remain in `fixes-r2-261030.md`; no new full-snapshot baseline was rebuilt.

### Retained failures and diagnostic controls

1. `admission-desired-order-01` exited **1**. The 2-X case observed X,Y,X and passed. During setup for 4-X, a `POST /v1/sessions` received 429 despite the 200 ms between-case pause. No 4-X ordering violation was observed in that run. `admission-desired-order-02` reran the identical helper and passed both cases, observing 429,429,X,Y,X for four X requests. The separate setup-release diagnostic permits bounded retries only for setup creates/metadata; concurrent X/Y requests are untouched and all original order assertions remain. It also exited 0, but needed no retry, so it does not retrospectively identify which response held the failed setup permit or measure its release time. The timing cause remains untraced. Both logs and aggregate HTTP suite exit 1 are retained.

2. `cost-prepare-01` exited **1**, `FAIL cannot write fixture`, because the output directory did not exist. No timing sample was taken. After `mkdir -p .temp/hybrid-b-r8-261030/cost`, the identical prepare command exited 0; all nine measured imports passed. The README now explicitly creates its output directory. This documentation-only post-build change alters no compiled code or assertion.

3. `continuation-control-expected-red-01` exited **1 as intended**, `FAIL imported greedy differs from observed continuation`, after mutating only the first observed continuation token in a private fixture. The real delta/prefix were linked read-only. This proves that the equivalence checker rejects a known wrong continuation. Its embedded identity/boundary controls accept valid distributions and reject independent KL-only/top1-only violations.

| Control plant | Gate narrowed | Named failing assertion | Survivor bound |
|---|---|---|---|
| Wrong first continuation | Observed tokens must match imported and independent greedy references | `continuation-control-expected-red-01`: imported greedy differs from observed continuation | Killed, exit 1; does not independently mutate native transfer. |
| KL-only distribution violation | Correct top-1 does not waive KL | Embedded `controls()`: KL-only gate control | Rejected in the positive checker process; no surviving plant in this scope. |
| Top1-only distribution violation | Low KL does not waive agreement | Embedded `controls()`: top1-only gate control | Rejected in the positive checker process; no surviving plant in this scope. |

Helper provenance: `R/adapted-paths.py` differs from the accepted R7 helper only in the binary path; `R/review_admission.py` changes only artifact and integrated fixture paths. Hashes and original paths are in `R/probe-provenance.json`. The setup-release diagnostic and all helpers remain under this task's `.temp/`; no A assertion or production implementation was weakened.

CPU only, Qwen3.5-0.8B-Q4_0.gguf only, SHA-256 `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`. No CUDA/Metal, 27B, 32k/128k, 50-prompt matrix, network shaping or full per-session numerical qualification of the 3-client x 4-session HTTP matrix is claimed. The latter matrix checks isolation and token/hash/state outcomes; the requested 2-client x 2-session run checks numerical equivalence. SWA imports and Anthropic Messages remain unsupported. RAM-only prefill storage and synchronous shadows remain prototype limitations. A second failure while restoring recurrent rollback state still requires resynchronization, as documented in the owner README.

A prior review corrected the startup wording: the HTTP socket may bind before model loading. The model hash is computed once during model initialization, before readiness and before imports are served; the request path uses that value. Historical reports are preserved rather than rewritten.

The brief supplies no attack-surface table or external landing property suite. This report records the explicitly requested gate and finite coverage; it does not claim broader landing acceptance. Only task-owned documentation changed beyond the merge resolutions. No push, PR, model download, shared-golden rewrite or A-workbench edit was performed.

All **14 recorded server PIDs** are absent from `/proc` at 2026-10-30T10:19:44Z; each harness terminated/reaped its own PIDs in `finally`, and `R/cleanup-final.json` records them. The proxy used a harness thread, which was shut down and joined. Minimum guarded configure/build free space: **43,576,999,936 bytes**. Final build tree: **86,412,492 bytes**. Final observed available disk: **42,563,022,848 bytes**. Largest sampled conservative sum of test-tree RSS: **4,277,219,328 bytes** (shared pages counted per process); samples are not a continuous peak measurement. `R/final-summary.json` checks all 34 required green records and explicitly retains the two setup failures and expected-red continuation control.

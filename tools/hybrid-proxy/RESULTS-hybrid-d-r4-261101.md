# Local-kept hybrid call latency, 2026-11-01

Ready for review. Branch: `m3-short-call`; base: `a25754271e19cfda2c44c702a52b752cb53cf623`. No push, GPU execution, model download or engine source change.

## Finding and change

`Proxy.prepare_session()` called `sync()` before checking the policy decision. On an immediate local-kept follow-up this performed a session GET and a shadow POST before dispatching the decoder. A new session instead required POST-create and GET, even with no suffix to send. Heterogeneous mode can add a health GET; it was disabled here. There is no per-call link probe in this path.

The node's `return_delta=false` append returns HTTP 202, but the current `tools/hybrid/prefill.cpp` handler loads state, evaluates the suffix and saves state before responding. Thus a supposedly asynchronous shadow append still blocks the caller for node inference/serialization as well as network RTT. A GET can also queue behind node work.

Move the existing `sync()` into the critical-offload branch. Local-kept calls commit the same token/message journal and leave their suffix for the existing replay worker. Offloads still synchronize and verify the prefix before importing anything. No new worker, queue, acknowledgement interpretation or durability shortcut is introduced.

## Measurement

CPU only on agents1, two threads per engine, both endpoints on loopback, servers inherited `nice -n 10`. Model: existing `Qwen3.5-0.8B-Q4_0.gguf`, SHA-256 `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`. One seed turn establishes 109 evaluated tokens; the measured follow-up adds 44 prompt tokens and samples five tokens. Temperature 0, seed 42, thinking disabled, q8_0 KV, 2048 context, batch/ubatch 256, one decoder slot. The unchanged default policy keeps both turns local.

Three pairs per revision, alternating local/hybrid order. Each member starts with a cold slot. The measured follow-up immediately follows generation, with the previous suffix still pending; controlled replay occurs after the response. This isolates the foreground shadow cost. A separate three-pair run enables the production one-second replay loop.

Existing CPU binaries were copied from C's r4 build, with every copied binary/library checked against its saved SHA-256 receipt. Engine build source `f46c03a3abb3f6d95eff0d79af0efa3bde6265ff` has no diff against this base in src, ggml, common, include, vendor, server, mtmd, hybrid, CMakeLists.txt or cmake. No build slot was consumed. Receipts: `.temp/hybrid-d-r4-261101/engine-identity.json` and each run's `identity.json`. The task-local binary copies were removed after validation; other producers' builds were untouched.

`bench_local.py` adds trace wrappers only inside the benchmark process. The ordinary proxy response format is unchanged. It records the bench's `remote_prepare_ms`, `server_shadow_ms`, native decoder timings, every foreground node RPC, template operations, mutating Store methods and SQL write counts. Raw traces are `before-final/pairs.json` and `after-final/pairs.json` under `.temp/hybrid-d-r4-261101/`; the numeric extraction is `stages.json`.

All times below are milliseconds. Receive is handler entry through JSON/header processing. Template is the union of renderer work and template/tokenizer RPCs. DB time includes serialization, SQL and FULL/WAL commit in methods that actually execute writes. Decoder request spans HTTP dispatch through the complete generation stream. Journal includes generated-token reconciliation and its durable save. These are inclusive timings: DB can overlap template/journal columns, so columns must not be summed. Slot admission, prefix reads and response/lease cleanup remain in the raw trace.

| Revision | Pair | Mode | Receive | Template | DB writes | DB ms | Decoder request | Journal | TTFT | Response total |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Before | 1 | local | 0.05 | 20.62 | 2 | 12.02 | 416.03 | 42.67 | 323.86 | 488.82 |
| Before | 1 | hybrid | 0.04 | 22.35 | 2 | 12.19 | 385.66 | 42.93 | 930.13 | 1084.62 |
| Before | 2 | local | 0.14 | 23.24 | 2 | 15.11 | 365.08 | 51.37 | 294.67 | 461.35 |
| Before | 2 | hybrid | 0.03 | 14.95 | 2 | 11.78 | 403.04 | 55.04 | 907.15 | 1101.51 |
| Before | 3 | local | 0.04 | 20.02 | 2 | 13.29 | 393.01 | 47.03 | 289.69 | 466.17 |
| Before | 3 | hybrid | 0.04 | 21.93 | 2 | 18.99 | 420.02 | 51.79 | 974.69 | 1148.88 |
| After | 1 | local | 0.06 | 17.64 | 2 | 12.71 | 375.70 | 45.29 | 289.68 | 446.13 |
| After | 1 | hybrid | 0.16 | 22.58 | 2 | 22.11 | 370.95 | 55.08 | 286.82 | 454.61 |
| After | 2 | local | 0.05 | 21.93 | 2 | 12.30 | 385.71 | 45.94 | 294.20 | 459.18 |
| After | 2 | hybrid | 0.04 | 33.37 | 2 | 11.28 | 419.85 | 47.50 | 340.86 | 508.57 |
| After | 3 | local | 0.06 | 27.18 | 2 | 26.97 | 432.85 | 69.20 | 323.19 | 538.42 |
| After | 3 | hybrid | 0.07 | 38.62 | 2 | 13.39 | 414.12 | 53.78 | 327.99 | 513.97 |

Every foreground node round trip on the measured follow-up:

| Revision | Pair | local RPCs | hybrid RPCs | GET session | POST shadow | Bench remote_prepare | Bench server_shadow |
|---|---:|---:|---:|---:|---:|---:|---:|
| Before | 1 | 0 | 2 | 1.33 | 626.32 | 627.71 | 614.88 |
| Before | 2 | 0 | 2 | 0.40 | 622.49 | 622.93 | 610.57 |
| Before | 3 | 0 | 2 | 0.71 | 648.46 | 649.22 | 637.40 |
| After | 1 | 0 | 0 | 0 | 0 | <0.001 | 0 |
| After | 2 | 0 | 0 | 0 | 0 | <0.001 | 0 |
| After | 3 | 0 | 0 | 0 | 0 | <0.001 | 0 |

Mean hybrid TTFT: 937.32 -> 318.56 ms; mean local: 302.74 -> 302.36 ms. The mean hybrid-minus-local gap falls from 634.58 to 16.20 ms. Mean remote preparation falls from 633.29 ms to <0.001 ms, while both revisions retain two database writes per measured call. This attributes the large overhead to shadow synchronization rather than journal/database writes or template work. Shared CPU scheduling accounts for visible pair-to-pair noise; three pairs do not establish a latency distribution.

The decoder's own `timings.prompt_ms` and `hybrid_bench.decode_ms` (`predicted_ms`) give the inference portions of the decoder-request interval above:

| Revision | Pair | local prompt_ms | hybrid prompt_ms | local decode_ms | hybrid decode_ms |
|---|---:|---:|---:|---:|---:|
| Before | 1 | 292.78 | 271.09 | 120.10 | 111.07 |
| Before | 2 | 248.93 | 263.28 | 113.75 | 134.87 |
| Before | 3 | 262.61 | 295.08 | 122.52 | 120.38 |
| After | 1 | 261.11 | 255.68 | 111.55 | 111.93 |
| After | 2 | 264.61 | 299.34 | 117.60 | 117.30 |
| After | 3 | 286.30 | 280.27 | 142.88 | 129.40 |

With the actual replay loop enabled after the change, hybrid follow-up TTFTs were 354.98, 326.53 and 307.23 ms. All three pairs retained equal sampled IDs and complete durable token sequences; subsequent node GETs matched the committed token count and prefix hash. These observations do not bound worst-case replay contention.

## Correctness and limits

Local and hybrid sampled IDs and full evaluated token journals match in every native pair, before and after, including the live-worker run. After replay, node count, prefix hash and acknowledged `remote_n` all match the durable local journal. The harness refuses a non-local policy path, inconsistent prefill meters, incomplete stream, token mismatch or unreconciled node state. It stops only its own recorded PIDs and retains cleanup receipts.

Relevant executable checks:

| Command/evidence | Actual exit | Result |
|---|---:|---|
| `python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py'` / `tests-01.log` | 0 | 72 tests; existing concurrency, journal, append, corruption and fallback regressions retained |
| `PYTHONPATH=tools/hybrid-proxy python3 -m unittest test_benchmark.BenchmarkProxyTests test_proxy.HTTPTests.test_delayed_202_shadows_critical_drain_without_token_loss` / `tests-focused-03.log` | 0 | Seven tests with the added assertions at the reviewed source |
| Baseline local-path regression / `regression-before-01.log` | 1 | Expected red: four production/hybrid streaming/non-streaming cases contacted the node |
| `bench-before-03`, `bench-after-02`, `background-01` logs/guard receipts | 0 each | Three native pairs each; actual engine inference, tokens and replay checks |
| Control identity / `identity-control-01.log` | 0 | Six focused tests green without a plant |

The first focused after attempt exited 1 because its corruption case appended only one token: that leaves no importable target after reserving the final prompt token for local logits. The corrected case appends two tokens and exercises the offload/prefix rejection path. Earlier 20-token exploratory and pre-final instrumentation runs are retained, but are not the table's evidence. One scratch runner initially overwrote its mutant index with its guard receipt; the control batch was rerun with separate filenames (`mutants.json`, latest `controls-03-guard.json`). The last control replay covers the reviewed import cleanup in the test file. No failed control is credited as green. Existing fixture ResourceWarnings were emitted by the suite; they were not test failures.

| Mutant | Gate narrowed to | Named failing test | Exit | Survivor bound |
|---|---|---|---:|---|
| Unconditional local sync | Local decisions issue no foreground node request | `BenchmarkProxyTests.test_policy_local_skips_sync_and_preserves_tokens` | 1 | Killed by assertion |
| Disable replay | Durable local suffix must survive restart/outage and reach the node | `BenchmarkProxyTests.test_policy_local_durable_replay_after_restart` | 1 | Killed by assertion |
| Treat shadow acknowledgement as applied | Queued 202 must not advance `remote_n` | `HTTPTests.test_delayed_202_shadows_critical_drain_without_token_loss` | 1 | Killed by assertion |

No surviving plants or setup errors in the three-plant batch. This is bounded regression evidence, not exhaustive mutation coverage. New tests also exercise the next local-to-offload transition and reject a divergent prefix without importing or changing durable tokens. No brief surface table was supplied; no formal catalog-wide coverage is claimed. No rows of the requested local-path contract were treated as out of contract.

A replay already holding the session ticket can still delay the next request, and long-running synchronous node appends are not cancellable here. Deferral also means the next offload can pay for an outstanding suffix. The existing FIFO, skip-busy replay, one-second retry loop and node prefix checks remain unchanged. Prefix divergence remains an explicit recovery error rather than silently overwriting node state. Generalizing replay ownership would need a separate design and tests; this change does not claim to solve that queueing case.

The Python syntax/ASCII/whitespace check and upstream model-naming check exited 0 (`validation-01.log`, `code-style-02.log`); `git diff --check` exited 0. `ty check --exit-zero-on-warning --output-format=github` and `editorconfig-checker` each exited 127 because those executables are unavailable (`type-check-01.log`, `editorconfig-01.log`); these checks are unrun. No hosted checks or independent acceptance are claimed. GPU, 5G, other weights/templates, tools/rewrites at native scale and long-depth performance were not run in this task. Existing unit coverage for those journal transitions was preserved. No full engine rebuild was needed for this Python-only fix.

Disk TMPDIR and `PYTHONDONTWRITEBYTECODE=1` were used. Guarded final before/after/live-worker/control batches observed at least 29.10 GB free and used a 28.4 GB abort margin. Earlier unit/exploratory work had spot checks, not a continuous disk record; the lowest recorded spot check was 29.12 GB. No floor breach was observed. No task-created build tree remains.

## Expected network/GPU effect

For this pending-suffix path, two serialized foreground node RPCs disappear. At 110 ms RTT the expected saving is about 220 ms plus the shadow's node evaluation/state serialization and request upload, minus negligible loopback RPC time. For this CPU fixture, retaining the same node work gives roughly 0.85 s saved (0.633 s measured loopback overhead + 0.220 s RTT). This is an estimate, not a 5G measurement. The cold empty-session path also removes two RPCs, but has no shadow inference cost. A fully synchronized existing session removes one GET, about 110 ms at that RTT; heterogeneous health validation can add another removed local-path GET.

On loopback GPU, the RTT term is small: the saving should track removed node work/scheduler wait. The supplied Glimmer, Gemma 4 and Mirai gaps are 0.69, 0.52 and 0.39 s. If those gaps come from this same shadow path, hybrid should approach their local TTFTs (1.04, 1.13 and 0.36 s), recovering most of the gap. This is a hypothesis for the commands below, not GPU qualification or a promise that the entire 1.5 s end-to-end gap disappears. In-flight replay, different templates, depths and CPU contention can retain other costs.

## Exact rerun commands

From this branch, prepare the unchanged baseline Python modules once. No second model copy, worktree or engine rebuild is needed because the candidate changes no engine code:

```bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
TASK_ROOT="$PWD/.temp/hybrid-d-r4-261101-rerun"
mkdir -p "$TASK_ROOT/tmp" "$TASK_ROOT/before-source"
export TMPDIR="$TASK_ROOT/tmp"
for module in proxy policy store journal concurrency release; do
    git show "a25754271e19cfda2c44c702a52b752cb53cf623:tools/hybrid-proxy/$module.py" > "$TASK_ROOT/before-source/$module.py"
done
```

On agents1, use existing CPU binaries and the existing model. These are the measured commands, with the artifact parent changed to avoid overwriting evidence:

```bash
BIN_DIR=/home/agent/Werkstatt/hybrid-c/llama.cpp/.temp/hybrid-c-r4-261101/build/bin
MODEL=/home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf
nice -n 10 python3 tools/hybrid-proxy/bench_local.py --binary "$BIN_DIR/llama-server" --prefill-binary "$BIN_DIR/llama-hybrid-prefill" --model "$MODEL" --pairs 3 --proxy-source "$TASK_ROOT/before-source" --artifacts "$TASK_ROOT/before" > "$TASK_ROOT/before.log" 2>&1
nice -n 10 python3 tools/hybrid-proxy/bench_local.py --binary "$BIN_DIR/llama-server" --prefill-binary "$BIN_DIR/llama-hybrid-prefill" --model "$MODEL" --pairs 3 --artifacts "$TASK_ROOT/after" > "$TASK_ROOT/after.log" 2>&1
```

On the GPU host, run the preparation block from the checkout containing this candidate, set `BIN_DIR` to the already built CUDA binaries and `MODEL` to an already present Qwen/ChatML GGUF (0.8B or Glimmer/Mirai). Then run these exact commands; no GPU operation below was run on agents1:

```bash
: "${BIN_DIR:?Set BIN_DIR to the existing CUDA build/bin directory}"
: "${MODEL:?Set MODEL to the existing local GGUF path}"
test -x "$BIN_DIR/llama-server"
test -x "$BIN_DIR/llama-hybrid-prefill"
test -f "$MODEL"
nice -n 10 python3 tools/hybrid-proxy/bench_local.py --binary "$BIN_DIR/llama-server" --prefill-binary "$BIN_DIR/llama-hybrid-prefill" --model "$MODEL" --gpu-layers 999 --pairs 3 --proxy-source "$TASK_ROOT/before-source" --artifacts "$TASK_ROOT/gpu-before" > "$TASK_ROOT/gpu-before.log" 2>&1
nice -n 10 python3 tools/hybrid-proxy/bench_local.py --binary "$BIN_DIR/llama-server" --prefill-binary "$BIN_DIR/llama-hybrid-prefill" --model "$MODEL" --gpu-layers 999 --pairs 3 --artifacts "$TASK_ROOT/gpu-after" > "$TASK_ROOT/gpu-after.log" 2>&1
nice -n 10 python3 tools/hybrid-proxy/bench_local.py --binary "$BIN_DIR/llama-server" --prefill-binary "$BIN_DIR/llama-hybrid-prefill" --model "$MODEL" --gpu-layers 999 --pairs 3 --background-replay --artifacts "$TASK_ROOT/gpu-live-replay" > "$TASK_ROOT/gpu-live-replay.log" 2>&1
```

Use binaries from this base or an engine-equivalent revision; the harness records their hashes, not their build provenance. Retain the build receipt and confirm CUDA placement in both server logs. The default framing contract is Qwen/ChatML, as in the measured CPU model; the native Gemma template is outside the existing renderer's supported frames. Reuse the orchestrator's already admitted Gemma smoke/template procedure to qualify that model rather than treating a template rejection here as a latency result. Each artifact directory must be new. Compare `pairs.json` TTFTs, bench fields and `node_rpc` events; retain the token and post-response replay assertions. The harness's 28 GB admission floor is also applied on the GPU host.

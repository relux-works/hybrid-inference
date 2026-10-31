# Hybrid benchmark operator guide

Run from a checkout containing these tools. Python 3.10+ and its standard library are sufficient for scripts. Native targets require Git, CMake and a C++ compiler. Linux CUDA additionally requires the driver, `nvidia-smi` and `nvcc`; macOS requires Xcode's Metal compiler. Install prerequisites using the refusal message, then retry. Scripts never install packages.

Current scope: dense `qwen35` GGUF, one visible GPU, one decode slot, `q8_0` KV and native f32 recurrent state. The prototype limits a node to 16,384 tokens and retained sessions/delta bodies to 512 MiB. The tools enforce both bounds; GPU capacity alone does not remove them. The 50-prompt, 32k/128k cross-backend acceptance matrix in specification section 11 remains future qualification. Do not treat a short smoke run as that acceptance.

## 1. Deploy the prefill node (Linux)

Obtain the same GGUF for both machines. Use a trusted SHA-256 from the model distributor or your own verified copy. Set `MODEL` to its absolute path, and `MODEL_SHA256` to that digest. Keep all runtime state outside tracked files.

```sh
mkdir -p .temp/bench
# Existing CPU build directory can be reused. Omit --cuda for CPU.
tools/hybrid-bench/deploy-prefill-node.sh \
  --state .temp/bench/node --build-lock .temp/bench/build.lock --model "$MODEL" --sha256 "$MODEL_SHA256" --cuda
```

The default ref is the current commit. `--ref COMMIT` builds that revision in an isolated source worktree when needed. Builds use only the named targets and `-j2`, under an advisory lock (`--build-lock PATH`, default `~/Werkstatt/hybrid-models/.build.lock`; its parent must already exist). At least 25 GiB must remain free. CPU is the Linux default; macOS clients select Metal. Do not run two separately sized nodes on a memory-constrained GPU without budgeting both allocations.

Download alternative on the deployment machine:

```sh
tools/hybrid-bench/deploy-prefill-node.sh --state .temp/bench/node --build-lock .temp/bench/build.lock \
  --hf-repo ORGANIZATION/REPOSITORY --hf-file MODEL.gguf --sha256 "$MODEL_SHA256" \
  --model-cache /existing/model/directory --cuda
```

Downloads use temporary files in the cache and publish only after SHA-256 verification. A failed setup removes a newly downloaded model. `--source-url file:///absolute/fixture` or loopback HTTP exercises this path without model downloads. Existing model files are never removed. Local `--model` files are hashed even when `--sha256` is omitted.

`--ctx N` requests a context; omission selects the largest bound that fits the measured available RAM/VRAM, GGUF geometry, 20% margin, 2 GiB compute reserve, host staging and prototype limits. CPU sizing accounts for weights; GPU sizing accounts for weights in VRAM and copies of native state in host RAM. This is conservative allocation arithmetic, not a guarantee against unrelated workloads consuming memory later. Metadata and the exact arithmetic are in `node-info.json`.

Startup prints only the private token file path. Copy `node-info.json` and the token file securely to the client; the token JSON must stay mode 0600. Never paste credentials into commands or returned results. By default the service binds `127.0.0.1:8091`. Remote exposure requires `--bind ADDRESS --public --token-file PRIVATE_JSON`; HTTP traffic is unencrypted. To generate a token file first:

```sh
tools/hybrid-bench/deploy-prefill-node.sh --state .temp/bench/node \
  --generate-token .temp/bench/node-token.json
```

## 2. Open the SSH tunnel (from the client)

```sh
ssh -N -L 8091:127.0.0.1:8091 prefill-host
```

Use your normal SSH host alias and authentication. Keep the tunnel running. No public HTTP listener is needed.

## 3. Set up the client (macOS Metal, Linux CUDA or CPU)

```sh
mkdir -p .temp/bench
chmod 600 /secure/copied/node-token.json
tools/hybrid-bench/setup-client.sh --state .temp/bench/client --build-lock .temp/bench/build.lock \
  --model "$MODEL" --sha256 "$MODEL_SHA256" \
  --node-info /secure/copied/node-info.json --token-file /secure/copied/node-token.json
```

Add `--cuda` on a Linux CUDA client; omit it for Linux CPU. The default prefill URL is `http://127.0.0.1:8091`; override with `--prefill-url URL`. On one machine use the node's own `node-info.json` and token file paths, with distinct ports. The default decode/proxy ports are 8090/8092. `--node-sha256 HASH` can replace `--node-info` at setup; the benchmark still needs `--node-info FILE` for the returned bundle. Both the supplied node identity and the live node must match the client's GGUF.

Setup builds `llama-server` and `llama-hybrid-bench-check`, starts the protected decoder and proxy, and waits for health. The proxy config, credential and PID files are private. `--skip-build` requires an existing matching build record and binary hashes; use it only after a successful build of the same ref/backend. The config includes `node_token_file`, `client_credential` and the existing proxy `clients` mapping; never share this file.

## 4. Smoke run

```sh
python3 tools/hybrid-bench/hybrid_bench.py --client-state .temp/bench/client \
  --scratch .temp/bench --output .temp/bench --label smoke --smoke --check
```

The smoke schedule has two small depth rungs and three small tool results per rung, with four generated tokens per fixed-schedule call. It runs `local`, `hybrid` and `offload` on the same client and seeded input schedule. Typical small-model CPU smoke time is tens of seconds, including one 64-token equivalence check. All native state/check inputs are hashed in `state-dump-hashes.jsonl` and immediately deleted. Native runtime databases and parked slots remain private until the operator removes their stopped state directories.

## 5. Full run

```sh
python3 tools/hybrid-bench/hybrid_bench.py --client-state .temp/bench/client \
  --scratch .temp/bench --output .temp/bench --label fixed --check
python3 tools/hybrid-bench/hybrid_bench.py --client-state .temp/bench/client \
  --scratch .temp/bench --output .temp/bench --label measured --schedule measured --seed 17 --check
```

The fixed ladder is 16,384 / 50,000 / 100,000 / 150,000 / 200,000 / 256,000 tokens. At each rung the conversation grows through synthetic tool results of 512, 2,048 and 10,000 tokens, interspersed with generated assistant messages. A `tool` message represents a synthetic result; no actual tool is executed. The depth row fills toward the rung, capped by both contexts and a generation/check reserve. Actual authoritative token depths are recorded, including chat framing; each tool result continues from the previous result's depth. Oversized subsequent results are skipped explicitly. This prototype therefore returns a partial full-ladder run, rather than claiming that long depths were exercised.

The measured schedule uses seeded piecewise-linear inverse-quantile sampling of aggregate tool-result/output counts at p10/p25/p50/p75/p90/p99; the tails clamp to p10/p99. `hybrid_bench.py` embeds these aggregates and their source hash/provenance from a 2026-10-30 measurement export. It never reads original conversations. Measured smoke limits generated sizes to eight tokens; the full schedule uses the sampled sizes.

Each mode starts a distinct explicit session. A changed generated-token count or depth across modes makes the comparison partial. The client fills KV/recurrent byte estimates from the GGUF. The hybrid policy uses the configured existing proxy cost model; defaults are estimates, not calibration for the current machine. Configure its `policy` rates using measured data before drawing a capacity or purchase conclusion.

`--check-depths N ...` selects the first forced-offload call at or above each chosen depth; defaults are 128 for smoke and 4,096/32,768/131,072 otherwise. The checker recomputes the exact observed prompt on a fresh remote session, imports its [0,N-1) delta and evaluates the held token locally. It compares against local same-partition and whole-prompt contexts over the same K continuation IDs (`--check-tokens`, default 64). Full-vocabulary teacher-forced mean KL must be <=0.01 nats and top-1 agreement >=95%; same-backend same-partition greedy must be K/K. Free-running divergence is informational. A cross-backend result reports the same reference thresholds with `cross_backend_thresholds_calibrated=false`; it never claims calibrated CUDA/Metal acceptance. Threshold failure exits 3 and preserves the numbers.

Plan wall time from smoke measurements, not the card name: `sum(local prompt tokens / measured local prefill rate + generated tokens / decode rate)` plus measured offload/shadow/transfer/import and equivalence costs. A future uncapped 256k ladder processes roughly 0.8M prompt tokens across three modes plus about 4.6k generated tokens. At 100-1,000 prompt tokens/s this is roughly 2.2 hours to 13 minutes before generation, transfer and checks; at 10-50 decode tokens/s add 8-2 minutes. These are planning assumptions, not hardware measurements. The current capped ladder is shorter and exits partial. Do not budget GPU or Mac time from the small-model CPU numbers.

### Memory planning for the 27B GGUF

Illustrative packing estimates: Q4_K_M `27e9 * 4.85/8 = 16,368,750,000` bytes (15.24 GiB), Q8_0 `27e9 * 8.5/8 = 28,687,500,000` bytes (26.72 GiB). Actual GGUF basenames/hashes/sizes decide deployment. The spec estimates 157,000,000 recurrent bytes. KV including q8_0 scales is `16 * 2 * 4 * 256 * 34/32 = 34,816` bytes/token. With capacities C treated as GiB for this table, memory-only depth is `floor((0.8*C*2^30 - weights - 2*2^30 - 157e6)/34816)`, rounded down to 256 and capped at 262,144. Check actual free bytes from `nvidia-smi`; advertising units and other allocations can change the result.

| Card memory (GB class) | Q4_K_M memory-only depth | Q8_0 memory-only depth |
|---:|---:|---:|
| 24 | 55,552 | does not fit |
| 32 | 252,928 | does not fit |
| 48 | 262,144 | 262,144 |
| 80 | 262,144 | 262,144 |
| 96 | 262,144 | 262,144 |
| 180 | 262,144 | 262,144 |

These are memory estimates, not operational context support. For this prototype, the 512 MiB retained-state bound alone reduces this illustrative 27B geometry to about 10k tokens; the launcher computes the exact bound from the actual GGUF. The long-context session-store/transport design must be extended and qualified separately before the larger memory-only depths can run.

## 6. Return results and stop

Send only `hybrid-bench-LABEL-TIME.tar.gz` and its `.sha256` sidecar. The archive contains JSONL, Markdown, sanitized node/client info and `SHA256SUMS`. Verify each member against that manifest after extraction. Never send runtime configs, credentials, database, slots, checker token files or native state.

```sh
tools/hybrid-bench/setup-client.sh --state .temp/bench/client --status
tools/hybrid-bench/setup-client.sh --state .temp/bench/client --stop
tools/hybrid-bench/deploy-prefill-node.sh --state .temp/bench/node --stop
```

Run node control on the node host. PID files include process start identity; stale/reused PIDs are not signalled. Stop the SSH command in its own terminal. Exit codes: benchmark 0 passed, 2 refusal/aborted, 3 equivalence outside reference thresholds, 4 partial/skipped; lifecycle 0 success, 2 refusal, status 3 stopped/missing. A killed/unreachable node aborts the benchmark; forced-offload fallback is a failure, while normal hybrid policy may legitimately choose local.

## Tools and trace contract

| Tool | Purpose and entry point | Artifacts |
|---|---|---|
| Shell launchers / `manage.py` | Deployment/client commands above; `--help` lists options | private state, logs, PID files, node/client info |
| CMake / compiler | Named targets through launchers, locked `-j2` builds | selected build dir, `hybrid-bench-build.json`, executable/library hashes |
| `hybrid_bench.py` | Seeded chat workload, link probe and checks | JSONL, Markdown, archive, SHA-256 sidecar |
| `llama-hybrid-bench-check` | Called by `--check`; standalone usage is printed without arguments | content-free result, temporary private inputs |
| `test_bench.py` | `python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v` | test log if redirected by operator |
| `test_native.py` | `python3 tools/hybrid-bench/test_native.py --help`; disposable live node/client required, kills its node | command ledger, refusal logs and aborted control bundles |

Proxy `benchmark=true` is an operator opt-in. Requests then accept `X-Hybrid-Bench-Mode: local|hybrid|offload`; absent this header, default policy behavior is unchanged. Local mode suppresses remote prefill and shadow replay for its sessions; offload attempts every transferable suffix. With a header, the final SSE object is `{"hybrid_bench":{...}}`, after durable reconciliation and before `[DONE]`. Non-streaming responses carry the same field. No trace includes token IDs, messages, session IDs or credentials. Interrupted streams without committed trace/DONE are failures.

Trace fields include authoritative before/prompt/after depths, new/generated counts, local/remote prompt work, path, critical `server_prefill_ms`, awaited `server_shadow_ms`, `wire_bytes` (delta envelope body only), body-read `transfer_ms`, import HTTP `import_ms` and inclusive `remote_prepare_ms`. HTTP/TCP/TLS/SSH overhead and asynchronous shadows are excluded from the critical byte/time fields. RTT and link download throughput include request overhead. TTFT is client-observed first nonempty streamed content/reasoning; decode rate uses native generation timing. Native prompt counters must equal only new tokens; a lost prior boundary or repeated prefix aborts the run. A telemetry control plant demonstrates that detector without claiming an actual engine regression.

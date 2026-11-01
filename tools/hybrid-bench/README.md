# Hybrid benchmark operator guide

Run from a checkout containing these tools. Python 3.10+ and its standard library are sufficient for scripts. Native targets require Git, CMake and a C++ compiler. Linux CUDA additionally requires the driver, `nvidia-smi` and `nvcc`; macOS requires Xcode's Metal compiler. Install prerequisites using the refusal message, then retry. Scripts never install packages. For a no-download run, pass `--offline --model /local/model.gguf` to both launchers. All launcher builds, including builds without `--offline`, pass `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF`: the benchmark uses no web UI, so disabling it removes the external download and latest-version fallback. `--offline` additionally requires a local model and prevents model downloads.

Current scope: dense `qwen35`, `gemma4` and `muse-glimmer` GGUF, one visible GPU, one decode slot, `q8_0` KV and native f32 recurrent state. The node defaults to a 16,384-token ceiling and 512 MiB retained-session budget; the launcher sets limits derived from the selected context and host memory. Native delta bodies remain bounded to 512 MiB and critical extends are chunked. The 50-prompt, 32k/128k cross-backend acceptance matrix in specification section 11 remains future qualification. Do not treat a short smoke run as that acceptance.

## Sliding-window geometry

Launchers accept `qwen35`, `gemma4` and `muse-glimmer` only. For the two sliding-window architectures, GGUF metadata supplies the layer mask, window size, full/window head dimensions, per-layer KV-head counts and Gemma shared-layer count. Sizing counts allocated KV sources once. `kv_bytes_per_token` counts full layers; `window_state_bytes` reserves the whole window's KV plus conservative cell metadata. `delta_geometry` passes this separately as `window_bytes`, while policy `state_bytes` includes the fixed transfer cost. The critical chunk bound is `(512 MiB - recurrent_bytes - window_bytes - 1 MiB) // (full_kv_bytes_per_token + 24)`, still subject to the existing header bound. Window allocation sizing includes microbatch/pad256 capacity. An impossible fixed window or unknown architecture refuses setup.

Existing benchmark/checker and task-evaluation interfaces remain usable; the checker uses reduced window allocations. Same-partition logit calibration remains the separate PR 9 change. Synthetic coverage and commands for real models are recorded in [RESULTS-hybrid-a-r16-261101.md](RESULTS-hybrid-a-r16-261101.md). Real Gemma 4/Muse Glimmer inference is unverified on this host.

## 1. Deploy the prefill node (Linux)

Obtain the same GGUF for both machines. Use a trusted SHA-256 from the model distributor or your own verified copy. Set `MODEL` to its absolute path, and `MODEL_SHA256` to that digest. Keep all runtime state outside tracked files.

```sh
mkdir -p .temp/bench
# Existing CPU build directory can be reused. Omit --cuda for CPU.
tools/hybrid-bench/deploy-prefill-node.sh \
  --state .temp/bench/node --build-lock .temp/bench/build.lock --model "$MODEL" --sha256 "$MODEL_SHA256" --cuda --bench-link
```

The default ref is the current commit. `--ref COMMIT` builds that revision in an isolated source worktree when needed. Builds use only the named targets and `-j2`, under an advisory lock (`--build-lock PATH`, default `build.lock` beside the state directory; its parent must already exist). At least 25 GiB must remain free. CPU is the Linux default; macOS clients select Metal. Do not run two separately sized nodes on a memory-constrained GPU without budgeting both allocations.

Download alternative on the deployment machine:

```sh
tools/hybrid-bench/deploy-prefill-node.sh --state .temp/bench/node --build-lock .temp/bench/build.lock \
  --hf-repo ORGANIZATION/REPOSITORY --hf-file MODEL.gguf --sha256 "$MODEL_SHA256" \
  --model-cache /existing/model/directory --cuda --bench-link
```

Downloads use temporary files in the cache and publish only after SHA-256 verification. A failed setup removes a newly downloaded model. `--source-url file:///absolute/fixture` or loopback HTTP exercises this path without model downloads. Existing model files are never removed. Local `--model` files are hashed even when `--sha256` is omitted.

`--ctx N` requests a context; omission selects the largest bound that fits the measured available RAM/VRAM, GGUF geometry, 20% margin, 2 GiB compute reserve, host staging and prototype limits. CPU sizing accounts for weights; GPU sizing accounts for weights in VRAM and copies of native state in host RAM. This is conservative allocation arithmetic, not a guarantee against unrelated workloads consuming memory later. Metadata and the exact arithmetic are in `node-info.json`.

Startup prints only the private token file path. Copy `node-info.json` and the token file securely to the client; the token JSON must stay mode 0600. Never paste credentials into commands or returned results. By default the service binds `127.0.0.1:8091`. Remote exposure requires `--bind ADDRESS --public --token-file PRIVATE_JSON`; HTTP traffic is unencrypted. On a public node, `/metrics` requires the bearer token; open health exposes identity and limits but omits activity counters. Authenticated health includes counters, including benchmark release accounting. To generate a token file first:

```sh
tools/hybrid-bench/deploy-prefill-node.sh --state .temp/bench/node \
  --generate-token .temp/bench/node-token.json
```

## 2. Open the SSH tunnel (from the client)

For a client partition experiment, `setup-client.sh --batch 64 --ubatch 64 ...` sets the decoder's logical and physical batch sizes and records them in client-info. Defaults remain 256/256; require `1 <= ubatch <= batch <= 65536`. The native node remains fixed at 256/256 and refuses other launcher values. A second client needs distinct `--decode-port`, `--port` and `--state`; see the [task evaluator](../hybrid-eval/README.md) for the identical-token control and aggregate memory requirements.

```sh
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -L 8091:127.0.0.1:8091 prefill-host
```

Use your normal SSH host alias and authentication. For an unstable link such as 5G, use `ServerAliveInterval 15` and `ServerAliveCountMax 8`, and run this SSH command under a supervisor that reopens the tunnel after it exits (for example, an existing service manager with restart enabled). Keepalives detect a dead connection; they do not reconnect it. A supervisor can restore connectivity for later cleanup attempts, but cannot resume an interrupted benchmark call. No public HTTP listener is needed.

## 3. Set up the client (macOS Metal, Linux CUDA or CPU)

```sh
mkdir -p .temp/bench
chmod 600 /secure/copied/node-token.json
tools/hybrid-bench/setup-client.sh --state .temp/bench/client --build-lock .temp/bench/build.lock \
  --model "$MODEL" --sha256 "$MODEL_SHA256" \
  --node-info /secure/copied/node-info.json --token-file /secure/copied/node-token.json
```

Add `--cuda` on a Linux CUDA client; omit it for Linux CPU. The default prefill URL is `http://127.0.0.1:8091`; override with `--prefill-url URL`. On one machine use the node's own `node-info.json` and token file paths, with distinct ports. The default decode/proxy ports are 8090/8092. `--node-sha256 HASH` can replace `--node-info` at setup; the benchmark still needs `--node-info FILE` for the returned bundle. By default both supplied and live node identities must match the client's exact GGUF SHA-256. To test different quantizations of the same dense Qwen35 base, supply `--hetero-quant` and `--node-info FILE`. The client compares its GGUF-derived version-1 layout fingerprint with node-info and live health, including ordered vocabulary bytes, layer geometry and recurrent geometry. A mismatched fingerprint refuses even with the opt-in. Exact model SHA-256 is required by default. Experimental `--hetero-quant` (`hetero_quant=true` in the proxy) admits a different SHA only after a matching layout fingerprint; the proxy then rewrites the delta envelope SHA, bypassing the native SHA guard. A layout match establishes shapes only, not base-weight identity or numerical equivalence. Quality is unverified and uncalibrated: the r11 CPU Q4_0 node / Q8_0 client measurement gave mean KL 0.0145 nats/token (above the 0.01 reference) and top-1 62/64. The Q8 file was a requantization of Q4 weights, not an independent higher-precision source. Heterogeneous benchmark results remain uncalibrated even when their measured values meet the reference. Both file hashes and quantization types appear in every JSONL row and bundle info file.

The generated proxy config sets `hetero_quant`, `node_model_sha256`, `layout_fingerprint` and `node_layout_fingerprint`. In this opt-in path the proxy validates live node identity before sync and rebinds only each delta envelope's model hash for the client decoder; native body bytes and checksums are unchanged and the decoder still validates its complete native layout. A whole node snapshot is not rebound: ahead-state recovery uses local continuation. This is an experimental accuracy comparison; heterogeneous checks receive uncalibrated status, even on one backend.

Use `--extend-chunk-tokens N` at setup to lower the critical-extend chunk size. Critical chunks use `(512 MiB - recurrent_bytes - 1 MiB) // (kv_bytes_per_token + 24)`, also limited by `(1 MiB - 4096) // 11` for the JSON header. The 24 bytes are native per-cell metadata for one sequence; the 1 MiB reserve covers fixed serialization overhead. The boundary includes this reserve: one more token violates the reserved planning bound, although it can still fit the raw body cap. The launcher writes separate `delta_geometry` values from GGUF metadata; policy cost estimates do not control this safety bound. Without launcher geometry the proxy and benchmark use a conservative 27B bound (34,816 KV bytes/token, 157,000,000 recurrent bytes), safe also for 0.8B (6,528 KV bytes/token, 20,201,472 recurrent bytes). Explicit chunk sizes above the planning bound refuse startup. These are the supported 0.8B/27B geometries; other geometry needs verified `delta_geometry` before use. Each `[n0,n1)` extend has its own delta import and prefix verification. A later failure retains verified earlier chunks for local continuation; a subsequent sync reconciles the remote prefix. Explicit sizes above those bounds refuse proxy startup.

Setup builds `llama-server` and `llama-hybrid-bench-check`, starts the protected decoder and proxy, and waits for health. Each launched process has `--startup-timeout` seconds (default 600) to become healthy. The decoder and the node hash the whole GGUF at load with a portable SHA-256, which takes minutes for a 27B file on some machines; a failed or timed-out start stops only the processes it launched and reaps them. The proxy config, credential and PID files are private. `--skip-build` requires an existing matching build record and binary hashes; use it only after a successful build of the same ref/backend. The config includes `node_token_file`, `client_credential` and the existing proxy `clients` mapping; never share this file.

## 4. Smoke run

```sh
python3 tools/hybrid-bench/hybrid_bench.py --client-state .temp/bench/client \
  --scratch .temp/bench --output .temp/bench --label smoke --smoke --check
```

Enable the optional node download probe with launcher `--bench-link`; without it the benchmark records an explicit skipped link stage. The node does not expose that endpoint by default. Every authenticated extend reports native prefill timing independently of this flag. Skipping this optional probe does not make an otherwise valid run partial. The smoke schedule has two small depth rungs and three small tool results per rung, with four generated tokens per fixed-schedule call. It runs `local`, `hybrid` and `offload` on the same client and seeded input schedule. Typical small-model CPU smoke time is tens of seconds, including one 64-token equivalence check. All native state/check inputs are hashed in `state-dump-hashes.jsonl` and immediately deleted. Native runtime databases and parked slots remain private until the operator removes their stopped state directories.

## 5. Full run

```sh
python3 tools/hybrid-bench/hybrid_bench.py --client-state .temp/bench/client \
  --scratch .temp/bench --output .temp/bench --label fixed --check
python3 tools/hybrid-bench/hybrid_bench.py --client-state .temp/bench/client \
  --scratch .temp/bench --output .temp/bench --label measured --schedule measured --seed 17 --check
```

For a bounded long-context CPU run, use `--smoke --depths 39500 40000 --check-depths 40000 --request-timeout 7200 --check`. `--depths` replaces only the ladder; smoke retains short tool results and four output tokens. Depths must increase strictly within 2..1048576 and remain subject to the actual context bound. Pass `--request-timeout 7200` at client setup as well so the proxy waits for the native node and decoder. Without the option, existing timeouts stay unchanged.

The fixed ladder is 16,384 / 50,000 / 100,000 / 150,000 / 200,000 / 256,000 tokens. At each rung the conversation grows through synthetic tool results of 512, 2,048 and 10,000 tokens, interspersed with generated assistant messages. A `tool` message represents a synthetic result; no actual tool is executed. The depth row fills toward the rung, capped by both contexts and a generation/check reserve. Actual authoritative token depths are recorded, including chat framing; each tool result continues from the previous result's depth. Oversized subsequent results are skipped explicitly. This prototype therefore returns a partial full-ladder run, rather than claiming that long depths were exercised.

The measured schedule uses seeded piecewise-linear inverse-quantile sampling of aggregate tool-result/output counts at p10/p25/p50/p75/p90/p99; the tails clamp to p10/p99. `hybrid_bench.py` embeds these aggregates and their source hash/provenance from a 2026-10-30 measurement export. It never reads original conversations. This sampling does not reproduce the source means: a 200k-draw review simulation found about 1,233 tool tokens and 580 output tokens versus source means 1,077.6 and 511.9 (about 14% and 13% higher). Measured smoke limits generated sizes to eight tokens; the full schedule uses the sampled sizes.

Each mode starts a distinct explicit session and releases its remote session and retired sessions on exit, including failures. Closed sessions cannot be recreated by background replay. Each release trace records live node session, retained-byte and eviction counters after DELETE. Each checker session is deleted in `finally`; normally at most two remote sessions (one mode plus one sequential check) are held by a run. Failed releases can leave more sessions. A changed generated-token count or depth across modes makes the comparison partial. The client fills KV/recurrent byte estimates from the GGUF. The hybrid policy uses the configured existing proxy cost model; defaults are estimates, not calibration for the current machine. Configure its `policy` rates using measured data before drawing a capacity or purchase conclusion.

As soon as a checker returns, its exit code and raw result JSON are saved in a `checker` JSONL row, followed by the validated `equivalence` row, with both flushed to disk before any DELETE or scratch removal. The final bundle includes these rows. A failed release adds a separate `release_error` row with the exception type, sanitized message and node session IDs that may remain. The Markdown and summary report the release failure; the benchmark exits 2 even if equivalence passed (an interrupted run retains exit 130). Original checker exit codes and metrics remain available. Malformed or missing checker output still preserves its exit code/raw output, but cannot produce a validated equivalence row. Cleanup failure also preserves an earlier benchmark exception.

Checker DELETE, client-to-proxy release and proxy-to-node DELETE each make at most three attempts for connection errors, with 0.5 s and 1 s backoff. HTTP refusals, including 401/403/503, are terminal and are not retried; 404 means already released. An error response from the proxy lists unreleased current/retired node IDs. If the proxy cannot be reached, those IDs are unknown and the row explicitly requires node reconciliation.

Checker DELETE uses a 10 s timeout per attempt; client-to-proxy release retains its 30 s timeout so the proxy can release current and retired sessions. Three timed-out attempts plus backoff take about 31.5 s and 91.5 s respectively, excluding scheduling overhead. The proxy's configured node timeout can be longer, so node reconciliation can still be required. Ctrl-C during either DELETE or its retry backoff writes a `KeyboardInterrupt` release-error row before propagating the interrupt. Checker IDs are retained; proxy-owned node IDs remain explicitly unknown if no response arrived. A second interrupt during journal writing or an uncatchable termination can prevent that write.

`--check-depths N ...` selects the first forced-offload call at or above each chosen depth; defaults are 128 for smoke and 4,096/32,768/131,072 otherwise. Two checks run sequentially: `fresh` recomputes the exact observed prompt; `append_chain` replays the run's recorded successful critical boundaries and intervening shadow spans into a new remote session. Both import the resulting sequence of bounded deltas and evaluate the held final token locally. The append replay exercises node save/load and client nonzero-n0 imports; it is a replay of observed tokens/boundaries, not a capture of the measured session's native bytes. The checker compares local same-partition and whole-prompt contexts over the same K continuation IDs (`--check-tokens`, default 64), then measures the client's own batch-boundary noise. The decisive same-partition comparison uses every logit over those checked positions; greedy continuation agreement is also reported.

### Equivalence status and local noise

- Homogeneous same-backend (CPU, CUDA or Metal): compare every full-vocabulary imported logit with an independent local computation using the exact same prefill partition. `same_partition_exact` means bit equality over all K checked positions, rather than greedy-token agreement. A second independent computation of that local partition measures run-to-run reproducibility on the device in use. A bit-identical repeat selects `same_backend_same_partition_exact` and requires bit-identical imported logits. If the repeat differs, `same_backend_same_partition_measured_floor` requires, at each checked position, imported/local KL and maximum absolute logit difference no larger than the corresponding repeat/local values. Positions with an exact repeat still require bit equality. There is no multiplier or epsilon. Both rules also require K/K greedy agreement. CPU additionally retains `numerical_within_bounds` against the whole prompt. Whole-prompt KL, top-1 and free-running divergence remain GPU diagnostics. One observed repeat bounds only this run; it is not a backend-wide tolerance or proof of state equivalence outside the checked positions.
- The checker reports `same_partition_kl_nats` (mean imported || local KL), `same_partition_max_abs_logit_diff`, repeat KL/max difference, repeat exactness, `same_partition_within_floor`, and the per-position measurements. Equal finite bit patterns report KL zero directly, avoiding roundoff in the KL subtraction. The rule uses the loaded device backend and checks the client label. Requested GPU offload without a GPU-capable build/device refuses with exit 2; a CPU build cannot select the GPU rule through its label. Older checkers missing the logit/noise fields refuse with a rebuild message as soon as their result is read, before any equivalence row can enter a bundle.
- Cross-backend or heterogeneous quantization: `uncalibrated_client_noise` always reports `uncalibrated`, exit 5. Hybrid KL is measured against the client's whole prompt, alongside the client's local noise and their ratio. Neither small KL nor same-partition agreement establishes cross-backend acceptance. The previous `uncalibrated_outside_reference` status is replaced by `uncalibrated`; CPU-reference metrics remain visible.

`--noise-chunks 128 255` is the default. Supply one to four sizes in 1..256 to bound cost, for example `--noise-chunks 128`. Each partition is evaluated on the same client backend, model, depth, inputs and context settings. Sizes stay within the existing 256-token batch/ubatch capacity. The checker records the actual decode endpoints of the imported/same-partition chain and of the whole prompt (internally chunked at 256). It reduces a requested size when necessary until its boundaries differ from both references and from earlier noise partitions, and its last boundary before the prompt endpoint differs from the hybrid holdback boundary. Equivalent duplicate candidates are omitted if no further distinct partition exists. The checker refuses when no eligible distinct partition exists; very short inputs may have none.

Each `noise_partitions` entry reports requested/effective chunk size, batch count, `noise_kl_nats` (alternate || whole prompt), `noise_top1_agreed`, `noise_top1_agreement`, and `noise_free_running_first_divergence`. All KL and top-1 comparisons teacher-force the original hybrid continuation IDs; local noise does not use a separately chosen teacher path. Free-running noise compares an independently greedy alternate continuation with the independently greedy whole-prompt continuation; divergence indexes are zero-based, with -1 meaning no divergence within K tokens. This is a bounded measurement of selected partitions, not an upper bound over all batch boundaries or prompts.

Top-level `noise_kl_nats` and `noise_kl_nats_max` are the maximum measured partition KL. Top-level noise agreement is the minimum, and noise divergence is the earliest observed divergence. `kl_to_noise_ratio` is hybrid whole-prompt KL / maximum local noise KL. A zero floor gives JSON null plus `kl_to_noise_ratio_status=both_zero` or `zero_noise_floor`; a positive floor gives `finite`. Both KL values, the ratio, each partition, and `status_rule` appear in JSONL, the Markdown comparison table and the sanitized bundle. Each equivalence status line names the applied rule.

The comparison now retains five contexts: imported, local, independent repeat, whole prompt and free-running whole prompt. They are released before each noise partition runs sequentially with three contexts. The repeat adds one same-partition prefill and K teacher-forced decode steps. Each noise partition adds two prompt prefills and 3K decode steps. Its alternate free-running stream loads a native sequence snapshot instead of repeating the alternate prefill; its first greedy token is retained separately because sequence snapshots do not restore logits. Two default partitions therefore add four prompt prefills, rather than the previous six, plus the independent repeat. Peak context count is five, and one temporary alternate state snapshot is additionally resident during restore. `noise_reference_variant=alternate_state_snapshot` identifies this path.

`noise_reference_seconds` measures the full sequential noise loop, including context creation, prefills, snapshot/restore, numerical comparisons and decode. `same_partition_repeat_seconds` measures the repeated prefill/decode calls, excluding context allocation and comparison overhead. Both appear in JSONL, Markdown and bundles. The r15 section of the [noise report](RESULTS-hybrid-a-r13-261101.md) records actual short CPU times; no 27B/128k GPU wall time has been measured here. Reduce to one partition for expensive depths. The checker loads another full model while the decoder remains resident; budget that weight copy and five contexts (plus the node model when sharing a GPU). Client sizing describes the decoder alone. Runtime threads stay at two compute/two batch threads, batch/ubatch 256, q8_0 KV and f32 recurrent state. No backend kernel or transfer format is changed.

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

These are memory estimates, not qualification evidence. The launcher derives `--max-ctx` from model/card memory and host staging limits, `--ram-budget-mib` for two simultaneous saved sessions, and `--max-request-mib` for worst-case ten-digit token IDs plus JSON framing. It records the divisor, estimates and actual launched node limits in `node-info.json`. Use node launcher `--ram-budget-mib N` to lower the derived retained-session budget for eviction/refusal experiments; it cannot exceed host admission. The uncapped derived budget is also recorded. The native wire body remains 512 MiB; bounded proxy extends avoid sending a whole long context in one delta. Node save/load still copies whole session state, which can dominate long-context time and RAM. 27B at 256k, rented CUDA hosts and CUDA-to-Metal remain target-machine qualification work. The node itself retains its old default 16,384/512 MiB/1 MiB limits when used without launcher overrides.

## 6. Return results and stop

Send only `hybrid-bench-LABEL-TIME.tar.gz` and its `.sha256` sidecar. The archive contains JSONL, Markdown, sanitized node/client info and `SHA256SUMS`. Verify each member against that manifest after extraction. Never send runtime configs, credentials, database, slots, checker token files or native state.

After a release failure, restore the tunnel or run on the node host. List sessions with authenticated `GET /v1/sessions`, then `DELETE /v1/sessions/SESSION_ID` for each leftover session using the same client token. The list is scoped to that client; do not delete another run's active sessions. IDs in `release_error.node_session_ids` can identify the affected sessions. If IDs are unknown, reconcile the list with other active runs first. A lost DELETE response may mean deletion succeeded already; 404 is safe to treat as released. Verify with GET again. Read the token from its private JSON file rather than placing it on a command line. Run the command below first without IDs to list sessions, then repeat with the leftover IDs after the URL:

```sh
python3 - /secure/copied/node-token.json http://127.0.0.1:8091 <<'PY'
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

token = json.load(open(sys.argv[1]))['bench']  # Use this client's configured key.
base = sys.argv[2].rstrip('/')
headers = {'Authorization': 'Bearer ' + token}
def request(path, method='GET'):
    with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers, method=method), timeout=15) as response:
        return response.read()
print(request('/v1/sessions').decode())
for sid in sys.argv[3:]:
    try:
        request('/v1/sessions/' + urllib.parse.quote(sid, safe=''), 'DELETE')
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
print(request('/v1/sessions').decode())
PY
```

```sh
tools/hybrid-bench/setup-client.sh --state .temp/bench/client --status
tools/hybrid-bench/setup-client.sh --state .temp/bench/client --stop
tools/hybrid-bench/deploy-prefill-node.sh --state .temp/bench/node --stop
```

Run node control on the node host. PID files include process start identity; stale/reused PIDs are not signalled. Stop the SSH command in its own terminal. Exit codes: benchmark 0 passed, 2 refusal/aborted, 3 same-partition failure (or CPU legacy numerical failure), 4 partial/skipped, 5 cross-backend or heterogeneous uncalibrated, 130 interrupted; lifecycle 0 success, 2 refusal, status 3 stopped/missing. Startup failure logs survive in a private `<state>.failed-*` directory; the refusal names it. Default builds use the selected source tree's `build/` and a `build.lock` beside state. After valid run metadata is loaded, runtime exceptions and interruptions produce an aborted/interrupted bundle. A killed/unreachable node aborts the benchmark; forced-offload fallback is a failure, while normal hybrid policy may legitimately choose local.

## Tools and trace contract

| Tool | Purpose and entry point | Artifacts |
|---|---|---|
| Shell launchers / `manage.py` | Deployment/client commands above; `--help` lists options | private state, logs, PID files, node/client info |
| CMake / compiler | Named targets through launchers, locked `-j2` builds | selected build dir, `hybrid-bench-build.json`, executable/library hashes |
| `hybrid_bench.py` | Seeded chat workload, link probe and checks | JSONL, Markdown, archive, SHA-256 sidecar |
| `llama-hybrid-bench-check` | Called by `--check`; standalone usage is printed without arguments | content-free result, temporary private inputs |
| `test_bench.py` | `python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v` | test log if redirected by operator |
| `controls.py` | `nice -n 10 python3 tools/hybrid-bench/controls.py --output .temp/cleanup-controls`; requires a new output directory; unchanged source must pass before source-file mutants run in fresh processes | source hashes, commands/PIDs, per-control JSON/logs, `controls.json`; setup errors, skipped tests and wrong assertions receive no kill credit |
| `llama-quantize` | Optional local identity fixture: `build/bin/llama-quantize --allow-requantize INPUT.gguf OUTPUT.gguf Q8_0 2` | explicitly selected model directory; requantizing Q4 weights does not recover source precision |
| `test_integrated.py` | `python3 tools/hybrid-bench/test_integrated.py --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model /local/small-qwen35.gguf --artifacts .temp/integrated-negatives`; CPU noise/corruption, checked fresh/append benchmarks, native defaults, real LRU replay and budget-refusal classification; requires an existing bench checker, no builds | private logs, PID files, snapshots and aborted benchmark bundle |
| `test_native.py` | `python3 tools/hybrid-bench/test_native.py --help`; disposable live node/client required, kills its node | command ledger, refusal logs and aborted control bundles |

Proxy `benchmark=true` is an operator opt-in. Requests then accept `X-Hybrid-Bench-Mode: local|hybrid|offload`; absent this header, default policy behavior is unchanged. Local mode suppresses remote prefill and shadow replay for its sessions; offload attempts every transferable suffix. With a header, the final SSE object is `{"hybrid_bench":{...}}`, after durable reconciliation and before `[DONE]`. Non-streaming responses carry the same field. Call traces omit token IDs, messages, session IDs and credentials. Release-error rows include session IDs for operator cleanup. Interrupted streams without committed trace/DONE are failures.

Trace fields include authoritative before/prompt/after depths, new/generated counts, local/remote prompt work, path, critical `server_prefill_ms`, awaited `server_shadow_ms`, `wire_bytes` (delta envelope body only), body-read `transfer_ms`, import HTTP `import_ms` and inclusive `remote_prepare_ms`, a `chunks` array with boundaries/status/bytes/timings, and a content-free `fallback_reason` (`ram_budget`, HTTP status class, transport or invalid delta). Budget refusal is reported explicitly rather than as generic offload unavailability. HTTP/TCP/TLS/SSH overhead and asynchronous shadows are excluded from the critical byte/time fields. RTT and link download throughput include request overhead. TTFT is client-observed first nonempty streamed content/reasoning; decode rate uses native generation timing. Native prompt counters must equal only new tokens; a lost prior boundary or repeated prefix aborts the run. A telemetry control plant demonstrates that detector without claiming an actual engine regression.

## Local M1 integration evidence

[CPU 0.8B integration run R11](RESULTS-hybrid-a-r11-261101.md) records the pinned A/B merge, native launcher limits, heterogeneous admission, three-mode 40k check, negative controls, exact exits and process limits. It also records the initial UI download deviation; offline launchers prevent that retrieval in later builds.

Short CPU noise/corruption regression (no downloads or builds; use existing binaries):

```sh
nice -n 10 python3 tools/hybrid-bench/test_integrated.py --case noise \
  --server-binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill \
  --checker-binary build/bin/llama-hybrid-bench-check \
  --model /local/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/noise-regression
```

The artifact directory must not already exist. `--baseline-checker /path/to/M1-checker` additionally compares every original CPU metric and exit code with the unchanged M1 executable. Set the library search path to the matching build when running a retained executable. The test retains the one-byte checksum mismatch control (exit 2, no metrics) and adds two bodies with valid recomputed canonical XXH3-128 and SHA-256 trailers: stride 1031/xor 0x40 corruption and stride 4099/xor 0x01 loss after byte 4096. Both must import successfully and fail the same-partition logit gate with exit 3; the lossy case must keep all eight argmax tokens. It reuses the existing proxy test helper and its system libxxhash dependency. A clean delta must pass. CPU-only builds also refuse a GPU request and a false CUDA label with exit 2. `--gpu-available` skips only the unavailable-GPU refusal control when running the suite on a GPU-capable host; this suite still selects CPU computations.

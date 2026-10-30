# Hybrid client proxy prototype

This directory contains the Python client side of the hybrid prefill/decode prototype. Run commands from the repository root. Python 3.10+ and its standard library are sufficient for the proxy. The synthetic stub additionally calls the already installed `libxxhash` through `ctypes`; it does not install a Python package. The real delta reader is shared with producer A in `tools/hybrid`.

## Run the synthetic path

The example contains public fixture credentials, not deployment credentials. Start these three commands in separate terminals:

```sh
mkdir -p .temp/hybrid-b-r1-261030/slots
python3 tools/hybrid-proxy/stub_server.py --role prefill --port 8081 --database .temp/hybrid-b-r1-261030/stub.sqlite3
python3 tools/hybrid-proxy/stub_server.py --role decode --port 8080 --slot-path .temp/hybrid-b-r1-261030/slots
python3 tools/hybrid-proxy/proxy.py --config tools/hybrid-proxy/config.example.json --database .temp/hybrid-b-r1-261030/client.sqlite3 --port 8082
```

```sh
curl --no-buffer http://127.0.0.1:8082/v1/chat/completions -H 'Authorization: Bearer fixture-a' -H 'Content-Type: application/json' -d '{"model":"stub","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

The response includes `X-Hybrid-Session`. Send it on later requests for explicit association. Without it, the proxy compares cumulative SHA-256 token hashes at 256-token boundaries and checks the remaining tokens exactly. A history that diverges before an existing session's end gets a fresh session. An explicit ID owned by another client returns 404. Explicit IDs must contain only ASCII letters, digits, hyphens and underscores, up to 128 characters.

The stub has a clearly labelled synthetic token body. It cannot be imported by the real engine. Its HTTP state-machine and corruption tests do not establish inference equivalence or real prefill-node isolation.

## Run with the engine

Set `model_sha256` to the SHA-256 of the one shared GGUF. Point `prefill_url` at A's prefill node and `decode_url` at the patched llama-server. Replace the fixture credentials in a private configuration file, and restrict its permissions. Increase `prefill_timeout` to cover real CPU prefill during correctness runs (for example, 120 seconds); its default of 0.5 seconds bounds waits when using an unavailable development server. `slot_path` must be the same local directory passed to llama-server as `--slot-save-path`, and `slots` must match its `--parallel` value. Keep context shifting, speculative decoding, automatic idle-slot caching, and RAM prompt caching disabled for this prototype (`--no-context-shift --no-cache-idle-slots --cache-ram 0 --ctx-checkpoints 0`). Text chat and upstream chat tool parsing are forwarded; multimodal inputs and multiple choices are not supported. Anthropic Messages is not implemented.

The decode node must expose `GET /slots/{id}/info` with `n`, `prefix_hash`, and `tokens`, and binary `POST /slots/{id}?action=import_delta`. The import uses A's shared checked reader. The proxy offloads all but the final prompt token, then submits the original chat request locally. That last token produces the client's exact logits; memory serialization alone does not carry logits.

SQLite stores each client's token journal, block hashes, remote acknowledgement position, slot, and last-use time. This contains conversation data; keep the database and slot directory private. Shadow replay runs in a background thread, and also precedes a critical extend. A 202 only acknowledges queuing; GET establishes the applied boundary. After a lost acknowledgement or restart, the boundary and hash determine the remaining suffix, preventing duplicate application. If a RAM-only server lost the session, the proxy creates a replacement remote session and replays the complete durable history. If the server is unavailable, the request uses local prefill. After a restart, the proxy checks slot state and the remote prefix before reuse. Slot eviction saves the least recently used session and restores it on demand; missing parked files fall back to local prefill from durable tokens.

A 409 logs both hashes. A behind server receives journal replay; an ahead server is recovered through its snapshot and the local slot restore action. The restored prefix must match the incoming history. Incompatible model/prefix metadata returns 409 before restore. A snapshot mismatch or journal failure restores the prior decoder snapshot. Recovery GET transport errors and retryable HTTP failures continue locally, then replay on reconnect. Turns currently serialize through one proxy lock, including slot operations. This bounds resource usage and preserves per-session ordering; parallel decode scheduling is future work.

## Policy and trace replay

The policy uses a depth-indexed local prefill table, server rate, link EWMA, fixed overhead, KV bytes/token, recurrent-state bytes, efficiency, margin and hysteresis. The defaults are estimates from the specification; configure measured rates for the target Mac and server. Transfer EWMA measures body bytes after response headers arrive, excluding server prefill time; it is application-level throughput, not a physical link measurement. `policy.always=true` forces an offload attempt when there are new transferable tokens; offline requests still run locally.

```sh
python3 tools/hybrid-proxy/policy.py .temp/turns.jsonl --config .temp/policy.json
```

The trace contains JSON objects with `delta`, `depth`, and optional `server_factor`; the config contains `Policy` fields. Output is JSONL with the selected path and both predicted times. No benchmarks are run on agents1.

## Tests and artifacts

```sh
python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v
python3 -m py_compile tools/hybrid-proxy/*.py
```

The tests use ephemeral localhost ports and temporary SQLite/slot fixtures. They cover policy decisions, explicit and inferred mapping, shared-prefix forks, restart persistence, streaming, offline/lost-ack replay, 409/snapshot recovery, foreign-client 404s on all prefill session operations, 3 clients x 4 interleaved sessions, and atomic synthetic corruption rejection. See `RESULTS.md` for exact run evidence and engine limitations. Runtime databases, slots, diagnostics and test logs belong under `.temp/hybrid-b-r1-261030/`.

Build the four named correctness targets listed below under the shared flock, with CPU-only settings and `-j2`. Check exact free bytes and retain the 25 GiB start/abort floor. Do not start a build while the orchestrator's disk hold is active.

```sh
build/bin/llama-hybrid-check-append ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf ~/Werkstatt/hybrid-models/golden/prompt-512 ~/Werkstatt/hybrid-models/golden/prompt-2048 ~/Werkstatt/hybrid-models/golden/prompt-8192
python3 tools/hybrid-proxy/test_decode.py --binary build/bin/llama-server --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --golden ~/Werkstatt/hybrid-models/golden/prompt-512 --artifacts .temp/hybrid-b-r1-261030/native-http
# Add A's existing prefill binary to exercise the complete real proxy path too:
python3 tools/hybrid-proxy/test_decode.py --binary build/bin/llama-server --prefill-binary /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-hybrid-prefill --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --golden ~/Werkstatt/hybrid-models/golden/prompt-512 --artifacts .temp/hybrid-b-r1-261030/real-e2e
```

The C++ checker compares incremental append with plain local prefill and A's 64-token golden sequence. The HTTP checker independently checks the actual slot routes, both slots' native save bytes after rejection, cross-slot import from A's one-stream source, actual cache reuse, and 64-token equality for the golden prompt (EOS masked consistently on both HTTP paths). With `--prefill-binary`, it also checks the full real proxy path for two clients with two sessions each, captures actual imported deltas, checks both numerical references described below, and exercises streaming forks, foreign-client 404, offline continuation and replay after losing the server RAM store. Native layout probes are bounded to the supplied Qwen35 fixtures. These commands are not passing evidence until `RESULTS.md` records their exit status.

## Review fix round and numerical checks

See [fixes-r2-261030.md](fixes-r2-261030.md) for the current signed fix, exact commands, build evidence and finite coverage, and [fixes-r1-261030.md](fixes-r1-261030.md) for the prior round. Historical build-hold notes in RESULTS.md are superseded by those rounds.

Build targets for the real path now include `llama-server`, `llama-hybrid-check-append`, `llama-hybrid-check-guards` and `llama-hybrid-check-distributions`. All builds use CPU `-j2`, the shared flock and the exact 25 GiB free-space guard. The end-to-end harness uses two proxy slots and a third independent local-reference slot, and stops its servers by PID.

Spec v0.3.4 section 11 distinguishes exact 64-token same-partition equivalence from whole-prompt numerical closeness. The numerical checker imports each actual delta and compares it with two independent token-only local contexts. It teacher-forces the same 64 tokens into imported and whole-prompt contexts, requiring mean KL <=0.01 nats/token and top1 agreement >=95%. Free-running whole-prompt divergence is retained as information. The real harness automatically calls the numerical checker beside `--binary`, or at `--distribution-checker`.

```sh
python3 tools/hybrid/test_recurrent_append.py --library build/bin/libllama.so --malformed
build/bin/llama-hybrid-check-guards ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf ~/Werkstatt/hybrid-models/golden/prompt-512
build/bin/llama-hybrid-check-distributions ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf ~/Werkstatt/hybrid-models/golden/prompt-512.tokens.json ~/Werkstatt/hybrid-models/golden/prompt-2048.tokens.json ~/Werkstatt/hybrid-models/golden/prompt-8192.tokens.json
```

| Tool | Purpose and entry point | Artifacts |
|---|---|---|
| Python standard library | Proxy/stub CLIs and unittest commands above | SQLite, slots and logs under `.temp/` |
| CMake and C++ compiler | CPU engine and the four named targets, under the shared flock | `build/`; monitor/build logs under `.temp/` |
| Native checkers | Append, recurrent rejection, SWA guard and distribution commands above | stdout/stderr logs under `.temp/` |
| Git | Scoped signed local commits (`git commit -S`), no pushes | Local `hybrid-client` history |
| a2a | `a2a post '#hybrid' --topic prototype-go --kind notice --to agents1_hybrid_orch TEXT` | Lane notice; local command logs under `.temp/` |

## Second fix round controls

Build the additional targets `llama-hybrid-check-client` and `llama-hybrid-check-codec` under the same lock and disk guard. The codec controls retain A80013538's valid maximum int32-token header and intended budget failures. The capacity control fills a 512-cell unified KV cache so the valid append fails after validation, then checks both sequences' native bytes and a successful retry after freeing cells.

```sh
build/bin/llama-hybrid-check-codec
build/bin/llama-hybrid-check-client ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --capacity ~/Werkstatt/hybrid-models/golden/prompt-512
python3 tools/hybrid-proxy/test_header_budget.py --binary build/bin/llama-server --prefill-binary /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-hybrid-prefill --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r6-261030/header-http
```

The HTTP header control generates one real 16,384-token state with A's existing binary, checks both header-budget rejections against saved native slot bytes, then imports the valid maximum. It stops both servers by recorded PID. The client checker also supplies the requested bounded append-cost measurement: `--prepare GOLDEN_DIR OUTPUT_DIR` creates three fixed-suffix fixtures, then `--measure STEM N0` measures one append in a fresh process and verifies source equality outside the timing interval. Exact before/after commands, RSS methodology and limits are in the round-2 report. All outputs go under `.temp/`.

If snapshot rollback fails, the proxy returns HTTP 503 and retains `SESSION-resync-backup.bin` in the configured slot directory. This file also marks that binding untrusted across a proxy restart. The next request erases the dirty slot and supplies tokens for a full prefill; the marker is removed after the decoder's evaluated tokens and the durable journal agree. Ordinary prefill-node outages still complete locally and replay on reconnect.

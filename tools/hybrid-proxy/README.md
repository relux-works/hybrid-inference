# Hybrid client proxy prototype

This directory contains the Python client side of the hybrid prefill/decode prototype. Run commands from the repository root. Python 3.10+ and its standard library are sufficient for the proxy. The synthetic stub additionally calls the already installed `libxxhash` through `ctypes`; it does not install a Python package. The real delta reader is shared with producer A in `tools/hybrid`.

## Run the synthetic path

The example contains public fixture credentials, not deployment credentials. Start these three commands in separate terminals:

```sh
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

Set `model_sha256` to the SHA-256 of the one shared GGUF. Point `prefill_url` at A's prefill node and `decode_url` at the patched llama-server. Replace the fixture credentials in a private configuration file, and restrict its permissions. Increase `prefill_timeout` to cover real CPU prefill during correctness runs (for example, 120 seconds); its default of 0.5 seconds bounds waits when using an unavailable development server. `slot_path` must be the same local directory passed to llama-server as `--slot-save-path`, and `slots` must match its `--parallel` value. Keep context shifting and speculative decoding disabled for this prototype. Text chat and upstream chat tool parsing are forwarded; multimodal inputs and multiple choices are not supported. Anthropic Messages is not implemented.

The decode node must expose `GET /slots/{id}/info` with `n`, `prefix_hash`, and `tokens`, and binary `POST /slots/{id}?action=import_delta`. The import uses A's shared checked reader. The proxy offloads all but the final prompt token, then submits the original chat request locally. That last token produces the client's exact logits; memory serialization alone does not carry logits.

SQLite stores each client's token journal, block hashes, remote acknowledgement position, slot, and last-use time. This contains conversation data; keep the database and slot directory private. Shadow replay runs in a background thread, and also precedes a critical extend. A 202 only acknowledges queuing; GET establishes the applied boundary. After a lost acknowledgement or restart, the boundary and hash determine the remaining suffix, preventing duplicate application. If a RAM-only server lost the session, the proxy creates a replacement remote session and replays the complete durable history. If the server is unavailable, the request uses local prefill. After a restart, the proxy checks slot state and the remote prefix before reuse. Slot eviction saves the least recently used session and restores it on demand; missing parked files fall back to local prefill from durable tokens.

A 409 logs both hashes. A behind server receives journal replay; an ahead server is recovered through its snapshot and the local slot restore action. The restored prefix must match the incoming history. An incompatible restored history returns 409 for investigation. Turns currently serialize through one proxy lock, including slot operations. This bounds resource usage and preserves per-session ordering; parallel decode scheduling is future work.

## Policy and trace replay

The policy uses a depth-indexed local prefill table, server rate, link EWMA, fixed overhead, KV bytes/token, recurrent-state bytes, efficiency, margin and hysteresis. The defaults are estimates from the specification; configure measured rates for the target Mac and server. Transfer EWMA includes request/server time and is a conservative effective throughput, not a physical link measurement. `policy.always=true` forces an offload attempt when there are new transferable tokens; offline requests still run locally.

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

After engine builds are admitted, build only `llama-server` and `llama-hybrid-check-append` under the shared flock, with CPU-only settings and `-j2`. Do not start a build while the orchestrator's disk hold is active.

```sh
build/bin/llama-hybrid-check-append ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf ~/Werkstatt/hybrid-models/golden/prompt-512 ~/Werkstatt/hybrid-models/golden/prompt-2048 ~/Werkstatt/hybrid-models/golden/prompt-8192
python3 tools/hybrid-proxy/test_decode.py --binary build/bin/llama-server --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --golden ~/Werkstatt/hybrid-models/golden/prompt-512 --artifacts .temp/hybrid-b-r1-261030/native-http
```

The C++ checker compares incremental append with plain local prefill and A's 64-token golden sequence. The HTTP checker independently checks the actual slot routes, native save bytes after rejection, cross-slot import from A's one-stream source, cache reuse, and 64-token equality with a plain local slot (EOS masked consistently on both HTTP paths). Native layout probes are bounded to the supplied Qwen35 fixtures. These commands are not passing evidence until `RESULTS.md` records their exit status.

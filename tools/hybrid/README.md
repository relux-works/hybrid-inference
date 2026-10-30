# Hybrid CPU prototype

Local prototype on upstream e0f142d5. No upstream submission. Implements the producer-A side of the hybrid lane specification with the orchestrator-approved v1 native-state envelope and one-token decode boundary.

## Tools and commands

Use CMake, a C++17 compiler, Make, Python 3 (standard library only), and `flock`. Vendored xxhash, SHA-256, nlohmann JSON and cpp-httplib provide checksums, JSON and HTTP. No model downloads are required.

```sh
# If cmake is absent, install a task-local tool (30 MB wheel, about 87 MB installed).
python3 -m pip install --no-cache-dir --target .temp/hybrid-a-r1-261030/cmake-tool cmake==4.4.4
CMAKE="$PWD/.temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake"
# This round permits incremental builds only, in the existing configured tree.
# Stop if exact available bytes are below 26843545600 (25 GiB), or a build hold is active.
df -B1 --output=avail /
du -sb build
flock ~/Werkstatt/hybrid-models/.build.lock "$CMAKE" --build build -j2 --target llama-hybrid-prefill llama-hybrid-check
du -sb build
```

Create a mode-0600 JSON credentials file mapping client IDs to distinct bearer tokens (16-256 characters). Keep it under `.temp/`; do not commit it. The node reads this file without logging credentials.

```sh
build/bin/llama-hybrid-prefill ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf .temp/clients.json 8091 8448
build/bin/llama-hybrid-check --codec
# Incremental append equivalence, without rewriting large fixtures:
build/bin/llama-hybrid-check ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf -
build/bin/llama-hybrid-check ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf ~/Werkstatt/hybrid-models/golden
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r3-261030/http
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r3-261030/slow --regression slow-readers
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r3-261030/max --regression max-delta
python3 tools/hybrid/test_recurrent_append.py --library build/bin/libllama.so --malformed
```

The node binds to `127.0.0.1`. Health and metrics are public; session routes require `Authorization: Bearer <token>`. Context size is bounded to 16384. All session IDs are server-generated 32-character hexadecimal strings. Token IDs must be nonnegative and in the loaded vocabulary. An empty extend is rejected. A caller never sees another client's sessions.

`GET /v1/sessions/{id}/snapshot` returns the upstream slot-file format (GGSQ version 4, token IDs, native memory state). Deltas use chunked HTTP/1.1, with the entire envelope prepared before transmission. The RAM store is bounded to 512 MiB and 128 sessions. One CPU context is restored from each session's RAM snapshot for its next extend.

Binary responses (deltas and snapshots) hold a permit until transmission completes or the connection closes. At most two are unfinished per client and three globally; four HTTP workers leave one available for admission and control requests. Excess binary requests return `429` with `Retry-After: 1`, before inference or session mutation. A retry uses the same boundary and tokens. Shadow requests do not consume a stream permit. Each connection serves one HTTP/1.1 request and then closes, so idle keepalive connections do not retain workers. These are the section-5 bounds approved in spec v0.3.3. Slow request headers and broader transport denial of service are outside this regression's claim.

## Native delta contract

`hybrid-state` is the shared CMake library. Include `hybrid-state.h`. `hybrid::encode_delta(header, native_state)` writes an envelope; `hybrid::read_delta(wire, expected_model_sha256, expected_n0)` checks it and returns `.header` and `.state`. Token IDs are available as `result.header.at("tokens").get<std::vector<llama_token>>()`. The import caller must validate the token-derived prefix and native layer layout before admitting a delta to its slot. `hybrid::prefix_hash(tokens)` is SHA-256 of little-endian signed-int32 token IDs.

The wire begins with a 4-byte little-endian JSON byte count (at most `min(1048576, 4096 + 11*(n1-n0))`). JSON carries `magic="HYBD"`, `version=1`, `model_sha256`, `session_id`, `n0`, `n1`, `tokens` (IDs for [n0,n1)), `prefix_hash_n1`, `kv_type="q8_0"`, `state_dtype="f32"`, `body_format="llama-native-seq-v1"`, and `sections=[{name:"llama_seq_state",offset:0,bytes:N}]`. The body is exactly N bytes of pinned native sequence state: 0xaf143cd8 magic, source sequence ID, filtered attention serialization, complete recurrent serialization. Trailer: 16 canonical big-endian XXH3-128 bytes for that section, then 32 raw SHA-256 bytes of the body. Trailing bytes are rejected. The range is validated before computing its budget. The fixed 4096 bytes cover metadata; each nonnegative int32 token needs at most ten digits and one separator. `encode_delta_header(header, state_bytes)` validates this budget without exporting native state; the node preflights it using the largest body-size metadata before inference. Writer and reader enforce the same budget. The framing, fields, single section and checksums are unchanged. When integrating this codec, the decode endpoint must allow 1048576 header bytes in its outer envelope-length cap.

`llama_state_seq_get_data_range(ctx,dst,size,seq,p0,p1,flags)` in `src/llama-ext.h` returns a native range buffer, or its size when `dst == nullptr`. Flags must be zero. Positions satisfy `0 <= p0 < p1`, and `p1` must equal the current sequence endpoint. Historical recurrent endpoints are rejected. Export synchronizes the context and leaves its state unchanged.

The caller implements one-token holdback: send all appended prompt tokens except the final token to the server. The server applies exactly those tokens and returns the corresponding range. After import, evaluate the held token locally to obtain logits, then decode. Shadow that token and all generated tokens back to the server before the next offload. The server does not silently drop a requested token.

## Golden fixtures and evidence limits

`llama-hybrid-check MODEL DIR [DEPTH]` writes deterministic fixtures for prompt lengths 512, 2048 and 8192, or the specified single depth. Each prompt repeats the tokenizer output of the literal prompt in `check.cpp`; the exact IDs and 64 greedy outputs are recorded in `prompt-N.tokens.json`. `n0=N-129`, `n1=N-1`; the final prompt token is held back. Files are `prompt-N.prefix.state` (raw sequence at n0), `prompt-N.delta` (128-token range), `prompt-N.full.delta` (range 0..n1), and the token manifest.

The check restores the prefix into a fresh context, rejects malformed native appends with an unchanged prefix, then appends the range and requires its native state to equal a full source export. After evaluating the held token, it compares 64 greedy tokens against an independent plain local prefill. It also checks stale/empty/negative range rejection and unchanged source state. The append importer is producer B's signed core change, integrated locally. `GOLDEN_DIR=-` suppresses artifact writes. Codec checks are independent of model inference; HTTP tests verify snapshot equality after rejected requests and 3 x 4 session isolation, without claiming decode equivalence for each concurrent session.

Current prototype limits: RAM only, no restart recovery, no NVMe encryption/TTL/eviction, no shadow batching. Admitted requests rotate clients first; within each client, eligible critical requests precede shadows while preserving per-session FIFO. Shadow appends are applied synchronously before returning 202. HTTP/2 and per-layer streaming are deferred by the orchestrator. A disconnected response can leave an accepted append committed; retry observes 409 and must resynchronize.

Task logs and tool installation live under `.temp/hybrid-a-r1-261030/`. No benchmark is run on this host.

Recorded results, negative cases and evidence limits for this run: [hybrid-a-r1-261030-results.md](hybrid-a-r1-261030-results.md). A mode-0600 runtime configuration for three clients is available locally at `.temp/hybrid-a-r1-261030/clients.json`; refer to the file for credentials.

Fix-round findings, controls and build byte accounting: [fixes-r1-261030.md](fixes-r1-261030.md).

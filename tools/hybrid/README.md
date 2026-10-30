# Hybrid CPU prototype

Local prototype on upstream e0f142d5. No upstream submission. Implements the producer-A side of the hybrid lane specification with the orchestrator-approved v1 native-state envelope and one-token decode boundary.

## Tools and commands

Use CMake, a C++17 compiler, Make, Python 3 (standard library only), and `flock`. Vendored xxhash, SHA-256, nlohmann JSON and cpp-httplib provide checksums, JSON and HTTP. No model downloads are required.

```sh
# If cmake is absent, install a task-local tool (30 MB wheel, about 87 MB installed).
python3 -m pip install --no-cache-dir --target .temp/hybrid-a-r1-261030/cmake-tool cmake==4.4.4
CMAKE="$PWD/.temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake"
df -BG /
# Stop if less than 22 GiB is available, or the orchestrator has a build hold.
flock ~/Werkstatt/hybrid-models/.build.lock "$CMAKE" -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF
df -BG /
flock ~/Werkstatt/hybrid-models/.build.lock "$CMAKE" --build build -j2 --target llama-hybrid-prefill llama-hybrid-check
```

Create a mode-0600 JSON credentials file mapping client IDs to distinct bearer tokens (16-256 characters). Keep it under `.temp/`; do not commit it. The node reads this file without logging credentials.

```sh
build/bin/llama-hybrid-prefill ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf .temp/clients.json 8091 8448
build/bin/llama-hybrid-check --codec
build/bin/llama-hybrid-check ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf ~/Werkstatt/hybrid-models/golden
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r1-261030/http
```

The node binds to `127.0.0.1`. Health and metrics are public; session routes require `Authorization: Bearer <token>`. Context size is bounded to 16384. All session IDs are server-generated 32-character hexadecimal strings. Token IDs must be nonnegative and in the loaded vocabulary. An empty extend is rejected. A caller never sees another client's sessions.

`GET /v1/sessions/{id}/snapshot` returns the upstream slot-file format (GGSQ version 4, token IDs, native memory state). Deltas use chunked HTTP/1.1, with the entire envelope prepared before transmission. The RAM store is bounded to 512 MiB and 128 sessions. One CPU context is restored from each session's RAM snapshot for its next extend.

## Native delta contract

`hybrid-state` is the shared CMake library. Include `hybrid-state.h`. `hybrid::encode_delta(header, native_state)` writes an envelope; `hybrid::read_delta(wire, expected_model_sha256, expected_n0)` checks it and returns `.header` and `.state`. The import caller must validate the token-derived prefix and native layer layout before admitting a delta to its slot. `hybrid::prefix_hash(tokens)` is SHA-256 of little-endian signed-int32 token IDs.

The wire begins with a 4-byte little-endian JSON byte count (at most 65536). JSON carries `magic="HYBD"`, `version=1`, `model_sha256`, `session_id`, `n0`, `n1`, `prefix_hash_n1`, `kv_type="q8_0"`, `state_dtype="f32"`, `body_format="llama-native-seq-v1"`, and `sections=[{name:"llama_seq_state",offset:0,bytes:N}]`. The body is exactly N bytes of pinned native sequence state: 0xaf143cd8 magic, source sequence ID, filtered attention serialization, complete recurrent serialization. Trailer: 16 canonical big-endian XXH3-128 bytes for that section, then 32 raw SHA-256 bytes of the body. Trailing bytes are rejected.

`llama_state_seq_get_data_range(ctx,dst,size,seq,p0,p1,flags)` in `src/llama-ext.h` returns a native range buffer, or its size when `dst == nullptr`. Flags must be zero. Positions satisfy `0 <= p0 < p1`, and `p1` must equal the current sequence endpoint. Historical recurrent endpoints are rejected. Export synchronizes the context and leaves its state unchanged.

The caller implements one-token holdback: send all appended prompt tokens except the final token to the server. The server applies exactly those tokens and returns the corresponding range. After import, evaluate the held token locally to obtain logits, then decode. Shadow that token and all generated tokens back to the server before the next offload. The server does not silently drop a requested token.

## Golden fixtures and evidence limits

`llama-hybrid-check MODEL DIR [DEPTH]` writes deterministic fixtures for prompt lengths 512, 2048 and 8192, or the specified single depth. Each prompt repeats the tokenizer output of the literal prompt in `check.cpp`; the exact IDs and 64 greedy outputs are recorded in `prompt-N.tokens.json`. `n0=N-129`, `n1=N-1`; the final prompt token is held back. Files are `prompt-N.prefix.state` (raw sequence at n0), `prompt-N.delta` (128-token range), `prompt-N.full.delta` (range 0..n1), and the token manifest.

The initial producer-A check restores the full-range native state into a fresh context, evaluates the held token in both contexts and compares 64 greedy tokens. It also checks stale/empty/negative range rejection and unchanged source state. Incremental append equivalence requires producer B's append importer and is reported separately until that importer is integrated. Codec checks are independent of model inference; HTTP tests verify snapshot equality after rejected requests and 3 x 4 session isolation, without claiming decode equivalence for each concurrent session.

Current prototype limits: RAM only, no restart recovery, no NVMe encryption/TTL/eviction, no cross-client priority scheduler or shadow batching. Shadow appends are applied synchronously before returning 202. HTTP/2 and per-layer streaming are deferred by the orchestrator. A disconnected response can leave an accepted append committed; retry observes 409 and must resynchronize.

Task logs and tool installation live under `.temp/hybrid-a-r1-261030/`. No benchmark is run on this host.

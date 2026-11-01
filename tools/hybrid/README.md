# Hybrid CPU prototype

For opt-in deployment options (`--host`, `--gpu-layers`, `--bench`), private launchers and CPU-qualified benchmark commands, see [the benchmark guide](../hybrid-bench/README.md). CUDA execution remains unqualified until tested on a GPU host. Existing positional arguments and default CPU/loopback behavior are unchanged.

Local prototype on upstream e0f142d5. No upstream submission. Implements the prefill node, native delta codec and client-side correctness tools of the hybrid lane specification with the orchestrator-approved v1 native-state envelope and one-token decode boundary.

## Tools and commands

Use CMake, a C++17 compiler, Make, Python 3 (standard library only), and `flock`. Vendored xxhash, SHA-256, nlohmann JSON and cpp-httplib provide checksums, JSON and HTTP. No model downloads are required.

```sh
# If cmake is absent, install a task-local tool (30 MB wheel, about 87 MB installed).
python3 -m pip install --no-cache-dir --target .temp/hybrid-a-r1-261030/cmake-tool cmake==4.4.4
CMAKE="$PWD/.temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake"
# Disable both unused UI sources before building; neither downloads nor local UI builds are needed.
flock ~/Werkstatt/hybrid-models/.build.lock "$CMAKE" -S . -B build -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF
# Incremental builds use the existing configured tree.
# Stop if exact available bytes are below 26843545600 (25 GiB), or a build hold is active.
df -B1 --output=avail /
du -sb build
flock ~/Werkstatt/hybrid-models/.build.lock "$CMAKE" --build build -j2 --target llama-hybrid-prefill llama-hybrid-check llama-hybrid-check-append llama-hybrid-check-guards llama-hybrid-check-distributions llama-hybrid-check-client llama-hybrid-check-codec llama-server
du -sb build
```

Create a mode-0600 JSON credentials file mapping client IDs to distinct bearer tokens (16-256 characters). Keep it under `.temp/`; do not commit it. The node reads this file without logging credentials. Every authenticated extend reports `X-Hybrid-Prefill-Ms`, measured around native prefill only; `--bench` enables only the authenticated link probe.

```sh
build/bin/llama-hybrid-prefill ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf .temp/clients.json 8091 8448
build/bin/llama-hybrid-check --codec
# Incremental append equivalence, without rewriting large fixtures:
build/bin/llama-hybrid-check ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf -
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r3-261030/http
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r3-261030/slow --regression slow-readers
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r3-261030/max --regression max-delta
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r6-261030/pending --regression pending
python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r6-261030/errors --regression errors
python3 tools/hybrid/test_recurrent_append.py --library build/bin/libllama.so --malformed
```

The node binds to `127.0.0.1`. Health and metrics are open on the default loopback-only node; public-node activity requires authentication as described below. Session routes require `Authorization: Bearer <token>`. The default context ceiling is 16384. `--max-ctx N` raises the ceiling; with no positional N_CTX it also selects N as the allocated context. With an explicit N_CTX, that allocation must be at least 256 and no larger than the ceiling. The unchanged no-flag allocation is 8448. All session IDs are server-generated 32-character hexadecimal strings. Token IDs must be nonnegative and in the loaded vocabulary. An empty extend is rejected. A caller never sees another client's sessions.

`GET /v1/sessions/{id}/snapshot` returns the upstream slot-file format (GGSQ version 4, token IDs, native memory state). Deltas use chunked HTTP/1.1, with the entire envelope prepared before transmission. The RAM store defaults to 512 MiB total across all clients, configurable with `--ram-budget-mib N`, and at most 128 sessions. The budget counts retained serialized state plus four-byte token IDs; model weights, the live context, empty-state template, allocator overhead and temporary request/state/wire buffers are additional allocations. One context is restored from each session's RAM snapshot for its next extend. An extend that needs space evicts the least recently used idle sessions across all clients. Successful info and snapshot reads refresh recency, as do creates and accepted extends; a monotonic sequence breaks same-second ties. Session responses pin their session until transmission finishes or disconnects, so an active extend or reader cannot be evicted. Queued requests acquire the session when scheduled and can observe an eviction as 404. The proxy recovers by creating a fresh session and replaying its token journal. Explicit DELETE keeps its existing behavior.

Admission plans all victims before removing any. If the new state plus pinned retained states cannot fit, the node returns `503` with `{"error":"ram budget","need":BYTES,"budget":BYTES}` and preserves every session. `need` is the total retained bytes required after reclaiming all eligible idle sessions, including the new candidate state. Rejected work can still allocate temporary buffers and evaluate tokens; this budget is not a process RSS ceiling.

`--max-request-mib N` controls the HTTP body and bounded JSON parser together, default 1 MiB. The byte limit is inclusive; one byte over returns 413 before the handler. Both MiB flags take positive decimal integers; invalid and overflowing arguments refuse startup. The fixed wire caps remain 1 MiB JSON header and 512 MiB native body, independent of these flags. Larger contexts require bounded append deltas; neither flag enlarges the wire envelope.

`GET /v1/health` returns JSON with `max_ctx`, `n_ctx`, `ram_budget_bytes`, `ram_bytes`, `sessions`, `evictions`, `max_request_bytes`, `model_sha256`, `layout_fingerprint` and `model_layout`. `/metrics` retains `hybrid_extends_total` and `hybrid_ram_bytes` and adds `hybrid_max_ctx`, `hybrid_ram_budget_bytes`, `hybrid_sessions`, `hybrid_evictions`, `hybrid_max_request_bytes`, and `hybrid_model_info{model_sha256="...",layout_fingerprint="..."} 1`. Counters are individually atomic; a sample during a commit is not a transactionally consistent multi-counter snapshot. On a public node (`--public` or a non-loopback `--host`), `/metrics` requires a valid bearer token and unauthenticated health omits sessions, ram_bytes and evictions. Authenticated health retains these counters. Loopback-only defaults are unchanged; no response includes file paths or credentials.

The version-1 layout fingerprint hashes compact JSON with lexicographically sorted object keys containing architecture, layer count, embedding size, each layer's query/KV head counts, head dimensions and recurrent flag, SSM convolution/inner/state/rank/group dimensions, vocabulary size and vocabulary hash. The vocabulary hash is SHA-256 over token-ID-ordered UTF-8 vocabulary bytes, each framed by an eight-byte little-endian byte count; embedded NUL bytes are preserved. Weight quantization and file names are excluded. The separate model SHA hashes the entire GGUF. Exact model SHA-256 is required by default. Experimental `--hetero-quant` (`hetero_quant=true` in the proxy) admits a different SHA only after a matching layout fingerprint; the proxy then rewrites the delta envelope SHA, bypassing the native SHA guard. A layout match establishes shapes only, not base-weight identity or numerical equivalence. Quality is unverified and uncalibrated: the r11 CPU Q4_0 node / Q8_0 client measurement gave mean KL 0.0145 nats/token (above the 0.01 reference) and top-1 62/64. The Q8 file was a requantization of Q4 weights, not an independent higher-precision source. Heterogeneous benchmark results remain uncalibrated even when their measured values meet the reference.

Every authenticated session request takes a permit before entering the model scheduler, including shadow extends, snapshots, list, info, create and delete. Session routes retain the model lease so DELETE cannot invalidate a running extend. Health and metrics bypass the lease; metrics read atomic counters. A response owns its permit through queuing, inference and transmission, including JSON errors and shadow acknowledgements. Disconnected queued requests leave the scheduler without changing session state; a disconnect during inference may still leave an accepted append committed. At most two permits are held per client and three globally; four HTTP workers leave one available for admission and control requests. Excess requests return `429` with `Retry-After: 1` before waiting, inference or session mutation. The existing error body remains `{"error":"binary response limit"}` for compatibility. A retry uses the same boundary and tokens. Each connection serves one HTTP/1.1 request and then closes. These are the section-5 bounds approved in spec v0.3.5. Slow request headers and broader transport denial of service are outside this regression's claim.

## Native delta contract

`hybrid-state` is the shared CMake library. Include `hybrid-state.h`. `hybrid::encode_delta(header, native_state)` writes an envelope; `hybrid::read_delta(wire, expected_model_sha256, expected_n0)` checks it and returns `.header` and `.state`. Token IDs are available as `result.header.at("tokens").get<std::vector<llama_token>>()`. The import caller must validate the token-derived prefix and native layer layout before admitting a delta to its slot. `hybrid::prefix_hash(tokens)` is SHA-256 of little-endian signed-int32 token IDs.

The wire begins with a 4-byte little-endian JSON byte count (at most `min(1048576, 4096 + 11*(n1-n0))`). JSON carries `magic="HYBD"`, `version=1`, `model_sha256`, `session_id`, `n0`, `n1`, `tokens` (IDs for [n0,n1)), `prefix_hash_n1`, `kv_type="q8_0"`, `state_dtype="f32"`, `body_format="llama-native-seq-v1"`, and `sections=[{name:"llama_seq_state",offset:0,bytes:N}]`. The body is exactly N bytes of pinned native sequence state: 0xaf143cd8 magic, source sequence ID, filtered attention serialization, complete recurrent serialization. Trailer: 16 canonical big-endian XXH3-128 bytes for that section, then 32 raw SHA-256 bytes of the body. Trailing bytes are rejected. The range is validated before computing its budget. The fixed 4096 bytes cover metadata; each nonnegative int32 token needs at most ten digits and one separator. `encode_delta_header(header, state_bytes)` validates this budget without exporting native state; the node preflights it using the largest body-size metadata before inference. Writer and reader enforce the same budget. The framing, fields, single section and checksums are unchanged. When integrating this codec, the decode endpoint must allow 1048576 header bytes in its outer envelope-length cap.

`llama_state_seq_get_data_range(ctx,dst,size,seq,p0,p1,flags)` in `src/llama-ext.h` returns a native range buffer, or its size when `dst == nullptr`. Flags must be zero. Positions satisfy `0 <= p0 < p1`, and `p1` must equal the current sequence endpoint. Historical recurrent endpoints are rejected. Export synchronizes the context and leaves its state unchanged.

The caller implements one-token holdback: send all appended prompt tokens except the final token to the server. The server applies exactly those tokens and returns the corresponding range. After import, evaluate the held token locally to obtain logits, then decode. Shadow that token and all generated tokens back to the server before the next offload. The server does not silently drop a requested token.

## Golden fixtures and evidence limits

`llama-hybrid-check MODEL DIR [DEPTH] [--max-ctx N]` writes deterministic fixtures for prompt lengths 512, 2048 and 8192, or the specified single depth. Each prompt repeats the tokenizer output of the literal prompt in `check.cpp`; the exact IDs and 64 greedy outputs are recorded in `prompt-N.tokens.json`. `n0=N-129`, `n1=N-1`; the final prompt token is held back. Files are `prompt-N.prefix.state` (raw sequence at n0), `prompt-N.delta` (128-token range), `prompt-N.full.delta` (range 0..n1), and the token manifest.

The check restores the prefix into a fresh context, rejects malformed native appends with an unchanged prefix, then appends the range and requires its native state to equal a full source export. After evaluating the held token, it compares 64 greedy tokens against an independent plain local prefill. It also checks stale/empty/negative range rejection and unchanged source state. The append importer is producer B's signed core change, integrated locally. `GOLDEN_DIR=-` suppresses artifact writes. Codec checks are independent of model inference; HTTP tests verify snapshot equality after rejected requests and 3 x 4 session isolation, without claiming decode equivalence for each concurrent session.

Current prototype limits: RAM-only LRU eviction, no restart recovery, no NVMe tier/encryption/TTL, no shadow batching. [NODE-STATE-DESIGN.md](NODE-STATE-DESIGN.md) proposes resident contexts and host/disk tiers with explicit 27B memory and time estimates. Admitted requests rotate clients first; within each client, eligible critical requests precede shadows while preserving per-session FIFO. Shadow appends are applied synchronously before returning 202. HTTP/2 and per-layer streaming are deferred by the orchestrator. A disconnected response can leave an accepted append committed; retry observes 409 and must resynchronize.

The retained CMake tool lives under `.temp/hybrid-a-r1-261030/`. M1 CPU benchmark and correctness evidence is in [the r12 report](../hybrid-bench/RESULTS-hybrid-a-r12-261101.md); target-GPU performance remains unqualified.

Recorded results, negative cases and evidence limits for this run: [hybrid-a-r1-261030-results.md](hybrid-a-r1-261030-results.md). A mode-0600 runtime configuration for three clients is available locally at `.temp/hybrid-a-r1-261030/clients.json`; refer to the file for credentials.

Fix-round findings, controls and build byte accounting: [fixes-r1-261030.md](fixes-r1-261030.md).

Admission-before-waiting fix, pending-request regressions and exact validation commands: [fixes-r2-261030.md](fixes-r2-261030.md).

## Client import tools

The client proxy, stub, native HTTP importer and end-to-end commands are documented in [../hybrid-proxy/README.md](../hybrid-proxy/README.md). Their artifacts belong under `.temp/`. All tools below use the same CPU model and integrated `hybrid-state` codec.

```sh
build/bin/llama-hybrid-check-append ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf ~/Werkstatt/hybrid-models/golden/prompt-512 ~/Werkstatt/hybrid-models/golden/prompt-2048 ~/Werkstatt/hybrid-models/golden/prompt-8192
build/bin/llama-hybrid-check-guards ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf ~/Werkstatt/hybrid-models/golden/prompt-512
build/bin/llama-hybrid-check-distributions ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf ~/Werkstatt/hybrid-models/golden/prompt-512.tokens.json ~/Werkstatt/hybrid-models/golden/prompt-2048.tokens.json ~/Werkstatt/hybrid-models/golden/prompt-8192.tokens.json
build/bin/llama-hybrid-check-client ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --capacity ~/Werkstatt/hybrid-models/golden/prompt-512
build/bin/llama-hybrid-check-codec
# Prepare fixtures for the bounded append cost/RSS control; no shared fixtures are rewritten.
mkdir -p .temp/append-cost
build/bin/llama-hybrid-check-client ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --prepare ~/Werkstatt/hybrid-models/golden .temp/append-cost
build/bin/llama-hybrid-check-client ~/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --measure .temp/append-cost/cost-512 512
```

`check-append` compares 64 greedy tokens after import to an independent local prefill at the same partition. `check-guards` tests portable single-sequence restore and sliding-window rejection. `check-distributions` compares full-vocabulary teacher-forced probabilities against whole-prompt prefill. `check-client --capacity` tests rollback after validated input exhausts free KV cells; `--measure` reports one append and checks its resulting state outside the measured interval. `check-codec` covers the maximum token range and both header budgets.

## Configurable-limit qualification

Use a local small hybrid GGUF; no test downloads models. State dumps and private logs must stay in the chosen `.temp/` directory. The deep fixture is an explicit opt-in CPU test and can take substantially longer than the small regressions. For the identity control, add `llama-quantize` to the named targets in the locked incremental build above, then create a local Q8_0 copy with the command below. Re-quantization changes the file and storage precision; it does not recover information lost by the original quantization. The output stays at the selected local model-cache path.

```sh
MODEL=/path/to/small-hybrid.gguf
REQUANTIZED=/path/to/local-requantized.gguf
ARTIFACTS=.temp/node-limits
build/bin/llama-quantize --allow-requantize "$MODEL" "$REQUANTIZED" Q8_0 2
build/bin/llama-hybrid-prefill "$MODEL" .temp/clients.json 8091 --max-ctx 65536 --ram-budget-mib 1024 --max-request-mib 2
build/bin/llama-hybrid-check-codec --max-ctx 65536
build/bin/llama-hybrid-check-codec --max-ctx 262144
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL" --artifacts "$ARTIFACTS/limits"
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL" --requantized "$REQUANTIZED" --artifacts "$ARTIFACTS/identity" --case identity
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL" --artifacts "$ARTIFACTS/deep" --case deep
build/bin/llama-hybrid-check-distributions "$MODEL" "$ARTIFACTS/deep/deep.tokens.json" --max-ctx 65536
```

`test_limits.py` checks LRU replay, atomic impossible-budget rejection, concurrent snapshot/extend pinning, all-pinned refusal and disconnect release, inclusive request limits, invalid CLI arguments, and metrics. `--case identity` compares two actual local quantizations. `--case deep` writes the node's 39999-token delta plus a 40000-token prompt; the distribution checker imports that exact delta, checks independent same-partition 64/64 and teacher-forced whole-prompt KL/top-1, and reports free-running divergence. Existing golden continuations remain checked when present; a new HTTP fixture obtains its continuation from the imported context and compares it with independently computed references. Header fixtures accept an exact range/absolute byte budget and reject one byte over in both writer and reader. Long contexts use the largest bounded worst-case-ID chunk rather than assuming a whole context fits one header.

Results and finite limits for the node-limits run are in [RESULTS-hybrid-a-r10-261031.md](RESULTS-hybrid-a-r10-261031.md). Hash and remove native state dumps after validation; keep only content-free numbers, manifests and logs for review.

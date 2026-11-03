# Recurrent state dtype on the wire

Run hybrid-a-r38-261103, based on accepted RULER head 18d3f114e46c4f801c4a09e2c6cf8ba2303ae42d. This is the private hybrid-inference prototype. The engine's recurrent tensors remain f32. Attention KV remains q8_0. fp16 is not implemented.

## Routes and negotiation

| Route | Serialization or import |
|---|---|
| Node `POST /v1/sessions/{id}/extend`, `return_delta=true` | `tools/hybrid/prefill.cpp` calls `hybrid::export_range`, then `encode_delta`. Omitted `state_dtype` means f32; bf16 is explicit. Unsupported dtype/model combinations refuse before prefill. |
| Chunked critical transfer | `tools/hybrid-proxy/proxy.py` sends the configured dtype on bf16 extends, checks the returned envelope, and sends `X-Hybrid-State-Dtype` to the decoder. The default f32 extend request retains its old shape. |
| Non-final chunk | `fixed_state=false`; q8_0 KV only. bf16 does not change its body. All chunks in one transfer must agree on dtype. |
| Final or single chunk | `fixed_state=true`; full KV delta followed by one recurrent cell. Only its recurrent tensor payloads shrink. |
| Client `POST /slots/{slot}?action=import_delta` | `tools/server/server-context.cpp` passes the expected dtype (header default f32) to `delta_import::stage`. Mismatch refuses before mutation. bf16 bytes expand to f32 host bytes before native validation, staging and transactional application. |
| Native benchmark and scorer | `tools/hybrid-bench/check.cpp` and `score.cpp` pass the fixture's expected dtype into the same staging path. `read_delta` returns expanded native bytes; `read_delta_view` intentionally returns the original wire dtype. |
| Offline state, node snapshots, rollback backups | Stay native f32. `GET .../snapshot`, `save_state`, slot save/restore and native range exports have no compression option. They are not bf16 transfer envelopes. Their origin must travel with checkpoint fixtures. |
| Shadow/catch-up extend, `return_delta=false` | No state transfer. Node compute and retained state stay f32. |
| vLLM bridge | Not changed or qualified by this task. It emits f32; it does not advertise bf16 state transport. |

Node health advertises `state_dtypes` (Qwen35: f32/bf16; SWA models: f32 only). Configure the proxy with `state_dtype: "bf16"`, or use `setup-client.sh --state-dtype bf16`. The launcher and proxy refuse a missing bf16 capability. The envelope, configured expectation and native import request must agree. Old f32-only nodes remain compatible with the default. An old client refuses a bf16 envelope. The Qwen35-only codec validates its native shape; it does not guess another architecture's layout.

A committed transfer appends `{n0,n1,state_dtype,lossy}` to the session's persisted `state_imports`. The message journal retains that history and latest dtype. Benchmark traces include dtype/lossy flags, and each staged/imported chunk names its dtype. This history also exists for token-only sessions in the session database. Existing SQLite databases migrate with f32 and empty-history defaults. Staging alone is not a committed import.

## Envelope and native body

The existing version-2 envelope stays `u32le JSON_length | ordered JSON | body | canonical XXH3-128(body),16B | SHA256(body),32B`. Checksums bind compressed bytes. The JSON keeps `body_format=llama-native-chunks-v3`, `kv_type=q8_0`, and one `llama_seq_state` section at offset zero. `state_dtype` is f32 or bf16; section bytes describe the transmitted body. Default f32 fields, order, body and checksum bytes are unchanged.

The native format is host ABI serialization, qualified here on little-endian Linux. The bf16 codec reads little-endian scalar fields and targets the existing Qwen35 layout; portability to another ABI is not claimed. At sequence zero, one stream, no transposition:

1. Native magic u32 and source sequence i32 (8B), stream count u32, KV cell count u32 (8B).
2. Each attention cell: position i32, sequence count u32 (must be one), M-RoPE x/y/token metadata (12B), sequence ID i32: 24B/token.
3. Attention data: transpose/layer-count/rotation metadata (16B); all K rows, then all V rows. Each row has ggml type i32 and native row bytes u64 (12B), followed by `cell_count * row_bytes`. These bytes stay q8_0.
4. Final state only: recurrent cell count u32 (one), position i32, sequence count u32 (zero), transpose u32 (zero), model layer count u32: 20B.
5. All recurrent R rows, then all recurrent S rows. Each has the original f32 type i32 and original f32 row bytes u64 (12B), then its payload. In bf16, each four-byte f32 value becomes a two-byte little-endian bf16 word. Row metadata deliberately retains expanded f32 lengths; the envelope dtype determines the payload width. The client restores the original native representation before import.

The converter bounds counts, row lengths and expanded size before copying/allocating, requires exactly the expected recurrent row count, and rejects short/odd or trailing data. Native shape/token validation still runs before any tensor write. The f32 path does not enter the converter. Limits stay conservative at their existing f32 chunk geometry; compression does not expand the admitted context or request size.

## Rounding and exactness

The codec uses the existing scalar `ggml_fp32_to_bf16` conversion, which preserves subnormals until rounding (the vector row routine can flush them on AVX512BF16). For finite values its integer RNE formula is `(bits + 0x7fff + ((bits >> 16) & 1)) >> 16`. Infinities and signed zero are preserved. NaNs remain NaNs, are quieted, retain sign and available upper payload bits; low payload bits are lost. f32 subnormals can round to zero. Finite f32 values at the top of the exponent range can round to infinity, as required by bf16 RNE. The relative-error bound `<= 2^-8` applies to normal inputs whose rounded result is finite; a universal finite relative-error bound at overflow would contradict RNE. The CPU real-state test verifies that bound across every normal recurrent value in its fixture. Synthetic tests cover exponent boundaries, ties, zero, NaN, Inf, underflow and overflow.

bf16 origin is always lossy, even when a particular rounded value or measured logit happens to be unchanged. Same-partition checks return `lossy`, exit 5, and cannot pass an exact import gate. Growing-session controls include a false `state_transport_exact` condition for bf16 and likewise return lossy/5. Numerical equality is retained separately as `same_partition_logits_exact` or `logits_exact`; raw per-position metrics are never fabricated to fit a verdict. f32 behavior and the heterogeneous rules remain unchanged. None of the CPU measurements qualifies long-context GPU quality.

## Qwen3.8-27B bytes at prefix depth 100000

Let `D` be the transferred suffix length and `S=156893184` bytes of f32 recurrent tensors. Payload is `34816*D + S` in f32 and `34816*D + S/2` in bf16. Prefix depth does not enter the delta payload. The native body also carries `24*D + 1588` bytes: 416B attention headers plus 1172B recurrent metadata. Full binary wire bytes are `body + JSON_length + 52`; HTTP/TCP/TLS framing is excluded.

| D | f32 tensor payload | bf16 tensor payload | f32 native body | bf16 native body |
|---:|---:|---:|---:|---:|
| 512 | 174718976 | 96272384 | 174732852 | 96286260 |
| 2343 | 238467072 | 160020480 | 238524892 | 160078300 |
| 10000 | 505053184 | 426606592 | 505294772 | 426848180 |

For a reproducible complete-envelope example, use 32-character session/transfer IDs, 64-character digests, every suffix token ID 100, one final chunk and compact JSON. These are arithmetic examples, not measured 27B transfers:

| D | f32 JSON bytes | bf16 JSON bytes | f32 binary wire | bf16 binary wire |
|---:|---:|---:|---:|---:|
| 512 | 2612 | 2612 | 174735516 | 96288924 |
| 2343 | 9936 | 9937 | 238534880 | 160088289 |
| 10000 | 40564 | 40565 | 505335388 | 426888797 |

Real header length depends on token digit counts and identifiers. Tensor savings are exactly 78446592B per offload; the final envelope difference can be one byte smaller because `bf16` is one character longer. Chunking repeats KV headers but ships recurrent state only once, in the final chunk.

GPU procedures: [KL study](../hybrid-bench/STATE-BF16-RUNBOOK.md) and [paired RULER arm](../hybrid-eval/RULER-BF16-RUNBOOK.md). CPU commands and finite limits: [r38 results](RESULTS-hybrid-a-r38-261103.md).

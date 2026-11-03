# r46b timing instrumentation

Enable `--timing-spans` on the growing driver and `LLAMA_HYBRID_TRACE=/absolute/private/native-%p.jsonl` before launching the node/client. Each process needs a new path; existing native files refuse. The driver sets its decoder trace under OUTPUT automatically. The launcher's calibration decoder uses the inherited path. Keep calibration and timing files separate. Native tracing is opt-in, with fixed memory buffers and an asynchronous writer: measured threads only read steady clocks, copy bounded metadata and enqueue records. No trace serialization, disk flush, extra backend synchronization or capture is added to the measured thread. Existing backend synchronizations, checksum passes, imports and rollback saves are timed where they already occur. Post-run diagnostic checkpoints drain the writer; dropped records or failed writes refuse qualification of the trace. This is instrumented production-path timing, with clock/enqueue overhead still present.

Each span has an ID, parent ID/domain, call/session/transfer/chunk IDs, n0/n1, start/end/duration_ns, bytes/tokens and critical/background origin. Python `monotonic_ns` and native `steady_clock` domains include host/PID. Only same-domain parents assert interval nesting. Remote parents are causal links and cannot be subtracted by clock offset. Asynchronous native task/stream roots retain request_id/domain, rather than pretending to fit inside an already-ended dispatch span. Batching attribution is supported for this packet's single active slot; a mixed multi-slot batch is not attributed to one call.

| Boundary | Span/field and interpretation |
|---|---|
| Stream | bench call, stream_start, headers_received, first_response_body_read observation, first_nonempty_delta, stream_end, post_call_verification_end. JSON encoding precedes stream TTFT. |
| Proxy | chat render/tokenize/layout/append, slot lease/info, prepare/sync, JSON encode, HTTP open/upload/headers, response body read/JSON decode. `transfer_ms` still starts after headers. |
| Node | header-receipt/body-complete observation, scheduler queue, full-prefix load_state, prefill, full-prefix save_state, export sizing/range/tensor-to-host and existing backend synchronization. |
| Wire | SHA256/XXH3 init, per-block update and final spans carry bytes/backend. Existing fused checksum loop remains fused; no new pass. Envelope/header/body copies and response writes have separate spans. |
| Import | header parse, loopback upload, staged native parse/checksum/prefix/validation, metadata backup, tensor-to-device copies, final apply. Old-slot save, tensor read, file write/close-flush are separate nested spans. Existing final import_ms already includes old-slot save. |
| Post import | info/token/hash readback, journal SQLite commit, abort_delta and unlink are outside import_ms. |
| Catch-up | critical/background sync and per-extend spans carry counts and causal IDs; timestamps identify actual overlap. The whole catchup_seconds field is never added to TTFT. |
| Decoder | task queue/execute, prompt batch/one-token eval, existing backend sync, first sample, detokenization, SSE write/wait. Native first sample is distinct from nonempty visible text. |
| Fidelity-only evidence | snapshot download/open, write/flush/fsync, hash, delta write/hash/chmod; instrumentation=true. Timing-only runs record zero capture counts. |

"Receipt" is an application observation at an existing read/write, not an exact kernel socket-byte arrival. HTTP upload and header wait share an urllib open span; native header/body observations help bound it without extra probes. Server first-body markers bracket the existing sink write. The bench records the first existing SSE line read and the first nonempty content/reasoning delta separately. Do not interpret an empty SSE/header or detokenized delay as the first sampled token.

The Python TTFT account clips leaves to [stream_start, first_nonempty_delta], takes their interval union, and records an explicit residual. Parent and child durations are never added together; overlapping leaves are counted once. Remote native trees remain separate. Residual includes uninstrumented parent self work, transport gaps and observer overhead, not an unexplained adjustment subtracted to manufacture production TTFT. Native phase totals cover the whole call, including output/reconciliation, and are labelled non-additive.

After node/client cleanup and post-run trace checkpoints, copy the private node JSONL beside the run and extract offline:

```bash
python3 tools/hybrid-bench/packets/r46/extract_timing.py --trace "$RUN/trace.json" --native "$NODE_TRACE" "$CLIENT_TRACE" --output "$RUN/timing-extract.json"
```

Pass one complete file per native process. If a node was shared with calibration, its earlier spans remain in that file and are selected by call ID. The reader rejects missing checkpoints, loss, malformed/truncated JSON, duplicate IDs, missing parents, bad nesting and changed persisted TTFT reconciliation. It preserves `local_tail_seconds`, `meter.local_prompt_ms`, `reconciliation_seconds`, total import, stage sum and final save/apply request. Raw spans remain the detailed evidence; the summary does not replace them. Retain binary/build/model/priority/batch/link evidence from the launcher, supervisor and driver; runtime trace does not rehash executables in TTFT.

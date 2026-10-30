# Producer B first-run evidence

Run: hybrid-b-r1-261030. Base: e0f142d5ddc1461ce721644d92af33db23545105. Branch: hybrid-client. Spec v0.3 was verified against SHA-256 e5858388ab249646c8c65df4d6b4f4c6009fc5ffd30c8f766cc6052b3cf0936b. Orchestrator decisions in #hybrid/prototype-go subsequently fixed the single native section, canonical checksum bytes, token IDs in the header, HTTP/1.1, and the one-token logits boundary.

## Proxy validation

- `python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v`: exit 0, 10 tests, 8.064 seconds. Log: `.temp/hybrid-b-r1-261030/proxy-tests-01.log`.
- `python3 -m py_compile tools/hybrid-proxy/*.py`: exit 0. Log: `.temp/hybrid-b-r1-261030/pycompile-03.log`.
- `git diff --check`: exit 0. Log: `.temp/hybrid-b-r1-261030/diff-check-01.log`.

Tests: streaming HTTP; shared 600-token system prefix then divergent conversations; exact prefix extension; explicit IDs and foreign-client 404; SQLite restart; offline local decode and durable replay; lost acknowledgement after durable server application; 409 replay and ahead-server snapshot restore; server-side 404 on foreign read/extend/snapshot/delete; 3 clients x 4 sessions with slot parking and mixed offload/local policy. Synthetic rejection cases cover wrong model, n0 gap/overlap, truncation, section and body checksum corruption, prefix mismatch and layout mismatch, with the fixture slot unchanged and a nearby valid append accepted.

These tests use a synthetic decode node. They do not prove native-memory atomicity, A's real server isolation, or 64-token engine equivalence. No model inference or performance run is claimed here.

## Environment and limits

Python 3.14.4; SQLite 3.46.1; installed system libxxhash.so.0 for the fixture checksum. No Python dependencies installed. CMake was absent on PATH at startup. No engine build or configure has run in B's checkout: the orchestrator explicitly held B builds and then all new builds/configures for disk headroom. A's golden directory was initially empty. Native append, server import and CPU equivalence evidence will be recorded separately after integration and build admission.

There is no brief surface table or landing property suite. This is a first-build private prototype under the explicit no-review-gate exception. Cross-review and orchestrator review remain future gates; no upstream submission or push is authorized.

## Source integration and additional probes

A's signed reader/range commits 30b992f3a9 and d8fe1bcddc were fetched locally and integrated by signed merge 76a230f647. B core append is signed commit 684e35a5b8 (source-only when committed). The C++ API uses explicit n0/n1, requires a contiguous attention suffix, replaces recurrent state, checks both hybrid endpoints, and restores a serialized destination backup on failure. Sliding-window append is explicitly unsupported so failed allocation cannot evict another session's masked cells.

`c++ -std=c++17 -Iinclude -Iggml/include -Isrc -Icommon -Ivendor -Ivendor/nlohmann -Itools/mtmd -Itools/hybrid -Itools/server -fsyntax-only src/llama-context.cpp src/llama-kv-cache.cpp src/llama-memory-hybrid.cpp tools/server/server-context.cpp tools/server/server.cpp tools/hybrid/check-append.cpp` exited 0. Log: `.temp/hybrid-b-r1-261030/cxx-syntax-03.log`. The first syntax check exited 1 because a new server field was placed in the slot instead of the context implementation; the field was moved, and subsequent checks passed. No compilation/link/runtime success is inferred from syntax-only success.

Two further synthetic HTTP tests exercise authentication/unsafe-ID rejection and full journal replay after the prefill server loses all RAM sessions. The updated 12-test suite passed in 10.548 seconds (exit 0, `.temp/hybrid-b-r1-261030/proxy-tests-02.log`). A subsequent bounded retry guard prevents unbounded recursion when a faulty server repeatedly creates IDs that cannot be read.

Native test sources now include `llama-hybrid-check-append` (golden prefix+delta append, fresh-slot rollback, malformed native stream/layout/recurrent endpoint, 64-token local/golden equivalence) and opt-in `tools/hybrid-proxy/test_decode.py` (real slot HTTP rejection, saved-state byte preservation, two-slot cache reuse and 64-token local equivalence). They remain unrun until binary build admission. No pre-existing reviewer tests were removed. This run adds engine/server changes outside the proxy as explicitly required by brief scope 2; it is not a rework.

The bounded-retry candidate was rechecked: the same 12-test command exited 0 in 10.328 seconds (`.temp/hybrid-b-r1-261030/proxy-tests-03.log`); Python compile and diff checks exited 0 (`pycompile-05.log`, `diff-check-02.log`).

Source review follow-up: single-sequence snapshots may contain a different source stream count from the decode node's parallel-slot count. The native reader now accepts one populated source stream for a single-sequence restore while preserving strict counts for full-context/mirrored restores. The HTTP probe restores A's one-stream snapshot into slot 1 of a two-slot server, keeps that neighbor populated during all invalid imports, and checks the neighbor is unchanged. This probe remains unrun under the build hold. The C++ syntax check for the portability candidate exited 0 (`cxx-syntax-04.log`).

The policy now rejects non-finite measurements and uses a body-transfer timer after response headers, excluding prefill latency from its link EWMA. The explicit transfer-timing control plus the prior suite passed: 13 tests, 10.328 seconds, exit 0 (`proxy-tests-05.log`). The previous 12-test measurement candidate also passed (10.388 seconds, `proxy-tests-04.log`). No performance benchmark was run; the timer control uses a deterministic mock clock.

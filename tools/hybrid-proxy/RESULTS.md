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

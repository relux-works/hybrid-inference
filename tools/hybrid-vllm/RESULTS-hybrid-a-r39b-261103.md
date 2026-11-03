# r39b loopback bridge timing preparation

Assigned base: `fca0fe31` (prototype main, PR #16). Branch: `m3-vllm-bridge-timing`. Local signed delivery only, no push. Ready for review as CPU-qualified instrumentation and rental scripts; GPU measurements remain unrun.

## Delivered scope and measurement limits

- `bridge_timing.py` drives the real node HTTP extend/conversion/transfer/native-import path over loopback into an exclusive CUDA client. All D=16384/65536/131072 and delta=2343/10000 pairs have a warmup and three rounds; exact aligned import and local-tail journals are required before TTFT can qualify.
- The exporter measures synchronized page/state copy and serialization. Backend and adapter expose separate evaluation, export and conversion headers. Existing `X-Hybrid-Prefill-Ms` and the exact finished.json completion contract remain unchanged.
- Evaluation is backend HTTP wall minus measured export, including scheduling/frontend and one output token. This is an explicit envelope, not a pure GPU kernel timer. Transfer is response body read after headers; import includes decoder HTTP/upload. TTFT includes all phases, journal checks and the local tail. Raw inclusive/backend and residual values stay visible.
- The wrapper uses existing source-pinned hook installation, codec startup qualification, build/client/node launchers and state-aware cleanup. It checks exact clean source, uses new ports and run directories, emits terminal done/failed and never stops foreign states. It builds the existing client/checker targets through the client launcher, with CUDA 120a-real on C.
- Both model identities and heterogeneous weights remain explicit. Successful timing/import is not a numerical-equivalence verdict. No economics replacement is made.

## CPU evidence and corrected failures

All logs are under `.temp/hybrid-a-r39b-261103/` in this worktree. Disk-backed TMPDIR and PYTHONDONTWRITEBYTECODE=1 were set. No GPU, package/model download or full engine build ran on agents1.

| Check | Exit | Evidence |
|---|---:|---|
| Initial six-shape timing arithmetic, phase refusal, SSE and backend accounting (4 tests) | 0 | timing-tests-01.log |
| Initial broad suite (71 tests, 4 failures, 1 skip) | 1 | bridge-suite-01.log |
| Native helper first link attempt against old installed CPU libraries | 1 | native-build-01.log |
| Native helper rebuilt against compatible existing r34 CPU libraries | 0 | native-build-02.log |
| Corrected broad suite (71 tests, 1 torch skip) | 0 | bridge-suite-02.log |
| Adapter phase-header control plus prior controls (5 tests) | 0 | timing-tests-02.log |
| Broad suite (72 tests, 1 torch skip) | 0 | bridge-suite-03.log |
| Append/import endpoint-mismatch cleanup plus prior controls (6 tests) | 0 | timing-tests-03.log |
| Final broad suite (73 tests, 1 optional torch skip) | 0 | bridge-suite-04.log |

The initial export instrumentation put timing.json beside the capture, violating `test_export_refusal_cleanup`; it now lives inside the atomically published capture directory. The other three initial failures were the pre-existing native roundtrip/chunk tests using the old protocol helper: it rejected HYBD v2. The first rebuild linked old libraries and failed on missing chunk APIs. The corrected rebuild links the existing `.temp/build` tree in `hybrid-a-r34-261102/prototype`; `git diff fca0fe31..18d3f114e -- src include ggml tools/hybrid/hybrid-state.cpp tools/hybrid/hybrid-state.h` is empty. Current native_check.cpp and hybrid-state.cpp are compiled, not replaced with stubs. The helper remains a synthetic CPU reader; it proves no GPU extraction or numerical equivalence.

Reproducible native helper command (existing libraries, no download):

```bash
r34=/home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r34-261102/prototype/.temp/build
c++ -std=c++17 -O1 -DGGML_BACKEND_SHARED -DGGML_SHARED -DLLAMA_SHARED \
  -Itools/hybrid -Isrc -Iinclude -Iggml/include -Ivendor \
  tools/hybrid-vllm/native_check.cpp tools/hybrid/hybrid-state.cpp \
  "$r34/vendor/hash/libvendor-hash.a" -L"$r34/bin" -lllama -lggml -lggml-base -lggml-cpu \
  -Wl,-rpath,"$r34/bin" -o .temp/hybrid-a-r39b-261103/native_check
```

System Python has no NumPy (readiness probe failed); the broad tests reuse the existing r34 `ruler-venv` with NumPy 2.4.2. Torch is absent, so the explicitly optional torch test skips; CUDA gather/codec/live bridge tests are unrun. The six new focused timing controls run with stdlib Python alone. Original 67 test methods are retained under their names; retention evidence is in the parent paper task's retention-01.log.

Self-review applied the repository code-review skill's scope, security, general, design and tests checks. No new model, core API, backend or quantization code is changed. Private token files are read through the existing private-file guard and are not logged or passed by value on command lines. Live URLs must be literal loopback. Missing phase headers, invalid timings, mismatched endpoints and incomplete SSE refuse before qualifying a measurement. No surface table was supplied (brief gap), and no mutation coverage map is claimed. No acceptance rows are silently excluded. Native/GPU qualification and independent review remain explicit gates. Changes outside tools/hybrid-vllm are only the required root README tool link.

Rental procedure, exact commands, outputs and estimates are in [BRIDGE-TIMING.md](BRIDGE-TIMING.md). Combined C stage estimate: 3-6 hours cached, 120-160 GiB free; bridge itself 30-90 minutes plus 10-30 minutes cold build. These are estimates, not measurements.

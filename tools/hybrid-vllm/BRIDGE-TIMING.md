# Loopback vLLM bridge timing (r39b)

Base: prototype main fca0fe31 (PR #16). The new local branch is m3-vllm-bridge-timing. This script delivery does not claim GPU execution. Run node C's RTX PRO 6000 first after its KL owner releases it, in the same rental as kit 1.6.8 step 73. Keep the shared vLLM 0.32.0 / torch 2.13.0 environment and FP8 cache.

## Run

Transfer a clean checkout of the signed timing head and set these paths on the rental:

```bash
export EXPECTED_REF=FULL_SIGNED_TIMING_HEAD
export RUN_DIR=/rental/stage-pro6000-168/bridge
export VLLM_VENV=/rental/work168/vllm-venv
export HF_HOME=/rental/work168/hf
export HF_MODEL="$HF_HOME/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
export Q4_GGUF=/rental/work168/models/Qwen3.8-27B-Q4_K_M.gguf
export CUDA_ARCH=120a-real
bash tools/hybrid-vllm/run-bridge-timing.sh
```

The kit's `stage-pro6000.sh` supplies the same variables after step 73 and checks the signed heads supplied in the handoff. The bridge wrapper downloads no models or packages. It requires a new private output directory, exact clean source and unused ports. It runs the existing gather gate, configures CUDA and lets the existing client launcher build llama-server plus its two required checker targets, then starts a fresh prototype node and an exclusive CUDA client using the existing state-aware launchers. The node's codec byte gate still runs in its adapter before readiness. Node context and client context are 147456; the node reserves 0.55 of GPU memory to coexist with the Q4 client. Record allocation refusal rather than enabling CPU fallback.

All six shapes are D=16384,65536,131072 and delta=2343,10000. Each uses one excluded warmup and three rounds, with a 15-second cooldown between shapes. The script first brings the node journal and exclusive decoder slot to D, outside timing. This bootstrap includes real server requests and a local CUDA prefill. It then starts the append timer, extends through the node HTTP adapter, converts and transfers aligned HYBD v2 chunks, imports them into the native CUDA decoder, verifies its exact prefix hash/token journal and streams the first local output token. The endpoint is `floor((D+delta-1)/block)*block`; the remaining prompt tail is evaluated on CUDA and reported. No alignment rounding silently changes the requested delta. The driver reuses the proxy's admitted heterogeneous envelope rebinding and checks live layout identities first. It executes the same node/slot protocol over loopback directly; proxy chat rendering, policy selection and browser latency are outside this timing.

## Timing boundaries

- `server_prefill_ms`: backend HTTP request wall time minus measured export. This includes vLLM admission, scheduler/frontend overhead and its one output token. It can include own-prefix cache reuse. It is an evaluation envelope, not an isolated CUDA kernel duration. Preserve the raw backend wall in each chunk; do not substitute it for cold prefill rate.
- `export_ms`: the live exporter synchronizes CUDA first, then measures page/state gather, device-to-host copy, serialization and validation up to capture publication. Waiting for prior GPU compute belongs to the evaluation envelope.
- `conversion_ms`: adapter's conversion to the complete HYBD delta, including codec and checksum; reading the output bytes afterward remains residual overhead.
- `transfer_ms`: loopback response-body read after headers, matching the existing Node transport boundary. Header wait, outgoing request JSON, envelope rebinding and decoder request upload remain residual or import overhead as appropriate. This does not estimate a WAN link.
- `import_ms`: decoder HTTP import call, including request upload and native validation/commit. Multi-chunk rows retain every receipt.
- `ttft_ms`: the whole measured append, from before the first node extend through the first native streamed output token. `other_and_local_tail_ms` retains unassigned overhead and the explicit local tail. No output-token event or terminal SSE means refusal.

The historical `X-Hybrid-Prefill-Ms` remains total adapter elapsed for compatibility. New headers expose `Evaluate`, `Export`, `Conversion`, and inclusive `Backend` milliseconds. They are distinct: backend = evaluate + export; conversion is outside backend. Timing metadata is separate from finished.json, so its original exact completion contract stays intact. Older hooks without phase metadata cannot pass the new driver.

Each request's timings are measured inside its existing serialization boundaries. Three-round medians are computed per component; component medians need not sum to the median TTFT. JSON preserves every raw round, exact exported endpoint, local tail and chunk receipt. Any missing header, invalid timing, failed import, mismatched journal, missing token or partial shape fails and leaves the remaining rows pending. The wrapper prints `done bridge-timing` only after successful state-aware client/node cleanup, otherwise `failed bridge-timing exit=N`. It stops only the states it creates; an existing RUN_DIR refuses without touching its contents.

The FP8 node and Q4 decoder have heterogeneous weights. Timing and successful import do not establish numerical equivalence; the separate KL/equivalence gates remain in force. No economics or paper rates are changed by these scripts.

## CPU tools and evidence

| Tool | Command | Output |
|---|---|---|
| Python stdlib | `python3 -m unittest discover -s tools/hybrid-vllm -p test_bridge_timing.py -v` | CPU arithmetic, phase refusal and SSE controls; no GPU/model/install |
| Existing NumPy venv + native helper | `HYBRID_VLLM_NATIVE_CHECK=/existing/native_check /existing/venv/bin/python -m unittest discover -s tools/hybrid-vllm -p 'test_*.py' -v` | Existing converter/adapter/runtime regression suite; no new build or download |
| Bash/CMake/CUDA, rental only | `bash tools/hybrid-vllm/run-bridge-timing.sh` | RUN_DIR numbered logs, exact source/binary hashes, own process states, codec receipt and timing/bridge-timing.json |
| Python driver, already launched exclusive client only | `python3 tools/hybrid-vllm/bridge_timing.py --client-state PRIVATE_RUN/client --out PRIVATE_RUN/timing` | New private JSON directory; loopback only, requires CUDA client metadata |

Set PYTHONDONTWRITEBYTECODE=1 and a pre-created disk TMPDIR for all commands. Preparation evidence lives under `.temp/hybrid-a-r39b-261103/`; persistent check results are in RESULTS-hybrid-a-r39b-261103.md. No surface table was supplied (brief gap); no mutation coverage map is claimed. No acceptance rows are silently excluded. The synthetic native helper is rebuilt against an existing compatible CPU library tree; new GPU execution remains unrun.

Estimated bridge time is 30-90 minutes plus a cold CUDA build (10-30 minutes). The combined C stage is budgeted at 3-6 hours with ~120-160 GiB free disk. Actual cache, network and maximum-session volume can extend this. Keep private tokens, model state and captures out of published reports.

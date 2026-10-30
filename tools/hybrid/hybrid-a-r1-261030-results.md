# Producer A: CPU prototype results

Run: hybrid-a-r1-261030, 2026-10-30 UTC. Base e0f142d5dd. Branch hybrid-server, local signed delivery only. No push or PR. Initial specification v0.3 SHA-256 e5858388ab249646c8c65df4d6b4f4c6009fc5ffd30c8f766cc6052b3cf0936b; later orchestrator decisions adopt a single native section, token IDs in its header, one-token holdback and HTTP/1.1. These supersede per-layer streaming and empty-suffix sampling for this first build.

## Implemented

- Staged range-export API, with a position filter in the attention writer and full current recurrent state. Both subcache endpoints must equal p1; unsupported flags, empty ranges and stale endpoints reject.
- Shared HYBD v1 writer/reader in `hybrid-state`, using the vendored XXH3-128 and SHA-256. The reader bounds all envelope lengths and validates version, model, positions, section layout, checksums and token metadata before returning native bytes. Append callers verify the hash of their merged prefix and validate native layer layout.
- CPU prefill node: create/list/info/extend/snapshot/delete/health/metrics; separate bearer-token namespaces; foreign IDs return 404. RAM snapshots, append-only n0/hash checks, bounded token/context input, 512 MiB payload budget, 128 sessions. HTTP/1.1 response chunks; shadow appends apply before 202. Scheduling rotates admitted clients first, then prioritizes eligible critical work within each client, preserving session FIFO.
- Producer B's signed transactional append core 684e35a5b8, verified and cherry-picked as 35f6cce096. This is distinct from B's later server endpoint and portable slot-restore changes.
- Golden fixtures at `/home/agent/Werkstatt/hybrid-models/golden/`, with exact token IDs, held token, greedy outputs, prefixes, deltas, full deltas, README and SHA256SUMS.

## Environment and commands

Model: `/home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf`, SHA-256 `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`. Architecture qwen35; CPU on both contexts, two threads, q8_0 K/V, flash attention enabled, f32 recurrence. No other model was downloaded and no benchmark was run.

CMake was absent. Installed task-local CMake 4.4.4 with:

```sh
python3 -m pip install --no-cache-dir --target .temp/hybrid-a-r1-261030/cmake-tool cmake==4.4.4
```

Exact build/run commands and tool outputs are documented in [README.md](README.md). Gate commands ran directly, without tee or pipeline masking. All logs below are relative to `.temp/hybrid-a-r1-261030/`.

## Validation

The engine and shared codec passed build 05 at c68df953. The final scheduler correction passed build 06 at 1c496c523d; only the two tool translation units rebuilt. A named `scheduler_tests` regression requires client B's shadow request to receive its turn despite client A's critical backlog; eligible critical work still precedes another session's shadow within the selected client.

| Prompt tokens | Prefix n0 | Delta endpoint n1 | Native delta bytes | Native full bytes | Greedy result |
|---|---|---|---|---|---|
| 512 | 383 | 511 | 21,040,756 | 23,550,172 | 64/64, exit 0 |
| 2048 | 1919 | 2047 | 21,040,756 | 33,614,044 | 64/64, exit 0 |
| 8192 | 8063 | 8191 | 21,040,756 | 73,869,532 | 64/64, exit 0 |

Commands for these persisted fixtures:

```sh
build/bin/llama-hybrid-check /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden 512
build/bin/llama-hybrid-check /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden 2048
build/bin/llama-hybrid-check /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf /home/agent/Werkstatt/hybrid-models/golden 8192
```

Logs: `append-512-01.log`, `append-2048-01.log`, `append-8192-01.log`. Each valid append had byte-identical serialized native state to the full source export. Each then evaluated the held last token and compared greedy output against a newly cleared context prefilling all tokens locally, without using the transferred state.

Negative controls:

| Surface | Executable checks | Observed behavior |
|---|---|---|
| Envelope | `codec_tests`: wrong model, gap/overlap, truncated/trailing bytes, corrupt section/body checksum, declared layout, token count, negative token, prefix hash | Exact intended rejection reasons; valid neighbor round trip accepted |
| Raw append | `append_negative_tests`: gap, overlap, wrong endpoint, truncation, magic, stream layout, cell-position gap, layer count | API returns 0 and serialized prefix remains byte-identical after every rejection; nearby valid append accepted |
| Range export | stale endpoint, empty range, negative p0, unsupported device flag | API returns 0 and source state remains byte-identical |
| HTTP | wrong model, gap/overlap, wrong prefix, invalid token types/bounds, missing bearer, foreign get/extend/snapshot/delete | 400/409/401/404 as applicable; existing count/hash/snapshot unchanged |
| Concurrent namespaces | 3 clients x 4 simultaneously resident sessions, round-robin client input with interleaved offload/shadow | All IDs distinct; expected counts and hashes; all deletions succeed and payload RAM accounting returns to zero |

Final all-depth sweep on signed source 1c496c523d, after the scheduler correction: `build/bin/llama-hybrid-check /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf -`, exit 0, `append-final-01.log`. All three depths again passed 64/64 with every negative control and exact state equality. This sweep suppresses artifact rewrites; the persisted goldens above retain their recorded c68df953 producer identity.

Latest standalone codec/scheduler command: `build/bin/llama-hybrid-check --codec`, exit 0, `codec-03.log`. HTTP command: `python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r1-261030/http-04`, exit 0, `http-04.log`. Node stdout/stderr and owned PID are retained below that artifact directory; the script stopped and waited for its own process.

## Resource controls and corrected failures

One build at a time under `/home/agent/Werkstatt/hybrid-models/.build.lock`, CPU -j2, focused targets only. The orchestrator paused new builds for disk pressure; A made no new build or configure during that hold. Keeper/orchestrator explicitly admitted builds 05 and 06 in the existing tree. Exact free-byte checks required >=22 GiB in addition to df output. Build 05 added 215,132 bytes (33,642,804 -> 33,857,936); build 06 added 2,471 bytes (33,857,936 -> 33,860,407). Build 06 started at 22.1891593933 GiB exact free.

Corrected setup failures are retained: build-engine-01 exited 2 for a missing stdexcept include; build-hybrid-02 exited 2 because the new target needed configuration; build-hybrid-03 exited 2 for a missing httplib include path. Builds 04, 05 and 06 exited 0. The initial Matrix post failed 403; joining the authorized room resolved it. One notice accidentally carried a commit placeholder and was immediately corrected to the exact signed SHA.

## Bounds and handoff gaps

The brief supplied no attack-surface table; this is a brief gap, not an assertion of exhaustive surface coverage. No source mutation campaign was run. Corrupt payload controls above are executable rejection tests with nearby valid controls.

This is a CPU first build ready for cross-review, not an accepted CUDA/Metal integration. Only three deterministic prompt depths are covered, rather than the specification's later 50-prompt/large-context matrix. The 12-session HTTP test checks namespace/state persistence and hashes; it does not decode 64 tokens for every concurrent session. NVMe storage, restart recovery, encryption at rest, TTL/LRU eviction, asynchronous shadow batching and per-layer streaming are deferred. The brief explicitly permits RAM-first storage; the orchestrator explicitly permits the single-section transport and one-token boundary. Production hardening is outside specification section 1. No claim is made for arbitrary cache-wrap layouts, other architectures, B's full server head, or cross-slot snapshot portability.

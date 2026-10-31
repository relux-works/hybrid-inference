# M1 node limits qualification

Run: hybrid-a-r10-261031. Branch: `m1-node-limits`, based on `b/m1-hybrid-bench` at `20964cee820951d904606354e4049b64bcf4104c`. Local signed delivery only; no push or upstream submission. This report is being populated while the long CPU gates run; pending rows are not passing evidence.

## Scope and resulting behavior

Changes are confined to `tools/hybrid/`. No changes to the engine, proxy, benchmark or launcher. Implements `--max-ctx` (ceiling default 16384), `--ram-budget-mib` (512 total across sessions), `--max-request-mib` (1), global idle LRU eviction with response-lifetime pinning, atomic impossible-budget refusal, and model identity plus counters on health/metrics. Existing positional allocation default remains 8448; with no positional context, an explicit `--max-ctx` selects that allocation. Explicit positional context cannot exceed the ceiling.

An evicted ID returns 404. Reads and accepted extends refresh logical LRU order. The scheduler serializes context access, and streaming responses retain a session pin until completion/disconnect. A queued request has not acquired its session and can observe 404 after eviction. Admission finds enough unpinned victims before erasing any. If it cannot fit, it returns `503 {"error":"ram budget","need":BYTES,"budget":BYTES}` without committing any session mutation or eviction. `need` includes the new session state and retained pinned states after every eligible idle state is hypothetically removed.

The retained-state budget counts serialized state and four-byte token IDs. Model weights, live context, empty-state template, allocations and temporary old/new state, token, JSON and wire buffers remain outside that counter. Rejection is after inference/serialization and does not bound transient RSS. Wire limits remain exactly 1 MiB header and 512 MiB body. Resident contexts and host/disk tiers are design-only in [NODE-STATE-DESIGN.md](NODE-STATE-DESIGN.md).

`GET /v1/health` contains JSON fields `max_ctx`, `n_ctx`, `ram_budget_bytes`, `ram_bytes`, `sessions`, `evictions`, `max_request_bytes`, `model_sha256`, `layout_fingerprint`, `model_layout`. Prometheus `/metrics` has corresponding `hybrid_` numeric fields and a `hybrid_model_info` label row. Counters are individually atomic, not one transactionally consistent multi-counter observation.

The layout SHA uses compact JSON with recursively sorted object keys. The actual model projection includes architecture, layer count, embedding size, per-layer query/KV head counts and dimensions, recurrent-layer bitmap and SSM dimensions, vocabulary count and a SHA-256 of vocabulary bytes in token-ID order with little-endian uint64 length framing. Quantization/file name are excluded. The independent benchmark GGUF reader agrees on every layout field and the canonical hash. Exact GGUF SHA remains mandatory for native state; matching layout alone does not establish numerical equivalence across quantizations.

## Environment and reproduction

CPU-only Linux x86_64, GNU C++ 15.2.0, Release, CMake 4.4.4, Python 3.14. No downloads or installations. Reused the existing configured CPU build. Every compile/configure is protected by the shared build lock, checks exact available bytes against 26,843,545,600 first, and compiles at `-j2`. The isolated negative-control executable compiles only the node translation unit and links the already-built libraries; it does not rebuild the engine.

Set these paths for reproduction; private machine paths and credentials are deliberately excluded from this public report:

```sh
MODEL_CACHE=/path/to/existing/model-cache
MODEL="$MODEL_CACHE/Qwen3.5-0.8B-Q4_0.gguf"
REQUANTIZED="$MODEL_CACHE/Qwen3.5-0.8B-Q8_0-requant.gguf"
GOLDEN="$MODEL_CACHE/golden"
CMAKE=.temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake
RUN=.temp/hybrid-a-r10-261031
# All actual argv, times, PIDs and exits are in the private $RUN/commands.jsonl.
flock "$MODEL_CACHE/.build.lock" "$CMAKE" --build build -j2 --target llama-hybrid-prefill llama-hybrid-check llama-hybrid-check-codec llama-hybrid-check-append llama-hybrid-check-guards llama-hybrid-check-distributions llama-hybrid-check-client
build/bin/llama-quantize --allow-requantize "$MODEL" "$REQUANTIZED" Q8_0 2
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL" --artifacts "$RUN/limits"
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL" --requantized "$REQUANTIZED" --artifacts "$RUN/identity-02" --case identity
python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model "$MODEL" --artifacts "$RUN/deep" --case deep
build/bin/llama-hybrid-check-distributions "$MODEL" "$RUN/deep/deep.tokens.json" --max-ctx 65536
```

The actual lock commands also execute the byte-floor guard inside the lock; see the command appendix and private logs for every guard and exit. The model cache retains the explicitly authorized local re-quantization. Native state fixtures are task-local, hashed and deleted after the gates.

## Results

| Gate / named claim | Evidence | Actual exit and result |
|---|---|---|
| Three sessions / small budget, LRU refresh, 404 and replay | `limits-final-01.log`, `test_lru_replay_and_atomic_refusal` | exit 0; replay tokens and native snapshot bytes identical |
| Impossible extend preserves every session | same final gate, `test_lru_replay_and_atomic_refusal` | Repeated 503 with need 59,538,100, budget 50,331,648; target and both neighbors byte-identical; later smaller append accepted |
| Oldest pinned snapshot and extend survive competing eviction | `limits-final-01.log`, `test_pinned_read_and_extend` | exit 0; active oldest response survives, idle other-client neighbor evicted |
| All-pinned refusal / disconnect release | same gate | exit 0; exact 503, no mutation/eviction, then disconnect permits a valid append |
| Inclusive request cap, one-over and larger configured cap | `limits-final-01.log`, `test_request_caps_and_metrics` | exit 0; 1 MiB and 2 MiB accepted exactly; cap+1 is 413 and unchanged snapshot |
| CLI limits | same gate, `test_invalid_limits` | parent exit 0; seven invalid startups each exit 1 as expected |
| Metrics | same gate | default values 16384 / 536870912 / 1048576 and exact numeric/identity agreement between health and Prometheus |
| Header valid maximum and one-over, default and long configured contexts | `scheduler-default-01.log`, `codec-65536-02.log`, `codec-canonical-01.log` | all exit 0; exact budgets 184320, 724992 and 1048576 bytes accepted; writer and reader reject one byte over |
| Local Q4_0 to Q8_0 re-quantization | `quantize-01.log` | exit 0, two threads; retained in local cache |
| Quantization-independent identity | `identity-02.log`, `identity-reference-01.log` | both exit 0; actual GGUF SHAs differ, every layout field and fingerprint match |
| Different geometry identity | `native-01.log` and codec controls | exit 0; native projected embedding change and synthetic field changes differ; independent second base model unavailable |
| 40000-token HTTP transfer and numerical references | `deep-http-01.log`, deep distribution gate | pending |
| Existing HTTP, native, malformed and distribution regressions | command ledger and appendix | pending remaining long gates |

Canonical small-model layout fingerprint: `33d9f73396507b1f8402bf1e439936f42caee2c8875fb46e127b06e0e656d2fa`. Original GGUF SHA-256: `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`. Re-quantized SHA-256: `f543fad263339779c3a38a21a531c6adbffcfdecdf5f749dd542de924b1e7302`; file size 811,843,840 bytes.

## Controls and corrections

| Mutant / control | Property narrowed by it | Named failing test | Survival bound |
|---|---|---|---|
| Eligibility ignores active response pins, while still excluding the current extender | LRU must skip a pinned oldest session, not merely happen to select another older victim | `test_pinned_read_and_extend`, snapshot leg: active session GET returns 404 instead of 200 | Killed: isolated `unpinned-control-01`, actual exit 1 on the intended assertion. No exhaustive mutation claim |
| Reversed token-distribution / KL-only / top1-only controls | Distribution checker rejects both metric violations, with an identical positive control and exact threshold boundaries | `controls()` in the distribution checker | Existing executable controls; final numerical gate result below |

First build exited 2 because the old CMake build graph lacked the benchmark-check target. No compiler failure; explicit configure of the existing tree exited 0 and subsequent incremental builds passed. The benchmark checker was not edited or used for the node's deep test.

The first two limits-suite attempts exited 1 due to a test setup error: a 2000-token session plus its neighbor exceeded 48 MiB during fixture creation, so the node correctly evicted sessions before the intended streaming-eviction step. Corrected the active fixture to 1000 tokens and made the pinned session explicitly older than its idle neighbor; the corrected targeted suite exits 0. Both full-suite logs retain their already-passing LRU and byte-preserving refusal legs.

The first default HTTP regression exited 1 on an allowed transient 429 during final DELETE cleanup, after its concurrent work had succeeded. Cleanup now uses the harness's existing overload retry; the corrected default run exits 0. Admission limits and product behavior were not weakened.

The first client-fixture preparation exited 1 because its output directory did not exist. Created the task-local directory and reran; subsequent results are listed in the appendix. The first identity implementation used insertion-order JSON hashing; before handoff it was corrected to sorted-key canonical JSON, with insertion-order controls and a fresh actual-quantization plus independent-reader check. Earlier fingerprint observations are superseded.

## Finite coverage and untested steps

No supplied surface table or external landing property suite. No prior reviewer tests removed; the RAM-error fixture retains its original test name and is updated for the new eviction contract. No rework module expansion. Self-review covers scope, security, lifetime, numeric bounds and tests under the local code-review skill; no hosted review, check or landing is claimed or requested.

No GPU, Metal, 27B, 128k/256k inference, rented hardware, transport shaping, restart recovery, disk tier or resident-context implementation. The deep test uses one deterministic synthetic 40000-token prompt on the existing 0.8B CPU model, not the specification's full 50-prompt matrix. Re-quantization verifies file/layout identity, not heterogeneous numerical acceptance. No other base-model GGUF is available without a forbidden download; actual different-base-model identity remains unrun, supplemented by synthetic structural and model-projection controls.

The installed environment lacks `ty` and `editorconfig-checker`; their hosted workflows were not run. Narrow Python compilation, C++ builds, whitespace/scope/public-data checks and the requested functional gates are recorded separately. The code-style workflow checks model source naming, which is outside this delta; no hosted CI is claimed.

Process cleanup, state-dump hash/removal, final commit signatures and the complete command/exit appendix are recorded after all running gates finish.

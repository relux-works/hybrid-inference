# bf16 recurrent state transport: r38 CPU evidence and GPU tools

Ready for review on local branch `m3-state-bf16`, based directly on accepted RULER head `18d3f114e46c4f801c4a09e2c6cf8ba2303ae42d`. No push, PR, landing, GPU execution or downloads. The signed delivery commit is the branch head; its SHA is sent separately to orchestration. Existing worktrees and branches were not switched.

The implementation and negative controls are described in [STATE-BF16.md](STATE-BF16.md). Only Qwen35 recurrent payloads change; attention remains q8_0. Default f32 envelopes have a frozen pre-change header fixture, independent checksum controls and native bit-exact import checks. bf16 uses the existing scalar ggml RNE conversion and expands before native import. Session records, journals and traces carry dtype/origin. Exactness gates classify bf16 as lossy/exit5 even when argmax or numerical values agree.

The RULER declared arm includes changed evaluator and service executable identities. All other manifest fields compare via canonical typed JSON digests; bool/int substitutions also refuse. Both baseline and candidate implementation hashes are retained. Baseline row/spec hashes, prompt identity, and completed archive contents are checked. Only committed examples are reused; only `offload_bf16` executes. A resumed completed arm preserves its sealed archive byte-for-byte. This declared software boundary is necessary to pair existing f32 observations with a newly implemented transport option; it is not a claim of unchanged executables or a temporally interleaved experiment.

## Commands and exits

All paths below are relative to this task worktree. Logs live in `.temp/`. Set `TMPDIR=$PWD/.temp/tmp` and `PYTHONDONTWRITEBYTECODE=1`. Python unit commands use `PYTHONPATH=tools/hybrid-proxy:tools/hybrid-bench:tools/hybrid-eval`. `PY_RULER` is the existing local Python3.14 environment at `/home/agent/Werkstatt/hybrid-a/llama.cpp/.temp/hybrid-a-r34-261102/prototype/.temp/ruler-venv/bin/python`; no packages were installed. CMake was reused from the local r1 installation after PATH readiness found it absent.

| Command / evidence | Actual exit | Claim and limit |
|---|---:|---|
| CMake configure, CPU Release, `GGML_NATIVE=ON`, `GGML_CUDA=OFF`, both UI sources off, tests/examples/app off | 0 | Offline configuration; `.temp/configure-01.log`. |
| `nice -n 10 cmake --build .temp/build -j4 --target llama-hybrid-prefill llama-hybrid-check-codec llama-hybrid-check-append llama-server llama-hybrid-bench-check llama-hybrid-score` | 0 | Final native sources compile; `.temp/build-05.log`. Hashes/configuration identity retained in `.temp/tested-identity.json` and readiness logs. |
| `python3 -m unittest test_proxy test_journal test_templates test_benchmark` | 0 |109 tests; final `.temp/proxy-suite-02.log`. Includes dtype mismatch, session persistence and existing journal/template/recovery behavior. |
| `python3 -m unittest test_bench test_growing_session test_hetero_study test_eval` | 0 |106 cases, one existing skip; `.temp/python-suite-01.log`. Includes bf16 exact-gate refusal and matched positive controls. |
| `$PY_RULER -m unittest discover -s tools/hybrid-eval -p test_ruler.py` | 0 |21 tests; final `.temp/ruler-tests-06.log`. Includes all old tests and single-arm reuse/resume, strict manifest/type mismatch, sealed-row substitution and analysis controls. |
| `.temp/build/bin/llama-hybrid-check-codec` | 0 | Frozen f32 header, independent checksums, RNE ties/exponent boundaries/NaN/Inf/zero/underflow/overflow; `.temp/codec-tests-03.log`. |
| `.temp/build/bin/llama-hybrid-check-append /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --bf16` | 0 | Real-state f32 byte/bit exactness, bf16 payload/formula, per-normal-value error, KV identity, odd/truncated checksum-valid bodies and dtype mismatch preserving the slot, nonfinal/final chunks, mixed-dtype refusal and valid retry; `.temp/native-bf16-04.log`. |
| `python3 tools/hybrid-bench/test_integrated.py --prefill-binary .temp/build/bin/llama-hybrid-prefill --server-binary .temp/build/bin/llama-server --checker-binary .temp/build/bin/llama-hybrid-bench-check --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/http-bf16-final --case bf16` | 0 | Real HTTP negotiation and no-mutation refusal, successful proxy import with dtype in session/journal/trace/chunks; `.temp/http-bf16-03.log`. |
| Same native harness, `--case noise --artifacts .temp/native-noise-final` | 0 | f32 remains exact at 8/8 positions, single-final chunks match old full-state controls; checksum error exits2, checksum-valid corrupt/lossy controls exit3, including 8/8 argmax despite nonzero logit error; `.temp/native-noise-02.log`. CPU-only GPU-request and false-backend controls also refuse without GPU execution. |
| `python3 tools/hybrid-bench/test_hetero_study_native.py --build-dir .temp/build --client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --reference /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --artifacts .temp/study-native` | 0 | Three local synthetic documents, N64/M16, real node/scorer, bf16 arm, byte savings, receipts, report statistics and refusal/resume controls; `.temp/study-native-01.log`. No Gutenberg or model download. |
| Final `hetero_study.py arm`, same study, `--kind imported --label bf16-ggml --state-dtype bf16 --chunk-tokens 8192 --score-batch 4 --gpu-layers 0 --ram-budget-mib 128 --keep-deltas`, same Q4 node/client and Q8 reference hash | 0 | Reusing ggml conversion produces byte-identical compressed native bodies and 48/48 identical score rows to the earlier integer-reference conversion; `.temp/native-ggml-arm-01.log`, `.temp/ggml-conversion-equivalence.json`. |
| `.temp/build/bin/llama-hybrid-bench-check MODEL .temp/native-exact-bf16.json ignored .temp/native-exact-bf16.result.json 0` | **5** | Expected lossy verdict, not a passing exact gate. Actual bf16 import, 8/8 top1, repeat exact, same-partition KL 0.0000437021 and max logit difference 0.425491; `.temp/native-exact-bf16-03.log`. |
| `ty check --python EXISTING_RULER_VENV --extra-search-path tools/hybrid-bench --extra-search-path tools/hybrid-proxy tools/hybrid-eval/ruler_eval.py tools/hybrid-eval/ruler_analysis.py` | 0 | Scoped evaluator static check; `.temp/type-check-04.log`. |
| AST parse all 16 changed Python files; `git diff --check`; reviewed source hashes versus `.temp/tested-identity.json` | 0 | Syntax, whitespace and exact tested source identity; `.temp/syntax-01.log`. |

The 236 unit cases include one inherited skip:235 passed. The opt-in native harnesses above are additional. Final code-review scope was General, Security, Approach and Tests plus the small server import plumbing. Native model/core/backend source was not changed. No attack-surface table was supplied (brief gap); no requested implementation/test/tool row is declared out of contract. fp16 was optional and is omitted. GPU measurements are explicitly excluded by the task, so tools/runbooks are delivered without invented quality or latency results.

## Named control plants

| Mutant | What it narrows the gate to | Named failing test | Result and surviving bound |
|---|---|---|---|
| Remove lossy-origin classification from `classify_equivalence` | Same partition numerical agreement could falsely accept a bf16 import | `Bf16GateTests.test_bf16_import_never_passes_exact_gate` | Killed, actual unittest exit 1; `.temp/mutant-gate-01.log`. No survivor in this plant. |
| Bypass the paired manifest comparison | A changed depth/seed/model/data/service field could pair to old rows | `Bf16RulerTests.test_manifest_every_nondeclared_field_refuses` | Killed, actual unittest exit 1 with named field subtests; final `.temp/mutant-manifest-03.log`. No survivor in this plant. |

These are bounded instrument controls, not exhaustive mutation coverage. Native negatives additionally reach the actual client validator using recomputed correct checksums, so checksum failure cannot stand in for bf16 length validation. Exact f32 imports and matched bf16 imports are the adjacent positive controls.

## CPU numerical observations

All normal recurrent values in the real 16-token Qwen3.5-0.8B test satisfy relative error <=2^-8; maximum observed 0.00389105. q8_0 attention bytes are unchanged. Recurrent tensor payload 20201472B becomes 10100736B. Actual complete envelope 20307547B becomes 10206812B (one extra dtype character).

RNE may map the largest finite f32 values to infinity and can underflow f32 subnormals; NaNs remain quiet NaNs and infinities/signed zero are preserved. Consequently the finite relative bound applies to normal inputs with finite rounded outputs, not the mathematical overflow boundary. These cases are executable synthetic controls, not silently excluded passing observations.

The tiny natural-text fixture has 3 documents x 16 positions. f32 local and same-model f32 import have identical metric rows and 48/48 top1; direct local-to-import KL is 0. bf16 likewise agrees 48/48 top1, with direct local-to-import mean KL about 0.0000666553 across those 48 positions. Against the Q8_0-requant reference, local/f32 mean KL is 0.0001383209 and bf16 is 0.0002219208. The reference is a local requantization fixture, not BF16 source weights. This is CPU protocol evidence, not the 27B stage-2 study or a long-context quality conclusion.

## Wire table and GPU budgets

Tensor payload per offload at 100k prefix depth, excluding metadata/envelope: `34816*D +156893184` (f32) or `34816*D +78446592` (bf16).

| D | f32 bytes | bf16 bytes |
|---:|---:|---:|
|512|174718976|96272384|
|2343|238467072|160020480|
|10000|505053184|426606592|

Full native-body/envelope formula and reproducible complete-wire examples are in [STATE-BF16.md](STATE-BF16.md). Arithmetic evidence is `.temp/wire-table.json`.

- [KL runbook](../hybrid-bench/STATE-BF16-RUNBOOK.md): existing stage-2 D20/N32768/M2048, score-batch32 and seed42; f32/bf16/local arms. Reserve1-2h as an unmeasured rate-based estimate; reuse the18.947GiB reference and allow at least7GiB extra free space plus logs/host floor, excluding models.
- [RULER single-arm runbook](../hybrid-eval/RULER-BF16-RUNBOOK.md):390 existing examples/depth. Measured three-arm15.4s at8k and about59s at32k imply one-third planning estimates33.4min and2.13h; reserve45-60min and3-4h. The client disk admission is `2*saved_session_bytes +1GiB` (about1.90GB at8k,4.63GB at32k), plus normal host floor and2GiB outputs. New bf16 times remain unmeasured.

The input geometry/noise context was read from the local FACTS section 3 stage-2 record, including the client's KL 0.0014-0.0112 partition band. That band is a comparison reference, not an exactness or universal quality threshold.

## Corrected controls and custody

Initial configure/tool readiness found CMake absent from PATH; the existing local r1 CMake was used. Initial CMake configuration, including its brief compiler probes, overlapped another lane's compile activity; every full native target build waited until `pgrep -u agent cc1plus` found none, checked `uptime`, and ran niced at -j4. No other session's process was stopped. This records the configuration overlap rather than claiming complete compilation exclusivity for that first probe.

Build 01 exited 2 because one local-only scorer call lacked the new dtype argument; fixed, with subsequent builds green. First proxy unit run exited 1 because a default f32 request acquired an extra field; default request shape was restored. First hetero unit run exited 1 because a legacy direct fixture omitted dtype; f32 default handling was retained. The initial full RULER unit run exited 1 under system Python missing the already installed RULER dependencies; the existing local venv passed. Initial evaluator type check exited 1 for mixed-dictionary inference and the unselected venv; annotations and explicit environment selection fixed it. Initial HTTP harness exited 1 due a test-only nonexistent store accessor after the product import had passed; corrected to the existing `occupant` API. The first manual bf16 checker command exited 2 for its omitted final CPU-layer argument; the corrected command explicitly selects 0 and returns the expected 5. All logs remain available; none of these red attempts is counted as green evidence.

Sampled disk remained above the 28GB floor, with the lowest observed free value about 31.3GB. Task-owned native children record PID/exit in fixture process/cleanup files and were reaped. Build version metadata was configured at the base commit; tested source hashes bind the reviewed working-tree changes that are committed on this branch. The final source/binary hash ledger is `.temp/tested-identity.json`. Build trees and large generated binary fixtures are removed after validation; small logs/JSON results/hashes remain. Cleanup and final signature/head evidence are recorded in `.temp/cleanup.json` and `.temp/delivery.json`.

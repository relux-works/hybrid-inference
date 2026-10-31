# M1 B review fixes and bounded append qualification

Ready for review. Local signed delivery on `m1-hybrid-bench`; no push, PR, acceptance or landing is claimed. A's node branch is not merged; that still requires the owner's instruction.

## Source and boundaries

Reviewed base: `20964cee820951d904606354e4049b64bcf4104c`. Implementation commits, each verified signature `G`:

- `dd414ff865f8dd143c905157904dab49906d8d37`: chunk loop, session release, identity admission, checker replay and launcher fixes.
- `6464cda054a6634837956f5eeabe441d414de7e3`: derived-limit fixture and distinct quantization/backend reporting.
- `154b8e7a43d4876e1347be0eefa8876343307ad4`: legacy replay exclusion, checker child cleanup, evicted-session cleanup and native middle-chunk refusal regression.

All tracked changes are in `tools/hybrid-proxy/` or `tools/hybrid-bench/`. No `src/`, server or node edits, no A workbench edits, no downloads, no new dependencies. The owner explicitly authorized continuing this branch and signed local commits rather than starting from main or publishing it. No task-board is configured; the existing `.temp/tasks.md` tracks the work.

Self-review used the repository code-review skill's Scope, Security, Approach, General and Tests checklists. All previous reviewer test names remain. There is no surface table in this brief; that is a brief gap, not exhaustive coverage. No rows were silently treated as out of contract: the acceptance limits below identify the unqualified target environments. Independent acceptance remains pending.

## Review disposition

| Finding | Change and verification |
|---|---|
| B1 | Each mode releases its current/retired remote sessions in `finally`; closed sessions cannot replay. Each sequential checker deletes its own session, accepting an eviction's 404. Planner reserves two simultaneous sessions, not all sequential modes. Both homogeneous native runs exited 0 and left the remote list empty; budget refusals remain identifiable. |
| M1 | Failed startup moves logs into a private `<state>.failed-*` directory before deleting state. Actual old-node/new-flags refusal exited 2 and left readable `prefill.log`; log preservation unit control passed. Those owned test logs were later hashed and removed during cleanup. |
| M2 | `--check` now runs `fresh` and `append_chain`. The latter replays observed critical endpoints and intervening shadow spans; the checker imports nonzero-n0 deltas and builds matching partition references. Native replay imported 16 deltas, 64/64, KL below 0.01. It replays recorded tokens/boundaries rather than capturing the measured session's native bytes. |
| M3 | Cross-backend results use `uncalibrated`/`uncalibrated_outside_reference`, exit 5, preserving metrics. Same-partition equality is informational in that classification. Unit positive/outside-reference controls passed. Heterogeneous runs use the same uncalibrated classification with a separate quantization marker. |
| m1 | Default lock is `build.lock` beside state. Non-HEAD source/path fixture verifies it without a host-specific home layout. |
| m2 | Ordinary GET remains HTTP 501; authenticated benchmark health remains opt-in. HTTP regression passed. |
| m3 | Default SSE has no `hybrid_bench` and exactly one DONE; direct replay exclusion test passed. |
| m4 | Mode marker persists before generation, survives interruption, and preserves legacy local meters across upgrade. Ordinary subsequent requests explicitly clear that exclusion. Named interrupted/release regression passed. |
| m5 | README explicitly budgets the extra checker model and four contexts while the decoder remains resident; decoder sizing alone does not guarantee checker fit. No GPU allocation claim. |
| m6 | Both info files record threads=2, batch threads=2, batch/ubatch=256. README states the possible CPU/GPU throughput bias and partition coupling. |
| m7 | README records sampling's observed mean bias (approximately +14% tool tokens, +13% output tokens); source aggregates and seeded sampler are unchanged. |
| m8 | Runtime `Exception` and `KeyboardInterrupt` produce aborted/interrupted bundles after valid metadata admission; interrupt exit is 130. Checker child has a recorded PID and is stopped on interruption. Bundle and child-cleanup controls passed. |
| m9 | Trace includes `ram_budget`, HTTP status class, transport or invalid-delta fallback reason. A real second-chunk 503 control preserves the explicit budget reason and valid native prefix. |
| m10 | Node download probe requires launcher `--bench-link`. Without it, the benchmark records a skipped probe. Launcher command fixture verifies default omission and opt-in inclusion. |
| m11 | Default build directory follows the selected source worktree, avoiding a different source's CMake cache. Non-HEAD build-path regression passed. |

The launcher derives node `--max-ctx`, `--ram-budget-mib`, `--max-request-mib` from actual GGUF/card/host bounds, records the two-session divisor and derived values, and records live limits when supplied by the new node. Direct native node defaults and wire constants were not edited. The checker retains its default 16k bound for old fixtures and accepts an explicit configured context; its private JSON input has a bounded 8 MiB reader for long token lists.

Critical extends split `[local_n,N-1)` under both the unchanged 512 MiB native-body and 1 MiB header bounds. Each chunk has its own return_delta/import/prefix validation and trace. Previously verified imports remain usable after a later refusal; local continuation and existing replay/resync reconcile the session. No large-delta wire change or whole-context delta allocation is introduced.

Heterogeneous mode is explicit on setup and proxy config. It checks matching version-1 layout fingerprints, validates the live node hash/fingerprint, then rebinds only the envelope model hash for the decoder. Native bytes and checksums remain unchanged and native import still validates layout. Whole snapshots are not rebound; an ahead node uses local continuation. Layout compatibility does not establish common base-weight provenance or accuracy equivalence.

## CPU numerical results

All primary runs used context 2048, two CPU threads and K=64, with three modes in one run and eight calls per mode. Both homogeneous runs exited 0; every mode release succeeded and the remote session list was empty afterward. The first forced-offload call in the 64-token-chunk run used three chunks. Checks were at observed depth 550, after several append turns.

| Run | Check | Deltas | Same partition | Whole top-1 | Mean KL nats/token | Status / exit |
|---|---|---:|---:|---:|---:|---|
| Homogeneous default chunk limit (unchunked at this depth) | fresh | 1 | 64/64 | 64/64 | 0.0000489421153 | passed / 0 |
| Homogeneous default chunk limit | append chain | 11 | 64/64 | 64/64 | 0.0000631833691 | passed / 0 |
| Homogeneous 64-token chunks | fresh | 9 | 64/64 | 64/64 | 0.0000489421153 | passed / 0 |
| Homogeneous 64-token chunks | append chain | 16 | 64/64 | 64/64 | 0.0000414602480 | passed / 0 |
| Q8_0 node -> Q4_0 client, 64-token chunks | fresh | 9 | 64/64 | 64/64 | 0.000222497349 | uncalibrated / 5 |
| Q8_0 node -> Q4_0 client | append chain | 16 | 64/64 | 64/64 | 0.000222497349 | uncalibrated / 5 |

These are separate local numerical references, not comparisons against the imported result itself. Reference thresholds are mean KL <=0.01 and top-1 >=95%; homogeneous same-backend same-partition also requires K/K. The homogeneous chunked and unchunked fresh checks have identical observed metrics. No timing speedup is claimed. Small hybrid turns still perform synchronous shadow work, so they can cost roughly twice the local baseline.

Model identities:

| Role/file | Bytes | SHA-256 | Type |
|---|---:|---|---|
| Client / `Qwen3.5-0.8B-Q4_0.gguf` | 507154688 | `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c` | Q4_0 |
| Heterogeneous node / `Qwen3.5-0.8B-Q8_0-r16.gguf` | 811843840 | `f543fad263339779c3a38a21a531c6adbffcfdecdf5f749dd542de924b1e7302` | Q8_0 |

Both GGUF-derived layout fingerprints: `33d9f73396507b1f8402bf1e439936f42caee2c8875fb46e127b06e0e656d2fa`. The Q8 file was made locally with `build/bin/llama-quantize --allow-requantize INPUT OUTPUT Q8_0 2` (exit 0) and retained in `~/Werkstatt/hybrid-models/` as authorized. It is a compatibility fixture made from Q4 weights, not recovered source precision. Another owner's same-hash requantized file was observed read-only and preserved.

**Node integration limit:** all session computation, deltas and decoder imports above used the current native node. For heterogeneous setup, a test-only HTTP bridge added the new health fingerprint field from the loaded GGUF's layout calculation; state/session requests were forwarded unchanged. The matching algorithm was read from A's working tree. This proves real cross-file state import behind a stubbed new health contract; it does not establish native A/B fingerprint interoperability. New launcher flags were checked with a command fixture and an expected old-node refusal, not a merged new-node deployment.

## Negative controls, regressions and exact exits

Evidence root is `.temp/hybrid-b-r16-261031/` (`RUN` below). Commands ran as standalone processes, without tee. Parent native ledgers preserve expanded launcher/benchmark arguments and exits; native refusal commands retain their real exit 2, unauthorized reads their HTTP 401, and uncalibrated results their exit 5.

| Command / evidence | Exit | Claim |
|---|---:|---|
| `python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v`, `proxy-tests-02.log` | 0 | 67 tests, 92.886s; all previous names preserved, default behavior and synthetic middle-chunk recovery included. |
| `python3 -m unittest discover -s tools/hybrid-proxy -p test_benchmark.py -v`, `proxy-focused-04.log` | 0 | Six focused tests after final test cleanup; foreign DELETE cannot release another client's session, legacy/interrupted local replay remains excluded. |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v`, `bench-unit-08.log` | 0 | 16 tests; limits/build paths, logs, budgets, identity, cross-backend status, exception/interrupt bundles and checker-child cleanup. |
| `python3 tools/hybrid-proxy/test_remote_append.py --binary build/bin/llama-server --prefill-binary build/bin/llama-hybrid-prefill --model $MODEL --artifacts $RUN/dumps/remote-append-fault-final`, `remote-append-04.log` | 0 | Four native PASS claims: checkpoint rewrite/retirement; local1+remote28=new29 now in 8-token chunks; explicit409/native-byte preservation and fresh fork; second-chunk budget refusal with first8 imported, exact local continuation and remote replay. |
| `python3 $RUN/native_r16.py`, `native-r16-01.log` | 1 | Unchunked run passed, then a distinct setup refusal for a recently stopped port aborted this harness; all engines stopped. Not a green parent run. |
| `python3 $RUN/native_r16_round2.py`, `native-r16-02.log` | 0 | Chunked all-mode/check run 0; r15 suite 0; heterogeneous default refusal 2; opt-in setup 0; numerical run 5 as expected. Distinct client ports used. |
| `python3 tools/hybrid-bench/test_native.py --client-state $RUN/dumps/round2/client-chunked --node-state $RUN/dumps/round2/node-q4 --model $MODEL --artifacts $RUN/dumps/round2/r15-negatives --output $RUN`, `round2-r15-refusals.log` | 0 | r15 checksum/model/public/context/download/authentication/unreachable/re-prefill/killed-node controls; refusal commands 2, auth reads 401; killed node was disposable and owned. |
| `python3 $RUN/fingerprint_control.py`, `fingerprint-control-01.log` | 0 | Actual CLI returns 2 with layout mismatch under hetero opt-in, without state/process creation. Nearby cross-file opt-in setup succeeded. |
| `python3 $RUN/checker_lifecycle.py`, `checker-lifecycle-01.log` | 0 | Final checker launcher records its child PID, computes native64/64 and releases its check session. |
| Python compilation and both `sh -n` entry points, `syntax-final-01.log`, `sh-node-01.log`, `sh-client-01.log` | 0 | Syntax only, not a substitute for runtime tests. |
| `git diff --check` | 0 | Whitespace check. |
| `ty --version`, `editorconfig-checker --version` | 127 / 127 | Tools absent; full external type/EditorConfig checks were not run. No installation/download attempted. |
| `python3 $RUN/audit_cleanup.py`, `cleanup-audit-01.log` | 0 | Bundle/hash/credential/process/port audit and owned scratch deletion. |

Each executable test's claim is stated beside it. Known-violation controls include inflated old-prefix work (`test_full_reprefill_control`), second-chunk RAM refusal, mismatched hashes/fingerprints, and the checker's built-in full-vocabulary KL controls (including changed distribution with unchanged top-1). No source-mutant campaign or universal coverage is claimed.

Earlier failures remain visible: focused rounds 01/03 exited 1 for fixture API/order mistakes; round 02 passed. The first real fault test (`remote-append-03.log`) exited 1 after reaching recovery because its assertion requested a token list absent from node GET; round 04 uses the documented length/prefix-hash contract and passed. These are corrected harness failures, not successful negative observations. Initial quantize help exits 1 with usage as expected. The actual new-flags/old-node command exits 2 and preserves logs.

## Build identity, bundles and cleanup

All builds were incremental CPU builds with the shared `~/Werkstatt/hybrid-models/.build.lock`, `-j2` and a continuously checked 25 GiB floor. Minimum recorded build free space: 40,163,524,608 bytes, above 26,843,545,600. Exact configure flags: `-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF`. Targets were the checker/quantizer initially, then node and server/checker through the launchers. Before/after executable/library hashes and build PIDs are in `build-record.json` and `build-records-final.json`.

The first diagnostic build honestly records `source_dirty=true` at base 20964cee82 and is not the final runtime qualification. Native launcher builds record clean sources at dd414ff865 and 6464cda054. Later changes affect only legacy/default replay markers, cleanup/PID handling, tests and docs; native checker computation, token partitioning, native source and model identities are unchanged. The final native fault regression exercises current proxy/store source, and the final checker lifecycle control executes current Python with the unchanged compiled checker. Earlier numerical and r15 evidence is reused within that explicit boundary.

Shareable archives (all verified against their own manifests and sidecars):

| Bundle stem | SHA-256 | Status |
|---|---|---|
| `hybrid-bench-native-unchunked-20261031T234027958220Z` | `43dd71fbaa85db2a0e3c26b2514af659b9eb97f62137fdfdba8b822a034a3cab` | passed 0 |
| `hybrid-bench-native-chunked-20261031T234223711246Z` | `68fb2c9a3101dfb39eac3bee321f002786835dc56b5543c7a3fd1724a130499c` | passed 0 |
| `hybrid-bench-native-hetero-20261031T234354517530Z` | `213232c23fc36a6674cbf5ccd1c09311cf7f16d6bf02697199c4b0dca4cce4cf` | uncalibrated 5 |
| `hybrid-bench-reprefill-control-20261031T234339150379Z` | `67f9dd702f97674aa479997b0498ecd7cc5041ecb12c30f4c41139bb9d566cef` | aborted 2, intended control |
| `hybrid-bench-killed-mid-run-20261031T234341931813Z` | `16b8da911f119c8b595afe05c676a2352c4bbaf8820c71aac7bc176d65d440b7` | aborted 2, intended control |

Every JSONL row includes both SHA-256 values, both quantization types and the hetero flag. All five archives passed credential exclusion checks. No native state, token IDs, raw conversations, config, private credentials or databases are bundled.

Cleanup hashed 151 closed scratch files / 1,511,687,559 bytes before deleting the owned `dumps/` tree. Another 80 checker artifacts / 1,276,878,525 bytes were hashed immediately before per-check deletion; those manifests are retained in `earlier-check-hashes.json`. Main manifest SHA-256: `d2abf56eae4d5600a6773da02e62241e61454e65e61671d88216fbb7b474e550`. Pure synthetic unit fixtures were automatically removed and are not included in these native-state counts. All 31 recorded PIDs were inactive and all 11 owned ports closed. Only the authorized r16 model fixture is intentionally retained outside scratch.

## Remaining acceptance limits

Per the brief's CPU/no-download limits and explicit future A-merge instruction, there was no 27B, 256k runtime, rented GPU, CUDA, Metal, A-node merge, independent re-review, hosted check or publication. Native A/B fingerprint and new-node deployment compatibility remain pending integration. Layout fingerprints do not identify base weights. Cross-backend and heterogeneous numerical thresholds remain uncalibrated. The checker holds another model and four contexts; long-context memory amplification and whole-session node save/load costs remain to be measured. This is a small CPU compatibility/regression result, not the section-11 50-prompt matrix or a production performance claim.

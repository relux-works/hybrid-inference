# Growing-session scenario, baseline and trace

The r38 section below supersedes r37 numerical acceptance; all earlier measurements remain historical. The original sections record r36. The [r37 revision](#r37-native-depth-and-session-measured-acceptance) below supersedes its depth and acceptance decisions; the original measurements remain historical evidence.

Implementation: `m3-growing-session`, based on exact main `528724c96894aa5f65e98b329317de59aa694afc`. Worktree: `/home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r36-261102/worktree`. The original `m3-import-speed-v2` checkout was not switched or edited. Implementation commits: `7dbb355b9d` and `cae2bca104`, signed by the configured Relux Nexus identity with the requested co-author trailer. No push or PR.

The scenario driver, paired local baseline, documented engine trace, CPU controls and CUDA/Mac runbooks are implemented. **The requested mixed-route same-backend exactness gate is not met.** Both measured CPU start modes exit 3 with complete traces and `status="failed"`; the acceptance harness exits 1. This is not a successful end-to-end paper experiment or an admitted review handoff. The exactness criterion and the uncapped 250k capacity choice remain [owner decisions](../../UNRESOLVED_QUESTIONS.md).

## Delivered behavior

- [Driver](growing_session.py): actual proxy handler/policy/journal/replay, one sequential session per arm, cold or pre-staged base, fixed-seed depth-conditioned input and measured output/gap sampling, greedy decode, and unchanged sequence replay on the same client for the local baseline.
- Real transport outage through a task-owned TCP relay, stopped/restarted by recorded PID; local progress during the outage and protocol catch-up before the next offload. The node process is preserved; RAM-loss restart is the separate inherited PR13 test.
- Content-free protocol events, policy inputs/estimates and actual route, TTFT component timings, delta-body bytes, catch-up/resync counts/time/body bytes, decode rate and committed journal hash. Attribution is by session, including asynchronous replay.
- Live native checkpoint snapshots at each tenth/final turn, scored over 64 teacher-forced full-vocabulary positions against the actual local session. Same-backend acceptance requires bit-exact logits. Different histories refuse comparison; CUDA-to-Metal is explicitly heterogeneous and uncalibrated.
- [Trace schema and runbooks](GROWING-SESSION.md): CPU controls, one-PRO-6000 CUDA rehearsal, Vast-to-M3-Max runs, pre-staging, resource/duration arithmetic and the 250k capacity constraint. Root and bench READMEs link the tools and outputs.

Snapshot scoring reuses `server_tokens::deserialize`; no new native snapshot parser or core/backend behavior is introduced. CMake links that existing server component only with `LLAMA_BUILD_SERVER=ON`. With it off, the old delta-chain checker still builds/runs, and snapshot requests explicitly refuse. Snapshot context capacity is bounded before allocation.

## CPU observations

Final observations are in this worktree's `.temp/native-05/`, with component summaries in `.temp/measurements.json` and independent trace audit `.temp/measurement-audit-02.log`. Python 3.14.4, CMake 4.4.4, GCC 15.2.0; existing Qwen3.5-0.8B-Q4_0 GGUF SHA-256 `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`. CPU-only binaries, two inference threads, context 4096, q8_0 KV, fixed fixture caps 64 input / 2 output, pauses zero, seed 1, D0=128, 12 turns, outage at turns 4-5, reconnection at 6.

The test policy uses 10 local tokens/s, 2000 server tokens/s, 4s fixed cost and a synthetic 1e12-byte/s initial link estimate to exercise both policy branches. These values are controls, not a measured target-hardware calibration. The timing samples are from a shared CPU host; they are not evidence of hybrid speedup.

Each arm evaluates 895 new tokens (including framing) and generates 23 tokens over the 12 measured turns, ending at depth 1046. All per-turn hybrid/local token journal hashes match. The independent audit verifies depth-conditioned sampling against actual prior depths, no old-prefix re-prefill/rollback, and an acknowledged node boundary before every successful first critical append. Both modes leave the node session list empty and record no failed remote cleanup.

| Start | Arm | Local / offload turns | Mean turn TTFT (s) | Sum call wall time (s) | Delta response bytes |
|---|---|---:|---:|---:|---:|
| Cold | Hybrid | 2 / 10 | 0.733335 | 9.935849 | 207073951 |
| Cold | Local | 12 / 0 | 0.416632 | 6.091462 | 0 |
| Pre-staged | Hybrid | 2 / 10 | 0.740888 | 10.029851 | 207073951 |
| Pre-staged | Local | 12 / 0 | 0.415519 | 6.072043 | 0 |

This table excludes start rows. Call wall time includes journal reconciliation and checkpoint saves, but not the separately executed distribution checker. Pre-stage save/restore and base fill costs remain separate in the trace. The baseline has zero catch-up tokens/time/bytes and zero offload bytes. Hybrid background/call catch-up covers 147 tokens in cold (2071 JSON upload bytes) and 145 during the pre-staged measured window (1930 bytes); two base tokens were caught up before the pre-staged window. Catch-up time overlaps other work and is not added to TTFT.

The first offload after reconnection catches up 121 journal tokens and sends 770 JSON body bytes. Protocol recovery time is 0.549791s cold / 0.525060s pre-staged, of which the catch-up extend is 0.538593s / 0.522070s. These are application timings/body sizes, not TCP/SSH/5G traffic counts.

| Checkpoint | Depth | Top-1 | Mean KL(local \|\| hybrid), nats/token | Max absolute logit difference | Exact gate / native exit |
|---|---:|---:|---:|---:|---|
| Turn 10, both starts | 886 | 64/64 | 0.000667384221574 | 0.852364540100 | Failed / 3 |
| Turn 12, both starts | 1046 | 64/64 | 0.000334264341730 | 0.773005485535 | Failed / 3 |

The two start modes give identical checkpoint numerical metrics. Token-journal agreement does not imply state/logit bit equality. Partition sensitivity is a plausible explanation, also exposed by the existing independent local-noise control; no kernel-level root-cause proof is claimed. Exact same-snapshot and same-partition controls pass. Neither the historical KL<=0.01 / top-1>=95% rule nor the existing repeat-floor rule was substituted for this experiment's requested exact gate.

Final trace SHA-256 values:

- Cold: `7be898212e65d69045beb6a4f8d2f5ffc3c3e143043fba6293e0c9f42ff45b90`.
- Pre-staged: `4babec9f6a5f7da21efceb49f1a51885bbecfa19bd9fa2fde17c45bf0a49ef5f`.

## Validation and real exit codes

Commands ran as standalone processes with their real statuses retained. No validation command was piped through tee. Exact build arguments, PIDs, binary identities and disk observations are retained in `.temp/build-driver-01.log`, `.temp/build-driver-02.log`, `.temp/cpu-03-build-record.json`, `.temp/no-server-build-record.json` and the private native metadata. All builds used the shared model-directory lock, nice 19 and at most four jobs, one build at a time. Existing tools/models were reused; no installs or downloads.

| Command / scope | Actual result | Evidence |
|---|---|---|
| CPU configure/build of server, node and checker | 0 | `build-driver-01.log`, `cpu-03-build-01.log` |
| `LLAMA_BUILD_SERVER=OFF` checker configure/build | 0 | `no-server-driver-01.log` |
| `python3 -m unittest discover -s tools/hybrid-bench -p 'test_*.py' -v` | 0; 76 tests, 6 native-environment skips | `bench-regression-02.log` |
| `python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v` | 0; all 73 tests, including the unchanged PR13 restart regression | `proxy-regression-01.log` |
| `test_growing_session_native.py --build-dir .temp/build --model LOCAL_GGUF --artifacts .temp/native-05` | **1; acceptance remains red** | `native-05.log`, both driver commands exit **3** in `native-05/commands.json` |
| Native same-live-snapshot 64-position exactness control | 0; exact, KL=0, top-1=64/64 | `native-05/checkpoint-controls/commands.json` |
| Native checksum, token-journal and context-capacity refusal controls | **2 each**, intended refusals with no result file | Same command ledger and named logs |
| Existing delta-chain/noise/corruption native suite | 0 overall; intended child exits 2 and 3 retained | `legacy-checker-01.log`, `legacy-checker-01/equivalence/*.command.json` |
| Final server-on and server-off legacy paths | 0 / 0; same-partition exact 8/8 | `configuration-checks-01.log` and command ledger |
| Server-off snapshot request | **2**, explicit unsupported-build refusal, no metrics | `configuration-checks/server-off-snapshot.log` |
| Trace audit: exact journals, route/outage/reconnect, no re-prefill, session attribution, cleanup | 0 | `measurement-audit-02.log` |
| Python syntax, ASCII additions and whitespace checks | 0 | Final scope audit |

Six bench skips require the explicitly supplied native identity/binary environments; they are not passing native tests. The opt-in native command above supplies this experiment's model/binaries directly. Full upstream CI, ty and editorconfig-checker were not run; the latter tools are absent from PATH. CUDA, Metal, GPU timing, 100k/250k CPU inference and any real network run were not executed on agents1.

Direct negative/control evidence is bounded to these tests; no broad mutation-coverage claim is made:

| Fault/control | Property checked | Named test/control | Observed result and limit |
|---|---|---|---|
| Explicit history disagrees with durable journal | No import or durable/native partial change | `ProtocolTests.test_journal_mismatch_refuses` | HTTP409, unchanged valid state; neighboring valid append succeeds |
| Remove resync after local outage turns | Offload cannot use a stale remote boundary | `ProtocolTests.test_disconnect_without_resync_refuses_offload` | No import, local fallback/http409; real sync then admits the next append; synthetic transport fixture |
| Corrupt a catch-up suffix with valid JSON | Prefix mismatch cannot reach critical import | `ProtocolTests.test_corrupted_catchup_refuses` | HTTP409 before import; durable/native valid prefix unchanged; recreated remote is positive control |
| Relax exactness to nonzero KL / unchanged top-1 | A weak verdict cannot masquerade as exact | `ScenarioTests.test_checkpoint_exactness_gate` | Intended Refusal assertions; valid exact and labelled heterogeneous controls retained |
| Interleave another session's background replay | Baseline cannot inherit hybrid catch-up counters | `ScenarioTests.test_catchup_attribution` | Other-session events excluded; native baseline counters independently audited as zero |
| Wrong native snapshot digest / token journal / excessive context | Checkpoint input validation precedes successful metrics | `checksum_refusal`, `journal_refusal`, `context_capacity_refusal` | Intended native exit2 and no output, alongside the valid exact-state control |
| Checksum-valid imported loss preserves argmax | Full-logit gate must reject despite top-1 agreement | Existing `test_equivalence_noise`, `valid-lossy` | Native exit3 with same-partition nonzero KL/difference and 8/8 argmax; old checker path only |

Initial/corrected observations are preserved. Unit attempt 01 exited 1 because its positive assertion used a meter field absent in the legacy synthetic fixture; corrected assertions check actual offload/import behavior. Native attempt 01 exited 1 after its child refused packed server-token metadata (checker2); it was corrected by using the existing server deserializer. Native 02-05 remain red for the exactness requirement, not setup errors. An audit of attempt 04 found pre-staged hybrid replay charged to the local baseline; session attribution and a named regression were added, then attempt 05 reran both full arms. Earlier timings/counters are not promoted as final evidence.

## Remaining decisions and limits

Default seed1/40 from 100k draws about 15.7k fixture input plus 27.1k output tokens, plus framing and about 789s of pauses per arm. From 250k it requires about 293k final capacity. The existing launcher caps context at GGUF training capacity (262144 for the available model). The runbook records two clean options: explicit 128/128 caps with all truncations exported, or a model/context already qualified beyond 300k. No extrapolation flag, silent clamp or forced offload was added. A capped run may naturally choose only local and then fails the required mixed-route gate.

The input depth bins publish only p50/p90, so the documented bounded CDF does not reproduce an unknown conditional tail or input/output/pause correlations. This fixture is deliberately content-neutral; no task-quality result is implied. Physical wire counts, persistent node RAM loss, arbitrary disconnects during native import, different quantizations and differing generated histories are outside this finite run. Exported traces report heterogeneous measurements only when histories still match; divergent histories abort rather than yielding a misleading equivalence number.

No surface table was supplied (brief gap). No requested row is declared out of contract; mixed-route exactness remains an unmet requested property. Self-review covered scope, approach, security, tests and general conventions. Native core/backend source is unchanged. Changes outside `tools/hybrid-bench/` are limited to the checker CMake linkage, README links and owner-decision document, with those reasons stated here.

At the 2026-11-02T01:06:09Z cleanup audit, all 81 recorded PIDs were inactive. Task-owned build trees, large snapshot/delta fixtures and expired fixture credentials were removed (787112608 bytes). Logs, trace exports, small inputs, hashes and command ledgers remain. Free disk was 33322864640 bytes before cleanup and 34121883648 afterward, above the 28GB floor. Cleanup evidence and pre-removal binary/source identities are `.temp/cleanup.json` and `.temp/pre-cleanup-identity.json`. The task-1 progress notice was delivered to `agents1_hybrid_orch` in `#hybrid`, topic `prototype-go` (`a2a-progress-02.log`, exit0). The first attempted kind `progress` was rejected and corrected to `notice`; only the successful notice is credited. The final short notice carries the signed head and the two unresolved gates. No task-board exists in this checkout; the persistent task plan is `.temp/tasks.md`.

## r37: native depth and session-measured acceptance

Scope: `hybrid-b-r37-261102`, continuing `m3-growing-session` in the same r36 worktree from signed head `2918d3d08b93613c28b9b96a4fc2cb03c006699c`, as requested. No merge/rebase, push, PR, GPU use or model download. All changes are in `tools/hybrid-bench/` except the root decision document, which records the resolved owner choices and remaining numerical acceptance gap. The original checkout remains unchanged.

The implementation, fresh CPU measurements and report are available for review. **Both start modes still fail numerical acceptance at turn 10 under the new criterion.** Turn 12 passes. Neither failure is relabelled as success; native driver exits are 3 and the acceptance harness exits 1. No partition settings were tuned after observing the band. This is a reproducible remaining numerical gap, not an accepted hardware experiment.

### Native depth

`--depth 100000` is retained; `--depth deep` derives D0 from the full sampled sequence, without YaRN. Preflight measures actual incremental ChatML framing using the existing renderer/tokenizer. For each candidate D0, it samples all N turns at their planned committed depths and includes every input delta, framing token and output. Descending 1024-token candidates select the largest start satisfying D0 + growth + 2048 <=262144. A sequence that cannot fit at 100000 refuses explicitly; no turn is dropped. Actual per-turn framing, conditioning depths and final depths must match the plan or the run aborts.

With seed 1, 40 turns, default uncapped quantile limits and this model's measured 14 framing tokens per turn:

| Arm | D0 | Input plus framing | Outputs | Total growth | Final depth | Final plus margin |
|---|---:|---:|---:|---:|---:|---:|
| 100k | 100000 | 16297 | 27142 | 43439 | 143439 | 145487 |
| Deep | 216064 | 16164 | 27142 | 43306 | 259370 | 261418 |

The next multiple, 217088, would end at 260394 and require 262442 including margin, so it refuses. These are exact sampling/admission calculations, not long-context inference measurements. The trace schema is now `hybrid-growing-session-v2`: `depth_plan` records D0, growth, framing, final depth, capacity and margin; `final_depth` records the verified actual depth per arm. Both roles must allocate 262144 for the deep arm. CPU smoke remains explicitly depth128/context4096.

### Measured criterion and CPU checkpoints

The original local baseline remains 256/256 batch/ubatch. A second fully local session restarts the decoder at 64/32 and replays the same sequence on the same model/backend with identical generation settings. Each arm has its own native state, base and journal; the partition base is prepared outside its pre-staged measurement window. All turn hashes must match across all three arms, including the starts. Snapshot hashes and actual native token journals are checked before comparison.

The checker compares 64 full-vocabulary positions with local-greedy teacher forcing. Define H = mean KL(local || hybrid), B = mean KL(local || local_partition), and P95 = nearest-rank 95th percentile of the 64 per-position partition KL values (sorted element61). Accept only H <=max(B,P95), top-1 >=63/64, and no differing token over 64 independent greedy steps restored from the hybrid/local checkpoints. Export H, B, H/B, P95, the limit and all position metrics. Zero B has a null ratio with an explicit both-zero or zero-band status. Heterogeneous backends still exit5 with uncalibrated measurements. Legacy two-snapshot inputs retain strict exactness.

Fresh final evidence is `.temp/r37/native-02/`, bound to `.temp/r37/qualification-identity.json`; the independent audit is `.temp/r37/audit-02.log` and its summary is `measurements-02.json`. Python 3.14.4, CMake 4.4.4, GCC 15.2.0, CPU only, two inference threads, q8_0 KV, existing Qwen3.5-0.8B-Q4_0 SHA256 `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`. Smoke caps 64 input/2 output, seed 1, pauses 0, D0=128, 12 turns, outage 4-5 and reconnect 6. The policy is the same synthetic routing control as r36, not a hardware calibration.

| Start | Checkpoint / depth | H: hybrid KL | B: partition KL | H/B | P95 / KL limit | Top-1 / free-running | Verdict / native exit |
|---|---|---:|---:|---:|---:|---|---|
| cold | 10 / 886 | 0.000667384221574 | 0.000148168021645 | 4.50423927 | 0.000393689511876 | 64/64 / no divergence in 64 | failed / 3 |
| cold | 12 / 1046 | 0.00033426434173 | 7.13366915975e-05 | 4.68572812 | 0.000353664365071 | 64/64 / no divergence in 64 | passed / 0 |
| pre-staged | 10 / 886 | 0.000667384221574 | 0.000148168021645 | 4.50423927 | 0.000393689511876 | 64/64 / no divergence in 64 | failed / 3 |
| pre-staged | 12 / 1046 | 0.00033426434173 | 7.13366915975e-05 | 4.68572812 | 0.000353664365071 | 64/64 / no divergence in 64 | passed / 0 |

KL is nats/token. The max(B,P95) limit equals P95 at both checkpoints. H is 1.69520 times the limit at turn 10 and 0.945146 times the limit at turn 12. The local-partition mean ratio is reported separately above; it is not the acceptance limit ratio. Both hybrid H values reproduce r36 exactly. This supports the expected batching sensitivity but does not make the failed bound pass or establish a kernel-level root cause.

Each start has 39 committed calls: three bases plus 12 turns per arm. Every arm adds 895 input/framing tokens and 23 output tokens, ending at 1046; all per-turn journal hashes are identical. Hybrid takes 2 local/10 offload turns and recovers a 121-token suffix after reconnection. Both local sessions use only local work, with zero remote bytes and zero catch-up counters. The audit also verifies no old-prefix re-prefill/rollback, acknowledged remote boundaries before critical appends, and empty remote cleanup. The native harness independently reads the node session list as empty.

### Decode catch-up decision and batch timing

No `--catchup-mode decode` option is added. `tools/hybrid-proxy/proxy.py::sync` sends `session["tokens"][n:]` as one flat `return_delta=false` append. `tools/hybrid/prefill.cpp` receives tokens, n0 and hashes but no evaluation schedule. During the outage the suffix contains locally batched prompt deltas as well as generated tokens; whole-history replay after node RAM loss has the same issue. Evaluating all catch-up tokens one by one would also change the prompt path and cannot reproduce the client's state. Clean exact replay needs durable prompt/decode boundaries, prefill partitions and reconciliation steps in the journal/protocol. This revision preserves that contract instead of claiming exactness from a node-only switch.

Consequently decode-mode CPU exactness and a decode/batch timing ratio were not run and are not claimed. Batch catch-up timing is measured from the node's existing `X-Hybrid-Prefill-Ms`, exported as per-event `node_prefill_seconds` and per-call `catchup_node_seconds`. It excludes node state restore/save, serialization and HTTP work; wall time includes those. The measured turn windows are:

| Start | Catch-up tokens | Node evaluation (s) | Catch-up wall (s) | JSON upload bytes |
|---|---:|---:|---:|---:|
| cold | 147 | 0.731995416 | 0.852663109 | 2071 |
| pre-staged | 145 | 0.699232132 | 0.779897970 | 1930 |

The pre-staged base catches up two tokens outside that window. These shared-host CPU timings are diagnostic observations; they do not establish speedup or target-machine performance.

### Validation, controls and limits

All validation commands run directly with real statuses retained. The final source/binary identity file, build command ledgers, PID files and logs are under `.temp/r37/`. The unchanged r36 exact-state/checksum/journal/capacity fixture cases were executed afresh. No numerical tolerance was applied to them.

| Command / scope | Real result | Evidence under `.temp/r37/` |
|---|---|---|
| CPU configure | 0 | `cpu-01-build-record.json`, configure log |
| CPU final server/node/checker/identity build, nice 19, -j4 | 0 | `cpu-03-build-record.json`, `cpu-03-build-01.log` |
| `python3 -m unittest discover -s tools/hybrid-bench -p 'test_*.py' -v` | 0; 78 tests, 6 explicit environment skips | `bench-02.log` |
| `python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v` | 0; 73 tests, including the unchanged PR13 restart witness | `proxy-01.log` |
| `test_mirai_identity.py --binary .../llama-hybrid-check-identity --headers /home/agent/lanes/hybrid/.temp/research/bonsai/gguf-dumps --artifacts .../identity -v` | 0; 12 tests, no skips | `identity-01.log`, `identity/` |
| `test_growing_session_native.py --build-dir .temp/r37/build --model LOCAL_GGUF --artifacts .temp/r37/native-02` | **1; measured criterion remains red**, both start commands **3** | `native-02.log`, `native-02/commands.json` |
| Native same-live-state exactness control | 0; bit-exact 64/64, KL0 | `native-02/checkpoint-controls/commands.json` |
| Native checksum, journal and capacity refusals | **2 each**, intended diagnostic, no metrics file | Same ledger and named logs |
| Existing delta-chain/noise/corruption CPU suite, `test_integrated.py --case noise` | 0 overall; intended child 2/3 retained | `legacy-01.log`, `legacy-01/equivalence/*.command.json` |
| Independent trace/depth/journal/route/cleanup audit | 0 | `audit-02.log`, `measurements-02.json` |

`ScenarioTests.test_native_depth_plan` checks maximality against every higher 1024 candidate, complete turn retention, both 100k/deep overflow refusals, and a rounding boundary near 100k. `test_checkpoint_partition_band` tests both valid/failed numerical outcomes, the inclusive bound, top-1 loss, free-running divergence, malformed/missing position metrics, altered p95/limit/ratio and zero-band behavior. Existing `test_checkpoint_exactness_gate`, catch-up attribution and protocol refusal tests retain their names and assertions. Native checker startup controls separately accept each measured bound and reject KL excess, top-1 loss, free-running divergence and nonzero KL against a zero band. These are bounded positive/negative controls; no broad mutation-coverage claim is made and no narrowing mutants were run in r37.

The initial oversized-sequence unit probe failed 1 with StopIteration beyond the finite last depth-bin marker; the sampler now retains the terminal published bin and reaches the intended explicit capacity refusal. Initial build driver exited 1 after build exit 2 because the identity target was mistyped; server/node/checker had built, and the corrected full target command exited 0. Exploratory native-01 predates final source/control refinements; only native-02 is credited above. Its measured checkpoint numbers repeat in native-02. Required suite names are retained; full upstream CI, CUDA/Metal, long-context inference, server-off rebuild, ty and editorconfig-checker were not run in r37. No new dependencies or tools were installed.

The new runbook updates native depth, three-session order, three checker contexts, doubled retained checkpoint count, disk admission, runtime estimates, trace fields and current numerical rule. No surface table was supplied (brief gap). No requested row is declared out of contract: decode catch-up follows the explicitly permitted exclusion path with its constraint documented; numerical acceptance remains failed. Self-review covered scope, approach, security, tests and general conventions. No backend, native identity, journal, proxy or node implementation changed. Independent acceptance and GPU/Mac measurements remain unclaimed.

Final trace SHA256 values:

- cold: `e96d2b415d7452b4d78c1d7011d8a83b562beac06e90da3a19ccf654b72cf092`.

- pre-staged: `63ceea7421cdf145c0535862512ec715c6da858c8ed46dbda4544e37b58ce992`.

Cleanup at 2026-11-02T01:24:56.531899+00:00: all 51 recorded PIDs inactive; removed this task's build tree (123817420 bytes), native snapshots/deltas and expired fixture credentials. Source and model files are preserved. Logs, trace exports, command ledgers and SHA256 identities remain. Free disk before/after cleanup: 33393344512 / 33887891456 bytes. The smallest recorded build sample was 33311567872 bytes, above the 28GB floor. Cleanup evidence: `.temp/r37/cleanup.json` and `cleanup-01.log` (exit 0).

## r38: factorial replay and pre-hardware criterion

Rework of Muse r7r, bound to incoming 5c6fcfe6, run hybrid-b-r38-261102. This section supersedes r37 numerical acceptance. **The generated-token-batching-only attribution is refuted as a sufficient explanation on these CPU fixtures. No factorial cell reproduces the live hybrid state across both checkpoints.** It does not exclude batching as one contributor. The implementation and evidence are ready for review; numerical/hardware qualification remains failed. No GPU, model download, push or upstream submission.

### F1: control and realistic-output observations

The checker now builds four fresh contexts from retained check tokens and trace boundaries. A uses local prompt processing (256-token batches, logits only at prompt end) and one-token generated decode. B1 changes generated spans to the exact node catch-up batch intersections; B2 changes prompt spans to node critical/catch-up chunk intersections and chunk-end logits, keeping generated decode. B preserves complete node batches, including mixed prompt/generated outage batches, through the last imported boundary, then evaluates the client's remaining tail. A one-factor cell must split a batch that crosses a prompt/generated boundary; only B preserves that interaction. All schedules must tile0..checkpoint exactly; unsupported primary local partitions refuse. No production replay/import behavior changed.

Every fresh live comparison uses the same 64 local-greedy teacher-forced IDs; the checker also restores each state for an independent 64-token greedy continuation. A-L checks local replay against the live local checkpoint. B-H checks full evaluation-only node replay against the live hybrid checkpoint. The checker retains hashed little-endian float32 full-vocabulary logits for L/H/P/A/B1/B2/B. It scores sequential contexts, rather than holding all replay contexts at once.

Final evidence: `.temp/r38/native-03/`, `.temp/r38/qualification-identity.json`, `.temp/r38/measurements.json`, and trace/position/hash audit02. Existing local Qwen3.5-0.8B-Q4_0 SHA-256 `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`; CPU, q8_0 KV, flash attention, two inference threads, original 256/256 client and 64/32 partition baseline, context 4096. No new tool/library dependency. NumPy readiness was unavailable; the independent audit uses existing Python stdlib arrays/math.

The realistic-output smoke samples the measured output CDF with cap 128 and input cap 48, fixed seed 281, 12 turns, pauses capped 1.1 s, outage 4-5/reconnect 6. Answer lengths are 30..128, not two tokens. Each arm adds 692 prompt/framing tokens and 1223 generated tokens: generated share 63.8642%, comparable to the paper workload's62%; final depth 2043 plus 2048 margin is 4091<=4096. It has 983 generated/1551 growth tokens at checkpoint 10 (63.3785%). The caps, seed, pauses and route schedule differ from r37, so this is a realistic-share scaling probe, not an output-length-only or monotonic scaling proof.

The numerical rows below were independently observed in both cold and pre-staged starts and are exactly identical across starts; the table applies to each start. `H_cell` is the mean KL for the named ordered pair, `band` is the diagnostic mean KL(L||P) at that checkpoint, and ratio=H_cell/band. H is the actual live hybrid only in the L-H row. B denotes the full-node factorial cell, not the partition band. p95 is nearest rank across 64 positions (sorted element 61). All KL units are nats/token. A zero band exports a null ratio with explicit both-zero/zero-band status, never an inferred infinite tolerance. Free column is the first independent greedy divergence (-1 means none).

Checkpoint 10, depth 1679, generated/growth=983/1551; both starts:

| Pair / cell | H_cell | Band | Ratio | p95 | Max abs logit difference | Top-1 | Free |
|---|---:|---:|---:|---:|---:|---:|---:|
| L-H | 0.000637837220046 | 0 | null (zero band) | 0.00186315189913 | 0.964870452881 | 64/64 | -1 |
| L-P | 0 | 0 | null (0/0) | 0 | 0 | 64/64 | -1 |
| A-A | 0 | 0 | null (0/0) | 0 | 0 | 64/64 | -1 |
| A-B1 | 0.00018071129068 | 0 | null (zero band) | 0.000879184946625 | 0.866317749023 | 64/64 | -1 |
| A-B2 | 8.29298923694e-05 | 0 | null (zero band) | 0.000546047878833 | 1.11637067795 | 64/64 | -1 |
| A-B | 0.000563020461777 | 0 | null (zero band) | 0.00121762819219 | 0.942632198334 | 64/64 | -1 |
| A-L | 0 | 0 | null (0/0) | 0 | 0 | 64/64 | -1 |
| B-H | 7.71985333402e-05 | 0 | null (zero band) | 0.000302589110353 | 0.792877197266 | 64/64 | -1 |

Checkpoint 12, depth 2043, generated/growth=1223/1915; both starts:

| Pair / cell | H_cell | Band | Ratio | p95 | Max abs logit difference | Top-1 | Free |
|---|---:|---:|---:|---:|---:|---:|---:|
| L-H | 0.00027947834459 | 0 | null (zero band) | 0.00100390301939 | 0.737212657928 | 63/64 | 0 |
| L-P | 0 | 0 | null (0/0) | 0 | 0 | 64/64 | -1 |
| A-A | 0 | 0 | null (0/0) | 0 | 0 | 64/64 | -1 |
| A-B1 | 0.000248037221206 | 0 | null (zero band) | 0.000306196323783 | 0.70611333847 | 64/64 | -1 |
| A-B2 | 0.00041999735555 | 0 | null (zero band) | 0.00138579233125 | 0.869745731354 | 63/64 | 0 |
| A-B | 0.000245036923118 | 0 | null (zero band) | 0.000767767496852 | 0.624326705933 | 64/64 | -1 |
| A-L | 0 | 0 | null (0/0) | 0 | 0 | 64/64 | -1 |
| B-H | 0.000274473850817 | 0 | null (zero band) | 0.00101512660845 | 0.771906852722 | 63/64 | 0 |

A-L and A-A are bit-exact controls at both checkpoints. At 10, generated-only B1 has mean 0.000180711 versus live H 0.000637837; full B is closer in mean but still has KL(B||H)=0.0000771985 and max logit difference 0.792877. At 12, B1's mean is closer to H, yet B-H remains 0.000274474 against H 0.000279478, and B-H diverges at the first free token. B2 shares the live turn 12 initial greedy divergence, but its distribution and maximum logit error differ. Similar mean magnitudes do not reproduce a state. KL ratios are magnitude comparisons, not additive causal shares. There is no single cell that explains the growing hybrid deviation over both checkpoints; the residual leaves node save/load/import/recurrent-state semantics as further factors, without selecting one as the root cause.

Retained r37 replay controls use original `.temp/r37/native-02/cold/check-{10,12}/input.json` and trace boundaries through the new tracked `growing_session.py replay` command. Outputs/commands/hashes are `.temp/r38/replay-cli-{10,12}/`; both checker exits 0. Original live H and partition band below are read from the retained r37 trace, not remeasured from deleted snapshots. A-B rows are fresh replay measurements. B-H and A-L are unavailable for these historical snapshots; the live realistic rerun above supplies those comparisons. Do not promote their absence to exactness.

Retained r37 checkpoint 10, generated/growth=19/758:

| Pair / cell | H_cell | Band | Ratio | p95 | Max abs logit difference | Top-1 | Free |
|---|---:|---:|---:|---:|---:|---:|---:|
| L-H (historical) | 0.000667384221574 | 0.000148168021645 | 4.50423926948 | 0.00564551197203 | 0.8523645401 | 64/64 | -1 |
| A-A | 0 | 0.000148168021645 | 0 | 0 | 0 | 64/64 | -1 |
| A-B1 | 0 | 0.000148168021645 | 0 | 0 | 0 | 64/64 | -1 |
| A-B2 | 0.000147236156961 | 0.000148168021645 | 0.993710757059 | 0.00158836625987 | 0.76303935051 | 64/64 | -1 |
| A-B | 0.000163292555749 | 0.000148168021645 | 1.10207691198 | 0.000955864264354 | 0.874028205872 | 64/64 | -1 |

Retained r37 checkpoint 12, generated/growth=23/918:

| Pair / cell | H_cell | Band | Ratio | p95 | Max abs logit difference | Top-1 | Free |
|---|---:|---:|---:|---:|---:|---:|---:|
| L-H (historical) | 0.00033426434173 | 7.13366915975e-05 | 4.68572811893 | 0.00239973821519 | 0.773005485535 | 64/64 | -1 |
| A-A | 0 | 7.13366915975e-05 | 0 | 0 | 0 | 64/64 | -1 |
| A-B1 | 0 | 7.13366915975e-05 | 0 | 0 | 0 | 64/64 | -1 |
| A-B2 | 8.32481363698e-05 | 7.13366915975e-05 | 1.16697500971 | 0.000590594355794 | 0.707780838013 | 64/64 | -1 |
| A-B | 0.00015386676396 | 7.13366915975e-05 | 2.15690916573 | 0.000833947835197 | 0.78058218956 | 64/64 | -1 |

Generated-only batching is exactly zero on the old 2-token controls while the original H is positive; neither prompt-only nor full-node replay reaches the original H. This refutes the old untested sufficient-cause assertion. The richer-output controls produce nonzero generated-batch differences and a turn 12 free-running divergence, which the old cap hid. They establish finite fixture coverage, not long-context equivalence or task quality.

### F2: one rule before hardware

Every three-snapshot same-backend checkpoint now gates on **mean full-vocabulary KL(L||H)<=0**, plus top-1>=63/64 and no free-running divergence over 64 steps. Code/driver/runbook use `session_mean_kl_zero`, `kl_limit=0`; p95 never sets a threshold. The partition band is reported only. Since no reference class has been qualified by F1, no positive tolerance or substitute reference is justified; zero is a conservative admission rule, not a fit to the measured H. Legacy two-snapshot controls remain bit-exact. Heterogeneous backends remain exit 5/unqualified.

Both starts exit 3 and the native qualification harness exits 1 after fully checking all 39 calls/start and remote cleanup. Both checkpoints fail KL; turn 12 also fails independent free-running agreement (top-1 is 63/64, divergence0). Turn 10 has 64/64 and no free divergence despite max logit shift 0.964870. Hardware remains blocked by numerical qualification. A report-only KL split is not implemented.

### F3/F4: routing and rehearsal

The driver simulates the entire sampled policy/hysteresis sequence before any base or measured arm. It uses configured link rate, labels the constant-link approximation and accounts for outage resets; live EMA can change decisions. Real native all-local and all-remote controls each exit 2 for the intended degeneracy, with `calls=[]`. Refusal is verified before any arm; decoder/tokenizer initialization is allowed. The live gate counts connected policy-local turns without fallback, successful connected policy-offloads, and connected catch-up of a locally evaluated prompt separately from fallback reasons.

| Start | Predicted connected local/offload | Actual connected policy-local/offload | Outage fallbacks | Connected local prompt catch-up | Background catch-up events | Reconnect suffix |
|---|---|---|---|---|---:|---|
| Cold | 2/8 | 2/8 | transport:2 | turn 1 | 9 | 401 tokens, 1959 upload bytes |
| Pre-staged | 2/8 | 2/8 | transport:2 | turn 1 | 9 | 401 tokens, 1959 upload bytes |

There are two policy-local turns while connected (1 and 3); turn 1's prompt is caught up while connected. Turn 3's later recovery overlaps the deliberate outage and is not credited as connected catch-up. The two outage fallbacks are turns 4/5 and do not satisfy policy-local coverage. Resync catch-up wall time is 1.79997 s cold/1.82961 s pre-staged; containing protocol time is 1.80451 s/1.83433 s. These are shared-host CPU plumbing observations, not Mac economics.

The GPU runbook now prescribes target-M3 canonical 512-token delta rates by depth 16k/50k/100k/150k/200k/249k:171.4/96.9/54.9/27.9/21.9/18.8 tok/s. It retains the 2k/10k rows and source JSONL SHA in fixtures/growing-provenance.json. Same-GPU local rates would predict all-local because remote work adds transfer/import. The rehearsal therefore labels target-M3 decision inputs on a CUDA client and limits its timing claim to plumbing.

Policy now has zero-default `import_fixed` and `import_per_token` fields, included once in remote cost and rejected when nonfinite/negative. The runbook uses the measured PR9 CUDA OLS 1.618657356721653 s +0.0004953084013066794 s/imported-token for the CUDA rehearsal client, with measured node/link/transport/geometry fields supplied separately. Calibration audit02 re-derived this fit from 18 positive-import raw records using remote_applied, verified source SHA b88f8fb440db84abbd48b9d5a606e07ec672b3ed11f16b73fef4a66ece82a634, and checked canonical/stat table values against their source hashes. The initial audit used new_tokens (including the held token) and failed; using imported-token units reproduces the published intercept and slope exactly. PR9 predates import-speed-v2 and is a labelled calibration surrogate. The M3 observation 42.4 s/39504tokens identifies no separate intercept/slope; a measured M3 fit is still needed for Mac economics. No zero intercept is described as measured.

### F5 and notes

`validate_checkpoint` reads hashed, size-bounded full-vocabulary float32 raw logits and independently re-derives each position, KL, top-1, exactness, maximum error, mean and p95, including factorial rows. Numerical consistency uses relative 1e-8/absolute 1e-12 for independently summed KL; positions, top-1, exactness and max-error comparisons are exact. The native gate still uses its reported floating-point mean with a zero limit. Independent greedy continuations remain native-checker evidence; raw teacher-forced logits cannot alone rederive them. The r7r mutant (positions 9.9/false with passing aggregate fields), max-error 12345, shifted positions, false mean/top-1 and altered raw bytes refuse. A valid exact and a non-exact analytical two-logit control both run, so always-rejecting validation cannot pass.

Resync aggregation now filters session ownership before sums, using the same helper as per-call catch-up. Trace v3 exports extend_chunk_tokens, delta_geometry, fixed node/checker 256/256 plus two threads, generation temperature 0/ignore_eos/thinking settings, sync source and route preflight/counts. Local-arm decision.result is a policy suggestion before forced-local benchmark execution; readers must pair it with meter.mode/path. Checked-in depth/gap/output tables carry stats.json/agents.all selectors and SHA; source audit02 matches them. The conditional input lower half implies p25=median/2 versus the measured global p25/median 0.20, oversizing small deltas; the flat p90 tail and independent marginals remain explicit. Growing output sampling adds(0,0), unlike hybrid_bench's p10 floor. Mac admission now explicitly names launcher ram_budget_mib/node_limits.

### Regressions, controls and real statuses

All validation commands ran as standalone subprocesses with real exits; no tee pipeline. Exact commands, PIDs and source/binary hashes are in `.temp/r38/` ledgers. Python3.14.4, existing CMake4.4.4/GCC15.2.0; nice19 builds, shared build lock, one build at a time, <=j4, on-disk TMPDIR, bytecode disabled. CPU identities are bound by qualification-identity.json; server-OFF binaries are bound by off-identity.json before deletion.

| Validation | Actual exit / result | Evidence under .temp/r38 |
|---|---|---|
| Server ON configure/build; incremental final checker | 0 | cpu-01/02/03-build-record.json |
| Server OFF configure/build | 0 | cpu-off01-build-record.json |
| Bench unittest discovery | 0;83 tests,6 environment skips | bench-02.log |
| Proxy unittest discovery | 0;74 tests | proxy-01.log |
| Native header/identity | 0;12 tests, no skips | identity-01.log |
| Native realistic cold/pre-staged qualification | **1** overall; both drivers **3**, each checkpoint **3** | native-03.log; native-03/commands.json |
| Same-live-snapshot exact positive | 0 | native-03/checkpoint-controls/commands.json |
| Checksum, journal, capacity negatives | 2 each, intended named refusal | same ledger/logs |
| Native degenerate-route controls | 2 each, zero arm calls | native-03/preflight-all-local/all-remote-command.json |
| Retained r37 factorial replay, checkpoints10/12 | 0 each | replay-cli-10/12/command.json |
| Existing delta-chain/noise/corruption controls, server ON/OFF | 0 each; intended child2/3 retained | legacy-01.log,legacy-off-01.log |
| Server-OFF snapshot and malformed replay tiling | 2 each, intended refusal/no result | off-controls/commands.json |
| Source-table/OLS audit | 0 | calibration-audit-02.log,json |
| Trace/position/raw-hash/source identity audit | 0 | audit-02.log,measurements.json |
| Mutation harness | 0;identity 0, seven mutants each 1 by named assertion | mutants-01.log,mutants/results.json |

All prior growing-session test names remain present and green. Added tests cover raw-logit rejection, route degeneracy, fallback counting, exact factorial ranges and provenance; proxy import-cost positive/negative controls also run. The r36 checksum-valid changed-distribution controls retain their intended failures, including same-top-1 loss. These are finite instrument controls, not broad mutation coverage or whole-project CI.

| Mutant | Narrows the gate to | Named failing test (exit 1) | Survivor bound |
|---|---|---|---|
| generated-cell-decodes | prompt-only replay masquerades as generated batching | test_factorial_schedule_boundaries | killed |
| partition-mean-authorizes | nonzero KL accepted under an unexplained partition band | test_checkpoint_partition_band | killed |
| raw-validation-disabled | checker summaries trusted without raw positions | test_checkpoint_raw_logits_reject_r7r_mutant | killed |
| degenerate-route-admitted | same-route preflight accepted | test_route_preflight_rejects_degenerate | killed |
| foreign-catchup-counted | another session charged to this turn | test_catchup_attribution | killed |
| depth-source-drift | sampler table differs from measured source | test_sampling_table_provenance | killed |
| legacy-exactness-weakened | contradictory legacy exact result admitted | test_checkpoint_exactness_gate | killed |

Review surface map: row 1 attribution/statistic -> factorial/A-L/B-H plus generated-cell/partition-mean mutants; row 2 driver -> raw/route/catch-up tests and mutants; row 3 trace/baseline -> source/provenance and foreign-session controls; row 4 runbooks -> native preflight negatives, import-cost tests and degenerate-route mutant; row 5 regressions -> bench/proxy/native identity/ON/OFF legacy suites, retained old test names and legacy-exactness mutant. No requested row is silently out of contract. There are no surviving mutants in this bounded set. Native fault injection of corrupted catch-up through the relay remains unrun; existing protocol tests still prove HTTP409/no import and preserved state under their synthetic fixtures. Production schedule protocol changes and root-cause kernel work were not added to compensate for the unexplained numerical failure.

Failed/corrected work is retained: unit01/02/03 failed on evolving synthetic fixtures/regex and a None-valued background count; unit04 and final bench02 pass. Native01 cold was diagnostic, then pre-staged refused on changed checker identity; neither is final qualification. Native02's two preflights refused correctly but the harness reused an exclusive ledger filename and failed 1; native03 writes distinct per-case records and completes both starts. Audit01 over-assumed the old 64/64/free-exact behavior and failed on the actual turn 12 result; audit02 checks the observed positions and gate honestly. Calibration audit01 used the wrong token-unit selector; corrected audit02 passes. None of these failures is counted as green evidence.

The final scientific gap remains explicit: no cell accounts for the live hybrid state across both checkpoints, so no numerical-equivalence acceptance or GPU/Mac admission is claimed. A next investigation can cross node save/load and import semantics against evaluation-only B; this rework does not guess a workaround, drop KL or claim token integrity proves fidelity.

cold observed 2026-11-02T02:01:57.951687+00:00 to 2026-11-02T02:09:25.442199+00:00; trace SHA256 `9e13a6495c1af5f24ad1bddf67f3a10f53d980e124b9112c2d59476ce3b48535`.

pre-staged observed 2026-11-02T02:09:25.948977+00:00 to 2026-11-02T02:16:41.961695+00:00; trace SHA256 `8eafc59958cd6cf283a466cbf3572fab3290866e275168932ef8dcdd3d3dc71a`.

Cleanup at 2026-11-02T02:19:58.734801+00:00: all recorded task-native/build PIDs inactive, both owned build trees and raw native snapshots/expired credentials removed. Final raw logits, retained input tokens/schedules, trace/results/logs/hash manifests remain private for review. Minimum observed free disk 31096446976 bytes (31.096 GB), above 28 GB; cleanup free 32000200704 bytes. The model and source worktree remain. Full CUDA/Metal/long-context inference and upstream CI are unrun under the CPU-only task scope.

## r39: residual source isolated by live node state and append imports

Run `hybrid-b-r39-261102`, based on signed r38 `f37962493a`, on the same `m3-growing-session` worktree. Evidence: `.temp/r39/measurements.json`, `qualification-identity.json`, `native-affinity-01/cold/check-{10,12}/`, and retained r38 raw logits. CPU only; existing Qwen3.5-0.8B-Q4_0, SHA-256 `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`, context 4096, q8_0 K/V, f32 recurrent state, flash attention, 256/256 primary partition. No GPU, model download, numeric-gate change or push.

The residual source is **mixed attention-cache provenance after append import**. The client keeps old KV rows computed locally, imports only the node's new KV range, and replaces the full recurrent state. Node catch-up recomputes old generated/outage tokens in batches, but that recomputed historical node KV is not transferred back. B uses the node path for the complete prefix; H retains the mixture. Replaying local evaluation and the actual delta imports reproduces H exactly. Replacing only its attention KV with the node snapshot makes the replay exact with B; replacing recurrent state alone leaves it exact with H. At the final import boundaries 1550/1914, recurrent bytes are already equal and attention bytes differ.

The smallest explanation for the entire CPU observation is evaluation-partition arithmetic **plus retaining the resulting different historical attention KV during append import**. Batch arithmetic alone was insufficient in r38 because its B cell replaced the provenance of that history. This run names the residual mechanism; it does not make L-H numerically acceptable.

KL and p95 are nats/token over 64 complete 248320-logit distributions. Pair labels specify KL(left || right); an exact cell also has max logit difference 0, top-1 agreement 64/64 and no free-running divergence. Both checkpoints use the same local greedy teacher IDs.

| Factor changed / comparison | KL10 | p9510 | max10 | Exact10 | KL12 | p9512 | max12 | Exact12 |
|---|---:|---:|---:|:---:|---:|---:|---:|:---:|
| L-H, original gate | 0.000637837220046 | 0.00186315189913 | 0.964870452881 | no | 0.00027947834459 | 0.00100390301939 | 0.737212657928 | no |
| B-H, matched 2/2 threads and CPU affinity | 7.71985333402e-05 | 0.000302589110353 | 0.792877197266 | no | 0.000274473850817 | 0.00101512660845 | 0.771906852722 | no |
| Live node state + identical tail vs B | 0 | 0 | 0 | yes | 0 | 0 | 0 | yes |
| Save/load each node extend vs B | 0 | 0 | 0 | yes | 0 | 0 | 0 | yes |
| Own range export/import vs A | 0 | 0 | 0 | yes | 0 | 0 | 0 | yes |
| Imported full node state vs B | 0 | 0 | 0 | yes | 0 | 0 | 0 | yes |
| Actual append-chain replay vs H | 0 | 0 | 0 | yes | 0 | 0 | 0 | yes |
| Replace attention KV alone vs B | 0 | 0 | 0 | yes | 0 | 0 | 0 | yes |
| Replace recurrent state alone vs H | 0 | 0 | 0 | yes | 0 | 0 | 0 | yes |

A-L, A-A, OWN_NATIVE-A and BSTATE-B are also exact at both checkpoints. BSTATE's complete native bytes equal the node snapshot at each last-import boundary. Each native restore in RESTORE and each OWN_NATIVE/OWN_RANGE/NODE_RANGE roundtrip is byte identity, in one process, followed by exact continuation relative to its corresponding unchanged reference. Export/import is not an extra lossy q8 quantization step.

### Node truth, threads, KV and remaining factors

The final diagnostic uses snapshots captured by the **live hybrid node process immediately after its critical extend**, through the existing authenticated snapshot endpoint, with the returned delta and SHA-256 hashes. It does not substitute a client reconstruction for node truth. A boundary-race control rejects a snapshot that has advanced beyond the extend, before writing files. An earlier isolated node replay of the retained r38 boundaries supplied the same turn 10 result; its CHAIN raw bytes equal the retained live r38 H bytes. The live captured run repeats that result at both checkpoints.

The existing node/client/checker compute and batch counts were already 2/2. Pinning their parent to CPUs 0,1 gives all three the same inherited mask. `qualification-identity.json` records observed live PID/thread masks, including the checker; every recorded worker mask is 0,1. The complete factorial dictionaries and A/B/H raw hashes match r38 at both checkpoints. No thread-split or affinity contribution is needed to explain this residual. This does not claim universal invariance to thread counts or CPU models.

Both sides request q8_0 K/V and the same model-derived Hadamard attention rotation; the live node/checker log K rotation 256 and V rotation 64, and the native loader enforces that layout on imported client state. `llama-kv-cache.cpp` enables rotation for quantized head geometry and verifies K/V rotation dimensions and types when restoring; `hybrid-state.cpp` exports native q8_0 ranges. Whole-node native/range restore agrees exactly with a same-partition local B context. Own-state byte identity and the malformed/valid append controls bound the import claim.

The local tail is held identical in every comparison. The recurrent-only replacement is a negative causal control: it changes no bytes or logits because the fixed state already matches the final node checkpoint. Attention-only replacement selects the residual independently of tail or fixed-state placement. B retains the node's per-chunk logits flags, while A retains local prompt-end flags; both reproduce their own live references. Since the complete hybrid import path now reproduces H, further tail/logits/build flag variants are unnecessary for naming the residual. Node, client and checker came from one CMake Release/CPU/native build and shared the same backend libraries; target flags and binary/source hashes are retained. No core, kernel, transfer-format or production protocol fix was introduced.

CUDA is unrun. This is not established as thread-dependent CPU matmul noise, so a change to GPU reductions is not assumed to remove it. The GPU rehearsal must repeat NODE-B, CHAIN-H, the roundtrips and both component replacements, alongside L-H/A-L. If batch and token paths yield different historical KV there, append import retains the same provenance mechanism. The existing uniform mean-KL-zero, top-1>=63/64 and free 64 gate remains authoritative.

### Validation and finite bounds

Bench: 85 tests exit 0 (six optional native-header skips); the separate native identity run executes all 12 tests with the existing exact header dumps, exit 0. Proxy: 74 tests exit 0. Final own-state/native controls pass with intended refusals for incomplete final logits, duplicate probe names, snapshot checksum, token journal, and capacity (each actual exit 2, no successful result). Node snapshot boundary-race and missing/gapped live-evidence tests are green. Native malformed recurrent append controls exit 0, including valid appends before/after rejections and byte-preserved neighbor state.

Server-ON/OFF CPU builds exit 0; both legacy noise/corruption suites exit 0, retaining checksum refusal 2 and checksum-valid corruption/lossy failures 3. Server-OFF snapshot and malformed replay tiling each refuse with actual exit 2 and their intended reason. The first legacy invocation refused an already-existing artifact directory with exit 1 before inference; corrected `legacy-02` runs green. These are not hidden passing attempts.

Nine narrowing mutants are killed, each by an assertion in the named test below; identity is green. No survivor is used as evidence.

| Mutant | Property narrowed | Named failing test | Actual exit |
|---|---|---|---:|
| probe-raw-validation-disabled | probe fields must agree with raw logits | test_checkpoint_raw_logits_reject_r7r_mutant | 1 |
| node-boundary-race-admitted | a snapshot must match the just-extended boundary | test_node_snapshot_capture_refuses_boundary_race | 1 |
| generated-cell-decodes | B1 must change the generated partition only | test_factorial_schedule_boundaries | 1 |
| partition-mean-authorizes | positive partition noise cannot authorize KL | test_checkpoint_partition_band | 1 |
| raw-validation-disabled | raw evidence must reject falsified positions | test_checkpoint_raw_logits_reject_r7r_mutant | 1 |
| degenerate-route-admitted | both connected policy routes are required | test_route_preflight_rejects_degenerate | 1 |
| foreign-catchup-counted | catch-up belongs to its own session | test_catchup_attribution | 1 |
| depth-source-drift | sampling parameters match recorded provenance | test_sampling_table_provenance | 1 |
| legacy-exactness-weakened | legacy exactness remains enforced | test_checkpoint_exactness_gate | 1 |

Native source coverage is finite: existing 0.8B, two checkpoints at depths 1679/2043, 64 teacher-forced and independent greedy positions, one CPU build/backend. This is not 27B/100k/deep or CUDA/Metal qualification. Raw-logit validation independently rederives per-position KL/top1/max/exactness and aggregates, including probe comparisons; free-running verdicts remain native observations. No brief surface table was supplied (brief gap); every requested factor is accounted for above. GPU/model downloads are outside the user's explicit scope. All previously committed reviewer/unit tests retain their names; no requested CPU row is silently excluded.

The runtime checker and driver were frozen for the live captures. The native harness gained additional roundtrip/refusal controls after the first cold launch; those final controls were run separately, and the pre-staged regression uses the final harness. Paths/line numbers printed in the first harness traceback can reflect that test-file update; recorded exit codes and measured JSON are the evidence. Numerical equivalence remains unqualified.

Both final starts complete 39 calls and both 64-position checkpoints: two connected policy-local turns, eight connected offloads, two transport fallbacks, connected prompt catch-up at turn 1 and nine background catch-ups. Both actual drivers exit 3; both native harnesses exit 1 at the unchanged zero-KL criterion. Their complete factorial dictionaries and A/B/H raw identities equal r38. `audit-02.log` exits 0, verifies both starts, journals/routes, all referenced private state/logit hashes, named probe exactness/component predicates, and source/binary identity. No passing-equivalence handoff is claimed.

native-affinity-01: observed 2026-11-02T02:40:27.731703+00:00 to 2026-11-02T02:56:42.999345+00:00; trace SHA-256 `fa93e162e7ce87f1cdf206130c013d30406079a320c669a9bb7d82ce2bcf97b9`.

native-prestaged-01: observed 2026-11-02T02:58:34.959318+00:00 to 2026-11-02T03:04:17.009433+00:00; trace SHA-256 `fba8528d2b8a7a3ffb44b9e116dbda80e32f9f28459ff1c4f844315b45621196`.

Audit01 failed 1 because it read exit_code from trace instead of the command ledger; corrected audit02 reads the persisted actual command exit and passes. A late additional affinity read failed its minimum-live-actor assertion after the processes had exited; it supplies no new mask evidence. The during-run qualification identity remains the affinity evidence. These failures do not replace or weaken any product gate.

Cleanup at 2026-11-02T03:09:01Z: all 82 recorded task PIDs inactive; both task-owned build trees removed (118424498 bytes). Minimum sampled free disk 29188509696 bytes (29.19 GB); cleanup free 36195479552 bytes. Live node snapshots/deltas, hashed raw logits, tokens, traces and logs remain private for review; identical logit files are hardlinked without changing bytes. Signed local delivery only; no push, GPU or model download.

## r40: mechanism-based append-import gate

Run `hybrid-b-r40-261102`, continuing signed r39 `796425fe` on `m3-growing-session`. The orchestrator has selected the acceptance contract below after the cause was isolated. This section supersedes the r38/r39 L-H zero-KL gate, while retaining those historical measurements and failures. CPU only; no GPU, downloads, push or production protocol change.

### Mechanism and paper wording

Append import keeps the client's historical attention KV rows, appends only the node's new KV range, and replaces the full recurrent state. The node computed that range over its own catch-up history. Batch evaluation and token-by-token evaluation can produce different historical attention KV, so the resulting hybrid state combines local history with a node suffix. Replaying the exact append chain reproduces the live hybrid continuation bit-exactly (CHAIN-H, maximum logit difference 0). Replacing attention KV alone selects the coherent node path B; replacing recurrent state alone leaves H unchanged. This explains why agreement with a differently evaluated fully local path L cannot be required to have zero mean KL for append import.

The gate requires exact CHAIN-H at every checkpoint and the complete r39 component controls: NODE-B, RESTORE-B, BSTATE-B, NODE_RANGE-B, OWN_NATIVE-A, OWN_RANGE-A, CHAIN_ATTENTION-B, CHAIN_RECURRENT-H and A-L. Exact here means all 64 full-vocabulary position vectors have identical float32 bytes, maximum difference 0, KL 0, top-1 64/64 and no independent greedy divergence over 64 positions. Restore/range/own roundtrips must preserve state bytes; BSTATE must match the live node snapshot bytes; both replacements must show equal recurrent bytes before replacement. Attention-byte differences are reported, rather than required on every backend. Model/journal/hash checks remain mandatory.

L-H has a separate token-integrity guard: top-1 at least 63/64 and no free-running divergence among tokens 0 through 7. Divergence at index 8 or later is allowed and reported. L-H mean KL, nearest-rank p95 and maximum absolute logit difference remain visible at every checkpoint as the measured cost of append import. They do not decide correctness. Both native checker and Python driver use `status_rule=session_append_replay_exact`, `kl_limit=null`, with each identity in `gate_controls`; Python independently rederives the teacher-forced metrics from hashed raw logits. Missing evidence aborts; a measured failed control or token guard gives exit 3. Existing two-snapshot exactness and legacy zero-band fixtures retain their original rules. CUDA-to-Metal remains heterogeneous, exit 5/uncalibrated.

State probes are mandatory in the driver, including live authenticated node snapshots and returned deltas. `--state-probes` remains accepted for old commands. These captures add HTTP/disk overhead and are not a performance result. The [GPU and Mac runbooks](GROWING-SESSION.md) repeat every control at every checkpoint, use pinned target-client delta rates, require routing preflight before the arms, and require a measured multi-point M3 import fit for Mac economics. The CPU fixture policy remains a labelled synthetic routing control.

### Alternative: refresh the full attention KV

A full attention refresh after catch-up removes the mixed history. It restores exactness to the same-partition full-node local replay B; equality to differently batched A/L additionally needs matching evaluation arithmetic. Refreshing node KV alone cannot make two distinct arithmetic schedules identical. This is an alternative state contract, not a change implemented here.

The illustrative 27B q8_0 attention geometry is 16 attention layers x 2 (K/V) x 4 KV heads x 256 dimensions x 34/32 bytes = **34816 bytes/token**. The dimensions are read from the checked-in exact-header fixture; `.temp/r40/geometry.json` also recomputes the complete seed-1/40-turn deep admission plan with the measured 14 framing tokens per turn.

| Per-offload depth | Full attention payload, bytes | Decimal GB | GiB |
|---|---:|---:|---:|
| 100000 | 3481600000 | 3.481600000 | 3.242492676 |
| Deep start 216064 | 7522484224 | 7.522484224 | 7.005859375 |

For subsequent offloads, use actual depth x 34816. These are tensor payload bytes only; add recurrent state (about 157000000 bytes), native row/layout metadata, envelope/hash framing and network overhead. With the illustrative 157000000-byte recurrent payload, the combined tensor payload would be about 3638600000 bytes at 100k and 7679484224 bytes at the deep start, before framing. Because append already sends the suffix, the incremental attention cost of full refresh is `(depth - imported_tokens) x 34816` bytes. No full-state bandwidth or M3 speedup is inferred from these payload calculations.

### Large snapshot control

Repeating BSTATE at 27B/100k exposed a pre-existing driver limit: its byte comparison loaded the whole native node snapshot through the 513 MiB delta reader. Required hardware snapshots are several GB. Only the comparison now streams through a 64 KiB buffer against the already materialized local native state, without another full snapshot copy; the delta-file bound remains unchanged. Checker startup keeps small exact/changed/truncated controls. An isolated fixture larger than the old cap checks exact equality, a changed final byte and truncation. Restoring the old cap fails `test_large_snapshot_matches`; no GPU/model is needed. The complete 40-turn replay/probe manifests also exceed the original 8 MiB JSON cap: synthetic all-offload plans with the full sampled growth serialize 21934079 bytes at 100k and 23101435 bytes at deep. These are CPU schedule-size controls, not inference or state truth. The bounded JSON cap is now 64 MiB, with nesting 32 retained. A 9 MiB valid exact-checkpoint input passes, the original checker refuses it at 8 MiB, and a 64 MiB+1 input refuses with no metrics. These instrument limits change no evaluation, import or acceptance arithmetic.

### CPU results and validation

The frozen live-run checker completed both three-session starts. The final checker differs only in streamed snapshot byte comparison, its startup controls and the JSON input bound; exact reconstruction of the original source hash proves that boundary (`late-delta-final-01.log`). Both final-checker replay runs recompute all 13 A/B1/B2/B/probe observation vectors from the retained live journals, node snapshots and deltas. Their raw SHA-256 values and all shared numerical/state fields equal the live-run vectors, including CHAIN equal to live H. The frozen live processes remain on CPU 0,1. Final replay 10 starts there and then moves to CPU 4,5 to avoid CPU contention (all thread masks and the transition are recorded); replay 12 uses CPU 2,3. Both keep two compute/two batch threads, and bit identity is checked rather than assumed. Final replay-only native exits are 0; they are not fresh live H/L/P captures. The full live-driver gate integration is from the frozen initial binary. The final replay binary predates the last JSON-cap-only change; its gate, import and arithmetic code is identical, and the final rebuilt binary passes the above 8 MiB exact-checkpoint admission control. All source/binary identities and exact command/PID logs are retained.

| Start | Turn / depth | L-H mean KL | L-H p95 KL | L-H max abs dlogit | Top-1 | First free divergence | Gate / native exit |
|---|---:|---:|---:|---:|---:|---:|---|
| cold | 10 / 1679 | 0.000637837220046 | 0.00186315189913 | 0.964870452881 | 64/64 | -1 | passed / 0 |
| cold | 12 / 2043 | 0.00027947834459 | 0.00100390301939 | 0.737212657928 | 63/64 | 0 | failed / 3 |
| pre-staged | 10 / 1679 | 0.000637837220046 | 0.00186315189913 | 0.964870452881 | 64/64 | -1 | passed / 0 |
| pre-staged | 12 / 2043 | 0.00027947834459 | 0.00100390301939 | 0.737212657928 | 63/64 | 0 | failed / 3 |

KL is nats/token; p95 is nearest rank. **Every checkpoint passes all ten replay/component identities, including CHAIN-H with max difference 0. Turn 10 passes despite positive L-H KL. Turn 12 fails solely because free-running divergence is at index 0, inside the first-eight-token guard; top-1 remains 63/64.** Both drivers therefore exit 3 and the whole native harness exits 1. This is a preserved measured token failure, not a passing-equivalence claim or a reason to suppress KL. Each start has 39 calls, exact paired journals, two connected policy-local turns, eight policy offloads, two transport fallbacks, connected prompt catch-up and background replay. A/L and every factorial/probe result reproduce r39. CPU 0.8B/q8_0/context 4096 and this synthetic policy do not qualify 27B/100k/deep or GPU/Mac economics.

| Start | Observed UTC interval | trace.json SHA-256 |
|---|---|---|
| cold | 2026-11-02T03:21:36.345545+00:00 to 2026-11-02T03:35:29.146325+00:00 | `d47a5ca631cb3aa366635bd92eee2d3c8342faf329c4692d57bed07fdc07da6b` |
| pre-staged | 2026-11-02T03:35:29.542092+00:00 to 2026-11-02T04:02:25.811971+00:00 | `d596eac5779f8e2c799e0e6f49b7251ff89baea71b4d00d3b0e7f5f558c5126f` |

All validation ran as standalone processes with real exits, on-disk TMPDIR and bytecode disabled. Build commands were nice 19/-j4 under the shared lock. Evidence/commands are in `.temp/r40/validation-ledger.json`, each native `command(s).json`, the build records, and `measurements.json`.

| Validation | Actual exit / scope | Evidence under .temp/r40 |
|---|---|---|
| CPU server ON/server OFF configure/build and final checker rebuilds | 0 | cpu-*-build-record.json |
| Bench discovery | 0; 89 tests, 6 header-environment skips | bench-regression-01.log |
| Proxy discovery | 0; 74 tests | proxy-regression-01.log |
| Explicit native identity | 0; 12 tests, covering all 6 discovery skips | identity-02.log |
| Recurrent malformed/state-preservation suite | 0 | recurrent-01.log |
| Legacy ON/OFF and final ON delta/noise/corruption suites | 0 overall; intended child 2/3 retained | legacy-01.log, legacy-off-01.log, legacy-final-01.log |
| Exact live-state, own native/range identity; checksum/journal/capacity/incomplete/duplicate refusals | 0 exact/roundtrip; intended 2 refusals | native-01/checkpoint-controls/commands.json |
| Native all-local/all-offload routing preflights | intended 2 each, zero arm calls | native-01/preflight-*-command.json |
| Snapshot server OFF, missing append controls, invalid replay tiling | intended 2 each, no metrics; final binaries | extra-controls-final/commands.json |
| Retained r37 replay 10/12 | 0 each | retained-replay-*/command.json |
| Final native replay of both complete probe sets | 0 each; 13 vectors per checkpoint identical | final-check-*/verification.json |
| Final native exact/own roundtrip/manifest and refusal controls | 0 positive; intended 2 refusals | controls-final-01.log |
| Large snapshot exact/changed/truncated controls | 0 | snapshot-compare-03.log |
| Complete 40-turn manifest size planning; above 8 MiB native positive / above 64 MiB refusal | 0 planning; native 0 / intended 2 | manifest-forecast.json, manifest-controls/commands.json |
| Full cold/pre-staged session gate | driver 3 each, harness 1; turn 12 first-eight guard | native-01.log, native-01/commands.json |
| Final artifact, journal, component, identity and route audit | 0 | audit-01.log, measurements.json |
| AST, ASCII additions, shell syntax and whitespace | 0 | style-final-01.log, whitespace-final-03.log |

Initial identity selection failed 1 because the header directory was wrong; retained exact-header fixtures corrected it. Initial retained replay invocation failed 2 for missing --trace/--turn; full corrected commands pass. The first geometry probe failed 1 because max_new=0 caps all input; default 10875 gives the declared deep start. The first isolated stream-test compile failed 1 for a missing cstdint include in its test preamble; the product already had the type. A preliminary old-cap plant refused before a named test and is not credited; the corrected plant explicitly fails test_large_snapshot_matches. These attempts are preserved separately. Full upstream CI, ty and editorconfig-checker were not run; the latter two are absent. No dependency install, model download, GPU run, PR, push or scientific/hardware acceptance is claimed.

### Narrowing mutants

| Mutant | What it narrows the gate to | Named failing test | Actual child exit |
|---|---|---|---:|
| append-replay-mismatch-admitted | CHAIN-H exactness cannot be omitted | test_checkpoint_append_replay_mismatch_fails | 1 |
| append-LH-KL-zero-required | positive L-H KL alone must pass | test_checkpoint_append_positive_kl_passes | 1 |
| append-early-divergence-admitted | free token 7 must fail; token 8 may pass | test_checkpoint_append_token_boundaries | 1 |
| append-roundtrip-bytes-ignored | own/range byte identity is mandatory | test_checkpoint_append_component_controls | 1 |
| probe-raw-validation-disabled | probe metrics must agree with raw vectors | test_checkpoint_raw_logits_reject_r7r_mutant | 1 |
| node-boundary-race-admitted | live snapshot must match the critical extend | test_node_snapshot_capture_refuses_boundary_race | 1 |
| generated-cell-decodes | B1 changes generated partition only | test_factorial_schedule_boundaries | 1 |
| partition-mean-authorizes | legacy partition band cannot authorize positive KL | test_checkpoint_partition_band | 1 |
| raw-validation-disabled | raw vectors must reject falsified positions | test_checkpoint_raw_logits_reject_r7r_mutant | 1 |
| degenerate-route-admitted | preflight requires both connected policy routes | test_route_preflight_rejects_degenerate | 1 |
| foreign-catchup-counted | catch-up belongs to the measured session | test_catchup_attribution | 1 |
| depth-source-drift | sampling matches pinned provenance | test_sampling_table_provenance | 1 |
| legacy-exactness-weakened | legacy exactness remains strict | test_checkpoint_exactness_gate | 1 |
| native-replay-mismatch-admitted | compiled checker gate must reject replay loss | test_append_replay_mismatch_fails | -6 |
| native-LH-KL-zero-required | compiled checker gate must accept positive L-H KL alone | test_append_positive_LH_KL_passes | -6 |
| old-manifest-cap | full replay/probe JSON must fit above 8 MiB | test_manifest_above_8MiB | 2 |
| old-size-cap | full snapshot comparison must work above delta-file bound | test_large_snapshot_matches | 1 |

All 17 listed mutants are killed at the named test; there are no survivors in this bounded set. Identity controls exit 0. The two compiled gate mutants use exact extracted production gate functions and verdict block on the retained real positive-KL turn 10 result; abort -6 is the intended named failed assertion with core dumps disabled. They are isolated gate controls, not fresh inference runs. The Python gate tests explicitly isolate raw validation; the unchanged r7r raw-vector test covers that independent instrument. No broad mutation-coverage claim is made.

No surface table was supplied (brief gap). No requested row is excluded. The owner-selected gate is implemented and ready for review; the remaining token/hardware failures are explicitly recorded.

Cleanup at 2026-11-02T04:07:17Z: all 115 recorded task PIDs are inactive. All three task build trees (209602643 bytes), compiled mutants and expired fixture credentials were removed. Minimum sampled free disk was 29538697216 bytes (29.539 GB); cleanup readback was 30843768832 bytes. Private node snapshots/deltas, journals, raw logits, hashes and logs remain available for review; only sealed byte-identical outputs share inodes. No push.

## r41: the reviewer-specified gate

Run `hybrid-b-r41-261102`, based on signed `e3814bac` on `m3-growing-session`, implements r7t F2 exactly. This supersedes the r40 free-divergence clause; historical r40 exit 3 remains recorded above. CPU only, existing local Qwen3.5-0.8B Q4_0, q8_0 KV, no GPU/model download/push. The fresh admission harness and both starts exit **0**. This is ready for review; no independent acceptance of r41 is claimed.

All ten `gate_controls` remain exact: NODE-B, RESTORE-B, BSTATE-B, NODE_RANGE-B, OWN_NATIVE-A, OWN_RANGE-A, CHAIN-H, CHAIN_ATTENTION-B, CHAIN_RECURRENT-H and A-L. Their native/Python implementation bodies and Python raw-logit validator are byte-identical to r40, with proof hashes in `.temp/r41/qualification-identity.json`. Byte roundtrips, live BSTATE bytes, recurrent equality before both replacements, journal/model/snapshot/delta hashes, and full raw-logit re-derivation are unchanged. L-H teacher-forced top-1>=63/64 is the sole L-H sanity gate. L-H free divergence is always reported and never an exit condition. Exact-control independent greedy equality remains required.

Each exported checkpoint adds report-only `fidelity_flag`, computed after validation. L-H is compared strictly with all three noise cells A-B1, A-B2 and A-B, like-for-like per statistic: mean/p95/max greater, top-1 smaller, free index earlier (-1 ranks as positive infinity). Equal to any noise cell leaves the relevant flag false. Neither a flag nor any L-H KL/max/free value can change the verdict. The native checker observes free continuations; the independent raw audit validates teacher-forced distributions and cannot re-derive free continuations.

### Admission measurements

Audit observed at 2026-11-02T05:10:39.685660+00:00; runtime source and binary SHA-256 identities are retained in qualification-identity.json.

Both cold and pre-staged have 39 calls, checkpoints at depth 1679/2043 (turn 10/12), all ten exact controls green, exact paired journals, connected policy-local/offload, outage fallback, connected prompt catch-up and background replay. The native harness runs the unchanged raw-logit validator at every checkpoint. Hashes, controls, flags, routes and source/binary identities are additionally audited in `.temp/r41/measurements.json`. The table includes each actual checkpoint, its three noise cells and B-H; KL is nats/token, p95 is nearest rank, top-1 is out of 64, free=-1 means none.

| Start | Turn / depth | Cell | Mean KL | p95 KL | Max abs dlogit | Top-1 | Free |
|---|---|---|---:|---:|---:|---:|---:|
| cold | 10 / 1679 | L-H | 0.000637837220046 | 0.00186315189913 | 0.964870452881 | 64/64 | -1 |
| cold | 10 / 1679 | A-B1 | 0.00018071129068 | 0.000879184946625 | 0.866317749023 | 64/64 | -1 |
| cold | 10 / 1679 | A-B2 | 8.29298923694e-05 | 0.000546047878833 | 1.11637067795 | 64/64 | -1 |
| cold | 10 / 1679 | A-B | 0.000563020461777 | 0.00121762819219 | 0.942632198334 | 64/64 | -1 |
| cold | 10 / 1679 | B-H | 7.71985333402e-05 | 0.000302589110353 | 0.792877197266 | 64/64 | -1 |
| cold | 12 / 2043 | L-H | 0.00027947834459 | 0.00100390301939 | 0.737212657928 | 63/64 | 0 |
| cold | 12 / 2043 | A-B1 | 0.000248037221206 | 0.000306196323783 | 0.70611333847 | 64/64 | -1 |
| cold | 12 / 2043 | A-B2 | 0.00041999735555 | 0.00138579233125 | 0.869745731354 | 63/64 | 0 |
| cold | 12 / 2043 | A-B | 0.000245036923118 | 0.000767767496852 | 0.624326705933 | 64/64 | -1 |
| cold | 12 / 2043 | B-H | 0.000274473850817 | 0.00101512660845 | 0.771906852722 | 63/64 | 0 |
| pre-staged | 10 / 1679 | L-H | 0.000637837220046 | 0.00186315189913 | 0.964870452881 | 64/64 | -1 |
| pre-staged | 10 / 1679 | A-B1 | 0.00018071129068 | 0.000879184946625 | 0.866317749023 | 64/64 | -1 |
| pre-staged | 10 / 1679 | A-B2 | 8.29298923694e-05 | 0.000546047878833 | 1.11637067795 | 64/64 | -1 |
| pre-staged | 10 / 1679 | A-B | 0.000563020461777 | 0.00121762819219 | 0.942632198334 | 64/64 | -1 |
| pre-staged | 10 / 1679 | B-H | 7.71985333402e-05 | 0.000302589110353 | 0.792877197266 | 64/64 | -1 |
| pre-staged | 12 / 2043 | L-H | 0.00027947834459 | 0.00100390301939 | 0.737212657928 | 63/64 | 0 |
| pre-staged | 12 / 2043 | A-B1 | 0.000248037221206 | 0.000306196323783 | 0.70611333847 | 64/64 | -1 |
| pre-staged | 12 / 2043 | A-B2 | 0.00041999735555 | 0.00138579233125 | 0.869745731354 | 63/64 | 0 |
| pre-staged | 12 / 2043 | A-B | 0.000245036923118 | 0.000767767496852 | 0.624326705933 | 64/64 | -1 |
| pre-staged | 12 / 2043 | B-H | 0.000274473850817 | 0.00101512660845 | 0.771906852722 | 63/64 | 0 |

| Start | Turn | Mean flag | p95 flag | Max flag | Top-1 flag | Free flag | Native checkpoint exit |
|---|---:|---|---|---|---|---|---:|
| cold | 10 | true | true | false | false | false | 0 |
| cold | 12 | false | false | false | false | false | 0 |
| pre-staged | 10 | true | true | false | false | false | 0 |
| pre-staged | 12 | false | false | false | false | false | 0 |

**Turn 10 flags mean+p95 and still passes. Turn 12 flags nothing and passes with top-1 63/64 and free divergence 0.** No ratio or noise-derived tolerance gates either checkpoint.

### Regression and negative evidence

All commands run directly with their real exits, on-disk `TMPDIR=.temp/r41/tmp`, `PYTHONDONTWRITEBYTECODE=1`, and niceness 19 for builds/native/tests. Toolchain is reused CMake 4.4.4 / GCC 15.2.0 / Python 3.14.4. One lane build at a time, `-j4`, shared lock after pgrep/uptime admission. Full commands/PIDs/exits are in `.temp/r41/*.command.json`, `cpu-02-build-record.json`, native `commands.json` and the instrument-control ledgers.

| Validation | Actual exit / result | Log under .temp/r41 |
|---|---|---|
| Current CPU configure/build: llama-server, llama-hybrid-prefill, llama-hybrid-bench-check, llama-hybrid-check-identity | 0 / 0 | cpu-02-{configure,build}-01.log |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_growing_session.py -v` | 0; 25 tests | growing-unit-01.log |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_*.py -v` | 0; 93 tests, six header-environment skips | bench-01.log |
| `python3 -m unittest discover -s tools/hybrid-proxy -p test_*.py -v` | 0; 74 tests | proxy-01.log |
| `test_mirai_identity.py --binary .temp/r41/build/bin/llama-hybrid-check-identity --headers /home/agent/lanes/hybrid/.temp/research/bonsai/gguf-dumps --artifacts .temp/r41/identity-01 -v` | 0; 12 tests, covers all six bench skips | identity-01.log |
| `test_growing_session_native.py --build-dir .temp/r41/build --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/r41/native-02 --state-probes` | harness 0; cold 0; pre-staged 0; four checkpoints 0 | native-02.log |
| Same-backend exactness, own roundtrips, valid 9 MiB manifest; >64MiB/incomplete/duplicate/checksum/journal/capacity refusals | three exits 0, six intended exits 2 | native-02/checkpoint-controls/commands.json |
| All-local/all-remote routing preflights | each intended exit 2, zero arm calls | native-02/{all-local,all-remote}.log |
| `.temp/r41/mutants.py`; `.temp/r41/native_gate_mutants.py` | harnesses 0; Python identity 0 / 18 mutants 1; compiled identity 0 / four mutants -6 | mutants-01.log; native-mutants-01.log |
| Runtime-source/native-binary and evidence audit | 0 | audit-01.log; qualification-identity.json; measurements.json |
| CI model-naming script, source AST/ASCII, whitespace | 0 each | code-style-01.log; syntax-ascii-01.log; whitespace-01.log |

The new tests replace the early-divergence boundary with named `test_checkpoint_append_top1_62_fails` and `test_checkpoint_append_divergence_0_passes` (all valid free indices exercised at 63/64). `test_checkpoint_fidelity_flag_never_changes_exit_code` covers pass 0, fail 3 and heterogeneous 5 under all-false/all-true flags. `test_checkpoint_append_control_free_divergence_fails` requires exact teacher metrics with free 3 to make CHAIN-H false; it kills the reviewer's previously surviving Python control mutant. `test_checkpoint_fidelity_flags` exercises strict all-cell comparisons, equality, directions and -1 ordering.

| Mutant | What it narrows the gate to | Named failing test | Actual exit / survivor bound |
|---|---|---|---|
| append-replay-mismatch-admitted | CHAIN-H exactness | test_checkpoint_append_replay_mismatch_fails | 1; killed, none |
| append-LH-KL-zero-required | Positive L-H KL can pass | test_checkpoint_append_positive_kl_passes | 1; killed, none |
| append-top1-floor-admitted | 63/64 teacher-forced floor | test_checkpoint_append_top1_62_fails | 1; killed, none |
| append-free-divergence-gated | L-H divergence 0 remains report-only | test_checkpoint_append_divergence_0_passes | 1; killed, none |
| fidelity-flag-gated | Flags cannot alter 0/3/5 | test_checkpoint_fidelity_flag_never_changes_exit_code | 1; killed, none |
| exact-control-free-divergence-ignored | Exact-control independent greedy equality | test_checkpoint_append_control_free_divergence_fails | 1; killed, none |
| fidelity-any-noise | Strict comparison against every noise cell | test_checkpoint_fidelity_flags | 1; killed, none |
| fidelity-free-minus1-first | Free=-1 ranks as infinity | test_checkpoint_fidelity_flags | 1; killed, none |
| append-roundtrip-bytes-ignored | Roundtrip byte identity | test_checkpoint_append_component_controls | 1; killed, none |
| probe-raw-validation-disabled | Probe raw-logit integrity | test_checkpoint_raw_logits_reject_r7r_mutant | 1; killed, none |
| node-boundary-race-admitted | Captured node journal boundary | test_node_snapshot_capture_refuses_boundary_race | 1; killed, none |
| generated-cell-decodes | Separate generated/prompt schedules | test_factorial_schedule_boundaries | 1; killed, none |
| partition-mean-authorizes | Legacy zero-band rule retained | test_checkpoint_partition_band | 1; killed, none |
| raw-validation-disabled | Full raw-logit integrity | test_checkpoint_raw_logits_reject_r7r_mutant | 1; killed, none |
| degenerate-route-admitted | Connected route mix before arms | test_route_preflight_rejects_degenerate | 1; killed, none |
| foreign-catchup-counted | Session-specific catch-up | test_catchup_attribution | 1; killed, none |
| depth-source-drift | Pinned workload provenance | test_sampling_table_provenance | 1; killed, none |
| legacy-exactness-weakened | Unchanged legacy exactness | test_checkpoint_exactness_gate | 1; killed, none |
| native-replay-mismatch-admitted | Native CHAIN-H exactness | test_append_replay_mismatch_fails | -6; killed, none |
| native-LH-KL-zero-required | Native positive L-H KL accepted | test_append_positive_LH_KL_passes | -6; killed, none |
| native-top1-floor-admitted | Native 63/64 floor | test_append_top1_62_fails | -6; killed, none |
| native-free-divergence-gated | Native divergence 0 accepted | test_append_divergence_0_passes | -6; killed, none |

All 22 plants are killed; none survives. Compiled plants extract the current production helper and verdict block and load the retained real r39 positive-KL result; their expected assertion failures are actual SIGABRT/-6 with core dumps disabled. They are bounded gate tests, not a mutated whole-model inference rerun. No failure is counted without its named assertion.

### Review coverage

| r7t surface row | Current evidence | Narrowing mutant / named failing test |
|---|---|---|
| 1. Replay mechanism | All ten controls exact at all four checkpoints; helper bodies unchanged | append-replay-mismatch-admitted / test_checkpoint_append_replay_mismatch_fails |
| 2. Gate | 62 fails, free 0 passes, flags cannot gate | append-top1-floor-admitted / test_checkpoint_append_top1_62_fails; append-free-divergence-gated / test_checkpoint_append_divergence_0_passes; fidelity-flag-gated / test_checkpoint_fidelity_flag_never_changes_exit_code |
| 3. F3-F5 and notes | Route preflights refuse before arms; raw validator unchanged; runbooks corrected | degenerate-route-admitted / test_route_preflight_rejects_degenerate; raw-validation-disabled / test_checkpoint_raw_logits_reject_r7r_mutant |
| 4. Regressions | Bench 93, proxy 74, native identity 12 and fresh smoke green | legacy-exactness-weakened / test_checkpoint_exactness_gate |

Runtime/test rework stays inside tools/hybrid-bench. The only outside path is the existing root README growing-session tool entry, updated because it advertised the obsolete free-divergence gate. No requested row is excluded.

### Runbooks, evidence limits and resource controls

Heterogeneous runs remain exit 5. The Mac runbook names CHAIN-H, A-L, OWN_NATIVE-A and OWN_RANGE-A (OWN_*-A) as client-backend identities expected exact even with a CUDA node; their mismatch is a finding. Cross-backend NODE/RESTORE/BSTATE/NODE_RANGE/CHAIN_ATTENTION-to-B comparisons remain report-only there. Probes are mandatory; stale optional/unchanged-L-H wording is removed.

Full attention refresh is **34,816 B/token** for the 27B q8_0 layout: 16 attention layers x 2 K/V x 4 KV heads x 256 width x 34/32 bytes. At 100,000 tokens that is 3,481,600,000 B (**3.4816 GB**); at 216,064 it is 7,522,484,224 B (**7.5225 GB** rounded). Add recurrent state, row metadata, envelope and transport overhead. A full refresh restores equality to B, not L. Refresh is documented only and not implemented.

The first build admission refused 1 because another lane was compiling; no configure/build ran until that lane stopped. First smoke `native-01` exited 1 after the nine instrument controls when the all-local preflight driver refused 2 on its conservative disk reservation before creating a trace. The following missing-trace error is not route-gate evidence. No criterion or integrity check was weakened: hash-verified sealed duplicate logits were hardlinked, retaining every path and byte, then the entire smoke was rerun fresh as `native-02`. Prior/current dedup records and disk observations are retained. Cleanup01 refused 1 because its PID check included its own helper; no files were removed. Report01 consequently exited 1 on missing cleanup evidence and wrote no section. Cleanup02 excludes only its current helper PID, verifies all other recorded PIDs inactive and exits 0; Report02 follows that successful evidence.

Minimum observed disk throughout r41 was **29.791694848 GB**, above 28 GB. After owned build/mutant executable/expired fixture-credential cleanup, free disk was **31.319572480 GB**. All recorded task runtime PIDs were inactive at cleanup; the task disk observer was stopped first. Source, exact tokens, state/delta evidence, raw logits, SHA identities, command logs and both failed/corrected controls are retained under `.temp/r41/`; identical sealed logits share inodes. The task build tree is deleted.

No requested row is out of contract. All pre-existing tests are retained except the explicitly requested replacement of the old free-divergence boundary test. Private self-review is `.temp/r41/self-review.md`; no board is configured, so the established `.temp/tasks.md` tracks scope/evidence. Missing ty/editorconfig-checker/ec means type/editorconfig tool checks were not run; hosted full-platform CI is unrun. GPU/Metal, 27B, 100k/deep and economics remain unqualified. The bounded CPU admission smoke is green; r41 awaits independent review.

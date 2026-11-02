# Growing-session scenario, baseline and trace

The original sections record r36. The [r37 revision](#r37-native-depth-and-session-measured-acceptance) below supersedes its depth and acceptance decisions; the original measurements remain historical evidence.

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

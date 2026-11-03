# r46: timing-only Qwen/Glimmer driver and operator packets

Ready for review on m3-growing-r46, based on the owner-pinned72566bca. Local signed delivery only; no push. Scope: driver/checker, fixture reader/data, focused tests and r46 packets/docs. Core server, proxy, state transport, model implementations and GPU backends are unchanged. The owner supplied the design and explicitly requested the commit/co-author trailer for this private fork; no upstream submission is prepared.

`--checkpoint-turns none` now disables the unconditional live-node capture path, checkpoint checker and partition arm. It refuses state-probe/retention flags before metadata reads. The trace records capture mode and counts, stable arm definitions and per-call transport profiles. Successful timing exits0/statusmeasured; it carries no fidelity verdict. An all-local policy on a slow link is measured without forced offloads; fidelity mode still requires both connected routes and recovered offload after outage. Production rollback saves, import validation and normal metering remain in timing.

The PRI31 Qwen fixture is reconstructed from the actual owner-pinned archive c2e83fbd4f59a38181e48b17206783e91d6d5b6734624f46d4cede68d7bcace5. It records raw/priority evidence hashes, model identity, every512/2000/10000 rate and weighted decode by level. Routing retains Policy's delta512 floor curve. At5.19/9.75MB/s the illustrative corrected Qwen plan has no turn offloads; at16/25MB/s it has5/7. These are plans, not new link measurements. The Mac payload measures its actual link at start, records5G/Wi-Fi and uses the existing PRI31 supervisor and authenticated r44 helpers.

Glimmer uses76000 start and131072 native capacity, requires window2048/no recurrence, includes window bytes in reservations, and substitutes CHAIN_WINDOW-H/window_equal for CHAIN_RECURRENT-H/recurrent_equal. The checker derives the component from model memory/layout and refuses a mismatched fixture. All ten exact controls and L-H>=63/64 remain; the packet explains each. Its native SWA split uses the existing iswa base-cache getter and counts serialized base bytes before the window suffix. CPU gate-algebra tests are not inference qualification of that split.

## Packets and estimates

- `packets/r46/ADDENDUM-MAC.md`: exact sequential Qwen and Glimmer capture-free payloads, calibration/PRI31 admission, link/exit/archive instructions and disabled bf16 design hook.
- `packets/r46/REHEARSAL-GLIMMER.md`: one-PRO6000 CUDA loopback,40 turns, checkpoint10/20/30/40, outage20+3, same ten-control/top1 gate.
- `packets/r46/README-NODE.md` and `stage-{qwen,glimmer}-node.sh`: pinned Q4_K_M model hashes, context196608/131072, fresh/stamped CUDA node builds and private metadata transfer.
- `packets/r46/planning.json`: reproducible arithmetic. Qwen3-4.5h,5.5-7.0GB including base/link probe, plus roughly10-30GB calibration;80GB run-volume free admission, RAM formula about73GiB for17GB weights. Glimmer hypothetical2-4.5h,1.2-1.8GB plus roughly2-8GB calibration;45GB free, about47GiB for18GB weights. Glimmer GPU rehearsal3-6h,120GB free,96GB host/GPU RAM, roughly40-60GB conservative loopback capture allowance. These are conditional operational estimates, not measurements or total physical traffic bounds.

Glimmer seed1/framing32 ends128011, leaving3061 tokens below131072, including the2048 margin. Framing16 ends127371. Actual template framing is checked before any arm. At80000 the sampled sequence fails; it is never truncated.

## Validation and actual exits

Scratch/evidence root: `/home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r46-261103/`. All runs CPU-only on agents1 with on-disk TMPDIR, PYTHONDONTWRITEBYTECODE=1 and no downloads. No child agents were launched.

| Check | Actual exit and evidence | Bound |
|---|---|---|
| Configure, native Release build (server/prefill/checker) | configure0; first build2 from a unique_ptr passed without.get(); corrected build0 in candidate-build-02.log/build-record-02.json | Existing CMake4.4.4/GCC15.2, nice19/-j4, shared native build lock; minimum free33.47GB during first build |
| Growing Python suite | 44 tests,0; python-06.log | All original36 test names retained; timing option/refusals, native-depth/window gate, fixture/model/priority/admission and packet command tests |
| Native C++ gate controls | 0; append-controls-01.log, native-validation.json | Recurrent and window exact/byte negative controls, count10 and63/62 boundary; no SWA inference |
| Backend label controls | 0; backend-labels-01.log | Original six controls retained |
| Existing native small-Qwen checkpoint controls | 0 overall; individual expected negatives2; native-controls-01.log/native-validation.json | Snapshot equality/roundtrips, checksum/journal/context/manifest/probe refusal; unchanged reviewer tests retained |
| Final real two-arm timing smoke | 0; native-timing-final-command-02.json, native-timing-final-02.log | Existing0.8B QwenQ4_0, depth128,12 sampled turns/input48/output128/pause1.1 caps, seed281; real imports, connected local/offload, outage/recovery, cleanup and zero node/checker capture |
| Real all-local timing control | 0; native-timing-final2/timing-all-local-command.json | Same existing CPU model, depth128,3 turns/input48/output16, no pauses, seed281; no forced turn offload, both arms measured, no captures |
| Static packet/source checks | 0; static-checks-01.log/static-checks.json | Bash and embedded Python syntax, new/added ASCII, JSON, workflow model naming, git diff --check, exact archive-fixture and planning regeneration |
| ty/editorconfig-checker | 127 each; type-check-01.log/editorconfig-01.log | Tools absent; checks unrun, no install/download attempted |

Earlier CPU timing runs were green but the final source was replayed after the route/profile changes; only the final2 run is the final-source native timing evidence. Failed first compile is retained, not presented as green. Target Metal/CUDA builds and real Mac/Glimmer inference are not performed here. Owned native build trees and large CPU fixture states/logits are removed after hashes/exit evidence are retained in cleanup.json; those deleted payloads are not available for offline recheck.

| Mutant | Gate narrowed to | Named failing test | Actual result / surviving bound |
|---|---|---|---|
| snapshot-added-to-timing | Timing extend never calls snapshot download | test_timing_node_request_has_no_snapshot | exit1, intended unexpected download assertion; killed |
| skip-window-byte-gate | A forged green summary cannot conceal unequal window bytes | test_window_gate_replaces_recurrent_without_weakening | exit1, required Refusal not raised; killed |
| accept-placeholder | A full fixture with placeholder status cannot enter measured routing | test_pri31_fixture_admission | exit1, required Refusal not raised; killed |

Plants ran in isolated in-memory module copies; source SHA, mutant SHA, PID and exact failure are in mutation-controls.json and named logs. No survivors among these three. These controls do not cover arbitrary instrumentation, the physical SWA serialization split, or target-device numerical behavior.

## Remaining qualification and review boundaries

No surface table was supplied (brief gap). No requested task row is excluded. The ten-control Glimmer map is in REHEARSAL-GLIMMER.md. This is a new owner-requested leaf, not reviewer rework; changes outside tools/hybrid-bench are the root README link only.

Glimmer's actual PRI31 step40 fixture is pending. The shipped placeholder is refused by the Mac payload and direct Metal driver. No small SWA weights were found locally (only vocabulary-only Gemma GGUFs); no SWA inference test was run. Native slot restore/compaction, beyond-window physical placement, range imports, component replacement and GPU exactness remain the rehearsal's explicit gates; synthetic gate tests cannot close them. No fallback control or looser threshold was added.

The dev-d r13 timestamp list has not arrived through the owner. The trace reserves versioned instrumentation fields; adding the requested cheap timestamps remains a later delta when the list arrives. hybrid-bf16 is named/reserved but unavailable pending dev-a r38 and independent protocol/import qualification; existing hybrid remains f32. The Mac timing run's fidelity stays with the earlier r44/r45 evidence. Hardware execution, independent review and acceptance are pending.

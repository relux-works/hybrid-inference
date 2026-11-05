# r46d rehearsal routing and client build repair

Ready for review on m3-growing-r46, based on accepted d06df13fd6496bb89f0dc9cb4ef69f42c9f8a4c4. No push, GPU use or downloads. Evidence is under /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r46d-261103; incremental packet under its packet/ directory. The source bundle requires accepted r46c history. No surface table was supplied in this brief; no requested contract is excluded.

## Behavior and source

The Qwen node-E aborted workload equals the unchanged seed1 sampler at100k/framing14/40 turns. The old16MB/s/local100/PR9 import policy reproduces37 local/0 offload and the exact preflight refusal. The real input range is6-2329. The replacement uses the existing r46b20MB/s planning scenario and measured PRI31 Qwen delta512 floor rates, with actual34816/156893184 geometry. It predicts30 connected local/7offload plus outage2-4. BF16_CHAIN replays those captured f32 boundaries: seven predicted turn imports beyond the base, final-import counts2/4/4/8 at checkpoints10/20/30/40. An independent bf16 policy would predict28 local/9offloads, separate from chain replay. Base-only chain replay now refuses before output allocation. Actual captured turn imports and cumulative counts are saved with replay metrics; live link EMA may change runtime routes.

Glimmer's real7072-byte attention geometry and43514896-byte window are pinned to the kit1.6.5 geometry source in REHEARSAL-POLICY.md. Its old16MB/s policy already gives26 local/11offload, so an old-policy degeneracy claim is contradicted by evidence. The new sourced20MB/s/interim100tok/s profile gives the same26/11 with unchanged76000 start, framing32 and outage20-22. Its negative uses2MB/s and requires37/0 plus the specific refusal. No window bytes, samples or turns are changed to force admission.

Both runbooks contain sequential operator command lists: stage node, run setup-rehearsal-client.sh, generate policy, stop launcher, run baseline. The wrapper omits --skip-build, selects the right context/node state, and invokes manage.py client at R46_HEAD/BUILD under NODE_ROOT/build.lock. All three client targets are built and recorded by the launcher. Native source, f32 wire/import, timing arms, calibration budgets and exact controls are unchanged. planning_policy is shared between estimate.py and the new rehearsal policy generator; existing planning JSON and LINK-PLANNING.md reproduce byte-for-byte.

Resource reservations remain Qwen6-10h,450GB work+30GB selective archive,128GB host/96GB GPU; Glimmer3-6h,120GB work,96GB host/GPU after weights/build. Qwen's prior$1.57/h scenario remains$9.42-$15.70, not a current quote. Route predictions do not reduce snapshot disk reservations or prove quality. Rehearsal policies are illustrative and satisfy no Mac calibration admission.

## Validation and finite limits

Direct final validation recorded22 subprocess commands and their actual exit codes in validation.json. Growing-session59 tests exit0; benchmark58 tests exit0 with one existing native-integration skip. All previously committed tests retain their names. Runbook client command lines are executed in CPU fixtures; real manage.build/compile_targets stamp/hash logic reproduces missing binaries and manual unrecorded binaries, preserves the node stamp after rejection, then records all targets, accepts reuse and rejects a changed binary. Compiler processes and git identity are fixtures: this does not claim native compilation. The25 GiB precompile floor refuses before any configure/build dispatch. Native source is unchanged and no native build was run for this repair.

The policy generator's real CLI succeeds on both actual geometries. Invalid geometry exits2 before either output file exists. The replay CLI rejects a base-only trace with exit2 before output allocation. All five packet shell scripts and four rehearsal Bash blocks pass bash -n; changed Python parses; added text is ASCII and git diff --check exits0. The seven-link planning JSON and Markdown are unchanged. ty/editorconfig-checker are unavailable on this host, not reported passing. task-board goal get exits1 because no .task-board exists; the established .temp/tasks.md tracker is used.

An intermediate59-test replay exited1 (three failures and one error) when shared free disk dropped to27.82GB and existing recheck logit admission refused. This failed run is retained as growing-disk-refusal.log. Removed24 owned regenerable timing-only decoder slot caches from earlier r46b/r46c runs,546234708 bytes, with paths/hashes in cleanup-prior-timing-slots.json. Traces, native timing spans, fidelity snapshots and raw logits were retained. Fresh full validation then exited0 with29.09GB free. No build tree or foreign process was created or removed. An initial tail -7 read failed on the host's CLI; tail -n7 corrected it.

## Narrowing controls

Each mutant was applied to an owned source file alone, tested as a standalone process, then restored byte-for-byte. Mutant commands all exited1 at the named tests; the control harness exited0. No survivors.

| Mutant | Contract narrowed | Named failing test |
|---|---|---|
| qwen-old-policy | Real seed1 schedule cannot admit the failed16MB/s/local100 policy | test_qwen_rehearsal_old_16mb_policy_refuses |
| glimmer-window-omitted | Policy cost must retain the fixed real SWA window | test_glimmer_rehearsal_low_link_refuses_and_old_policy_is_mixed |
| client-skip-build | Actual runbook command must build/record client targets | test_rehearsal_client_build_records_missing_targets |
| base-only-chain | Repeated-rounding rehearsal needs turn imports beyond base | test_bf16_rehearsal_base_only_chain_refuses |

CUDA compilation, large-model inference, actual four-checkpoint f32/bf16 quality, live route counts and Glimmer SWA native identities remain operator execution and independent review gates. CPU fixtures and route arithmetic do not close them. Prior lossy/exit5 semantics, L-H and L-BF16>=63/64 gates, KL references and free-running divergence reporting remain intact.

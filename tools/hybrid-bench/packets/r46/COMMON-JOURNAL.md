# r46e declared common-journal fidelity design

This design is recorded before implementation and before any new GPU execution. The node-F Glimmer trace at7d512238 aborted on differing free-running journals before any checker ran. Its SHA256 is2c462979aff26108efefa3db9b948470855bf7b686f6a3b672368369f10e165f. The actual template framing is7, final depth127011 from76000/40/seed1, within native131072. Changing batching or the base prefill path can change Glimmer's greedy output. Such a difference does not contradict teacher-forced fidelity on identical inputs.

## Explicit admission

The default contract remains independent free-running arms with required equal journals. Qwen keeps that contract and every existing equality/refusal/control. Common-journal fidelity requires an explicit --pairing-contract declaration, bound to the exact GGUF hash, architecture, sliding window and absent recurrence, declaring free_running_batch_invariant=false. The packet supplies this measured-model declaration for the pinned Glimmer file, with the node-F trace as evidence. The driver verifies and records the declaration before any output or arm. A missing declaration never triggers fallback after a mismatch. The common mode requires fidelity checkpoints and state probes; it is not a production timing mode. Qwen's recurrent layout cannot use this declaration.

## Common journal and reference histories

Only the hybrid arm generates the40-turn session. Its exact sampled token IDs, not re-tokenized answer text, are authoritative. At each checkpoint the live H snapshot and common journal are retained. The native checker replays the entire captured journal in independent local L and partition P contexts: incoming prompt spans at256/256 for L,64/32 for P, and each recorded generated token evaluated individually. Both include the same forced base and all post-outage history. No local or partition greedy token may replace a journal token.

L and P are teacher-forced reference histories, not live free-running arms and not production TTFT measurements. Each reference crosses a native save/restore boundary immediately before the last common token, matching the H snapshot's unreconciled-token boundary. A independently replays the local schedule and must equal L exactly. This keeps A-L as an independent replay versus native roundtrip identity rather than dropping it or aliasing two observations. P is a partition sensitivity measurement, not an exact L-P identity.

The checker then scores64 teacher-forced positions on the same local greedy continuation for H/L/P and every applicable replay/probe. It records the actual teacher token IDs and free-running token IDs for those reference observations. Python verifies equal teacher inputs and journal binding. KL(L||H), position agreement and top1>=63/64 retain their existing definitions. It also reports the first independent free-running divergence and the token streams at each checkpoint, with no rejection based on H-L or L-P free-running divergence. These64-token probes do not claim to measure a full alternate40-turn free-running conversation.

## Exact controls and custody

All ten existing controls remain: NODE-B, RESTORE-B, BSTATE-B, NODE_RANGE-B, OWN_NATIVE-A, OWN_RANGE-A, CHAIN-H, CHAIN_ATTENTION-B, CHAIN_WINDOW-H and A-L. Window/state bytes, native and range save/restore, imported chain boundaries, hashes and journals remain exact. Divergence inside an exact identity remains a failed identity. Only cross-path H-L/P free-running differences are report-only. No tolerance or failed-control exception is introduced.

The driver checks checkpoints during the hybrid arm, since no independent local/partition generating arms follow. The trace labels live and replay references explicitly. Preserved checkpoint recheck reconstructs the same one-H-snapshot fixture, declaration, captured schedule and references, and verifies all hashes before computation. The existing Qwen three-snapshot path remains unchanged. The native reference replay primitive is model-independent; a locally available small Qwen may test its mechanics without changing Qwen driver admission or qualifying Glimmer SWA. No small SWA weights are currently available on agents1, so native Glimmer window behavior remains a GPU qualification gap.

## Fresh GPU budget

Node F measured about25 minutes for hybrid alone including pauses at the real framing7. The new run adds four full checkpoint programs: two teacher-forced references, four factorial schedules and nine state/transport probes, plus64-token observations and storage/model-load overhead. Plan4-7 GPU hours for the full fresh run including build/setup, rather than treating25 minutes as the total. This is an engineering allowance, not a measured total: time checkpoint10 and revise the remaining estimate from actual phase timings. Keep the existing120GB free after weights/build,96GB host/GPU RAM, and28GB runtime floor. No GPU time is spent by this CPU-only repair.

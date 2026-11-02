# Unresolved questions

R40 resolves the acceptance decision: growing-session append imports use exact CHAIN-H plus the r39 component controls, exact journals/hashes, top-1>=63/64 and no free-running divergence in the first8 tokens. L-H KL/p95/max are reported costs, with no zero-KL gate. See [runbook](tools/hybrid-bench/GROWING-SESSION.md) and [r40 results](tools/hybrid-bench/RESULTS-hybrid-b-r36-261102.md#r40-mechanism-based-append-import-gate). Historical r36-r39 numerical rules remain labelled in their evidence.

The CPU mechanism is mixed attention-cache provenance: append keeps older local KV, appends the node suffix and replaces recurrent state. Full attention refresh would remove that mixture, at34816bytes/token for illustrative27B q8_0 geometry. It restores the full-node same-partition reference B; differently batched local L also needs matching arithmetic for exactness. No production protocol/state change is authorized or implemented here.

Native deep start remains the largest1024-token start that fits all sampled growth plus2048 margin within262144. No250k start, dropped turn, hidden cap or YaRN substitution.

GPU/Metal and100k/deep measurements remain operator-host work. Both runbooks repeat all state controls. CUDA-to-Metal remains heterogeneous, exit5/uncalibrated. CPU0.8B/4096 evidence cannot qualify hardware or M3 economics. The CPU turn12 free-running divergence at0 remains a token-integrity failure under r40.

Canonical M3 Max delta-prefill rates and measured PR9 CUDA import OLS have pinned source hashes. CUDA rehearsal uses target-M3 routing rates with its labelled CUDA import fit. Mac economics require a measured multi-point M3 import fit; the single42.4s/39504-token observation cannot identify intercept and slope. Import-speed-v2 CPU speedups are not a measured M3 fit.

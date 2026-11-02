# Unresolved questions

Growing-session r38 supersedes r37's partition-band acceptance rule. The factorial replay does not establish that generated-token batching explains the live hybrid deviation. The same-backend gate is uniformly mean KL(local || hybrid)=0, with top-1>=63/64 and no independent greedy divergence over 64 tokens. Partition mean/p95 remain diagnostic. See [r38 results](tools/hybrid-bench/RESULTS-hybrid-b-r36-261102.md#r38-factorial-replay-and-pre-hardware-criterion) and [runbook](tools/hybrid-bench/GROWING-SESSION.md). Numerical equivalence and hardware admission remain unqualified; no positive tolerance or changed reference class is authorized by these controls.

Native deep start remains resolved as the largest1024-token start that fits all sampled growth plus 2048 margin within 262144. No250k start, dropped turn, hidden cap or YaRN substitution is introduced.

The r39 CPU probes identify the residual B-H source as mixed attention-cache provenance: append imports retain older local KV rows, append node KV for the new range, and replace recurrent state with the node's state. The live node snapshot and B are exact; own-state roundtrips are identity; replaying the actual append chain reproduces H. Replacing attention KV alone makes that chain exact with B, while replacing recurrent state alone leaves it exact with H. See the r39 section of the growing-session results for both checkpoints and the matched-thread/affinity controls.

Exact production catch-up still needs an owner-selected evaluation/state contract. The diagnostic replay does not change the flat production protocol or replace old client KV. Batch/token evaluation differences plus retaining their different historical KV explain these CPU observations; making L-H exact requires addressing that contract rather than relaxing the gate. CUDA/Metal must repeat the state-path controls; a GPU is not assumed to remove the mechanism.

Canonical M3 Max delta-prefill rates and the measured PR9 CUDA import OLS are retained with source hashes. The single M3 import point42.4 s/39504tokens cannot identify both an intercept and slope. The CUDA rehearsal uses its labelled measured CUDA import fit with target-M3 routing rates; M3 import economics require a measured M3 fit. Import-speed-v2 CPU speedups are not a measured M3 fit.

GPU/Metal and 100k/deep hardware measurements remain operator-host work. CPU 0.8B/4096 controls do not qualify them, and the CPU numerical failure remains blocking.

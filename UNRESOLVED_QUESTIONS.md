# Unresolved questions

## hybrid-b-r36-261102: live-session numerical gate

The requested exact CPU/CUDA checkpoint gate is stricter than the current live proxy/local baseline behavior. The first corrected CPU checkpoint at turn 10 has top-1 64/64 and KL(local || hybrid) 0.0006673842215735586 nats/token, with nonidentical logits. Different prefill partitions are a known source of numerical variation. The driver retains the strict gate and reports failure; it does not substitute the older numerical or repeat-floor bounds. Decide whether exactness remains blocking or whether the live-session experiment may use a stated numerical bound, keeping exactness for partition-matched controls. Evidence: [r36 results](tools/hybrid-bench/RESULTS-hybrid-b-r36-261102.md), task `.temp/research.md`, and `checkpoint-retry-01.log`.

## hybrid-b-r36-261102: 250k growth capacity

The default seed-1 sequence needs about 293k tokens from D0=250000. The launcher caps context at GGUF training context; the existing Qwen3.5 fixture advertises 262144. Choose an explicitly capped 40-turn sequence (for example 128 input and 128 output tokens per turn), or a model/configuration already qualified for at least 300k context. No context-extrapolation override or silent clamping was added. The capped sequence changes the measured call mix and must be identified separately. The GPU and Mac runs remain unexecuted on agents1 by instruction.

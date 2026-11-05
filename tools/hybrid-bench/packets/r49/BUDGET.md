# Staged Mac budget (FACTS-rate planning)

Run `python3 tools/hybrid-bench/packets/r49/budget_wall.py > NEW-BUDGET.json` from the exact source. BUDGET.json retains all inputs, source artifact hashes, arm components and unscaled local-fill brackets. These are public-reference planning estimates, not owner or target-model results. The owner set is unavailable.

| Path MB/s | Public: four arms, hours | Main C/A: ten arms, hours | Optional B/A: ten arms, hours | C turn offloads /68 |
|---|---:|---:|---:|---:|
| 3.75 | 7.07 | 16.71 | 16.75 | 1 |
| 6 | 6.81 | 16.21 | 16.66 | 20 |
| 18.75 | 5.85 | 14.42 | 14.43 | 58 |
| 27.875 | 5.68 | 14.07 | 14.07 | 62 |
| 50 | 5.53 | 13.74 | 13.74 | 63 |
| 100 | 5.43 | 13.53 | 13.53 | 65 |

1. Public rehearsal on iPhone USB: 5.68-5.85 h at 18.75-27.875 MB/s, plus one-time setup/calibration 0.5-1 h. Allow 6-8 h including the raw-fill uncertainty bracket. It must pass every gate before main.
2. Main C/A fast: 14.07-14.42 h for five public-like pairs, plus the cold/checker raw-fill bracket. Reserve 15-18 h initially, and reprice from actual public checker timings before owner inference. A 512-token/turn sensitivity adds approximately 10.0 h of decode at the recorded depths; reserve about 25-30 h for such histories. This sensitivity holds depths fixed and omits extra-context slowdown, so it is not a new selected workload or an upper bound.
3. Decision before any optional block: B/A fast adds about 14 h short-response planning; home B/A or C/A adds about 16-17 h each. Each has its own five-pair local baselines. Favourable n=3 costs roughly three-fifths of a five-pair block only if its actual work matches this proxy; price the frozen set before authorization. Optional blocks remain separate comparisons, never partial headline results.

Quality placement on the delivered reference: turns 29 and 68, depths 146142 and 203809. Turn 29 is the earlier of equally near eligible following 60 s waits. Two fresh local reference fills cost about 1.09 h per hybrid arm, zero per local arm, compared with the reviewer's four-per-arm 2.05 h. Checkpoint observation/load adds 60 s each. Reduced coverage is explicit in the amendment; the 64-position >=63/64 gate is unchanged.

Warm timing uses estimate.py with the measured PRI31 depth fixture, 1097 tok/s prior server surrogate, import 1.498 s + 0.0487 ms/token and bf16 state 78,448,192 B + 34,816 B/KV token. Cold/checker fill integrates segment rates, scaled by the FACTS 59.7 min cumulative-to-249k observation; raw unscaled fill retained. Cooldown 300 s, decoder load allowance 90 s and deferred hash allowance 60 s per arm. Public recorded waits total 856 s; output lengths are about seven tokens. Real owner waits and output lengths are still unknown.

Option (b) qualification: pre/post T/R/T uses 2.25 GiB per block; ordinary 384 MiB policy probes before/after add 0.75 GiB. Thus public two blocks uses 6 GiB and main one block 3 GiB, beyond inference envelopes and <=2 GB shared calibration. Qualification hours are in the table at nominal payload rates; practical overhead and the 80% minimum increase them. Option (a) spends no Mac bandwidth on the six-grid loopback test and needs about one VM-hour including setup. Ordinary policy probes still run per Mac block.

Home Wi-Fi is expected to show little or no gain, retained and explained through the fixed portion bins. A headline still requires every sampled main session positive and median >=10%. n=5 bootstrap interval is [min,max]. No Mac budget decision changes that claim rule.

Actual 200k checker time is a go/no-go budget input: if it exceeds this bracket, stop before main and report the revised hours. Do not reduce the gate, select faster sessions, or reroll a failed main series.

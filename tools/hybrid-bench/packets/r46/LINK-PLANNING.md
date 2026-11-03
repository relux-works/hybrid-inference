# r46b link-rate planning

Decimal MB/s. Seed1,40 turns, outage2-4. Base is forced offload by the current pre-staged driver at every positive link rate:100k Qwen,76k Glimmer. It is outside the measured turn window. A non-empty *turn* offload arm has the profile-specific threshold below; including the forced base, every arm is non-empty for any positive link.

Wire is estimated request/delta application bodies, excluding calibration, link probes, HTTP/SSH/TCP overhead/retransmissions and captures. Fixed state occurs only in the last chunk. The hybrid-bf16 arm is enabled with accepted transport; this planning model changes recurrent tensor bytes only. BF16 import costs and quality remain unmeasured and unchanged in this calculation; the Mac run requires its own bounded fit. Glimmer curves are interim hypothetical sensitivity cases (prefill/decode tok/s), not PRI31 fixtures.

TTFT totals include base plus40 turns; turn-only totals are also shown. They model prompt/policy costs, one held local prompt token and1/decode_rate first-sample proxy; queue, node load/save/serialization, readback and other residual costs remain unmodeled. No pauses or remaining output decode are added. RTT0.1406s, eta0.85, prior Qwen import1.498s+0.0000487s/token; Glimmer uses that fit as an explicitly interim portability assumption.

## Qwen f32 PRI31

First connected turn offload above approximately 10.234848 MB/s (strict policy inequality; not a measured link).

| MB/s | Turn offloads /40 | Base offload | Wire GB | Hybrid total TTFT s | Local total TTFT s | Hybrid turn TTFT s | Local turn TTFT s |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 2 | 0 | yes, forced | 3.643 | 2458.82 | 950.97 | 253.43 | 253.43 |
| 3 | 0 | yes, forced | 3.643 | 1744.77 | 950.97 | 253.43 | 253.43 |
| 5 | 0 | yes, forced | 3.643 | 1173.52 | 950.97 | 253.43 | 253.43 |
| 10 | 0 | yes, forced | 3.643 | 745.08 | 950.97 | 253.43 | 253.43 |
| 20 | 7 | yes, forced | 5.122 | 468.72 | 950.97 | 191.29 | 253.43 |
| 30 | 8 | yes, forced | 5.307 | 365.52 | 950.97 | 159.49 | 253.43 |
| 50 | 10 | yes, forced | 5.660 | 277.49 | 950.97 | 128.59 | 253.43 |

## Qwen hybrid-bf16 wire estimate

First connected turn offload above approximately 6.867503 MB/s (strict policy inequality; not a measured link).

| MB/s | Turn offloads /40 | Base offload | Wire GB | Hybrid total TTFT s | Local total TTFT s | Hybrid turn TTFT s | Local turn TTFT s |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 2 | 0 | yes, forced | 3.565 | 2412.68 | 950.97 | 253.43 | 253.43 |
| 3 | 0 | yes, forced | 3.565 | 1714.00 | 950.97 | 253.43 | 253.43 |
| 5 | 0 | yes, forced | 3.565 | 1155.06 | 950.97 | 253.43 | 253.43 |
| 10 | 6 | yes, forced | 4.379 | 696.41 | 950.97 | 213.99 | 253.43 |
| 20 | 9 | yes, forced | 4.700 | 426.92 | 950.97 | 154.10 | 253.43 |
| 30 | 10 | yes, forced | 4.797 | 332.39 | 950.97 | 129.44 | 253.43 |
| 50 | 11 | yes, forced | 4.889 | 255.07 | 950.97 | 108.02 | 253.43 |

## Glimmer interim 50/12

First connected turn offload above approximately 1.095276 MB/s (strict policy inequality; not a measured link).

| MB/s | Turn offloads /40 | Base offload | Wire GB | Hybrid total TTFT s | Local total TTFT s | Hybrid turn TTFT s | Local turn TTFT s |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 2 | 3 | yes, forced | 0.793 | 788.80 | 2020.80 | 413.25 | 500.71 |
| 3 | 7 | yes, forced | 1.007 | 613.47 | 2020.80 | 352.31 | 500.71 |
| 5 | 10 | yes, forced | 1.162 | 430.52 | 2020.80 | 260.88 | 500.71 |
| 10 | 10 | yes, forced | 1.162 | 294.05 | 2020.80 | 193.05 | 500.71 |
| 20 | 12 | yes, forced | 1.254 | 220.23 | 2020.80 | 153.55 | 500.71 |
| 30 | 15 | yes, forced | 1.390 | 192.67 | 2020.80 | 137.43 | 500.71 |
| 50 | 18 | yes, forced | 1.524 | 168.58 | 2020.80 | 122.48 | 500.71 |

## Glimmer interim 100/18

First connected turn offload above approximately 2.290039 MB/s (strict policy inequality; not a measured link).

| MB/s | Turn offloads /40 | Base offload | Wire GB | Hybrid total TTFT s | Local total TTFT s | Hybrid turn TTFT s | Local turn TTFT s |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 2 | 0 | yes, forced | 0.585 | 626.43 | 1010.97 | 250.91 | 250.91 |
| 3 | 1 | yes, forced | 0.667 | 494.06 | 1010.97 | 232.94 | 250.91 |
| 5 | 3 | yes, forced | 0.793 | 369.21 | 1010.97 | 199.61 | 250.91 |
| 10 | 10 | yes, forced | 1.162 | 244.18 | 1010.97 | 143.22 | 250.91 |
| 20 | 10 | yes, forced | 1.162 | 175.95 | 1010.97 | 109.30 | 250.91 |
| 30 | 10 | yes, forced | 1.162 | 153.20 | 1010.97 | 98.00 | 250.91 |
| 50 | 12 | yes, forced | 1.254 | 133.46 | 1010.97 | 87.40 | 250.91 |

## Glimmer interim 150/24

First connected turn offload above approximately 3.598488 MB/s (strict policy inequality; not a measured link).

| MB/s | Turn offloads /40 | Base offload | Wire GB | Hybrid total TTFT s | Local total TTFT s | Hybrid turn TTFT s | Local turn TTFT s |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 2 | 0 | yes, forced | 0.585 | 542.96 | 674.17 | 167.46 | 167.46 |
| 3 | 0 | yes, forced | 0.585 | 428.56 | 674.17 | 167.46 | 167.46 |
| 5 | 1 | yes, forced | 0.667 | 324.20 | 674.17 | 154.61 | 167.46 |
| 10 | 4 | yes, forced | 0.849 | 226.43 | 674.17 | 125.48 | 167.46 |
| 20 | 10 | yes, forced | 1.162 | 159.13 | 674.17 | 92.51 | 167.46 |
| 30 | 10 | yes, forced | 1.162 | 136.39 | 674.17 | 81.20 | 167.46 |
| 50 | 10 | yes, forced | 1.162 | 118.19 | 674.17 | 72.16 | 167.46 |

## Qwen three-arm total

Run order: hybrid, local, hybrid-bf16. Body bytes include both forced bases; one403MB link probe and the shared <=2GB calibration are additional. Hours include generated output, sampled pauses and bases, with25-50% operational allowance still to add. The bf16 row assumes the f32 import fit only for planning.

| MB/s | Total body GB | Three-arm hours before allowance |
|---:|---:|---:|
| 2 | 7.208 | 4.46 |
| 3 | 7.208 | 4.06 |
| 5 | 7.208 | 3.75 |
| 10 | 8.023 | 3.50 |
| 20 | 9.822 | 3.35 |
| 30 | 10.104 | 3.30 |
| 50 | 10.549 | 3.25 |

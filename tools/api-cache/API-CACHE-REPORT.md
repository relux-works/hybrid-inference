# API cache report: hybrid-d-r15b-261103

Preregistration: signed commit d43113de61f01a69d32d1337cbf38049c3903f01, verified G before any paid request. Frozen endpoint list and criteria are in PROTOCOL.md and snapshots/2026-11-03.json. Private workbench branch m3-api-cache-test; no push.

## Chutes smoke, 2026-11-03 14:08:38-14:08:56 UTC

Three streaming Qwen calls pinned to chutes/fp8, no fallback, all serving-provider fields Chutes, all HTTP 200 and accepted. Total API-reported cost USD0.002841708, below USD0.50. Initial key usage was USD0.

| Call | Prompt tokens | Output tokens | Cached tokens | API USD | TTFT seconds |
|---|---:|---:|---:|---:|---:|
| 0 | 32824 | 2 | 0 | 0.001282636 | 6.583840 |
| 1 | 33824 | 2 | 32000 | 0.000198436 | 2.507999 |
| 2 | 34824 | 2 | 0 | 0.001360636 | 6.677305 |

Cached input fraction: 31.5358% across all three calls (46.615% across the two follow-ups); 1/3 calls, or 1/2 follow-ups, had a cache hit. Mean uncached input: 23,157.3 tokens/call. This is below the owner's 98.1% cache fraction by 66.5642 percentage points, and above the owner's approximately 3,200 uncached tokens/call by 19,957.3. The paper assumes 1,065 new tokens/call. One confirmed hit is not stable caching or admission. No cheapest/median baseline is selected from smoke.

Own smoke availability was 3/3 within 18 seconds. Sampled OpenRouter daily Chutes uptime was 100%; repeated windows overlap. Neither statistic proves long-term 95% availability. Two completion tokens arrived together, so decode speed is unknown rather than an invented value. No intentional pauses were added.

Timing correction: original smoke pause fields omitted post-response bookkeeping. UTC start/end timestamps give actual follow-up gaps of 0.956684 and 0.238897 seconds. Original records are retained in results/smoke-2026-11-03/; pause-correction.json records the correction. Scheduler correction and named deterministic regression test precede full run. No criteria, providers or prices changed.

## Full run

Runtime outputs are under .temp/hybrid-d-r15b-261103/full/ from the repository root, with per-call JSON, calls.csv, summary.json and its API-CACHE-REPORT.md. The launch log and PID are in .temp/hybrid-d-r15b-261103/full-02.log and full.pid. A terminal.json and final done/failed log line distinguish completion from a running or interrupted experiment. Verified detached launch at 2026-11-03T14:11:05.644500Z: PID 2910864, SID 2910864, reparented to PID 1229. The first plain-nohup launcher disappeared before creating output or recording an attempt; full-01.log records that failed control. The corrected launcher uses nohup plus setsid --fork. Consult the live files for current status; smoke does not establish full-run completion.

The matrix has 139 sessions / 1390 planned calls: 61 Qwen sessions and 78 GLM sessions. All-cold nominal cost is about USD88.13 before explicit write surcharges, so the USD45 guard can truncate coverage. Unmeasured or failed cells remain visible. Equal target-depth weights substitute for an unavailable empirical paper depth distribution and are labelled as an analysis assumption. Full admission also needs all planned attempts, >=95% own availability, known cache hits, and >=95% daily uptime at three distinct sample times.

## Instrument validation

Twelve standard-library CPU tests pass, exit 0 (.temp/hybrid-d-r15b-261103/tests-05.log). Tests cover provider mismatch/missing identity, budget refusal before POST, usage bounds and missing costs, literal-secret redaction, a planted leaking-log negative control, production-runner mock smoke, unexpected billed-cost stop, network destinations, key permissions, endpoint selection, deterministic prefix growth and complete pause accounting. Python compilation and scoped source checks passed. ty and editorconfig-checker are unavailable and were not downloaded; hosted type checking was not run. No C++ build is relevant to this isolated Python tool. No surface table was supplied; mock tests qualify the instrument, not providers.

| Mutant | Property narrowed | Named failing test | Outcome |
|---|---|---|---|
| Restore pause reference after bookkeeping | Gap includes time from previous response completion | test_pause_includes_post_response_bookkeeping_regression | Killed: 300 != 400, exit 1; pause-mutant-01.log |

Matrix notifications and remote-base freshness require an exception to the owner's OpenRouter-only outbound-network rule. That question was asked; without an exception no Matrix or GitHub network access is performed. Requested messages are saved locally with the run evidence.

## Initial full-attempt instrument failures and signed v2 correction

The initial full attempt was stopped after eight response records. GLM rejected disabled reasoning, and a Qwen 250k request was rejected because the proxy estimated 325235 input tokens from long-word text. These request-configuration errors do not establish provider availability or caching. All original intents and responses remain under .temp/hybrid-d-r15b-261103/full/ with a failed terminal record. An in-flight response was recorded before terminating the owned PID; no automatic retry or uncertain duplicate call was made.

Reported initial-full cost was USD0.1438775125; with smoke the known total is USD0.1467192205. The live spend endpoint showed only USD0.09313342 at reconciliation, so full-v2 must retain the higher known total as its prior-spend floor. It leaves optional reasoning unspecified, uses compact space-prefixed lowercase letters, and gives every session a fresh r15b-v2 prefix. It keeps the same endpoints, prices, admission criteria, target depths, pauses and 16-token completion cap. HTTP400 now stops the experiment for diagnosis. This is preregistered in PROTOCOL.md before full-v2.

Fourteen CPU tests pass after the correction (tests-07.log), including a mandatory-reasoning mock that rejects the old request flag and accepts default reasoning, and a restart-spend-floor negative control. The corrected experiment writes to .temp/hybrid-d-r15b-261103/full-v2/ with log full-v2-01.log and PID full-v2.pid. The earlier full/ attempt is retained separately and is not pooled into provider admission statistics. For current experiment status use full-v2/terminal.json or the live log; absence of terminal.json means the experiment has not reached a terminal record.

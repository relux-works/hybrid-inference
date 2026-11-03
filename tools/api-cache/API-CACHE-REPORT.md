# API cache report: hybrid-d-r15b-261103

## Current status

The corrected full-matrix workflow is detached under supervisor PID/SID 3001509. Its log is .temp/hybrid-d-r15b-261103/full-v3-01.log and its PID file is full-v3.pid. It is waiting for the outstanding full-v2 Alibaba request (PID 2935739) to return before starting fresh full-v3 sessions. No corrected full-v3 measurement is claimed yet. The supervisor refuses to proceed if an earlier request has no response or a charge is unresolved. It also refuses a candidate source change while waiting. The current-request STOP file prevents a following v2 request.

The corrected full matrix will write per-call JSON, redacted SSE journals, calls.csv, summary.json and API-CACHE-REPORT.md into .temp/hybrid-d-r15b-261103/full-v3/. terminal.json and a done/failed log line distinguish completion from running or interrupted work. At terminal state the supervisor writes notification-terminal.txt for the requested report. Matrix sending requires an exception to the owner's OpenRouter-only outbound-network rule; that question remains unanswered. No Matrix or GitHub network action was taken, and nothing was pushed.

Known reported costs before the outstanding v2 request: USD0.1467192205, including smoke. Its additional charge is not yet known. Every prior reported charge, or higher live spend, becomes the new run's budget floor. The cumulative ceiling remains USD45; no ambiguous paid request is automatically replayed.

## Preregistration and instrument

Original provider list, admission criteria and protocol: signed commit d43113de61f01a69d32d1337cbf38049c3903f01, verified G before smoke. The frozen list contains 23 Qwen and 40 GLM endpoints. Context/cache criteria produce 139 sessions and 1390 planned calls (61 Qwen sessions, 78 GLM sessions). All-cold nominal cost is approximately USD88.13 before write surcharges, so budget truncation remains possible and never qualifies an endpoint. See PROTOCOL.md and snapshots/2026-11-03.json for the unchanged inclusion/admission rules.

The corrected candidate is signed commit e0e7fb401, after explicit dated protocol corrections and preservation of the original attempts. Qwen keeps the reasoning-off configuration measured in smoke; GLM uses its mandatory reasoning at the supported low effort, both capped at 16 completion tokens. Full-v3 uses compact seeded space-prefixed letters, fresh r15b-v3 session prefixes, exact order/only tags plus ignores for frozen regional descendants, a separate Linux 300-second wall-clock request deadline, flushed redacted event journals, and the carried prior-spend floor. These are uniform model/routing rules, not post-hoc provider exclusions.

Full-session depths are terminal targets to fit exactly-1M endpoints, with nine approximately 1000-token additions. Actual usage must match the declared target tolerance. Report actual and intended pause levels; a serial scheduler can lengthen intervals. Effective costs use 1065 new input tokens, measured prefix retention and 512 output tokens. An empirical paper depth distribution was not supplied: the equal-depth mixture is labelled as an assumption, and per-depth prices are retained for reweighting. Missing depth/cache evidence yields unknown aggregate prices. No cheapest or median provider is selected before admission.

## Chutes smoke, 2026-11-03 14:08:38-14:08:56 UTC

Three streaming Qwen calls pinned to chutes/fp8, no fallback, serving-provider fields all Chutes, all HTTP200 and accepted. API-reported total USD0.002841708, below USD0.50. Initial key usage USD0. Original evidence: results/smoke-2026-11-03/.

| Call | Prompt tokens | Output tokens | Cached tokens | API USD | TTFT seconds |
|---|---:|---:|---:|---:|---:|
| 0 | 32824 | 2 | 0 | 0.001282636 | 6.583840 |
| 1 | 33824 | 2 | 32000 | 0.000198436 | 2.507999 |
| 2 | 34824 | 2 | 0 | 0.001360636 | 6.677305 |

Cached input fraction: 31.5358% over three calls, or 46.6146% over the two follow-ups. One of three calls, or one of two follow-ups, hit cache. Mean uncached input: 23157.3 tokens/call. Against the owner's logs, this is 66.5642 percentage points below 98.1% caching and 19957.3 tokens/call above approximately 3200 uncached tokens. The paper assumes 1065 new tokens/call. This confirms a cache hit but does not establish stable caching or admission.

Own smoke availability: 3/3 within 18 seconds. Sampled daily Chutes uptime: 100%; repeated rolling windows overlap. Neither establishes long-term 95% reliability. Both completion tokens arrived together, so decode speed is unknown. No intentional pause was added. Original smoke pause fields excluded bookkeeping; unchanged UTC timestamps give corrected follow-up gaps 0.956684 and 0.238897 seconds, recorded separately in pause-correction.json. The subsequent scheduler correction has a deterministic regression test.

## Preserved interrupted attempts

| Cohort | Status | Evidence and reason |
|---|---|---|
| First plain-nohup launch | failed before a request | PID 2910696 disappeared; no output or intent; full-01.log |
| full, PID 2910864 | failed, 8 response records | GLM mandatory-reasoning HTTP400; a proxy estimate rejected the long-word 250k prompt as 325235 tokens; USD0.1438775125; full/terminal.json and stop-reconciliation.json |
| full-v2, PID 2935739 | current request outstanding, STOP pending | Defaults left optional Qwen reasoning at xhigh; request exceeds socket idle timeout despite no terminal response; separate wall-clock deadline added for v3 |

The v1 in-flight response was recorded before terminating the owned PID; all eight intents have response records. The intermediate live /key counter lagged at USD0.09313342; the higher known reported total is retained. Interrupted cohorts are instrument evidence and costs, not pooled as provider-availability or cache-admission results. No failed provider was removed from the corrected matrix.

## CPU validation and evidence limits

Seventeen CPU tests pass, exit 0 (.temp/hybrid-d-r15b-261103/tests-10.log). Coverage includes exact root/region and provider pinning, wrong/missing serving identity, budget refusal before POST, known prior-spend floors, usage bounds/missing costs, key redaction, a planted leaking-log negative, mandatory reasoning compatibility, deterministic growth, mock-runner smoke, unexpected billed-cost stop, network destinations, key permissions, deadline cleanup, event journaling and pause accounting. The task-specific supervisor's positive cost-floor and negative unresolved-cost/intent fixtures pass (supervisor-tests-01.log).

| Mutant | Property narrowed | Named failing test | Outcome |
|---|---|---|---|
| Restore pause reference after bookkeeping | Gap includes previous-response bookkeeping | test_pause_includes_post_response_bookkeeping_regression | Killed: 300 != 400; exit 1, pause-mutant-01.log |

Python compilation, scoped whitespace/ASCII checks and git diff checks passed. ty and editorconfig-checker are unavailable and were not downloaded; hosted type checking was not run. No C++ source changed, so a native-model build is inapplicable. No surface table or independent provider review was supplied. Local mocks establish instrument behavior only. Daily OpenRouter uptime samples overlap and do not prove long-term reliability. Local main freshness is unverified because the network rule excludes GitHub fetches.

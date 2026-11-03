# API cache report - Amendment 1 (hybrid-d-r15c-261103)

Status at launch: queued, with no Amendment 1 API calls or measured provider results yet. The detached runner PID/SID is 3066191, independently observed reparented to PID1229. It leaves v1 PID3001509, its source and its runtime outputs untouched. It waits for prior processes and the same exclusive key lock, reconciles previous charges, then measures under the combined USD45 ceiling. There is no second USD45 allowance. Queued ledger values are initialization placeholders, not a measured prior account balance.

Live log: .temp/hybrid-d-r15c-261103/run-01.log. Live state: run/state.json. Amendment data/report: run/calls.csv, call-*.json, stream-*.jsonl, summary.json and API-CACHE-REPORT.md under that same task directory. A terminal.json and done/failed line show termination. Detailed skipped_or_pending entries identify every missing provider/depth/cohort and reason. notification-terminal.txt contains the eventual terminal summary.

## Preregistration

Signed protocol and manifest commit 3d84bf168bbbe0be7c21087a46eeb711701a7fa7 preceded all amendment requests. Tested implementation and clarifications are signed in 5c749c65a. Both signatures verified successfully before launch. Branch: m3-api-cache-test. No push.

Reasons, declared before data: the owner's analyst requested pauses matched to the deep Claude Code profile (median about8s, p90 about35s), and the million-token section is the strongest hybrid case. These are supplied motivation/profile facts. The experiment measures whether cold 1M TTFT is minutes; it does not insert the expected answer.

The frozen candidate list has 42 endpoints: Qwen Alibaba/Novita/Phala and 39 GLM endpoints with at least1M declared context and cache-read pricing. Depths are 524288 and1000000. Core sessions have ten growing calls, with six8s gaps and three35s gaps. Independent one-shot probes target each nominal depth and report miss TTFT only with explicitly zero cached input and observed tokens. Separate long-pause sessions use60/300/900/3600s. All prefixes are fresh and deterministic; actual depth/pause and timeout limits are retained. Expected complete matrix:840 core calls,84 probes and420 long-pause calls, before budget skips.

Admission retains context, declared caching, measured hits and >=95% availability. Core coverage requires20 attempts, known cache counters, a hit at both depths, >=95% own core availability and >=95% sampled daily uptime at three timestamps. Overlapping daily windows do not prove long-term reliability. Long-pause results and cold-probe availability/latency remain separate.

Ranking is frozen at measurement startup from completed v1 evidence: cheapest and median actually admitted Qwen1M providers first, then GLM, remaining Qwen last. If admitted evidence is unavailable, use the preregistered frozen-price fallback and label the choices provisional. No admission is invented. The analyst's USD5/provider figure is an estimate; each call reserves its actual conservative price bound and budget refusals remain explicit skips.

Core effective pricing uses1065 new input tokens,512 output tokens and equal weights over512k/1M, because empirical depth weights were not supplied. Reports compare every endpoint with the supplied logs'98.1% cached input and approximately3200 uncached tokens/call. No winner or price claim is made from queued or incomplete data.

## CPU and local validation

All33 CPU tests pass, exit0 (.temp/hybrid-d-r15c-261103/tests-03.log):17 unchanged v1 tests and16 amendment tests. Loopback controls cover the whole miniature amendment, priority/admission labels, combined-budget refusal, held custody with no key read, prior-charge reconciliation, provider refusal, explicit miss/hit/unknown TTFT classification, growing prefixes, clock bookkeeping, zero-duration deadline crossing, separate core/long summaries, missing counters, budget skips with snapshot provenance, missing/unexpected costs, source identity and a planted-key log negative. These establish instrument behavior, not real provider or tokenizer performance. The end-to-end fixture records32 mock calls with separate20 core/10 long/2 probe results.

CLI readiness, Python compilation/AST, ASCII/whitespace and staged diff checks passed, exit0. No C++ source changed; a native build is inapplicable. ty/editorconfig-checker remain unavailable and were not downloaded. No surface table or independent review was supplied; no killed-mutant coverage is claimed for this amendment.

The original engine SHA256 remains5921dde6be676011da7122e2d810c0b074fe3f91b4e251026131b8eee09d362e, matching both v1 and the amendment manifest. This prevents changing the active v1 source to accommodate the amendment. On-disk TMPDIR and PYTHONDONTWRITEBYTECODE=1 are set. Disk free was above28GB at launch; the runner checks the floor before each paid call.

## Requested notification

The summary to agents1_hybrid_orch in #hybrid/topic prototype-go is saved at .temp/hybrid-d-r15c-261103/notification-summary.txt. It was not transmitted: the repeated network rule permits HTTPS only to openrouter.ai. A narrowly scoped Matrix exception was requested and has not been granted. The runner saves its terminal summary locally as well.

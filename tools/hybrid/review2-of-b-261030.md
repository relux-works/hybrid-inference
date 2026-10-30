# Round-2 cross-review of producer B

Reviewer: producer A, run `hybrid-a-r5-261030`, 2026-10-30 UTC. Subject: `hybrid-client` at signed head `d7e40062a8e86766d4d33a05b1a89ab9bf200720` (signature verification exit 0). A runtime/source head: `e8e15025a07d79e00da3b3843a2ef0aaab9162b7`. Specification: `/home/agent/lanes/hybrid/docs/prototype-spec.md`, v0.3.4, sections 4a, 5, 6, 7 and 11. First review: [review-of-b-261030.md](review-of-b-261030.md). B's response: `tools/hybrid-proxy/fixes-r1-261030.md` in B's checkout.

**Verdict: changes required.** R1-R4 are fixed within the tested contracts, and the delayed-202 coverage concern is resolved by an existing delayed-application fixture. The supplied CPU end-to-end evidence follows section 11's two-reference method and passes. However, B has not integrated A's codec limits: its shared reader still rejects headers larger than 65,536 bytes and does not enforce the per-range budget. This remains a section-6 interoperability and admission defect at the reviewed head.

This is an internal prototype review, with no upstream submission. Applied the repository code-review skill's scope, security, general, API, server and test checks to the requested fix scope. No builds, new input generators, downloads, pushes or B-workbench writes. Existing binaries were used read only. Runtime outputs are under A's `.temp/hybrid-a-r5-261030/` (abbreviated `R` below). All file:line references in the item table refer to the pinned B tree unless marked A.

## Item verdicts

| Item | Verdict | Existing test and command | Change / relevant location |
|---|---|---|---|
| R1: recurrent count 2 aborts before rollback | **Fixed.** The two-live-sequence control accepts a valid append, rejects the original count-2 witness plus 14 neighboring malformed records normally, preserves both native sequences byte for byte, then accepts a valid append again. Real HTTP count-2 and endpoint negatives also preserve both slots. | `test_recurrent_cell_count_two_rejected_without_abort`, C2 and C2A, exit 0; HTTP `native-recurrent-two-cells`, C3, exit 0. Logs `two-sequence-b-01.log`, `two-sequence-a-01.log`, `e2e-01.log`. | `src/llama-context.cpp:3535` preflights the entire record without scheduling writes; `src/llama-memory-recurrent.cpp:866` requires exactly one append cell; `:1022` bounds ordinary single-sequence restoration. |
| R2: rejected incompatible resync changes decoder but leaves old journal | **Fixed for the reported failure.** Incompatible prefix/model metadata gets 409 before any decoder call; inconsistent snapshot and SQLite commit failures restore the old destination and preserve the durable row. Compatible-ahead recovery still succeeds. These state-machine tests use synthetic decoder state. | `HTTPTests.test_resync_incompatible_prefix_is_atomic`, `test_resync_incompatible_model_is_atomic`, `test_resync_snapshot_and_journal_failures_roll_back`, with positive `test_409_and_snapshot_resync`; C1, exit 0. | `tools/hybrid-proxy/proxy.py:130` and `:136` validate identity/prefix before restore; `:151` saves destination; `:154` verifies restored state; `:158` commits journal before changing the session; `:162` rolls back on failure. Tests at `tools/hybrid-proxy/test_proxy.py:217`, `:235`, `:250`. |
| R3: recovery GET failures escape offline fallback | **Fixed.** Both divergence and 409 recovery branches complete locally on connection loss, HTTP 503 and truncated HTTP bodies, then replay the exact journal. Genuine incompatible-history 409 stays explicit. | `HTTPTests.test_recovery_get_outage_completes_locally_and_replays` (six subcases), `TransferTests.test_truncated_http_body_is_transport_error`, and compatible-ahead positive control; C1, exit 0. | `tools/hybrid-proxy/proxy.py:48` converts interrupted HTTP bodies to transport errors; `:171` implements common recovery fallback; `:276` and `:282` use it from both exception handlers. |
| R4: neighbor preservation checks metadata only | **Fixed.** Fourteen invalid native HTTP imports return 400 with a message and leave destination and neighbor slot files byte-identical. Successful append and one-token append also preserve neighbor bytes. | `test_decode.py:run`, especially the candidate loop at `:98`; C3, exit 0. Existing golden control generates exactly 64 tokens and proves reuse with `cache_n=n1`, `prompt_n=1`. | `tools/hybrid-proxy/test_decode.py:57` captures neighbor native bytes; `:110` compares them after each rejection; `:118` compares after valid append; `:189` checks one-token append neighbors. |
| Delayed-202 concern | **Fixed as a coverage concern.** Three stale replay calls enqueue identical unapplied shadows; a critical extend drains FIFO, the first shadow applies, stale duplicates return 409, the critical path still offloads, and final remote/local tokens and acknowledged n agree exactly. This does not claim real asynchronous scheduler timing coverage. | `HTTPTests.test_delayed_202_shadows_critical_drain_without_token_loss`; C1, exit 0. | `tools/hybrid-proxy/test_proxy.py:317` adds the delayed fixture and assertions; existing `tools/hybrid-proxy/proxy.py:93` treats 202 as queued, not applied, and `:117` advances acknowledgement only from GET. No production deduplication change is claimed. |
| Section 11: exact and numerical end-to-end references | **Confirmed for the supplied CPU cases.** All four actual HTTP conversations pass 64/64 same-partition equality and both teacher-forced thresholds. | `test_decode.py:run_proxy` invokes `llama-hybrid-check-distributions`; C3, exit 0. Golden 512/2048/8192 numerical cases: C8. | `tools/hybrid-proxy/test_decode.py:263` captures actual prompt IDs/deltas; `:308` invokes checker for all four; `tools/hybrid/check-distributions.cpp:85` creates references, `:108` locally prefills split/whole paths, `:114` requires observed greedy IDs on both exact paths, `:118` compares full-vocabulary distributions, `:124` teacher-forces the same token into every context, `:132` gates KL/top1. |
| Section 6: 1 MiB reader cap and per-range header budget | **Not fixed / not integrated.** B retains the old 64 KiB limit and has no range-aware header-size check. See C11 and the open finding below. A's codec controls pass, but are not represented as B runtime coverage. | Existing A `test_header_budget_int32_max_16384`, `reader-range-header-budget`, `reader-absolute-header-budget`: C5, exit 0. Pinned A/B source comparison: C11, exit 1 because the files differ. No B runtime large-header rejection/acceptance test was added. | `tools/hybrid/hybrid-state.cpp:88` old writer cap; `:99` old reader cap; `:105` position checks followed by no range-budget validation. `tools/server/server-context.cpp:3244` uses this reader; `:6845` retains old body-plus-header allowance. |
| Section 5: 429 retryability/local fallback | **Confirmed.** Critical offload HTTP 429 produces a local completion, clears offload, keeps the full journal and replays it exactly after reconnect. Recovery also classifies 429 as retryable; background replay retains pending work. | Reused A `ReviewFailures.test_429_critical_falls_back_locally_and_replays`, C9, exit 0. Existing A `test_slow_readers_do_not_starve_critical` and `test_stream_budget_lifetime`, C4S, exit 0, establish actual node 429 admission and unchanged state. These are separate node/proxy controls, not a combined live overload-through-proxy run. | `tools/hybrid-proxy/proxy.py:177` recovery status list; `:277` primary fallback status list; `:185` background replay keeps retryable failures pending. No immediate retry or Retry-After sleep is required for the local-fallback choice. |

## Open finding: codec contract was not integrated

Severity: **major; blocks acceptance of the reviewed head against v0.3.4 section 6**. The allowed header budget is `min(1048576, 4096 + 11 * (n1 - n0))`, with valid range checks first. A's existing codec control constructs a valid 16,384-token header of 180,611 bytes with maximum body-size metadata, round-trips it, and rejects both range-budget overflow and absolute-cap overflow. B's reader rejects every header above 65,536 bytes before parsing it. For short ranges it also lacks the required tighter padding limit, even when a padded header fits under 65,536 bytes.

B's passing short-prompt/golden runs do not cover this boundary. `parse_json`'s separate 1 MiB text cap at `hybrid-state.cpp:17` does not fix the earlier binary-envelope cap at `:99`. B's fix report does not claim to have imported this codec change, and the pinned files confirm it is absent.

Required rework: integrate the already implemented shared codec budget from A (`tools/hybrid/hybrid-state.cpp` and `.h`, in A's signed `7349f2a152` scope), align the decode-route envelope allowance at `tools/server/server-context.cpp:6845`, and reuse A's existing maximum-range, range-budget and absolute-cap controls against B's reader. Do not change the agreed wire contract. The positive large-header and negative over-budget controls must both qualify B's implementation before this item closes.

## Remaining recovery bound

An existing reviewer observation control, C12, also reproduces a narrower residual: when compatible metadata is followed by an inconsistent snapshot and the decoder connection then fails during rollback, `recover` returns normally with the wrong synthetic slot, old durable tokens and the slot still marked verified. The rollback's `OSError` at `tools/hybrid-proxy/proxy.py:164` is swallowed by the generic fallback at `:174`, and the cleanup at `:169` removes the backup file. The control exits 0 because it asserts this observation; **it is not passing evidence of atomic recovery**.

The original incompatible-metadata R2 witness is fixed, as the item table states. This compound failure remains an open recovery concern: distinguish a decoder rollback failure from prefill unavailability and retain a recoverable or explicitly invalidated destination. The control does not execute the subsequent real completion, so no wrong completed response or irrecoverable journal loss is claimed. Native rollback across a decoder outage remains unqualified.

## Section 11 evidence

The checker reconstructs three independent CPU contexts: actual captured delta import plus the held token, token-only local prefill with the same partition plus the held token, and whole-prompt token-only prefill. Neither local reference receives imported state. At every one of the 64 continuation positions, imported and split greedy IDs must equal the observed HTTP continuation before that token is fed to all contexts. The whole-prompt comparison uses stable full-vocabulary softmax, double accumulation for `KL(imported || whole)`, and raw-logit top1. EOS masking is confined to greedy token selection where `ignore_eos` requests it; it does not mask the KL vocabulary.

Fresh C3 results reproduce B's report:

| Conversation (both clients independently) | Same partition | Mean KL, nats/token | Whole-prompt top1 | First whole-prompt free-running divergence |
|---|---|---|---|---|
| Counting | 64/64 | 8.67437e-15 | 64/64 | none within 64 |
| Primary colors | 64/64 | 0.00203662 | 63/64 (98.4375%) | zero-based index 38 / token 39 |

Both numerical thresholds pass: mean KL <=0.01 and top1 >=95%. Free-running whole-prompt divergence is reported as informational, as required. C6 also passes append/golden 64/64 at depths 512, 2048 and 8192. C8 reports golden mean KL 3.2963e-16, 7.69261e-7 and 2.01502e-7 respectively, with 64/64 top1 at each depth. Detailed fresh numerical output is in `R/real-e2e/distribution-check.log`; whole-prompt outputs and split diagnostics remain in `comparison-*.json`, `partition-*.json` and `greedy-equality.json`.

This confirms the method and supplied cases, not all section-11 acceptance coverage: 50 prompts at 4k/32k/128k, CUDA/Metal, production concurrency and performance remain unqualified. The actual HTTP mapping run covers two clients with two conversations each plus streaming/offline recovery. The synthetic suite and A prefill check separately cover 3 clients x 4 sessions, without proving per-session native equivalence at that concurrency.

## Commands and observed results

Run from A's repository root. These shell variables make the commands below exact without copying long paths into every table row:

```sh
A=/home/agent/Werkstatt/hybrid-a/llama.cpp
B=/home/agent/Werkstatt/hybrid-b/llama.cpp
M=/home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf
G=/home/agent/Werkstatt/hybrid-models/golden
R="$A/.temp/hybrid-a-r5-261030"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$R/tmp"
```

| ID | Exact command (stdout/stderr saved separately under R) | Exit / result | Log |
|---|---|---|---|
| C1 | `python3 -m unittest discover -s "$B/tools/hybrid-proxy" -p 'test_*.py' -v` | 0; 19 tests | `proxy-suite-01.log` |
| C2 | `python3 tools/hybrid/test_recurrent_append.py --library "$B/build/bin/libllama.so" --malformed` | 0; positive, 15 malformed cases, positive again; both sequences preserved | `two-sequence-b-01.log` |
| C2A | `python3 tools/hybrid/test_recurrent_append.py --library build/bin/libllama.so --malformed` | 0; same cases on A | `two-sequence-a-01.log` |
| C3 | `python3 "$B/tools/hybrid-proxy/test_decode.py" --binary "$B/build/bin/llama-server" --prefill-binary "$A/build/bin/llama-hybrid-prefill" --model "$M" --golden "$G/prompt-512" --artifacts "$R/real-e2e"` | 0; native HTTP plus full real proxy/numerical run | `e2e-01.log` |
| C4 | `python3 tools/hybrid/test_prefill.py --binary "$A/build/bin/llama-hybrid-prefill" --model "$M" --artifacts "$R/prefill-http" --port 18295` | 0; error atomicity, authentication, isolation, interleaved sessions | `prefill-errors-a-01.log` |
| C4S | `python3 tools/hybrid/test_prefill.py --binary "$A/build/bin/llama-hybrid-prefill" --model "$M" --artifacts "$R/slow-readers" --port 18296 --regression slow-readers` | 0; response admission/lifetime/429 controls | `admission-a-01.log` |
| C5 | `build/bin/llama-hybrid-check --codec` | 0; codec limits, envelope negatives and scheduler controls | `codec-a-02.log` |
| C6 | `"$B/build/bin/llama-hybrid-check-append" "$M" "$G/prompt-512" "$G/prompt-2048" "$G/prompt-8192"` | 0; native/envelope negatives and 64/64 at all depths | `append-b-01.log` |
| C7 | `"$B/build/bin/llama-hybrid-check-guards" "$M" "$G/prompt-512"` | 0; non-SWA positive, three SWA rejection/preservation controls | `guards-b-01.log` |
| C8 | `"$B/build/bin/llama-hybrid-check-distributions" "$M" "$G/prompt-512.tokens.json" "$G/prompt-2048.tokens.json" "$G/prompt-8192.tokens.json"` | 0; 64/64 at all three depths, KL/top1 gates pass | `distributions-b-01.log` |
| C9 | `PYTHONPATH="$A/.temp/hybrid-a-r4-261030" python3 -m unittest probe_failures.ReviewFailures.test_429_critical_falls_back_locally_and_replays -v` | 0; reused existing reviewer HTTP control | `proxy-429-01.log` |
| C10 | `"$B/build/bin/llama-hybrid-check-distributions" "$M" .temp/hybrid-a-r4-261030/wrong-continuation.tokens.json` | **1, expected red**: `imported greedy differs from observed continuation` | `wrong-continuation-red-01.log` |
| C11 | `git diff --no-index -- tools/hybrid/hybrid-state.cpp "$B/tools/hybrid/hybrid-state.cpp"` | **1**, source differences, not a passing B codec test | `codec-source-diff-01.log` |
| C12 | `PYTHONPATH="$A/.temp/hybrid-a-r4-261030" python3 -m unittest probe_failures.ReviewFailures.test_failed_resync_rollback_is_swallowed_as_prefill_outage -v` | 0; observation of inconsistent synthetic state after failed rollback, not a passing atomicity property | `rollback-outage-observation-01.log` |

C9 and C12 import the prior attempt's exported B Python fixtures. Their bytes, including proxy, policy, store, stub and test suite, were verified identical to the pinned current B files before reuse. C10 reuses the already existing wrong-continuation fixture unchanged; it introduces no new generator or plant.

| Existing control | What it narrows the gate to | Named rejecting test/check and observed result |
|---|---|---|
| Wrong first observed continuation token, C10 | Exact observed 64-token continuation must be checked, rather than merely comparing distributions | `check-distributions.cpp:run`, imported-greedy assertion at `:115`; exit 1 with the intended mismatch. Unmodified captured conversations pass C3. |
| Built-in identity, boundary, KL-only and top1-only controls | Each numerical threshold must independently reject its violation, with valid identity and boundary neighbors | `check-distributions.cpp:controls`, called by C3's checker; internal invalid cases rejected and positive cases accepted, checker exit 0. |

No new mutants were created. These controls do not claim to detect every possible defect.

## Identity, process cleanup and finite limits

`R/evidence-manifest.json` records observation time, source heads, SHA-256 of the specification, small model, executed binaries/libraries, source agreement with B's build evidence, and every owned server PID. B's recorded build-source commit `e8d86f693d22e723c8c8104da94129c01eb550b6` differs from the reviewed head only in four documentation files; all 13 recorded implementation/test/configuration source hashes match. Existing checker hashes match B's recorded build evidence; server and library hashes also match B's `binary-hashes-01.txt`. No rebuilt or hypothetical binary is substituted for these observations.

The native HTTP harness stops both servers by recorded PID in `finally`; A's HTTP harnesses do the same. All five recorded server PIDs were absent from `/proc` at the manifest observation. Fixture HTTP servers shut down in their existing teardown. B's checkout remains clean.

R2's rollback tests exercise the proxy state machine with a synthetic decoder, including real SQLite aborts; they do not establish native snapshot rollback under a decoder outage or process-crash atomicity across two services. R4's native checks are real slot-file comparisons. Delayed 202 is a deterministic fixture, while A's current prefill node applies shadows synchronously. SWA guard coverage is allocated-cache rejection, not SWA-model inference. Anthropic Messages, long-depth/backend equivalence and production load remain outside the supplied CPU review evidence. No attack-surface table was supplied; no exhaustive coverage is claimed. Codec integration blocks acceptance; the compound rollback-outage concern above also remains open with its stated bound.

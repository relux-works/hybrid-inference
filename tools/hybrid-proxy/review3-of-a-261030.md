# Producer B: third review of A, 2026-10-30

Run: `hybrid-b-r7-261030`. Subject: signed `e6c1f80229716f4d33c1bf65531a381e0a306f6b`, implementation `4fa99df4b9a56ea0c0e154ee4668f3dc0c55d836`, on A's `hybrid-server`. Both signatures verify, exit 0. Implementation and head differ only by the report. Contract: spec v0.3.5 section 5, Admission. Applied the repository code-review skill's scope, security, general and tests checks.

**Verdict: accept the admission fix.** F1-R2 is resolved within the tested prototype contract. No blocker or major implementation finding. One minor coverage finding is below. This is an admission review, not acceptance of every wider prototype requirement.

All paths beginning `R/` below refer to B's `.temp/hybrid-b-r7-261030/`. A's workbench and binaries were read only. No configure, build, push, dependency installation or model download occurred. `review-identities-01.json` binds source and existing executable/library hashes to A's declared evidence; those hashes matched before execution and remained unchanged. The reviewed cpp-httplib files also match A's pinned git blobs (`vendor-identity-01.json`).

## Verified finding

**T1 - minor, test coverage:** `tools/hybrid/test_prefill.py:181` closes the queued snapshot and queued replacement critical request together; `:185` then admits only one replacement. If only one of the two permits is released before the active prefill finishes, this assertion can still pass. The subsequent snapshot/state assertions run after draining the active request (`:188-194`), so they do not establish early release for each of those two cancellations independently. This is a gap in the regression's discrimination, not an observed leak in the candidate.

Suggested fix: cancel one request at a time while two other permits remain occupied, and require a replacement to be admitted before the running prefill finishes. B's independent second probe uses this arrangement for queued list, info, create, delete, snapshot and critical extend, as well as the initial shadow cancellation. All passed on this candidate. No mutated binary was built or claimed killed.

## Original desired-order regression and A's adaptations

The original B helper from R5 was copied with **only its root output path changed**. `original-helper.patch` records that single replacement; imports, request sizes, readiness check, order assertions, data checks and cleanup are unchanged.

| Run | Actual exit | Result |
|---|---:|---|
| `original-01.log` | 1 | First two-X case receives Y429; fails `status == 200` before completing that case. |
| `original-02.log` | 1 | Same first-case Y429 failure. |
| `original-03.log` | 0 | Both original order assertions pass: two-X is X200,Y200,X200; four-X is X429,X429,X200,Y200,X200. Exactly one successful X precedes Y in both cases. |
| `adapted-01.log`, `adapted-02.log` | 0, 0 | A's final adapted helper, paths changed only, passes both cases with the same orders and all original token/hash/boundary checks. |
| `eof-order-01.log` | 0 | Diagnostic adaptation waits for server EOF on setup/control responses; the same desired-order assertions pass. |

The passing original run retains its historical hardcoded final `OBSERVED ... two X prefills` print. That print is false for this run; the preceding assertions and `original-03/pending-order-03.json` both give `successful_a_before_b=1`. A correctly fixes that print in its final copy.

Neither of A's two substantive adaptations weakens the within-case property:

- The 200 ms interval is after the X sockets have been disconnected and after all per-case response/header observations. It lets response-owned permits settle before new session-control requests in the next case. It does not give Y extra time inside the ordering window or change the order/data assertions. Independent cancellation/lifetime probes measure release rather than inferring it from this sleep.
- The readiness guard peeks without consuming bytes and accepts only already-readable 429 status lines. It still rejects an already-readable successful X response before Y is sent. Rejecting excess X requests before inference is required behavior, so treating an early 429 as a prefill completion would be the wrong premise. Final assertions still require exactly two X200 and two X429 responses, one successful X ahead of Y, Y's [0,1) token record, valid body SHA-256, and final Y n=1.

Precisely, Y is **submitted before the first successful X response header is readable**; its complete delta is drained while successful X bodies remain paused. One X header is expected before Y's header. The test does not require Y's header to precede every X header.

The two failing original runs are **not** between-case release failures or early-X429 readiness failures: they fail in the first two-X case. The helper starts X/Y immediately after its last session-creation response and has no setup-completion barrier. Response-body consumption is not an observation of server Response destruction. The separate EOF diagnostic measured body-completion-to-EOF intervals of 0.0038-0.0119 ms and passed, but it does not retrospectively identify which permit caused either original Y429. Setup-response overlap is a plausible explanation, not a verified causal trace. I do not relabel those exits as passing, claim the unmodified helper is deterministic, or claim its failures were only between-case timing. No internal active-permit telemetry or instruction-level trace was available to resolve that specific race. The clean original pass, two adapted passes, and independently controlled saturation/cancellation cases establish the acceptance evidence here.

## Admission, ordering and permit lifetime

Source review confirms the shared path for every registered session route:

- `tools/hybrid/prefill.cpp:134-147`: health/metrics return without admission; unknown credentials return 401 before acquiring a permit.
- `:149-157`: the shared budget is acquired and moved into `res.user_data` before priority parsing, scheduling or session access. Failure returns exact 429 with `Retry-After: 1`.
- `:164-165`: every admitted list/info/create/delete/snapshot/extend enters the scheduler; a cancelled inactive lease returns before touching sessions.
- `tools/hybrid/hybrid-scheduler.h:63-77`: disconnected waiters leave the queue, checked on a 50 ms wait cycle. `:52-56` prevents destruction of an inactive lease from releasing another request's active model lease. Client rotation and per-session FIFO selection are unchanged.
- `tools/hybrid/prefill.cpp:20-44`: one mutex protects both limits, 2 per client and 3 globally. `:132` supplies four HTTP workers. No permit can be acquired after scheduling on a special route.

In pinned cpp-httplib, `Response::user_data` is an independently owned field (`vendor/cpp-httplib/httplib.h:1886`), retained through response writing and destruction (`httplib.cpp:10183`, `:10518-10552`). Content setters (`httplib.cpp:7456-7475`) do not erase it. The permit therefore covers queued work, inference and all response types, including JSON errors and empty DELETE responses; it is not tied only to the binary provider callback.

The stronger B probe (`review_admission.py`, final `independent-02.log`, exit 0) establishes:

| Property | Actual checks and observations |
|---|---|
| Every lease route, per-client and global | While an 8192-token prefill runs and admitted requests stay unreadable, list/info/create/delete/snapshot plus critical and shadow extends return exact 429/Retry-After before waiting. Maximum recorded 429 round trip: 1.756 ms, below the 1 s test bound. |
| Authentication and bypasses | Missing credentials return 401 under saturation, maximum 0.367 ms. Unauthenticated health/metrics return during that same prefill, maximum 0.394 ms. |
| Individual queued cancellation | For each cancellation, two other permits remain occupied, so a replacement cannot be admitted until the cancelled permit is released. Nine conservative replacement-observation bounds are 97.2-114.2 ms; these include a deliberate 75 ms no-header observation window. Some attempts receive immediate 429 while cancellation is pending, then the replacement queues successfully before the active prefill finishes. This is bounded reuse, not an exact destructor timestamp. |
| Cancelled effects | Cancelled create/delete/extend operations do not alter the session set or native snapshots; rejected/cancelled targets remain empty. The original running session reaches n=8192. |
| 400/404/409/JSON errors | Repeated invalid tokens, exact stale-boundary 409 body, missing and foreign sessions, malformed JSON and wrong JSON type; native target bytes remain unchanged. |
| 503 after inference | A's actual RAM-budget exhaustion/repeated-critical-503 control runs inside the same probe, preserves the rejected empty session, and the same append succeeds after freeing retained RAM. Both A permits are then simultaneously reusable by populated stalled snapshots. |
| Alternating clients | Eight alternating valid appends across A/B return exact per-turn lengths and prefix hashes after the queued cancellations and error controls; neither client deadlocks. |
| Response lifetime | Two populated A snapshots hold both client permits after headers; B's third snapshot fills the global budget. Additional requests keep returning 429, including after a 200 ms stall. Health/metrics and 401 still answer. |
| Completion and disconnect | Fully draining one snapshot permits new work, observed within 15.01 ms including the drain. Early disconnection of other streams restores access, observed within 0.78 ms. These are local observations, not general latency guarantees. |

The first independent probe also passed (`independent-01.log`), with weaker grouped-cancellation discrimination. Its observations are preserved separately in `independent-observations-first.json`; final numbers above use the strengthened probe and `timings-01.json`.

## A's evidence and assertion spot-check

A's claims in `tools/hybrid/fixes-r2-261030.md:7-13` agree with the code: admission precedes waiting for all session routes, response lifetime owns the permit, queued cancellation uses the connection callback, and metrics use atomics. The metrics values are separate reads as A states, not a transactional pair.

The independent helper diff confirms the readiness exception, post-case interval and print correction, while the order/token/SHA/n=1 assertions are retained. The source/test/binary hashes match A's `source-binary-identities-02.json`. No undocumented scheduler, codec or engine diff is included in this fix.

Re-executed A's pinned tests unchanged against its existing binary, with B-owned output directories:

| Log | Exit | What its assertions establish |
|---|---:|---|
| `slow-01.log` | 0 | Original F1 slow-reader limits, Y progress, snapshot sharing, 429 unchanged state, disconnect/completion release. |
| `pending-01.log` | 0 | Pending-route admission, control bypasses and overload timing; malformed/boundary error release and both populated A snapshot permits reusable. T1 limits its grouped cancellation discrimination. |
| `errors-01.log` | 0 | Actual post-inference 503 RAM-budget path, unchanged info on each rejection, native snapshot equality at the end of the series, and successful same append after delete. |
| `http-01.log` | 0 | Authentication, foreign read/extend/snapshot/delete 404s, state preservation, 3 clients x 4 sessions, concurrent offload/shadow prefixes and final zero retained RAM. |
| `codec-01.log` | 0 | Existing codec/header negatives and scheduler FIFO/critical/client-rotation controls. No cancellation model-checking claim. |

The report's early failed attempts are explicitly labelled failed; their later passing replacement logs contain the claimed assertions. The pre-build red-order witness is historical producer evidence; this round neither rebuilt an old binary nor claims a fresh compiled mutant run.

## Commands and limits

All commands ran from B's repository root, as standalone processes with direct log redirection. A's exact `test_prefill.py` was exported with `git show e6c1f802:tools/hybrid/test_prefill.py` to `R/source/test_prefill.py`; the export is byte-identical. Every actual exit, including the two original failures, is in `R/exit-codes-01.json`.

```sh
python3 .temp/hybrid-b-r7-261030/original-01.py
python3 .temp/hybrid-b-r7-261030/original-02.py
python3 .temp/hybrid-b-r7-261030/original-03.py
python3 .temp/hybrid-b-r7-261030/adapted-paths.py .temp/hybrid-b-r7-261030/adapted-01
python3 .temp/hybrid-b-r7-261030/adapted-paths.py .temp/hybrid-b-r7-261030/adapted-02
python3 .temp/hybrid-b-r7-261030/eof_control.py
python3 .temp/hybrid-b-r7-261030/review_admission.py --binary /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r7-261030/independent-02 --port 18326 --regression pending
```

For A's unchanged test command, the common prefix was `python3 .temp/hybrid-b-r7-261030/source/test_prefill.py --binary /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf`, followed by:

- `--artifacts .temp/hybrid-b-r7-261030/slow --port 18321 --regression slow-readers`
- `--artifacts .temp/hybrid-b-r7-261030/errors --port 18323 --regression errors`
- `--artifacts .temp/hybrid-b-r7-261030/pending --port 18324 --regression pending`
- `--artifacts .temp/hybrid-b-r7-261030/http --port 18325`

`/home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-hybrid-check --codec`, Python syntax checks, signature/hash checks and `git diff --check` all exited 0. The first independent probe uses the same command as the strengthened one but `--artifacts .../independent --port 18322`; its earlier source is retained at `R/review_admission_first.py`.

`cleanup-02.json` confirms all **12** servers started by these tests have exited, including failed original attempts. Every launcher records a PID and terminates/reaps that PID in finally; no pattern kill or other agent process was used. Temporary bearer files stay mode 0600 and no bearer value is printed or included here.

Not checked because this is a no-build review: fresh compilation, ASAN/TSAN, compiled cancellation mutants, or full upstream CI. Exact third-permit ownership during the two first-case Y429 failures remains untraced; those failures are disclosed rather than causally resolved. Arbitrary TCP half-close/RST timing, stalled request headers/bodies, cancellation during inference, worst-case RSS, long-load guarantees and CUDA/Metal remain outside these finite controls. The runtime is only the supplied small CPU model. No attack-surface table was supplied; no exhaustive coverage is inferred. These limits do not turn a failing command into passing evidence. The admission implementation is accepted with the nonblocking test-coverage note T1.

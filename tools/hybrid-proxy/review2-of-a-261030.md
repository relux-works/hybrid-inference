# Producer B: second cross-review of producer A

Run: hybrid-b-r5-261030. Reviewed signed head **e8e15025a07d79e00da3b3843a2ef0aaab9162b7** (G), branch hybrid-server. Executable implementation is **7349f2a152d16c77e86e34aceaa686cb9dc938af**, with shared-core cherry-pick **a7fe591d8f711e5a9f8181425207279d73d61cfc** (G). The later head changes documentation only. Locations below refer to that reviewed A head, not B's checkout.

**Verdict: changes required.** F2 is fixed, and no R1 validation was lost in the shared-core conflict resolution. F1 now enforces the outgoing-response ceilings and passes the original slow-reader scenario, but its claimed reserved admission worker can still be consumed by handlers waiting for the compute scheduler. This is an A-side fix review, not acceptance of the entire hybrid prototype.

## Verified finding

### F1-R2 - major: scheduler waiters bypass the HTTP admission reserve

- **Location:** `tools/hybrid/prefill.cpp:142` calls the blocking scheduler before response admission at `prefill.cpp:198` and `prefill.cpp:232`; `tools/hybrid/hybrid-scheduler.h:62` waits inside an HTTP worker. A's regression at `tools/hybrid/test_prefill.py:58` sends the next X request only after the previous response headers, so it does not exercise concurrent pending handlers.
- **Concrete failure:** send four valid client-X critical extends concurrently, each for 1024 tokens in its own session, and pause the successful response readers. Send client-Y's one-token critical request before any X response becomes readable. One X handler prefills while three X handlers occupy the other HTTP workers inside `scheduler.enter`, without yet acquiring response permits. Y therefore cannot reach the client-fair queue. The observed response/commit order is X200, X200, X429, X429, Y200. Both X prefills commit before Y is admitted. With only two concurrent X requests, the positive control gives X200, Y200, X200: the existing scheduler rotates correctly once Y can enter it.
- **Why the response counters are insufficient:** the two-per-client and three-global counters are real for acquired response leases. They do not count pending handlers parked before acquisition, so they cannot guarantee the specification's free admission worker. This counterexample uses complete authenticated requests, the supplied small model, and ordinary critical extends; it does not depend on slow request headers, keepalive, invalid tokens or a timing benchmark.
- **Reproduction and control:** `pending-probe-02.log` and `pending-order-02.json` verify the bad ordering with a two-X positive control. Both cases assert that Y is sent while no X socket is readable. The same live regression asserting the intended one-X-before-Y order exits **1** with `admission reserve lost: Y waited behind two X prefills despite being sent before the first X response` (`pending-regression-red-01.log`, `pending-order-03.json`). Y's full delta has the correct boundary, token ID and SHA-256, and its final n is exactly one; the failure is admission ordering, not a malformed response or setup error.
- **Suggested fix:** make admission cover pending binary handlers before they block in `scheduler.enter`, or give admission/control handling capacity independent of workers waiting on compute. Retain the shared snapshot/delta budget, namespace isolation, pre-inference 429 and RAII release guarantees. Add the concurrent-before-first-response regression, keeping the two-X positive control so normal compute serialization is not misdiagnosed as unfair admission.

## What passes

### F1 response ceilings and lifetime

The independent round-1 scenario now receives X statuses `[200,200,429,429]`; Y obtains and validates its full critical delta while the two successful X sockets remain stalled. A third held response from another client reaches the global limit; further delta and snapshot attempts return 429 without changing the target prefix, while health and foreign-session 404 operations remain reachable.

`admission-probe-02.log`, exit **0**, additionally verifies:

- snapshots and deltas share the same two-per-client quota;
- stalled requests asking for keepalive receive `Connection: close`;
- four idle keepalive requests do not consume persistent workers, and two pipelined health requests receive only one response before the socket closes;
- disconnect and normal completion return permits;
- HEAD on this node returns 404 and does not consume a permit (HEAD is not a promised API operation);
- repeated invalid-token 400, stale-boundary 409 and foreign-snapshot 404 errors leave the empty victim snapshot unchanged and do not lose either client permit;
- after filling the actual retained-payload RAM budget, four critical extends each return the post-inference 503 rather than leaking leases and degrading to 429; the victim remains unchanged. Deleting one retained state admits the exact same critical request with 200 and n=1.

The first review helper expected HEAD to use the GET admission path and failed with actual 404 (`admission-probe-01.log`, exit 1). Its expectation was corrected; that helper failure is not a product finding. The sequential response-pressure repair is verified even though the concurrent pending-worker repair remains incomplete.

### F2 header budget

The current codec computes `min(1048576, 4096 + 11*(n1-n0))` after bounding n0/n1 (`hybrid-state.cpp:80-86`). The reader bounds the absolute declared header length before slicing/parsing (`hybrid-state.cpp:114`), then validates positions and the range budget (`hybrid-state.cpp:121-122`). The subtraction and multiplication occur only after the positions are bounded, so no arithmetic overflow was identified here.

The node checks tokens/context at `prefill.cpp:214`, then generates a conservative header using the largest allowed body-length metadata at `prefill.cpp:239`, before context restore at 241 and inference at 242. The same writer is used for the final envelope. Pre-inference ordering is established by source review; the snapshot assertions alone would not prove that no discarded computation occurred.

Freshly running A's exact copied test against its existing node accepted **16384** six-digit token IDs with a **115101-byte** header, exact tokens, prefix hash, body SHA-256 and native snapshot equality. **16385** tokens returned 400 and preserved the empty session info and snapshot byte for byte (`max-delta-01.log`, exit **0**). No large response payload was written into A's workbench.

The current compiled codec/scheduler suite also exits **0** (`codec-01.log`): maximum-int32 16384-token headers round-trip, conservative header size is **180611 bytes**, and reader/writer per-range and absolute reader bounds reject with the intended errors. This separates the codec's int32 serialization bound from the model's vocabulary bound.

### Shared-core conflict resolution

Compared A's a7fe591d with B's 4d7cad76, not just the commit messages. Five core files plus the recurrent regression script are byte-identical. The complete difference in `src/llama-kv-cache.cpp` is A's pre-existing stream-count/one-populated-stream policy versus B's earlier portable-restore/SWA policy; see `core-resolution-01.diff` and `core-resolution-01.json`.

All R1 additions remain: the validation flag is forwarded to KV metadata parsing; validation neither allocates cache cells nor clears failed state; host validation does not schedule tensor writes; recurrent cardinality, IDs, layout and lengths are checked before mutation. B's earlier portable restore and SWA capability guards were not part of the requested R1 cherry-pick and remain intentionally absent from A. This is not a lost R1 hunk.

A's unchanged existing library freshly passes the original two-cell regression and 14 adjacent malformed cases, with byte-identical target and neighbor sequence states and valid append before/after (`recurrent-01.log`, exit **0**). No abort was observed.

## Spot-check of A's evidence

- The reported candidate-to-head diff contains only README/fix-report changes. The current binary and library SHA-256 identities were recorded before and after the review and are unchanged (`binary-identities-01.json`, `binary-identities-02.json`). Producer build logs bind them to 7349f2a152; this review did not independently rebuild them.
- The max-range test actually checks exact token IDs, count, prefix, SHA-256 and native snapshot bytes, and checks unchanged empty state after the 16385 rejection. The compiled codec test separately checks the XXH3/SHA framing and its corruption controls; the HTTP max-range test itself asserts only SHA-256, not XXH3.
- A's slow-reader test asserts both per-client and global response quotas, shared snapshot admission, state preservation and permit reuse. Its two printed PASS labels are groups in one function. It waits for each X response header before issuing the next X request, which is precisely why the new pending-worker failure is missed.
- The reported old-binary F1 witness is a real timeout at Y's critical extend, not a setup error. A correctly labels the old 16384-token 180-second timeout inconclusive rather than a demonstrated header rejection.
- A's append log contains 64/64 passes at 512/2048/8192 and its malformed-state rejections. The independent local-prefill checker path remains present; these full-depth equivalence runs were spot-checked from evidence rather than rerun in this review. No teacher-forced distribution or cross-backend acceptance claim is inferred from that report.

## Commands and custody

All fresh evidence is under `.temp/hybrid-b-r5-261030/` in B's tree. Python probes use generated mode-0600 credential files, ephemeral ports and recorded subprocess PIDs. They stop and wait for their own nodes in `finally`; no pattern kill is used. The copied A HTTP test writes its artifacts only under B's directory. `cleanup-01.json` and a live process scan confirm no owned prefill node remains. `exit-codes-01.json` records the actual command outcomes; `review-identities-01.json` records the source/specification hashes and observation time.

```sh
/home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-hybrid-check --codec
python3 tools/hybrid/test_recurrent_append.py --library /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/libllama.so --malformed
python3 .temp/hybrid-b-r5-261030/probe_admission.py
python3 .temp/hybrid-b-r5-261030/probe_pending_queue.py
python3 .temp/hybrid-b-r5-261030/regression_pending_queue.py
python3 .temp/hybrid-b-r5-261030/source/test_prefill.py --binary /home/agent/Werkstatt/hybrid-a/llama.cpp/build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-b-r5-261030/reproduce-max --port 18095 --regression max-delta
```

The live admission-reserve regression is intentionally red on the reviewed candidate (exit 1). The observation probe exits 0 when it verifies the counterexample; that is not a passing product gate. Other fresh validations listed above exit 0. No configure, build, benchmark, push, A-workbench edit or golden rewrite was performed.

## What could not be checked

No repaired admission implementation exists in this review, and no build/configure was allowed, so the suggested fix has not been validated. Compiled parser/scheduler mutants, sanitizer instrumentation, arbitrary cancellation races, slow request headers, maximum simultaneous outgoing-memory residency, other architectures/backends, and the wider long-prompt matrix remain unchecked. Retained-payload accounting and response-count bounds are not a process-RSS guarantee. The available source and runtime probes did not establish another codec or shared-core defect; no unverified suspicion is promoted to a finding.

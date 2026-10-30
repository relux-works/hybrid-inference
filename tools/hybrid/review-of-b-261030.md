# Cross-review of producer B

Reviewer: producer A, run hybrid-a-r2-261030, 2026-10-30 UTC. Subject: signed commit `98aaee97686988d0e13ce94ac2ffcbd20bf9b475`, branch hybrid-client, against upstream `e0f142d5ddc1461ce721644d92af33db23545105`. Signature verified G. Fetched into `refs/remotes/b/hybrid-client`; B's checkout and branch were not modified. This verdict is bound to that exact subject, not a later B head.

Verdict: **changes required before prototype acceptance**. One blocker, two major findings and one minor coverage finding are verified below. No verified cross-client leakage was found in the reviewed proxy mapping and credential routing.

Reviewed against specification v0.3.2, sections 4a-9 and 11, plus the #hybrid/prototype-go decisions adopting one native section, the one-token boundary, token IDs in the header and HTTP/1.1. Applied the repository code-review skill's scope, security, API, server, design, tests and general checks. This is local review evidence, with no upstream submission or hosted review.

## Verified findings

### R1 - blocker: malformed recurrent row count aborts the process before rollback

Locations: `src/llama-memory-recurrent.cpp:1010`, `:1023`, `:1042`, `:1051`; caller `src/llama-context.cpp:3546`; exposed through `tools/server/server-context.cpp:3264`.

The single-sequence recurrent reader checks only `cell_count <= size`, where size grows with parallel slots. With two slots, a malformed state declaring two recurrent cells passes that check. `ubatch_reserve(cell_count, 1)` and `find_slot` allocate one recurrent cell for one sequence, but the assertions afterward assume a block of cell_count cells. The duplicate-row payload hits `GGML_ASSERT(cells[head + cell_count - 1].pos == ubatch.pos[cell_count - 1])` and calls abort. A C++ catch cannot handle this, so the transactional append backup cannot restore the prefix, and every session in the server process is lost.

Concrete witness: start an isolated unified two-sequence context with the provided Qwen3.5 model; load the 512-token golden prefix into sequences 0 and 1. A valid 128-token append succeeds and leaves sequence 1 unchanged. Restore sequence 0, change the native recurrent count from 1 to 2, and insert a second eight-byte position/sequence-count metadata row. Attention metadata, positions and declared n0/n1 remain valid. The malformed call aborts at recurrent-reader line 1051. The envelope reader treats the native section as opaque; updating its byte count and recomputing its checksums permits this payload to reach the same append call over HTTP.

Verification: reproduced with A's existing libllama, without compiling. Positive control exited 0; malformed control exited **134 (SIGABRT)**, a failed clean-rejection property, not a passing atomicity test. The exact append, hybrid and recurrent source paths are identical between A's tested binary source and B's pinned head. B differs in KV stream admission/SWA guards, neither of which rejects this unified non-SWA witness. This is not a claim that B's server binary was executed. The recurrent reader predates this patch, but the new HTTP native-import path exposes the failure.

Suggested fix: reject an invalid recurrent cardinality for a single-sequence append before removing sequence metadata or allocating a batch. A nonempty delta must carry exactly one current recurrent state. Keep whole-context restoration separate; empty ordinary snapshots need their own allowed case. Add two-live-sequence tests for malformed counts, with a valid neighbor control, destination and neighbor native-byte comparisons, and a process-survival assertion. A singleton test alone masks this defect because count 2 already exceeds size 1.

Evidence: `.temp/hybrid-a-r2-261030/repro_native.py`, `native-positive-01.log`, `native-recurrent-count-01.log`. Reproduce from A's repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 .temp/hybrid-a-r2-261030/repro_native.py
PYTHONDONTWRITEBYTECODE=1 python3 .temp/hybrid-a-r2-261030/repro_native.py --malformed
```

The malformed control sets RLIMIT_CORE to zero and operates only on its own in-memory contexts and read-only golden/model inputs. Its nonzero result is expected evidence of the defect and must remain reported as failing.

### R2 - major: rejected snapshot resync leaves the decoder and durable journal inconsistent

Locations: `tools/hybrid-proxy/proxy.py:141`, `:145`, `:147`, `:149`.

`resync` restores the remote snapshot before checking whether its tokens match the incoming conversation. If that compatibility check fails, it raises HTTP 409 before saving the new token journal. The decoder now contains the incompatible remote history while SQLite still contains the previous local history. Thus a rejected recovery operation has destructive effects instead of preserving the old state or committing one consistent replacement. This is a resync failure, distinct from the native import failure in R1.

Concrete witness: complete and journal a valid conversation, then make its remote session contain `[8, 8, 8]`, incompatible with the local conversation. Call resync for an extension of the original prompt. The call returns 409, the decoder's slot changes to `[8, 8, 8]`, and the durable session retains the original tokens. The original prefix cannot be resumed consistently without further recovery or a new session.

Verification: reproduced through B's pinned proxy using its existing synthetic HTTP fixtures. The fixture demonstrates the real proxy state-machine ordering; real native snapshot-byte preservation was not tested. A's independent source trace confirms the actual restore request precedes the failing check. B's existing successful ahead-snapshot test is a nearby positive control, not a rejection/rollback test.

Suggested fix: validate snapshot identity and its token prefix against the incoming conversation before restoring it, or back up the local slot and roll it back on any failed compatibility check. Commit the decoder/journal transition consistently. Add a mismatched-snapshot test checking both durable tokens and slot state after the 409; once native tests can run, compare saved slot bytes too.

Evidence: `ReviewControls.test_failed_resync_mutates_slot`, `.temp/hybrid-a-r2-261030/repro_proxy.py`, `proxy-review-controls-02.log`.

### R3 - major: recovery-time network failures escape the offline fallback

Locations: `tools/hybrid-proxy/proxy.py:245`, `:250`, `:256`, `:258`, `:338`.

The resync calls execute inside except handlers rather than inside the original protected try. An OSError from resync in the ValueError branch is therefore not caught by the later OSError branch. It reaches the HTTP handler, which returns 400 instead of continuing with local prefill/decode. The 409 branch similarly catches OSError locally but lets retryable upstream HTTP failures escape. This violates specification section 7.4's local continuation when the server is unreachable.

Concrete witness: the remote server is one token ahead of the durable journal, and that ahead prefix is valid for the incoming conversation. The first GET succeeds and naturally triggers the existing divergence/resync path. Inject a connection failure only on the recovery GET. The user request returns HTTP 400 with no local completion. Remove that single fault and the identical ahead-prefix request recovers and completes successfully.

Verification: reproduced through the actual proxy HTTP route. Only the recovery transport was faulted; sync and resync logic were not mocked. The same input then passed with the network available. The existing offline test faults the server before initial sync and does not cover this nested recovery branch.

Suggested fix: apply the offline policy around the entire synchronization/recovery operation, including calls made from exception handlers. Catch retryable transport/upstream failures and proceed to local prefill while preserving the durable journal. Keep genuine model or incompatible-history errors explicit. Add separate recovery-GET connection-loss and retryable-HTTP-error tests, paired with the successful ahead-prefix control.

Evidence: `ReviewControls.test_resync_network_failure_blocks_local_completion`, `.temp/hybrid-a-r2-261030/repro_proxy.py`, `proxy-review-controls-02.log`.

### R4 - minor: the neighbor-slot preservation assertion covers metadata, not native state

Locations: `tools/hybrid-proxy/test_decode.py:90`, `:91`, `:92`, `:99`; related claim `tools/hybrid-proxy/RESULTS.md:33`.

The native HTTP probe compares the neighbor's `/info` response after each rejected import, but saves and compares native bytes only for slot 0. A corruption of slot 1's K/V or recurrent tensors that preserves its token list and positions would pass the neighbor check. Slot 1 is then erased before the decode comparison, which removes the remaining opportunity to detect that corruption. The claimed unchanged-neighbor check is therefore bounded to metadata.

Verification: traced the assertions; no actual neighbor-tensor corruption is claimed. This probe remains unrun in B's evidence.

Suggested fix: save slot 1 before and after each invalid import and compare its native bytes, as for slot 0. Optionally continue decoding that preserved neighbor against an independent reference before erasing it. Until then, describe the assertion as unchanged neighbor metadata.

## Suspicion and additional coverage gap, not a verified token-loss finding

S1: `tools/hybrid-proxy/proxy.py:115` and `:116`, versus the synchronous shadow stub at `tools/hybrid-proxy/stub_server.py:130` and `:135`. Delaying shadow application while returning 202 lets repeated sync calls enqueue the identical n0/suffix twice. The reviewer observed two queued copies; when applied through the strict fixture, the first succeeded and the second returned 409. No duplicate token application or token loss was observed, so this is not promoted to a correctness finding. A genuinely asynchronous queue, retries before application, and a subsequent critical extend draining that queue remain unverified. Add that delayed-application integration case before claiming complete queue/replay coverage.

## Checks performed and limits

All source and test references above use B's pinned commit. Review inputs and reproduction code were exported into A's `.temp/hybrid-a-r2-261030/subject/` and task-scoped files; nothing was executed from or written into B's workbench. The existing A shared library source identity is `1c496c523d`; the recurrent file blob is `5fd644645cb206f4f82817583a52075aa1e32b4c` in both trees. `shared-native-path-01.diff` confirms no append/hybrid/recurrent source differences. File hashes, runtime versions and subject identity are recorded in `evidence-manifest.json`.

| Command/control | Result | Evidence under .temp/hybrid-a-r2-261030 |
|---|---|---|
| Pinned B `git verify-commit` | exit 0, signature G | verify-b-01.log |
| `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s .temp/hybrid-a-r2-261030/subject/tools/hybrid-proxy -p 'test_*.py' -v` | exit 0, 13 tests | proxy-tests-01.log |
| `python3 .../repro_native.py` | exit 0, valid append and unchanged neighbor | native-positive-01.log |
| `python3 .../repro_native.py --malformed` | exit 134, clean-rejection property fails at native assertion | native-recurrent-count-01.log |
| `python3 .../repro_proxy.py` | exit 0, three reviewer observation controls; two reproduced failure scenarios, one async-queue observation | proxy-review-controls-02.log |

Temporary SQLite and HTTP fixture files were confined using task-scoped TMPDIR, with bytecode generation disabled. Servers/threads started by the Python fixtures were shut down and joined. The isolated native process used the existing small model, CPU, two threads and two sequence slots; it started no shared service. No benchmark, dependency installation, build, configure or push was performed. Exact disk bytes were checked before source export/test writes. A later disk reclaim did not override the user's no-build scope.

The native repro is not a test of B's new KV portability/SWA guards, nor of its real HTTP server. Because no B server/checker was built or available in the admitted review workflow, the following remain unrun: B's real import/info HTTP routes, source/destination slot-count portability, native neighbor-byte preservation, the SWA rejection path, the full real proxy/prefill/decode integration, and CUDA/Metal equivalence. The supplied Qwen35 model also does not exercise a non-NONE SWA cache. The 13 passing tests accurately label their engine state as synthetic. B's explicit build-hold limitations are retained; they are not converted into passing runtime evidence.

No attack-surface table was supplied for this review. The findings and finite checks above do not claim exhaustive coverage. Anthropic support, streaming interruption journal recovery, and restart behavior with in-flight generation were not qualified here. The review adds no production code and requests no change to the shared wire contract.

# Round-3 cross-review of producer B

Reviewer: producer A, run `hybrid-a-r7-261030`, 2026-10-30 UTC. Reviewed B `hybrid-client` at `f893d961405769435cca926d35c67979d230cdf1`; A source head `e6c1f80229716f4d33c1bf65531a381e0a306f6b`. Contract: `/home/agent/lanes/hybrid/docs/prototype-spec.md` v0.3.5, sections 5, 6, 7 and 11. Prior findings: [review2-of-b-261030.md](review2-of-b-261030.md). B response: `tools/hybrid-proxy/fixes-r2-261030.md`.

**Verdict: accept for the requested second fix round.** Codec convergence, C12 recovery handling, recurrent-only rollback, startup hashing and listener admission qualify within the bounds below. No blocking implementation finding remains in this review scope. There is one nonblocking correction to B's startup-listening claim. This verdict does not grant the full section-11 prototype acceptance or qualify untested backends/depths.

Applied `skills/code-review/SKILL.md`: scope, security, general, API, server and test checks. This is an internal prototype review. No builds, configuration, downloads, new performance measurements, pushes or B-workbench edits. All runtime outputs are in A's `.temp/hybrid-a-r7-261030/` (`R`). File:line references below refer to the pinned B checkout unless marked A.

## Item table

| Requested item | Verdict and actual observation | Test / command | Relevant file:line |
|---|---|---|---|
| 1. Codec source convergence | **Fixed.** Both no-index diffs are empty, exit 0. Both files also equal A's `80013538` blobs. Reader and writer enforce `min(1048576, 4096 + 11 * range)`. | C1/C2; blob comparison in `report-inputs-01.log`. | `tools/hybrid/hybrid-state.h:16`; `tools/hybrid/hybrid-state.cpp:80`, `:97`, `:114`, `:121`, `:122`. |
| 1. Codec controls and slot preservation | **Fixed.** A's retained codec controls pass against B's linked reader: 16,384 INT32_MAX tokens round-trip, 180,611-byte maximum-metadata header; intended writer/range-reader/absolute-reader errors. Fresh native HTTP accepts the real 16,384-token record with a 115,101-byte header and 127,550,068-byte state, then compares saved native KV/recurrent bytes with A's source record. Both padded-header negatives return the intended 400 and leave the saved destination file byte-identical. | C3 `test_header_budget_int32_max_16384`, `reader-range-header-budget`, `reader-absolute-header-budget`; C4 `test_native_header_budget_16384`. | `tools/hybrid/check-codec.cpp:55`; `tools/hybrid-proxy/test_header_budget.py:112`, `:116`, `:122`, `:127`, `:129`, `:134`. |
| 1. Import-route envelope allowance | **Fixed.** Route and listener allow 512 MiB native body + 1 MiB JSON header + 52 framing bytes, totaling 537,919,540 bytes. Framing is 4 length bytes + 16 section checksum bytes + 32 body checksum bytes. | C4 crosses the previous frontend cap successfully and rejects cap+1. | `tools/server/server-context.cpp:6856`; `tools/server/server-http.cpp:220`; `tools/hybrid/hybrid-state.cpp:138`. |
| 2. C12 failed decoder rollback | **Fixed.** Distinct `RollbackError`/503 escapes generic offline recovery; verification is revoked and the backup retained. Reacquisition erases a dirty slot before reuse, including after store restart with a superficially compatible extension. A local completion then agrees with the durable tokens and removes the marker. Wrong state is not left verified. | C5, `HTTPTests.test_recovery_rollback_failure_retains_backup_and_resets_slot`, exit 0, including cleanup-outage subcases. C6 reruns A's original observation against current B modules: **exit 1, expected red**, because `recover` now raises `RollbackError` instead of returning normally. | `tools/hybrid-proxy/proxy.py:169`, `:175`, `:180`, `:191`, `:215`, `:219`, `:244`, `:316`; `tools/hybrid-proxy/test_proxy.py:303`, `:329`, `:333`, `:342`, `:347`. |
| 3. Core rollback and high flag bits | **Fixed.** PARTIAL_ONLY skips non-SWA attention serialization/restoration and retains recurrence. Failure discards deferred writes, restores recurrence, then removes positions `[n0, end)` without clearing the retained attention prefix. KV/hybrid append failures defer cleanup to this transaction. APPEND/VALIDATE use bits 30/31 with a public-flag overlap assertion. | C7: A's existing unified two-sequence positive append, 15 malformed cases and positive append again, all preserving neighbor bytes. C8: actual post-preflight capacity failure preserves both sequences; freeing the neighbor permits the same record. C9: late endpoint/recurrent failures preserve destination, then 64/64 append/golden equality at 512/2048/8192. C10: non-SWA positive and three SWA rejection controls. | `src/llama-context.cpp:3554`, `:3575`; `src/llama-kv-cache.cpp:2060`, `:2143`, `:2208`; `src/llama-memory-hybrid.cpp:146`, `:214`; `src/llama-memory.h:19`; `tools/hybrid/check-client.cpp:91`; `tools/hybrid/check-append.cpp:97`. |
| 3. Timing evidence | **Confirmed as recorded measurements, with provenance limit below.** Recomputed every median from B's 18 raw sample records; all equal the report. Measurement code times one successful 128-token append at fixed context 9216 and compares native bytes with source afterward. | Audit of `cost-{before,after}-*.log`, `cost-identities-01.json`, `cost-summary.json`; C11 audit log. No timing rerun. | `tools/hybrid/check-client.cpp:47`, `:66`, `:79`, `:87`; `tools/hybrid-proxy/fixes-r2-261030.md:43`. |
| 4. Model hash at startup | **Fixed.** The only server-context `file_hash` call is in model loading, before readiness; import uses the initialized value and errors if absent. Fresh C4 launch logs exactly one hash, 1141.54 ms, before the later listening announcement. **The socket itself starts before model loading**, so B's stronger wording needs the correction below. | C4 server log; C11 startup-order audit; source trace. Reload rehash is source-inspected, not exercised. | `tools/server/server-context.cpp:1215`, `:1634`, `:3252`; `tools/server/server.cpp:660`, `:674`, `:682`, `:697`. |
| 5. Previous 100 MiB listener ceiling | **Fixed.** Vendor cpp-httplib defaults to 104,857,600 payload bytes, rejecting a valid roughly 127.7 MB wire body before route validation. The conditional setter raises the frontend cap to the same finite envelope allowance as the route. It applies to the listener's routes when hybrid slot support is enabled, rather than only to import. | C4 `listener_budget`: cap+1 returns 413 with unchanged slot metadata; valid maximum record then succeeds. Header negatives additionally compare native files. | `vendor/cpp-httplib/httplib.h:142`, `:2433`; `tools/server/server-http.cpp:220`; `tools/hybrid-proxy/test_header_budget.py:16`. |
| 6. Claims against assertions | **Substantially supported; one nonblocking wording correction.** C12 checks retained backup, old durable row, verification revocation, erase and subsequent exact local tokens. Capacity test asserts both native sequences, not just positions. Header tests assert the rejection reason and native-file equality. Section-11 checker separately requires observed greedy IDs on imported and token-only same-partition references, teacher-forces the same continuation into all paths, and gates KL/top1. | C5/C8/C4 fresh; numerical checker/source and B's `real-e2e-02/distribution-check.log` inspected, not rerun. | `tools/hybrid-proxy/test_proxy.py:329`; `tools/hybrid/check-client.cpp:114`; `tools/hybrid-proxy/test_header_budget.py:122`; `tools/hybrid/check-distributions.cpp:105`, `:116`, `:118`, `:131`; `tools/hybrid-proxy/test_decode.py:308`. |

## Rollback reasoning and timing bounds

In non-SWA KV state write/read, PARTIAL_ONLY returns before serializing or modifying attention. Hybrid state operations still traverse the recurrent component, so the backup contains the native record framing plus recurrence. On rejected append, the old recurrent position is restored to below n0 before hybrid `seq_rm` checks recurrence; suffix removal then leaves that recurrent state intact and removes only appended attention cells. Clearing the entire attention sequence in either lower-level exception handler would defeat this design; `15745a35` removes both destructive append cleanups. A restore/removal failure raises an error instead of pretending the transaction recovered. SWA is separately rejected and is not qualified for this rollback method.

The fresh capacity witness matters because malformed preflight rejection alone does not exercise rollback after attention allocation begins. The fresh late-endpoint/recurrent controls exercise failures after valid-layout import work as well. These are bounded CPU state comparisons, not proof for every allocator/backend failure.

| n0 | Before median ms (range) | After median ms (range) | Before / after extra peak RSS bytes, median |
|---|---|---|---|
| 512 | 4.721 (4.529-4.979) | 4.349 (4.135-4.700) | 0 / 0 |
| 2048 | 11.671 (11.426-15.533) | 4.792 (4.457-6.826) | 33,611,776 / 0 |
| 8192 | 24.363 (24.255-24.758) | 4.006 (3.975-4.446) | 73,596,928 / 0 |

The old full snapshots are 23,556,724 / 33,620,596 / 73,876,084 bytes; PARTIAL_ONLY is 20,201,932 bytes at each depth. Raw samples and source support removal of the prefix-dependent attention copy in these cases. This does not prove constant total append cost or zero allocation: parsing/cache metadata and recurrent copies remain, and RSS can reuse allocator pages. Each point has only three processes. B discloses short control overlap during after samples. The before shared-library hash was **not captured at measurement time**, as `cost-identities-01.json` explicitly records; its provenance is baseline source plus build logs. I verified the recorded before/after source hashes and fixture hashes, and the current after library hash, but cannot retroactively certify the before executable identity without a rebuild, which this review forbids.

## Nonblocking report correction

At `tools/hybrid-proxy/fixes-r2-261030.md:16`, replace the assertion that hashing precedes listening with: hashing occurs during model initialization, before model readiness and request processing. The five recorded log comparisons only establish ordering against the later `server_on_listening` announcement. `tools/server/server.cpp:660` explicitly starts HTTP before `load_model`; `tools/server/server-http.cpp:536` launches `listen_after_bind`. Loading-time middleware returns 503 (`tools/server/server-http.cpp:312`). The requested removal of first-import hashing is implemented correctly; no listener lifecycle change is needed.

Additional assertion bounds: the cap+1 control compares slot metadata, whereas the two header-overflow controls compare native files; these are not interchangeable claims. Header negatives start from an empty destination. C12 uses synthetic decoder HTTP fixtures and reopens the SQLite store in the same process; it does not simulate process termination or a native decoder outage. Its final backup/state assertions follow the cleanup-failure subcase, while each cleanup subcase independently asserts the 503 exception. No native crash-atomicity claim follows.

## Commands and observed exits

Run from A's root, with these path abbreviations:

```sh
A=/home/agent/Werkstatt/hybrid-a/llama.cpp
B=/home/agent/Werkstatt/hybrid-b/llama.cpp
M=/home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf
G=/home/agent/Werkstatt/hybrid-models/golden
E="$B/.temp/hybrid-b-r6-261030"
R="$A/.temp/hybrid-a-r7-261030"
export PYTHONDONTWRITEBYTECODE=1
```

Every gate ran directly, without a pipeline or build. Log files below contain stdout/stderr; actual observed exits are also recorded in `R/exit-codes-01.json`.

| ID | Command | Exit and evidence |
|---|---|---|
| C1 | `git diff --no-index -- tools/hybrid/hybrid-state.cpp "$B/tools/hybrid/hybrid-state.cpp"` | 0, empty `codec-cpp-diff-01.log` |
| C2 | `git diff --no-index -- tools/hybrid/hybrid-state.h "$B/tools/hybrid/hybrid-state.h"` | 0, empty `codec-h-diff-01.log` |
| C3 | `"$B/build/bin/llama-hybrid-check-codec"` | 0, `codec-b-01.log` |
| C4 | `python3 "$B/tools/hybrid-proxy/test_header_budget.py" --binary "$B/build/bin/llama-server" --prefill-binary "$A/build/bin/llama-hybrid-prefill" --model "$M" --delta-fixture "$E/header-http-02/maximum.delta" --artifacts "$R/header-http"` | 0, `header-http-01.log`; one decoder PID recorded and reaped in finally |
| C5 | `python3 -m unittest discover -s "$B/tools/hybrid-proxy" -p test_proxy.py -v` | 0, 21 tests, `proxy-tests-01.log` |
| C6 | `python3 - < "$R/c12_observation.py"` | **1, expected red**, `c12-old-observation-red-01.log`; original inline stdin payload saved at this path |
| C7 | `python3 tools/hybrid/test_recurrent_append.py --library "$B/build/bin/libllama.so" --malformed` | 0, `two-sequence-b-01.log` |
| C8 | `"$B/build/bin/llama-hybrid-check-client" "$M" --capacity "$G/prompt-512"` | 0, `capacity-b-01.log` |
| C9 | `"$B/build/bin/llama-hybrid-check-append" "$M" "$G/prompt-512" "$G/prompt-2048" "$G/prompt-8192"` | 0, `append-b-01.log`; 54 named passing lines, 64/64 at all three depths |
| C10 | `"$B/build/bin/llama-hybrid-check-guards" "$M" "$G/prompt-512"` | 0, `guards-b-01.log` |
| C11 | `python3 "$R/audit_evidence.py"` | 0, `evidence-audit-02.log`, `report-inputs-01.log`, `evidence-manifest.json`; refreshed source/log audit, median recomputation and PID cleanup |

C6 preloads current B `test_proxy` and `proxy` before loading A's unchanged old observation, so the old helper's exported-subject path cannot silently select obsolete implementation modules. The log records their current absolute paths. Its failure is the intended `RollbackError` at recovery, not a fixture/setup failure. C5 supplies the positive assertions of retained backup and eventual correct state; C6 alone would not establish them.

## Narrowing controls and finite coverage

The following B recorded mutant evidence was inspected, not rebuilt or replayed. These are actual failing gates on violations; no surviving mutant is reported as killed.

| Mutant/control | What it narrows the gate to | Named failing test and result | Survival bound |
|---|---|---|---|
| Recurrent-only backup with old destructive KV/hybrid cleanup | Failures after preflight must preserve the attention prefix | `test_append_capacity_failure_preserves_prefix_and_neighbor`, `capacity-mutant-01.log`: exit 1, `capacity rejection changed prefix KV or recurrent bytes` | Killed in B's evidence; fresh corrected C8 passes. |
| Pre-fix proxy with C12 regression | Rollback loss must escape offline fallback | `test_recovery_rollback_failure_retains_backup_and_resets_slot`, `rollback-control-03.log`: exit 1, `RollbackError not raised` | Killed in B's evidence; fresh corrected C5 passes. |
| Changed first observed continuation token | Same-partition checks must enforce actual observed IDs | `check-distributions.cpp:run`, `continuation-control-red-01.log`: exit 1, `imported greedy differs from observed continuation` | Killed in B's evidence; numerical suite was not rerun here. |

`R/evidence-manifest.json` records actual UTC observation time, both source heads, model/spec hashes, 16 verified B source hashes, eight binary/library hashes, recomputed timing values, borrowed delta hash and owned process cleanup. All match B's supplied source/binary evidence at inspection. B's four supplied commits verify with the configured signing identity, exit 0 (`report-inputs-01.log`). The single fresh native server PID is absent from `/proc`; fixture servers shut down in their existing teardown. A and B had clean tracked/untracked status before this review; B remains untouched.

The previous codec/C12 blockers are closed within the stated contracts. Remaining unqualified section-11 work includes 50 prompts at 4k/32k/128k, calibrated CUDA/Metal thresholds, native numerical equivalence for the full 3-client x 4-session matrix, and shaped-network/trace performance. B's existing four-session numerical logs were checked against their assertions, not claimed as a fresh end-to-end run. SWA imports and Anthropic Messages remain unsupported. There is no supplied attack-surface table or landing property suite; no exhaustive coverage is inferred. These limits are preserved from B's report and are not waived by this fix-round acceptance.

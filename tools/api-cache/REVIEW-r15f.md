# r15f qualification and self-review

Scope: tools/api-cache only, based on signed b8efa3f9462b14ab0d01214d388cb8759b8586cf on m3-api-cache-r15e. The live old95671982b source checkout is unchanged. The owner explicitly authorized this restart and historical output-only repair. No push, PR, session link, external dependency or download. No independent Muse verdict is claimed; review is expected alongside the restarted run.

The local code-review skill's scope, security, approach, tests and general checks were applied. Costs stay decimal-checked and receipt-bound; partial progressive usage cannot settle a canceled request; byte limits are labelled conservatively rather than miscounting SSE chunks as tokens. The response context closes on cancellation after the triggering event is flushed. Wrong model/provider, measured-depth failure, unresolved charges and excessive debit retain their halts. Historical repair validates all proposed changes before writing and preserves original JSON, receipt and ledger. The shared request lease also covers the repair CLI. Source diffs are bounded to this module, its tests and documentation.

CPU suite command: PYTHONDONTWRITEBYTECODE=1 TMPDIR=<task>/tmp python3 -m unittest discover -s tools/api-cache -p 'test_openrouter*.py' -v. Final evidence: .temp/hybrid-d-r15f-261103/tests-05.log, exit0,91 tests including all77 earlier names. AST validation covers all9 Python modules, exit0. Existing r15e entry-point refusals, planted cost/price failures, pooled/per-phase label, seed/smoke exclusion and all charge/timeout controls remain green. No C++ or accelerator source changed; no C++ build is required for the standard-library scripts.

Earlier tests-02.log exited1 on a new mock response whose default truthy __exit__ swallowed the guard exception. The fixture was corrected to return false, as a real HTTP response does; tests-03/04/05 then passed89/90/91 tests. This was a control-fixture failure, not a provider observation. Every negative below ran in a disposable loopback fixture or an in-memory source mutant, without real key reads or paid requests.

| In-memory mutant | Property narrowed to | Named failing test | Observed result |
|---|---|---|---|
| reject-overcap | Acceptance of full-usage non-compliant output | test_over_cap_completion_is_accepted_with_flag_and_continues | exit1; acceptance assertion fails |
| reserve16 | Reservation includes1024 output tokens at live price | test_reservation_covers_1024_token_completion | exit1; expected USD0.0153368 vs mutated USD0.0128168 |
| disable-cancel | Actual interruption before the final usage event | test_cancel_guard_past_bound_preserves_unknown_usage_and_continues | exit1; guard-trigger assertion fails |
| accept-wrong-model | Explicit model mismatch still stops the next POST | test_named_negative_wrong_model_identity_halts | exit1; phase wrongly succeeds |

Evidence: .temp/hybrid-d-r15f-261103/mutation_control.py and mutant-*-02.log. These are expected-red processes with real exit1, not passing tests. All four were killed by the intended assertion; no survivors and no unrelated errors. Originals were never edited by the controls. The final suite ran against the unchanged candidate source.

Important bounds: no surface table was supplied. Loopback qualification is instrument evidence, not provider qualification. Client cancellation cannot prove the upstream has stopped billing hidden work; retained reservations and cumulative reconciliation remain mandatory. A byte-triggered partial call is censored and an availability failure, while complete over-cap HTTP200 usage is successful. Missing SSE model identity is recorded, with exact identity also checked by the read-only generation audit. Whole-window availability is request-sampled, not continuous uptime. Finite cutoffs and caps can leave explicit coverage gaps; no pause is compressed. These limits preserve the pre-registered analysis contract rather than excluding providers for low cache hits.

Launch and actual settlement/migration evidence will be recorded outside the source worktree in .temp/hybrid-d-r15f-261103/SUMMARY.md, with canonical paid outputs remaining in .temp/hybrid-d-r15d-261103/. Every new process PID must be observed separately. The new end observer repeats r15e generation-cost and price-drift checks at the extended workflow end without changing the paid runtime.

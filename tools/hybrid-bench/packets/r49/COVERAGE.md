# R7az source coverage and execution bounds

The review supplies surface rows R1-R8. Every changed source contract has a passing CPU control and a killed narrowing mutant below. Native/GPU/Metal execution is not implied by these controls. All previous reviewer tests retain their names; the full bench206 (7 optional native skips) and proxy141 suites pass.

| Surface | CPU controls / retained evidence | Narrowing mutant | Bound |
|---|---|---|---|
| R1 selection/prereg integrity | test_select_entrypoint_cannot_skip_publication; test_published_freeze_requires_live_event_matching_inventory; test_exclusion_message_and_api_availability_are_retained; immutable signed doc checks | publication-content; a2a-mixed-selectors | actual a2a event content/routing required; tests inject isolated event data, no owner population published |
| R2 identical work/native replay | test_recorded_output_mismatch_never_commits; unchanged r49b native-replay-03 and native-paired-02 source/test identities verified | replay-output-equality | Python commit boundary and replay request verified; native C++ unchanged, no fresh native build |
| R3 waits/background | test_checkpoint_nearest_wait_tie_and_final_only; test_capture_budget_and_actual_wait_both_bound_save; test_wait_lateness_measured_after_sleep; retained journal/background tests | wait-placement; save-budget | fixed wait selection and two independent bounds; target save latency awaits public run |
| R4 order/link/cost/admission | test_cooldown_delegates_to_real_kit_interfaces; test_four_arm_scheduler_loop_and_no_reroll_on_post_failure; existing accounting/bin tests | cooldown-shortened | adapter calls actual kit contract; test uses fakes, Darwin priority/memory execution remains open |
| R5 analysis/claim | test_analyze_entrypoint_preserves_n1_n3_estimates; test_claim_threshold_and_one_negative_session; retained cluster/no-gain tests | descriptive-pairs; claim-threshold | descriptive n1/n3, distinct main clusters and unchanged claim threshold |
| R6 public/API | test_public_api_format_keeps_base_and_recorded_messages; immutable pinned source/public artifacts | public-api-base-dropped | complete text/context available; provider replay/calls belong to dev-d |
| R7 transport qualification | test_real_path_both_sides_independence_and_fast_binding; test_tc_grid_missing_or_slow_path_refuses; test_independent_curl_ranges_and_auth_negative; test_real_trials_interleave_and_preserve_progress | reference-independence | rejects shared reference, slow/missing sides, short ranges and wrong path; actual tc/real-path throughput unmeasured |
| R8 whole-loop/test/packet | test_four_arm_scheduler_loop_and_no_reroll_on_post_failure; test_public_full_receipt_positive_missing_arm_and_failed_quality_negative; test_public_missing_post_qualification_refuses; checksums/signatures | public-failed-arm; progress-exclusive; standalone-import | all scheduler iterations and progress updates exercised with fake driver/hardware; actual full Mac public loop is a mandatory gate, not claimed green |

## Mutation controls

Each unmutated positive is green. All thirteen plants exit1 at the named test; none survives. The progress-exclusive mutant fails on the related FileExistsError inside the valid four-arm scheduler, reproducing the original progress-write failure. Other plants fail named assertions. An earlier harness attempt found two progress call sites rather than one and stopped before planting that mutant; two later admission checks stopped below28.5GiB and left source restored. Corrected successful controls are recorded separately.

| Mutant | Narrows the contract to | Named failing test | Exit |
|---|---|---|---|
| publication-content; a2a-mixed-selectors | accepts a changed inventory behind a real event | test_session_study.StudyTests.test_published_freeze_requires_live_event_matching_inventory | 1 |
| wait-placement | chooses midpoint without an eligible following wait | test_session_study.StudyTests.test_checkpoint_nearest_wait_tie_and_final_only | 1 |
| save-budget | bounds the save only by the wait, dropping the declared budget | test_session_study.FullSeriesAdapterTests.test_capture_budget_and_actual_wait_both_bound_save | 1 |
| descriptive-pairs | retains counts but drops public/favourable pairs | test_session_study.EntryPointTests.test_analyze_entrypoint_preserves_n1_n3_estimates | 1 |
| public-failed-arm | accepts a failed rehearsal arm | test_session_study.StudyTests.test_public_full_receipt_positive_missing_arm_and_failed_quality_negative | 1 |
| reference-independence | accepts a shared reference server path | test_session_study.StudyTests.test_real_path_both_sides_independence_and_fast_binding | 1 |
| claim-threshold | claims gain for a median below ten percent | test_session_study.StudyTests.test_claim_threshold_and_one_negative_session | 1 |
| progress-exclusive | reintroduces exclusive-create progress writes, stopping on the second arm | test_session_study.FullSeriesAdapterTests.test_four_arm_scheduler_loop_and_no_reroll_on_post_failure | 1 |
| replay-output-equality | accepts divergent decoder output into the recorded journal | test_session_study.UnchangedContractTests.test_recorded_output_mismatch_never_commits | 1 |
| cooldown-shortened | shortens the actual kit cooldown to one interval | test_session_protocol.AdmissionAdapterTests.test_cooldown_delegates_to_real_kit_interfaces | 1 |
| public-api-base-dropped | drops the API replay base while keeping native tokens | test_session_study.UnchangedContractTests.test_public_api_format_keeps_base_and_recorded_messages | 1 |

| standalone-import | requires an inherited in-process proxy import path, so the runbook command fails | test_session_study.StandaloneEntryTests.test_runbook_entrypoints_import_in_clean_process | 1 |

| a2a-mixed-selectors | Combines event lookup with topic history, which the real CLI refuses | test_session_study.A2ASelectorTests.test_a2a_event_read_never_combines_topic_or_last | 1 |

## Declared limits and scope

No source contract is silently excluded. Actual target execution in R2/R4/R7/R8 is outside this CPU-only run and remains an explicit pre-main gate, following the task's CPU-only and transport-choice clauses. Provider API execution in R6 is outside this lane per the dev-d ownership clause. These are unexecuted qualifications, not passing tests. The source rework remains inside tools/hybrid-bench; no server/proxy/codec/model code changes. Documentation, planning and test additions in that directory implement the requested amendment. Self-review used scope/security/general/tests/design checklists; native area was checked by exact unchanged source identity. No new dependency is installed. nginx, actual Darwin and a CAP_NET_ADMIN VM are operator-side requirements. Optional ty/editorconfig-checker are unavailable here; AST/compile/ASCII, CLI import/help, shell syntax and whitespace checks run instead, not reported as equivalent hosted type checking.

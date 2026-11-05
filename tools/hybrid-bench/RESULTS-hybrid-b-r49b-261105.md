# R49b preregistered replay preparation

Source base: signed r49 34ddd785de53ad5af38230efa5dea517ebcddb28, branch m3-growing-r46. The separate signed preregistration commit is dfffda35763edb17b832f758f586367999ac8978; its PREREG-E2E.md SHA256 is 56c0670d55e6366e532fe25c6df2814512a6338f9f52ad093207fc90b05ac1d2. Registration precedes owner selection and study inference. CPU fixture inference/tokenization after registration is development evidence, not a study series. No push, target deployment, GPU or Metal execution occurred.

## Outcome and source contracts

Main selection now freezes a complete timestamped owner population and samples five distinct qualifying histories by the preregistered hash rank, without portion-size/routing/performance filters. It retains the natural number of turns. Three disjoint large-output histories are a separately labelled favourable case. Missing eligible counts block rather than repeat/reroll a history. Zero-length recorded responses are preserved; their fully evaluated checkpoints use a fixed token0 probe extension only in the identical quality references, with the recorded/live journal untouched. This binding is CPU-fixture verified, not a new target-model quality result. The selector derives actual response-end-to-next-request waits, preserves long natural waits and publishes exclusions/distributions. Paired analysis uses five session clusters, fixed portion bins and the committed success test; empty hybrid arms remain valid no-gain outcomes. Cold routing also follows policy.

The native recorded-token replay mode performs normal one-position sampling/evaluation, then accepts the recorded output ID. Every arm emits the recorded length and has identical next-turn context. EOS cannot stop a recorded response early. Invalid mode/count/token, stop, grammar, probability-output or reasoning-budget controls refuse before mutation. Without the explicit mode, normal native sampling remains unchanged. The paper statement in PREREG-E2E.md explicitly measures system time, not agent task success.

Checkpoint capture consumes the existing recorded waiting interval, followed only by the remaining sleep. A capture overrun invalidates timing; it is never padded, subtracted from TTFT or used to give background extra time. Hashes/persistence/checkers are deferred after timing. The executor alternates independent local/hybrid and local/background pairs across distinct sessions and records every failed/not-attempted item. It delegates Mac cooldown/priority/owned-child memory monitoring to source-verified existing kit interfaces. The exact requested accepted r47 binding remains external; no CPU test claims Darwin admission.

A fresh eight-stream chosen-node probe occurs after cooldown immediately before each pair block, with an end probe retained for drift. Home/fast series reverse block order and keep identities/sets separate. TLS bytes and node synchronized compute/evaluation wall clocks remain measured cost inputs, with physical-interface/rental-bill boundaries explicit. The API replay/trace schema is documented for dev-d; no provider call or private transcript transmission occurred.

## Real data available

Owner cohort: zero sessions available for selection; actual owner depth range and selected tool-wait distribution are unknown. Do not mistake synthetic controls for a selected cohort. The retained aggregate stats file SHA256 is 8c5f1f2c44e344b78d32d0b4c216d0ebdf2ecdae3cde0f1a983cb329e488e2a7. Claude gap p50/p90=8.1/60.6 s (n=282262), all-agent gap=7.8/34.8 s (n=1511502). The producing code computes consecutive call timestamp differences, not verified tool-only or response-end-to-next-request times. Therefore these observations cannot verify the analyst's tool-time claim for the new cohort. owner-data-status.json preserves this distinction.

Public reproducibility reference: one fully rendered constructed session, 68 turns, 101711 -> 203809 tokens, from the pinned public llama.cpp tree e0f142d5ddc1461ce721644d92af33db23545105. Full source text, MIT license, file hashes, recorded acknowledgements and text/token context are delivered. Tokenization uses the existing local Qwen3.5-0.8B-Q4_0 model, with no inference for session construction. Its waits are deliberately constructed (8 s, every tenth 60 s): n=68, median8, p90=8, p99=60 s. They are not owner measurements. The actual pinned 27B tokenizer must render/freeze the target artifact before its study run; the builder is supplied. CPU reference gzip SHA256 7c42962e0d5b8185e07084a788a89dc041df964285191f61e07ea4a8eede85ab; source gzip SHA256 959fc89a5597c5bfcb2b8dfa7259450a00be06ae1dd994773ffa7a4360f68b50.

## Validation

Evidence root: /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r49b-261105. TMPDIR stays on disk, PYTHONDONTWRITEBYTECODE=1. Native builds were admitted after pgrep/load/disk checks, niced19/-j4 under the shared lane lock, using task-local pinned CMake3.31.6; no model download. Build/tool trees are removed after preserving hashes. Minimum observed free disk stayed above28.5GiB. No brief surface table was supplied. No requested contract is silently excluded; hardware/input prerequisites are listed below.

| Check | Real result |
|---|---|
| Full benchmark Python suite | 183 tests, exit0, seven optional native/SWA fixture skips; bench-suite-final4.log |
| Native replay contract | native-replay-03 exit0: non-greedy recorded IDs, four native decode positions, exact evaluated journal, EOS continuation, ten invalid-request variants without state mutation, unchanged default greedy control |
| Real native paired mechanics | native-paired-02 exit0: three arms, four controlled turns plus base, identical recorded journal hashes, real TLS/bf16 imports/background catch-up and stale-ACK injection; mechanics-only, not a new quality/target-model result |
| Native append controls | append-controls-02 exit0; old exact/lossy rejection contracts retained |
| Actual public construction | public-session-01 exit0, native CPU tokenizer only, complete publishable reference produced |
| Actual freeze/select/plan/analysis CLI controls | corrected freeze02/select01/series-plan01/analysis01 exit0; explicitly synthetic8-history input gives five main/three favourable, twenty planned attempts, missing-attempt analysis remains unqualified |
| Prior proxy suite | r49's141-test exit0 reused: proxy/transport/store source and tests are unchanged in r49b; no new proxy behavioral claim |
| Shaped throughput | Still unmeasured; r49 actual namespace attempt exit1 at uid_map. No unshaped substitute |

The first freeze CLI control used a mistyped trusted hash and correctly refused exit1 before output; its log is retained. The corrected command uses the computed input hash and exits0. This was a corrected invocation, not a claimed new intentional negative. Earlier successful/unsuccessful native/build logs are retained separately; current-source native evidence is the third replay and second paired run. Native paired mechanics deliberately omits large raw quality arrays; unchanged quality verifier behavior retains r49 evidence, not a fresh 27B qualification.

Static checks cover Python AST/ASCII, shell syntax, exact source/hash bindings, immutable preregistration blob, packet checksums and whitespace. Optional ty/editorconfig-checker remain unavailable from r49; no new claim that they ran. Existing growing/Qwen equality tests and exact controls remain present. New out-of-benchmark changes are limited to native server task/schema/decode support for one-token recorded replay; hybrid codecs and native import implementation are unchanged.

## Mutation controls

Each unmutated positive passes. Each mutant exits1 at the named contract test, not during setup; the main-filter mutant raises the wrongly narrowed eligibility refusal inside its positive control. No survivors are claimed.

| Mutant | Narrows the contract to | Named test that fails |
|---|---|---|
| Sleep a full wait after evidence | Gives background the capture time plus the full tool interval | test_evidence_uses_existing_tool_deadline_no_free_background_time |
| Use previous request start | Counts generation time as tool opportunity | test_measured_wait_distribution_and_chronology_negative |
| Require large portions in main pool | Promotes a favourable case to the main sample | test_main_accepts_all_small_portions_without_offload_filter |

## Before a GPU/Mac study

Obtain the complete ordered owner v2 export, trusted hash and extraction/tokenizer receipt; freeze it, select the preregistered sets and calculate their actual wait/depth/portion distributions and planning. Bind the exact accepted r47 source/ref and execute its real Darwin admission/cooldown interfaces. Obtain an authorized isolated tc/netem runner and verified >=80% results. Render/freeze the public target-tokenizer session, qualify the reviewed target build/import fit and quality, then measure the chosen node's home/fast link with this transport. No source delivery or CPU control makes these executed gates green. Allow roughly a day for input adaptation, target-tokenizer preparation and qualification after the missing inputs arrive; actual twenty-arm series planning may require several device-days and remains data-dependent.

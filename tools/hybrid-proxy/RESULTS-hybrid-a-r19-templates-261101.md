# Template journal prototype - hybrid-a-r19-261101

Local prototype, ready for review after the validation below. Base: `74f8f4c3afd50e9f0ce34b1581b5019d8ca234c7`, fetched from `https://github.com/relux-works/hybrid-inference m3-swa` and verified with signature G and `git verify-commit` exit 0. Existing commits are preserved. No push, PR, GPU use, model access or downloads. Code commit: 1d33648187c690f4077f9ceee937ba3fa2b74f0b.

## Changes and exactness

- Plain token pieces must reconstruct the rendered UTF-8 bytes exactly. Initial IDs must match the admitted IDs. With no added specials, both piece arrays must agree. Otherwise only one declared, single-ID BOS is accepted, followed by precisely the plain IDs/pieces, with no BOS spelling also present in the render. The BOS consumes a token position and zero rendered bytes. All layout positions retain the decoder's real n, including nonzero byte offsets within a piece.
- Live nonce probes derive each message header, reasoning span and tool wrapper. Family selection verifies a user/assistant/user render, closings and the generation header without history changes. Gemma/Glimmer additionally require matching BOS/EOS properties and delimiters in the live template source. Unsupported families and missing probes fail closed with 409. Each actual message probe must preserve its preceding context.
- Append verifies the exact text and token suffix, a single ending token and anchor overlap. The ordinary reply closings are template-family specific. Gemma tool responses share a model turn: a checked argument probe locates the response opener, not the earlier call terminator. Its response bytes remain in the appended tail and layout. Glimmer probes preserve resolved tool names when removing the preceding calls for a content probe.
- `user_boundaries` and the proxy rollback trial use the same live header derivation. The snapshot encloses header derivation in the existing rollback-protected operation; the original save/restore/rename failure regression remains unchanged.

## Real fixture evidence

The three files under `fixtures/templates/` are copied byte for byte from `/home/agent/lanes/hybrid/.temp/template-fixtures/`. Their decoder build is b12094-74f8f4c3a. Each contains 8 shapes x generation prompt on/off x thinking disabled/default = 32 renders, plus both token modes: 96 renders and 192 captured tokenizations overall. GGUF hashes are absent in the supplied captures; no weight identity is inferred.

| Model | Leading addition | Ordinary reply stop / captured ID | Tool behavior |
|---|---|---|---|
| Qwen3.8 27B | None | `<|im_end|>` / 248046 | Both shapes render; responses are user turns with `<tool_response>`; surrounding content is trimmed by the template. |
| Gemma 4 31B | `<bos>` / 2 | `<turn|>` / 106, also the captured EOS | Unmatched bench tool response is silently dropped. A matched call/response renders inside one model turn; its tool-call stop is `<|tool_response>` / 50. Historical reasoning is dropped when the template's last-user/preserve gate excludes it. |
| Muse Glimmer 30B | `<|begin_of_text|>` / 200000 | `<|eot|>` / 200008 | Both shapes render as named tool turns with `<tool_output>`; matched call IDs resolve to the function name. Reasoning uses a separate `assistant to=self` segment. |

The supplied Jinja sources establish these frame closings. `src/llama-vocab.cpp` explicitly adds `<|im_end|>`, `<turn|>`, `<|eot|>` and Gemma's `<|tool_response>` to the EOG set. Thus Glimmer's `<|eot|>` is an end-of-generation token even though `/props.eos_token` is `<|end_of_text|>`. Gemma's `<tool_call|>` / 49 is an earlier call terminator, not the source-selected generation stop. Actual sampled GPU endings still need observation; prompt captures do not establish generated output.

Fixture SHA-256 values:

- `gemma4-31b.json`: `d50e88d81aecd5b747dba2eecb381255e0376689d7d5008b844b9859a442b05f`
- `muse-glimmer-30b.json`: `81b24f6b30bc2bf00ba0566e16ae6d07124fce523514452b89e589e4a8817f3f`
- `qwen3.8-27b.json`: `46601101c9338a0e8a1ebbc705b8c4719c874e8f4c195fd96c43f19bfd61b472`

`FixtureTemplate` extends the stub with functional text renderers. All 96 renders compare byte for byte with captures before marker-probe evidence is accepted. Both captured tokenizations replay through `/tokenize`. Encoding a suffix/header uses captured aligned token spans, not an invented general tokenizer. The fixture harness is bounded to these text shapes and calls without root tool definitions. Arbitrary model tokenization, multimodal inputs and unobserved template options are outside its claim.

Layout succeeds in 92 cases; the four Gemma unmatched-tool variants explicitly reject. User boundaries cover every user in those supported shapes. Append covers 56 variants, including reasoning, matched calls/responses and two rounds, with T1 retained, ordinary/tool stop overlap and mapped positions checked. Six HTTP cases exercise local and forced-offload append for all three models. A separate byte-array/UTF-8 fixture checks a boundary within one token and a zero-byte BOS.

## Bench workload proposal (not implemented)

Qwen and Glimmer preserve the bench's tool message. Gemma omits it, so treating the bench as if it transferred those bytes would be false. The proxy rejects the missing message probe rather than journaling fabricated work.

The smallest valid tool workload change is to obtain a matching assistant tool call before submitting the same response blob: declare a no-op payload function, request/force that call, and use its actual returned call ID for the tool response. Keep the response payload and requested target depths, and record the call/wrapper overhead separately; adjust filler if target depth includes framing. The matched fixture demonstrates the required Gemma rendering. Do not graft `tool_calls` onto the already recorded A1: that edits immutable history. Simply inserting a synthetic assistant after A1 can also remove Gemma's end-of-turn boundary because consecutive assistant messages share a turn. This proposal needs a separate authorized workload change and real generation qualification. No bench or eval workload code is changed here.

## Validation and finite limits

| Check | Result | Evidence under `.temp/hybrid-a-r19-261101/` |
|---|---|---|
| `python3 -B -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v` | exit 0, 84 tests, no skips (69 existing + 15 new) | `proxy-03.log` |
| `HYBRID_SWA_CHECKER="$PWD/build/bin/llama-hybrid-check-swa" python3 -B -m unittest discover -s tools/hybrid-bench -p test_bench.py -v` | exit 0, 46 tests, no skips | `bench-03.log` |
| `python3 -B -m unittest discover -s tools/hybrid-eval -p test_eval.py -v` | exit 0, 11 tests, no skips | `eval-02.log` |
| `python3 -B tools/hybrid-proxy/template_controls.py --output .temp/hybrid-a-r19-261101/controls-06` | driver exit 0; identity 0; 11 expected-red mutants each 1 | `controls-06.log`, `controls-06/controls.json` |
| `python3 -m py_compile tools/hybrid-proxy/*.py tools/hybrid-bench/*.py tools/hybrid-eval/*.py` | exit 0, Python build/validation | `compile-02.log`; disk `PYTHONPYCACHEPREFIX` under task scratch |
| `build/bin/llama-hybrid-check-swa` | exit 0, 68 PASS, synthetic CPU geometry/state only, existing binary | `swa-readiness-01.log` |
| `git diff --check` | exit 0 | `whitespace-01.log` |
| Fixture copy and pre-existing test preservation audit | exit 0, 3 fixture files and 16 existing test files unchanged | `source-identity.json` |

The 141 unit tests are all green. Fixture variant counts and mutation worker executions are reported separately, not added to that unit count. `controls-06/source-identity.json` matches the delivered Python sources and captures. The existing native checker's SHA-256 is recorded in `validation.json`; it uses synthetic allocated caches and does not read model files.

Every command runs directly with its real exit code, `nice -n 10`, and disk TMPDIR `$PWD/.temp/hybrid-a-r19-261101/tmp` for tests. Full native model/CLI and GPU suites are unrun by the CPU/no-model task constraint. No C++ change or rebuild is needed; Python compilation is the affected-code build check. Existing proxy, bench and eval test files are byte-identical to the pinned base.

Development evidence is retained, including failures: journal-01 exited 1 because the unchanged rollback test injected a decoder outage before the newly derived header; snapshot ordering now preserves its original recovery contract. templates-01 exited 1 (negative tests initially resent shorter history, missing Glimmer call-name preservation and a wrong assumed Qwen ID). templates-05 exited 1 because forced-offload deliberately has no local user checkpoint; the positive test now checks local checkpoints and offload layout separately. eval-01 exited 1 because broad discovery collided with the two standalone `test_native` modules; the documented `test_eval.py` command passes. bench-01 passed with one checker-dependent skip; bench-03 runs that same test green with the existing checker. controls-01's plain-byte/context labels were not accepted as property kills: secondary guards rejected them. The final controls use both token modes for byte corruption and a synthetic tokenizer for context corruption, and require the intended assertion. No setup failure or wrong assertion is credited.

## Mutation controls

Final copied-source evidence: `.temp/hybrid-a-r19-261101/controls-06/`. The driver exits 0; identity exits 0 and survives. Each of 11 mutants exits 1 with a named intended assertion, no errors or skipped tests. The driver verifies module paths/bound namespaces, records source hashes, commands, PIDs and exact exits, and does not credit skipped or unapplied plants.

| Mutant | Gate narrowed to | Named failing test | Result / survivor bound |
|---|---|---|---|
| `identity` | unchanged implementation | none (positive identity) | survives; bounded fixture/HTTP contracts only, no native or GPU claim |
| `plain-bytes` | ID equality without byte equality | `TemplateHTTPTests.test_altered_piece_atomic` | killed |
| `foreign-bos` | prefix position without BOS authority | `TemplateHTTPTests.test_foreign_special_atomic` | killed |
| `double-bos` | BOS spelling without exact multiplicity | `TemplateHTTPTests.test_double_bos_atomic` | killed |
| `bos-in-text` | exact prefix IDs without visible-BOS exclusion | `TemplateHTTPTests.test_bos_in_text_atomic` | killed |
| `bos-bytes` | token count with BOS spelling counted as rendered bytes | `TemplateTests.test_layout_and_user_boundaries` | killed |
| `unknown-family` | fallback despite failed live-family proof | `TemplateHTTPTests.test_unknown_template_atomic` | killed |
| `split-eot` | exact suffix without a single ending token | `TemplateHTTPTests.test_split_eot_atomic` | killed |
| `exact-suffix` | ending token without full text/token suffix equality | `TemplateHTTPTests.test_changed_suffix_atomic` | killed |
| `gemma-tool-stop` | call terminator instead of decoder response opener | `TemplateTests.test_append` | killed |
| `generation-history` | byte reconstruction without generation-prefix invariance | `TemplateHTTPTests.test_generation_rewrites_history_atomic` | killed |
| `probe-context` | header detection without preceding-context invariance | `TemplateHTTPTests.test_probe_rewrites_context_atomic` | killed |

## Review and remaining GPU evidence

Self-review applied the code-review skill's scope, security, approach, general and tests checklists. Review notes are in `.temp/hybrid-a-r19-261101/review-01.md`; logs and identities remain under that task directory. No supplied attack-surface table or external landing property suite exists in the brief; this is an explicit brief gap. Existing reviewer tests are retained. Out-of-contract rows are the full native/model/GPU validations excluded by the CPU-only/no-model rule, plus the unchanged bench workload requested as proposal-only. This is local review handoff, with no independent review, push or landing claimed.

The GPU host must still establish actual sampled stop IDs and complete streamed IDs; parse/reasoning/tool reconciliation; real tokenizer suffix/prefix behavior beyond captured aligned spans; native checkpoint rollback for all three families; remote transfer/import identity, cache reuse and exact metering; and same-partition numerical/quality equivalence with the real models. Re-run local first-call and append qualification before another full benchmark. Fixture replay does not establish GPU state correctness, throughput or real generation quality.


## r20 rework

Additive rework on `b2e75e357ecfa67e67fe6c0583a4556e9ce4bce6` after review r6o. No rewrite, push, downloads, model access or GPU work. Evidence: `.temp/hybrid-a-r20-261101/`. The framing, guard and source-redaction work is ready for review. The authorized bench change remains pending the orchestrator's real tools captures; no workload edit or acceptance is claimed for that part.

Gemma continued assistants now start at their own content/reasoning prefix rather than an earlier `<|turn>` opener. The probe runs before the continued message's calls and preserves the preceding bytes. Both consecutive assistants and call/tool/answer/user have separate, byte-exact ranges. The reasoning alias is covered. Equal starts reject with 409; noncontiguous assistant content rendered after its tool responses also rejects. The stub reproduces continued-turn opening/closing behavior. HTTP tests verify persisted answer ranges and keep unmatched-tool rejection atomic across journal, slots and remote prefix.

All original 96 derived renders and 192 tokenizations are unchanged. Only full `chat_template` and `chat_template_tool_use` properties were replaced by `*_sha256` (null for an absent source). Oracle tests load original capture JSON from `HYBRID_TEMPLATE_FIXTURES`, default `/home/agent/lanes/hybrid/.temp/template-fixtures/`, verify both digests, and explicitly skip absent sources or missing Jinja2. Present mismatched source fails. The adapted r6o oracle matches 88 original renders; eight Gemma/Glimmer assistant-prefill renders require server assembly outside its scope. Live oracle probes agree with the stub for layout/user/append on all 84 supported variants. Both continued-turn shapes also pass real Jinja probes with synthetic character-piece tokenization; this does not qualify real uncaptured tokenizations.

Redaction applies to the current tree. Inherited r19 objects still contain the full sources and are preserved under the no-rewrite instruction. Muse Glimmer redistribution/licensing remains unresolved before publication of that history. No session links or new full template sources are committed.

| Check | Result | Evidence under `.temp/hybrid-a-r20-261101/` |
|---|---|---|
| Full proxy unittest discovery | exit 0, 106 tests, no skips | `proxy-02.log` |
| Bench `test_bench.py` | exit 0, 46 tests, 1 checker-dependent skip | `bench-01.log` |
| Previously skipped bench identity test with existing `HYBRID_SWA_CHECKER` | exit 0, 1 test, no skips | `bench-checker-01.log` |
| Eval `test_eval.py` | exit 0, 11 tests, no skips | `eval-01.log` |
| External-source oracle tests after helper cleanup | exit 0, 6 tests, no skips | `oracle-02.log` |
| Source-absent oracle control | exit 0, 1 passing fixture-contract test and 5 explicitly explained skips | `source-absent-01.log` |
| Copied-source template controls | driver exit 0; identity and 3 equivalent mutants survive with exit 0; 30 mutants exit 1 at named assertions | `controls-04/controls.json`, `controls-04.log` |
| Existing allocated-cache SWA checker (CPU, model-free) | exit 0, 68 PASS; existing binary, no rebuild | `swa-readiness-01.log` |
| Python compile validation without bytecode writes | exit 0 | `compile-02.log`, `compile_check.py` |
| Derived-output and original-test preservation audit | exit 0 | `source-audit-01.log`, `source-audit.json` |
| `git diff --check` | exit 0 | `whitespace-01.log` |
| `ty check --exit-zero-on-warning --output-format=github`; `editorconfig-checker` | both exit 127, executables absent; checks unrun, no install/download | `type-check-01.log`, `editorconfig-01.log` |

Bench's other 45 tests are reused unchanged, and its checker-dependent test is executed separately under the available binary. All 46 distinct bench tests have green evidence; the skipped invocation itself is not passing evidence for that case. Full native/model/GPU suites are unrun by the CPU-only constraint. Commands, observed start/end times and owned PIDs are in the corresponding `*-process.json`. Every Python check uses disk TMPDIR under task scratch and `PYTHONDONTWRITEBYTECODE=1`; CPU checks use `nice -n 10`. No C++ change requires rebuilding the engine.

Negative tests now isolate each family BOS/EOS/source property, generation header, assistant/user transitions, final closing and generation-prefix preservation; append's closing prefix; both Gemma tool-probe guards; equal/reversed starts; and checkpoint header text. Duplicate assistant content probes also reject even with one reasoning probe. Positive controls precede each negative. No setup error, skip or secondary rejection is credited as an intended kill.

| Mutant | Gate narrowed to | Named failing test | Result / survivor bound |
|---|---|---|---|
| `identity` | unchanged source under the same tests | `none (positive survivor)` | survives, exit 0; unchanged source under the same tests |
| `plain-bytes` | IDs alone cannot prove byte preservation | `TemplateHTTPTests.test_altered_piece_atomic` | killed, exit 1 |
| `foreign-bos` | piece position alone does not authorize a foreign special | `TemplateHTTPTests.test_foreign_special_atomic` | killed, exit 1 |
| `double-bos` | BOS identity alone cannot limit its multiplicity | `TemplateHTTPTests.test_double_bos_atomic` | killed, exit 1 |
| `bos-in-text` | exact IDs alone do not permit a visible and invisible BOS | `TemplateHTTPTests.test_bos_in_text_atomic` | killed, exit 1 |
| `bos-bytes` | BOS spelling cannot advance a rendered-byte offset | `TemplateTests.test_layout_and_user_boundaries` | killed, exit 1 |
| `unknown-family` | a failed live-family check cannot select a fallback | `TemplateHTTPTests.test_unknown_template_atomic` | killed, exit 1 |
| `split-eot` | an exact suffix alone does not prove one ending token | `TemplateHTTPTests.test_split_eot_atomic` | killed, exit 1 |
| `exact-suffix` | a recognizable ending alone cannot authorize a changed suffix | `TemplateHTTPTests.test_changed_suffix_atomic` | killed, exit 1 |
| `gemma-tool-stop` | the call terminator cannot replace the decoder response opener | `TemplateTests.test_append` | killed, exit 1 |
| `generation-history` | exact pieces alone do not protect history from generation prompts | `TemplateHTTPTests.test_generation_rewrites_history_atomic` | killed, exit 1 |
| `probe-context` | a recognizable header alone does not protect its preceding context | `TemplateHTTPTests.test_probe_rewrites_context_atomic` | killed, exit 1 |
| `continued-turn-frame` | opening-frame search without continued-model boundaries | `ContinuedTurnTests.test_consecutive_assistants` | killed, exit 1 |
| `continued-tool-frame` | opening-frame search after an embedded tool response | `ContinuedTurnTests.test_call_tool_answer` | killed, exit 1 |
| `family-source-props` | live renders without BOS/EOS/source authority | `TemplateGuardTests.test_family_source_properties` | killed, exit 1 |
| `family-bos` | source framing without the declared BOS | `TemplateGuardTests.test_family_source_properties` | killed, exit 1 |
| `family-eos` | source framing without the declared EOS | `TemplateGuardTests.test_family_source_properties` | killed, exit 1 |
| `family-source-opening` | live frames without an opening in the template source | `TemplateGuardTests.test_family_source_properties` | killed, exit 1 |
| `family-source-closing` | live frames without a closing in the template source | `TemplateGuardTests.test_family_source_properties` | killed, exit 1 |
| `family-gen-header` | preserved history without a valid generation header | `TemplateGuardTests.test_family_generation_header` | killed, exit 1 |
| `family-closing-after-last` | valid transitions without an exact final closing | `TemplateGuardTests.test_family_closings` | killed, exit 1 |
| `family-closing-before-assistant` | initial user framing without a valid assistant transition | `TemplateGuardTests.test_family_closings` | killed, exit 1 |
| `family-closing-after-assistant` | initial user framing without a valid next-user transition | `TemplateGuardTests.test_family_closings` | killed, exit 1 |
| `family-generation-prefix` | valid generation header without preserved history | `TemplateGuardTests.test_family_generation_prefix` | killed, exit 1 |
| `append-closing-prefix` | exact suffix without its closing prefix | `TemplateGuardTests.test_append_closing_prefix` | killed, exit 1 |
| `gemma-tool-count` | matching tool suffix without an exact probe count | `TemplateGuardTests.test_gemma_tool_probe_count` | killed, exit 1 |
| `gemma-tool-closing` | matching response suffix without a valid call terminator | `TemplateGuardTests.test_gemma_tool_probe_closing` | killed, exit 1 |
| `layout-overlap` | valid frames without strict message range ordering | `TemplateGuardTests.test_layout_overlap` | killed, exit 1 |
| `layout-equal-starts` | ordered frames that still allow an empty message range | `TemplateGuardTests.test_layout_overlap` | killed, exit 1 |
| `users-header-text` | matching prefix tokens without matching live header bytes | `TemplateGuardTests.test_users_header_text` | killed, exit 1 |
| `assistant-probe-count` | unique reasoning probe without unique content | `TemplateGuardTests.test_duplicate_assistant_probe` | killed, exit 1 |
| `frame-nonassistant-count` | non-assistant reasoning is absent; the unique-marker and duplicate-marker guards imply this count | `none (positive survivor)` | survives, exit 0; non-assistant reasoning is absent; the unique-marker and duplicate-marker guards imply this count |
| `bos-len` | one extra piece plus exact bos_ids + plain_ids already implies one BOS ID | `none (positive survivor)` | survives, exit 0; one extra piece plus exact bos_ids + plain_ids already implies one BOS ID |
| `plain-equal-branch` | when IDs agree, returned bytes come from the already checked plain pieces | `none (positive survivor)` | survives, exit 0; when IDs agree, returned bytes come from the already checked plain pieces |

`frame-nonassistant-count`, `bos-len` and `plain-equal-branch` remain the three review-named equivalent mutants, with their exact survivor bounds above. Identity/equivalence workers run the 31 source-free fixture/HTTP/guard/continued-turn tests; optional external oracle skips do not invalidate or manufacture their outcomes.

Retained development failures: `templates-02` exited 1 because an empty assistant can own its closing bytes (the corrected witness places it before another continued assistant) and because the oracle count excluded four valid Qwen prefill captures. `controls-01` exited 1: the final-closing and generation-prefix plants tripped secondary guards (invalid), and the before-assistant plant survived. Corrected, property-isolating plants are used in controls-02 through controls-04; no initial mis-targeted kill is claimed. Subsequent oracle-filter cleanup was rerun green in oracle-02 and the delivered control identity is controls-04.

Self-review covers scope, security, general and tests; notes are in `review-01.md`. Every pre-existing test method is retained; all other pre-existing proxy/bench/eval test files are unchanged. README and unresolved-question edits document the external-source tool and remaining capture/licence dependency. No formal attack-surface table or external landing property suite was supplied in the brief (gap); r6o rows R1-R8 are used as review coverage: R1 existing exactness tests; R2 continued-turn/guard/oracle tests; R3 append/tool-prefix tests; R4 unchanged Qwen tests and unchanged bench code; R5 hash-bound oracle comparisons; R6 identity/intended-kill/equivalence controls; R7 tool ranges and atomic unmatched-tool HTTP rejection; R8 source-redaction audit. Out-of-contract evidence remains full native/model/GPU state, actual sampled stops, throughput and numerical/quality qualification, per the CPU-only requirement. Independent review and acceptance remain pending.

### Bench capture dependency

No new capture directory has been supplied or observed in `#hybrid/prototype-go` at this checkpoint. Existing fixtures have no root tools definitions, so they cannot prove the requested workload change. The deterministic candidate is a fresh user request before a matching synthetic assistant call, followed by the unchanged response blob and continuation user; the user delimiter preserves A1's existing closing without rewriting immutable history or adding a sampled decode round. A call inserted directly after A1 would merge the Gemma turn and is unsuitable. Validate the candidate against captured tools renders and exact append before implementation. See the tracked design note `.temp/hybrid-a-r20-261101/bench-plan.md`. Qwen workload bytes remain unchanged at this checkpoint.

## r21 matched synthetic tool turns

Additive continuation of signed r20 head `4aeba4a2fe0a06df67448bb06bbfb949e1fa104b`; no rewrite or push. CPU only, no downloads or target-model/GPU access. Evidence is under `.temp/hybrid-a-r21-261101/`. This local prototype is ready for review after the checks below; independent acceptance is not claimed.

The bench probes the active `/apply-template` with an unmatched nonce result using the benchmark's thinking setting. A single preserved nonce retains the old workload. A dropped result or HTTP 400/422 rejection selects a fresh user request, matching assistant `read` call, unchanged synthetic tool result and existing continuation user. The fresh user request preserves the preceding sampled assistant's closing delimiter, without an extra generation round. Duplicate unmatched probes, missing/duplicate matched probes and HTTP authentication/server/transport failures abort. The verdict is recorded as `tool_call_added` in probe and call rows (false for depth rows). Payload construction, token-count validation, schedule and RNG are unchanged.

Qwen and Glimmer retain the original unmatched-tool path. `SyntheticToolTests.test_qwen_request_bytes` compares old message lists, renders and actual `stream_chat` request JSON bytes over both complete fixed/measured schedules. `test_main_call_verdict` exercises main's actual requests and persisted result fields for Qwen and Gemma. Gemma matching-call append retains the sampled anchor; the direct A1 -> call capture merges the model turn and therefore cannot use A1's former closing as an incremental anchor. This case is explicitly rejected and the inserted user solves it.

### Captures, ranges and redaction

Three new `fixtures/templates/*-r20.json` files preserve all 96 real renders and 192 real tokenizations from b12094-74f8f4c3a, with both full template properties replaced by SHA-256 only. Existing fixtures are unchanged. The source filename selects the corresponding external capture for digest verification. Missing sources/Jinja2 still skip with a reason; a present wrong digest fails. Inherited r19 source-bearing objects remain unchanged under the no-rewrite instruction; the existing publication/licence question remains open.

The fixture renderer accepts only the captured fixed `read` tool schema, borrowing its captured static system preamble while independently rendering each message/probe. No general schema or uncaptured-tokenizer claim is made. The external Jinja oracle matches all 96 r20 renders and their layout/user/append probes. Its test-only tool conversion restores the server wrapper field order and observed fixed-schema parameter order lost in the sorted capture JSON; this is documented as a finite oracle limit. The original oracle coverage remains 88 renders/84 supported probe variants, excluding eight server-assembled assistant-prefill renders.

`CapturedR20Tests` checks all eight new shapes, all generation/thinking variants and both tool-definition choices. Starts are independently found in captured bytes, not inferred from agreement between two frame implementations. Consecutive Gemma assistants, continued calls and call/tool/answer segments own disjoint ranges. Every bench tool range contains the actual four captured filler token IDs: Gemma 1123 x4; Glimmer 831 x4; Qwen 87 then 830 x3 (its pre-existing leading-space trim). Both matched bench shapes retain all four tokens. Twenty-four HTTP admissions cover every family/shape under benchmark generation/thinking flags and verify persisted captured IDs and ranges against decoder stub slot state. Production journal code required no change.

Fixture SHA-256 after redaction:

- `gemma4-31b-r20.json`: `007c5d23c336b84608af71aad27a8f91635f29bf21c020802b860cc3d1d9cd48`
- `muse-glimmer-30b-r20.json`: `e56eeec8913355d6bf7a09daa324122556d6cc8aa9567daabd1553d2714141ed`
- `qwen3.8-27b-r20.json`: `766102cd17dbe12726202fbec2490469929d1c401e535703f06b3f2f3ec8c709`

### Validation

Commands use `TMPDIR=$PWD/.temp/hybrid-a-r21-261101` on disk and `PYTHONDONTWRITEBYTECODE=1`. No upstream hosted checks or C++ rebuild are claimed; C++ source is unchanged.

| Check | Result | Evidence |
|---|---|---|
| `python3 -B -m unittest discover -s tools/hybrid-proxy -p 'test_*.py'` | exit 0, 112 tests | `proxy-02.log` |
| `HYBRID_SWA_CHECKER=$PWD/build/bin/llama-hybrid-check-swa python3 -B -m unittest discover -s tools/hybrid-bench -p test_bench.py` | exit 0, 54 tests, no skips | `bench-04.log` |
| `python3 -B -m unittest discover -s tools/hybrid-eval -p test_eval.py` | exit 0, 11 tests | `eval-01.log` |
| Full template suite with external sources | exit 0, 43 tests, no skips | `templates-03.log` (96 HTTP variants before narrowing that check to 24 benchmark-flag admissions) |
| `python3 -B -m unittest discover -s tools/hybrid-proxy -p test_templates.py -k test_ranges -v` | exit 0, explicit captured filler-token count assertions | `payload-tokens-01.log` |
| Source-absent full template suite | exit 0, 43 tests, 6 explicit source-absence skips | `source-absent-01.log` |
| Source-absent bench oracle | exit 0, 1 explicit source-absence skip | `bench-source-absent-01.log` |
| Changed Python syntax/ASCII, exact derived-capture equality, source hashes and prior test-name retention | exit 0 | `audit-01.log` |
| `git diff --check` | exit 0 | `diff-check-01.log` |

The exact upstream `ty` and `editorconfig-checker` tools are absent; no downloads/installations were attempted. Their checks are unrun, not passing. The code-style workflow's C++ model-naming gate is outside this Python/fixture delta.

Development failures retained: `templates-01.log` exited 1 on Qwen/Glimmer tool-schema JSON ordering before the oracle conversion was corrected; no captured bytes were modified to match the oracle. `bench-01.log` exited 1 because the existing decoder mock referenced a wrong variable while being updated to preserve probe content (11 assertion failures and one error); the mock correction is included. `bench-02` and `bench-03` pass with only the existing optional checker skip; `bench-04` enables that checker and passes without skips. Controls are repeated after test-source changes so the delivered identities include the explicit filler-token assertions.

### Controls and review bounds

`python3 -B tools/hybrid-proxy/template_controls.py --output .temp/hybrid-a-r21-261101/controls-03` exits 0. Identity survives all 43 source-free fixture/HTTP/guard/continued-turn/bench tests, exit 0. All 40 fault mutants exit 1 at their named intended assertions, with no errors/skips; the three declared equivalent mutants survive, exit 0. `controls-03/controls.json` contains the full table, commands, copied-module bindings, PID records, failure traces and source/test/fixture identities. Identity digests were rechecked against the delivered sources. The 30 existing r20 fault mutants retain their named tests and narrowing bounds from the preceding table and are all killed again. New controls:

| Mutant | Narrows the gate to | Named failing test | Outcome |
|---|---|---|---|
| `bench-probe-duplicate` | an unmatched probe may appear at most once | `SyntheticToolTests.test_probe_preservation_guards` | killed, exit 1 |
| `bench-matched-preservation` | fallback must retain exactly one matched probe | `SyntheticToolTests.test_probe_preservation_guards` | killed, exit 1 |
| `bench-rejection-status` | only client rejections authorize matched fallback | `SyntheticToolTests.test_rejection_and_outage` | killed, exit 1 |
| `bench-force-call` | preserving unmatched templates keep their old requests | `SyntheticToolTests.test_qwen_request_bytes` | killed, exit 1 |
| `bench-never-call` | dropping templates need a matching call | `SyntheticToolTests.test_live_probe` | killed, exit 1 |
| `bench-call-id` | the assistant call ID must match its tool response | `SyntheticToolTests.test_payload_and_matching_call` | killed, exit 1 |
| `bench-payload` | matching-call framing must retain the measured payload | `SyntheticToolTests.test_payload_and_matching_call` | killed, exit 1 |
| `bench-user-anchor` | a fresh user turn must preserve the sampled closing | `SyntheticToolTests.test_gemma_append_anchor` | killed, exit 1 |
| `bench-row-verdict` | each call row records whether a call was actually inserted | `SyntheticToolTests.test_main_call_verdict` | killed, exit 1 |
| `fixture-tools-schema` | a captured preamble cannot qualify a different tool schema | `CapturedR20Tests.test_uncaptured_tools` | killed, exit 1 |

| Surviving mutant | Named suite | Bound stated by survival |
|---|---|---|
| `frame-nonassistant-count` | 43 source-free identity tests | non-assistant reasoning is absent; the unique-marker and duplicate-marker guards imply this count |
| `bos-len` | 43 source-free identity tests | one extra piece plus exact bos_ids + plain_ids already implies one BOS ID |
| `plain-equal-branch` | 43 source-free identity tests | when IDs agree, returned bytes come from the already checked plain pieces |

Self-review covers scope, security, general and tests (`review-01.md`). All previous named reviewer tests remain present. The brief supplies no formal surface table or external landing property suite (declared gap); r6o R1-R8 remain mapped to exactness, continued frames, append boundaries, unchanged Qwen, hash-bound oracle, identity/mutants, tool ownership and redaction. Out of contract under the CPU-only brief: target GPU/native sampled stops and state identity, checkpoint/numerical equivalence, throughput, arbitrary schemas/multimodal and uncaptured model tokenizations. No changes outside the bench and its proxy fixture/test/documentation support, except the root unresolved-question update explaining the satisfied capture dependency.

Parallel r6p report `hybrid-review-proto/llama.cpp/.temp/REVIEW-templates-rework-claude.md` was still absent at 2026-11-01 16:38:18 UTC. No r6p acceptance or unresolved finding closure is inferred. Delivery/source/PID checks are in `delivery-audit-01.log`; tracker and local no-board evidence are in `.temp/tasks.md` and `goal-01.log`.

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

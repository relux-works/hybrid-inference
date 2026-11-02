# Hybrid SWA and chunk-state integration

Run: hybrid-a-r27-261101. Local branch `m3-swa-chunk`; ready for review, no push. Reviewed commits keep their identities. Fetched `https://github.com/relux-works/hybrid-inference` main at exactly `1309bcb8b02e8e8de764f2456e5a6af94d7de476`.

## Signed history and conflict resolutions

1. Merge `bbb06def7fe44f9253c457618c4bcaa6857836aa`: first parent `5c8b8c045223466a27a8481b076cfd511a2d4e5d` (chunk state), second parent `ac28dbf4731829903a103baf3ba80d53899d5d73` (SWA r6).
2. Merge `25e2d4398ec02503c71f1d5f37d06cbe64d820b1`: first parent the merge above, second parent exact fetched main `1309bcb8`. Both signatures verify G. No rebase, history rewrite, push or PR.

There were four content conflicts in the first merge and five in the main merge. Every resolution is below.

| Merge | File | Resolution |
| --- | --- | --- |
| SWA | `src/llama-kv-cache-iswa.cpp` | Keep the `!(flags & KV_ONLY)` wrapper and the r6 inner call `kv_swa->state_write_range(io, seq_id, -1, p1, flags | KV_LAYOUT)`. Early chunks omit the entire window; final chunks preserve physical layout. |
| SWA | `tools/hybrid/check-swa.cpp` | Retain both branches' appended tests. Convert the late-apply chunk window fixture to v2. Keep legacy fixtures only on intentional generic-state/refusal paths. Remove the duplicate include. |
| SWA | `tools/hybrid/README.md` | Keep both the final-state chunk and physical-layout descriptions; document HYBD 2 / body format v3 and link this report. |
| SWA | `tools/hybrid/UNRESOLVED_QUESTIONS.md` | Retain the diagnosed timing failure and production qualification gaps; update the combined-tree CPU item to this report. |
| Main | `README.md` | Keep both SWA transfer/checker guidance and PR 11 reference-scoring guidance; link this report. |
| Main | `tools/hybrid-bench/README.md` | Keep template qualification and all PR 11 reference/study/Mirai sections. |
| Main | `tools/hybrid-bench/bench_common.py` | Retain all SWA metadata prefixes plus `mirai.version`; keep both SWA planner helpers and Qwen trunk/Mirai validation helpers. Dispatch identity construction to SWA layout or Qwen trunk as appropriate. Explicitly retain Qwen-only admission for Mirai-marked metadata. |
| Main | `tools/hybrid-proxy/README.md` | Keep captured-template qualification and PR 12 deferred shadow-sync documentation, including the restart-test link. |
| Main | `tools/hybrid/CMakeLists.txt` | Keep `llama-hybrid-check-swa`, `llama-hybrid-check-identity` and `llama-hybrid-score`. |

Clean auto-merges were also reviewed. `src/llama-context.cpp` needs the manual layout requirement below despite no conflict. `src/llama-memory.h`, `hybrid-state.cpp`, server import/staging and `prefill.cpp` keep both contracts. The production proxy auto-merge retains synchronous prefix validation for critical offloads and PR 12's deferred sync for local-kept calls. Its r26 restart test needed the two protocol-health reads and the new critical-extend transfer fields in its expected trace; the original test name, restart/replay assertions and no-stale-import invariant remain. PR 11's benchmark collector, launchers, reference scorer and Mirai fixtures remain present.

## Invariants and new regression coverage

- Chunk validation and apply share the same `read(validate)` lambda. Its final-window call now includes `KV_LAYOUT | (validate ? VALIDATE : 0)`. Both reject v0; neither can silently compact an old chunk-only node's final window.
- HYBD remains version 2; native `body_format` is now `llama-native-chunks-v3`. Native decoding refuses chunk-only v2 envelopes with `unsupported delta layout` before parsing the body. Codec and real HTTP controls exercise this with valid checksums. Synthetic Python token-state fixtures keep their own explicit format.
- `KV_ONLY` is included among the public flags in the disjointness assertion, using `llama-ext.h` for its definition.
- Deferred writes remain pending across all chunks; rejection discards every reader and restores both base and fixed metadata. The KV metadata snapshot includes allocator heads. The window floor remains `{max(0,n1-n_swa),n1}`; layout parsing extends it to saved occupied cells.
- Generic ISWA sequence and PARTIAL_ONLY state paths are byte-identical to r6, with independent legacy-byte tests. Only hybrid session save/load explicitly request layout. The accepted threshold implementation is byte-identical to r6.
- The new public-API `swa_chunk_legacy_window_refused` test covers fresh and populated destinations, both `llama_state_seq_validate_chunk` and `llama_state_seq_set_data_chunks`, and unchanged rejected state. Nearby v2 positive controls commit successfully. Three-chunk tiny-model transfers cover interruption after k=1,2, byte/physical-layout equality with one import and identical held-token logits for fresh and append.
- Existing allocated Gemma/Glimmer tests cover unified/separate caches, early-window omission, two-chunk equivalence/refusals, late capacity failure, neighbor preservation, non-prefix indices, masked metadata, v0/v1 refusal and checkpoint legacy bytes. Existing reviewer test names were retained.
- Merged planner: Gemma fixed-window budget 445,680,656 bytes, full row 43,520 bytes, final width 2,070. Including 24 metadata bytes/token and 1 MiB reserve gives 536,865,312 bytes, below 536,870,912; one more token exceeds the cap. Glimmer fixed budget is 43,514,896 bytes. Early native chunks carry no fixed window; the planner conservatively reserves the fixed budget when choosing every chunk width.

`invariants-01` checks ordered ancestry, signatures, flags, export/import gates, rollback heads, generic/threshold source identity and retained SWA tests. `validation-identity.json` binds non-document source files and binaries to this tree. Final documentation changes do not alter those tested sources.

## CPU validation

Evidence root: `.temp/hybrid-a-r27-261101/`. Numbered JSON records hold exact argv, PID, real exit, times and sampled free disk; logs have the same stems. Commands run as direct processes, without a tee pipeline. Linux x86_64, GCC 15.2.0, CMake 4.4.4, Python 3.14.4, Release AVX2/FMA with `GGML_NATIVE=OFF`, CPU only. Existing 0.8B files and local header fixtures only; no model downloads or GPU. Builds are nice 19, at most -j4, serial under the shared model-cache lock. Tests are niced, TMPDIR is task-local disk, Python bytecode is disabled. Owned build/control binaries and native scratch are removed after validation; top-level command logs, ledgers and source/binary hashes remain. Minimum sampled free disk was 34,416,803,840 bytes; cleanup left 35,506,511,872 bytes, above the 28 GiB reserve. All 15 inherited/merge commits introduced above main verify G (`history-signatures-01.log`).

| Gate | Real exit / result | Evidence stems |
| --- | --- | --- |
| Configure, corrected builds and restored candidate | 0 | `configure-01`, `build-02` through `build-05`, `build-restored-01` |
| SWA allocated caches and tiny model | 0; 18/18 original depth/chain cases exact across all eight logits at 64 continuation positions; fresh/append chunk controls pass | `swa-final-02` |
| Reviewer public-API multi-slot F16 repro, unchanged source | 0 at 512/520/600; 3,224-byte checkpoints, restore succeeds | `repro-{512,520,600}-01` |
| Server component | 0; 50 tests, 235 assertions, including 44 threshold assertions | `server-component-final-01` |
| Q4_0 and Q8_0 chunks, fresh and append | 0 each; four-chunk byte equality and held-token logits; interruptions k=1,2,3; malformed/state/order/version/checksum/late-apply refusals | `chunks-q4-01`, `chunks-q8-01` |
| Real HTTP/node/server/proxy refusals and transfers | 0; original slot unchanged on rejection/interruption; final-only commit and stale staging replacement; old body format refused | `http-01` |
| Proxy | 0; 123 tests | `proxy-02` |
| Benchmark | 0; 81 tests, six optional native tests skipped in discovery | `bench-02` |
| Mirai native headers | 0; 12 tests, including all six optional native cases; exact local header identities verified | `mirai-native-01` |
| Codec / guards / malformed recurrent | 0 each | `codec-01`, `guards-01`, `recurrent-01` |
| Golden append and distributions, 512/2048/8192; capacity and golden checker | 0 each; original 64-token controls retained | `append-01`, `distributions-01`, `capacity-01`, `golden-01` |
| Focused chunk/noise, LRU and real RAM budget | 0; intended corruption child commands exit 2 or 3, real 503 leaves zero retained sessions | `integrated-noise-01`, `integrated-chunk-01`, `extra-checks-01` |
| Full inherited integrated driver | 1; timing-no-link benchmark exits 3 for fresh and append_chain, whole-prompt top-1 7/8, mean KL 0.00852205499931497; same-partition exact, max diff/KL 0 | `integrated-all-01` |
| Reference native scoring and refusal controls | 0; real hetero-node orchestration, native refusal exits 2, checker exit 5 retained as uncalibrated | `reference-native-01` |
| Planner boundary / static syntax, ASCII, whitespace and model naming / GPU recipe shell syntax | 0 each | `planner-01`, `static-checks-01`, `gpu-recipe-syntax-01` |

The integrated driver's red result matches the previously accepted r24/r7g CPU diagnostic metrics. No fresh baseline build was substituted or claimed here. Its abort precedes timing-link/eviction/budget; eviction and budget were therefore run separately and pass. It is not an all-green timing qualification. Imported-versus-same-partition and whole-prompt results stay distinct; no threshold or model was changed.

Golden files were not edited. Task-local envelope copies use version 2/body format v3 with unchanged native bodies and checksums; `golden-conversion.json` records those hashes and both headers. No SWA v0 body was upgraded this way.

Earlier failures and corrections are recorded in `early-failures.md`: initial build exit 2 (missing prefill offsets), initial proxy/restart trace failures, initial benchmark refusal-message mismatch, and the inverted fixed-state fixture. The first failed SWA log was overwritten by a reused driver stem; only its observed diagnostic summary remains, not a complete failed log. The first MC classifier expected the older incidental post-check failure; the merged test rejects earlier at its named chunk transaction gate. Its diagnostic now names that test, and the rerun exits 1 at that assertion. None of these earlier attempts is presented as green.

`ty`, `editorconfig-checker` and `clang-format` are unavailable, so those checks are unrun. No upstream/all-backend CI, production SWA/27B, CUDA, Metal, heterogeneous equivalence, 5G performance, fatal-backend/process crash recovery or persistent decoder-outage qualification is claimed. No formal surface table or independent landing-property suite was supplied (brief gap). No requested row is declared out of contract; this local merge is ready for independent review, with the timing red and unrun production qualification explicit.

## Mutant reruns

The targeted set has 22/22 kills, zero survivors: SWA producer controls 4/4, r7l SWA reader/session controls 7/7, chunk reviewer M1-M6 6/6 plus M6b, the two retained r24 producer controls, the new merge-layout control and main's R2 restart control. Each compile/link exits 0; each credited expected-red test actually exits 1. Interposition libraries and copied Python modules affect only owned test processes. The N2 threshold source is restored and rebuilt before final green tests. Candidate engine sources/libraries remain unchanged by interposition controls.

| Mutant | What it narrows the gate to | Named failing test / diagnostic |
| --- | --- | --- |
| SWA F1, write masked payload | Size scales with retained K/V rows | `swa_masked_rows_metadata_only` |
| SWA F2, mark generic writes | Generic bytes remain legacy | `swa_generic_state_legacy_bytes` |
| SWA F3, ignore saved layout | Non-prefix physical placement survives | `swa_nonprefix_physical_indices` |
| SWA N2, n_swa > 1 | One-token windows use SWA semantics | `prompt_reuse_threshold`, `one-token SWA window` |
| SWA MA, compact saved indices | Saved indices are consumed | `swa_nonprefix_physical_indices` |
| SWA MB, drop masked metadata | Masked cells remain physically occupied | `swa_chunk_single_equivalence: physical layout or bytes differ` |
| SWA MC, reader threshold +1 | Reader consumes the writer's exact payload | `swa_chunk_transaction_equivalence_and_refusals: SWA chunk transaction failed` |
| SWA MD, accept v1 layout | Prior physical version is refused | `swa_previous_layout_version_refused` |
| SWA ME, accept v0 layout | Legacy compacted append is refused | `swa_legacy_compacted_record_refused` |
| SWA MF, legacy node save/load | Session save/load preserves physical cells/head | `swa_physical_cells_and_head` |
| SWA MH, mark only PARTIAL_ONLY | Checkpoint bytes stay legacy too | `swa_checkpoint_legacy_bytes` |
| Chunk M1, apply non-final validation | Early validation publishes nothing | `swa_chunk_transaction_equivalence_and_refusals`, `SWA early chunk changed cache` |
| Chunk M2, omit reader discard | Late failure publishes no deferred KV | `swa_chunk_late_window_rollback: deferred keys written after rejection` |
| Chunk M3, omit fixed restore | Late failure restores window metadata/head | `swa_chunk_late_window_rollback: metadata or neighbor changed` |
| Chunk M4, skip lost-ack restore | Original slot restored after uncertain final commit | `test_lost_final_ack_restores_original_slot` |
| Chunk M5, skip endpoint check | Final recurrent endpoint must match range | `chunk_late_endpoint_rollback` |
| Chunk M6, persist staged state | Durable applied prefix stays committed-only | `test_staged_chunks_do_not_persist_applied_state_or_drop_checkpoints` |
| Chunk M6b, clear staged checkpoints | Staging retains recovery checkpoints | Same M6 named test |
| R24 skip per-chunk validation | Early native fixed state refuses immediately | `chunk_early_native_state` |
| R24 no KV metadata restore | Late failure restores old KV metadata | `chunk_late_endpoint_rollback: chunk failure changed cells or state` |
| Merge, omit chunk layout flag | Both public chunk APIs must require v2 windows | `swa_chunk_legacy_window_refused: validate accepted v0` |
| Main R2, bypass prefix GET | Restart cannot import through a stale remote ID | `test_burst_node_restart_then_offload` |

Ledger: `controls.json` (use the latest record per mutant), `proxy-mutants.json`, `r24-mutants.json`, `n2-mutant.json`. The early MC matcher miss is retained as an earlier uncredited classification; the corrected, named rerun is the final evidence.

The inherited template-control suite also ran: 40 killed mutants, three intentionally bounded equivalent survivors, and one passing identity control (44 executions). Its command exits 0 according to that declared contract. `template-controls/controls.json` records each killed mutant's named failing test, exact command and assertion. All template controls below are separate from the 22 targeted controls. The three survivors are not counted as kills.

| Template mutant | What it narrows the gate to | Named failing test | Result / survivor bound |
| --- | --- | --- | --- |
| `plain-bytes` | IDs alone cannot prove byte preservation | `TemplateHTTPTests.test_altered_piece_atomic` | Killed, exit 1 |
| `foreign-bos` | piece position alone does not authorize a foreign special | `TemplateHTTPTests.test_foreign_special_atomic` | Killed, exit 1 |
| `double-bos` | BOS identity alone cannot limit its multiplicity | `TemplateHTTPTests.test_double_bos_atomic` | Killed, exit 1 |
| `bos-in-text` | exact IDs alone do not permit a visible and invisible BOS | `TemplateHTTPTests.test_bos_in_text_atomic` | Killed, exit 1 |
| `bos-bytes` | BOS spelling cannot advance a rendered-byte offset | `TemplateTests.test_layout_and_user_boundaries` | Killed, exit 1 |
| `unknown-family` | a failed live-family check cannot select a fallback | `TemplateHTTPTests.test_unknown_template_atomic` | Killed, exit 1 |
| `split-eot` | an exact suffix alone does not prove one ending token | `TemplateHTTPTests.test_split_eot_atomic` | Killed, exit 1 |
| `exact-suffix` | a recognizable ending alone cannot authorize a changed suffix | `TemplateHTTPTests.test_changed_suffix_atomic` | Killed, exit 1 |
| `gemma-tool-stop` | the call terminator cannot replace the decoder response opener | `TemplateTests.test_append` | Killed, exit 1 |
| `generation-history` | exact pieces alone do not protect history from generation prompts | `TemplateHTTPTests.test_generation_rewrites_history_atomic` | Killed, exit 1 |
| `probe-context` | a recognizable header alone does not protect its preceding context | `TemplateHTTPTests.test_probe_rewrites_context_atomic` | Killed, exit 1 |
| `continued-turn-frame` | opening-frame search without continued-model boundaries | `ContinuedTurnTests.test_consecutive_assistants` | Killed, exit 1 |
| `continued-tool-frame` | opening-frame search after an embedded tool response | `ContinuedTurnTests.test_call_tool_answer` | Killed, exit 1 |
| `family-source-props` | live renders without BOS/EOS/source authority | `TemplateGuardTests.test_family_source_properties` | Killed, exit 1 |
| `family-bos` | source framing without the declared BOS | `TemplateGuardTests.test_family_source_properties` | Killed, exit 1 |
| `family-eos` | source framing without the declared EOS | `TemplateGuardTests.test_family_source_properties` | Killed, exit 1 |
| `family-source-opening` | live frames without an opening in the template source | `TemplateGuardTests.test_family_source_properties` | Killed, exit 1 |
| `family-source-closing` | live frames without a closing in the template source | `TemplateGuardTests.test_family_source_properties` | Killed, exit 1 |
| `family-gen-header` | preserved history without a valid generation header | `TemplateGuardTests.test_family_generation_header` | Killed, exit 1 |
| `family-closing-after-last` | valid transitions without an exact final closing | `TemplateGuardTests.test_family_closings` | Killed, exit 1 |
| `family-closing-before-assistant` | initial user framing without a valid assistant transition | `TemplateGuardTests.test_family_closings` | Killed, exit 1 |
| `family-closing-after-assistant` | initial user framing without a valid next-user transition | `TemplateGuardTests.test_family_closings` | Killed, exit 1 |
| `family-generation-prefix` | valid generation header without preserved history | `TemplateGuardTests.test_family_generation_prefix` | Killed, exit 1 |
| `append-closing-prefix` | exact suffix without its closing prefix | `TemplateGuardTests.test_append_closing_prefix` | Killed, exit 1 |
| `gemma-tool-count` | matching tool suffix without an exact probe count | `TemplateGuardTests.test_gemma_tool_probe_count` | Killed, exit 1 |
| `gemma-tool-closing` | matching response suffix without a valid call terminator | `TemplateGuardTests.test_gemma_tool_probe_closing` | Killed, exit 1 |
| `layout-overlap` | valid frames without strict message range ordering | `TemplateGuardTests.test_layout_overlap` | Killed, exit 1 |
| `layout-equal-starts` | ordered frames that still allow an empty message range | `TemplateGuardTests.test_layout_overlap` | Killed, exit 1 |
| `users-header-text` | matching prefix tokens without matching live header bytes | `TemplateGuardTests.test_users_header_text` | Killed, exit 1 |
| `assistant-probe-count` | unique reasoning probe without unique content | `TemplateGuardTests.test_duplicate_assistant_probe` | Killed, exit 1 |
| `bench-probe-duplicate` | an unmatched probe may appear at most once | `SyntheticToolTests.test_probe_preservation_guards` | Killed, exit 1 |
| `bench-matched-preservation` | fallback must retain exactly one matched probe | `SyntheticToolTests.test_probe_preservation_guards` | Killed, exit 1 |
| `bench-rejection-status` | only client rejections authorize matched fallback | `SyntheticToolTests.test_rejection_and_outage` | Killed, exit 1 |
| `bench-force-call` | preserving unmatched templates keep their old requests | `SyntheticToolTests.test_qwen_request_bytes` | Killed, exit 1 |
| `bench-never-call` | dropping templates need a matching call | `SyntheticToolTests.test_live_probe` | Killed, exit 1 |
| `bench-call-id` | the assistant call ID must match its tool response | `SyntheticToolTests.test_payload_and_matching_call` | Killed, exit 1 |
| `bench-payload` | matching-call framing must retain the measured payload | `SyntheticToolTests.test_payload_and_matching_call` | Killed, exit 1 |
| `bench-user-anchor` | a fresh user turn must preserve the sampled closing | `SyntheticToolTests.test_gemma_append_anchor` | Killed, exit 1 |
| `bench-row-verdict` | each call row records whether a call was actually inserted | `SyntheticToolTests.test_main_call_verdict` | Killed, exit 1 |
| `fixture-tools-schema` | a captured preamble cannot qualify a different tool schema | `CapturedR20Tests.test_uncaptured_tools` | Killed, exit 1 |
| `frame-nonassistant-count` | non-assistant reasoning is absent; the unique-marker and duplicate-marker guards imply this count | `None; survives test_templates` | Survives, exit 0: non-assistant reasoning is absent; the unique-marker and duplicate-marker guards imply this count |
| `bos-len` | one extra piece plus exact bos_ids + plain_ids already implies one BOS ID | `None; survives test_templates` | Survives, exit 0: one extra piece plus exact bos_ids + plain_ids already implies one BOS ID |
| `plain-equal-branch` | when IDs agree, returned bytes come from the already checked plain pieces | `None; survives test_templates` | Survives, exit 0: when IDs agree, returned bytes come from the already checked plain pieces |

## GPU qualification of the combined head

Unrun here. Provision model files through the existing owner workflow on an existing GPU host. Use the full delivered signed local head as SOURCE, clean source, one identical GGUF per node/client/checker case, CUDA on both sides, and sufficient aggregate VRAM for services and checker contexts. Run the three cases sequentially. Do not enable heterogeneous opt-in or change batch/ubatch: 256/256 keeps the SWA ring capacities matched.

SWA depths 512/1536/3072/6144 cross both Glimmer's 2048-token window/2304-cell ring and Gemma's 1024/1280 boundaries. The combined planner causes multi-chunk Gemma calls as needed; only the last chunk carries its fixed window. The Qwen 27B case uses 20k and 40k token targets in context 49152. At the accepted 16-attention/48-recurrent geometry, 40,000 tokens have 1,550,496,020 native bytes across four chunks of at most 10,871 tokens, including one 156,894,356-byte recurrent record. It is well above 512 MiB; 20k also exceeds that cap. Actual chat framing/token IDs and transfer boundaries change envelope sizes; verify observed counts rather than assuming every turn has four chunks. The codec reproduces the 40k arithmetic without a 27B allocation.

Set SOURCE and the three absolute model paths, then run this exact block from the clean combined checkout:

```sh
#!/usr/bin/env bash
# Operator-only recipe. Run on an existing GPU host; no downloads.
set -eu
source_commit=${SOURCE:?Set SOURCE to the delivered full signed head}
git verify-commit "$source_commit"
test "$(git rev-parse HEAD)" = "$source_commit"
git diff --quiet
git diff --cached --quiet
export PYTHONDONTWRITEBYTECODE=1
: "${GLIMMER_MODEL:?Absolute path to an existing Glimmer GGUF}"
: "${GEMMA4_MODEL:?Absolute path to an existing Gemma 4 GGUF}"
: "${QWEN27B_MODEL:?Absolute path to an existing Qwen 3.5 27B GGUF}"
run_case() (
    set -eu
    label=$1
    model_path=$2
    context=$3
    shift 3
    run_path="$PWD/.temp/hybrid-a-r27-gpu/$label"
    test ! -e "$run_path"
    mkdir -p "$run_path/results" "$run_path/tmp"
    export TMPDIR="$run_path/tmp"
    git rev-parse HEAD > "$run_path/source-head.txt"
    trap 'tools/hybrid-bench/setup-client.sh --state "$run_path/client" --stop || :; tools/hybrid-bench/deploy-prefill-node.sh --state "$run_path/node" --stop || :' EXIT
    tools/hybrid-bench/deploy-prefill-node.sh --offline --cuda --ref "$source_commit" \
        --model "$model_path" --ctx "$context" --bench-link --port 18941 \
        --startup-timeout 7200 --state "$run_path/node" --build-lock "$run_path/build.lock"
    tools/hybrid-bench/setup-client.sh --offline --cuda --ref "$source_commit" \
        --model "$model_path" --ctx "$context" --batch 256 --ubatch 256 \
        --startup-timeout 7200 --request-timeout 7200 --prefill-url http://127.0.0.1:18941 \
        --port 18942 --decode-port 18940 --node-info "$run_path/node/node-info.json" \
        --token-file "$run_path/node/client-tokens.json" \
        --state "$run_path/client" --build-lock "$run_path/build.lock"
    python3 tools/hybrid-bench/hybrid_bench.py --client-state "$run_path/client" \
        --scratch "$run_path" --output "$run_path/results" --label "$label-r27" \
        --smoke --modes offload --depths "$@" --check-depths "$@" \
        --check --check-tokens 64 --request-timeout 7200
)
run_case muse-glimmer "$GLIMMER_MODEL" 8192 512 1536 3072 6144
run_case gemma4 "$GEMMA4_MODEL" 8192 512 1536 3072 6144
run_case qwen27b-chunks "$QWEN27B_MODEL" 49152 20000 40000
```

Require exit 0 for each benchmark and retain eight SWA equivalence rows per architecture (fresh and append_chain at four depths), plus both Qwen depths in both modes. Require `same_partition_exact=true`, `same_partition_max_abs_logit_diff=0` and zero same-partition KL across 64 positions; retain whole-prompt, repeat and partition-noise fields separately. Greedy agreement alone does not qualify physical exactness. Below-window SWA append calls must evaluate only the new prompt tokens.

Inspect Qwen's first deep forced-offload transfer and Gemma transfers crossing its chunk limit: multiple chunks, early statuses `staged`, exactly one final `imported`, no early applied/durable advance, final `remote_applied` equal to the complete range, fixed state only in the last body and no fallback. Every body must be at most 536,870,912 bytes; sum actual wire bytes and retain chunk endpoints. Repeat interruption/invalid-final controls with the combined server if changing deployment partitions. Health must advertise `delta_version: 2`, source/build records must name SOURCE and headers must use body format v3. Old body-format refusal must not be relabeled as numerical success.

Keep JSONL, Markdown, sanitized archive/SHA sidecar, command exits, client/node source records and decoder logs. Keep credentials, session databases and native states private. Exit 2 is refusal, 3 numerical failure, 4 partial, 5 uncalibrated; none accepts GPU exactness. An unsupported real file must refuse without bypassing layout fingerprints, unknown-tensor checks or MTP admission.


## r30 - current-main merge and landing evidence

Run: hybrid-a-r30-261102. Signed source merge `23198b4f3fc4537c016ff75ae18c910b9db7d8e9` on `m3-swa-chunk`; first parent `03f1b4fff4c48e2cdb1f54066f070a62ae714215`, second parent exact fetched main `528724c96894aa5f65e98b329317de59aa694afc` (PR #7 fast hashing). `git verify-commit` exits 0, signature G. All 17 introduced commits above main verify G; the subsequent report-only delivery commit is checked separately. No rebase, push, GPU execution or model/tool download.

### Every conflict and its resolution

The merge command returned 1 for exactly two content conflicts; the resolved merge commit returned 0.

| File | Resolution |
| --- | --- |
| `tools/hybrid/hybrid-state.h` | Keep main's `file_hash(path, read_bytes = 65536, backend = nullptr)` signature and the reviewed branch's `require_supported_architecture` declaration. Retain all fixed-state/chunk declarations. |
| `tools/hybrid/README.md` | Union the build-target lists so both `llama-hybrid-check-hash` and `llama-hybrid-check-swa` remain. Retain both hash and chunk/SWA documentation. |

Clean auto-merges were checked: CMake retains every target; `hybrid-state.cpp` contains main's exact complete hash implementation plus the reviewed branch's unchanged remaining logic. No manual implementation edits were needed.

### Reviewed-diff comparison

Common base is `1309bcb8b02e8e8de764f2456e5a6af94d7de476`. Replayed the exact binary patch `git diff --binary 1309bcb8 03f1b4ff` onto main with a separate Git index. The replay reports exactly the same two conflicts; resolving only the declared unions yields the complete expected tree. There are 58 branch-only paths byte-identical to r27, eight main-only paths byte-identical to main, and four overlapping paths: the header, README, CMake and helper implementation. The SHA-NI/portable/CommonCrypto block is byte-identical to main; removing only that hash delta from the merged helper reproduces the entire r27 helper byte-for-byte.

```text
reviewed patch + main + two resolutions -> 6e6c7849b20b826f60387872a2293a8fc551fbdf
actual signed merge tree               -> 6e6c7849b20b826f60387872a2293a8fc551fbdf
git diff --exit-code 6e6c7849b20b826f60387872a2293a8fc551fbdf 23198b4f3fc4537c016ff75ae18c910b9db7d8e9
exit 0; empty diff
```

Thus `git diff 528724c9..23198b4f3fc4537c016ff75ae18c910b9db7d8e9` is exactly the reviewed branch diff carried onto PR #7 with the two listed resolutions. `reviewed-r27.patch`, `merged-vs-main.patch`, `merge-overlap.patch`, `replay-reviewed-patch.log`, `comparison-initial.json` and `comparison-replay.json` show the comparison. The later delta consists only of this requested r30 report appendix; final comparison evidence lists that one documentation path separately.

### CPU validation and actual exits

Evidence root: `.temp/hybrid-a-r30-261102/`. Exact commands/PIDs/exits, times and free-disk samples are in numbered JSON/log files and `command-ledger.json`; native control compile/link commands are also in `controls.json`. Tested source tree `6e6c7849b20b826f60387872a2293a8fc551fbdf` is bound to 329 source/test/configuration hashes and 21 native binary hashes in `validation-identity.json`. Linux x86_64, GCC 15.2.0, existing local CMake 4.4.4, Make, Python 3.14.4; same r27 Release AVX2/FMA CPU configuration with `GGML_NATIVE=OFF`, `GGML_CUDA=OFF`. Builds and controls were niced to 19, serial under the shared model-cache lock, with at most -j4. All tests were niced; TMPDIR stayed on task-local disk and `PYTHONDONTWRITEBYTECODE=1`. Only existing local Qwen 0.8B weights, original golden files and local header fixtures were used.

| Gate / evidence stem | Real exit | Claim / limit |
| --- | ---: | --- |
| `configure-01` | 0 | CPU Release configure |
| `build-01` | 0 | All requested native consumers and checkers |
| `swa-final-02` | 0 | Allocated Gemma/Glimmer caches, original depth/chain cases, fresh/append chunks and legacy refusals |
| `server-component-final-01` | 0 | 50 tests, 235 assertions, including prompt reuse threshold |
| `chunks-q4-01` | 0 | Q4_0 fresh/append chunks, interruption, byte/logit equivalence and refusal/rollback controls |
| `chunks-q8-01` | 0 | Q8_0 same chunk contract |
| `http-01` | 0 | Real HTTP node/server/proxy transfer, interruption, atomic refusal and old-body rejection |
| `proxy-01` | 0 | 123 proxy tests |
| `bench-01` | 0 | 81 benchmark tests; six optional native cases skipped here |
| `mirai-native-01` | 0 | 12 real-header tests, including the six optional native cases |
| `codec-01` | 0 | Codec and malformed envelopes |
| `guards-01` | 0 | Native guards and unchanged rejected state |
| `recurrent-01` | 0 | Malformed recurrent state |
| `append-01` | 0 | Golden append at 512/2048/8192, original 64-token controls |
| `distributions-01` | 0 | Full distributions at 512/2048/8192 |
| `capacity-01` | 0 | Capacity refusal |
| `golden-01` | 0 | Golden checker |
| `integrated-noise-01` | 0 | Focused same-partition and partition-noise checks |
| `integrated-chunk-01` | 0 | Focused multi-chunk integration |
| `integrated-all-01` | 1 | Full inherited integrated driver; existing timing diagnostic remains red |
| `extra-checks-01` | 0 | LRU eviction/replay and actual RAM-budget 503 with zero retained sessions |
| `reference-native-01` | 0 | Reference scoring and native refusal controls |
| `hash-original-01` | 0 | Original four hash tests, automatic dispatch, real local GGUF |
| `hash-original-portable-01` | 0 | Original four hash tests, forced portable |
| `hash-ni-02` | 0 | Eight SHA-NI/portable tests; random boundaries, odd reads, NIST, sparse >4 GiB, invalid sizes and EIO |
| `isa-isolation-01` | 0 | 32 SHA256RNDS2 and 12 each MSG1/MSG2 instructions confined to the attributed function, no global SHA ISA flags |
| `build-hash-sanitized-01` | 0 | ASan+UBSan hash checker build |
| `hash-sanitized-01` | 0 | Six sanitizer cases; linked vendor/engine libraries remain unsanitized |
| `planner-01` | 0 | Fixed-window chunk-width boundary |
| `invariants-01` | 0 | Retained generic bytes, flags, rollback, layout and v3 invariants |
| `static-checks-01` | 0 | Changed Python syntax, ASCII additions, whitespace and model-naming workflow |
| `gpu-recipe-syntax-01` | 0 | Shell syntax only; recipe not executed |
| `compare-merge-tree-01` | 0 | Expected replay tree equals actual signed merge |
| `repro-{512,520,600}-01` | 0 each | Original reviewer public-API F16 unified multi-slot checkpoint source, unchanged; 3,224-byte checkpoint and successful restore at all three depths |

The full inherited integrated driver remains an actual exit 1, with its native timing checker exiting 3. Its same-partition result is exact (KL/max difference 0); fresh and append_chain both reproduce r27: whole-prompt top-1 7/8 and mean KL 0.00852205499931497 (`timing-diagnostic.json`). Whole-prompt numerical acceptance remains the prior CPU diagnostic. Focused chunk/noise, LRU/budget and reference tests were executed independently; their passing results do not change the failed driver outcome. No threshold, model, fixture expectation or engine implementation was changed to turn the full driver green. Original golden files were untouched; task-local copies change only the HYBD envelope to v2/body v3 while preserving native bodies and checksums (`golden-conversion.json`).

PR #7 digests remain exactly `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c` for the 507,154,688-byte local Q4 GGUF, and `01faa4689b7b9762b0c965822d43e087f61558f572609c0b384e6e981600231f` for the 4,294,967,361-byte sparse stream with markers a/b/c at 0, 2,147,483,655 and the final byte. Automatic SHA-NI, forced portable, hashlib and the original test's sha256sum reference agree. Hash throughput measurements were not repeated; this task validates digests and behavior.

Readiness and corrected attempts remain explicit in `early-failures.md`: system CMake/Ninja are absent, so the already installed local CMake and Make were used; the first SWA launcher refused before executing a test because of an overly conservative GiB admission threshold; the already-running first hash sweep was then terminated by that earlier threshold (recorded -15/SIGTERM, wrapper shell exit 241), so it is incomplete, not passing. The complete `hash-ni-02` rerun is green. The first inherited r24 control driver exited 1 because its original scratch-source directory had been removed; the corrected driver uses the exact retained r27 witness copies and exits 0 as `r24-controls-02`. No failed log was overwritten.

The requested disk floor is 28 GB (28,000,000,000 bytes). Minimum sampled free disk across the recorded commands was 29,537,103,872 bytes. Owned build/native scratch cleanup and its final free-disk readback are recorded in `cleanup.json`; command logs, test text evidence, source/control copies and hashes remain. `ty`, `clang-format` and `editorconfig-checker` are unavailable and unrun. No Mac, new GPU, heterogeneous backend/quantization, 5G or process-crash/outage qualification is claimed. No new surface table or separate landing-property suite was supplied; the inherited brief gap remains explicit. No requested row was treated as out of contract.

### Narrowing controls

All 22 r27 targeted controls were rerun and killed, plus all four PR #7 hash controls. Every credited control compile/link exits 0; every expected-red test below actually exits 1. Candidate engine files and libraries were never mutated. No targeted survivor is claimed as a kill.

| Mutant | What it narrows the gate to | Named failing test / diagnostic | Actual exit |
| --- | --- | --- | ---: |
| `chunk-m1` | Early validation publishes nothing | `swa_chunk_transaction_equivalence_and_refusals / SWA early chunk changed cache` | 1 |
| `chunk-m2` | Deferred KV is discarded after late failure | `swa_chunk_late_window_rollback: deferred keys written after rejection` | 1 |
| `chunk-m3` | Window metadata/head roll back | `swa_chunk_late_window_rollback: metadata or neighbor changed` | 1 |
| `chunk-m5` | Final endpoint matches the declared range | `chunk_late_endpoint_rollback` | 1 |
| `merge-layout-gate` | Both chunk APIs require v2 physical windows | `swa_chunk_legacy_window_refused: validate accepted v0` | 1 |
| `swa-f1` | Masked rows do not inflate K/V payload | `swa_masked_rows_metadata_only` | 1 |
| `swa-f2` | Generic bytes stay legacy | `swa_generic_state_legacy_bytes` | 1 |
| `swa-f3` | Saved physical indices are used | `swa_nonprefix_physical_indices` | 1 |
| `swa-ma` | Indices are not compacted | `swa_nonprefix_physical_indices` | 1 |
| `swa-mb` | Masked occupied cells retain metadata | `swa_chunk_single_equivalence: physical layout or bytes differ` | 1 |
| `swa-mc` | Reader and writer use the same retention threshold | `swa_chunk_transaction_equivalence_and_refusals: SWA chunk transaction failed` | 1 |
| `swa-md` | Previous physical version is refused | `swa_previous_layout_version_refused` | 1 |
| `swa-me` | Compacted v0 layout is refused | `swa_legacy_compacted_record_refused` | 1 |
| `swa-mh` | PARTIAL_ONLY checkpoints retain legacy bytes | `swa_checkpoint_legacy_bytes` | 1 |
| `swa-mf` | Session save/load preserves physical layout/head | `swa_physical_cells_and_head` | 1 |
| `chunk-m4` | Lost final ACK restores the original slot | `test_benchmark.ChunkAndLifecycleTests.test_lost_final_ack_restores_original_slot` | 1 |
| `chunk-m6` | Staged chunks publish no durable applied state | `test_benchmark.ChunkAndLifecycleTests.test_staged_chunks_do_not_persist_applied_state_or_drop_checkpoints` | 1 |
| `chunk-m6b` | Staging retains checkpoints | `test_benchmark.ChunkAndLifecycleTests.test_staged_chunks_do_not_persist_applied_state_or_drop_checkpoints` | 1 |
| `main-r2` | Restart cannot import through a stale remote ID | `test_benchmark.BenchmarkProxyTests.test_burst_node_restart_then_offload` | 1 |
| `r24-validation` | Early fixed state is refused per chunk | `FAIL chunk_early_native_state` | 1 |
| `r24-rollback` | Late failure restores old metadata | `FAIL chunk_late_endpoint_rollback: chunk failure changed cells or state` | 1 |
| `swa-n2` | One-token SWA window uses SWA semantics | `prompt_reuse_threshold / one-token SWA window` | 1 |
| `hash-no-sha` | Capable host dispatches to SHA-NI | `FAIL: test_dispatch` | 1 |
| `hash-ignore-switch` | Only explicit portable override selects fallback | `FAIL: test_dispatch` | 1 |
| `hash-bad-round` | SHA rounds match fixed NIST answers | `FAIL: test_nist_vectors` | 1 |
| `hash-ignore-read-error` | Read EIO yields no digest | `FAIL: test_read_error` | 1 |

The template-control suite also exits 0 under its unchanged contract: 40 named kills, three bounded equivalent survivors and one positive identity control. The template mutant table above remains the per-mutant coverage map for this rerun; `template-controls/controls.json` contains the fresh commands and named failures. The three survivors retain exactly the declared bounds in that table (`frame-nonassistant-count`, `bos-len`, `plain-equal-branch`); they are not kills. No claimed property is inferred from agreement between two implementations alone. The four hash mutants add fixed-answer, dispatch/override and real EIO controls; the swallowed-EIO mutant additionally returns the empty-file digest on `/proc/self/mem` (`hash-eio-witness`, actual exit 0), demonstrating the planted violation.

### Prior GPU qualification carried with provenance

The owner supplied r27/r7n acceptance and swa7 GPU qualification on 2026-11-02, PRO 6000, CUDA. Read the existing results under `/home/agent/lanes/hybrid/.temp/swa7-results/` without executing GPU work. `gpu-prior-evidence.json` records exact local JSONL paths/hashes and passed summary/checker exits; `gpu-prior-summary.json` verifies all 20 equivalence cases are same-partition exact with KL/max-logit difference zero. These are prior r27 results, not new GPU execution on the merge.

| Model | Same-partition exact cases | Supplied whole-prompt KL example | Example / partition noise | Largest whole-prompt KL in that result file |
| --- | ---: | ---: | ---: | ---: |
| Glimmer | 8 | 1.61550083e-05 | 0.379628x | 0.00753164053 |
| Gemma 4 | 8 | 0.000459921188 | 0.15893x | 0.000459921188 |
| Qwen 27B | 4 | 0.0133429409 | 1.24383x | 0.0133429409 |

All three result summaries and every checker exit are 0. The supplied examples round to KL 1.6e-5, 4.6e-4 and 0.0133 at 0.38x, 0.16x and 1.24x partition noise. They are example rows, not universal maxima. Qwen's whole-prompt example is above the retained 0.01 absolute KL diagnostic (`numerical_within_bounds=false`), while its same-partition contract and checker pass. No numeric threshold was changed or cross-backend equivalence inferred.

### Landing note

PR title draft: `hybrid : add atomic chunk state with physical SWA layout`.

PR body draft:

> Transfer native hybrid state in bounded chunks, with recurrent and SWA window state only in the final chunk. Preserve physical SWA cell placement and allocator head, validate before publishing, and restore the original slot after rejected or interrupted transfers. Generic sequence/checkpoint bytes retain their reviewed legacy format; incompatible old chunk envelopes are refused.
>
> Preserve the r7n-reviewed r27 changes and merge main 528724c9 (PR #7). The two conflicts retain both the hash API and architecture guard, and both hash/SWA build targets; the complete hash implementation is unchanged from main. Replaying the reviewed patch with these resolutions reproduces the exact source tree.
>
> CPU validation, real exits and prior swa7 GPU evidence are in this report's r30 section. Same-partition comparisons pass, and the inherited whole-prompt CPU timing diagnostic remains explicitly red. Prior PRO 6000 CUDA examples for Glimmer/Gemma 4/Qwen 27B have KL 1.6e-5/4.6e-4/0.0133, or 0.38x/0.16x/1.24x measured partition noise; the report also lists maxima. No r30 GPU execution was performed.
>
> AI assistance: Codex resolved the merge conflicts, verified source scope and reran CPU checks under the contributor's specified design. Local draft for the hybrid-inference fork.

Exact tested source head: `23198b4f3fc4537c016ff75ae18c910b9db7d8e9`. The report-only delivery commit is signed separately; `.temp/hybrid-a-r30-261102/landing-note.md` records that exact final delivery head, title/body draft and clean/signature evidence after the appendix commit. No PR was opened or updated, and nothing was pushed. Ready for review with the stated timing diagnostic and qualification bounds.

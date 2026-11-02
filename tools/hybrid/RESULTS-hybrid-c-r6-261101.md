# SWA physical window transfer: r7e rework

Run: hybrid-c-r6-261101. Branch: `m3-swa-fix-r6`, continuing signed r33 head `cc1f454fc057c3ad897ae42b0b31a1f516ded04b` and preserving its history on `c4e0faae`. Private fork, local signed delivery, no push. Ready for review; production GPU qualification is pending.

## Changes

F1: version-2 SWA records retain every occupied row's physical index, position and sequence metadata. Only rows unmasked at the saved maximum have K/V payload. The reader derives those payload rows from the validated positions, scatters them to the original indices, and restores the allocator head. Masked rows keep finite initialized or stale destination K/V. CPU continuation compares fresh zero-initialized imports and repeated imports against independently computed local state; masked values differ without changing logits. No tolerance changed.

F2: generic ISWA `state_write` no longer adds the physical-layout flag. Ordinary sequence/whole-context state, checkpoints, prompt caches and slot files use the same legacy writer branches and bytes as c4e0faae. Only hybrid range export and the node's explicit save/load helpers request physical records. The internal flag is stripped before the full-attention cache save/load. Ordinary legacy records remain readable. The new multi-slot regression uses two sequences, unified KV, a 1024-token context, 256/256 batches, a four-token window, Q8_0 KV, and a three-token second slot before restore. Both whole-sequence and PARTIAL_ONLY restores preserve the neighbor at depths 10/256/300/512/520/600.

F3: an independent fixture removes positions 0..6 from an eleven-row window cache, leaving physical indices [7,11). Import into an empty cache must preserve those indices, metadata, head and retained tensors. The test runs for both admitted architectures and both unified/non-unified layouts. A reader that uses `find_slot` instead fails this assertion.

N1: SWA record version 2 uses stream tag `0xc0000000`; r33 version 1 used `0x80000000`, and legacy compacted records were unmarked. Hybrid append and node load require version 2. Hybrid append validation and apply refuse v0/v1 with `unsupported sliding-window record version`, preserving both caches and the neighbor. Legacy generic state reads still work. Both old readers refuse v2: c4e0faae sees a stream count above LLAMA_MAX_SEQ; r33 clears bit 31 but the remaining bit 30 still exceeds that bound, before mutation. This reverse-compatibility statement is from inspection of the actual old guards, not an executed old-client binary.

N2/N4: n_swa=1 has ten threshold assertions; the accepted threshold implementation is byte-identical to r33. The attention claim comment now belongs to `inference`, rather than the cell-print helper.

HYBD version, framing, checksums, full-attention chunk representation and chunk staging are unchanged. This does not combine `m3-chunk-state` / `675cc8f7`; that integration remains separate.

## Record sizes

Bytes below are calculated for Q8_0, one populated stream, no cell extensions, maximum occupied ring and the supplied production geometries. These are geometry/format calculations, not measurements from the unavailable production models. The retained tensor-row widths are independently reproduced by the planner test.

For C occupied cells, W retained positions, L allocated window layers and R K/V bytes per token, the SWA record is `32 + 24*L + 16*C + W*R`. The native outer magic/sequence, full-attention section and HYBD envelope are additional. Planner fixed bytes conservatively use `W*R + pad256(W+256)*28 + 16`, with the existing 1 MiB header margin and full-attention token budget applied separately. Allocation still reserves all ring K/V cells in memory.

| Geometry | Window / ring cells | K/V row | Retained K/V | Physical metadata | SWA record | Planner fixed bytes | r33 fixed bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Glimmer, 39 window layers | 2048 / 2304 | 21,216 | 43,450,368 | 36,864 | 43,488,200 | 43,514,896 | 48,946,192 |
| Gemma 4, 50 window layers | 1024 / 1280 | 435,200 | 445,644,800 | 20,480 | 445,666,512 | 445,680,656 | 557,091,856 |

Gemma's planner now admits context 8192 and leaves a safe chunk of 2070 full-attention tokens at 43,520 bytes/token under 536,870,912 bytes including the margin. A 16x enlarged window still refuses with `window exceeds delta budget`. The native synthetic size test independently checks exact record bytes with 273 occupied rows but only four K/V rows; changing the writer back to full masked payload fails it.

## CPU evidence

Evidence root: `.temp/hybrid-c-r6-261101/`. Each numbered gate log has a JSON ledger with exact argv, PID, exit code, timestamps and observed free disk. Build start timestamps include queueing on the shared lock; completion timestamps are observed. `run.py`, `validate.py` and `mutants.py` retain the commands; source/binary manifests bind the evidence. CMake 3.31.10 (existing local tool), GNU C++ 15.2.0, Release CPU, no CUDA/model downloads; all builds nice 19, -j4, shared model-cache build lock. TMPDIR is on disk and Python bytecode is disabled. One owned build tree is used, including incremental controls.

| Gate | Actual result | Evidence |
| --- | --- | --- |
| Configure and candidate builds | Exit 0 | `configure-01`, `build-01`, `build-candidate-02`, `build-legacy-fixtures-03`, `build-restored-01-02` |
| SWA checker with deterministic tiny Muse Glimmer | Exit 0; 18/18 same-partition cases bit-exact, all eight logits at 64 continuation positions, max diff 0 | `swa-candidate-04` |
| Unified multi-slot Q8_0 checkpoint / whole-sequence restores | Exit 0; 12/12, neighbor unchanged | `swa-candidate-04` |
| Original reviewer public-API F16 repro | Exit 0 at 512/520/600; each checkpoint 3,224 bytes | `unified-repro-{512,520,600}-02` |
| Non-prefix indices, metadata-only sizing, legacy byte fixtures, version/position/capacity/index/truncation refusals and rollback | Exit 0 for both architectures/cache layouts; 19 original named refusal/layout tests retained | `swa-candidate-04`, `retained-tests.json` |
| Server component suite | Exit 0; 50 tests, 235 assertions; threshold 44 assertions | `server-component-03` |
| Benchmark suite with native SWA checker enabled | Exit 0; 55 tests, no skips | `bench-unit-03` |
| Proxy suite | Exit 0; 112 tests | `proxy-unit-01` |
| Codec and legacy guard suite | Exit 0 | `codec-01`, `guards-01` |
| Qwen golden append / distributions at 512/2048/8192, capacity rollback, golden checker and live node/proxy integration | Exit 0 each; original golden continuations unchanged; live checksum/lossy-state, timing, chunk-budget, LRU and RAM-refusal controls pass | `append-01`, `distributions-01`, `capacity-01`, `golden-01`, `integrated-01` |
| Code-style workflow, Python syntax, ASCII additions, whitespace and GPU-recipe shell syntax | Exit 0 | `code-style-01`, `gpu-recipe-syntax-01`, task log |

| Mutant | What it narrows the gate to | Named failing test | Actual result / survivor bound |
| --- | --- | --- | --- |
| F1: write all masked tensor rows | Serialized bytes must scale with retained K/V, not ring capacity | `swa_masked_rows_metadata_only` | Exit 1, killed |
| F2: auto-mark every ISWA save | Generic writes must retain legacy serialization | `swa_generic_state_legacy_bytes` | Exit 1, killed; original reviewer repro also fails 3/3 with occupied-cell collisions |
| F3 / r7e M4: ignore saved indices and compact | A non-prefix physical index set must survive import | `swa_nonprefix_physical_indices` | Exit 1, killed |
| N2 / r7e t7: n_swa > 1 | One-token windows must use SWA semantics | `prompt_reuse_threshold`, `one-token SWA window` | Exit 1, killed (10/44 assertions fail) |

All controls compiled successfully (exit 0); expected-red commands above really exited 1. No surviving mutant in this bounded table. Sources were restored after each control, rebuilt, and the candidate SWA/server suites passed. `mutants-02.json` and `mutant-f2-unified-*-02` retain final-checker evidence. The first control round remains separately recorded. The independent legacy-byte fixture was added after the long runtime gates; only the affected checker/control and benchmark suites were repeated, since product sources and the other test/environment identities were unchanged.

One scheduling mistake is preserved: the first standalone reviewer-repro invocation ran before its queued compile had acquired the build lock, exited 127 because the binary was absent, and is not passing evidence (`unified-repro-512.log`). After compile exit 0, all three corrected invocations passed. `a2a post --help` is unsupported; root help and whoami work. ty and editorconfig-checker are unavailable (readiness exit 1), so their full checks are unrun; upstream/all-backend CI is also unrun. No surface table was supplied (brief gap); no requested behavior is declared out of contract. No prior reviewer test was removed. Changes outside core SWA serialization are its planner/tests, node state helper, threshold tests and requested documentation. The optional tiny CPU witness does not establish GPU exactness or the magnitude of production-model noise.

## Exact GPU block

Unrun here. Provision Glimmer and Gemma 4 GGUFs on the GPU host first, pin SWA_HEAD to the delivered signed head, and check out that head cleanly. All three binaries must come from that ref and use the same model file. Run architectures sequentially with fresh state directories. Keep node/client 256/256 partitions: larger client ubatches allocate a different ring. Context 8192 accommodates every rung; depths 512/1536/3072/6144 cross both windows (2048/1024) and both ring capacities (2304/1280).

```sh
# Run on the GPU host with both model files already provisioned.
# Pin SWA_HEAD to the delivered signed head, then use the same ref for all binaries.
: "${SWA_HEAD:?Set SWA_HEAD to the delivered commit}"
test "$(git rev-parse HEAD)" = "$SWA_HEAD"
git diff --quiet
git diff --cached --quiet
run_swa_rework() (
    set -eu
    model_path=$1
    arch_label=$2
    run_path="$PWD/.temp/hybrid-c-r6-gpu/$arch_label"
    test ! -e "$run_path"
    mkdir -p "$run_path/results" "$run_path/tmp"
    export TMPDIR="$run_path/tmp" PYTHONDONTWRITEBYTECODE=1
    git rev-parse HEAD > "$run_path/source-head.txt"
    trap 'tools/hybrid-bench/setup-client.sh --state "$run_path/client" --stop || :; tools/hybrid-bench/deploy-prefill-node.sh --state "$run_path/node" --stop || :' EXIT
    tools/hybrid-bench/deploy-prefill-node.sh --offline --cuda --ref "$SWA_HEAD" \
        --model "$model_path" --ctx 8192 --bench-link --port 18941 \
        --startup-timeout 7200 --state "$run_path/node" --build-lock "$run_path/build.lock"
    tools/hybrid-bench/setup-client.sh --offline --cuda --ref "$SWA_HEAD" \
        --model "$model_path" --ctx 8192 --batch 256 --ubatch 256 \
        --startup-timeout 7200 --request-timeout 7200 --prefill-url http://127.0.0.1:18941 \
        --port 18942 --decode-port 18940 \
        --node-info "$run_path/node/node-info.json" \
        --token-file "$run_path/node/client-tokens.json" \
        --state "$run_path/client" --build-lock "$run_path/build.lock"
    python3 tools/hybrid-bench/hybrid_bench.py --client-state "$run_path/client" \
        --scratch "$run_path" --output "$run_path/results" --label "$arch_label-r6" \
        --smoke --depths 512 1536 3072 6144 --check-depths 512 1536 3072 6144 \
        --check --check-tokens 64 --request-timeout 7200
)
run_swa_rework /absolute/path/to/existing/muse-glimmer-30b.gguf muse-glimmer
run_swa_rework /absolute/path/to/existing/gemma4-31b.gguf gemma4
```

Require benchmark exit 0 and all eight equivalence rows per architecture (fresh and append_chain at four depths), each with `same_partition_exact=true`, `same_partition_max_abs_logit_diff=0`, and zero same-partition KL across 64 continuation positions. Require below-window calls to evaluate only new prompt tokens. Keep repeat/whole-prompt/partition-noise fields separate; top-1 alone and measured-noise-floor success do not accept this run. Exit 2 is refusal/abort, 3 numerical failure, 4 partial, 5 uncalibrated. Return sanitized JSONL, Markdown, archive/sidecar and decoder logs; keep credentials, session databases and native states private.

## Touched files (r6 delta only)

- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`
- `src/llama-kv-cache-iswa.cpp`
- `tools/hybrid/hybrid-state.cpp`
- `tools/hybrid/check-swa.cpp`
- `tools/hybrid-bench/bench_common.py`
- `tools/hybrid-bench/test_bench.py`
- `tests/test-server-component.cpp`
- `tools/hybrid/README.md`
- `tools/hybrid/SWA-STATE-DESIGN.md`
- `tools/hybrid/UNRESOLVED_QUESTIONS.md`
- `tools/hybrid/RESULTS-hybrid-c-r6-261101.md`

Custody at 2026-11-01T21:41:02.521419+00:00: 62 recorded owned child PIDs inactive; no active owned PID remained. Minimum observed free disk across completed gate ledgers was 28,678,180,864 bytes, above the 28 GB floor; these are sampled observations. The task build tree (127,417,075 accounted file bytes) and six validated native delta dumps were removed after hashing/accounting. Existing models, tool installations and other task trees were preserved. Final observed free disk after build removal was 29,107,838,976 bytes. Evidence: `final-audit.json`, `build-cleanup.json`, `native-state-cleanup.json`, source/binary manifests.

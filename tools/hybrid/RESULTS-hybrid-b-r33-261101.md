# SWA reuse and physical state transfer

Run: hybrid-b-r33-261101. Branch: `m3-swa-fix`, parent `c4e0faae269ef47fcbe210f7068f9b64dd35c0ca`. Local signed delivery, no push. CPU only, existing local tools/models, no downloads. Ready for review; production GPU acceptance remains pending.

## Cause and correction

The server's strict `pos_min < threshold` test wrongly rejected a complete origin-zero SWA cache when the old threshold clamped to zero. `server_prompt_reuse_threshold` now clamps SWA to one. The non-SWA/recurrent expression remains exactly `max(0, pos_eval)`, including its zero boundary. The existing server component suite calls the helper without a model for both `has_new_tokens` values, below/at/above the window, complete caches, the first required position, one missing boundary cell and `n_swa=0`.

The CPU reproduction establishes a second defect: window export discarded masked occupied cells and import compacted retained rows through `find_slot`. Positions, sequence membership and each retained K/V row could match while physical indices, allocator head and padded attention shape differed. For a four-token window at prefix 512, the local retained positions 508..511 occupy cells 508..511; the old import puts them in cells 0..3. Query 512 goes to cell 0 locally and cell 4 after import. The unmasked set is still 509..512, but its reduction order changes. At prefix 514 the window already straddles the ring boundary. This is not an off-by-one error in the mathematical window.

Allocation checks reusable cells against the previous maximum plus one, before evaluating the next microbatch. Export runs after the final microbatch. At endpoint p1, the required exported window is [max(0,p1-n_swa),p1); the next query needs n_swa-1 earlier positions. Those boundaries are unchanged.

The fix preserves all occupied SWA sequence cells, their physical indices and the allocator head, including masked rows that preserve padded KV shape. The existing mirrored-layout scatter reader places tensors at those indices; it refuses collisions rather than relocating rows. Node sequence save/load uses the same representation. Full-attention append and recurrent data remain unchanged. Planner fixed bytes now reserve `pad256(window+256) * (window_row_bytes+28) + 16`, plus the existing header margin.

The native window stream-count word marks this representation with bit 31. Each populated stream adds capacity, head and a bounded unique index list before its ordinary metadata/tensors. Required-window completeness, retained position continuity, indices, head and tensor layout are validated before deferred writes commit. Apply/post-check failures restore both caches and allocator heads. New readers still accept legacy records; old readers reject marked records. Rebuild node, decoder and checker together. Whole-cache records with seq_id=-1 retain their existing format.

## Evidence and limits

`R=.temp/hybrid-b-r33-261101`; full commands, PIDs and logs are retained there. CMake 4.4.4, GNU C++ 15.2.0, Release, CPU, Q8_0 KV, flash attention, two compute/batch threads, 256/256 batches. Builds use `nice -n 19`, `-j2` and the shared lock. TMPDIR is on disk and Python bytecode is disabled.

No production SWA GGUF is installed. The existing checker now optionally generates a deterministic four-layer, width-64 Muse Glimmer GGUF with an eight-token vocabulary and a four-token window. It removes its own GGUF after use and refuses to overwrite an existing file. Fresh imports and seven-token append chains use depths 3/4/5/19/273/510/512/514/531. Every chain boundary compares all retained physical indices, positions, sequence metadata, allocator heads and K/V bytes against independent local prefill. Short contexts intentionally have smaller capacities; only the capacity field is normalized in that comparison. Every case then compares all eight logits bytewise at 64 teacher-forced positions.

| Gate | Actual result | Evidence |
| --- | --- | --- |
| Old transfer CPU witness | Exit 1 as required; six ring-wrap cases non-exact, max difference 5.36442e-7 | `swa-inference-before-01.log`, `baseline-lib/SHA256SUMS` |
| Fixed CPU witness | Exit 0; all 18 cases bit-exact, max difference zero | `swa-inference-after-03.log` |
| Threshold test | Exit 0; 34 assertions | `threshold-01.log` |
| Server component suite | Exit 0; 50 tests, 225 assertions | `server-component-01.log` |
| Benchmark suite with native SWA geometry | Exit 0; 54 tests, no skips | `bench-unit-04.log` |
| Proxy suite | Exit 0; 21 tests | `proxy-unit-01.log` |
| Live benchmark/checker controls | Exit 0; fresh/append checks, corrupt/lossy deltas, timing, defaults, chunk budget, LRU replay and RAM refusal | `integrated-01.log` |
| Golden append/distribution and capacity checks | Exit 0; 512/2048/8192 prompts, 64/64 same-partition continuation and byte-preserving capacity refusal | `native-checks.json` |
| Codec and legacy guards | Exit 0; non-SWA positive and all existing SWA refusal cases | `codec-01.log`, `guards-01.log` |

The old transfer witness ran before any engine edits, so it exercises the transfer code common to c4e0faae and threshold-only probe 6747331d. Its maximum error is small on this synthetic CPU model. It proves a physical-placement defect; it does not independently prove the magnitude or every cause of the 52-layer GPU mismatch. The GPU block below must verify that correction on both actual files. No checker tolerance or acceptance rule changed.

| Control | What it narrows the gate to | Named failing test | Result |
| --- | --- | --- | --- |
| Old SWA zero clamp | Complete origin-zero cache must remain reusable | `prompt_reuse_threshold` | Old extracted production helper exits 1; fixed helper exits 0 |
| Old window compaction | Same positions/K/V bytes alone do not establish exact logits | `swa_beyond_window_same_partition_logits` | Old CPU engine exits 1; fixed engine exits 0 |

There are no surviving controls in this table. Exact malformed-layout errors are tested for capacity, head, duplicate/out-of-range indices, incomplete windows and duplicate/out-of-range positions in validation and apply modes. A post-check explicitly observes a changed physical head before rejection and checks complete rollback. All existing named synthetic tests remain present. Their original sequence-owned byte comparisons exclude the shared stream head; the new physical-state checks include it.

Configure 01 exits 0. Build 01 exits 2 because the new fixture lacked `llama-ext.h`; the driver exits 1 and the include was corrected. Subsequent builds exit 0. The existing-output fixture exits 1 with its intended refusal and preserves the original bytes. `ty` and `editorconfig-checker` are unavailable, so those tools are unrun. Repository code-style workflow, changed Python syntax, added-line ASCII and whitespace checks exit 0. Full upstream/all-backend CI is unrun. No surface table was supplied (brief gap); no requested behavior is declared out of contract. Independent review and production GPU qualification remain open.

Gemma results were observed at 2026-11-01 20:36 UTC (`R/gemma-status.json`, `R/gemma-probe-rows.json`): probe 6747331d, window 1024, benchmark exit 3. Depth 517 passes bit-exact in both modes; depths 1598/3134/6206 fail in both modes, same-partition KL 3.8e-5..3.2e-4 and maximum logit differences 1.07..7.09 despite 64/64 top-1. The supplied Glimmer archive likewise fails beyond window 2048. These independently repeated threshold-only failures support the need to preserve physical state; neither probe accepts this new head on GPU.

## Exact GPU block for the orchestrator

Unrun here. Use the signed delivered head for all three binaries, with the same existing GGUF file on both sides, one architecture at a time. Use a fresh state directory. The node, decoder and checker model copies/contexts must fit together. Keep homogeneous same-CUDA acceptance bit-exact, including every checked logit; top-1 or measured-floor success alone does not accept the run.

```sh
run_swa_fix() (
    set -eu
    model_path=$1
    arch_label=$2
    run_path="$PWD/.temp/hybrid-b-r33-gpu/$arch_label"
    test ! -e "$run_path"
    mkdir -p "$run_path/results" "$run_path/tmp"
    export TMPDIR="$run_path/tmp" PYTHONDONTWRITEBYTECODE=1
    git rev-parse HEAD > "$run_path/source-head.txt"
    trap 'tools/hybrid-bench/setup-client.sh --state "$run_path/client" --stop || :; tools/hybrid-bench/deploy-prefill-node.sh --state "$run_path/node" --stop || :' EXIT

    tools/hybrid-bench/deploy-prefill-node.sh --offline --cuda \
        --model "$model_path" --ctx 8192 --bench-link --port 18941 \
        --state "$run_path/node" --build-lock "$run_path/build.lock"
    tools/hybrid-bench/setup-client.sh --offline --cuda \
        --model "$model_path" --ctx 8192 --batch 256 --ubatch 256 \
        --request-timeout 7200 --prefill-url http://127.0.0.1:18941 \
        --port 18942 --decode-port 18940 \
        --node-info "$run_path/node/node-info.json" \
        --token-file "$run_path/node/client-tokens.json" \
        --state "$run_path/client" --build-lock "$run_path/build.lock"
    python3 tools/hybrid-bench/hybrid_bench.py --client-state "$run_path/client" \
        --scratch "$run_path" --output "$run_path/results" --label "$arch_label-r33" \
        --smoke --depths 512 1536 3072 6144 --check-depths 512 1536 3072 6144 \
        --check --check-tokens 64 --request-timeout 7200
)

run_swa_fix /absolute/path/to/existing/gemma4-31b.gguf gemma4
run_swa_fix /absolute/path/to/existing/muse-glimmer-30b.gguf muse-glimmer
```

Require benchmark exit 0 and all eight equivalence rows per architecture (fresh and append_chain at four depths) with `same_partition_exact=true`, `same_partition_max_abs_logit_diff=0`, and zero same-partition KL. Below-window requests must perform only new-token prompt work. Retain repeat/whole-prompt/noise fields separately; do not loosen the exact rule. Exit 2 is refusal/abort, 3 numerical failure, 4 partial and 5 uncalibrated. Return sanitized JSONL, Markdown, archive/sidecar and decoder log; keep credentials, databases and native state private.

Final custody audit: 27 recorded child PIDs inactive; binaries/source hashes are in `R/final-audit.json`. The task-owned build tree and native delta fixtures were hashed/accounted for and removed (`R/cleanup.json`); pre-existing build trees and local models were preserved. Minimum observed free disk was 26,977,726,464 bytes, above 25 GiB. The stopped disk-watch process exits 143 by the recorded owned PID.

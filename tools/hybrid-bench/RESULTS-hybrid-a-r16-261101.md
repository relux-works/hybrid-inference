# Sliding-window hybrid transfer: hybrid-a-r16-261101

Ready for review. Local branch `m3-swa`, implementation commit `d83e4ed3b053bca4d70edd0b878f1f3057942b23` (signature G; `git verify-commit` exit 0). Started from public main `0124e89a933872b54bd445921b9ac2e1c9c43a09`: the requested fetch had advanced beyond `81dc3c5c`. No push, PR, model download, package installation or GPU execution was performed.

## Implemented behavior

The source study was written to [SWA-STATE-DESIGN.md](../hybrid/SWA-STATE-DESIGN.md) before implementation. The examined upstream ISWA, model and hparam sources match fork base `e0f142d5`. Gemma 4 and Muse Glimmer own independent full-attention and window caches, with no recurrent section. The design records native framing, metadata, masking/eviction, physical ordering, Gemma source-layer reuse and separate full/window head sizes.

The native range record now contains full-layer KV for `[p0,p1)` followed by the entire live window `[max(0,p1-window),p1)`. Initial and append imports use the existing native record format. HYBD still has one native section; its section and body checksums cover both cache records. No checksum algorithm, native model-SHA admission, heterogeneous-quant opt-in or calibration rule changed.

The public append API parses both complete records without writes first. It checks exact position sets/counts, window/full endpoint agreement, source metadata, tensor layout and complete buffer consumption. Wrapped window order is allowed. The coordinated ISWA import appends full KV and replaces window KV. Both caches' physical cell metadata and allocator heads are copied before apply. Host tensor writes stay deferred until success; an apply exception discards them and restores the metadata copies. The ordinary cache's destructive failure clearing is suppressed in this coordinated transaction. Bare SWA-cache append remains refused. Existing Qwen recurrent rollback is retained.

Native and GGUF-derived fingerprints bind the window/type, per-layer sliding mask, actual head dimensions, KV-head counts and source-layer reuse. Qwen35's version-1 fingerprint remains unchanged. Launchers and the node admit `qwen35`, `gemma4` and `muse-glimmer`, and refuse other architectures. Geometry counts allocated sources once. `kv_bytes_per_token` describes full layers; `window_state_bytes` includes window-layer rows and conservative 24-byte cell metadata. `delta_geometry.window_bytes` is a fixed chunk-planning cost; policy `state_bytes` includes that cost too. Allocation sizing includes window microbatch/pad256 headroom. Node/checker contexts explicitly request reduced window allocation.

The benchmark schedule, delta envelope and numerical evaluator interface are unchanged. The checker has only a context allocation flag change. The complete local pending PR 9 diff, `0124e89a93..d5a597a15d`, applies cleanly to this source, including its logit checker, launcher, tests and docs. PR 9 was not merged into this branch. Full same-partition logit acceptance still requires that separate change. No separate task evaluator is present in this checkout; its real-model task-quality run is unverified, and no evaluator implementation or output schema was changed here.

## Executed checks

All commands ran as standalone child processes at `nice -n 19`. Evidence root `R=.temp/hybrid-a-r16-261101/`; `R/checks.json` records commands, timestamps, PIDs and real exit codes. `R/tested-source.json` and `R/binary-identities.json` bind source and runtime identities. Test scratch uses `R/tmp/` in the final run.

| Check / final log | Exit | What it establishes |
|---|---:|---|
| `build/bin/llama-hybrid-check-swa`, `swa-native-04.log` | 0 | 56 checks across Gemma/Muse synthetic layouts and unified/separate streams: full append, wrapped whole-window export/replacement, short initial import, neighbor preservation and 12 negatives per configuration |
| `HYBRID_SWA_CHECKER=$PWD/build/bin/llama-hybrid-check-swa python3 tools/hybrid-bench/test_bench.py`, `python-unit-04.log` | 0 | 26 tests; actual GGUF metadata parsing for both prefixes, per-layer sizing/source reuse, native/Python fingerprint equality, fixed-window planning bounds and malformed/unknown layout refusals |
| `build/bin/llama-hybrid-check-codec`, `codec-02.log` | 0 | Existing framing/checksum/model/layout/header/body bounds and positive controls |
| `build/bin/llama-hybrid-check-guards MODEL GOLDEN`, `guards-01.log` | 0 | Original non-SWA positive and bare standard/chunked/symmetric SWA append refusals retained |
| `python3 tools/hybrid/test_recurrent_append.py --library build/bin/libllama.so --malformed`, `recurrent-02.log` | 0 | Existing Qwen valid append and malformed recurrent/native header cases preserve both sequences |
| `build/bin/llama-hybrid-check-append MODEL GOLDEN`, `append-512-02.log` | 0 | Real Qwen depth-512 append, native/envelope refusals and 64/64 greedy agreement |
| `python3 tools/hybrid-bench/test_integrated.py --prefill-binary $PWD/build/bin/llama-hybrid-prefill --server-binary $PWD/build/bin/llama-server --model MODEL --artifacts $PWD/.temp/hybrid-a-r16-261101/integrated-02`, `integrated-02.log` | 0 | Actual node/decoder/proxy: native chunk slope, timing with/without link, fresh/append equivalence, LRU replay, budget-503 abort and session release |
| `python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py'`, `proxy-unit-02.log` | 0 | All 68 proxy tests, including a new fixed-window maximum/one-over startup control |
| Python compile, `python-compile-01.log` | 0 | All changed Python modules compile |
| PR 9 full patch apply check, `pr9-compat-05.log` | 0 | Pending branch's complete diff applies without conflict |
| Whitespace and source/prior-test/PID audit, `whitespace-03.log`, `audit-01.log` | 0 | No removed existing Python test names, added code ASCII, exact source/binary hashes and no live recorded PIDs |

`MODEL=/home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf`; `GOLDEN=/home/agent/Werkstatt/hybrid-models/golden/prompt-512`. These existing local files were reused without downloads. Native synthetic tests allocate real Q8_0 K/V caches without loading model weights or running inference graphs. The synthetic IO fixture follows the production host reader's deferred-write/discard contract; real ISWA context-wrapper inference and device execution remain integration work.

### Bounded mutation controls

Native mutants are task-local C++ interposition libraries used only with the owned checker process through `LD_PRELOAD`. They do not modify the repository or installed library. The Python mutant changes only its own test process. No mutant survived this bounded set.

| Mutant | What it narrows the gate to | Named failing test | Actual exit / evidence |
|---|---|---|---|
| Disable both metadata restore callbacks | Recoverable apply failure must restore both caches, rather than merely discard tensor writes | `failed_apply_restores_both_caches_exactly` | 1, expected red; `mutant-rollback-red-01.log` reports changed cache bytes/cells |
| Export window only over `[p0,p1)` | Export must carry retained window positions preceding the new full-cache range | `full_append_whole_window_replace_export` | 1, expected red; `mutant-window-red-01.log` reports wrong full-range/whole-window bytes |
| Ignore `window_bytes` in planning | Fixed window must constrain the reserved body budget | `test_swa_metadata_geometry_and_chunk_budget` | 1, expected red; `mutant-policy-red-01.log`: 553442368 exceeds 536870912 |

Limits: these controls do not establish process-crash recovery, fatal backend/OOM atomicity, arbitrary concurrent mutation of the source buffer, or all sliding-window architectures. No surface table or separate landing-property suite was supplied. Original reviewer tests remain; this is a producer self-review using scope, security, general, native API and test checklists, with independent review still pending. No accepted delivery/landing is claimed.

### Build and diagnostic record

Every configure/build, including the two tiny mutant libraries, used `/home/agent/Werkstatt/hybrid-models/.build.lock`, `nice -n 19` and the existing 25 GiB admission/monitor helper. CMake 4.4.4, existing Release CPU tree, `-j2`, UI/prebuilt UI/CURL disabled. No model or UI download was requested. The minimum observed free space across these operations was 36,673,900,544 bytes, above the 26,843,545,600-byte floor. Final product build tree: 132,606,201 bytes. Exact build commands and PIDs are in the monitor logs.

```sh
flock /home/agent/Werkstatt/hybrid-models/.build.lock nice -n 19 \
  python3 .temp/hybrid-a-r3-261030/build_guard.py --minimum-start-gib 25 --abort-gib 25 \
  --log-prefix .temp/hybrid-a-r16-261101/build-05 -- \
  .temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake --build build -j2 --target \
  llama-hybrid-check-swa llama-hybrid-prefill llama-hybrid-bench-check \
  llama-hybrid-check-append llama-hybrid-check-guards llama-hybrid-check-codec llama-server
```

Configure exits 0. Build 01 exits 2 for a missing `<stdexcept>` include, corrected before build 02. Builds 02-06 exit 0; build 06 recompiles only the strengthened synthetic test. Mutant library builds exit 0. Configure warns that the extra `HF_ENABLED=OFF` variable is unused; UI downloads are disabled through the actual UI flags. The pre-existing checker signedness warning remains.

Synthetic run 01 exits 1 because its expected export omitted empty non-unified streams; the fixture was corrected, and subsequent runs pass. An initial PR 9 checker apply check exited 1 because the context flag overlapped a patch hunk; its placement was corrected. Later full-patch checks exited 1 for the new README section at EOF; moving that section preserved both docs changes and the final full-patch check exits 0. These failed diagnostics are retained rather than counted as passing gates. `ty` and `editorconfig-checker` are not installed (readiness exits 1), so those checks are unrun. Full upstream CI, all backends, macOS, large-model inference and task-quality evaluation are unrun.

## Commands for the rented GPU

Unrun here. Provision one local GGUF of each architecture with keeper approval, and use the same exact file for node and client. No download command is included. Start with the combined source after PR 9 is integrated for full logit-level same-partition checks; this branch's checker retains the existing pre-PR-9 acceptance behavior. The commands otherwise use existing operator interfaces unchanged.

The example runs node, decoder and a sequential numerical checker on one CUDA host. Their model copies and checker contexts must fit together; launcher sizing covers each service independently and cannot guarantee aggregate checker VRAM. Use separate hosts/tunneling as described in the operator guide if needed. Run the architectures sequentially. Context 8192 and depths through 6144 cross both the 1024 and 2048 windows; small tool turns and append-chain checks exercise nonzero-position window replacement.

```sh
run_swa() (
    set -eu
    model_path=$1
    arch_label=$2
    run_path="$PWD/.temp/hybrid-a-r16-gpu/$arch_label"
    mkdir -p "$run_path/results"
    trap 'tools/hybrid-bench/setup-client.sh --state "$run_path/client" --stop || :; tools/hybrid-bench/deploy-prefill-node.sh --state "$run_path/node" --stop || :' EXIT

    tools/hybrid-bench/deploy-prefill-node.sh --offline --cuda \
        --model "$model_path" --ctx 8192 --bench-link --port 18941 \
        --state "$run_path/node" --build-lock "$run_path/build.lock"
    tools/hybrid-bench/setup-client.sh --offline --cuda \
        --model "$model_path" --ctx 8192 --request-timeout 7200 \
        --prefill-url http://127.0.0.1:18941 --port 18942 --decode-port 18940 \
        --node-info "$run_path/node/node-info.json" \
        --token-file "$run_path/node/client-tokens.json" \
        --state "$run_path/client" --build-lock "$run_path/build.lock"
    python3 tools/hybrid-bench/hybrid_bench.py --client-state "$run_path/client" \
        --scratch "$run_path" --output "$run_path/results" --label "$arch_label-window" \
        --smoke --depths 512 1536 3072 6144 --check-depths 512 1536 3072 6144 \
        --check --request-timeout 7200
)

run_swa /absolute/path/gemma4-31b.gguf gemma4
run_swa /absolute/path/muse-glimmer-30b.gguf muse-glimmer
```

Do not pass `--hetero-quant` for this homogeneous control. Retain the generated `node-info.json`/client sizing and verify architecture, layer-source mapping and nonzero `window_state_bytes`. The default critical chunk size uses that fixed window budget. Expected benchmark exit is 0 for homogeneous same-CUDA checks; 2 is a refusal/abort, 3 is a numerical failure, 4 is partial, and 5 is uncalibrated. Keep those statuses truthful. With PR 9, retain its actual same-partition full-logit/repeat-floor fields, rather than treating greedy agreement alone as logit equivalence. Return only the sanitized archive and SHA-256 sidecar, never token/config/database/native-state files.

Real Gemma 4/Muse Glimmer logit agreement, throughput, long-context execution and task quality remain open qualification items, also recorded in `tools/hybrid/UNRESOLVED_QUESTIONS.md`. The CPU synthetic results above are not a substitute for these runs.

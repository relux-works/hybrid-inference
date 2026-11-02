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

## r17: rebase and review notes

Ready for review, run `hybrid-a-r17-261101`. Source candidate `e1abcd9f8cf3e550a6b1829e4f40ae7365fc218b`, branch `m3-swa`, based on exact public main `81dc3c5c857e1914cacfe3ef12e4fb02bb57fa21`. The old r6i ACCEPT was bound to `1bc72963`; it is not presented as independent acceptance of this new head. No push, PR creation, downloads, package installation or GPU execution. All actual model runs used the existing local Qwen3.5 0.8B Q4_0 file, CPU and `nice -n 19`.

### Rebase evidence

The requested `git fetch /home/agent/Werkstatt/hybrid-prototype.git main` succeeded but returned stale local bare `refs/heads/main` at `0124e89a`. The same bare repository's `refs/remotes/origin/main` held the requested `81dc3c5c`. A read-only `git ls-remote --heads git@github.com:relux-works/hybrid-inference.git main` independently confirmed public main at `81dc3c5c`. No shared bare ref was changed. Fetching its remote-tracking main and `m2-noise-floor` supplied both exact objects. Logs: `R/fetch-01.log`, `R/fetch-02.log`, `R/public-main-01.log`, where `R=.temp/hybrid-a-r17-261101`.

```sh
git rebase --gpg-sign --onto 81dc3c5c857e1914cacfe3ef12e4fb02bb57fa21 \
  0124e89a933872b54bd445921b9ac2e1c9c43a09 m3-swa
git range-diff --no-color \
  0124e89a933872b54bd445921b9ac2e1c9c43a09..1bc72963f07decd3176fe31b02f7dd069d2b19b5 \
  81dc3c5c857e1914cacfe3ef12e4fb02bb57fa21..e1abcd9f8cf3e550a6b1829e4f40ae7365fc218b
```

Both commands exited 0; no rebase conflicts. Range-diff (`R/range-diff-02.log`):

```text
1:  d83e4ed3b0 = 1:  2adb2999b1 hybrid : transfer full KV ranges with complete sliding windows
2:  1bc72963f0 = 2:  137566ba0b hybrid : record SWA transfer checks and GPU qualification commands
-:  ---------- > 3:  e1abcd9f8c hybrid : guard server imports and roll back failed SWA post-checks
```

The two `=` rows preserve the reviewed patches while re-signing their new objects. All three introduced commits have signature G, `git verify-commit` exit 0 and the requested `Co-authored-by: Codex gpt-6.1-sol <noreply@openai.com>` trailer. Main's launcher startup timeout/zombie handling and task evaluator/client `--batch`/`--ubatch` changes are retained.

### N1 and N2

N1: the context's existing post-apply consumed-byte/endpoint check now executes through a callback inside the ISWA apply/restore scope. Failure discards deferred tensor writes and restores both caches' metadata and allocator heads. The validation pass does not call the callback. The ordinary non-SWA path retains its check and recurrent restore behavior.

`postcheck_endpoint_rollback` injects a wrong expected endpoint after both caches changed. `postcheck_length_rollback` leaves a trailing byte after both records. Each requires the intended exception, then compares the whole cache and neighbor sequence with their original bytes. The nearby valid append calls the callback, imports successfully, preserves the neighbor and exports the expected complete window. Validation explicitly proves it does not invoke the apply callback. These are synthetic allocated-cache tests, not real-model context-wrapper inference; they do not add observability of allocator heads (r6i N7).

N2: `SLOT_IMPORT_DELTA` calls `hybrid::require_supported_architecture()` before parsing delta data or changing memory/token bookkeeping. This same helper gates model identity. Only `qwen35`, `gemma4`, `muse-glimmer` pass. The native `server_import_architecture_admission` test pairs all three valid architectures with explicit refusals of unknown, llama, gemma2, gemma3, gpt-oss, cohere2 and qwen35moe, checking the exact refusal message. The general libllama standard-ISWA API is not broadened or restricted by this server-specific admission policy.

An additional task-local interposition control sends an actual HTTP import request to the real Qwen decoder. Without the control, the admitted architecture reaches the delta parser and returns `truncated delta`; with the control, the helper evaluates a synthetic Gemma 3 model and returns `unsupported hybrid architecture` before parsing. Both return HTTP 400 and preserve slot information plus saved native state byte-for-byte. It does not load a real Gemma 3 file. Existing integrated tests separately exercise successful valid Qwen delta imports.

### Checks and controls

`R/checks.json` records standalone commands, times, PIDs and exit codes. All native/model tests used `nice -n 19`. Build logs and monitor logs identify exact commands, compiler child PIDs and disk admission. Final source/binary hashes are recorded in `R/tested-source.json` and `R/binary-identities.json`.

| Check / log under R | Exit | Result and bound |
|---|---:|---|
| `build-01`, `build-02`, `build-03` | 0 | Named CPU targets, shared `/home/agent/Werkstatt/hybrid-models/.build.lock`, existing 25 GiB admission/monitor helper, CMake 4.4.4, Release, `-j2`, no UI/CURL downloads. Last two rebuild only strengthened tests. |
| `swa-native-02.log` | 0 | 68 PASS across Gemma/Muse synthetic layouts, unified/separate streams; includes new architecture and both post-check rollback tests. |
| `python-unit-02.log` | 0 | 31 benchmark tests; native/Python identity, SWA bounds, retained launcher lifecycle and client partition tests from main. |
| `task-eval-01.log` | 0 | 11 task evaluator unit tests from main; no evaluator changes in r17. |
| `codec-01.log`, `guards-01.log` | 0 | Codec controls, non-SWA positive control and bare standard/chunked/symmetric SWA append refusals retained. |
| `recurrent-01.log` | 0 | Real Qwen valid/malformed append; both native sequences preserved on refusal. |
| `append-512-01.log` | 0 | Existing local Qwen golden fixture, 64/64 greedy agreement and import controls. |
| `integrated-01.log` | 0 | Real node/decoder/proxy, fresh/append imports, timing/link variants, LRU replay and budget refusal/session release. |
| `proxy-unit-01.log` | 0 | All 68 proxy tests. |
| `server-route-control-01.log` | 0 | Actual server HTTP gate refusal and admitted-parser control, unchanged slot/native bytes. |
| `report-syntax-01.log`, `report-metadata-control-01.log` | 0 | All four r17 shell blocks parse with `bash -n`; the exact offline metadata snippet runs on the local Qwen control without NumPy. GPU commands remain unexecuted. |
| `mtp-analysis-01.log` | 0 | Synthetic positive-nextn metadata retains raw Python counts (8 vs native 7, 4 vs native 3); local Qwen metadata control is 24/0/24. Does not load a real SWA/MTP model. |
| `pr9-merge-tree-02.log` | 0 | Exact PR 9 head and source candidate merge cleanly; details below. |
| `staged-whitespace-01.log`, `audit-02.log` | 0 | Whitespace, source/binary identity, prior test retention, ASCII additions and stopped recorded PIDs. |

Bounded mutants are intentionally failing commands, not passing gates:

| Mutant | What it narrows the gate to | Named failing test | Exit / evidence |
|---|---|---|---|
| Run the post-check after the ISWA restore scope | A post-check exception must roll back metadata as well as discard writes | `postcheck_endpoint_rollback` | 1, expected red; `mutant-postcheck-red-02.log` |
| Admit Gemma 3 in the shared architecture helper | Standard SWA capability alone must not grant prototype import admission | `server_import_architecture_admission` | 1, expected red; `mutant-admission-red-02.log` |

No survivors in this two-mutant set; this is not exhaustive mutation coverage. Both mutant builds also used the shared lock, nice priority and disk guard. Initial admission-mutant build exited 1 because the link command used `-llama` instead of the library's actual name; the first attempted run exited 127 because that binary did not exist. Explicit versioned library paths corrected the build (`mutant-admission-build-02`, exit 0), then the actual mutant failed for the intended admission assertion. The failed setup is not counted as a killed mutant. Initial audit 01 exited 1 because it compared the intentionally uncommitted report against the source commit; audit 02 excludes that documentation delta and checks the unchanged tested source. `GGUFReader` readiness exited 1 because NumPy is absent; no package was installed and no tensor-inventory check using that reader is claimed. The standard-library metadata checks below work without it. `editorconfig-checker` is absent (readiness 1), so that check is unrun. Native builds retain the pre-existing checker signedness warning. No supplied leaf surface table or separate landing-property suite; no prior reviewer test was removed. Source delta is confined to the SWA transaction, hybrid helper/checker, server import gate and their documentation. The pre-existing review's real-SWA/GPU, fatal-device/OOM and allocator-head-observability bounds remain open.

### N3: inspect nextn/MTP before GPU qualification

The full explanation and bounded proposal are in [SWA-STATE-DESIGN.md](../hybrid/SWA-STATE-DESIGN.md#nextn--mtp-files). Current behavior:

| File characteristic | Native loader / fingerprint | Launcher outcome |
|---|---|---|
| No nextn declaration, ordinary trunk tensors | Effective layers equal raw `block_count`; normal supported-layout checks | No MTP-specific refusal; other geometry/identity/budget gates still apply. |
| Positive `nextn_predict_layers`, native trunk load succeeds | Native fingerprints `block_count - nextn`; Python fingerprints/sizes raw `block_count` | Node refuses `native/GGUF layout fingerprint mismatch` after health, stops its owned process, retains private failed logs. |
| Additional nextn/MTP tensors not visited by Gemma/Muse loaders | `done_getting_tensors()` can refuse `wrong number of tensors` before health | Startup fails before any fingerprint comparison; default `load_mtp=false` alone does not account for unvisited tensors. |
| MTP-only, missing trunk or incompatible fused dimensions | Missing/dimension/tensor-count load failure | Refused; not an optional-tensor-only compatibility case. |

The homogeneous client launch checks full-file SHA and does not itself establish native/Python fingerprint equality; launch and check the node first. Full-file SHA includes every byte of optional tensors and metadata. Matching tensor geometry is not model-weight equivalence.

Run this standard-library-only inspection for the two actual files before launch. It reads metadata/tokenizer strings, not weight payloads, and makes no downloads:

```sh
PYTHONPATH=tools/hybrid-bench nice -n 19 python3 - \
  /absolute/path/gemma4-31b.gguf /absolute/path/muse-glimmer-30b.gguf <<'PY'
import json
import sys
import bench_common as common
for path in sys.argv[1:]:
    meta = common.model_metadata(path)
    arch = meta['general.architecture']
    raw = meta.get(arch + '.block_count')
    nextn = meta.get(arch + '.nextn_predict_layers', 0)
    print(json.dumps(dict(file=path, architecture=arch, block_count=raw,
        nextn_predict_layers=nextn, native_effective_layers=raw-nextn,
        shared_kv_layers=meta.get(arch + '.attention.shared_kv_layers', 0)), sort_keys=True))
    try:
        identity = common.model_identity(path)
        print(json.dumps(dict(python_layers=identity['model_layout']['layer_count'],
            python_fingerprint=identity['layout_fingerprint']), sort_keys=True))
    except common.Refusal as error:
        print('Python layout refusal:', error)
PY
```

A positive nextn count flags the layer-count mismatch even if there are no optional MTP tensor payloads. This inspection does not prove the optional tensor inventory is complete. Keep the subsequent native startup log: distinguish missing/dimension/unaccounted tensors before health from the fingerprint comparison after health. The current cleanup wrapper reports a generic `startup failed (Refusal)` for a post-health mismatch and preserves only `*.log`, not the health response or node-info; do not expect it to echo the internal mismatch string or retain credentials/configs. Correlate the offline counts with the native log rather than assuming the final launcher error identifies the stage. Check raw-length layer masks/head arrays, Gemma's `raw block_count - shared_kv_layers` source boundary, effective trunk presence and every extra tensor before calling a refusal optional-only. No such real-file inventory is available on this host.

If either real file is refused solely because it includes an unused optional nextn set, the proposed follow-up is explicit native accounting/shape validation and skipping of that architecture's recognized nextn tensors with MTP disabled, plus Python identity/sizing over the effective trunk. Validate raw array lengths, filter inactive nextn layers after the native Gemma source-boundary calculation, retain full-file SHA and exact fingerprint equality, and add complete/trunk-only/missing-trunk/unknown-extra/MTP-only fixtures with real-file equality evidence. No blanket partial load, arbitrary extra-tensor skipping, GGUF rewriting or fingerprint bypass. This is a proposal, not a fix implemented by r17; actual tensor names/dimensions are needed to settle that follow-up scope.

### N4: pinned source commands for the rented GPU

These replace the historical unpinned r16 GPU recipe above. Unrun here. This exact candidate deliberately excludes PR 9; it can qualify real-model import/window behavior, but its pre-PR-9 checker cannot establish the full-logit/repeat-floor acceptance that PR 9 adds. For that acceptance, integrate/review PR 9 separately and pin the resulting actual combined SHA in both launcher `--ref` arguments; no nonexistent combined commit is claimed here.

On the GPU checkout containing the candidate object, use the exact source commit below. Keep node/client/checker aggregate VRAM in budget, as described in the original GPU caveat; run the two architectures sequentially, with the same exact local file for node/client. No provisioning or download command is included.

```sh
set -eu
source_commit=e1abcd9f8cf3e550a6b1829e4f40ae7365fc218b
test -z "$(git status --porcelain --untracked-files=no)"
git verify-commit "$source_commit"
git switch --detach "$source_commit"
test "$(git rev-parse HEAD)" = "$source_commit"
mkdir -p .temp/hybrid-a-r17-gpu
git rev-parse HEAD > .temp/hybrid-a-r17-gpu/source-commit.txt

run_swa() (
    set -eu
    model_path=$1
    arch_label=$2
    run_path="$PWD/.temp/hybrid-a-r17-gpu/$arch_label"
    build_lock="$PWD/.temp/hybrid-a-r17-gpu/build.lock"
    mkdir -p "$run_path/results"
    git rev-parse HEAD > "$run_path/results/source-commit.txt"
    test "$(cat "$run_path/results/source-commit.txt")" = "$source_commit"
    trap 'tools/hybrid-bench/setup-client.sh --state "$run_path/client" --stop || :; tools/hybrid-bench/deploy-prefill-node.sh --state "$run_path/node" --stop || :' EXIT
    tools/hybrid-bench/deploy-prefill-node.sh --ref "$source_commit" --offline --cuda \
        --model "$model_path" --ctx 8192 --bench-link --port 18941 \
        --state "$run_path/node" --build-lock "$build_lock"
    tools/hybrid-bench/setup-client.sh --ref "$source_commit" --offline --cuda \
        --model "$model_path" --ctx 8192 --request-timeout 7200 \
        --prefill-url http://127.0.0.1:18941 --port 18942 --decode-port 18940 \
        --node-info "$run_path/node/node-info.json" \
        --token-file "$run_path/node/client-tokens.json" \
        --state "$run_path/client" --build-lock "$build_lock"
    python3 tools/hybrid-bench/hybrid_bench.py --client-state "$run_path/client" \
        --scratch "$run_path" --output "$run_path/results" --label "$arch_label-window-r17" \
        --smoke --depths 512 1536 3072 6144 --check-depths 512 1536 3072 6144 \
        --check --request-timeout 7200
)
run_swa /absolute/path/gemma4-31b.gguf gemma4
run_swa /absolute/path/muse-glimmer-30b.gguf muse-glimmer
```

The launchers' build records and per-run source files must equal the pinned SHA. Do not interpret the older checker as PR 9 logit calibration. Retain actual exits: 0 homogeneous control, 2 refusal/abort, 3 numerical failure, 4 partial, 5 uncalibrated. Do not pass heterogeneous opt-in for this same-file control. Return only sanitized benchmark bundles/sidecars; keep credentials, native states and failed startup logs private.

### PR 9 exact-head compatibility

```sh
git merge-tree --write-tree e1abcd9f8cf3e550a6b1829e4f40ae7365fc218b \
  f65c59483b699ab5cb3b3de32a677fc1fbd7f8b1
```

Exit 0, resulting tree `82e6ae20769e357ee2913bc4f6d4f3f27ffbaa2b`, no conflicts. This checks the complete PR head, including checker and `hybrid_bench`, against the source candidate with all three main additions. It writes only Git tree objects and does not merge PR 9 into `m3-swa`. It proves textual merge compatibility, not execution of the combined revision or GPU/task quality. Final report-head compatibility and signature verification are recorded in `R/delivery.json`.

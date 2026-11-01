# Final-state chunks - hybrid-a-r24-261101

Ready for review on local branch `m3-chunk-state`; no push. Base: exact supplied `c4e0faae269ef47fcbe210f7068f9b64dd35c0ca`, containing the reviewed SWA, chat-template and main numerical-checker changes. Local `m3-swa` was stale; the task was moved to the pinned source before delivery. The server reuse threshold is untouched. All execution here is CPU-only with existing local 0.8B files; no GPU access or downloads.

## Behavior

HYBD protocol 2 carries full-attention KV in each chunk and fixed recurrent/SWA state only in the last chunk. The decoder validates and stages host records while the original slot remains visible. Final application defers every tensor write until all records and the final endpoint pass; failure restores cache metadata as well. The proxy counts application only after final import/readback, aborts staging on exit, and saves a local recovery snapshot just before final application to handle a lost final acknowledgement. An unconfirmed restore retains that file and detaches the slot from reuse. Existing local fallback and replay follow rollback.

The protocol, bounds, version refusal, window semantics and failure contract are in [NODE-STATE-DESIGN.md](NODE-STATE-DESIGN.md#chunk-transaction-protocol-2). Native snapshots and model fingerprints retain their existing formats. Upgrade node, proxy and decoder together. Conservative chunk planning still reserves fixed-state space for every chunk; this change removes repeated payloads without increasing the body limit or weakening either checksum.

Core changes outside `tools/hybrid` are necessary for KV-only export and one deferred transaction across all native records. `tools/server` owns per-slot staging and publication; `tools/hybrid-proxy` carries transfer identity and handles rollback; the benchmark collector/checker now use the same protocol. No reviewer test names were removed. The former expectation that a middle failure retains an imported first chunk is explicitly replaced by the requested original-slot invariant. No formal surface table or independent landing-property suite was supplied; that is a brief gap, not a coverage claim.

## Exact synthetic wire size

`llama-hybrid-check-codec` uses layout arithmetic and real envelope encoding, without allocating a 27B model or a multi-GB buffer. Fixture: 40,000 token IDs equal to 100, four ranges of at most 10,871 tokens, 64 layers with 16 attention and 48 recurrent layers, four KV heads, key/value head dimensions 256/256, q8_0 blocks of 34 bytes per 32 elements; recurrent inner/state/groups/conv are 6144/128/16/4. Hashes/session/transfer IDs have fixed documented lengths in the checker.

- KV tensors: `16 * 4 * (256 + 256) * 34 / 32 = 34,816` bytes/token.
- Native cell metadata: 24 bytes/token; per-chunk KV framing: 416 bytes.
- Recurrent tensors: `48 * 4 * (128 * 6144 + 3 * (6144 + 2 * 16 * 128)) = 156,893,184` bytes.
- Native recurrent record: tensor bytes plus 1,172 bytes of metadata, or 156,894,356 bytes.

| Bytes | Protocol 1, fixed state in every chunk | Protocol 2, fixed state only last | Reduction |
|---|---:|---:|---:|
| Native bodies | 2,021,179,088 | 1,550,496,020 | 470,683,068 |
| Complete wire, including headers and checksums | 2,021,340,960 | 1,550,658,451 | 470,682,509 |

The body reduction is exactly three native recurrent records: 470,679,552 tensor bytes plus 3,516 metadata bytes. Protocol-2 envelopes add 559 bytes across this fixture. JSON token values and chunk boundaries affect envelope lengths, so these exact wire totals describe this synthetic case, not an unrun reproduction of the owner's measured 39,504-token request. The recurrent tensor geometry agrees with that run's existing `node-info.json` sizing of 156,893,184 bytes.

## CPU evidence

Logs, commands, PIDs, identity records and real exits are under `.temp/hybrid-a-r24-261101/`. `native-checks-03.json` records standalone native command execution. Old golden envelope headers were converted only in task scratch to protocol 2; original golden files and their native bodies remain unchanged.

| Check | Evidence | Result |
|---|---|---|
| CPU Release build, node/server/all selected checkers on pinned base | `build-09.log`, `build-10.log` and monitors | 0 |
| Codec bounds, checksums, exact wire arithmetic | `codec-03.log` | 0 |
| Gemma/Glimmer allocated SWA, unified and separate streams | `swa-03.log` | 0; chunk/single bytes, final window, refusals and neighbor preservation |
| Four-chunk 0.8B Q4_0 and Q8_0 imports | `chunks-q4-04.log`, `chunks-q8-04.log` | 0 each; fresh and populated slots byte-identical to single-shot, held-token logits identical |
| Interrupt after k=1,2,3 of four chunks | same native logs | 0; fresh/populated cells/state unchanged; every existing populated-slot next-token logit unchanged |
| Missing/early state, order, duplicate, missing middle/final, version, transfer ID, checksum | same native logs | 0; intended refusal and unchanged state/logits |
| Late recurrent endpoint mismatch after KV metadata changes | `chunk_late_endpoint_rollback` in same logs | 0; deferred writes discarded, metadata and logits preserved |
| Existing malformed recurrent, golden append and guards | `recurrent-03.log`, `append-04.log`, `guards-03.log` | 0 each; original 64-token greedy regression retained |
| Full proxy and inherited template regressions | `proxy-12.log` | 0; 118 tests |
| Benchmark unit tests with allocated native SWA checker enabled | `bench-02.log` | 0; 54 tests, no skips |
| Real node/server/proxy, interrupted bytes, malformed state/order/version, middle-failure fallback | `http-03.log`, `e2e-checks-03.json` | 0 |
| Final-state chunks through inherited full-logit numerical checker | `noise-chunks-04.log` | 0; all eight same-partition positions exact, original corruption controls retained |
| Native LRU replay and RAM-budget refusal | `focused-integration-04.log` | 0; 25 exact replay tokens and no decoder mutation; real 503 budget fields and zero retained sessions |
| Full inherited numerical/timing driver | `integrated-03.log` | 1; whole-prompt top-1 gate failed, as detailed below |
| Controlled candidate versus unchanged-base checker/import path | `partition-diagnostic-02.log`, `partition-diagnostic/commands.json` | diagnostic 0; both checker processes actually exit 3 with identical metrics |

The claim comments beside the tests state their actual scope. Native Qwen tests use the existing Q4_0 and Q8_0 files in `~/Werkstatt/hybrid-models/`; SWA tests allocate synthetic caches and do not establish real Gemma/Glimmer inference. Expected-red controls in the numerical suite actually exit 2 for checksum/device refusal and 3 for checksum-valid altered state; those commands are failing controls, not passing checker results.

| Mutant | Property narrowed | Named failing test | Actual result |
|---|---|---|---|
| Skip per-chunk native validation | Early fixed state must be refused immediately, before final apply | `chunk_early_native_state` | Exit 1, expected red; `mutant-validation-red-04.log` |
| Disable KV metadata restore | A late final failure must restore the original KV prefix | `chunk_late_endpoint_rollback` | Exit 1, expected red; `mutant-rollback-red-04.log` |

Both task-local interposition libraries compile with exit 0 and affect only owned checker processes. No mutant survived this bounded set. These controls do not establish arbitrary backend-failure recovery.

Builds use nice 19 and `-j2`, with a disk-backed `TMPDIR` and `PYTHONDONTWRITEBYTECODE=1`. A monitor refuses/aborts below the requested 25,000,000,000-byte floor. Early attempts were refused or stopped while unrelated host scratch allocations crossed that floor: builds 01/02 refused (78), builds 03/04 stopped their owned children (-15, guard 78). Build 05 failed (2) on a duplicate variable in the new arithmetic test; corrected before build 06. Build 07 was cancelled while waiting for the shared lock, with no build started; builds 08-10 use the allowed second build slot while the shared lock serializes other builders. No foreign files/processes were removed. Initial proxy runs exposed short fixture timeouts; the outer fixture timeout is now 30 s and its node timeout 5 s. Two new assertions initially compared synthetic `last_used` timestamps and were corrected to compare the actual slot contract. The first HTTP regression run correctly rejected its obsolete partial-import expectation; the test now requires staged KV to be discarded.

Base correction: the final ancestry gate disproved the early assumption that local `m3-swa` contained the pinned head. A signed local checkpoint preserved the patch, then a signed rebase moved it onto exactly `c4e0faae`. The single conflict in the synthetic server was resolved by retaining both protocol-2 delta production and the inherited template-fixture routing. Core/native source bytes were verified identical across this move. The inherited PR-9 numerical checker and template suites are included in final validation; stale-base evidence is not substituted for those checks. The early `integrated-02` budget gate failed because it observed HTTP 429 instead of the intended RAM-budget 503, so that attempt is not passing budget evidence.

The full inherited timing driver remains red under this CPU build (`GGML_NATIVE=OFF`, AVX2/FMA enabled). For its 132-token synthetic chat prompt, both fresh and append-chain checks have exactly zero same-partition/repeat KL and logit difference, but whole-prompt top-1 agreement is 7/8 (87.5%), below the unchanged 95% gate; mean KL is 0.00852205499931497. This is not passing whole-prompt qualification. A checker extracted byte-for-byte from `c4e0faae` was compiled against the same native libraries and protocol-2 envelope decoder; it uses the unchanged legacy native append path. On the identical captured IDs and full native delta, baseline and candidate both exit 3 and produce identical exactness, maximum-logit difference, top-1 count and mean KL. The source of the legacy append function is unchanged. No threshold, model, test expectation or build flag was adjusted to turn this gate green. The task's single-shot-versus-chunked equivalence contract passes; broader cross-partition acceptance remains a declared gap. The focused numerical, replay and budget gates were run separately and do not erase the failed full-driver result.

Toolchain: Linux x86_64, GCC 15.2.0, CMake 4.4.4, Python 3.14.4, Release `-O3 -DNDEBUG`, CPU q8_0 KV, two model threads and 256/256 batching. `editorconfig-checker`, `ty` and `clang-format` are unavailable and were not run. Python AST, added-line ASCII, whitespace and threshold-scope checks are recorded separately. The task build tree and temporary control executables are removed after validation; source/binary identities and exit records remain in task scratch.

Not qualified here: CUDA/Metal compilation or inference, real 27B/SWA models, the 5G link, crash recovery, fatal backend/OOM failures, or persistent decoder unavailability during rollback. No independent acceptance or landing is claimed. Recovery files stay private; an unconfirmed restore refuses slot reuse. The reuse threshold follow-up remains separate.

## Reproduce local checks

Use a CPU build of this exact signed head. Logs and model state stay in private task scratch; do not add models or download weights.

```sh
export TMPDIR="$PWD/.temp/hybrid-a-r24-261101/tmp"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
model=$HOME/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf
nice -n 19 build/bin/llama-hybrid-check-codec
nice -n 19 build/bin/llama-hybrid-check-swa
nice -n 19 build/bin/llama-hybrid-check-append "$model" --chunks
nice -n 19 build/bin/llama-hybrid-check-append \
  "$HOME/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-r16.gguf" --chunks
nice -n 19 python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py'
nice -n 19 python3 tools/hybrid-proxy/test_remote_append.py \
  --binary "$PWD/build/bin/llama-server" --prefill-binary "$PWD/build/bin/llama-hybrid-prefill" \
  --model "$model" --artifacts "$PWD/.temp/hybrid-a-r24-261101/http-requal"
nice -n 19 python3 tools/hybrid-bench/test_integrated.py \
  --server-binary "$PWD/build/bin/llama-server" --prefill-binary "$PWD/build/bin/llama-hybrid-prefill" \
  --checker-binary "$PWD/build/bin/llama-hybrid-bench-check" \
  --model "$model" --artifacts "$PWD/.temp/hybrid-a-r24-261101/integrated-requal"
```

The full integrated command above reproduces the diagnosed timing-fixture failure on this CPU configuration. Add `--case noise` to run the scoped full-logit chunk transaction and existing checksum/state controls. Do not describe the complete driver as green.

## GPU: r17 block recipe

Unrun operator commands. Deliver the local signed source object to both existing checkouts first, set `SOURCE` to the full delivered commit ID, and keep that exact revision for node, client and checker. Run architectures sequentially with identical existing files, enough aggregate VRAM for services plus checker contexts, and no heterogeneous opt-in. This retains r17's bounded block/window ladder.

```sh
set -eu
source_commit=${SOURCE:?full delivered commit ID required}
git verify-commit "$source_commit"
test "$(git rev-parse HEAD)" = "$source_commit"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p .temp/hybrid-a-r24-gpu/tmp
export TMPDIR="$PWD/.temp/hybrid-a-r24-gpu/tmp"

run_block() (
    set -eu
    model_path=$1
    arch_label=$2
    run_path="$PWD/.temp/hybrid-a-r24-gpu/$arch_label"
    build_lock="$PWD/.temp/hybrid-a-r24-gpu/build.lock"
    mkdir -p "$run_path/results"
    git rev-parse HEAD > "$run_path/results/source-commit.txt"
    trap 'tools/hybrid-bench/setup-client.sh --state "$run_path/client" --stop || :; tools/hybrid-bench/deploy-prefill-node.sh --state "$run_path/node" --stop || :' EXIT
    tools/hybrid-bench/deploy-prefill-node.sh --ref "$source_commit" --offline --cuda \
        --model "$model_path" --ctx 8192 --bench-link --port 18941 \
        --state "$run_path/node" --build-lock "$build_lock"
    tools/hybrid-bench/setup-client.sh --ref "$source_commit" --offline --cuda \
        --model "$model_path" --ctx 8192 --request-timeout 7200 \
        --prefill-url http://127.0.0.1:18941 --port 18942 --decode-port 18940 \
        --node-info "$run_path/node/node-info.json" --token-file "$run_path/node/client-tokens.json" \
        --state "$run_path/client" --build-lock "$build_lock"
    python3 tools/hybrid-bench/hybrid_bench.py --client-state "$run_path/client" \
        --scratch "$run_path" --output "$run_path/results" --label "$arch_label-window-r24" \
        --smoke --depths 512 1536 3072 6144 --check-depths 512 1536 3072 6144 \
        --check --request-timeout 7200
)
run_block /absolute/path/gemma4-31b.gguf gemma4
run_block /absolute/path/muse-glimmer-30b.gguf muse-glimmer
```

Keep r17's optional-MTP/source-layer admission caveats: an unsupported real file must refuse, not bypass fingerprints or discard unknown tensors. Require node health `delta_version: 2`; inspect generated source/build records and nonzero window geometry. Multi-chunk calls should show early `staged` entries and one final `imported` entry. Keep both fresh and append-chain checker outcomes. A same-backend run's expected success exit is 0; 2 is refusal, 3 numerical failure, 4 partial and 5 uncalibrated. Require the inherited full-vocabulary same-partition/repeat-floor fields, including exact same-partition logits. Greedy agreement alone is insufficient. The checker also retains whole-prompt and varied-partition noise measurements.

## GPU and Mac: 40k over 5G

Unrun. Use the same delivered `SOURCE` and the same existing 27B GGUF on both machines. A 49,152-token context covers the requested ladder and continuation. Run the SWA block above separately; this case targets the observed recurrent duplication.

On the GPU host, retain the node for the Mac run:

```sh
set -eu
source_commit=${SOURCE:?full delivered commit ID required}
git verify-commit "$source_commit"
test "$(git rev-parse HEAD)" = "$source_commit"
export PYTHONDONTWRITEBYTECODE=1
run_path="$PWD/.temp/hybrid-a-r24-40k"
mkdir -p "$run_path/results" "$run_path/tmp"
export TMPDIR="$run_path/tmp"
tools/hybrid-bench/deploy-prefill-node.sh --ref "$source_commit" --offline --cuda \
    --model /absolute/path/27B-Q4_K_M.gguf --ctx 49152 --bench-link --port 18941 \
    --state "$run_path/node" --build-lock "$run_path/build.lock"
```

Transfer only `node-info.json` and the private `client-tokens.json` to the owner's private Mac task directory using the existing authorized operator channel; keep the token file mode 0600. On the Mac, use an existing SSH host alias in `GPU_SSH`. The tunnel is an owned foreground process moved to the shell background with its PID recorded, then stopped by that PID. Metal is selected by the existing macOS launcher.

```sh
set -eu
source_commit=${SOURCE:?same full delivered commit ID required}
git verify-commit "$source_commit"
test "$(git rev-parse HEAD)" = "$source_commit"
run_path="$PWD/.temp/hybrid-a-r24-40k-mac"
mkdir -p "$run_path/results" "$run_path/tmp"
export TMPDIR="$run_path/tmp"
export PYTHONDONTWRITEBYTECODE=1
chmod 600 "$run_path/client-tokens.json"
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=8 \
    -L 18941:127.0.0.1:18941 "${GPU_SSH:?existing GPU SSH alias required}" \
    > "$run_path/tunnel.log" 2>&1 &
tunnel_pid=$!
printf '%s\n' "$tunnel_pid" > "$run_path/tunnel.pid"
trap 'tools/hybrid-bench/setup-client.sh --state "$run_path/client" --stop || :; kill "$tunnel_pid" 2>/dev/null || :; wait "$tunnel_pid" 2>/dev/null || :' EXIT

tools/hybrid-bench/setup-client.sh --ref "$source_commit" --offline \
    --model /absolute/path/27B-Q4_K_M.gguf --ctx 49152 --request-timeout 7200 \
    --prefill-url http://127.0.0.1:18941 --port 18942 --decode-port 18940 \
    --node-info "$run_path/node-info.json" --token-file "$run_path/client-tokens.json" \
    --state "$run_path/client" --build-lock "$run_path/build.lock"
python3 tools/hybrid-bench/hybrid_bench.py --client-state "$run_path/client" \
    --scratch "$run_path" --output "$run_path/results" --label mac-metal-40k-r24 \
    --smoke --depths 39500 40000 --check-depths 40000 --request-timeout 7200 --check
```

Retain the real process exit, sanitized bundle and SHA sidecar. CUDA-to-Metal numerical results remain uncalibrated (exit 5), even when within the inherited numerical bounds. Verify the first forced-offload call has four chunks, three `staged` and one `imported`, `remote_applied` equal to the complete imported range, and no fallback; compare its actual wire bytes with the prior run, accounting for actual token IDs and chunk endpoints. Small subsequent turns legitimately carry one fixed state each. On interruption, retain the result/error evidence and verify original slot bytes before local fallback through the HTTP/native controls; do not count staged ranges as applied. After the Mac client releases its sessions, stop the GPU node with `deploy-prefill-node.sh --state "$run_path/node" --stop` in the GPU shell. Return sanitized bundles only, not credentials or native states.

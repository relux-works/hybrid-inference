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

## r25 - r7g follow-ups

Continuation of accepted r24 head `675cc8f7f41da3023bd101f41ed68126616355c2` on `m3-chunk-state`, run `hybrid-a-r25-261101`. Local signed delivery only; no push. Review input: `/home/agent/Werkstatt/hybrid-review-proto/llama.cpp/.temp/REVIEW-chunk-state-claude.md`. Evidence, exact commands, PIDs and exits: `.temp/hybrid-a-r25-261101/`. No changes to SWA window export or any `src/` file. The separate `m3-swa-fix` merge and its combined-tree validation remain pending.

### Regressions and stale staging

`test_staged_chunks_do_not_persist_applied_state_or_drop_checkpoints` opens a fresh SQLite connection before each next chunk request. In token-journal and message-turn modes it requires the old durable tokens/remote boundary and zero applied count throughout staging; message-turn checkpoints must survive until final commit. The same transfer then succeeds and clears the checkpoints. The reviewer's exact M6 deletion now fails on the fresh reader observing two uncommitted tokens. Producer error cleanup and local fallback cannot conceal this observation.

`swa_chunk_late_window_rollback` allocates real CPU Gemma/Glimmer caches with a 512-cell window. The destination occupies two window cells and its neighbor occupies 510. Both chunks validate. Apply queues base KV writes, removes the destination's old window metadata, then fails to allocate the five-cell final window. The test checks byte-exact metadata/neighbor snapshots and all physical K storage, including unused cells where an undiscarded queued write would otherwise escape serialization checks. Freeing three neighbor cells admits the identical transaction (`swa_chunk_late_window_capacity_control`). This new allocation-pressure case uses unified storage on both architectures; the existing separate/unified fixtures remain green. It does not establish real-model SWA logits or arbitrary backend-failure recovery.

Stale staging was reproduced before the fix through the real node/server: two abandoned chunks, then a new chunk zero on the same live prefix, returned HTTP 400 with `position gap or overlap` (`http-red-01.log`, actual exit 1; server log records the intended rejection). The decoder used the abandoned staged endpoint as the expected start. That rejection is spurious; the old slot is still intact. A new transfer now replaces host staging only when its ID differs and its chunk index is zero, then undergoes all normal live-prefix, session, model, endpoint, vocabulary, checksum and native checks. Same-transfer duplicates, changed IDs on later chunks, wrong-prefix replacements, foreign sessions and corrupt replacements still refuse with unchanged state/logits. The real HTTP regression now accepts the first replacement request, keeps slot bytes unchanged before its final chunk, and commits the complete replacement at the end. No retry, fallback exception allowance or reuse-threshold change was added.

| Mutant | Property narrowed | Named failing test | Actual result |
|---|---|---|---|
| M6: remove the proxy's non-final `continue` | Staged chunks cannot enter the durable applied prefix | `test_staged_chunks_do_not_persist_applied_state_or_drop_checkpoints` | Exit 1; `m6-mutant-02.log` |
| M2: remove failed chunk readers' `discard()` | Queued physical KV writes cannot publish on late SWA failure | `swa_chunk_late_window_rollback` | Exit 1, `deferred keys written after rejection`; `m2-swa-red-01.log` |
| M3: remove `restore_fixed()` | Late SWA failure must restore the old window metadata | `swa_chunk_late_window_rollback` | Exit 1, `metadata or neighbor changed`; `m3-swa-red-01.log` |

No mutant survived this bounded set. The native mutants compile/link independently with exit 0 from copies under task scratch; the candidate library hash remains unchanged. `controls.json` records the direct processes and actual exits. These controls supplement the retained r24 refusals and do not claim exhaustive failure injection.

### Layout formulas and real 0.8B measurement

Let `A` be attention layers, `Q` recurrent layers, `s` KV streams, `P` old committed token depth, and `D` newly transferred tokens. For these Qwen layouts, q8_0 blocks hold 32 values in 34 bytes, each cell has 24 metadata bytes, and recurrent tensors use four-byte floats. Non-transposed K/V, one selected sequence, text-only tokens and no draft state are assumed.

```text
K = sum_attention(kv_heads * (key_dim + value_dim) * 34 / 32)
R = 4 * Q * (state * inner + (conv - 1) * (inner + 2 * groups * state))
F = R + 20 + 24 * Q                         # one native recurrent record
C(s) = 8 + 4 + 4*s + 16 + 24*A              # native KV chunk framing, including magic/sequence
B(chunk_tokens, final) = (K + 24)*chunk_tokens + C(s) + (final ? F : 0)
Save(P,s) = (K + 28)*P + F + C(s) + 20      # occupied P > 0, no checkpoint appendix
Appendix(q > 0) = 12 + sum_checkpoint(40 + target_bytes + draft_bytes + spec_bytes)
```

`Save` includes the native slot-file header (12 bytes), packed server token framing (16 plus four bytes/token), and memory state without the chunk's eight-byte magic/sequence prefix. Empty streams each add four bytes; empty slots have different zero-cell framing. `F` includes one recurrent cell header and two type/row-size descriptors per recurrent layer. Existing Qwen partial recurrent checkpoints have `target_bytes = F + 8`; without draft/spec state, a nonempty appendix is `12 + q*(F + 48)`. It is not included in the base-file table below.

For `m = ceil(D/w)` equal-width chunks, the last possibly shorter, retained native-body bytes immediately before the final chunk are `(K+24)*w*(m-1) + C(s)*(m-1)`. Once that chunk has been parsed and staged, they are `(K+24)*D + C(s)*m + F`. This counts body string lengths, not allocator capacity or RSS. Additional allocations include retained JSON token arrays and object/string storage, prefix/cumulative token vectors, the current wire and HTTP/task buffers, model/cache storage and snapshot I/O buffers. The wire/body copies can coexist with all previously staged bytes. Deferred tensor readers refer to these bodies; they do not remove the staging cost.

Measured with the existing `Qwen3.5-0.8B-Q4_0.gguf`, real node exports and real decoder `action=save`, CPU only: `A=6`, `Q=18`, `inner=2048`, `state=128`, `groups=16`, `conv=4`, two KV heads and 256/256 dimensions. Thus `K=6,528`, `R=20,201,472`, `F=20,201,924`, node `C(1)=176`, decoder `C(3)=184`. The node makes four chunks of four appended tokens after each old prefix. Nine snapshots were saved immediately before the final import and deleted after reading their size. The following predictions matched every observed byte count; `measure.py` asserts the formulas from health's actual layout (`measurement-03.log`, `measurement.json`, exit 0).

| Old depth P | Full node body at P | Recovery file, predicted = observed | Median server save_ms, 3 runs | Median request wall ms |
|---|---:|---:|---:|---:|
| 32 | 20,411,764 | 20,411,920 | 5.736 | 7.385 |
| 128 | 21,040,756 | 21,041,296 | 6.851 | 8.516 |
| 512 | 23,556,724 | 23,558,800 | 7.460 | 10.990 |

Each early four-token body was exactly 26,384 bytes and the final body 20,228,308. Three staged early bodies retain 79,152 bytes; all four retain 20,307,460. A separate real two-token local generation produced one checkpoint: at P=33 the base file is 20,418,476 bytes, the target checkpoint buffer 20,201,932, its appendix 20,201,984, and the complete recovery file 40,620,460. Actual server save time was 35.888 ms (`checkpoint-measure-01.log`, exit 0). This confirms that ordinary retained checkpoints materially increase recovery I/O. Save timing is CPU/host/filesystem-specific, includes page-cache writes, and is not an fsync durability or Mac latency measurement. Earlier sizing runs 01/02 also passed; run 03 additionally checks every formula and packed-token/stream framing explicitly.

### 27B at 40k and 131k

For the existing 27B layout, `A=16`, `Q=48`, four KV heads, 256/256 dimensions, `inner=6144`, `state=128`, `groups=16`, `conv=4`: `K=34,816`, `R=156,893,184`, `F=156,894,356`, `C(1)=416`. The table uses one KV stream and the r24 arithmetic fixture's explicit width `w=10,871`. The current proxy default is 10,873; explicit exact-layout geometry admits 10,876. These are layout calculations, not a downloaded/loaded 27B run. `131k` below means 131,072 tokens.

| Cost in bytes | 40,000 tokens | 131,072 tokens |
|---|---:|---:|
| Recovery file at old depth P, no checkpoints | 1,550,654,792 | 4,723,967,560 |
| Recovery file with one recurrent checkpoint | 1,707,549,208 | 4,880,861,976 |
| Retained early bodies before final, new range D | 1,136,238,168 | 4,544,952,672 |
| All KV bodies, excluding final recurrent record | 1,393,601,664 | 4,566,553,888 |
| All staged bodies including final state | 1,550,496,020 | 4,723,448,244 |

The base recovery values are 1.551/4.724 decimal GB (1.444/4.400 GiB). The pre-final staging values are 1.136/4.545 GB; the full staged payload is 1.550/4.723 GB. Thus the reviewer's roughly 1.4 GB corresponds to the aggregate 40k KV term, not the actual three early chunks with this width. Three decoder streams add only eight bytes to the base recovery file. Every retained checkpoint adds 156,894,404 bytes plus the appendix's one-time 12 bytes; with 32 checkpoints, total recovery files would be 6,571,275,732 / 9,744,588,500 bytes. That is a conditional size, not an observed checkpoint count at those depths. If 131k means 131,000, base recovery is 4,721,458,792 bytes and full staging 4,720,939,764; `sizing-27b.json` records both conventions.

P and D are independent. A short offloaded append at an old depth of 40k still writes the roughly 1.55 GB base backup but stages only that append plus fixed state. A fresh 40k transfer stages the large range while its empty old-slot recovery file is small. Each offloaded turn that reaches its final chunk saves the whole old slot, including single-chunk turns. This operation is currently counted inside proxy `import_ms`; the 27B/Mac rerun should meter `action=save` bytes and latency separately. No 27B timing is extrapolated from the 0.8B timings.

### Recovery proposal, not implemented

Replace the per-turn full snapshot with an undo record owned by the decoder transaction. Before changing live state, retain the old recurrent/SWA payload, old token/checkpoint bookkeeping and allocator metadata, and capture any KV cells that can be overwritten. Full-attention append normally needs suffix removal plus metadata restoration; that alone cannot restore recurrent state or overwritten SWA cells. Existing `PARTIAL_ONLY` export can supply fixed state, but the current in-process metadata restore closures are not a persistent undo format.

Keep the undo record and transfer identity through final application until an explicit client confirmation. A lost final response must permit an idempotent abort that restores the old slot, or a status query followed by that abort; accepting the new state merely because status says committed would change the current rollback guarantee. Reject decode/reuse while confirmation is unresolved. If preserving recovery across decoder/process restarts is required, write a checksummed, generation-bound undo intent durably before mutation and recover it on startup; an in-memory undo alone would narrow that guarantee. Retain the old token journal/checkpoints until confirmation and bind every operation to the exact session, transfer and prefix. Refuse before mutation if undo allocation/persistence fails.

For Qwen this changes routine backup data from `O(P*K + F + checkpoints)` to `O(F + overwritten_KV + changed_metadata/checkpoints)`, potentially about 157 MB of recurrent payload on 27B instead of copying the old full-attention prefix. A retained snapshot plus an append-only KV journal is another option if a restartable on-disk format is wanted. Both need explicit retry, lost-confirmation, crash, checkpoint and capacity tests before replacing the current full save. Neither optimization is implemented in r25.

### Validation and limits

| Check | Result and evidence |
|---|---|
| CPU Release selected targets, nice 19, -j4 maximum | Build 01, 03, 04 exit 0; exact commands/PIDs/disk samples in build logs |
| Candidate codec and allocated SWA | Exit 0 each; `codec-01.log`, `swa-02.log` |
| Fresh/populated four-chunk Q4_0 and Q8_0, same-partition bytes/logits, retained refusals and new replacement refusals | Exit 0 each on final checker; `chunks-q4-02.log`, `chunks-q8-02.log`, `final-checks.json` |
| Real node/server/proxy HTTP, first-request stale replacement, existing replay/fallback/checkpoint controls | Exit 0; `http-01.log`, `checks.json` |
| Malformed recurrent-state checks | Exit 0; `recurrent-01.log` |
| Full proxy suite including new M6 regression | 119 tests, exit 0; `proxy-01.log` |
| Benchmark unit suite with native SWA checker | 54 tests, no skips, exit 0; `bench-01.log` |
| Real layout/snapshot/staging arithmetic and checkpoint appendix | Exit 0; `measurement-03.log`, `checkpoint-measure-01.log` |
| Bounded M2/M3/M6 controls | Intended failures, actual exit 1 each; compile/link and asserting drivers exit 0 |

Build 02 failed with exit 2 because the new test initially accessed private staging fields; it was corrected to exercise replacement through the public stage API. One queued shared-lock controls invocation was cancelled before it started (143); the actual controls ran in the allowed second build slot, serially under the task lock. Own builds were always niced, no more than one ran at a time, with one other host builder observed. Guard admission was above 28.5 GB and abort threshold above 28.1 GB; sampled free space stayed above 28 GB. All TMPDIR paths are on disk and Python bytecode is disabled. Build 01 retrieved the default prebuilt UI assets; no model was downloaded and no GPU was used.

No existing reviewer test was removed. All changes stay in the hybrid helper/tests/docs and proxy tests; production proxy and SWA export are unchanged. The original brief has no formal surface table; the r7g rows provide the review context, with r25 specifically strengthening R1/R2/R8 and retaining R3-R6's executable controls. R7's documented baseline whole-prompt discrepancy, real SWA inference, GPU/Mac qualification and the combined `m3-swa-fix` export/layout qualification remain outside this follow-up's acceptance claim. The full inherited timing driver was not rerun: this change uses exact same-partition/native/HTTP gates and does not repair its already reproduced whole-prompt discrepancy. Type/style tools `ty`, `editorconfig-checker` and `clang-format` are unavailable; those checks are unrun. Local syntax, ASCII, diff/retained-test and signature audits are recorded separately. Temporary build/control executables and saved binary fixtures are removed before handoff; model files are retained. Ready for review.

Final audit: `audit-02.log` exits 0 for syntax/ASCII, retained Python test names, actual gate/control exits, untouched core/export scope and 23 inactive recorded process IDs. Audit 01 failed on its own incorrect expectation that the probe node exits 0; the probe deliberately terminates that PID with SIGTERM (-15) after its successful assertions. This was an audit assumption correction, not a product/test change. `validation-identity.json` binds source/binary hashes and observation time. Cleanup exits 0: own build tree, temporary control libraries/objects and saved binary fixtures are absent; logs and build metadata remain. Free disk after cleanup at 2026-11-01T21:41:16Z was 28,598,861,824 bytes, above the 28 GB floor.

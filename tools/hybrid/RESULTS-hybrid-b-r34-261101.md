# Client delta import speed

Run: hybrid-b-r34-261101. Base: `a25754271e19cfda2c44c702a52b752cb53cf623`, fetched from `https://github.com/relux-works/hybrid-inference`. Branch: `m3-import-speed`. Local signed delivery, no push. Ready for review; GPU and Metal measurements remain unrun.

## Result and changes

Final CPU medians: fresh import 169.1 -> 324.0 MB/s (1.92x), occupied-prefix append 238.6 -> 511.0 MB/s (2.14x). Checksums and full-body copies dominate. Native parsing and tensor writes do not dominate this workload.

- Reuse the SHA-NI compression/CPUID dispatch and Apple CommonCrypto approach from PR #7 head `41514e77c8e9da5433d77fa9ddd46466ebf44386`. One streaming SHA helper now serves model-file and delta-body hashing. `HYBRID_HASH_PORTABLE=1` retains the portable control; unsupported CPUs retain the portable implementation.
- Feed XXH3-128 and SHA-256 from the same 64 KiB chunks. Both digests still cover every body byte and retain their exact wire representations.
- Add an explicitly borrowed delta reader for the decoder task's owned envelope. The existing owning reader remains available. The task owns the body until synchronous deferred tensor writes finish; rvalue envelopes are refused at compile time.
- Move the HTTP adapter's body into its request object. Together with the borrowed reader this removes two full-body copies: four body copies become two. The remaining task copy keeps queued work independent of HTTP cancellation. Encoding reserves one output buffer. Heterogeneous proxy rebinding joins a memoryview tail into one output allocation instead of copying the tail twice.
- Add opt-in stage timers and reproducible profiling/identity tools. No tensor layout, numeric operation, native acceptance rule, wire version, body/header limit, fingerprint admission, rollback or checksum requirement changes.

Inspection corrected one initial hypothesis: this base already groups contiguous KV destinations into layer ranges and calls `ggml_backend_tensor_set_n`. Adding another batching mechanism was unnecessary. The ordinary non-transposed 0.8B record schedules 12 KV ranges plus recurrent tensor ranges, rather than one call per cell.

## Method and stage definitions

CPU only on agents1, CMake 4.4.4, GCC 15.2.0, Release, existing Qwen3.5-0.8B Q4_0 and Q8_0-r16 GGUFs. No downloads. Native inference uses two compute/batch threads, Q8_0 KV, flash attention, 256/256 batches and zero GPU layers. Builds are serialized through the shared model-cache lock, niced to 19, and use `-j2`. TMPDIR is on disk and Python bytecode is disabled.

`R=.temp/hybrid-b-r34-261101`. `baseline-identity.json`, `candidate-identity.json`, build records and decoder logs bind the measurements to source/binary identities. The baseline is pinned main plus timers and the identity diagnostic, with original hashing/copy behavior. `import-profile-baseline.patch` reproduces those six instrumented source files byte for byte; a separate temporary Git index verified this against the built baseline. It changes no inference or acceptance behavior.

The in-memory synthetic record expands the installed 511-cell full golden record to 45,056 cells. Body: 315,409,012 bytes; wire: 315,499,563 bytes. It repeats real Q8_0 KV rows and retains all real recurrent tensors. Fresh mode declares [0,45056); append mode first imports the genuine 511-token prefix, then declares [511,45567). The latter exercises recurrent backup/replacement. Synthetic positions, framing, checksums and endpoint/token checks are valid; this synthetic record is a throughput fixture, not an inference-quality oracle. Numerical identity uses genuine golden records separately.

The actual proxy `Node` downloads the body, then sends it through the actual decoder HTTP slot route. The homogeneous proxy does not hash the body itself: the decoder verifies both checksums before native append. Receive below is the proxy's body-read timer over loopback, including allocation. Import excludes that receive and starts at the proxy's POST to the decoder, matching the original benchmark's import boundary.

Copy sums HTTP body copy, request handoff, queued task copy and state extraction. Parse sums envelope parsing/checks plus native validation and metadata application, excluding checksum/copy/tensor I/O. Native validation skips tensor payloads after checking framing/layout, so its MB/s is an effective body rate. Tensor-write time surrounds the existing batched backend call and includes backend waits. Synchronization is separately timed; waits inside tensor I/O are not separated from that call. `transport_dispatch_free` is the measured POST residual after named stages: decoder socket receive/assembly, task/result dispatch, allocation/free and response overhead. It is not presented as a pure network or memcpy measurement.

All MB/s use decimal MB and the logical body size. Copy/parse/residual rates are effective rates, not hardware bandwidth. Sync and backup do not scan a whole body and have no body MB/s. Separate synchronization can overlap backup and is not added again to the total. Stage medians need not sum to the median total. Pair order alternates before/after, after/before, before/after. This is a shared host: absolute rates vary with other work. Final datasets are `fresh-paired-03/measurements.json` and `append-paired-03/measurements.json`; earlier pilots/diagnostic repetitions are retained separately.

## Three paired runs

| Workload/pair | Before ms | After ms | Before MB/s | After MB/s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh 1 | 1743.158 | 976.413 | 180.9 | 323.0 | 1.79x |
| Fresh 2 | 1895.400 | 973.339 | 166.4 | 324.0 | 1.95x |
| Fresh 3 | 1865.694 | 783.734 | 169.1 | 402.4 | 2.38x |
| Append 1 | 1321.968 | 582.538 | 238.6 | 541.4 | 2.27x |
| Append 2 | 1282.771 | 656.088 | 245.9 | 480.7 | 1.96x |
| Append 3 | 1336.991 | 617.271 | 235.9 | 511.0 | 2.17x |

Fresh stages; rates list pairs 1, 2, 3:

| Stage | Before median ms | After median ms | Before MB/s, three pairs | After MB/s, three pairs |
| --- | ---: | ---: | --- | --- |
| Proxy receive | 225.489 | 175.153 | 1401.1, 1398.8, 1307.6 | 1945.1, 1186.7, 1800.8 |
| Checksum | 833.111 | 221.671 | 371.7, 378.6, 386.5 | 1305.8, 1422.9, 1692.1 |
| Copy | 445.658 | 258.361 | 729.4, 707.7, 619.7 | 1177.5, 1220.8, 1526.9 |
| Parse | 15.663 | 17.473 | 20137.2, 25996.5, 19245.7 | 16712.7, 18051.6, 19405.3 |
| Tensor writes | 42.446 | 29.522 | 7413.5, 9884.4, 7430.9 | 10683.7, 7900.0, 10931.7 |
| Transport/dispatch/free residual | 481.776 | 418.592 | 780.9, 550.9, 654.7 | 753.5, 723.6, 912.5 |
| Empty-prefix backup | 0.006 | 0.008 | n/a | n/a |
| Synchronization | 0.003 | 0.003 | n/a | n/a |

Occupied-prefix append stages; rates list pairs 1, 2, 3:

| Stage | Before median ms | After median ms | Before MB/s, three pairs | After MB/s, three pairs |
| --- | ---: | ---: | --- | --- |
| Proxy receive | 115.224 | 103.526 | 2329.5, 2737.4, 3001.5 | 3046.7, 2993.1, 3158.9 |
| Checksum | 676.370 | 160.476 | 466.3, 482.4, 465.6 | 1953.7, 1965.5, 2130.0 |
| Copy | 310.415 | 153.639 | 1024.7, 996.1, 1016.1 | 1990.9, 2052.9, 2108.6 |
| Parse | 9.956 | 9.915 | 27321.4, 31680.1, 35885.3 | 23751.0, 33247.6, 31811.6 |
| Tensor writes | 23.502 | 23.902 | 13455.4, 13420.8, 12584.7 | 12661.2, 13710.4, 13195.7 |
| Transport/dispatch/free residual | 297.389 | 282.672 | 1060.6, 1144.7, 1009.7 | 1438.7, 1029.2, 1115.8 |
| Recurrent backup, about 20.2 MB | 3.272 | 3.115 | n/a | n/a |
| Synchronization | 0.002 | 0.002 | n/a | n/a |

Final checksum logs explicitly select `sha-ni`. Earlier unchanged-native runs under different shared load also improved: dataset 02 fresh median 206.2 -> 424.0 MB/s, append 260.5 -> 556.6 MB/s. These are corroborating observations, not substituted for the final three pairs.

## Correctness and refusal evidence

`check-client --identity` exports the entire native state, then hashes every vocabulary logit at 64 teacher-forced positions. `test_import_identity.py` requires exact equality against the instrumented main build for both fresh and append at genuine golden depths 511 and 8191, using each installed Q4_0 and Q8_0-r16 model: eight comparisons, no tolerance. The Q8 control imports the same compatible native fixture into both Q8 builds; it does not claim that Q4 and Q8 inference are identical. HTTP tests additionally hash the actual imported slot file. Both builds return `152e2df8583025c1db149573f7e2c538615ab304d995ce355b429caa581b8532`.

| Gate | Result / evidence |
| --- | --- |
| Cross-build native state and full logits | Exit 0; `identity-test-01/`, `identity-comparison.json`; eight fresh/append identities match |
| Actual HTTP slot import / refusals | Both builds exit 0; `http-before-02.log`, `http-after-02.log`; 18 named HTTP 400 cases preserve target and neighbor bytes, followed by valid import and 64/64 greedy continuation |
| Checksums, view/owning control, header bounds | Auto and forced portable codec commands exit 0; `codec-auto-02.log`, `codec-portable-02.log`; 134 binary body lengths cover block/chunk boundaries against independent portable SHA and one-shot XXH3 |
| Fingerprint/config/live refusal and proxy behavior | Final proxy suite exit 0, 69 tests; `proxy-unit-02.log`; existing fingerprint-mismatch test retained with valid control |
| Server components | Exit 0, 49 tests / 191 assertions, no failures/exceptions/skips; `server-component-01.log` |
| HTTP URL replay mode | Exit 0 against a real golden body; `url-smoke-01.log` |

HTTP cases cover version, model identity, gap/overlap, prefix hash, declared layout, session, truncation, both checksum trailers, corrupt payload, both header budgets, native truncation, native positions/layer layout, two recurrent cells and recurrent endpoint mismatch. Rejection checks compare actual slot files, including KV and recurrent data. The new tests have concise claim comments. Existing test names remain present.

| Mutant | What it narrows the gate to | Named failing test | Result / survivor bound |
| --- | --- | --- | --- |
| First SHA round constant changed | Accelerated compression must match independent portable SHA | `test_streaming_checksum_block_chunk_and_binary_boundaries` | Exit 1, killed |
| Ignore SHA trailer | XXH agreement alone must not admit the body | `corrupt-body-hash` | Exit 1, killed |
| Ignore XXH trailer | SHA agreement alone must not admit the body | `corrupt-section-hash` | Exit 1, killed |
| Change the emitted full-logit identity after real native execution | Cross-build comparator must reject unequal identities | `test_cross_build_state_and_full_logits` | Exit 1, killed; `identity-control-01.log` |

All three compiled mutant compiles/links exit 0; expected-red executions really exit 1. Their control driver exits 0 (`mutant-record.json`, `controls-01.log`). The identity control executes the real native checker successfully, changes only the emitted logit hash, then the unittest comparator exits 1 with its intended identity-mismatch assertion. No survivor is claimed covered. These controls supplement the product refusals, not replace them.

Successful baseline configure/build and candidate/incremental builds exit 0. The first attempted build was refused by a stricter 28 GiB admission check. A later attempt stopped its own recorded descendants when a concurrent host disk fluctuation crossed the requested 28 GB floor; its driver exits 1 and make logs show termination. The exact triggering free-byte sample was not persisted, so no global minimum is claimed for that stopped attempt. Successful recorded builds stayed above 28,000,000,000 bytes; the lowest recorded successful-build sample is 28,047,904,768 bytes. This distinguishes the requested GB floor from GiB. The first profiling pilot exits 1 because the test launcher omitted the server's required slot-save path (HTTP 501); corrected replay exits 0. No failed attempt is counted green.

Repository model-naming workflow, Python syntax, added-line/new-file ASCII, shell syntax and whitespace checks exit 0. `ty` and `editorconfig-checker` readiness commands exit 127; those checks are unrun. No tool was installed. Full upstream/all-backend CI, actual GPU/Metal imports, Apple compilation and real 27B numerical/performance qualification are unrun. No surface table was supplied (brief gap); no requested contract row is treated as out of scope. Independent review remains pending.

## Metal differences and M3 Max projection

Inspection of `ggml-metal-device.m` shows shared Metal buffers use memcpy for each range in the existing batched call. Private buffers encode the batch into one blit command buffer and wait once. Many small independent `tensor_set` calls would be costly; this import path already uses the batch interface. Fragmentation, transposed layouts and other backend paths can change range counts. Shared-memory CPU results do not establish private Metal blit or synchronization speed. An active GPU queue, allocator pressure or compressed/swap memory can dominate on the Mac.

Conditional budget for the owner's roughly 2.004 GB transfer in four chunks: assume CommonCrypto+XXH3 at 1-3 GB/s, decoder loopback/assembly at 1-3 GB/s, the remaining two body copies at 10-30 GB/s, tensor writes and roughly 471 MB of recurrent backups at 5-20 GB/s, metadata/dispatch fixed overhead <=0.5 s and additional waits <=0.8 s. Sum: roughly 2-7 s of import, including a small allowance for heterogeneous rebinding. These are assumed effective rates, not M3 measurements. The CPU speedup ratio alone does not justify applying the same ratio to the old Mac timing.

If those assumptions hold, replacing 42.4 s of the original 201.7 s TTFT gives about 161-166 s total, with import around 1-4% instead of 21%. The 123.2 s 5G transfer, 10.6 s prefill and other measured overhead stay in that arithmetic. A slower CommonCrypto path, loopback, private-buffer transfer or memory pressure invalidates this budget. Acceptance requires actual Mac runs; no Mac gain is claimed delivered here.

## Exact reproduction on agents1

Run from this delivered source. Reuse existing local tools/models. Baseline patch is diagnostic only; apply it to pinned main in a separate checkout, never to the candidate. A locally installed CMake can replace `cmake` below. Serialize both builds with the existing shared lock on agents1; the task used Python `fcntl.flock` on `~/Werkstatt/hybrid-models/.build.lock` and a 28,000,000,000-byte disk admission/watch guard.

```sh
export PYTHONDONTWRITEBYTECODE=1
R="${IMPORT_RUN_DIR:-$PWD/.temp/import-reproduction}"
export TMPDIR="$R/tmp"
mkdir -p "$TMPDIR"
case "${IMPORT_DEVICE:-cpu}" in
    cpu) backend_flags=(-DGGML_CUDA=OFF -DGGML_METAL=OFF) ;;
    cuda) backend_flags=(-DGGML_CUDA=ON -DGGML_METAL=OFF -DCMAKE_CUDA_ARCHITECTURES=native) ;;
    metal) backend_flags=(-DGGML_CUDA=OFF -DGGML_METAL=ON) ;;
    *) exit 2 ;;
esac
git worktree add --detach "$R/baseline" a25754271e19cfda2c44c702a52b752cb53cf623
git -C "$R/baseline" apply --unidiff-zero "$PWD/tools/hybrid/import-profile-baseline.patch"
for arm in before after; do
    src="$PWD"
    test "$arm" != before || src="$R/baseline"
    cmake -S "$src" -B "$R/$arm-build" -DCMAKE_BUILD_TYPE=Release \
        "${backend_flags[@]}" -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_APP=OFF
    nice -n 19 cmake --build "$R/$arm-build" -j2 --target llama-server llama-hybrid-check-client llama-hybrid-check-codec
done
MODEL="$HOME/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf"
GOLDEN="$HOME/Werkstatt/hybrid-models/golden/prompt-512"
python3 tools/hybrid/profile_import.py --binary before="$R/before-build/bin/llama-server" \
    --binary after="$R/after-build/bin/llama-server" --model "$MODEL" --golden "$GOLDEN" --output "$R/fresh"
python3 tools/hybrid/profile_import.py --binary before="$R/before-build/bin/llama-server" \
    --binary after="$R/after-build/bin/llama-server" --model "$MODEL" --golden "$GOLDEN" --append --output "$R/append"
python3 tools/hybrid/test_import_identity.py --before "$R/before-build/bin/llama-hybrid-check-client" \
    --after "$R/after-build/bin/llama-hybrid-check-client" --model "$MODEL" \
    --model "$HOME/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-r16.gguf" \
    --golden "$GOLDEN" --golden "$HOME/Werkstatt/hybrid-models/golden/prompt-8192" --artifacts "$R/identity"
"$R/after-build/bin/llama-hybrid-check-codec"
HYBRID_HASH_PORTABLE=1 "$R/after-build/bin/llama-hybrid-check-codec"
```

## GPU host and Mac commands

Unrun here. Use Bash, existing model files, and the existing authenticated prefill node. Set the paths/SSH alias to their real local values. Keep the same GGUF on the producing node and both decoder arms; identity mismatch must refuse. All data/log paths are private. The replay does no model download and does not expose the node's credential.

Before the common build block above, set `export IMPORT_DEVICE=cuda IMPORT_RUN_DIR="$PWD/.temp/import-device"` on the GPU host or `export IMPORT_DEVICE=metal IMPORT_RUN_DIR="$PWD/.temp/import-device"` on the Mac. This selects the exact backend flags and keeps build/output paths consistent. Use the same compiler/toolchain/settings for both arms. Only one decoder is resident at a time in the loop below; model startup is outside the import timer. Do not overwrite an existing result directory. The agents1 0.8B/golden commands following the build block are for agents1; use the device replay below on GPU/Mac.

On the GPU host, produce one genuine large fresh 27B delta (8000 tokens is about 435 MB with the measured 27B geometry), then keep a loopback replay source alive:

```sh
umask 077
export R="$PWD/.temp/import-device"
export TMPDIR="$R/tmp" PYTHONDONTWRITEBYTECODE=1
export PREFILL_URL=http://127.0.0.1:8091
export NODE_TOKEN_FILE=/absolute/path/to/existing/client-tokens.json
export IMPORT_CLIENT=bench
mkdir -p "$R/wire" "$TMPDIR"
python3 - <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, 'tools/hybrid-proxy')
from proxy import Node
from store import prefix_hash
tokens = json.loads(Path(os.environ['NODE_TOKEN_FILE']).read_text())
node = Node(os.environ['PREFILL_URL'], tokens[os.environ['IMPORT_CLIENT']], timeout=7200)
health = node.request('GET', '/health')
sid = node.request('POST', '/v1/sessions', {'model_sha256': health['model_sha256']})['session_id']
try:
    wire = node.request('POST', '/v1/sessions/' + sid + '/extend',
                        dict(n0=0, prefix_hash_n0=prefix_hash([]), tokens=[1] * 8000, return_delta=True), binary=True)
    destination = Path(os.environ['R']) / 'wire/fresh.delta'
    with destination.open('xb') as out:
        out.write(wire)
finally:
    node.request('DELETE', '/v1/sessions/' + sid)
PY
python3 -m http.server 18991 --bind 127.0.0.1 --directory "$R/wire" > "$R/wire-source.log" 2>&1 &
IMPORT_SOURCE_PID=$!
printf '%s\n' "$IMPORT_SOURCE_PID" > "$R/wire-source.pid"
```

With `R` pointing to the device build/output directory and `MODEL` pointing to that machine's existing identical GGUF, define this function in both Bash sessions. GPU replay measures loopback receive; Mac URL replay measures the actual transferred body over the tunnel before each import. Each Mac invocation performs one setup download and one measured download; the six invocations transfer twelve bodies.

```sh
MODEL=/absolute/path/to/existing/27b.gguf
run_import_pairs() {
for pair in 1 2 3; do
    arms="before after"
    test "$pair" != 2 || arms="after before"
    for arm in $arms; do
        python3 tools/hybrid/profile_import.py --binary "$arm=$R/$arm-build/bin/llama-server" \
            --model "$MODEL" --gpu-layers 99 "${SOURCE_ARGS[@]}" --runs 1 --output "$R/pair-$pair-$arm"
    done
done
}
```

On the GPU host:

```sh
SOURCE_ARGS=(--wire "$R/wire/fresh.delta")
run_import_pairs
```

Mac tunnel in the same Bash session (use the already configured GPU SSH alias):

```sh
GPU_SSH_ALIAS=your-existing-gpu-alias
ssh -N -L 18991:127.0.0.1:18991 "$GPU_SSH_ALIAS" > "$R/tunnel.log" 2>&1 &
IMPORT_TUNNEL_PID=$!
printf '%s\n' "$IMPORT_TUNNEL_PID" > "$R/tunnel.pid"
SOURCE_ARGS=(--url http://127.0.0.1:18991/fresh.delta)
python3 - <<'PY'
import time, urllib.request
deadline = time.monotonic() + 60
while True:
    try:
        request = urllib.request.Request('http://127.0.0.1:18991/fresh.delta', method='HEAD')
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
        break
    except OSError:
        if time.monotonic() >= deadline:
            raise
        time.sleep(0.2)
PY
run_import_pairs
kill "$IMPORT_TUNNEL_PID"
wait "$IMPORT_TUNNEL_PID" || test "$?" -eq 143
```

Keep the tunnel alive through the paired loop; the cleanup lines follow it. After the Mac run, stop the GPU source by its recorded `IMPORT_SOURCE_PID`, then remove only the task-owned wire file/build trees/worktree after retaining source/binary hashes, `measurements.json` and decoder logs. Return all six device records, not only medians. Inspect `sha256_backend=commoncrypto` on the Mac and compare checksum, copy, tensor-write, synchronization and residual stages independently. A busy/mismatched/invalid import is a refusal, not a performance sample.

## Custody and limits

Task tracker: existing `.temp/tasks.md`; no configured task-board, so no board status or primary-goal mutation. Research, failed commands, controls and raw measurements remain under `R`. Closed HTTP snapshots are hashed before removal (`early-cleanup.json` and per-run slot manifests). Large synthetic bodies never touch disk. Final cleanup and source/binary/PID evidence are recorded under `R`; pre-existing builds, models, fixtures and other sessions are preserved. No publication, GPU execution, model download or upstream submission was performed.

Final cleanup exits 0: all 66 recorded child PIDs inactive, both task-owned build/binary trees removed (117,381,962 file bytes including mutant objects/binaries), and the remaining 159,830,676 snapshot bytes removed after hashing. Earlier cleanup removed 296,109,108 snapshot bytes. Free disk readback: 29,469,147,136 bytes. Small logs, control sources, hashes and measurements remain. `final-audit.json` binds runtime sources and binaries before deletion; runtime source hashes still match those used for profiling. Both arms' `libllama.so` hashes are identical; engine numerical operations were not changed. Binaries were built from the recorded base plus the reviewed working-tree delta before the local signed commit; source hashes identify that delta exactly.

## r35: rebase after PR #7

Run: hybrid-b-r35-261101. Base: `528724c96894aa5f65e98b329317de59aa694afc`, fetched from the authorized GitHub repository. `m3-import-speed-v2` is a signed rebase of r34; `m3-import-speed` remains at `765afe4d2e7833ff816283ced42fd66b9591d1ef`. No push. PR #7 (`41514e77`) and its owner-qualified Mac path are ancestors. This section supersedes r34's copied-code packaging and pending-main statements; the old measurements above remain historical.

The conflict was only in `hybrid-state.cpp`. Extracted main's backend state/init/update/finalize dispatch into one internal `sha256_stream` API and routed both `file_hash` and the fused delta traversal through it. Main's CPUID, SHA-NI compression and partial-block update functions are byte-identical; no kernel was added by this branch. There is exactly one CommonCrypto Init/Update/Final site. The existing `file_hash(path, read_bytes = 65536, backend = nullptr)` interface, 1..65536 read bound, read-error refusals and backend reporting remain intact. The bounded CommonCrypto updates keep the 32-bit length guard. PR #7's vendored SHA optimization and its patch provenance are inherited unchanged.

The fused checksum loop, all decoder requirements, borrowed reader, server/HTTP copy hunks, native engine/timers, codec tests, identity checker and profiling tool are byte-identical to r34. The task continues to own the wire until synchronous native application ends. Both trailers cover every byte before application. Two body copies remain removed. `source-contract-audit.json` records these comparisons. The inherited main changes include the deferred shadow-sync/restart regression; they were not reverted.

### File-by-file rebase delta

Compare r34 with this branch, excluding unchanged r34 files:

| File | Delta and ownership |
| --- | --- |
| `README.md` | Inherit main's two hash-tool entries. |
| `scripts/sync_vendor.py` | Inherit main's vendor SHA patch registration. |
| `tools/hybrid-bench/RESULTS-hybrid-a-r14-261101.md` | Inherit main's portable-hash evidence. |
| `tools/hybrid-proxy/README.md` | Inherit main's shadow-sync/restart documentation. |
| `tools/hybrid-proxy/RESULTS-hybrid-a-r26-261101.md` | Inherit main's restart regression evidence. |
| `tools/hybrid-proxy/RESULTS-hybrid-d-r4-261101.md` | Inherit main's deferred-sync evidence. |
| `tools/hybrid-proxy/UNRESOLVED_QUESTIONS.md` | Inherit main's deferred-sync note. |
| `tools/hybrid-proxy/bench_local.py` | Inherit main's local-kept benchmark. |
| `tools/hybrid-proxy/proxy.py` | Inherit deferred shadow sync; retain r34's memoryview join. |
| `tools/hybrid-proxy/test_benchmark.py` | Inherit restart/deferred-sync tests; retain r34 tests. |
| `tools/hybrid-proxy/test_proxy.py` | Inherit main's explanatory assertion comment. |
| `tools/hybrid-proxy/test_decode.py` | Add optional `--reference-slot` and `test_http_import_matches_reference_slot`; compare actual applied HTTP slot bytes with separate-main output. |
| `tools/hybrid/CMakeLists.txt` | Inherit PR #7's check-hash target. |
| `tools/hybrid/README.md` | Inherit hash documentation; document the shared API, baseline three-way application, shasum reference and HTTP comparison switch. |
| `tools/hybrid/RESULTS-hybrid-a-r23-261101.md` | Inherit PR #7's SHA-NI evidence. |
| `tools/hybrid/RESULTS-hybrid-b-r34-261101.md` | Append this r35 report; preserve r34 evidence. |
| `tools/hybrid/UNRESOLVED_QUESTIONS.md` | Record the rebased CommonCrypto Mac qualification gate. |
| `tools/hybrid/check-hash.cpp` | Inherit PR #7 unchanged, including portable-stream controls. |
| `tools/hybrid/hybrid-state.cpp` | Resolve include conflict and retain main's kernels; preserve its file_hash arguments, validation and backend output around the shared streaming state. Delta/server behavior unchanged from r34. |
| `tools/hybrid/hybrid-state.h` | Retain main's defaulted three-argument file_hash declaration and r34's borrowed reader declarations. |
| `tools/hybrid/test_file_hash.py` | Inherit PR #7; use macOS `shasum -a 256` if `sha256sum` is absent. All test names and digest assertions retained. |
| `tools/hybrid/test_file_hash_ni.py` | Inherit PR #7 unchanged. |
| `vendor/hash/sha256/sha256-blocks.patch` | Inherit PR #7's vendor patch unchanged. |
| `vendor/hash/sha256/sha256.c` | Inherit PR #7's vendor optimization unchanged. |

All other r34 files retain their reviewed content. This is a fork-local signed delivery with the requested Codex trailer, without upstream submission or publication.

### CPU verification

CMake 4.4.4, GCC 15.2.0, Release, CPU only on agents1. Both builds were niced to 19, serialized under the shared model-cache build lock and limited to `-j4`. On-disk TMPDIR and `PYTHONDONTWRITEBYTECODE=1` were used. Baseline: exact main `528724c9` plus the six-file r34 timer/identity patch, applied with `git apply --3way --unidiff-zero`; the preserved `baseline-instrumentation.patch` changes no hashing/copy/admission behavior. Candidate runtime sources are the signed rebased commit `b0cda0d33e195e85e363986e60c926032090b7aa`; the subsequent commit changes tests/docs only. `final-audit.json` binds both arms' sources and binaries. Engine sources match, while library hashes may differ due to build-path strings.

All commands below exited 0 in both outer modes (unset `HYBRID_HASH_PORTABLE`, and `HYBRID_HASH_PORTABLE=1`). Exact argv, PIDs, exit codes, environment mode and sampled free disk are in `.temp/hybrid-b-r35-261101/*-01.json`; complete stdout/stderr is in the corresponding `.log`. Let `B=.temp/hybrid-b-r35-261101/candidate-build/bin`, `M=$HOME/Werkstatt/hybrid-models`, `G=$M/golden/prompt-512`.

| Command / gate | Evidence in each mode |
| --- | --- |
| `python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py'` | 73 tests, no skips; includes inherited main regressions and all r34 tests. |
| `$B/llama-hybrid-check-codec` | 23 PASS; original named checksum, borrowed/owning and refusal tests retained. |
| `python3 tools/hybrid/test_file_hash.py --binary $B/llama-hybrid-check-hash --model $M/Qwen3.5-0.8B-Q4_0.gguf -v` | 4 tests, no skips; real GGUF, fragmented vendor control and independent digests. |
| `python3 tools/hybrid/test_file_hash_ni.py --binary $B/llama-hybrid-check-hash -v` | 8 tests, no skips; NIST, all block/padding/read boundaries, unaligned/empty vendor updates, sparse 4 GiB + 65 bytes, byte mutations, dispatch/read errors/read bounds. Harness explicitly tests auto and forced portable internally on both invocations. Observed `sha-ni` / `portable`. |
| `test_file_hash.py` with PATH containing only the real `shasum` reference tool | Additional auto run: all 4 tests pass, including the real GGUF; fallback verified without a fake digest utility. |
| `$B/test-server-component` | 49 tests / 191 assertions; zero failures, exceptions or skips. |
| `$B/llama-hybrid-check-client $M/Qwen3.5-0.8B-Q4_0.gguf --capacity $G` | Capacity failure and rollback controls pass. |
| `$B/llama-hybrid-check-append MODEL GOLDEN_512 GOLDEN_2048 GOLDEN_8192` | 54 PASS, all three depths; guards also pass 4/4 via `llama-hybrid-check-guards MODEL GOLDEN_512`. |
| `test_import_identity.py --before BASE/bin/llama-hybrid-check-client --after $B/llama-hybrid-check-client --model Q4 --model Q8-r16 --golden GOLDEN_512 --golden GOLDEN_8192 --artifacts OUTPUT` | 8/8 native state and 64-position full-vocabulary logit identities match main in each mode, fresh/append; `auto-identity-01/result.json`, `portable-identity-01/result.json`. |
| `test_decode.py --binary $B/llama-server --model MODEL --golden GOLDEN_512 --artifacts OUTPUT --reference-slot BASE_HTTP/slots/imported.bin` | All 18 HTTP 400 cases preserve target/neighbor; valid import matches every main slot byte and gives 64/64 greedy continuation. The separate baseline run also passes. Slot SHA remains `152e2df8583025c1db149573f7e2c538615ab304d995ce355b429caa581b8532`. |
| r7i `http_probe.py`, unchanged script copied into this run | Main auto plus v2 auto/portable, both models, depth 8192, fresh/append: 72/72 checksum refusals with unchanged slot bytes (24 main, 48 candidate). All 12 valid imported slot digests agree across arms/modes per model. `probe-comparisons.json`; snapshots hashed before removal. |
| HTTP URL replay via `profile_import.py --url ... --runs 1` | Real golden-body replay passes in both modes. |

Mutants are compiled/relinked outside the normal candidate sources and binaries. Compile/link steps exit 0; credited negative executions exit 1 for the intended contract, paired with green positive controls. All original reviewer control names are retained. No credited mutant survives.

| Mutant | What it narrows the gate to | Named failing test | Result / bound |
| --- | --- | --- | --- |
| `sha-round` | SHA-NI compression matches the portable reference | `test_streaming_checksum_block_chunk_and_binary_boundaries` | Killed, auto exit 1. Portable does not execute this kernel, so that mode is not credited for this control. |
| `skip-sha` | SHA trailer is mandatory even if XXH agrees | `corrupt-body-hash` | Killed, auto/portable exit 1. |
| `skip-xxh` | XXH trailer is mandatory even if SHA agrees | `corrupt-section-hash` | Killed, auto/portable exit 1. |
| `m1-skip-sha-chunk` | SHA covers chunks after the first | `test_streaming_checksum_block_chunk_and_binary_boundaries` | Killed, auto/portable exit 1. |
| `m1b-skip-xxh-last-chunk` | XXH covers the last chunk | `test_streaming_checksum_block_chunk_and_binary_boundaries` (XXH assertion) | Killed, auto/portable exit 1. |
| `m2-second-half-unhashed` | Independent full-body reference defeats two agreeing partial digests | `test_streaming_checksum_block_chunk_and_binary_boundaries` | Killed, auto/portable exit 1. |
| `m3-apply-before-verify` | Refused corrupt input leaves readable, unchanged state | HTTP `section-checksum`, post-refusal slot assertion | Killed, auto/portable exit 1: post-refusal slot unreadable, HTTP 400. |
| `m4-mutate-after-verify` | Applied HTTP bytes equal independently verified/imported bytes, including unused rows | `test_http_import_matches_reference_slot` | Killed, auto/portable exit 1. This was r7i's survivor; the new optional reference assertion closes its observed gap. |
| Emitted-logit identity comparator control | Comparator rejects changed logits after actual native execution | `test_cross_build_state_and_full_logits` | Killed, exit 1; native child itself exits 0. |

Raw controls: `mutants/records.json`, `mutants2/records.json`, `identity-control-01.log`. The first m3 setup appended `server-context.o` beside the original archive member rather than replacing `server-context.cpp.o`. Its test exited 1 for accepting bad input, not for the intended verify-after-apply invariant; the driver correctly failed and that attempt is not credited. The corrected relink verifies the archived member bytes against the compiled object and then produces the named ordering failure. Exact green m3 compilation was reused; no product source needed correction.

One paired run per workload, after our builds and correctness jobs ended, same synthetic body as r34: 315,409,012 bytes. Raw measurements and stage events: `speed-fresh-01/measurements.json`, `speed-append-01/measurements.json`.

| Workload | Main ms / MB/s | v2 ms / MB/s | Speedup | Checksum main -> v2 ms | Copy main -> v2 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh | 1252.97 / 251.73 | 732.19 / 430.78 | 1.71x | 579.33 -> 155.12 | 352.12 -> 157.90 |
| Occupied-prefix append | 1158.62 / 272.23 | 614.37 / 513.38 | 1.89x | 515.01 -> 137.02 | 316.17 -> 146.01 |

These are single pairs, not medians. The baseline now includes PR #7's faster vendored portable SHA, unlike r34's original base. The candidate retains r34's throughput gains and two removed body copies against this faster main. Peak RSS was not remeasured in r35: r7i's approximately 630 MB reduction is reused for the unchanged copy/ownership hunks, rather than represented as a new measurement. This CPU fixture gives no new M3 forecast; the earlier 2-7 s estimate remains conditional best-case arithmetic, not acceptance evidence.

Model-naming, Python syntax, added-line ASCII and whitespace checks pass (exit 0). `editorconfig-checker` and `ty` readiness exit 127; those checks remain unrun. Bare CMake was absent from PATH; the already installed r34 toolchain was used, without an install. No GPU, download, upstream submission or push. No surface table was supplied; no requested contract row is excluded. Mac/CommonCrypto qualification of this rebased head is the remaining platform landing gate.


Cleanup exits 0: 146 recorded child PIDs inactive; both task-owned build trees, compiled mutants, large slot snapshots and the instrumented baseline worktree removed after preserving hashes and its patch. Removed build/binary/snapshot file bytes: 1,315,461,010, plus the baseline source checkout. The lowest recorded gate sample was 33,569,853,440 free bytes; cleanup readback was 35,142,762,496 bytes, above the 28 GB floor. Small logs, sources, manifests and the signed delivery bundle remain under `.temp/hybrid-b-r35-261101/`.

### Mac lane: mandatory before landing

Unrun on agents1. PR #7's earlier M3 qualification does not qualify the extracted streaming CommonCrypto state on this rebased head. Give the Mac lane the signed Git bundle retained with the handoff, the full head SHA, and an existing local GGUF path. Fetch the feature head from the bundle; it has not been pushed. The build enables Metal to match r7i's requested server build; the tests below execute codec/hash code only and need no GPU inference or download. Use Bash; set the three inputs before running. A failed command blocks landing.

```sh
set -euo pipefail
: "${R35_BUNDLE:?absolute path to the transferred m3-import-speed-v2.bundle}"
: "${EXPECTED_R35_HEAD:?full signed head from the orchestrator handoff}"
: "${R35_MODEL:?absolute path to an existing local GGUF}"
git fetch https://github.com/relux-works/hybrid-inference main
git fetch "$R35_BUNDLE" refs/heads/m3-import-speed-v2:refs/remotes/r35/m3-import-speed-v2
test "$(git rev-parse refs/remotes/r35/m3-import-speed-v2)" = "$EXPECTED_R35_HEAD"
git verify-commit "$EXPECTED_R35_HEAD"
git merge-base --is-ancestor 528724c96894aa5f65e98b329317de59aa694afc "$EXPECTED_R35_HEAD"
R="$PWD/.temp/hybrid-b-r35-mac"
test ! -e "$R"
mkdir -p "$R/tmp" "$R/logs"
export TMPDIR="$R/tmp" PYTHONDONTWRITEBYTECODE=1
unset HYBRID_HASH_PORTABLE
git worktree add --detach "$R/worktree" "$EXPECTED_R35_HEAD"
cd "$R/worktree"
mkdir -p .temp
cmake --version > "$R/logs/cmake-version.log"
c++ --version > "$R/logs/compiler-version.log"
python3 --version > "$R/logs/python-version.log"
cmake -S . -B "$R/build" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF -DGGML_METAL=ON -DGGML_VULKAN=OFF \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_APP=OFF \
    > "$R/logs/configure.log" 2>&1
nice -n 19 cmake --build "$R/build" -j4 \
    --target llama-server llama-hybrid-check-codec llama-hybrid-check-hash \
    > "$R/logs/build.log" 2>&1
for mode in auto portable; do
    if test "$mode" = portable; then export HYBRID_HASH_PORTABLE=1; else unset HYBRID_HASH_PORTABLE; fi
    nice -n 19 "$R/build/bin/llama-hybrid-check-codec" > "$R/logs/codec-$mode.log" 2>&1
    nice -n 19 python3 tools/hybrid/test_file_hash.py \
        --binary "$R/build/bin/llama-hybrid-check-hash" --model "$R35_MODEL" -v \
        > "$R/logs/hash-$mode.log" 2>&1
    nice -n 19 python3 tools/hybrid/test_file_hash_ni.py \
        --binary "$R/build/bin/llama-hybrid-check-hash" -v \
        > "$R/logs/hash-ni-$mode.log" 2>&1
done
unset HYBRID_HASH_PORTABLE
git rev-parse HEAD > "$R/logs/head.txt"
git status --short > "$R/logs/status.txt"
```

Return the exact SHA, platform/tool versions, build/codec/hash exit codes and complete logs, including `test_dispatch` output `automatic=commoncrypto`, `forced=portable`. The NI harness intentionally selects both modes internally even when invoked from the portable outer mode. The sparse >4 GiB fixture is sparse on disk; retain the disk floor. Preserve logs and binary identities before removing the Mac-owned build tree/worktree. Signature verification requires the machine's configured public signer trust; do not substitute identities. No landing is authorized by an agents1-only green result.


## r31: merge current main for landing (2026-11-02)

Exact tested signed merge head: `a0149dc769aff5665cf0ba0cb7ea7c3d4516d896`. Parents, in order: reviewed `dba3c14c8dc27c25ca505acc33e594b48e3a4a3c` and main `683caeed94c64ad69699d5f69a8fb533d4de6737` (PR #14). Remote `b` was fetched from dev-b's local workbench. Both source commits above `528724c96894aa5f65e98b329317de59aa694afc` verify with valid signatures; the reviewed objects were preserved. Muse r7o CPU acceptance is owner-supplied source qualification. R31 self-review and CPU evidence cover the merge resolutions. The following report commit changes documentation only; its exact delivery head is recorded in `.temp/hybrid-a-r31b-261102/delivery.json` and the orchestrator handoff. No push or PR submission was performed.

### Landing title and body

Title: `hybrid : accelerate chunked delta import hashing and remove body copies`

Body: Apply the reviewed streaming SHA-256/XXH3 implementation and borrowed delta reader to protocol-2 chunk imports on current main. Each staged chunk owns its request string until final synchronous application, so body views remain valid without extracting another body copy. Preserve PR #14's atomic staging, abort/replacement handling, SWA layout, and fixed state only in the final chunk. Wire bytes, both checksum trailers, imported state and continuation logits match main.

CPU build, codec, full-file hash, hash-NI, import, chunk/SWA, proxy and profiling gates pass in automatic and forced-portable modes. The reviewed source passed its M3 Max CommonCrypto/portable gate on 2026-11-02. Exact tested merge head: `a0149dc769aff5665cf0ba0cb7ea7c3d4516d896`; signed local delivery only.

### Every conflict and its resolution

Five files had eight conflict regions. `.temp/hybrid-a-r31-261102/conflicts-01.patch` preserves all original regions; `resolutions.md` records the full adaptation ledger.

| File | Resolution |
| --- | --- |
| `README.md` | Retain main's combined SWA/chunk qualification link and the branch's import-profiling tool entry. |
| `src/llama-context.cpp` | Retain main's SWA-aware backup condition inside the existing timing scope; retain main's `read_append` layout/range/endpoint validation inside the parse timing scope. |
| `tools/hybrid/UNRESOLVED_QUESTIONS.md` | Retain main's qualification limits and the source Mac-gate history; the r31 note records the returned gate. |
| `tools/hybrid/hybrid-state.cpp` | Retain both memory and timing includes. Keep main's bounded header reader, protocol 2 and body format v3 validation inside the borrowed reader; keep the owning compatibility wrapper. |
| `tools/server/server-context.cpp` | Retain main's staging, abort, final-only apply and token bookkeeping. Retain the task-copy timing scope and main's abort flag. Replace the superseded one-shot import block with main's transaction. |

Required adaptations outside conflict markers: `delta_import` takes the wire by value and the server moves its task string into it; heap-owned strings keep body-view addresses stable as vectors grow. Clear destroys views before their owners. The existing chunk test now destroys/reuses moved request buffers before application and still requires exact state/logit equality. The HTTP version-refusal fixture uses version 1, since version 2 is valid. The profiling helper describes the new transfer bounds, retains checksum/native-stage timings, and reports zero full recurrent backup on the chunk path; metadata recovery closure overhead stays in the residual. No admission limit, tensor layout, native transaction, checksum, or acceptance threshold was weakened.

### CPU gates and identity evidence

Evidence root: `.temp/hybrid-a-r31-261102/`. Each named command has complete `.log` and `.json` records with argv, actual exit code, PID, environment, times and sampled free disk. CMake 4.4.4 and GCC 15.2.0, Release/non-native AVX2/FMA, CPU only; CUDA/Metal/Vulkan and other GPU backends disabled. Builds were serialized under the shared model-cache lock, checked with `pgrep -u agent cc1plus` and `uptime`, niced to 19 and limited to `-j4`. `TMPDIR` was on disk and `PYTHONDONTWRITEBYTECODE=1`. One candidate build serves the two runtime hash modes; no duplicate build is claimed.

Reference runtime is exact main `683caeed`. Only `check-client.cpp` and `check-append.cpp` were instrumented in the isolated reference worktree to emit observations; `main-test-harness.patch` preserves the exact delta, with no runtime change. The candidate observation checker uses the same source. Legacy installed golden headers were wrapped as protocol 2/body v3; every body and both trailers remain unchanged (`golden-conversion.json`). Q8 identity checks use the same compatible native golden input on both builds; they do not claim equality between Q4 and Q8 inference.

Let `B=.temp/hybrid-a-r31-261102/build/bin`, `R=.temp/hybrid-a-r31-261102`, `M=$HOME/Werkstatt/hybrid-models`, `G=$R/golden`. Every gate below exits 0 in both modes: unset `HYBRID_HASH_PORTABLE`, and `HYBRID_HASH_PORTABLE=1`.

| Command / gate | Result per mode |
| --- | --- |
| `$B/llama-hybrid-check-codec` | 25 PASS; original borrowed/owning, streamed digest, header-budget and refusal controls retained. |
| `python3 tools/hybrid/test_file_hash.py --binary $B/llama-hybrid-check-hash --model $M/Qwen3.5-0.8B-Q4_0.gguf -v` | 4 tests, no skips; real GGUF and independent hashlib/sha256sum digests. |
| `python3 tools/hybrid/test_file_hash_ni.py --binary $B/llama-hybrid-check-hash -v` | 8 tests, no skips; NIST, boundaries, read errors, dispatch, mutations and sparse 4 GiB + 65-byte fixture. Observed `sha-ni` / `portable`; harness also selects both modes internally. |
| `$B/test-server-component` | 50 tests, 235 assertions; no failures, exceptions or skips. |
| `$B/llama-hybrid-check-swa $R/MODE-tiny.gguf` | 192 PASS, including allocated-cache windows, physical layout, rollback, interruption, legacy refusal and chunk/single equivalence. |
| `$B/llama-hybrid-check-client MODEL --capacity $G/prompt-512` and `$B/llama-hybrid-check-guards MODEL $G/prompt-512` | Capacity failure/neighbor preservation and valid positive control; guards 4/4. |
| `$B/llama-hybrid-check-append MODEL $G/prompt-512 $G/prompt-2048 $G/prompt-8192` | 54 PASS, including exact refusal state and 64-token greedy continuations at each depth. |
| `test_import_identity.py --before $R/reference/bin/llama-hybrid-check-client --after $B/llama-hybrid-check-client --model Q4 --model Q8-r16 --golden $G/prompt-512 --golden $G/prompt-8192 --artifacts OUTPUT` | Four model/depth comparisons covering eight fresh/append state and 64-position full-vocabulary logit identities; all exact. |
| `test_decode.py --binary $B/llama-server --model MODEL --golden $G/prompt-512 --artifacts OUTPUT --reference-slot $R/main-http/slots/imported.bin` | 18 HTTP 400 refusals preserve target/neighbor; actual imported slot bytes equal main; 64/64 greedy continuation. Slot SHA-256 remains `152e2df8583025c1db149573f7e2c538615ab304d995ce355b429caa581b8532`. |
| `test_remote_append.py --binary $B/llama-server --prefill-binary $B/llama-hybrid-prefill --model MODEL --artifacts OUTPUT` | Real HTTP chunk interruption, malformed order/fixed-state cases, abort, abandoned-transfer replacement and final commit controls pass. |
| `python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py'` | 123 tests, no skips; inherited restart, chunk persistence and recovery checks retained. |
| `llama-hybrid-check-chunk-identity MODEL --chunks` on Q4/Q8 | Existing chunk checker plus moved-request lifetime regression: 42 PASS per model, fresh/append. Compare main's 24 files byte-for-byte and by SHA-256 per mode, including wire envelopes, final state and held-token logits; 16 chunk trailers also match independent XXH3/hashlib references. Fixed state occurs only in chunk 3 of 4. |
| `http_probe.py` at depth 8191 on Q4/Q8, fresh/append | 24 checksum refusals per mode, unchanged slot bytes. All valid imported slot digests match main; baseline adds 24 refusals, for 72 total. Only prefix-transfer metadata was adapted from the r35 probe; mutation offsets and rejection assertions are unchanged. |
| `profile_import.py --url ... --runs 1`, plus synthetic fresh/append at 1024 tokens | Local HTTP golden replay and both synthetic protocol-2 transfers pass; no `state_copy` stage occurs. Smoke runs establish behavior, not a new throughput or RSS claim. |
| `test_integrated.py ... --case chunk` | PR #14 real short-delta slope and conservative chunk-budget test passes; no 79k-token prefill is claimed. |

`auto-suite-results.json`, `portable-suite-results.json`, `auto/portable-chunk-comparisons.json`, `probe-comparisons.json` and `final-audit.json` bind the results. Each mode has 17 suite gates, plus three profiling smoke gates and the integrated chunk gate. Across modes: 48 direct chunk byte identities, 32 independently checked chunk trailers and eight depth-8191 HTTP import digest identities.

| Mutant | What it narrows the gate to | Named failing test | Result / bound |
| --- | --- | --- | --- |
| Flip byte 1024 in an otherwise identical exported native-state fixture | Cross-build comparator must detect a one-byte state mismatch | `test_chunk_import_bytes_match_main: fresh-state.bin` | Expected-red exit 1, killed; identical fixture control exit 0. No survivor. |

Source audit confirms main's native chunk transaction differs only by opt-in timers and `delta_import::apply` is identical. The reviewed `sha256_stream` and streamed checksum function are unchanged; original reviewer control names remain. Source/binary hashes are in `validation-identity.json`; the final audit verifies they still match. No surface table was supplied (brief gap); no requested CPU contract is excluded. Rented GPU, production-model numerical qualification, Mac-over-5G timing, the separate inherited whole-prompt timing diagnostic, RSS and new performance benchmarking are outside this merge's requested CPU gates; no new result is claimed for them. Initial readiness failures and the corrected audit-boundary error remain recorded in `early-failures.md`, rather than counted as green.

### Recovery of interrupted run

Run `hybrid-a-r31b-261102` resumed the unchanged signed merge after the earlier runtime-capacity interruption. At 07:04:25Z, the merge, all 20 recorded source-file hashes and all 32 binary entries matched the earlier validation identity. Accelerated mode already had 17/17 green suite records; portable mode had 11/17, and both integrated chunk commands had complete exit-0 records. Those passing tests were reused. The partial `portable-proxy-01.log` had no exit record and is unrun evidence; the completed replacement is `portable-proxy-02.json` (123 tests, exit 0). The remaining portable native chunk checks, HTTP checksum probes and byte comparator use `-02.json` records; portable profile smoke gates use their first `-01.json` records. Every suite-summary row resolves to its actual command record. No build or source change was needed during recovery.

Fresh recovery evidence is under `.temp/hybrid-a-r31b-261102/`: `resume-identity.json`, `merge-comparison.json`, `mac-verification.json`, `command-audit.json`, and the recovery log. The independent comparison repeats the original alternate-index construction, checks all eight conflict regions in five files, and confirms both tree equality and byte equality of the main-to-merge patch. The compared patch SHA-256 is `6611ea8d7c68761d1740586d8f9f16de43e8fd47161f29675bb3ae5615b7c2db`. The final report delta is checked separately and contains only this results file and the Mac-gate readback note in `UNRESOLVED_QUESTIONS.md`.

### Reviewed diff plus resolutions

`reviewed-branch.patch` is exact `git diff --binary 528724c9..dba3c14c`. Applying it to main's tree using an alternate index reports the same five conflicted files (expected exit 1). Substituting only the nine paths named in `compare-merge-tree.json` with the listed resolutions/adaptations produces tree `5908af162d01d688e02c0561d57be9b95f430874`, exactly the signed merge tree (equality gate exit 0). Thus `git diff 683caeed..a0149dc769aff5665cf0ba0cb7ea7c3d4516d896` is the reviewed branch change carried onto current main plus the enumerated resolutions. The later report commit is verified separately as documentation-only; no unlisted runtime delta is hidden by the comparison.

### Returned Mac gate

On 2026-11-02, M3 Max/macOS 27.2 arm64 passed its Metal-enabled build and both codec/hash modes on exact reviewed head `dba3c14c8dc27c25ca505acc33e594b48e3a4a3c`: codec 23 PASS, full-file hash 4 tests, NI harness 8 tests, no skips; dispatch `automatic=commoncrypto`, `forced=portable`. Configure, build and all six test commands actually exited 0. Final `gate-progress.json` records completion at `2026-11-02T06:34:02.907737+00:00`; its final results, rather than the older running `STATUS.json`, are the gate evidence.

Archive: `/home/agent/lanes/hybrid/.temp/mac-results-261102/r35-mac-gate-evidence.tar.gz`. Verified full SHA-256: `901b27fe465b3a2216efe59144a312b6238b7fde4cfb12a4d68ff14d14f96c2a`; all 53 internal manifest entries match. Mac signature trust limitations and the source-owner amendment remain preserved in the archive; agents1 verifies the source signatures normally. This closes r35's returned CommonCrypto/portable source gate. R31 did not run a new Mac or GPU job; CPU merge tests and the unchanged hash implementation bound reuse of that source qualification.

Cleanup removes the owned main/candidate build trees and native snapshots after preserving identities, patches, logs and manifests. The lowest recorded command sample before report generation was 28,579,352,576 free bytes, above the 28 GB floor (28,000,000,000 bytes); cleanup readback is in `.temp/hybrid-a-r31-261102/cleanup.json`, and the exact signed delivery head is in `.temp/hybrid-a-r31b-261102/delivery.json`. No model files were removed, no downloads or GPU execution were used, and no push was performed. Ready for review/local landing.

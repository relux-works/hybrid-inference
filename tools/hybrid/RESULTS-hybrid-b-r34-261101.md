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

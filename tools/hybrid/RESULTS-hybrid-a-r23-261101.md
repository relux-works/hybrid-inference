# SHA-NI model hashing - hybrid-a-r23-261101

## Result and scope

Cached full-file SHA-256 throughput on agents1 increased from median 622.11 MB/s portable to 2198.15 MB/s SHA-NI: 3.53x throughput, 71.7% less hashing time. All six paired measurements returned the same digest with zero process disk-input blocks. Identity still covers every byte.

Continued local branch `m2-fast-hash` from the exact r6g-accepted PR #7 head `41eb6ca445d77d907e21a981b8f82758f006c784`. The local branch was already present and the checkout was clean. No fetch, download, GPU run or push was needed. This is a local fork prototype, with the user-requested signed commit and Codex co-author trailer; no upstream submission or session link.

The added compression function uses `_mm_sha256rnds2_epu32`, `_mm_sha256msg1_epu32` and `_mm_sha256msg2_epu32`, compiled only for non-Apple x86-64 GCC/Clang with `target("sha,ssse3,sse4.1")`. CPUID leaf 1 must report SSSE3 and SSE4.1 and leaf 7/subleaf 0 must report SHA. Unsupported builds/CPUs keep the vendored implementation. `HYBRID_HASH_PORTABLE=1` forces it; unset, empty, `0` and `true` preserve automatic selection.

The implementation keeps the vendor's state, 64-bit byte count, partial-block layout and finalizer. Complete blocks use SHA-NI; final padding uses the existing portable finalizer. Loads accept unaligned data and never read a partial block. Default reads stay at 65536 bytes. Every call opens and reads the whole file; identity remains ordinary SHA-256. CommonCrypto initialization, update, finalization and their return checks remain in the default macOS path. Existing hash tests, vendored SHA source/patch and CMake configuration are byte-identical to the base.

The hash checker now returns the backend actually selected, and `--file-chunk N` sends odd-sized native reads through that backend (1..65536). Its original positional chunk mode still tests unaligned portable streaming and empty updates. Optional `file_hash` parameters provide these observations; existing one-argument source callers retain their default behavior.

## Paired measurement

2026-11-01 UTC, agents1, Linux x86_64 KVM, 8 visible CPUs, AMD Ryzen 7 9700X 8-Core Processor. GCC/G++ 15.2.0 (Ubuntu 15.2.0-16ubuntu1), CMake 4.4.4, existing Release CPU build. Hash C++ flags: `-O3 -DNDEBUG -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi`. Vendor C uses `-O3 -DNDEBUG -fPIC` plus its existing warnings and per-source `-w`. No global `-msha`, `-mssse3`, `-msse4.1` or `-march` was added to the hash target. The compression function alone carries the ISA target attribute.

Dense disk fixture: 4000000000 bytes (4 GB decimal), 4000075776 allocated bytes, SHA-256 `ce6e4e524aa9678439d5eb29b8769556f4a775f43228b664063bdbd9c7967404`. Contents repeat a seeded random 8 MiB block. The fixture is fully written and fsynced before warming; it is not sparse. It was deleted after the measurements.

All measurements inherit `nice -n 10`, use one hashing thread, and are unpinned. Pair order is portable/SHA-NI, SHA-NI/portable, portable/SHA-NI. The helper's timer includes open/read/hash/finalization and excludes process startup, the size stat and measurement instrumentation. MB/s is decimal. Shared-host activity was not stopped. The driver keeps a read-only shared file mapping open, warms both the file and executable/libraries, verifies every file page resident with `mincore` before each hash, and records residency again afterward. The hashing child still uses the normal ifstream path. Each child's final `wait4` input-block count is 0; live `/proc/PID/io` samples report `read_bytes=0`. Live `rchar` samples are lower bounds, not final counters.

| Pair | Order | Portable MB/s | SHA-NI MB/s | Gain | Portable / SHA-NI seconds |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | portable, SHA-NI | 622.11 | 2234.61 | 3.59x | 6.429752 / 1.790021 |
| 2 | SHA-NI, portable | 621.82 | 2179.69 | 3.51x | 6.432695 / 1.835126 |
| 3 | portable, SHA-NI | 625.50 | 2198.15 | 3.51x | 6.394852 / 1.819715 |

Median portable time: 6.429752 s; SHA-NI: 1.819715 s. Median throughputs: **622.11 -> 2198.15 MB/s (3.53x)**. Measurement interval: 2026-11-01T19:54:31.099267+00:00 through the cleanup observation 2026-11-01T19:54:57.025258+00:00. All 976563 fixture pages were resident before and after each accepted run. The final six-run series needed no retries.

Warmup/instrumentation development excluded seven attempts (each driver exit 1); every fixture was removed. The accepted series uses the full pre-run residency check and zero process disk-input counters. A page evicted after it was consumed does not invalidate a cached read; post-run residency is recorded without requiring retention.

| Excluded attempt | Evidence and correction |
| --- | --- |
| 1 | Zombie `/proc/PID/io` access denied; switched to live samples plus final `wait4` counters |
| 2 | Sequential warmup still led to 4225036288 read bytes; 429.44 MB/s excluded |
| 3 | Writable private mapping residency did not establish shared file-cache residency; 4689563648 read bytes and 505.40 MB/s excluded |
| 4 | Read-only MAP_POPULATE left only 698229/1048577 pages resident; refused before timing, added explicit page touches |
| 5 | File fully resident, but 9674752 process read bytes remained; excluded, then warmed executable/libraries too |
| 6 | Zero read bytes and zero final input blocks, but conservative post-run residency gate refused after 12172 consumed pages were evicted; excluded and repeated the whole series with the sufficient I/O gate |
| 7 | Further shared-host eviction caused 102137856 read bytes despite initial full residency; 419.83 MB/s excluded |

The final fixture uses exactly 4 GB decimal, meeting the measurement size requirement with less cache pressure than the earlier 4 GiB + 65 byte fixture. Cache/I/O eligibility uses no timing threshold and applies identically to both modes; no rejected sample is included. No cache flush, service change or resource-limit change was used.

## Validation

All listed passing gates ran directly and exited 0. Full logs, commands, source/binary/flags hashes and exit files are under `.temp/hybrid-a-r23-261101/`; `source-identity-01.json` binds the tested candidate and executable. The report and README are the only later documentation changes.

| Gate / log | What it establishes |
| --- | --- |
| `build-03.log` | CPU-only build of hash checker, prefill node, server, hybrid checker, codec/append/guards/distributions/client checkers and benchmark checker; `-j2`, nice 10, shared model-cache build lock |
| `hash-original-01.log`, `hash-original-portable-01.log` | Unchanged original suite, 4/4 each, including the read-only 507154688-byte GGUF; automatic and forced-portable digests both `444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c` |
| `hash-ni-02.log` | Final new suite 8/8 in 148.526 s; every test has its own claim comment |
| `hash-sanitized-01.log` | ASan+UBSan, 6/6 in 45.857 s: NIST, dispatch, all odd-read cases, open/read errors, byte changes, invalid chunks; vendor library remains the existing unsanitized build |
| `codec-01.log`, `codec-02.log` | Existing codec checks pass after consumer rebuild |
| `static-01.log`, `isa-isolation-01.log` | Python compilation, ASCII/newline/whitespace, unchanged base artifacts, actual model-naming workflow (166 mappings), and all SHA instructions confined to the attributed function |

NIST answers cover empty, `abc`, the 448-bit and 896-bit messages, and one million `a`. Random coverage uses 14248 distinct lengths, 0..300001: every 64-byte boundary +/-1, all residues at small sizes and near 300000, with non-periodic seeded data. Both paths agree with hashlib. Odd native read sizes are 1, 3, 7, 55, 63, 65, 127, 4093 and 65535 across 0..128 and large/read-boundary inputs.

The r6g sparse layout was recreated at exactly 4294967361 bytes, with markers `a`, `b`, `c` at offsets 0, 2147483655 and 4294967360. Hashlib, automatic SHA-NI and forced portable all give `01faa4689b7b9762b0c965822d43e087f61558f572609c0b384e6e981600231f`. This tests the original size/offset geometry with explicit reproducible marker values. The sparse fixture and dense measurement fixture were deleted.

Dispatch reports `sha-ni` normally and `portable` with the switch; the test independently checks Linux CPU flags. Missing files, Linux directory reads and `/proc/self/mem` EIO fail with exit 1 and no digest; valid neighboring controls pass. Byte changes at the head, block/read boundaries and final tail change both digests and match hashlib. Invalid native read sizes return exit 1 without a digest.

The bounded controls modify isolated source copies, never the candidate. All four were killed; none survived. Every mutant test below actually exits 1 (expected failure); the control driver exits 0 only after verifying the named failures. The swallowed-EIO mutant also directly returns the empty-file digest on `/proc/self/mem`, confirming the planted violation.

| Mutant | Property narrowed | Named failing test | Surviving bound |
| --- | --- | --- | --- |
| Force portable dispatch | Capable-host automatic acceleration must run | `test_dispatch` | Killed |
| Ignore portable switch | Value 1 must select portable | `test_dispatch` | Killed |
| Corrupt first SHA round constant | Hardware compression must match fixed answers | `test_nist_vectors` | Killed |
| Remove read-error check | EIO must reject rather than return a digest | `test_read_error` | Killed |

Initial build attempt exited 2 because the generated tree lacked the hash-checker target; CPU-only reconfiguration and both subsequent builds exited 0. Initial scratch mutant link exited 1 due to an incorrect library option; explicit existing library paths fixed it, and every control compile/link and sanitizer build exited 0. Candidate hash sources produced no compiler warning; the broader build retained the existing benchmark signedness warning and missing-ccache notice.

Self-review used the repository code-review skill's scope, security, approach, general and test checks. No prior reviewer tests were removed or renamed. No surface table or separate landing property suite was supplied: these are brief gaps. No rows of the requested SHA-256 contract are excluded. No board is active (`task-board goal get` exit 1, `.task-board` absent); the existing `.temp/tasks.md` tracks the run. This is ready for independent review, with publication intentionally withheld.

Limits: no macOS SDK/runtime, unsupported-CPU machine or GPU-host timing was available. CommonCrypto remains unqualified on Mac; the r6g note about libc++ read errors appearing as EOF and concurrent file changes remains a pre-existing limit. Whole inference/HTTP suites and full upstream CI were not rerun for this hashing-only delta. `ty` and `editorconfig-checker` are unavailable; those exact checks were not run and nothing was installed. The standalone sanitizer subset does not cover the large sparse/random sweeps. These results do not establish an EPYC or macOS speedup or full startup latency.

## Reproduce

Use the existing CMake documented in README; do not install or download tools. All CMake builds in this run used the shared build lock and `-j2`. Set a disk-backed `TMPDIR` and disable bytecode:

```sh
export TMPDIR=/home/agent/Werkstatt/hybrid-a/.tmp PYTHONDONTWRITEBYTECODE=1
CMAKE=.temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake
nice -n 10 flock /home/agent/Werkstatt/hybrid-models/.build.lock "$CMAKE" -S . -B build -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF -DGGML_CUDA=OFF
nice -n 10 flock /home/agent/Werkstatt/hybrid-models/.build.lock "$CMAKE" --build build -j2 --target llama-hybrid-check-hash llama-hybrid-prefill llama-server
nice -n 10 python3 tools/hybrid/test_file_hash.py --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf -v
HYBRID_HASH_PORTABLE=1 nice -n 10 python3 tools/hybrid/test_file_hash.py --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf -v
nice -n 10 python3 tools/hybrid/test_file_hash_ni.py -v
nice -n 10 python3 .temp/hybrid-a-r23-261101/measure.py
```

The measurement script creates its own dense 4 GB decimal disk fixture, warms it, checks residency, alternates three pairs, verifies each digest against hashlib, records timestamps/PIDs/live I/O/final input blocks in `measurements.json`, and deletes the file in `finally`. It is a local evidence artifact; the README checker commands also work against any caller-supplied cached file.

## Draft updated PR #7 description

Hashing a large GGUF at each scorer/node/server startup leaves the GPU idle. This adds runtime-selected SHA-NI compression on x86-64 GCC/Clang while retaining full-file SHA-256 identity, the portable fallback and the existing macOS CommonCrypto default.

Since r6g: add CPUID checks for SHA, SSSE3 and SSE4.1, a function-level ISA target, `HYBRID_HASH_PORTABLE=1`, checker backend reporting and odd native reads. The original tests and vendored optimization remain unchanged. New tests cover fixed NIST answers, 14248 random lengths through 300001 bytes, odd chunks, >4 GiB sparse input, dispatch, read errors and byte changes against hashlib; bounded fault controls and sanitizer checks pass.

On agents1 (Ryzen 7 9700X, GCC 15.2.0, Release -O3 -DNDEBUG), three paired niced reads of a fully cached 4 GB dense file improve median full-file throughput from 622.11 to 2198.15 MB/s (3.53x, 71.7% less hash time). All six digests agree and disk-input counters stay zero.

macOS build/digest/timing qualification is still pending. Linux measurements are from agents1, not the EPYC GPU hosts. AI assistance: Codex implemented and tested the SHA-NI extension under the contributor's specified design. This text is a local draft; the PR was not changed or pushed.

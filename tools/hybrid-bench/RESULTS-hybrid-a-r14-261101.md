# Fast native model identity hash - hybrid-a-r14-261101

## Result and source

The whole-file identity is unchanged. The portable implementation now feeds complete 64-byte blocks directly to SHA-256; macOS file hashing uses system CommonCrypto. The launcher still hashes independently and native startup still reads the entire file. No digest cache, supplied hash, size/mtime record, or skip path was introduced.

Branch: `m2-fast-hash`. Exact base: `7e1a477b2e8dfc8146744f4f81d73abca25e7d4a`. Tested implementation commit: `c26311e6a6e5ae5878080c47420fc45ad42daf92`. The report commit follows it; its parent identifies the tested implementation. All new commits are local, signed with the configured host identity and contain the user-requested Codex co-author trailer. No push, PR, upstream submission or session link.

The prescribed `git fetch /home/agent/Werkstatt/hybrid-prototype.git main` succeeded but returned `0124e89a933872b54bd445921b9ac2e1c9c43a09`. The mirror also contained the requested base as `m1-test-host-aware`; fetching that exact object and branching from it preserved the explicit pin. The starting checkout was clean. No active task-board was available (`goal get` exit 1, `.task-board` absent); the existing `.temp/tasks.md` tracks this run.

## What was slow

`sha256_update` copied each byte into its 64-byte buffer, incremented the 64-bit byte count and checked the block boundary for every byte. `sha256_write_byte_block` already compressed complete blocks; the wasted work was byte-at-a-time staging before compression. Compression itself is portable scalar C, without the optimized digest provider used by this host's Python hashlib.

The update now counts an input span once, completes a buffered partial block with a bounded copy, compresses complete blocks directly from the input, and copies only the final tail. Byte loads preserve unaligned access and endian behavior. Empty updates accept a null data pointer as before. Padding, 64-bit length accounting, digest formatting and the public vendor API remain unchanged. The existing vendor-sync patch mechanism preserves this local change; no vendor download was run.

On Apple builds, only `hybrid::file_hash` switches to `CC_SHA256_Init/Update/Final`. The fixed 65536-byte read bounds every `CC_LONG` update; 32-bit update lengths do not truncate the whole-file length. Other platforms use the improved vendor implementation. Open/read failures retain the existing checks. Apple documents the [public API and libSystem linkage](https://github.com/apple-oss-distributions/CommonCrypto/blob/main/include/CommonDigest.h); its [implementation routes SHA-256 through corecrypto](https://github.com/apple-oss-distributions/CommonCrypto/blob/main/lib/CommonDigest.c). No additional link dependency or package was added.

## Measurements

Measured on 2026-11-01 UTC, Linux x86_64 KVM, 8 visible AMD Ryzen 7 9700X CPUs, GCC 15.2.0, CMake 4.4.4, existing Release build (`-O3 -DNDEBUG` for vendor C). CPU-only; benchmark and test processes inherit `nice -n 10`. Model: local `Qwen3.5-0.8B-Q4_0.gguf`, exactly 507154688 bytes (507.154688 decimal MB).

All reference/native/model runs returned:

```text
444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c
```

The helper times the actual `hybrid::file_hash` call, including open/read/finalization, excluding process startup and the initial size stat. The read size stays 64 KiB. These are repeated warm-cache reads; caches were not flushed and other host workloads were not stopped. No disk-throughput or full-startup claim follows from them.

The original native implementation was measured before any SHA implementation edit. Each initial series used one excluded warmup and three timed reads; all commands exited 0:

| Implementation | MB/s samples | Median MB/s | Median seconds |
| --- | --- | ---: | ---: |
| Original vendor | 500.62, 487.05, 553.81 | 500.62 | 1.013051 |
| Whole-block vendor | 612.12, 611.92, 601.97 | 611.92 | 0.828789 |

Initial median gain: 1.22x throughput, 18.2% less hash time. Baseline variation warranted a final interleaved original/candidate comparison after functional tests had ended. The saved baseline executable contains the original static SHA implementation; both native tools use the same file-read path. Three alternating pairs, followed by three Python hashlib runs with the same 64 KiB reads, all exited 0:

| Implementation | MB/s samples | Median MB/s | Median seconds |
| --- | --- | ---: | ---: |
| Original vendor | 550.73, 552.63, 562.85 | 552.63 | 0.917717 |
| Whole-block vendor | 626.06, 624.28, 634.86 | 626.06 | 0.810076 |
| Python hashlib (OpenSSL) | 1968.22, 1930.00, 1992.11 | 1968.22 | 0.257671 |

Use the tighter paired result for the conservative host conclusion: **552.63 -> 626.06 MB/s, 1.13x throughput, 11.7% less hashing time**. Python's `_hashlib.HASH` backed by OpenSSL 3.5.5 measured 1968.22 MB/s median. This shows the remaining scalar compression cost; removing byte staging alone gives a modest gain here.

Expected macOS gain: the system digest path should provide a larger reduction in hash CPU time than removing staging alone; a several-fold improvement is plausible, bounded by file-read throughput. This is an engineering expectation, with no measured macOS MB/s, numerical speedup guarantee or startup-time prediction. There is no Apple SDK/runtime on this host, so CommonCrypto compilation, linkage, digest equality and timing remain unverified on macOS. The owner's 362 s healthy-start observation includes work beyond hashing and is not reproduced here. No GPU measurements were made.

Private measurement evidence: `.temp/hybrid-a-r14-261101/baseline-01.log`, `blocks-01.log`, `paired-01.log`. The first two record one warmup plus three samples; the paired log records implementation, actual PID, exit code, duration, digest and bytes. `tested-source.json` binds native source/test/configuration hashes to the checked candidate.

## Correctness and negative controls

Each new test method has a claim comment in `tools/hybrid/test_file_hash.py`. `llama-hybrid-check-hash FILE` calls the real startup function; `FILE UPDATE_BYTES` tests vendor streaming separately, including unaligned input and interspersed null/zero-length updates. No timing threshold is an assertion.

| Named test | Claim and finite coverage |
| --- | --- |
| `test_file_boundaries` | Native startup hash equals both sha256sum and hashlib for 0, 1, 55, 56, 63, 64, 65, 119, 120, 127, 128, 129, 65535, 65536, 65537 and 131073 bytes. These cover SHA blocks, padding and file-read boundaries. |
| `test_fragmented_updates` | Both references agree with vendor streaming at sizes 0, 1, 55, 56, 63, 64, 65, 127, 128, 129 and 4097; update sizes 1, 3, 55, 56, 63, 64, 65, 127, 128, 129 and 65536 cover partial heads, direct blocks and tails. |
| `test_one_byte_mutations` | A 65537-byte valid fixture first matches both references. Independent single-bit mutations at bytes 0, 63, 64, 65535 and 65536 each change the reference digest; both native startup and 65-byte vendor updates must equal the changed digest and differ from the original. |
| `test_model` | The read-only 507154688-byte GGUF equals sha256sum and hashlib via startup hashing and vendor streaming with 65537-byte updates. |

All four test methods passed on the original implementation and on the candidate (exit 0). The final candidate run took 2.55 s. The real model was included; no skipped model test. Optional launcher record reuse was omitted, so changed-record/wrong-size/missing-record refusal tests do not apply. Every native call still hashes bytes.

The vendor patch was applied offline to a task-local fixture containing the original vendored source and reproduced the tested source byte for byte (`git apply` exit 0). An incompatible-source control changed the old staging statement: application then failed at the intended SHA update hunk (`git apply` exit 1, expected failure), and the fixture remained byte-identical. Both assertions passed in `vendor-patch-02.log`. No code mutation campaign or universal equivalence proof is claimed.

## Existing suites and build evidence

Every row below ran as a direct process; logs contain complete output and `checks.json` records exact argv, PID, elapsed time and real exit code. No command output was piped through tee. These are local results, not hosted CI or platform review statuses.

| Gate | Exit | Evidence/claim |
| --- | ---: | --- |
| `hash-final-01.log` | 0 | 4 new methods; fixtures, fragmented updates, mutation controls and GGUF |
| `bench-unit-01.log` | 0 | 25 existing benchmark tests |
| `proxy-unit-01.log` | 0 | 68 existing proxy/journal/concurrency tests; existing ResourceWarning messages retained |
| `codec-01.log` | 0 | Existing scheduler, SHA/layout and delta acceptance/rejection tests |
| `codec-header-01.log` | 0 | Default codec/header budgets |
| `codec-64k-01.log` | 0 | Codec/header budgets at max_ctx 65536 |
| `codec-256k-01.log` | 0 | Codec/header budgets at max_ctx 262144 |
| `model-512-01.log` | 0 | 64/64 greedy tokens equal independent prefill; malformed append preserves prefix state |
| `http-default-01.log` | 0 | Existing authenticated node/session HTTP suite |
| `limits-01.log` | 0 | LRU replay, byte-preserving budget refusals, snapshot/extend pinning, request caps, invalid CLI and public access |
| `identity-01.log` | 0 | Two existing local quantizations have distinct file SHA-256 and matching layout identity |
| `http-slow-readers-01.log` | 0 | Existing bounded response/slow-reader regression |
| `http-max-delta-01.log` | 0 | 16384-token delta checksum equals hashlib; native snapshot matches; one over refuses without mutation (195.55 s) |
| `http-pending-01.log` | 0 | Existing pending admission regression |
| `http-errors-01.log` | 0 | Existing HTTP error-state regression |

Configure, baseline helper build, block helper build, and final native builds all exited 0. Every configure/build used `nice -n 10`, the shared `~/Werkstatt/hybrid-models/.build.lock`, at most `-j2`, and a >=26843545600-byte free-space guard inside the lock. Minimum available bytes recorded at admission: 38663761920. The final target list included `llama-server`, `llama-hybrid-prefill`, hash/codec/append/guards/distributions/client checkers and the benchmark checker. No downloads, package installs or GPU execution.

Python compilation of the new test and modified sync script exited 0. ASCII/final-newline/trailing-space checks passed for the new C++/Python files. Initial staged whitespace check exited 2 because blank context lines inside the generated patch contained spaces. Removing those context spaces preserved patch application; the repeated positive/refusal patch controls and staged whitespace check exited 0. No native source changed during that formatting correction.

Self-review applied the scope, security, general and tests checklists. No prior reviewer tests were removed or renamed. No supplied surface table or external landing property suite was available; that is a brief gap. No rows are excluded from the requested whole-file SHA-256 contract. The explicitly optional launcher reuse path was omitted. Full upstream model/backend suites, long-context inference beyond the existing bounded 16k HTTP regression, macOS, and GPU checks were not run. `ty` and `editorconfig-checker` are absent; their workflows were not run, and no download was authorized. No independent or hosted review/landing is claimed for this local-only prototype. After tests, all recorded test/measurement PIDs were absent; task-created credential files were removed. Native state fixtures were held in memory by these suites, with no retained state dump over 1 MiB in the task directory. See `cleanup-01.log` for the observation time and counts.

## Reproduce

Use the installed local CMake or set `CMAKE` to an existing tool. Keep the shared node-guide lock for every build. This run reused the existing build tree:

```sh
CMAKE=.temp/hybrid-a-r1-261030/cmake-tool/cmake/data/bin/cmake
LOCK="$HOME/Werkstatt/hybrid-models/.build.lock"
MODEL="$HOME/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf"
# Run the free-byte guard inside the lock for every configure/build invocation.
nice -n 10 flock "$LOCK" bash -c 'available=$(df -B1 --output=avail . | tail -n1); echo "available_bytes=$available"; test "$available" -ge 26843545600 || exit 75; exec "$@"' guard "$CMAKE" -S . -B build -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF
nice -n 10 flock "$LOCK" bash -c 'available=$(df -B1 --output=avail . | tail -n1); echo "available_bytes=$available"; test "$available" -ge 26843545600 || exit 75; exec "$@"' guard "$CMAKE" --build build -j2 --target llama-hybrid-check-hash llama-hybrid-prefill llama-hybrid-check llama-hybrid-check-codec llama-hybrid-check-append llama-hybrid-check-guards llama-hybrid-check-distributions llama-hybrid-check-client llama-hybrid-bench-check llama-server
nice -n 10 build/bin/llama-hybrid-check-hash "$MODEL"
nice -n 10 python3 tools/hybrid/test_file_hash.py --model "$MODEL" -v
```

For before/after comparison, build the same helper against the pinned base and keep its executable before rebuilding the vendor implementation. The saved baseline binary in the private evidence directory was made this way. Use the no-chunk invocation for native startup timing; the chunk argument deliberately selects the portable vendor path on every platform.

Exact functional commands from `checks.json` (all exit 0; substitute a fresh `.temp/` artifacts path for reruns):

```sh
nice -n 10 python3 tools/hybrid/test_file_hash.py --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf -v
nice -n 10 python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v
nice -n 10 python3 -m unittest discover -s tools/hybrid-proxy -p 'test*.py' -v
nice -n 10 build/bin/llama-hybrid-check --codec
nice -n 10 build/bin/llama-hybrid-check-codec
nice -n 10 build/bin/llama-hybrid-check-codec --max-ctx 65536
nice -n 10 build/bin/llama-hybrid-check-codec --max-ctx 262144
nice -n 10 build/bin/llama-hybrid-check /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf - 512
nice -n 10 python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r14-261101/http-default --port 18094
nice -n 10 python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r14-261101/limits
nice -n 10 python3 tools/hybrid/test_limits.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --requantized /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --artifacts .temp/hybrid-a-r14-261101/identity --case identity
nice -n 10 python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r14-261101/http-slow-readers --port 18094 --regression slow-readers
nice -n 10 python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r14-261101/http-max-delta --port 18094 --regression max-delta
nice -n 10 python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r14-261101/http-pending --port 18094 --regression pending
nice -n 10 python3 tools/hybrid/test_prefill.py --binary build/bin/llama-hybrid-prefill --model /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --artifacts .temp/hybrid-a-r14-261101/http-errors --port 18094 --regression errors
```

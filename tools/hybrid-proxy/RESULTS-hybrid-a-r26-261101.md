# Proxy node-restart regression

Base: `b5159303fadfb8da96d0dced7b403d8d5ca1314c`, fetched from `https://github.com/relux-works/hybrid-inference` main. Branch: `m3-proxy-restart-test`. Production code is unchanged.

Source requirement: r7j review N1 in `/home/agent/Werkstatt/hybrid-review-proto/llama.cpp/.temp/REVIEW-short-call-claude.md` and its `review_tests/test_review_shortcall.py::test_burst_node_restart_then_offload`. Evidence is under `.temp/hybrid-a-r26-261101/`; `validation-identity.json` records source hashes, environment and command exits.

## Named regression

`test_benchmark.BenchmarkProxyTests.test_burst_node_restart_then_offload` uses the existing synthetic HTTP fixture and standard-library unittest/mocks. It runs two local-kept turns, replays to establish a remote session, then commits another local-kept turn without flushing its suffix. Deleting the fixture's node session rows simulates RAM loss while preserving the proxy journal and decoder slot.

The next forced offload must GET the old session and observe 404, create and GET a replacement, replay the entire durable prefix at n0=0, and critically extend that replacement before importing. Assertions check both extend bodies and hashes, one successful import under the replacement ID, exact durable tokens, and final node tokens/hash/acknowledged length after replay. A successful local fallback cannot satisfy the test. All recorded extends use the replacement path; no stale session ID may reach the decoder import.

## R2 control

The reviewer's unchanged `scripts/mutate.py` was copied to the evidence directory and run against an isolated copy of `tools/hybrid-proxy`. It requires exactly one source match; plant exit was 0. `r2.diff` confirms this sole production-code delta:

```python
# Original
path = self.sync(session, client)
# R2: skip the initial prefix sync when a remote ID already exists.
path = ("/v1/sessions/" + urllib.parse.quote(session["remote_id"], safe="")) if session["remote_id"] else self.sync(session, client)
```

| Mutant | Contract narrowed by the control | Named failing test | Result / surviving bound |
|---|---|---|---|
| R2 offload without initial prefix GET | Recover a lost node session and actually import from its replacement; local fallback is insufficient | `test_benchmark.BenchmarkProxyTests.test_burst_node_restart_then_offload` | Killed: exit 1, `AssertionError: 0 != 1` at the import-count assertion. No survivor in this one-plant control. |

In this fixture R2's later behind-prefix sync can recreate the session, but the critical request still uses the old path and falls back without importing. The failure is the missing required offload, not a harness/setup error. No mutant was applied to the working production source.

## Commands and outcomes

Run from the repository root with the on-disk temporary directory and bytecode writes disabled:

```sh
mkdir -p .temp/hybrid-a-r26-261101/tmp
export TMPDIR="$PWD/.temp/hybrid-a-r26-261101/tmp"
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s tools/hybrid-proxy -p test_benchmark.py -k test_burst_node_restart_then_offload -v
python3 .temp/hybrid-a-r26-261101/mutate.py .temp/hybrid-a-r26-261101/r2-proxy/proxy.py R2_offload_without_prefix_get
python3 -m unittest discover -s .temp/hybrid-a-r26-261101/r2-proxy -p test_benchmark.py -k test_burst_node_restart_then_offload -v
python3 -m unittest discover -s tools/hybrid-proxy -p 'test_*.py' -v
python3 -m unittest discover -s tools/hybrid-bench -p 'test_*.py' -v
```

The isolated mutant directory is a copy of the candidate proxy directory, including the new test; the mutator comes from r7j's scripts directory above. Each test command ran as its own process with stdout/stderr redirected directly to its log.

| Check | Exit | Evidence |
|---|---|---|
| Focused test, unchanged production code | 0, 1 test passed | `restart-positive-02.log` |
| Exact R2 focused control | 1, expected assertion failure | `r2-mutant-01.log`, `r2.diff` |
| Proxy suite | 0, 73 tests passed, no skips | `proxy-01.log` |
| Bench suite | 0, 68 tests, 6 native skips (62 passed) | `bench-01.log` |
| Diff whitespace check | 0 | `diff-check-01.log` |
| Source hashes, retained test names, scope, syntax, ASCII and disk audit | 0 | `audit-01.log` |

Initial focused run `restart-positive-01.log` exited 1 because the assertion expected a GET body of None while the HTTP fixture represents it as empty bytes. The corrected assertion checks the required method, path and 404 without constraining that irrelevant representation. The subsequent positive and R2 runs use the same test source.

## Gap harness decision

Do not retain `bench_gap.py` as a second tool. Compared with the existing `bench_local.py`, r7j's file only imports `os` and adds `time.sleep(float(os.environ.get("GAP", "0")))` before the follow-up. Copying the full harness would duplicate instrumentation, and its unchanged immediate-follow-up comment is inaccurate with a nonzero gap. The gap idea is useful for a future measurement, but should be an explicit validated option in the existing harness if requested. No gap measurement or latency improvement is claimed here.

## Scope and limits

No surface table was supplied. The bounded coverage row is simulated RAM-session loss between a deferred local turn and an offload, mapped to the named regression and killed R2 control above. Existing reviewer/producer tests are retained. This adds no Python package or native dependency; the reused synthetic fixture already accesses the host's libxxhash through standard-library ctypes.

Native process restart, inference equivalence, GPU, real-network/5G timing, and other mutant classes are outside this test's claim. The six bench skips require explicitly supplied native binaries and headers. No native build, model download, GPU run, hosted checks, push or PR was performed. The test addition is validated by Python import/execution and the proxy suite rather than a native build. Self-review covers scope, security, general conventions and tests; the only tracked changes are this report, the test and its README entry.

Python: 3.14.4 on Linux x86_64. Free-space samples: initially 36,098,973,696 bytes, later 35,784,372,224 bytes, and 36,093,448,192 bytes at the source audit, all above the 28,000,000,000-byte floor. Task logs and isolated mutant artifacts remain under `.temp/`.

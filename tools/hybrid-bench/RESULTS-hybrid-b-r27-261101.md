# R27: natural-text hetero-quant driver

Local prototype, ready for review on `m3-reference-logits`, appending to r26 `5dea95a8c9ccf1f3239aabc00f707989e90ee1aa`. No push. The requested Codex co-author trailer overrides the upstream contribution convention for this private prototype. CPU evidence is under `.temp/hybrid-b-r27-261101/`. No books or model weights were downloaded, and no GPU process was run on agents1.

## GPU host commands

Run from the combined prototype checkout containing this change and the required Bonsai/Mirai support. Use already verified local model files; the paths below are operator inputs, not download commands. Stock and Mirai scorer builds must both contain the schedule fixes described below. Mirai needs its effective 64-layer trunk, MTP disabled and the separately required greedy-token comparison against the alesha-pro fork; this driver does not establish those prerequisites.

### 1. Private disk, builds and model paths

```sh
set -eu
umask 077
export PYTHONDONTWRITEBYTECODE=1
STUDY_HOME=/gpu-disk/hetero-r27
mkdir -p "$STUDY_HOME/tmp" "$HOME/.cache/hybrid-study"
export TMPDIR="$STUDY_HOME/tmp"
DRIVER=tools/hybrid-bench/hetero_study.py
BUILD=build-cuda
MIRAI_BUILD=build-mirai-cuda
D_MODEL=/models/Qwen3.8-27B-BF16.gguf
Q8_MODEL=/models/Qwen3.8-27B-Q8_0.gguf
Q4_MODEL=/models/Qwen3.8-27B-Q4_K_M.gguf
BONSAI=/models/Bonsai-2-27B-Q2_0.gguf
MIRAI=/models/Qwen3.8-27B-S-mirai.gguf
STUDY="$STUDY_HOME/qwen27b"
D_SHA=$(sha256sum "$D_MODEL" | cut -d ' ' -f 1)
# Use the host's admitted build slot and existing CUDA configuration; no UI downloads.
nice -n 10 cmake --build "$BUILD" -j4 --target llama-server llama-hybrid-score llama-hybrid-prefill
nice -n 10 cmake --build "$MIRAI_BUILD" -j4 --target llama-hybrid-score llama-hybrid-prefill
```

Record `git rev-parse HEAD`, model SHA-256/size, `nvidia-smi`, driver/toolchain versions and both build identities alongside the private study. Only one visible GPU is assumed. Imported scoring holds two contexts on the client model; schedule probes run sequentially after those contexts are freed. Leave room for KV, recurrent state, logits buffers and compute beyond weight size. Do not run any of these stages concurrently with another GPU job.

### 2. Fetch on the GPU host, pin once

```sh
tools/hybrid-bench/fetch --manifest tools/hybrid-bench/gutenberg_sources.json --output "$STUDY_HOME/books"
```

The checked-in file is a URL list of 20 Gutenberg plain-text books, not a corpus pin. Fetch uses curl with HTTPS-only redirects, a 64 MiB/file cap and a five-minute transfer timeout. It writes private books and `books/manifest.json` with `id`, `url`, `sha256` and exact byte `size`. Preserve that pinned manifest for reproducibility. Later fetches compare both local files and newly downloaded bytes with the original pins, refusing any mismatch without replacing the pinned books. Initial pinning is trust on first use. A pinned input manifest can also be supplied to fetch.

Prepare accepts only the pinned form, for example:

```json
{"documents":[{"id":"pg1342","url":"https://www.gutenberg.org/cache/epub/1342/pg1342.txt","sha256":"<64 lowercase hex digits>","size":123456}]}
```

The example hash/size are schema placeholders. Use the actual fetch output. The CPU fixtures are synthetic and never masquerade as downloaded Gutenberg content.

### 3. Tokenize locally and prepare D=20, N=32768, M=2048

```sh
nice -n 10 "$BUILD/bin/llama-server" -m "$D_MODEL" -ngl 999 -c 35072 -np 1 \
  --host 127.0.0.1 --port 8099 --no-webui > "$STUDY_HOME/tokenizer.log" 2>&1 &
TOKENIZER_PID=$!
printf '%s\n' "$TOKENIZER_PID" > "$STUDY_HOME/tokenizer.pid"
trap 'kill "$TOKENIZER_PID" 2>/dev/null || :; wait "$TOKENIZER_PID" 2>/dev/null || :' EXIT INT TERM
# Wait for this owned server; never continue if it dies or readiness times out.
TOKENIZER_READY=0
for attempt in $(seq 1 300); do
  kill -0 "$TOKENIZER_PID"
  if curl --silent --fail http://127.0.0.1:8099/health > /dev/null; then TOKENIZER_READY=1; break; fi
  sleep 1
done
test "$TOKENIZER_READY" = 1
python3 "$DRIVER" prepare --manifest "$STUDY_HOME/books/manifest.json" --documents "$STUDY_HOME/books" \
  --model "$D_MODEL" --tokenizer-url http://127.0.0.1:8099 --output "$STUDY" --d 20 --n 32768 --m 2048
kill "$TOKENIZER_PID"
wait "$TOKENIZER_PID" || :
trap - EXIT INT TERM
unset TOKENIZER_PID
```

The trap and kill refer only to the child just started in this shell. Stop and reap it before scoring. Prepare never retrieves a document: its only HTTP calls are `/props` and `/tokenize` on numeric loopback, with redirects and proxies disabled. It verifies every selected source before tokenization, checks `/props` model path against the given local GGUF, adds special tokens, disables special-string parsing, and takes the first N+M IDs. No chat template, stripping or generation is applied. A short document refuses instead of padding, repeating or substituting another book. Index/fixtures are 0600 under a 0700 study directory. Keep local model files immutable while their tokenizer server is running.

### 4. Produce D from local prefill, then the local arms

```sh
nice -n 10 python3 "$DRIVER" reference --study "$STUDY" --model "$D_MODEL" --build-dir "$BUILD" --gpu-layers 999 --score-batch 32
nice -n 10 python3 "$DRIVER" arm --study "$STUDY" --kind local --label D-local --client-model "$D_MODEL" \
  --reference-model-sha256 "$D_SHA" --build-dir "$BUILD" --gpu-layers 999 --score-batch 32
nice -n 10 python3 "$DRIVER" arm --study "$STUDY" --kind local --label C-Q4-local --client-model "$Q4_MODEL" \
  --reference-model-sha256 "$D_SHA" --build-dir "$BUILD" --gpu-layers 999 --score-batch 32
nice -n 10 python3 "$DRIVER" arm --study "$STUDY" --kind local --label A-Bonsai-local --client-model "$BONSAI" \
  --reference-model-sha256 "$D_SHA" --build-dir "$BUILD" --gpu-layers 999 --score-batch 32
nice -n 10 python3 "$DRIVER" arm --study "$STUDY" --kind local --label A-Mirai-local --client-model "$MIRAI" \
  --reference-model-sha256 "$D_SHA" --build-dir "$MIRAI_BUILD" --gpu-layers 999 --score-batch 32
```

D means BF16 weights with q8_0 KV, native f32 recurrent state and flash attention. Every reference retains its measured float16 encoding error. D-local checks the stored-reference floor on the actual documents/backend. It does not measure all numerical variability of a different client.

### 5. Imported arms, one process at a time

```sh
for client in Bonsai Mirai; do
  if test "$client" = Bonsai; then CLIENT_MODEL="$BONSAI"; CLIENT_BUILD="$BUILD"; else CLIENT_MODEL="$MIRAI"; CLIENT_BUILD="$MIRAI_BUILD"; fi
  for node in BF16 Q8 Q4; do
    case "$node" in BF16) NODE_MODEL="$D_MODEL";; Q8) NODE_MODEL="$Q8_MODEL";; Q4) NODE_MODEL="$Q4_MODEL";; esac
    nice -n 10 python3 "$DRIVER" arm --study "$STUDY" --kind imported --label "B-$client-$node-imported" \
      --client-model "$CLIENT_MODEL" --node-model "$NODE_MODEL" --hetero-quant \
      --reference-model-sha256 "$D_SHA" --build-dir "$CLIENT_BUILD" --node-build-dir "$BUILD" \
      --gpu-layers 999 --score-batch 32 --chunk-tokens 4096 --ram-budget-mib 4096
  done
done
```

Each document starts its own authenticated loopback node, exports a delta chain through N-1, releases the session with HTTP 204, stops/reaps that owned PID, then runs the client scorer. The scorer evaluates the final context token and teacher-forces all M real continuation tokens. Original node hashes remain in the delta envelopes. Cross-model imports require `--hetero-quant` and matching native vocabulary/layout identity; there is no silent local fallback. The driver does not implement resync B2. A different node/client build is allowed only when their actual native state/layout contracts agree.

For JSON arm input, save `{"kind":"imported","label":"B-Bonsai-BF16-imported","client_model":"/models/Bonsai-2-27B-Q2_0.gguf","node_model":"/models/Qwen3.8-27B-BF16.gguf"}` in a private file and replace the four arm arguments with `--spec /private/arm.json`. Keep the explicit hetero flag, reference pin and build/settings arguments.

### 6. Numeric reports and paired differences

```sh
python3 "$DRIVER" report --study "$STUDY" --arms D-local C-Q4-local A-Bonsai-local B-Bonsai-BF16-imported B-Bonsai-Q8-imported B-Bonsai-Q4-imported \
  A-Mirai-local B-Mirai-BF16-imported B-Mirai-Q8-imported B-Mirai-Q4-imported --seed 42 --output "$STUDY_HOME/report27b-seed42"
```

CSV/Markdown contain numeric tables only, plus identifier/metric columns. JSON also records seed, 10,000 resamples and index identity. `positions` gives mean/median/p90/p99/max KL(D || arm), in nats, and top-1 agreement over all positions. `document` lists document means. `documents` gives the mean of document means and bootstrap percentile 95% intervals over whole documents. `paired` gives later-listed arm minus earlier-listed arm on those same documents, including bootstrap intervals. Individual document means have no within-document CI. `schedule_probes` gives separate arm-side schedule KL/top-1 observations at three positions per document. It is not an acceptance threshold or a bound on unprobed positions. `score_batch` is visible in every report row.

For the Bonsai 2B/4B/9B ladder, repeat steps 3-6 with a new `STUDY`, the corresponding Qwen3.5 BF16 D, client, Q8/Q4 node files and D hash. Reuse the frozen books, but tokenize afresh for each rung and keep separate indexes/references. Repeat step 6 with other explicit seeds for bootstrap reproducibility. The driver refuses accidental cross-family references; static compatibility and actual GPU runs remain prerequisites.

### 7. Resume and process custody

Rerun the identical reference or arm command. Completed documents are skipped only when fixture/index, GGUF, native binary/shared-library, helper, driver, settings and output hashes agree. Missing deltas, changed fixtures, reference model/document mismatches and damaged outputs refuse. Keep the input books, pinned manifest, native builds and model paths available. A separate configuration or seed does not relabel old evidence. Use a new study/arm label when intentionally changing the experiment.

The host-wide advisory lock defaults to `$HOME/.cache/hybrid-study/gpu.lock`; overrides go before the subcommand and must be shared by all cooperating studies. Child processes inherit its descriptor, so an orphan keeps the reservation after an uncatchable parent exit. Interrupt/TERM cleanup signals only live Popen children created by this invocation. No old PID file authorizes a kill, and no process name is used for cleanup. A SIGKILL can leave an attempt without a receipt; it remains diagnostic scratch, and a later invocation creates a new attempt after the inherited lock becomes available. Never stop an unrelated lock holder.

The lock does not reserve the device against unrelated programs or the externally launched tokenizer. The orchestrator must serialize those too. Every native command has a private log and `*.process.json` containing PID, argv and actual exit. No new report files are opened until all selected documents, references, arm rows and delta hashes validate. A prior valid report is preserved on refusal. Artifact receipts are transactional completion markers; this is not a filesystem-wide transaction across disk failures.

## Disk and GPU memory budgets

Reference rows cost exactly `2*V+44` bytes per continuation position, excluding their shared JSON header. At V=248320, M=2048, D=20: **496684 bytes/row**, **1017208832 bytes/document**, **20344176640 bytes = 18.947 GiB** for the D reference payload. D is shared by every arm within one study. The driver saves scalar per-position arm rows plus native JSON measurements, not a duplicate full-vocabulary reference for every arm.

| Item | Expected disk use at the defaults |
|---|---|
| Shared D references | 18.947 GiB plus headers |
| Local arm | O(D*M) metric JSON, logs and receipts; allow 100 MiB/arm |
| Imported arm, retained deltas | `D*((N-1)*(KV_bytes_per_token+24) + ceil((N-1)/chunk_tokens)*recurrent_state_bytes)` plus envelope headers and fixed serialization |
| Imported 27B example, KV=34816 B/token, state=157000000 B, chunk=4096 | 47,952,045,600 bytes = 44.659 GiB, plus headers/fixed serialization and metric/log allowance |
| Extra full-distribution file if separately requested with the native producer | Another 18.947 GiB/arm at this V/M/D |

The delta estimate is geometry-based, not a measured 27B file size. At eight deltas/document, the existing 1 MiB fixed-serialization reserve adds at most about 160 MiB across 20 documents, plus envelope headers. Every delta body must remain below the native 512 MiB cap; reduce chunk size for a larger actual geometry. Smaller chunks repeat recurrent state more often and increase disk use. The default six imported 27B arms retain roughly **268 GiB** of deltas, in addition to shared D, models and attempts. Reuse is explicit and checked; this implementation does not deduplicate node chains across arm labels or delete their evidence after scoring. Interrupted attempts require additional space. Measure actual files and keep free space accordingly.

The scorer retains at most two primary contexts on imported runs and one on local/reference runs. Full-vocabulary logits are bounded by score_batch (32 by default, native maximum 256); at V=248320, 32 float32 rows are about 30.3 MiB per context. Probe snapshots retain at most three full-precision distributions per arm (about 11.4 MiB total for two arms), then primary contexts are released and probe contexts run one at a time. Probes add up to three local prefills and three imported replays per document; original study runtime estimates do not include that cost.

## R6q findings incorporated

The independent r26 review is `/home/agent/Werkstatt/hybrid-review-proto/llama.cpp/.temp/REVIEW-reference-logits-claude.md`, bound to 5dea95a8, verdict changes_requested. F1 (`client-schedule-floor-unmeasured`) showed Q4 continuation batch-shape variation; F2 (`row0-arm-schedule-asymmetry`) showed the local full-prefill and imported scalar last token were different numerical schedules. R27 binds score_batch and `split-last-v1` in references, uses N-1 plus one locally, and records `fresh-prefix-spots-v1` full-precision arm-side probes at 0/M/2/M-1. Old references must be regenerated. The additional probes expose selected schedule variation rather than attributing it to float16 or claiming a universal floor. No independent acceptance of these fixes is claimed.

## CPU evidence

Validation commands run standalone with real exits, `PYTHONDONTWRITEBYTECODE=1`, disk TMPDIR and nice 10. Build holds the existing shared build lock, uses -j2, and installs nothing. Only 0.8B local Q4_0 (`444406ddd926550c724ec18d5120a9d40ded44908a063b0e66e9a7e5464c652c`) and Q8_0-requant (`f543fad263339779c3a38a21a531c6adbffcfdecdf5f749dd542de924b1e7302`) are used. Q8 requant is a CPU protocol reference, not a BF16 paper measurement.

Validation results and finite gaps are recorded below after the final audit.

| Standalone command / evidence under the run directory | Actual exit | What it verifies |
|---|---:|---|
| Existing CMake `--build build -j2 --target llama-hybrid-score`, nice 10 under `/home/agent/Werkstatt/hybrid-models/.build.lock`; `build-01.log`, `build-01.process.json`, `build-record-01.json` | 0 | Native schedule fixes compile against the existing CPU build. Before/after executable and shared-library hashes are retained. |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_hetero_study.py -v`; `unit-04.log` | 0 | Seven tests: pins/short/private prepare, changed fixture/no report, independent seeded bootstrap, shared lock, inherited lock and child cleanup, curl pin/mismatch controls without downloads, tokenizer identity/loopback/redirect refusal. |
| `python3 tools/hybrid-bench/test_hetero_study_native.py --client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --reference /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --artifacts .temp/hybrid-b-r27-261101/native-03`; `native-03.log` | 0 | Real tokenizer, N64/M16/D3, D || D encoding floor, local/imported/homogeneous Q4, JSON spec, unchanged-PID-record resume, independent recomputation of position/document/paired statistics and CIs, byte-identical seeded reports, all requested refusals. |
| Driver subcommands in `native-03/commands.json` | 11 commands at 0; 10 at **2** | Explicit refusals for manifest mismatch, short document, another reference document/model, missing delta and changed fixture, including resume/report paths. Every report refusal asserts that its output directory was never created; an earlier valid report remains byte-identical. |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v`; `bench-01.log` | 0 | All 42 existing benchmark tests. |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_reference.py -v`; `reference-unit-01.log` | 0 | Both existing metadata/framing tests. |
| `python3 tools/hybrid-bench/test_reference_native.py --client ...Q4_0.gguf --reference ...Q8_0-requant.gguf --artifacts .temp/hybrid-b-r27-261101/reference-native-01`; `reference-native-01.log` | 0 | Existing reference/native import/position/numerical witnesses retained; new schedule pin refusal and Q4 probe regression. 31 recorded commands: 14 at 0, 16 intended refusals at **2**, one intended wrong-row witness at **1**. |
| Existing checker inside three reference orchestration runs | **5** each | Heterogeneous equivalence remains `uncalibrated`; this is not a passing acceptance gate. The native scorers return 0. |
| `python3 .temp/hybrid-b-r27-261101/controls.py`; `controls-01.log`, `controls-01/controls.json` | 0 | Old r26 scorer executes at 0; the two new named witnesses actually exit **1** with the intended assertions on that result. |
| `python3 .temp/hybrid-b-r27-261101/audit.py`; `audit-01.log` | 0 | All 154 recorded owned PIDs inactive, native node ports closed, two successful study trees' receipts/hash/private modes valid, current native identities match, driver credentials removed, source syntax/ASCII. Final audit is repeated after this report is written. |

| Old behavior / plant | What it narrows the gate to | Named failing test and actual process exit | Survivor bound |
|---|---|---|---|
| r26 scorer without client schedule probes | Client-side variation must be reported, not inferred from Q8 encoding error | `test_client_schedule_floor`, **1**, `client schedule floor was omitted` | Killed; probe coverage is first/middle/last on the fixed CPU fixtures |
| r26 full local prefill versus scalar imported row zero, homogeneous Q4 delta | Same-model/same-partition row zero must not carry the former schedule asymmetry | `test_homogeneous_row0_control`, **1**, `row0 schedule asymmetry remains` | Killed; old row-zero KL 0.00019763826790027974, new 0 on the same document |
| Existing checksum-valid wrong-position / wrong-distribution reference plants | Structural position identity and an independent prefix-distribution witness | `shifted-position-control`, **2**; `test_prefix_position_witness`, **1** | Killed on the retained reference fixture; no universal GPU qualification |

No surviving planted control is claimed. The old-scorer controls use its saved r26 executable linked to the current build's unchanged-core sources; binary/result hashes and actual witness traces are retained. They do not stand in for an independent review of r27.

CPU numeric observations from the 48 continuation positions and nine schedule probes per arm:

| Arm | Mean KL(D to arm) | Max KL(D to arm) | Schedule-probe mean KL | Schedule-probe max KL |
|---|---:|---:|---:|---:|
| Q8 self | 1.3427056498631188e-08 | 9.486352014004368e-08 | 0 | 0 |
| Q4 local | 0.00013832093464185964 | 0.0021722407764195326 | 0.00012898168708906998 | 0.0008500437565400864 |
| Q8-node/Q4 imported | 6.593599100552661e-05 | 0.0012588329838999152 | 2.5692869532951722e-05 | 0.00019924436691334956 |

These three short, repeated synthetic documents are protocol and numerical witnesses. They provide no natural-document or model-quality inference. The report numbers match direct row recomputation; bootstrap resampling units are the three documents. Q4 same-model single-delta import has row-zero local-to-imported KL 0 in all three documents. The schedule observations are finite samples, not a claimed maximum over kernel paths, positions or devices.

Initial `native-01` harness exited **1** after imported-arm driver exit **2**, because my driver expected DELETE200 instead of the native contract DELETE204. The node was stopped/reaped and credentials removed; after correcting the expectation, native-02 and final native-03 exit 0. Unit-01 had a raw-regex SyntaxWarning, corrected before unit-04. No failed run is relabeled as passing. `ty` and `editorconfig-checker` are unavailable and were not run; no packages were installed.

Self-review used the repository code-review skill's scope/security/design/general/tests checklists. Changes are bounded to `tools/hybrid-bench`; no core/model/backend/API changes. Original delta hashes, finite native metric checks, versioned schedule identity, private files, transactional receipts, owned-child cleanup and numeric-loopback-only preparation were inspected. No supplied surface table exists, recorded as a coverage-map brief gap; no requested behavior is declared out of contract. Previously committed producer/reviewer tests are retained under their original names. Independent acceptance, hosted CI/landing, GPU capacity, BF16/Bonsai/Mirai values, ladder statistics, long-context stability and cross-backend effects remain unqualified. Publication was explicitly forbidden; delivery is a signed local handoff for review.

## r28: r6t driver fixes and bounded delta retention

Local rework on r27 `d4112d5496e2a9f5e50073058085bc349a2afb53`, without rewriting it or pushing. Review input: `/home/agent/Werkstatt/hybrid-review-proto/llama.cpp/.temp/REVIEW-hetero-study-driver-claude.md`. Ready for review; independent acceptance and target-GPU qualification remain pending. This section supersedes the earlier retained-delta and full-input report verification descriptions. The GPU commands above still apply with aligned chunks and a fresh study directory for the r28 receipt format.

All node HTTP, including DELETE204, now uses the numeric-loopback `api()` opener with proxies and redirects disabled. A release error cannot replace an earlier extend error. The capture regression uses a real loopback HTTP proxy, first proves capture with an unauthenticated GET returning 502, clears the capture, then runs the real imported arm under lower/uppercase HTTP/HTTPS proxy settings with no proxy bypass list. The arm succeeds and capture remains empty. No bearer value is printed or saved in capture evidence.

Successful imported scoring records delta SHA-256 and size alongside the reference/input binding and the row/result hashes in a checksummed receipt, then deletes that document's deltas by default. `--keep-deltas` retains successful deltas for debugging. Failed generation, scoring or catchable interruption removes only that attempt's deltas, even with `--keep-deltas`; logs remain. SIGKILL/host failure can still leave orphans and does not run cleanup. Missing debug delta files are allowed; their recorded hashes remain part of the receipt chain. Resume verifies live arm inputs, reference bytes, scored outputs and recorded bindings. A regenerated reference receipt or changed completed-document binding refuses stale rows before native work.

`report` verifies fixture hashes, each small row/result file's own hash, receipt checksums, and recorded index/fixture/token/arm/reference/delta bindings, including reference model/document/schedule metadata. Cached reference headers avoid reading the reference payload. It does not open model/build files, books, manifests or delta files. JSON `settings.verification` states this scope. It does not verify current existence/content of those large inputs, and checksums do not authenticate coordinated malicious rewrites of metadata. This deliberately differs from producing/resuming an arm. Pre-r28 receipts lack the cached headers and chain checksums and must be regenerated in a fresh study.

Chunks must be multiples of 256 in 256..8192. Before native arm work, stdout prints arm label, pending documents, delta reserve, row/log reserve, budget, margin and free bytes. Admission requires free space >= budget + margin. Delta reserve uses the existing 514 MiB HTTP response bound per possible chunk, multiplied by the largest pending document chain under default deletion or all pending chains with `--keep-deltas`. Row/log reserve is 100 MiB per pending document; margin is max(1 GiB, ceil(10% of budget)). Models and shared references already consume the measured filesystem space and are excluded from this incremental allowance. This is a conservative admission estimate, not a filesystem reservation against other writers or an unlimited-log bound.

At N32768, chunk4096, D20, the default budget is 6,408,896,512 bytes plus 1,073,741,824 bytes margin = **6.96875 GiB required free**, including a 4.015625 GiB upper bound for one live delta chain. Default six-arm execution no longer retains the earlier estimated 268 GiB of scored deltas. `--keep-deltas` reserves 88,332,042,240 bytes plus 8,833,204,224 bytes margin = **90.4921875 GiB required free** for one pending imported arm. Actual 27B geometry remains unmeasured; the earlier 44.659 GiB retained-arm estimate is geometry-based. Run each arm sequentially and keep shared D references.

### r28 evidence

All commands used `TMPDIR=$PWD/.temp/hybrid-b-r28-261101/tmp`, `PYTHONDONTWRITEBYTECODE=1`, and `nice -n 10`. No downloads, GPU use, installs, configure or native build. Existing r27 CPU binaries were reused unchanged; the audit records their full hashes and verifies the current engine/helper identities against the actual study receipts. Python source compilation and real native orchestration validate the changed driver; no C++ source changed.

| Command / evidence under `.temp/hybrid-b-r28-261101/` | Actual exit | Claim and limits |
|---|---:|---|
| `python3 -m unittest discover -s tools/hybrid-bench -p test_hetero_study.py`; `unit-02.log` | 0 | 12 tests. Default deletion/debug retention, deleted-input report with bounded hash-read allowlist, row corruption, failure/cancellation cleanup and foreign-file preservation, exact disk threshold/one-byte-below control, aligned/unaligned chunks, extend-vs-release error preservation, plus retained prepare/lock/bootstrap tests. Storage/scoring fixtures are synthetic here. |
| `python3 tools/hybrid-bench/test_hetero_study_native.py --client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --reference /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --build-dir build --artifacts .temp/hybrid-b-r28-261101/native-01`; `native-01.log` | 0 | Real CPU N64/M16/D3 tokenizer/reference/local/imported/homogeneous schedule witnesses and direct seeded statistics remain green. Proxy isolation and both stale-reference/binding paths refuse as specified. 25 CLI commands: 11 at 0, 14 intended refusals at **2**; no refused report published, previous valid report byte-identical. |
| `python3 .temp/hybrid-b-r28-261101/plants.py`; `plants-01.log`, `plants.json` | 0 | Unchanged binding and proxy controls exit 0; each F1/B/C mutant witness actually exits **1** at the named regression. The final tightened proxy assertion was rerun here on the real arm. |
| `python3 .temp/hybrid-b-r28-261101/binding-controls.py`; `binding-controls-01.log`, `plant-*-binding-03.log`, `valid-reference-control/control.json` | 0 | Final checksum-valid changed-reference witness: unchanged binding control 0, B/C mutants **1** at the intended arm/report assertions. The existing native scorer separately accepts the changed reference at **0**, ruling out a corrupt-row fixture. |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py`; `bench-01.log` | 0 | 42 existing benchmark tests. |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_reference.py`; `reference-01.log` | 0 | 2 existing reference tests. |
| `python3 -m unittest discover -s tools/hybrid-eval -p test_eval.py`; `eval-01.log` | 0 | 11 existing eval tests. |
| `python3 .temp/hybrid-b-r28-261101/audit.py`; `audit-02.log`, `audit.json` | 0 | Source ASCII/syntax, exact driver/build/helper identities, 25 expected CLI exits, 15 real completed receipts, all 63 recorded owned PIDs inactive, zero retained deltas/credentials in the native study tree, private JSON modes. Includes the extra proxy positive control, failed F1 attempt and final valid-reference controls. |

| Mutant | What it narrows the gate to | Named failing test and actual exit | Survivor bound |
|---|---|---|---|
| F1: DELETE through environment-sensitive `urlopen` | Every authenticated node call must bypass the capture proxy, including release | `test_arm_proxy_isolation`, **1**, `proxy received an arm request or bearer`; arm actually returns **2** after capture502 | Killed on the real CPU node/client path under the tested proxy environment |
| B: arm resume `receipt(path)` without expected binding | A changed valid reference receipt must invalidate a completed arm row before skip | `test_completed_binding_refused`, **1**, `stale-reference-arm` actually returns **0**, expected **2** | Killed at the arm refusal assertion |
| C: report `receipt(...)` without expected binding | A changed valid reference receipt must invalidate a row before report publication | `test_completed_binding_refused`, **1**, `stale-reference-report` actually returns **0**, expected **2** | Killed at the report refusal assertion; planted stale output remains isolated as evidence |

The binding regression also changes only a completed document's binding and recomputes its receipt checksum, so it reaches the expected-binding gate instead of stopping at corruption detection. The changed-reference case flips a finite float16 mantissa bit, recomputes that row's checksum, updates the recorded file hash, and reseals the reference receipt while preserving document/model/schedule metadata. Arm and report both refuse with `stale resume identity`, without new native processes or output. A separate real scorer run accepts that checksum-valid changed reference. Mutant modules preserve the unchanged driver's on-disk identity solely to isolate these gates from the earlier driver-change configuration gate; only the documented call is replaced. No surviving planted mutant is claimed.

Initial `unit-01.log` is an actual **1** with three synthetic-fixture setup errors (missing node/layout fields). Those fixture fields were corrected; `unit-02.log` is green. The native suite passed before the proxy assertion was tightened to run even after a CLI failure; that changed helper was separately rerun green and red by `plants.py`. Self-review then strengthened the changed-reference fixture from an altered checksum byte to valid changed content; the final binding helper and B/C plants were rerun by `binding-controls.py`. `ty` and `editorconfig-checker` are unavailable (readiness exit1), so their checks were not run. Existing native reference/scorer sources and tests are unchanged; their r27 evidence remains applicable, and the current real study reruns the affected orchestration. No hosted checks, independent review, GPU/BF16/Bonsai/Mirai/ladder or long-context qualification is claimed.

Self-review uses scope/security/general/tests/design checks from `skills/code-review/SKILL.md`. Diff stays inside `tools/hybrid-bench`. No surface table was supplied (coverage-map brief gap); no requested behavior is out of contract. Existing named reviewer regressions remain present. The requested co-author trailer and signed local commit are private-prototype delivery instructions; no upstream contribution or publication is attempted.

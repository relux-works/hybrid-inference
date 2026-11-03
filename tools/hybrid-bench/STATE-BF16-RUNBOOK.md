# bf16 state KL and top-1 on a fresh RTX PRO 6000

No GPU run or download was performed on agents1. `stage_kl_bf16.py` is an operator script for one exclusive 96 GB RTX PRO 6000 and a data volume of at least 160 GB. It rebuilds its own BF16-weight reference and serializes model/server lifetimes. All three treatment arms use the identical Q4_K_M GGUF: local, Q4-node import in f32, Q4-node import in bf16. The D reference uses BF16 **weights**; transport dtype is a separate variable.

## Historical reproduction is currently blocked

The surviving `kl-report-stage2.tar.gz` contains only CSV/Markdown/JSON reports in `report27b-seed42` and `report27b-local-seed42`. JSON contains `rows` and `settings`: document IDs, aggregate metrics, seed 42, 10000 bootstrap resamples and index digest `fcff54eac27feffe925d4895b02bf3a68bca08c719c9a218d33d771a1d51b206`. It contains **no document byte hashes or token hashes**. An aggregate index digest cannot recover them. A fresh absolute path also changes the original index's file records, so matching document/token hashes is the portable identity check; an index digest alone is insufficient.

There is also **no homogeneous Q4-to-Q4 imported row**. The historical Q4-node rows are `B-Bonsai-Q4-imported` and `B-Mirai-Q4-imported`; their clients differ from Q4. The retained `/home/agent/lanes/hybrid/.temp/vast/stage-kl2.sh` confirms that mapping. Comparing either heterogeneous row against the requested homogeneous f32 arm as a reproduction test is invalid. `C-Q4-local` is available as a limited sanity check, not an imported-arm reproduction control.

Model pins were recovered independently from `/home/agent/lanes/hybrid/.temp/vast/host-setup.sh` and copied into `state_bf16_model_pins.json`:

| GGUF | Repository revision | Bytes | SHA256 |
|---|---|---:|---|
| Qwen3.8-27B-Q4_K_M.gguf | `ggml-org/Qwen3.8-27B-GGUF` at `0669b98607d47046c7c2b3f801011d54a08cfccf` | 18973870432 | `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| Qwen3.8-27B-BF16.gguf | same revision | 53808281952 | `5a3eedc837bcbd1365cdbf5b71e698df3122e76586ba872f07ce3ed4a9bfa97e` |

Audit the saved report without downloads/builds/GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/hybrid-bench/stage_kl_bf16.py \
  --historical-archive /path/kl-report-stage2.tar.gz --audit-only
```

With the currently surviving archive this prints the missing evidence and ends `failed kl-bf16-stage: historical provenance absent; exact reproduction is blocked`, exit 1. That is the intended refusal. Reproduction mode requires an independently recovered provenance lock and a homogeneous historical imported comparator, which the current reports do not provide. Recover those inputs or explicitly choose the newly pinned study below. Do not insert invented hashes or relabel a heterogeneous comparator.

## Operator alternative: explicitly pin a new study

This is a new experiment with the historical document IDs/order and settings, **not exact stage-2 reproduction**. It needs the owner's choice of this alternative. On a fresh GPU node install the reviewed source commit in a clean separate checkout using the operator's normal source transfer; install CUDA/CMake/compiler/Python beforehand. The scripts install no tools or packages. Copy only the small surviving report archive. No old D reference is needed.

```bash
export PYTHONDONTWRITEBYTECODE=1
export SRC=/root/proto/src-state-bf16
export ROOT=/root/kl-state-bf16
export BUILD="$SRC/build-state-bf16"
export CANDIDATE_SHA='REPLACE_WITH_REVIEWED_40_HEX_SHA'
export OLD_REPORT=/root/kl-report-stage2.tar.gz
mkdir -p "$ROOT"
# Explicit first-use pinning of new Gutenberg bytes and tokens, on the GPU host only.
python3 "$SRC/tools/hybrid-bench/stage_kl_bf16.py" \
  --historical-archive "$OLD_REPORT" --mode new-study --bootstrap-new-lock \
  --root "$ROOT" --build "$BUILD" --ref "$CANDIDATE_SHA" --execute
# Preserve new-provenance-lock.json off-node before the expensive reference/arms.
python3 "$SRC/tools/hybrid-bench/stage_kl_bf16.py" \
  --historical-archive "$OLD_REPORT" --mode new-study \
  --provenance-lock "$ROOT/new-provenance-lock.json" \
  --root "$ROOT" --build "$BUILD" --ref "$CANDIDATE_SHA" --execute
```

Bootstrap downloads the two pinned models using immutable revision URLs, checks size and SHA256 before use, fetches exactly the 20 IDs in `gutenberg_sources.json`, and freezes each downloaded document's SHA256/size. It tokenizes the complete UTF-8 document with Q4's `/tokenize(add_special=true, parse_special=false)`, takes the first N=32768 tokens plus the next M=2048, and freezes SHA256 over their concatenated little-endian int32 IDs. There is no random document/window selection in `hetero_study.prepare`; the fixed source order and prefix rule are deterministic given bytes/tokenizer. Seed 42 is used for the statistical bootstrap. Changed or too-short documents refuse. New pins are labelled `newly-pinned-study`, never represented as recovered historical hashes.

The measurement phase validates the lock, source ref, GPU identity, settings and disk before work. It verifies all source/token hashes again before reference or arm execution. It builds once, niced with `-j4`, under a native build lock after `pgrep`/`uptime`. The original index/fixtures and every source/model/build hash are retained. It computes D with the BF16 model, score batch 32 and the same split-last/fresh-prefix schedule probes as the original driver. It then runs local Q4, homogeneous f32 import and homogeneous bf16 import at chunk size 4096, with successful deltas removed after scoring. A shared GPU lock serializes tokenizer, D and arm processes. No `--keep-deltas` is used.

The report contains KL against D, top-1 agreement, per-document values, paired document differences and 10000-resample document bootstrap intervals, seed 42. See `report/report.json`, `.csv`, `.md`; the native per-document results also retain client local-to-import KL/top-1 and schedule probes. Compare bf16-minus-f32 and bf16-minus-local, and use the client's historical partition-noise KL band 0.0014-0.0112 as context, not a universal threshold. Top-1 agreement is against D's argmax on the same teacher-forced positions, not free-running answer identity.

`reproduction.json` labels new-study mode `new-study-no-exact-reproduction`, with no f32 historical verdict. Its local-Q4 historical sanity check tests both current document-mean KL and top-1 against `C-Q4-local`'s saved 95% bootstrap intervals. A failure is retained as `passed=false`; passing cannot establish historical document identity, equivalence or imported-arm reproduction. In reproduction mode, only a recovered **homogeneous** imported comparator is allowed; both f32 means must lie inclusively within its saved KL/top-1 intervals, or the run is flagged and exits 1 after retaining all results. The supplied archive cannot satisfy this mode.

## Provenance lock and recovery

The version-1 lock has `historical_report_sha256` (SHA256 of the uncompressed full report JSON), `settings` exactly `{seed:42,d:20,n:32768,m:2048,score_batch:32}`, `models` with the pinned repository/revision/primary file/files list shown in `state_bf16_model_pins.json`, and `documents` with ordered `{id,url,size,sha256,token_sha256}` records. `provenance` states `kind` (`newly-pinned-study` or `recovered-stage2-lock`) and a nonempty `source`. A recovered homogeneous comparator also needs `f32_comparator` with `arm=B-Q4-Q4-imported`, `node_model_sha256` and `client_model_sha256` matching Q4. That arm must actually exist in the saved report; no such row is synthesized.

Retry measurement with the identical command: receipts skip unchanged completed reference/documents/arms; changed identities refuse. Do not rerun bootstrap after a valid new lock exists. Preserve failed attempts and `.partial` model files; inspect/move only those owned partials before retrying a failed download. A changed model/hash never becomes a successful download. Scripts stop only child PIDs they created and retain tokenizer PID/exit evidence. Manifest, lock, GPU identity, process/command exit records and `outputs.json` bind the report. Copy the complete private `ROOT` (including documents, fixtures, receipts and 18.9 GiB D reference) to durable storage if independent recalculation is required; the small report alone was insufficient for this reconstruction.

Terminal success is `done kl-bf16-stage seconds=...`, exit 0. Any refusal/error or failed f32 reproduction has `failed kl-bf16-stage: ...`, exit 1. A successful new-study run explicitly retains its non-reproduction label. Control-C/SIGTERM cleans up owned tokenizer/study children; SIGKILL/power loss needs inspection of owned PID records before retry. Never kill by process-name pattern.

## Estimates

Model download is **72,782,152,384 B (72.78 GB)** plus roughly tens of MB of Gutenberg text. At 100/50/10 MB/s the model bytes alone take 12.1/24.3/121.3 min. Reserve 20-60 min on a fast link, or 2-3 h at 10 MB/s, including verification; bandwidth is unmeasured. No additional model is downloaded.

D occupies **20,344,176,640 B (18.947 GiB)**. Models plus D total 93.13 GB. Allow 6 GiB for the CUDA build, 7 GiB for transient deltas/work, 2 GiB for logs/rows/archives, and the 28 GB free floor. Admission requires approximately 136.2 GB initially free for a fresh run, on a volume of at least 160 GB. The script recomputes remaining model/reference space on resume and rechecks free space before each major step. GPU-stage checks accept `--disk-floor-bytes INTEGER` (default 28,000,000,000; minimum 1,073,741,824 = 1 GiB); all model/reference, build, download and transient extras stay on top of the chosen floor. Keep the default for this fresh-node plan. The floor is recorded in `bootstrap-plan.json`, `bootstrap.json` and the measurement `manifest.json`, and disk-gate events in terminal output and `disk.log`/`build.log`/`models/download.log`. A resumed measurement must use the same floor as its manifest. BF16 wire does not shrink the D logits reference or native f32 slot snapshots. Do not retain successful delta chains without another roughly 48 GB/f32 arm or 35.4 GB/bf16 arm.

Reference runtime was not retained in the surviving reports. Reserve **2-4 h for regenerating D**, **1-2 h for the three Q4 arms**, and **10-25 min for the first CUDA build**; with fast downloads and I/O reserve **4-8 h total**. These are unmeasured planning bounds. After the first D document and first imported document, extrapolate measured elapsed time x20 plus 25% reserve before committing the full reservation. Cleanup the candidate build only after all owned processes have stopped, preserving its record/logs; retain models, reference and provenance for replication.

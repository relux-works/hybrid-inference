# Combined KL study and Mirai S, hybrid-c-r2-261101

Ready for review. CPU validation on agents1, 2026-11-01. No push, GPU execution, model/book download or quality ranking.

## Heads and merge resolution

- Accepted Mirai parent: `1d79f755f5d06cacb96981f1207915b452782ae9`.
- NOTICE commit on `m3-mirai-s`: `eed20cace1c42a3ec094b84a7410a4d3c7c4ffb6`, signed and verified (`G`, verify-commit exit 0). Only `tools/mirai-s/NOTICE` changed. It explicitly identifies the CUDA file's sm_80/HIP/MUSA guards and the Ampere admission gate in `ggml-cuda.cu`.
- `m3-study-mirai` starts at that NOTICE commit and merges the exact fetched `m3-reference-logits` head `46822cd52b148ddcef66865478891808baec27ee` from `https://github.com/relux-works/hybrid-inference`. The containing merge preserves both parent histories without squash or rebase. These user-selected parents supersede the usual current-main starting rule.

`git merge-tree --write-tree` and the actual `git merge --no-ff --no-commit` each exited 1 with these two content conflicts. Both are resolved:

| File | Resolution |
|---|---|
| `tools/hybrid/CMakeLists.txt` | Keep both `llama-hybrid-check-identity` from Mirai and `llama-hybrid-score` from the study, each linked to `hybrid-state`. |
| `tools/hybrid-bench/UNRESOLVED_QUESTIONS.md` | Retain both entries: Mirai GPU/greedy/state-import qualification and r28 study receipt/schedule/capacity/portability limits. |

Automatic merges were inspected: root README keeps both tools' documentation; `manage.py` keeps explicit client `--spec-type none` and adds the study scorer to client build targets. `source-preservation.json` verifies 50 files byte-for-byte against the parent that changed them. The four shared paths are exactly README, manage, CMake and unresolved questions. Scorer, driver, reference format, Mirai CPU/CUDA math, model integration, MTP accounting and existing tests retain their reviewed source. The only new executable content is two tests in `test_mirai_identity.py`.

## CPU validation

Evidence root `R=.temp/hybrid-c-r2-261101`; `B=$R/build`; CMake `C=.temp/hybrid-c-r1-261101/venv/bin/cmake` (3.31.10, readiness exit 0, no new installation). Build and runtime validation used disk `TMPDIR=$PWD/$R/tmp`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=2` and `nice -n 10`; Python tests used `PYTHONPATH=tools/hybrid-proxy:tools/hybrid-bench`. One build invocation, four jobs, PID in `build-01.pid`; no other host build was observed at admission. Compiler GNU 15.2.0, Linux x86_64. The build embeds pre-merge HEAD `eed20cace`; source and binary identities are retained in the audit.

| Command (build/runtime commands prefixed with `nice -n 10`) | Actual exit | Count / evidence |
|---|---:|---|
| `$C -S . -B $B -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF` | 0 | `configure-01.log` |
| `$C --build $B --target llama-server llama-hybrid-score llama-hybrid-prefill llama-hybrid-check-identity llama-hybrid-bench-check llama-hybrid-check-codec llama-hybrid-check test-backend-ops -j4` | 0 | `build-01.log`; CPU only |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_hetero_study.py -v` | 0 | 12/12; `study-unit-01.log` |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_reference.py -v` | 0 | 2/2; `reference-unit-01.log` |
| `python3 -m unittest discover -s tools/hybrid-bench -p test_bench.py -v` | 0 | 42/42; `bench-unit-01.log` |
| `python3 -m unittest discover -s tools/hybrid-eval -p test_eval.py -v` | 0 | 11/11; `eval-unit-01.log` |
| `python3 tools/hybrid-bench/test_mirai_identity.py --binary $B/bin/llama-hybrid-check-identity --headers /home/agent/lanes/hybrid/.temp/research/bonsai/gguf-dumps --artifacts $R/identity-native-01 -v` | 0 | 12/12, no skips; 34 native loader calls: 11 at 0, 23 intended refusals at 1; `identity-native-01.log` |
| `python3 tools/hybrid-bench/test_reference_native.py --client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --reference /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --build-dir $B --artifacts $R/reference-native-01` | 0 | 31 recorded scorer/witness commands: 14 at 0, 16 intended refusals at 2, one intended witness failure at 1. Analytic self-test 0; three legacy equivalence checks remain uncalibrated at 5. `reference-native-01.log`, `commands.json` |
| `python3 tools/hybrid-bench/test_hetero_study_native.py --client /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf --reference /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q8_0-requant.gguf --build-dir $B --artifacts $R/study-native-01` | 0 | 25 CLI commands: 11 at 0, 14 intended refusals at 2. D3/N64/M16 tokenizer, reference, local/imported/same-model arms, schedule witnesses, live proxy capture control, stale bindings and direct seeded statistics. `study-native-01.log`, `commands.json` |
| `$B/bin/test-backend-ops test -b CPU -o MIRAI_MUL_MAT` | 0 | 39/39; 33 random-code CPU/reference comparisons plus six independent known-output cases; `mirai-backend-01.log` |
| `$B/bin/test-backend-ops test -b CPU -o GATED_DELTA_NET -p 'n_seq_tokens=1,n_seqs=1,'` | 0 | 5/5 selected recurrence cases; `gdn-regression-01.log` |
| `$B/bin/test-backend-ops test -b CPU -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=4,nr23=\[1,1\],kv=512,nb=1,'` | 0 | 8/8 selected attention cases; `fa-regression-01.log` |
| `$B/bin/llama-hybrid-check-codec` | 0 | Existing codec positive/negative controls; `codec-regression-01.log` |
| `$B/bin/llama-hybrid-check /home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf $R/qwen-golden 512` | 0 | 64/64 greedy equality, range append and unchanged-state rejection controls; token JSON exactly matches r1; `qwen-regression-01.log` |
| `python3 .temp/hybrid-c-r1-261101/code-style.py` | 0 | Workflow model-naming checker, 166 mappings; `code-style-01.log` |
| `python3 $R/audit.py`; `git diff --check`; `git diff --cached --check`; `bash -n $R/gpu-commands.sh` | 0 each | `audit-03.log`, syntax/ASCII, source/binary identities, all 87 recorded owned PIDs inactive; GPU script syntax only |

Total unittest methods: 79. Selected backend cases: 52. These are separate counts from the native subprocess controls.

`test_study_mirai_bf16_hetero_identity` uses the real external header bytes for driver arm SHA admission and shared layout admission, then compares native loader identities for Mirai and stock BF16. Distinct header hashes require explicit `--hetero-quant`; absence of the flag refuses. Both native layouts have 64 trunk layers and fingerprint `d4fb520322100b1e438d60aa641ee86bd0d22576198a6dc49890c04f181924a8`. `test_study_unknown_extra_refused` adds `blk.64.nextn.surprise.weight` to a header fixture preserving real tokenizer bytes. Python rejects `unknown extra tensors`; the scorer's native model loader, exercised via its no-allocation checker adapter, returns normal exit 1 with `wrong number of tensors` and empty identity output. Valid Mirai/BF16 controls pass. External header hashes are checked before tests. Header tests establish static admission, not weight payload validity, 27B scoring or GPU parity.

Retained live narrowing controls:

| Plant | Gate narrowed to | Named failing test / real exit | Outcome / bound |
|---|---|---|---|
| Checksum-valid shifted absolute row position | Position binding beyond checksums | `shifted-position-control`, native exit 2, `reference position mismatch` | Killed; metadata binding only |
| Correct marker/checksum with a next-position distribution | Predictive distribution at an independently prefilled prefix | `test_prefix_position_witness` via `wrong-row-witness`, exit 1; scorer itself exits 0 | Killed on this finite CPU fixture |

No surviving planted control is claimed. Existing r1/r28 mutation campaigns were not repeated or relabeled as current-run evidence.

The first audit exited 1 because it assumed all `.pid` files contain bare integers; one existing harness writes a JSON PID/identity record. The parser was corrected; `audit-02.log` and the final audit are green. No product/test source was changed for that correction. Expired test-node credentials were removed only after all recorded owned PIDs were inactive. No suite failed on the merged candidate. Existing compiler warnings remain visible in the build log. `ty` and `editorconfig-checker` readiness each exited 1 (unavailable); their hosted checks were not run. No CUDA/HIP/MUSA/Metal build, hosted CI, GPU/model conversion, actual 27B state transfer, natural-book scoring or performance/quality result is claimed.

Self-review used scope, security, general and tests checks, plus inspection of backend/model preservation and the CMake integration. All original test names remain; no requested CPU case is declared out of contract. No normative surface table was supplied (coverage-map brief gap). Independent review and GPU qualification remain separate gates. Delivery is signed and local per the explicit no-push instruction and requested Codex co-author trailer.

## GPU-host Mirai study commands (unrun)

Run from this combined branch on the GPU host, using already verified local weights. `build-study-mirai-cuda` is a CUDA build of the combined tree, not the agents1 CPU build. Both client `--build-dir` and node `--node-build-dir` below point to it. Require an admitted Ampere-or-newer NVIDIA device, existing CUDA toolchain, available build slot and exclusive GPU use. Record toolchain/driver versions and build identities with the study. Complete r1's moderate-context greedy-token parity against alesha-pro before admitting Mirai measurement arms. This run does not establish that gate.

These are r27's fetch/pin, prepare D20/N32768/M2048, shared BF16 reference, local controls, three sequential Mirai imported arms and seed42 report steps, with r28 receipts and default delta deletion. Use a fresh study directory. MTP remains disabled: scorer uses the unchanged default `load_mtp=false`, prefill explicitly disables it, and the client launcher retains `--spec-type none`. Imported scoring needs room for two client contexts and sequential schedule probes. At the default 20 documents/chunk4096, r28 requires about 6.96875 GiB free for a pending imported arm; shared BF16 reference payloads add about 18.947 GiB at V248320. Keep the resulting book pins, immutable local model/build files and shared D references for resume. Do not add `--keep-deltas` unless its larger disk budget is available.

```bash
#!/usr/bin/env bash
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
STUDY_HOME=/gpu-disk/hybrid-c-r2-261101
mkdir -p "$STUDY_HOME/tmp" "$HOME/.cache/hybrid-study"
export TMPDIR="$STUDY_HOME/tmp"
DRIVER=tools/hybrid-bench/hetero_study.py
BUILD="$PWD/build-study-mirai-cuda"
D_MODEL=/models/Qwen3.8-27B-BF16.gguf
Q8_MODEL=/models/Qwen3.8-27B-Q8_0.gguf
Q4_MODEL=/models/Qwen3.8-27B-Q4_K_M.gguf
MIRAI=/models/Qwen3.8-27B-S-mirai.gguf
STUDY="$STUDY_HOME/qwen27b"
D_SHA=$(sha256sum "$D_MODEL" | cut -d ' ' -f 1)

git rev-parse HEAD > "$STUDY_HOME/source-head.txt"
git merge-base --is-ancestor eed20cace1c42a3ec094b84a7410a4d3c7c4ffb6 HEAD
git merge-base --is-ancestor 46822cd52b148ddcef66865478891808baec27ee HEAD
nice -n 10 cmake -S . -B "$BUILD" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF
nice -n 10 cmake --build "$BUILD" -j4 --target llama-server llama-hybrid-score llama-hybrid-prefill llama-hybrid-check-identity
sha256sum "$D_MODEL" "$Q8_MODEL" "$Q4_MODEL" "$MIRAI" > "$STUDY_HOME/model-sha256.txt"
nvidia-smi > "$STUDY_HOME/nvidia-smi.txt"
"$BUILD/bin/llama-hybrid-check-identity" "$D_MODEL" > "$STUDY_HOME/bf16-identity.json"
"$BUILD/bin/llama-hybrid-check-identity" "$MIRAI" > "$STUDY_HOME/mirai-identity.json"

# Fetch books on the GPU host and retain the resulting pinned manifest.
tools/hybrid-bench/fetch --manifest tools/hybrid-bench/gutenberg_sources.json --output "$STUDY_HOME/books"
nice -n 10 "$BUILD/bin/llama-server" -m "$D_MODEL" -ngl 999 -c 35072 -np 1 \
  --host 127.0.0.1 --port 8099 --no-webui > "$STUDY_HOME/tokenizer.log" 2>&1 &
TOKENIZER_PID=$!
printf '%s\n' "$TOKENIZER_PID" > "$STUDY_HOME/tokenizer.pid"
trap 'kill "$TOKENIZER_PID" 2>/dev/null || :; wait "$TOKENIZER_PID" 2>/dev/null || :' EXIT INT TERM
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

nice -n 10 python3 "$DRIVER" reference --study "$STUDY" --model "$D_MODEL" \
  --build-dir "$BUILD" --gpu-layers 999 --score-batch 32
nice -n 10 python3 "$DRIVER" arm --study "$STUDY" --kind local --label D-local --client-model "$D_MODEL" \
  --reference-model-sha256 "$D_SHA" --build-dir "$BUILD" --gpu-layers 999 --score-batch 32
nice -n 10 python3 "$DRIVER" arm --study "$STUDY" --kind local --label C-Q4-local --client-model "$Q4_MODEL" \
  --reference-model-sha256 "$D_SHA" --build-dir "$BUILD" --gpu-layers 999 --score-batch 32
nice -n 10 python3 "$DRIVER" arm --study "$STUDY" --kind local --label A-Mirai-local --client-model "$MIRAI" \
  --reference-model-sha256 "$D_SHA" --build-dir "$BUILD" --gpu-layers 999 --score-batch 32
for node in BF16 Q8 Q4; do
  case "$node" in BF16) NODE_MODEL="$D_MODEL";; Q8) NODE_MODEL="$Q8_MODEL";; Q4) NODE_MODEL="$Q4_MODEL";; esac
  nice -n 10 python3 "$DRIVER" arm --study "$STUDY" --kind imported --label "B-Mirai-$node-imported" \
    --client-model "$MIRAI" --node-model "$NODE_MODEL" --hetero-quant \
    --reference-model-sha256 "$D_SHA" --build-dir "$BUILD" --node-build-dir "$BUILD" \
    --gpu-layers 999 --score-batch 32 --chunk-tokens 4096 --ram-budget-mib 4096
done
python3 "$DRIVER" report --study "$STUDY" --arms D-local C-Q4-local A-Mirai-local \
  B-Mirai-BF16-imported B-Mirai-Q8-imported B-Mirai-Q4-imported --seed 42 --output "$STUDY_HOME/report27b-seed42"
```

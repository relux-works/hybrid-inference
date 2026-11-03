# bf16 state KL and top-1 on one RTX PRO 6000

No GPU experiment was run on agents1. Run on one exclusive 96GB RTX PRO 6000 with the existing models, stage-2 corpus pins, prepared index and reference receipts. This compares transport dtype, not BF16 versus quantized model weights. All treatment models must be the same Q4_K_M GGUF unless a separate heterogeneous study is explicitly named.

Use the exact stage-2 study from FACTS section 3: seed 42, D=20 Gutenberg documents, N=32768 context tokens, M=2048 continuation positions/document, score batch 32, BF16-weight D reference. Reuse the original `index.json`, token fixtures and `references/`; do not fetch/tokenize new books or regenerate the reference silently. There are 40960 scored positions per arm. Reference receipts, file sizes, model SHA, tokenizer layout, token hashes and score schedule are checked. If any differs, stop and locate the frozen artifacts. Original local/f32 arm receipts can be included in a report without rerunning them; new labels record candidate executable identities. The old study's imported arms can have different node weights: use the homogeneous f32 control below to isolate state dtype.

Set absolute paths to the existing study, immutable candidate build and local model files. Build CUDA once using the existing offline build procedure; disable both UI asset sources. Do not replace a running service's binaries. The driver serializes node and scorer lifetimes, so the BF16 reference, Q4 node and client need not all occupy VRAM together.

```bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/disk/tmp-bf16
mkdir -p "$TMPDIR"
export STUDY=/disk/kl-stage2/study27b
export BUILD=/disk/hybrid-state-bf16/build
export Q4_MODEL=/disk/models/Qwen3.8-27B-Q4_K_M.gguf
export D_MODEL=/disk/models/Qwen3.8-27B-BF16.gguf
export D_SHA=$(sha256sum "$D_MODEL" | cut -d' ' -f1)
export DRIVER=tools/hybrid-bench/hetero_study.py
export LOCK=/disk/hybrid-gpu.lock

nice -n 10 python3 "$DRIVER" --lock "$LOCK" arm --study "$STUDY" \
  --kind local --label state-Q4-local --client-model "$Q4_MODEL" \
  --reference-model-sha256 "$D_SHA" --build-dir "$BUILD" --gpu-layers 999 --score-batch 32
nice -n 10 python3 "$DRIVER" --lock "$LOCK" arm --study "$STUDY" \
  --kind imported --label state-f32 --client-model "$Q4_MODEL" --node-model "$Q4_MODEL" \
  --state-dtype f32 --reference-model-sha256 "$D_SHA" --build-dir "$BUILD" \
  --gpu-layers 999 --score-batch 32 --chunk-tokens 4096 --ram-budget-mib 4096
nice -n 10 python3 "$DRIVER" --lock "$LOCK" arm --study "$STUDY" \
  --kind imported --label state-bf16 --client-model "$Q4_MODEL" --node-model "$Q4_MODEL" \
  --state-dtype bf16 --reference-model-sha256 "$D_SHA" --build-dir "$BUILD" \
  --gpu-layers 999 --score-batch 32 --chunk-tokens 4096 --ram-budget-mib 4096
nice -n 10 python3 "$DRIVER" --lock "$LOCK" report --study "$STUDY" \
  --arms state-Q4-local state-f32 state-bf16 --seed 42 --output /disk/kl-stage2/report-state-bf16
```

No model/corpus download occurs in these commands. The existing D reference payload remains needed. For a retained original local arm, replace `state-Q4-local` in the report with its exact label and omit its arm command. A stale new arm label refuses; choose a new label instead of editing receipts.

Read `report.json`, `report.csv`, `report.md`, the per-document arm receipts and their native result files. Report KL against the common D reference, top-1 agreement, paired document differences and document bootstrap intervals. Native results also contain `local_to_hybrid` KL/top-1 against that client's local prefill and full-precision schedule probes. The measured client's partition-noise band KL 0.0014-0.0112 is a contextual reference, not a universal acceptance threshold. Report both bf16-minus-f32 and bf16-minus-local changes; a small difference against a large weight-quantization baseline is not proof of exactness. Preserve source/build/model/index hashes with every report.

For latency, use the RULER runbook's retained `seconds`, `wire_bytes`, `transfer_ms`, `import_ms` and per-chunk traces, paired against f32. On one card/loopback this measures functional import and encode costs; it does not establish Wi-Fi TTFT. Native import time includes bf16 expansion and validation.

Budget: existing shared D reference is 20344176640B (18.947GiB), plus model files. Q4 and BF16 model weights together are roughly 17GB and 54GB; use actual file sizes. One successful 32k document chain at chunk4096 retains at most about 2.40GB f32 or 1.77GB bf16 until scoring; default successful deltas are then removed. The conservative driver admission uses eight 514MiB response caps plus 100MiB/document and a margin: allow at least 7GiB extra free space for a new 20-document arm, above the host's required floor, even though normal retention is smaller. Allow another 2GiB for logs/rows/failed attempts. Do not use `--keep-deltas` unless budgeting about 48GB per f32 arm or 35.4GB per bf16 arm. Reference data, models and previous attempts are additional.

Time is unmeasured for this arm. For planning, 20 x 32768 prefills at the published PRO 6000 M1-M3 rates (~1.1k-4.3k tok/s) is 2.5-10 minutes for one pass. Each imported scorer also computes its own local prefill, teacher-forced continuation and up to three full-prefix probes plus imported replays; process/model loading is additional. Reserve 1-2 hours for the three-arm comparison with existing references, then replace this estimate with the first document's elapsed time x20 plus 25% reserve. This is a geometry/rate estimate, not a bf16 speed result.

The driver owns its PIDs and stops only those children. Preserve incomplete attempt logs and process records; resume identical commands after cleanup. Keep the shared GPU lock and serialize unrelated GPU jobs too.

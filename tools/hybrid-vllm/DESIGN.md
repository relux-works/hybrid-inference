# vLLM 0.32.0 prefill state bridge

Status: CPU prototype, GPU qualification pending. Base: `a3752b04cbd905bf61ffe6d9a4e7df738f26f4c2`. Scope: the approved private fork, no upstream submission. Source inspected: vLLM tag `v0.32.0`, commit `5df63a7203cdb9328d824a27b682e77c7b9928be`. No models or GPU were used on agents1.

## Route comparison

| Route | Source finding | Decision |
|---|---|---|
| A: V1 KV connector | `SupportsHMA.request_finished_all_groups` receives block IDs for all cache groups before release. NIXL implements HMA and explicit conv/SSM descriptors for Mamba/GDN, including TP mapping. Thus it can carry hybrid state; an attention-only connector cannot be assumed to. | Viable longer-term. NIXL adds device transfer registration, peer descriptors and completion ownership. It does not emit llama-native state, q8_0 or our envelope. |
| B: post-forward runner hook | The runner retains request token counts and group block tables while bound layers expose typed KV or conv/SSM views. A synchronous copy after `_model_forward`, before postprocess/sampling, captures a known prompt endpoint before release. | Chosen for this one-request prototype. Small explicit source patch, no scheduler modification or transport dependency. |
| C: offline `LLM` and direct tensor reads | An in-process executor could reach the same runner. After `generate()` the scheduler may already have released/recycled the request's blocks; the public result contains no safe state lease. Multiprocess execution further separates the runner. | Useful as a driver only if paired with B. Reading cache tensors after generation without retained ownership is rejected. |

## Source evidence

All vLLM paths below are relative to the pinned source root. Links use that immutable commit, not main. Local research file hashes are in `.temp/hybrid-a-r22-261101/source-files-01.json`; the downloaded clone is removed after qualification of the patch location.

| Claim | Source at inspected commit |
|---|---|
| HMA completion receives every group's blocks and can retain them until async save finishes | [`vllm/distributed/kv_transfer/kv_connector/v1/base.py:100-126`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L100-L126) |
| NIXL supports HMA; Mamba descriptors distinguish conv and SSM state | `.../nixl/connector.py:78`, [`.../nixl/base_worker.py:1828-1889`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L1828-L1889), `.../nixl/tp_mapping.py:35-37` |
| NIXL has special N-1 replay semantics for Mamba consumers | `.../nixl/base_scheduler.py:375-418`; this bridge does not inherit that protocol |
| Request state includes prompt tokens, computed count and block IDs by group | `vllm/v1/worker/gpu_input_batch.py:35-48` |
| State update precedes forward; forward precedes logits and sampling | [`vllm/v1/worker/gpu_model_runner.py:3940-4290`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu_model_runner.py#L3940-L4290); hook immediately before the unique postprocess marker, originally line 4215 |
| Logical dense KV is [B,H,N,2D], independent of physical stride order | `vllm/v1/kv_cache_interface.py:319-334,377-444`; `vllm/v1/kv_cache_layout.py:15-37`; `vllm/v1/attention/backends/flash_attn.py:1251-1274` |
| FP8 storage can be uint8, interpreted as E4M3FN/E5M2; separate per-layer K/V scales dequantize it | `vllm/v1/attention/backends/flashinfer.py:2110-2125`; `.../flash_attn.py:1289-1291` |
| Manager pages can be split into smaller kernel pages | `vllm/v1/worker/utils.py:460-496,635-680`; hook gathers manager page p from kernel pages p*r through (p+1)*r, then restores [H,N,2D] |
| Bound Mamba cache unpacks raw pages into typed state views | `vllm/model_executor/layers/mamba/abstract.py:28-42`; `vllm/v1/worker/utils.py:866-884` |
| Conv shape is SD=[kernel-1,channels] or DS=[channels,kernel-1]; GDN state is [value_heads,value_dim,key_dim] | [`vllm/model_executor/layers/mamba/mamba_utils.py:28-50,98-110,280-302`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/model_executor/layers/mamba/mamba_utils.py#L280-L302); `vllm/model_executor/models/qwen3_5.py:609-640` |
| GDN forward updates conv and recurrent state; recurrent final state replaces selected slots | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1390-1398,1449-1463,1590-1619` |
| The [V,K] state convention is mathematical, not inferred from equal dimensions | `vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py:87-116`; `qwen_gdn_linear_attn.py:234-280` passes the same state to FlashInfer |
| Align mode selects block_table[(seq_len-1)//block_size] and preserves scheduler endpoints, not arbitrary historical states | [`vllm/v1/attention/backends/utils.py:1163-1200`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/attention/backends/utils.py#L1163-L1200); `vllm/config/cache.py:189-196`; `vllm/v1/kv_cache_interface.py:1082-1096` |

Native references at the requested base: `src/llama-context.cpp:3471-3491` adds host magic and source sequence ID; `src/llama-memory-hybrid.cpp:198-203` writes suffix attention then full endpoint recurrent state; `src/llama-kv-cache.cpp:2219-2308` writes cell metadata, rotation tags, all K layers then all V layers; `src/llama-memory-recurrent.cpp:905-985` writes all conv R rows then all S rows. `tools/hybrid/hybrid-state.cpp:35-40,128-180` defines fingerprint and envelope/checksums. The native synthetic round-trip exercises these readers and writers without weights.

`conversion/qwen.py:485-523,621-672` reorders V heads from grouped `[key_head,repeat]` to tiled `[repeat,key_head]`. The converter applies that permutation to the V channels of conv history and the recurrent head axis. It retains `[V,K]` within each head, matching `ggml/src/ggml-cpu/ops.cpp:11212-11258`. Conv rows become `[channel,time]` with oldest-to-newest history. Full-attention keys are already post-RoPE; there is no additional RoPE or weight permutation here. `src/llama-kv-cache.cpp:304-345` enables normalized Hadamard rotation for q8_0 (K=256, V=64); the converter rotates before quantizing, or writes [0,0] only for a client explicitly launched with `LLAMA_ATTN_ROT_DISABLE=1`. Fingerprints do not encode rotations; native rotation tags independently enforce that contract.

## Restrictions and unverified behavior

- Exactly 64 blocks, attention layers 3,7,...,63 and 48 GDN layers. Four 256-dimensional KV heads; GDN channels 10240, history 3, state [48,128,128]. Model class and HF geometry are checked in the hook. Target layout comes from the existing GGUF identity reader; its canonical JSON fingerprint is recomputed, including vocabulary hash.
- BF16 or per-tensor FP8 E4M3FN/E5M2 KV only; BF16/f32 conv and recurrent state only. FP8 scale must be finite and positive. FP16, per-token/per-head quantization, FP4, unknown state records, mixed groups, different head shapes and speculative checkpoint slots refuse. Weight FP8 does not imply KV FP8: the launch below first uses BF16 KV.
- One request, TP/PP/DP/CP=1, original GPUModelRunner, synchronous eager execution, no adapters, images, embeddings, speculative decode or MTP. The patch refuses a runner whose original SHA-256 differs from `4cf95efe4ea8caa3110dcd231ff95ceb97e8136f5056e6bdd388300a484c4a81`.
- Block size is read from the actual cache specs, must agree across groups, be a multiple of 16 and be at most 4096. Both endpoints must be whole manager blocks. Cached-token counts alone are not a state lease. Null or recycled block indices refuse. Too large a cached hit cannot be split retrospectively because old GDN endpoints are unavailable.
- The capture's `source_contract` names the inspected tag contract, not a claim about a wheel's build commit. The patch hashes the actual runner file; the hook also checks runtime version and live shapes. CUDA cache binding, kernel page mapping on that host, conv history chronology, state numerical conventions, BF16/FP8 scaling and checkpoint ownership still need the live GPU run.
- The target GGUF hash identifies the receiving file, not the FP8 server weights. Matching layout does not establish matching weights, tokenizer semantics or logits. Operators must bind the same checkpoint/token IDs and record source provenance. Both Q4_K_M and BF16 comparisons are experimental and uncalibrated until measured. No speedup is claimed for the Python conversion.
- Import carries no boundary logits. The existing bench checker consumes one identical probe token after import, then compares full-vocabulary next-token logits with local prefill. It therefore measures N+1 and later, not the already-computed vLLM logits at N. This limit must remain visible in results.

## GPU-host procedure for the orchestrator

Use the existing GPU host's environment (`vllm==0.32.0`, `torch==2.13.0`, `transformers==5.19.0`, `flashinfer-python==0.7.0.post1`). No install or model download is part of this procedure. Run the following blocks in one Bash shell from this branch's repository root. Reserve exclusive use of this vLLM environment while its runner is patched. Start with one aligned block, then two; only then try the depth ladder. Use a fresh RUN_DIR for each capture attempt or depth.

```sh
set -eu
export PYTHONDONTWRITEBYTECODE=1
export RUN_DIR="$PWD/.temp/hybrid-d-r1-261101-gpu"
mkdir -m 700 "$RUN_DIR"
mkdir "$RUN_DIR/tmp"
export TMPDIR="$RUN_DIR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_MODEL=/existing/Qwen3.8-27B-FP8
export BF16_GGUF=/existing/Qwen3.8-27B-BF16.gguf
export Q4_K_M_GGUF=/existing/Qwen3.8-27B-Q4_K_M.gguf
export TOKEN_FILE=/existing/prompt-token-ids.json
export EXPORT_N=1568
export CONTEXT_SIZE=32768
export PROBE_TOKEN=1
export PYTHONPATH="$PWD/tools/hybrid-vllm:$PWD/tools/hybrid-bench${PYTHONPATH:+:$PYTHONPATH}"

wait_health() {
    local pid="$1" url="$2" deadline=$((SECONDS + ${3:-1800}))
    while kill -0 "$pid" 2>/dev/null; do
        if curl --fail --silent --max-time 5 "$url" > /dev/null; then return 0; fi
        if (( SECONDS >= deadline )); then
            printf 'Health timeout: %s\n' "$url" >&2
            return 1
        fi
        sleep 5
    done
    printf 'Server exited before health: PID %s\n' "$pid" >&2
    return 1
}
stop_pid() {
    local pid="$1" status=0
    if kill -0 "$pid" 2>/dev/null; then kill "$pid"; fi
    wait "$pid" || status=$?
    printf 'PID %s wait exit: %s\n' "$pid" "$status"
    case "$status" in 0|143) return 0 ;; *) return "$status" ;; esac
}

(
set -eu
VLLM_BRIDGE_PID=
VLLM_RUNNER=$(python3 - <<'PY'
import importlib.util
from pathlib import Path
print(Path(importlib.util.find_spec('vllm').origin).parent / 'v1/worker/gpu_model_runner.py')
PY
)
python3 - "$VLLM_RUNNER" <<'PY'
import hashlib, sys
from pathlib import Path
from patch_vllm import RUNNER_SHA256
assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() == RUNNER_SHA256, 'require original pinned runner'
PY
cp -p "$VLLM_RUNNER" "$RUN_DIR/gpu_model_runner.py.original"
cleanup_capture() {
    local status=$?
    trap - EXIT
    if [ -n "$VLLM_BRIDGE_PID" ]; then stop_pid "$VLLM_BRIDGE_PID" || status=1; fi
    cp -p "$RUN_DIR/gpu_model_runner.py.original" "$VLLM_RUNNER" || status=1
    cmp "$RUN_DIR/gpu_model_runner.py.original" "$VLLM_RUNNER" || status=1
    unset HYBRID_VLLM_CONFIG
    exit "$status"
}
trap cleanup_capture EXIT
python3 tools/hybrid-vllm/patch_vllm.py "$VLLM_RUNNER"
python3 - <<'PY'
import json, os
from pathlib import Path
from bench_common import model_identity, sha256
root = Path(os.environ['RUN_DIR'])
model = Path(os.environ['BF16_GGUF'])
target = dict(model_identity(model), model_sha256=sha256(model))
tokens = json.loads(Path(os.environ['TOKEN_FILE']).read_text())
if isinstance(tokens, dict):
    tokens = tokens['tokens']
n = int(os.environ['EXPORT_N'])
assert len(tokens) >= n
config = dict(target=target, tokens=tokens[:n], output=str(root/'captures'), rotations=[256,64])
(root/'bridge.json').write_text(json.dumps(config))
(root/'request.json').write_text(json.dumps(dict(model='bridge', prompt=tokens[:n], max_tokens=1, temperature=0, ignore_eos=True)))
PY
export HYBRID_VLLM_CONFIG="$RUN_DIR/bridge.json"
export VLLM_USE_V2_MODEL_RUNNER=0
vllm serve "$HF_MODEL" --served-model-name bridge --host 127.0.0.1 --port 8100 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 --max-num-seqs 1 \
  --dtype bfloat16 --kv-cache-dtype auto --enable-prefix-caching --mamba-cache-mode align \
  --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype float32 \
  --max-model-len "$CONTEXT_SIZE" --max-num-batched-tokens 7840 --enable-chunked-prefill \
  --no-async-scheduling --enforce-eager > "$RUN_DIR/vllm-01.log" 2>&1 &
VLLM_BRIDGE_PID=$!
printf '%s\n' "$VLLM_BRIDGE_PID" > "$RUN_DIR/vllm.pid"
wait_health "$VLLM_BRIDGE_PID" http://127.0.0.1:8100/health 1800
curl --fail --max-time 1800 -H 'Content-Type: application/json' \
  --data-binary "@$RUN_DIR/request.json" http://127.0.0.1:8100/v1/completions > "$RUN_DIR/response.json"
# Release vLLM's GPU reservation before conversion, checker or client startup.
stop_pid "$VLLM_BRIDGE_PID"
VLLM_BRIDGE_PID=
)
```

The capture subshell restores and byte-compares the original runner on success, request refusal, health timeout or server failure. Its hook environment also ends with the subshell. A matching hook is a no-op for `patch_vllm.py`, but the runbook deliberately starts from the original hash. SIGKILL or host failure cannot run a trap: use the saved runner backup to restore that exact path before restarting vLLM. Export refusals remove the private staging directory; only fully validated manifests and tensors are renamed to `capture-N/`. Existing captures refuse replacement.

Inspect the log's resolved block sizes. The observed BF16-KV/FP32-SSM geometry is expected to align at 784; it is not hardcoded. If it differs, choose N and chunk size as multiples of the actual size, regenerate the config in a fresh RUN_DIR, and restart. A request with max_tokens=1 samples one token but never forwards it before capture. Require a final manifest with n1=N; no manifest means no successful export.

Create a target-parameterized converter. It computes a fresh GGUF identity and hash for each leg, never reuses `bridge.json`'s BF16 target for Q4_K_M, and refuses a discontinuous chain or layout mismatch:

```sh
cat > "$RUN_DIR/convert.py" <<'PY'
import json, os, sys
from pathlib import Path
from bench_common import model_identity, sha256
from bridge import convert
root = Path(os.environ['RUN_DIR'])
quant, model = sys.argv[1], Path(sys.argv[2])
assert quant in ('bf16', 'q4_k_m')
target = dict(model_identity(model), model_sha256=sha256(model))
config = json.loads((root/'bridge.json').read_text())
paths = sorted((root/'captures').glob('capture-*/snapshot.json'))
assert paths
out = root/('deltas-' + quant)
out.mkdir(exist_ok=False)
(root/('target-' + quant + '.json')).write_text(json.dumps(target))
deltas, n = [], 0
for snapshot in paths:
    raw = json.loads(snapshot.read_text())
    assert raw['n0'] == n
    dest = out/('%08d.delta' % raw['n1'])
    header = convert(snapshot, target, dest)
    assert header['model_sha256'] == target['model_sha256']
    deltas.append(str(dest))
    n = raw['n1']
assert n == len(config['tokens']) == int(os.environ['EXPORT_N'])
fixture = dict(tokens=config['tokens']+[int(os.environ['PROBE_TOKEN'])], k=64,
               max_ctx=int(os.environ['CONTEXT_SIZE']), model_sha256=target['model_sha256'],
               deltas=deltas, backend='cuda', uncalibrated=True)
(root/('check-' + quant + '.json')).write_text(json.dumps(fixture))
print(quant, target['model_sha256'], deltas[0])
PY
python3 "$RUN_DIR/convert.py" bf16 "$BF16_GGUF" > "$RUN_DIR/convert-bf16.log" 2>&1
python3 "$RUN_DIR/convert.py" q4_k_m "$Q4_K_M_GGUF" > "$RUN_DIR/convert-q4_k_m.log" 2>&1

check_target() {
    local quant="$1" model="$2" first status=0
    first=$(python3 - "$RUN_DIR/check-$quant.json" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())['deltas'][0])
PY
) || return $?
    build/bin/llama-hybrid-bench-check "$model" "$RUN_DIR/check-$quant.json" \
      "$first" "$RUN_DIR/logits-$quant.json" 999 || status=$?
    printf '%s checker exit: %s\n' "$quant" "$status" > "$RUN_DIR/check-$quant.exit"
    case "$status" in 0|5) return 0 ;; *) return "$status" ;; esac
}
check_target bf16 "$BF16_GGUF"
check_target q4_k_m "$Q4_K_M_GGUF"
```

Each conversion leg has its own `target-<quant>.json`, `deltas-<quant>/` and `check-<quant>.json`. Preserve the printed target hashes and the exact checker exits. The checker deliberately exits 5 for `uncalibrated=true`; the wrapper permits evidence collection, and does not label 5 passing. The provisional KL <=0.01 and top-1 >=95% references are context, not calibrated acceptance. The checker compares N+1 after an identical probe, not the vLLM boundary logits at N. It can need five simultaneous contexts: at long depth prefer the scorer on `m3-reference-logits` after verifying that branch's actual CLI, or measure shorter depths first. No unverified scorer command is supplied here.

Import both chains through the actual client HTTP path, one server/model at a time. The subshell stops the recorded client PID on success or failure:

```sh
import_target() (
set -eu
quant="$1"
model="$2"
mkdir "$RUN_DIR/slots-$quant"
LLAMA_BRIDGE_PID=
cleanup_client() {
    local status=$?
    trap - EXIT
    if [ -n "$LLAMA_BRIDGE_PID" ]; then stop_pid "$LLAMA_BRIDGE_PID" || status=1; fi
    exit "$status"
}
trap cleanup_client EXIT
build/bin/llama-server -m "$model" -ngl 999 -c "$CONTEXT_SIZE" -np 1 \
  -ctk q8_0 -ctv q8_0 -fa on --host 127.0.0.1 --port 8101 \
  --slots --slot-save-path "$RUN_DIR/slots-$quant/" > "$RUN_DIR/client-$quant.log" 2>&1 &
LLAMA_BRIDGE_PID=$!
printf '%s\n' "$LLAMA_BRIDGE_PID" > "$RUN_DIR/client-$quant.pid"
wait_health "$LLAMA_BRIDGE_PID" http://127.0.0.1:8101/health 1800
python3 - "$quant" <<'PY'
import json, os, sys, urllib.request
from pathlib import Path
root = Path(os.environ['RUN_DIR'])
quant = sys.argv[1]
fixture = json.loads((root/('check-' + quant + '.json')).read_text())
for path in fixture['deltas']:
    request = urllib.request.Request('http://127.0.0.1:8101/slots/0?action=import_delta',
        data=Path(path).read_bytes(), method='POST',
        headers={'Content-Type':'application/octet-stream','X-Hybrid-Session':'vllm-prototype'})
    with urllib.request.urlopen(request, timeout=1800) as response:
        print(response.read().decode())
with urllib.request.urlopen('http://127.0.0.1:8101/slots/0/info', timeout=30) as response:
    info = json.load(response)
assert info['tokens'] == fixture['tokens'][:-1]
(root/('import-info-' + quant + '.json')).write_text(json.dumps(info))
PY
)
import_target bf16 "$BF16_GGUF"
import_target q4_k_m "$Q4_K_M_GGUF"
```

For the Mac leg, build this base's `llama-server` and `llama-hybrid-bench-check` with Metal; copy each target identity, fixture and matching delta directory, then correct fixture paths/backend to `metal`. Run the same checker and import functions with each receiver GGUF. The receiver must match that leg's target hash. Record source checkpoint provenance, target hashes, tokenizer identity, N, block sizes, dtype/scales, runtime versions, conversion time, peak RSS, import time, KL, top-1 and max logit difference. For FP8 KV, restart with `--kv-cache-dtype fp8_e4m3`; recalculate the block multiple (expected 1568), N and prefill chunk size. Never reuse a capture directory or include the sampled token in the captured prefix.

Depth ladder after the smoke: aligned N=15680, 65072 and 249312 for block 784. Increase CONTEXT_SIZE for the source, both clients and both checker fixtures; use chunks <=7840. Run warm-prefix deltas only after cold export and ownership are verified. Large prefix hits exceeding the wire limit refuse; they need a new capture at intermediate boundaries.

The reviewer measured about 38.5 ms/token across 16 attention layers on agents1, single-threaded Python (`r6x/logs/throughput-01.log`). Extrapolation per target, excluding fixed recurrent work, hashing and I/O: 15680 tokens about 10 min (0.17 h), 65072 about 42 min (0.70 h), 249312 about 160 min (2.67 h). Two target legs approximately double conversion time. These are CPU estimates, not GPU-host measurements. A faster converter would need batched BF16/FP8 decode, vectorized or compiled Hadamard/q8_0 kernels matching f32 rounding and byte order, and bounded layer/chunk parallelism with the same independent byte tests. No faster converter is implemented in this rework.

## Disk and memory

Q8_0 KV is 34816 B/token. Native GDN tensor payload is 156893184 B. At this exact base the full native body, including constant metadata, is `156894772 + 34840*delta_tokens` B: 24 B/token attention-cell metadata includes MRoPE positions and token ID. The brief's 156894784 fixed figure is 12 B larger; the native reader/re-export test verifies the source-derived size, so the writer does not pad it. Envelope JSON plus a 4-byte length and 48-byte checksum trailer are additional. The 512 MiB body ceiling admits at most 10906 tokens; a 7840-token operational chunk is 430040372 B (410.1 MiB).

With BF16 KV, BF16 conv and FP32 SSM, a raw capture costs `65536*delta_tokens + 153944064` B. With FP8 KV the token term halves. Captures deliberately retain raw data for audit and are shared by the two target legs; each leg retains its own deltas. At 15680/65072/249312 tokens and chunks <=7840, there are 2/9/32 captures; total disk is raw captures plus deltas, plus at most two temporary body-sized files during conversion. Delete audited captures only after recording their hashes. Tensor conversion works one layer at a time; its temporary Python arrays and KV decode buffers depend on chunk size, not whole context depth. Budget about 1 GiB extra CPU RAM for the converter at 7840 and about 1 GiB scratch disk, in addition to retained raw captures/deltas.

| Aligned tokens | Captures | BF16 raw captures GiB | Deltas GiB | One target retained GiB | Both targets retained GiB |
|---:|---:|---:|---:|---:|---:|
| 15680 | 2 | 1.244 | 0.801 | 2.045 | 2.846 |
| 65072 | 9 | 5.262 | 3.426 | 8.689 | 12.115 |
| 249312 | 32 | 19.805 | 12.765 | 32.570 | 45.335 |

The GPU source still needs the FP8 model weights, paged KV capacity, recurrent caches, allocator padding, graphs/workspace and vLLM's memory reservation. Logical BF16 KV at 249312 is 16.34 GB; FP8 is 8.17 GB. These are occupied bytes, not the allocator's reservation. A 27B BF16 receiver's weights alone are about 54 GB; Q4_K_M about 16.4 GB (actual files decide). The checker allocates five contexts, so at 249312 it can add roughly 44 GB of q8_0 cache beyond weights and workspace. Do not infer that a 96 GB GPU or a particular Mac can run every checker depth. No memory peak or speed was measured on the GPU.

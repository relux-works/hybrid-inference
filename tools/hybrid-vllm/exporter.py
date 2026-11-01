"""Post-forward hook. All torch and vLLM imports stay inside the GPU path."""

import json
import os
import tempfile
from pathlib import Path

from bridge import ATTENTION, RECURRENT, SOURCE, body_size, require, validate, MAX_BODY


def boundary(n0, computed, scheduled, prompt_length, block):
    n1 = computed + scheduled
    require(0 <= n0 <= computed < n1 <= prompt_length, "request is no longer in prefill")
    require(n0 % block == n1 % block == 0, "partial last block")
    require(body_size(n1 - n0) <= MAX_BODY, "prefill chunk exceeds delta body limit")
    return n1


def check_config(runner, target):
    config = runner.vllm_config
    parallel = config.parallel_config
    require(all(getattr(parallel, key) == 1 for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size")), "unsupported parallelism")
    require(config.speculative_config is None, "speculative decoding is unsupported")
    require(config.scheduler_config.max_num_seqs == 1 and not config.scheduler_config.async_scheduling, "synchronous single-request scheduling required")
    require(config.cache_config.mamba_cache_mode == "align", "Mamba cache mode must be align")
    hf = config.model_config.hf_text_config
    require(config.model_config.hf_config.architectures == ["Qwen3_5ForConditionalGeneration"], "unsupported model class")
    expected = dict(num_hidden_layers=64, num_key_value_heads=4, head_dim=256,
                    linear_conv_kernel_dim=4, linear_num_key_heads=16, linear_num_value_heads=48,
                    linear_key_head_dim=128, linear_value_head_dim=128)
    for key, value in expected.items():
        require(getattr(hf, key, None) == value, "wrong source geometry: " + key)
    layout = target["model_layout"]
    require(hf.hidden_size == layout["embedding_size"] and hf.vocab_size == layout["vocab_size"], "source/target embedding or vocabulary mismatch")
    require(all(x["heads"] == hf.num_attention_heads for x in layout["layers"]), "source/target query heads mismatch")
    require(list(hf.layer_types) == ["full_attention" if i in ATTENTION else "linear_attention" for i in range(64)], "wrong source layer order")


def export_snapshot(runner, scheduler_output, config, n0):
    import torch
    import vllm
    from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    require(vllm.__version__ == "0.32.0", "unsupported vLLM version")
    target = config["target"]
    check_config(runner, target)
    ids = list(scheduler_output.num_scheduled_tokens)
    require(len(ids) == 1 and len(runner.requests) == 1, "bridge requires one live request")
    req = runner.requests[ids[0]]
    require(not req.mm_features and req.prompt_embeds is None and req.lora_request is None and not req.output_token_ids, "text-only base-model prefill required")
    require(req.prompt_token_ids == config["tokens"], "request token journal mismatch")
    groups = runner.kv_cache_config.kv_cache_groups
    require(groups and len({g.kv_cache_spec.block_size for g in groups}) == 1, "cache groups have different block sizes")
    block = groups[0].kv_cache_spec.block_size
    require(type(block) is int and 16 <= block <= 4096 and block % 16 == 0, "wrong block size")
    n1 = boundary(n0, req.num_computed_tokens, scheduler_output.num_scheduled_tokens[ids[0]], len(config["tokens"]), block)
    snapshot = dict(version=1, source_contract=SOURCE, tp_size=1, pp_size=1, cache_mode="align", block_size=block,
                    n0=n0, n1=n1, tokens=req.prompt_token_ids[:n1], model_layout=target["model_layout"],
                    conv_layout=get_conv_state_layout(), rotations=config.get("rotations", [256, 64]), layers=[])
    destination = Path(config["output"]) / ("capture-%08d" % n1)
    require(not destination.exists(), "capture already exists")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="." + destination.name + "-", dir=destination.parent) as partial:
        folder = Path(partial)
        context = runner.vllm_config.compilation_config.static_forward_context

        def save(name, tensor, dtype=None):
            dtype = dtype or str(tensor.dtype).removeprefix("torch.")
            data = tensor.detach().cpu().contiguous()
            with (folder / name).open("xb") as out:
                out.write(data.view(torch.uint8).numpy().tobytes())
            return dict(file=name, shape=list(tensor.shape), dtype=dtype)

        # Device work and state copies finish before the request can release its blocks.
        torch.cuda.synchronize()
        for gid, group in enumerate(groups):
            spec = group.kv_cache_spec
            blocks = req.block_ids[gid]
            for name in group.layer_names:
                i = extract_layer_index(name)
                layer = context[name]
                row = dict(layer=i)
                if i in ATTENTION:
                    require(type(spec) is FullAttentionSpec and spec.sliding_window is None and spec.attention_chunk_size is None, "unsupported attention cache spec")
                    require(spec.num_kv_heads == 4 and spec.head_size == spec.head_size_v == 256, "wrong head geometry")
                    cache = layer.kv_cache
                    require(cache.ndim == 4 and cache.shape[1] == 4 and cache.shape[3] == 512, "unsupported attention tensor layout")
                    kernel = cache.shape[2]
                    require(kernel > 0 and block % kernel == 0, "wrong kernel block size")
                    ratio = block // kernel
                    dtype = str(cache.dtype).removeprefix("torch.")
                    mode = runner.cache_config.cache_dtype
                    if mode in ("fp8", "fp8_e4m3", "fp8_e5m2"):
                        require(dtype == "uint8", "unsupported FP8 storage dtype")
                        dtype = "float8_e5m2" if mode == "fp8_e5m2" else "float8_e4m3fn"
                        scales = [float(layer._k_scale.item()), float(layer._v_scale.item())]
                    else:
                        require(mode == "auto" and dtype == "bfloat16", "unsupported KV dtype")
                        scales = [1.0, 1.0]
                    filename = "%02d-kv.bin" % i
                    with (folder / filename).open("xb") as output:
                        for logical in range(n0 // block, n1 // block):
                            physical = blocks[logical]
                            require(physical > 0 and (physical + 1) * ratio <= cache.shape[0], "invalid attention block ownership")
                            page = cache[physical * ratio:(physical + 1) * ratio].permute(1, 0, 2, 3).reshape(4, block, 512)
                            output.write(page.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                    row.update(kv=dict(file=filename, shape=[(n1 - n0) // block, 4, block, 512], dtype=dtype), scales=scales)
                else:
                    require(i in RECURRENT and type(spec) is MambaSpec and spec.mamba_cache_mode == "align"
                            and spec.num_speculative_blocks == spec.num_prefill_checkpoint_blocks == 0, "unsupported Mamba cache spec")
                    slot = blocks[(n1 - 1) // block]
                    require(len(layer.kv_cache) == 2, "unsupported recurrent state records")
                    conv, state = layer.kv_cache
                    require(slot > 0 and slot < conv.shape[0] and slot < state.shape[0], "invalid recurrent block ownership")
                    row.update(conv=save("%02d-conv.bin" % i, conv[slot]), state=save("%02d-state.bin" % i, state[slot]))
                snapshot["layers"].append(row)
        validate(snapshot, target)
        (folder / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        folder.rename(destination)
    return n1


def after_prefill(runner, scheduler_output):
    path = os.environ.get("HYBRID_VLLM_CONFIG")
    if not path:
        return
    config = json.loads(Path(path).read_text())
    n0 = getattr(runner, "_hybrid_bridge_n0", 0)
    if n0 == len(config["tokens"]):
        return
    runner._hybrid_bridge_n0 = export_snapshot(runner, scheduler_output, config, n0)

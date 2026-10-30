"""Recurrent append regression: valid append works; malformed headers preserve both native sequences.

Run: python3 tools/hybrid/test_recurrent_append.py --library build/bin/libllama.so --malformed
Uses only the prototype model and golden files; an abort is a failing test.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import resource
import struct

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
P = C.c_void_p
class ModelParams(C.Structure):
    _fields_ = [(n, P) for n in ('devices', 'tensor_buft_overrides')] + [(n, C.c_int32) for n in ('n_gpu_layers','split_mode','load_mode','lazy_mode','main_gpu')] + [(n,P) for n in ('tensor_split','progress_callback','progress_callback_user_data','kv_overrides')] + [(n,C.c_bool) for n in ('vocab_only','check_tensors','use_extra_bufts','no_host','no_alloc','load_mtp')]
class ContextParams(C.Structure):
    _fields_ = [(n,C.c_uint32) for n in ('n_ctx','n_batch','n_ubatch','n_seq_max','n_rs_seq','n_outputs_max','n_outputs_max_per_seq')] + [(n,C.c_int32) for n in ('n_threads','n_threads_batch','ctx_type','rope_scaling_type','pooling_type','attention_type','flash_attn_type')] + [(n,C.c_float) for n in ('rope_freq_base','rope_freq_scale','yarn_ext_factor','yarn_attn_factor','yarn_beta_fast','yarn_beta_slow')] + [('yarn_orig_ctx',C.c_uint32),('defrag_thold',C.c_float),('cb_eval',P),('cb_eval_user_data',P),('type_k',C.c_int),('type_v',C.c_int),('moe_cache_size',C.c_size_t),('abort_callback',P),('abort_callback_data',P)] + [(n,C.c_bool) for n in ('embeddings','offload_kqv','no_perf','op_offload','swa_full','kv_unified')] + [('samplers',P),('n_samplers',C.c_size_t),('ctx_other',P)]
assert C.sizeof(ModelParams) == 80 and C.sizeof(ContextParams) == 168
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--library', required=True)
parser.add_argument('--malformed', action='store_true')
args = parser.parse_args()
lib=C.CDLL(str(Path(args.library).resolve()))
lib.llama_model_default_params.restype=ModelParams
lib.llama_context_default_params.restype=ContextParams
lib.llama_model_load_from_file.argtypes=[C.c_char_p,ModelParams];lib.llama_model_load_from_file.restype=P
lib.llama_init_from_model.argtypes=[P,ContextParams];lib.llama_init_from_model.restype=P
lib.llama_state_seq_set_data.argtypes=[P,P,C.c_size_t,C.c_int32];lib.llama_state_seq_set_data.restype=C.c_size_t
lib.llama_state_seq_get_size.argtypes=[P,C.c_int32];lib.llama_state_seq_get_size.restype=C.c_size_t
lib.llama_state_seq_get_data.argtypes=[P,P,C.c_size_t,C.c_int32];lib.llama_state_seq_get_data.restype=C.c_size_t
lib.llama_free.argtypes=[P];lib.llama_model_free.argtypes=[P]
append=getattr(lib,'_Z31llama_state_seq_set_data_appendP13llama_contextPKhmiii')
append.argtypes=[P,P,C.c_size_t,C.c_int32,C.c_int32,C.c_int32];append.restype=C.c_size_t
lib.llama_backend_init()
mp=lib.llama_model_default_params();mp.n_gpu_layers=0
model=lib.llama_model_load_from_file(b'/home/agent/Werkstatt/hybrid-models/Qwen3.5-0.8B-Q4_0.gguf',mp)
assert model
cp=lib.llama_context_default_params();cp.n_ctx=1024;cp.n_batch=256;cp.n_ubatch=256;cp.n_seq_max=2;cp.n_threads=2;cp.n_threads_batch=2;cp.type_k=8;cp.type_v=8;cp.flash_attn_type=1;cp.offload_kqv=False;cp.op_offload=False;cp.kv_unified=True
ctx=lib.llama_init_from_model(model,cp);assert ctx
root=Path('/home/agent/Werkstatt/hybrid-models/golden')
prefix=(root/'prompt-512.prefix.state').read_bytes()
wire=(root/'prompt-512.delta').read_bytes();hsize=struct.unpack_from('<I',wire)[0];h=json.loads(wire[4:4+hsize]);raw=wire[4+hsize:-48]
for seq in (0,1):
    assert lib.llama_state_seq_set_data(ctx,prefix,len(prefix),seq)==len(prefix)
def state(seq):
    size=lib.llama_state_seq_get_size(ctx,seq);out=C.create_string_buffer(size)
    assert lib.llama_state_seq_get_data(ctx,out,size,seq)==size
    return out.raw
before=state(0);neighbor=state(1)
assert append(ctx,raw,len(raw),0,h['n0'],h['n1'])==len(raw)
assert state(1)==neighbor
print('PASS positive append on unified two-sequence context; neighbor unchanged',flush=True)
assert lib.llama_state_seq_set_data(ctx,prefix,len(prefix),0)==len(prefix)
assert state(0)==before
if args.malformed:
    cells = struct.unpack_from("<I", raw, 12)[0]
    at = 16 + 24 * cells
    vtrans, layers, _, _ = struct.unpack_from("<4I", raw, at)
    assert vtrans == 0
    at += 16
    for _ in range(2 * layers):
        row = struct.unpack_from("<Q", raw, at + 4)[0]
        at += 12 + row * cells
    count, pos, nseq = struct.unpack_from("<IiI", raw, at)
    assert count == 1 and nseq == 0

    def reject(name, data):
        # A failed import must return normally and preserve KV and recurrent bytes in both sequences.
        assert append(ctx, data, len(data), 0, h["n0"], h["n1"]) == 0, name
        assert state(0) == before and state(1) == neighbor, name + ": native state changed"
        print("PASS " + name + ": rejected; both sequences byte-identical", flush=True)

    def changed(offset, fmt, value):
        data = bytearray(raw)
        struct.pack_into(fmt, data, offset, value)
        return bytes(data)

    def test_recurrent_cell_count_two_rejected_without_abort():
        bad = raw[:at] + struct.pack("<I", 2) + raw[at + 4:at + 12] * 2 + raw[at + 12:]
        reject("test_recurrent_cell_count_two_rejected_without_abort", bad)

    test_recurrent_cell_count_two_rejected_without_abort()
    for name, offset, fmt, value in [
        ("recurrent_count_overflow", at, "<I", 2**32 - 1),
        ("recurrent_count_zero", at, "<I", 0),
        ("recurrent_sequence_ids", at + 8, "<I", 1),
        ("recurrent_sequence_ids_overflow", at + 8, "<I", 2**32 - 1),
        ("recurrent_negative_position", at + 4, "<i", -1),
        ("recurrent_transposed", at + 12, "<I", 1),
        ("recurrent_layer_count", at + 16, "<I", 2**32 - 1),
        ("recurrent_tensor_type", at + 20, "<i", -1),
        ("recurrent_tensor_row_size", at + 24, "<Q", 2**64 - 1),
        ("source_sequence_id", 4, "<i", -1),
    ]:
        reject(name, changed(offset, fmt, value))
    for name, data in [
        ("recurrent_truncated_metadata", raw[:at + 10]),
        ("recurrent_truncated_tensor_header", raw[:at + 25]),
        ("recurrent_truncated_tensor", raw[:-1]),
        ("recurrent_trailing_bytes", raw + b"x"),
    ]:
        reject(name, data)
    # The same untouched prefix still accepts the valid record after every negative probe.
    assert append(ctx, raw, len(raw), 0, h["n0"], h["n1"]) == len(raw)
    assert state(1) == neighbor
    print("PASS positive append after malformed records; neighbor byte-identical", flush=True)
lib.llama_free(ctx)
lib.llama_model_free(model)

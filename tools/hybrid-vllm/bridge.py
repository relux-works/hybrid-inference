"""Strict native codec for the vLLM 0.32.0 single-rank Qwen35 bridge."""

import argparse
import array
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

ATTENTION = tuple(range(3, 64, 4))
RECURRENT = tuple(i for i in range(64) if i not in ATTENTION)
MAX_BODY = 512 * 1024**2
DTYPES = {"bfloat16": 2, "float32": 4, "float8_e4m3fn": 1, "float8_e5m2": 1}
SOURCE = "5df63a7203cdb9328d824a27b682e77c7b9928be"


class Refusal(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise Refusal(message)


def fingerprint(layout):
    return hashlib.sha256(json.dumps(layout, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def token_hash(tokens):
    return hashlib.sha256(struct.pack("<" + "i" * len(tokens), *tokens)).hexdigest()


def validate_layout(layout):
    require(layout.get("architecture") == "qwen35" and layout.get("version") == 1, "unsupported architecture")
    require(layout.get("layer_count") == 64 and len(layout.get("layers", [])) == 64, "wrong layer count")
    require(layout.get("recurrent_layout") == dict(conv=4, inner=6144, state=128, rank=48, groups=16), "wrong recurrent geometry")
    for i, layer in enumerate(layout["layers"]):
        require(layer.get("recurrent") is (i in RECURRENT), "wrong layer order")
        require(layer.get("kv_heads") == 4 and layer.get("key_dim") == layer.get("value_dim") == 256
                and type(layer.get("heads")) is int and layer["heads"] > 0 and layer["heads"] % 4 == 0, "wrong head geometry")
    require(type(layout.get("embedding_size")) is int and layout["embedding_size"] > 0, "wrong embedding geometry")
    require(type(layout.get("vocab_size")) is int and 0 < layout["vocab_size"] <= 2_000_000, "wrong vocabulary geometry")
    digest(layout.get("vocab_sha256"), "vocabulary hash")


def digest(value, name):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "invalid " + name)


def validate(snapshot, target):
    require(snapshot.get("version") == 1 and snapshot.get("source_contract") == SOURCE, "unsupported capture version")
    validate_layout(target["model_layout"])
    require(target["layout_fingerprint"] == fingerprint(target["model_layout"]), "layout fingerprint mismatch")
    digest(target.get("model_sha256"), "target model hash")
    require(snapshot.get("model_layout") == target["model_layout"], "source/target layout mismatch")
    require(snapshot.get("tp_size") == snapshot.get("pp_size") == 1 and snapshot.get("cache_mode") == "align", "unsupported execution geometry")
    block = snapshot.get("block_size")
    require(type(block) is int and 16 <= block <= 4096 and block % 16 == 0, "wrong block size")
    n0, n1 = snapshot.get("n0"), snapshot.get("n1")
    require(type(n0) is int and type(n1) is int and 0 <= n0 < n1 <= 1_048_576, "invalid token boundary")
    require(n1 % block == 0, "partial last block")
    tokens = snapshot.get("tokens")
    require(isinstance(tokens, list) and len(tokens) == n1 and all(type(t) is int and 0 <= t < target["model_layout"]["vocab_size"] for t in tokens), "invalid token journal")
    layers = snapshot.get("layers", [])
    require(len(layers) == 64 and sorted(x["layer"] for x in layers) == list(range(64)), "wrong layer count")
    require(snapshot.get("conv_layout") in ("SD", "DS"), "unsupported conv layout")
    require(snapshot.get("rotations") in ([256, 64], [0, 0]), "unsupported KV rotation")
    require(body_size(n1 - n0) <= MAX_BODY, "delta exceeds body limit; capture smaller prefill chunks")
    for layer in layers:
        if layer["layer"] in ATTENTION:
            require(layer["kv"]["shape"] == [(n1 - n0 + n0 % block) // block, 4, block, 512], "wrong block or head geometry")
            require(layer["kv"]["dtype"] in ("bfloat16", "float8_e4m3fn", "float8_e5m2"), "unsupported KV dtype")
            require(len(layer.get("scales", [])) == 2 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in layer["scales"]), "invalid FP8 scales")
            require(layer["kv"]["dtype"] != "bfloat16" or layer["scales"] == [1.0, 1.0], "BF16 scales must be one")
        else:
            conv = [3, 10240] if snapshot["conv_layout"] == "SD" else [10240, 3]
            require(layer["conv"]["shape"] == conv and layer["state"]["shape"] == [48, 128, 128], "wrong recurrent geometry")
            require(layer["conv"]["dtype"] in ("bfloat16", "float32") and layer["state"]["dtype"] in ("bfloat16", "float32"), "unsupported state dtype")


def body_size(count, fixed_state=True):
    return 416 + (156_894_356 if fixed_state else 0) + count * (34_816 + 24)


def chunk_transfer(transfer, n0, n1, session_id):
    if transfer is None:
        transfer = dict(transfer_id=session_id, transfer_n0=n0, transfer_n1=n1, chunk_index=0, chunk_count=1)
    require(set(transfer) == {"transfer_id", "transfer_n0", "transfer_n1", "chunk_index", "chunk_count"}, "incomplete chunk metadata")
    require(isinstance(transfer["transfer_id"], str) and 0 < len(transfer["transfer_id"].encode("utf-8")) <= 128, "invalid transfer id")
    for key in ("transfer_n0", "transfer_n1", "chunk_index", "chunk_count"):
        require(type(transfer[key]) is int and 0 <= transfer[key] <= 2**31 - 1, "invalid chunk metadata")
    begin, end = transfer["transfer_n0"], transfer["transfer_n1"]
    index, count = transfer["chunk_index"], transfer["chunk_count"]
    require(begin <= n0 < n1 <= end and 0 < count <= end - begin and index < count
            and (index != 0 or begin == n0) and (end == n1 if index + 1 == count else end > n1), "invalid chunk boundary")
    return dict(transfer, fixed_state=index + 1 == count)


def tensor_bytes(root, spec):
    require(spec["dtype"] in DTYPES, "unsupported tensor dtype")
    path = Path(spec["file"])
    require(not path.is_absolute() and len(path.parts) == 1 and path.name not in ("", ".", ".."), "unsafe tensor path")
    path = root / path
    require(not path.is_symlink(), "unsafe tensor symlink")
    size = math.prod(spec["shape"]) * DTYPES[spec["dtype"]]
    require(path.stat().st_size == size, "truncated or trailing tensor bytes")
    return path.read_bytes()


def floats(data, dtype):
    if dtype == "float32":
        out = array.array("f")
        out.frombytes(data)
        if sys.byteorder != "little":
            out.byteswap()
    elif dtype == "bfloat16":
        out = array.array("f", (struct.unpack("<f", b"\0\0" + data[i:i + 2])[0] for i in range(0, len(data), 2)))
    else:
        require(dtype in ("float8_e4m3fn", "float8_e5m2"), "unsupported tensor dtype")
        mantissa, bias = (3, 7) if dtype == "float8_e4m3fn" else (2, 15)
        table = []
        for byte in range(256):
            exponent, fraction = (byte & 127) >> mantissa, byte & ((1 << mantissa) - 1)
            value = math.ldexp(fraction / (1 << mantissa), 1 - bias) if exponent == 0 else math.ldexp(1 + fraction / (1 << mantissa), exponent - bias)
            if (dtype == "float8_e4m3fn" and byte & 127 == 127) or (dtype == "float8_e5m2" and exponent == 31):
                value = math.nan
            table.append(-value if byte & 128 else value)
        out = array.array("f", (table[b] for b in data))
    require(all(math.isfinite(x) for x in out), "nonfinite tensor value")
    return out


def f32_bytes(values):
    out = array.array("f", values)
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def rotate(values, size):
    if not size:
        return values
    out = array.array("f", values)
    for start in range(0, len(out), size):
        width = 1
        while width < size:
            for at in range(start, start + size, 2 * width):
                for j in range(width):
                    a, b = out[at + j], out[at + j + width]
                    out[at + j], out[at + j + width] = a + b, a - b
            width *= 2
        for j in range(start, start + size):
            out[j] /= math.sqrt(size)
    return out


def q8_0(values):
    require(len(values) % 32 == 0 and all(math.isfinite(x) for x in values), "invalid q8_0 input")
    out = bytearray()
    for i in range(0, len(values), 32):
        row = values[i:i + 32]
        scale = ctypes.c_float(max(abs(x) for x in row) / 127).value
        require(scale <= 65504, "q8_0 scale overflow")
        inverse = ctypes.c_float(1 / scale).value if scale else 0
        out.extend(struct.pack("<e", scale))
        quant = [math.copysign(math.floor(abs(ctypes.c_float(x * inverse).value) + .5), x) for x in row]
        out.extend(struct.pack("32b", *(int(x) for x in quant)))
    return bytes(out)


def kv_rows(root, layer, block, value, rotation):
    spec = layer["kv"]
    data = floats(tensor_bytes(root, spec), spec["dtype"])
    scale = layer["scales"][int(value)]
    for page in range(spec["shape"][0]):
        for token in range(block):
            row = []
            for head in range(4):
                at = ((page * 4 + head) * block + token) * 512 + (256 if value else 0)
                row.extend(ctypes.c_float(x * scale).value for x in data[at:at + 256])
            yield q8_0(rotate(row, rotation))


def recurrent_row(root, layer, state, layout):
    spec = layer["state" if state else "conv"]
    raw = tensor_bytes(root, spec)
    data = floats(raw, spec["dtype"])
    if state:
        heads = [(h % 16) * 3 + h // 16 for h in range(48)]
        return b"".join(f32_bytes(data[h * 16384:(h + 1) * 16384]) for h in heads)
    # GGUF reorders only V channels from grouped heads to tiled heads.
    channels = list(range(4096)) + [4096 + ((h % 16) * 3 + h // 16) * 128 + d for h in range(48) for d in range(128)]
    return f32_bytes(data[t * 10240 + c] if layout == "SD" else data[c * 3 + t] for c in channels for t in range(3))


def write_body(output, snapshot, root, codec=None, fixed_state=True):
    n0, n1 = snapshot["n0"], snapshot["n1"]
    def pack(fmt, *args):
        output.write(struct.pack("<" + fmt, *args))
    pack("IiII", 0xAF143CD8, 0, 1, n1 - n0)
    for pos in range(n0, n1):
        pack("iIiiii", pos, 1, pos, pos, snapshot["tokens"][pos], 0)
    pack("IIII", 0, 16, *snapshot["rotations"])
    layers = {x["layer"]: x for x in snapshot["layers"]}
    for value in (False, True):
        for i in ATTENTION:
            pack("iQ", 8, 1088)
            rows = (codec.kv_rows if codec else kv_rows)(root, layers[i], snapshot["block_size"], value, snapshot["rotations"][int(value)])
            skip = n0 % snapshot["block_size"] * 1088
            for row in rows:
                output.write(row[skip:])
                skip = max(0, skip - len(row))
    if not fixed_state:
        return
    pack("IiIII", 1, n1 - 1, 0, 0, 64)
    for state in (False, True):
        for i in RECURRENT:
            row = (codec.recurrent_row if codec else recurrent_row)(root, layers[i], state, snapshot["conv_layout"])
            pack("iQ", 0, len(row))
            output.write(row)


def xxh128_file(source):
    class Hash(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint64), ("high", ctypes.c_uint64)]
    library = ctypes.CDLL(ctypes.util.find_library("xxhash") or "libxxhash.so.0")
    library.XXH3_createState.restype = ctypes.c_void_p
    library.XXH3_128bits_reset.argtypes = [ctypes.c_void_p]
    library.XXH3_128bits_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.XXH3_128bits_digest.argtypes = [ctypes.c_void_p]
    library.XXH3_128bits_digest.restype = Hash
    library.XXH3_freeState.argtypes = [ctypes.c_void_p]
    state = library.XXH3_createState()
    require(bool(state), "cannot allocate xxhash state")
    sha = hashlib.sha256()
    try:
        require(library.XXH3_128bits_reset(state) == 0, "xxhash reset failed")
        for chunk in iter(lambda: source.read(1024**2), b""):
            require(library.XXH3_128bits_update(state, chunk, len(chunk)) == 0, "xxhash update failed")
            sha.update(chunk)
        value = library.XXH3_128bits_digest(state)
        return struct.pack(">QQ", value.high, value.low) + sha.digest()
    finally:
        library.XXH3_freeState(state)


def convert(snapshot_path, target, destination, backend="numpy", session_id="vllm-prototype", transfer=None):
    snapshot_path, destination = Path(snapshot_path), Path(destination)
    require(snapshot_path.stat().st_size <= 16 * 1024**2, "capture manifest too large")
    snapshot = json.loads(snapshot_path.read_text())
    validate(snapshot, target)
    n0, n1 = snapshot["n0"], snapshot["n1"]
    transfer = chunk_transfer(transfer, n0, n1, session_id)
    fixed_state = transfer["fixed_state"]
    header = dict(magic="HYBD", version=2, body_format="llama-native-chunks-v3", kv_type="q8_0", state_dtype="f32",
                  model_sha256=target["model_sha256"], layout_fingerprint=target["layout_fingerprint"],
                  session_id=session_id, n0=n0, n1=n1, tokens=snapshot["tokens"][n0:],
                  prefix_hash_n1=token_hash(snapshot["tokens"]),
                  sections=[dict(name="llama_seq_state", offset=0, bytes=body_size(n1 - n0, fixed_state))], **transfer)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    require(len(encoded) <= min(1024**2, 4096 + 11 * (n1 - n0)), "delta header exceeds range budget")
    # Publish only a fully converted, checksummed file; failures preserve an existing destination.
    from fast_codec import codec_for
    codec = codec_for(backend)
    with tempfile.TemporaryFile(dir=destination.parent) as body:
        write_body(body, snapshot, snapshot_path.parent, codec, fixed_state)
        require(body.tell() == body_size(n1 - n0, fixed_state), "native body size mismatch")
        body.seek(0)
        checksums = xxh128_file(body)
        fd, temp = tempfile.mkstemp(prefix=".vllm-delta-", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(struct.pack("<I", len(encoded)) + encoded)
                body.seek(0)
                for chunk in iter(lambda: body.read(1024**2), b""):
                    output.write(chunk)
                output.write(checksums)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, destination)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    return header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("target", help="target identity JSON from bench_common.model_identity, plus model_sha256")
    parser.add_argument("output")
    parser.add_argument("--backend", choices=("numpy", "torch", "auto", "reference"), default="numpy")
    args = parser.parse_args()
    convert(args.snapshot, json.loads(Path(args.target).read_text()), args.output, args.backend)


if __name__ == "__main__":
    main()

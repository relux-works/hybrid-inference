"""Explicit GPU-host byte gate; never part of CPU discovery."""

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

import bridge
from fast_codec import TorchCodec
from test_fast_codec import synthetic


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def check():
    bridge.require(torch.cuda.is_available(), "CUDA required for this explicit GPU gate")
    codec = TorchCodec()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Actual CUDA butterflies and q8_0 must agree with scalar FP8/BF16 pages, including non-unit scales.
        for dtype in ("bfloat16", "float8_e4m3fn", "float8_e5m2"):
            snapshot, target = synthetic(root, 32, dtype)
            for value in (False, True):
                for rotation in (0, 64, 256):
                    layer = snapshot["layers"][3]
                    expected = b"".join(bridge.kv_rows(root, layer, 16, value, rotation))
                    actual = b"".join(codec.kv_rows(root, layer, 16, value, rotation))
                    bridge.require(actual == expected, "CUDA KV bytes differ: " + str((dtype, value, rotation)))
        # Tie/zero controls and a complete envelope also bind GPU arithmetic to native serialization/checksums.
        ties = np.array([127, -127, .5, -.5, 1.5, -1.5, 0, -0.] * 4, dtype=np.float32)
        bridge.require(codec.quantize(ties, 0) == bridge.q8_0(ties.tolist()), "CUDA q8_0 half ties differ")
        snapshot_path = root / "snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot))
        for backend in ("reference", "torch"):
            bridge.convert(snapshot_path, target, root / (backend + ".delta"), backend)
        bridge.require(sha256(root / "reference.delta") == sha256(root / "torch.delta"), "CUDA full delta differs")
    sources = {name: sha256(Path(__file__).with_name(name)) for name in
               ("bridge.py", "fast_codec.py", "cuda_codec_check.py", "test_fast_codec.py", "test_bridge.py")}
    return dict(status="passed", torch=torch.__version__, cuda=torch.version.cuda,
                device_index=torch.cuda.current_device(), device=torch.cuda.get_device_name(),
                device_properties=str(torch.cuda.get_device_properties(torch.cuda.current_device())), sources=sources,
                claim="scalar/CUDA synthetic page and complete delta byte equality")


if __name__ == "__main__":
    print(json.dumps(check()))

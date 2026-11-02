"""CPU byte equivalence against the unchanged scalar converter."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import bridge
from fast_codec import NumpyCodec, TorchCodec
from test_bridge import fixture


def synthetic(root, count=16, dtype="float8_e4m3fn", seed=7):
    snapshot, target = fixture()
    snapshot.update(n1=count, tokens=[i % 1000 for i in range(count)])
    rng = np.random.default_rng(seed)
    if dtype == "bfloat16":
        raw = (rng.standard_normal((count // 16, 4, 16, 512)).astype(np.float32).view(np.uint32) >> 16).astype("<u2")
    else:
        raw = rng.integers(0, 120, (count // 16, 4, 16, 512), dtype=np.uint8)
        raw |= rng.integers(0, 2, raw.shape, dtype=np.uint8) << 7
    (root / "kv.bin").write_bytes(raw.tobytes())
    for name, shape in (("conv", (3, 10240)), ("state", (48, 128, 128))):
        (root / (name + ".bin")).write_bytes(rng.standard_normal(shape).astype("<f4").tobytes())
    for layer in snapshot["layers"]:
        if layer["layer"] in bridge.ATTENTION:
            layer.update(kv=dict(file="kv.bin", shape=list(raw.shape), dtype=dtype), scales=[1., 1.] if dtype == "bfloat16" else [.137123456789, 1.23456789])
        else:
            layer["conv"]["file"], layer["state"]["file"] = "conv.bin", "state.bin"
    return snapshot, target


class FastCodecTests(unittest.TestCase):
    # Random pages cover FP8/BF16, two block counts, separate non-f32 scales and every rotation pair.
    def test_random_kv_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, codec = Path(tmp), NumpyCodec()
            for dtype in ("bfloat16", "float8_e4m3fn", "float8_e5m2"):
                snapshot, _ = synthetic(root, 32, dtype)
                layer = snapshot["layers"][3]
                for rotation in (0, 64, 256):
                    for value in (False, True):
                        expected = b"".join(bridge.kv_rows(root, layer, 16, value, rotation))
                        self.assertEqual(b"".join(codec.kv_rows(root, layer, 16, value, rotation)), expected, (dtype, rotation, value))

    # Full envelopes, native metadata, recurrent permutations and both checksums equal the scalar path.
    def test_full_delta_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, target = synthetic(root)
            path = root / "snapshot.json"
            path.write_text(json.dumps(snapshot))
            for backend in ("reference", "numpy"):
                bridge.convert(path, target, root / (backend + ".delta"), backend)
            def sha(path):
                digest = hashlib.sha256()
                with path.open("rb") as src:
                    for chunk in iter(lambda: src.read(1024**2), b""):
                        digest.update(chunk)
                return digest.digest()
            self.assertEqual(sha(root / "reference.delta"), sha(root / "numpy.delta"))
            codec = NumpyCodec()
            for dtype in ("float32", "bfloat16"):
                for state in (False, True):
                    layer = copy.deepcopy(snapshot["layers"][0])
                    spec = layer["state" if state else "conv"]
                    raw = np.fromfile(root / spec["file"], dtype="<f4").reshape(spec["shape"])
                    spec.update(file="recurrent.bin", dtype=dtype)
                    for layout in ("SD", "DS"):
                        data = raw.T if not state and layout == "DS" else raw
                        spec["shape"] = list(data.shape)
                        data = data if dtype == "float32" else (data.view(np.uint32) >> 16).astype("<u2")
                        (root / spec["file"]).write_bytes(data.tobytes())
                        self.assertEqual(codec.recurrent_row(root, layer, state, layout), bridge.recurrent_row(root, layer, state, layout))

    # Half ties and quantization overflow/nonfinite refusals preserve the reference's accepted bytes.
    def test_rounding_and_refusals(self):
        codec = NumpyCodec()
        data = np.array([127, -127, .5, -.5, 1.5, -1.5, 0, -0.] * 4, dtype=np.float32)
        self.assertEqual(codec.quantize(data, 0), bridge.q8_0(data.tolist()))
        self.assertEqual(codec.quantize(np.zeros(32), 0), bytes(34))
        for value, reason in ((np.nan, "invalid q8_0"), (np.inf, "invalid q8_0"), (1e10, "scale overflow")):
            with self.assertRaisesRegex(bridge.Refusal, reason):
                codec.quantize(np.full(32, value), 0)
        for dtype, value in (("float8_e4m3fn", 127), ("float8_e5m2", 124)):
            with self.assertRaisesRegex(bridge.Refusal, "nonfinite tensor"):
                codec.decode(np.array([value], dtype=np.uint8), dtype)

    # The CUDA arithmetic can also be checked on torch CPU without a vLLM installation.
    def test_torch_cpu_if_present(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch absent; CUDA path remains unqualified")
        torch.set_num_threads(1)
        codec = TorchCodec("cpu")
        for rotation in (0, 64, 256):
            data = np.random.default_rng(44).standard_normal((4, 1024)).astype(np.float32)
            expected = bridge.q8_0(bridge.rotate(data.reshape(-1).tolist(), rotation))
            self.assertEqual(codec.quantize(data, rotation), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)

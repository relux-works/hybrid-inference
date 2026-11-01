import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace as Namespace
import unittest
from unittest import mock

import bridge
import exporter
import patch_vllm


def fixture():
    layout = dict(version=1, architecture="qwen35", layer_count=64, embedding_size=5120,
                  layers=[dict(heads=24, kv_heads=4, key_dim=256, value_dim=256, recurrent=i in bridge.RECURRENT) for i in range(64)],
                  recurrent_layout=dict(conv=4, inner=6144, state=128, rank=48, groups=16), vocab_size=1000, vocab_sha256="a" * 64)
    target = dict(model_layout=layout, layout_fingerprint=bridge.fingerprint(layout), model_sha256="b" * 64)
    tensor = lambda file, shape, dtype="float32": dict(file=file, shape=shape, dtype=dtype)
    layers = [dict(layer=i, kv=tensor("kv.bin", [1, 4, 16, 512], "bfloat16"), scales=[1.0, 1.0]) if i in bridge.ATTENTION else
              dict(layer=i, conv=tensor("conv.bin", [3, 10240]), state=tensor("state.bin", [48, 128, 128])) for i in range(64)]
    snapshot = dict(version=1, source_contract=bridge.SOURCE, tp_size=1, pp_size=1, cache_mode="align", block_size=16,
                    n0=0, n1=16, tokens=list(range(16)), model_layout=layout, conv_layout="SD", rotations=[256, 64], layers=layers)
    return snapshot, target


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.snapshot, self.target = fixture()

    # Imports and boundary selection work on a CPU without vLLM, torch or NumPy.
    def test_lazy_imports_and_boundary(self):
        self.assertNotIn("vllm", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        self.assertEqual(exporter.boundary(0, 0, 784, 1568, 784), 784)
        self.assertEqual(exporter.boundary(0, 784, 784, 1568, 784), 1568)
        with self.assertRaisesRegex(bridge.Refusal, "no longer in prefill"):
            exporter.boundary(0, 784, 785, 1568, 784)

    # Live configuration admits the specified model and refuses parallel, async, speculative and geometric variants.
    def test_source_config(self):
        parallel = Namespace(**{key: 1 for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size")})
        hf = Namespace(num_hidden_layers=64, num_key_value_heads=4, head_dim=256, linear_conv_kernel_dim=4,
                       linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128, linear_value_head_dim=128,
                       hidden_size=5120, vocab_size=1000, num_attention_heads=24,
                       layer_types=["full_attention" if i in bridge.ATTENTION else "linear_attention" for i in range(64)])
        config = Namespace(parallel_config=parallel, speculative_config=None,
                           scheduler_config=Namespace(max_num_seqs=1, async_scheduling=False),
                           cache_config=Namespace(mamba_cache_mode="align"),
                           model_config=Namespace(hf_text_config=hf, hf_config=Namespace(architectures=["Qwen3_5ForConditionalGeneration"])))
        runner = Namespace(vllm_config=config)
        exporter.check_config(runner, self.target)
        for obj, key, value, reason in [(parallel, "tensor_parallel_size", 2, "parallelism"),
                                        (config.scheduler_config, "async_scheduling", True, "single-request scheduling"),
                                        (config, "speculative_config", {}, "speculative decoding"),
                                        (hf, "linear_num_value_heads", 16, "source geometry"),
                                        (hf, "layer_types", ["full_attention"] * 64, "source layer order")]:
            with self.subTest(key=key), mock.patch.object(obj, key, value), self.assertRaisesRegex(bridge.Refusal, reason):
                exporter.check_config(runner, self.target)

    # Missing/duplicate layers cannot pass with otherwise valid geometry.
    def test_wrong_layer_count(self):
        bridge.validate(self.snapshot, self.target)
        self.snapshot["layers"].pop()
        with self.assertRaisesRegex(bridge.Refusal, "wrong layer count"):
            bridge.validate(self.snapshot, self.target)

    # Unaligned manager sizes and mismatched logical page shapes have distinct guards.
    def test_wrong_block_size(self):
        self.snapshot["block_size"] = 8
        for layer in self.snapshot["layers"]:
            if layer["layer"] in bridge.ATTENTION:
                layer["kv"]["shape"] = [2, 4, 8, 512]
        with self.assertRaisesRegex(bridge.Refusal, "wrong block size"):
            bridge.validate(self.snapshot, self.target)

    # Head geometry is checked before fingerprint or tensor conversion.
    def test_wrong_head_geometry(self):
        self.target["model_layout"]["layers"][3]["key_dim"] = 128
        self.target["layout_fingerprint"] = bridge.fingerprint(self.target["model_layout"])
        with self.assertRaisesRegex(bridge.Refusal, "wrong head geometry"):
            bridge.validate(self.snapshot, self.target)

    # A recurrent endpoint inside the final page cannot be treated as a whole-block export.
    def test_partial_last_block(self):
        self.snapshot["n1"] = 15
        self.snapshot["tokens"].pop()
        with self.assertRaisesRegex(bridge.Refusal, "partial last block"):
            bridge.validate(self.snapshot, self.target)
        with self.assertRaisesRegex(bridge.Refusal, "partial last block"):
            exporter.boundary(0, 0, 15, 15, 16)

    # An oversize capture is refused before allocating or serializing a body.
    def test_body_budget(self):
        with self.assertRaisesRegex(bridge.Refusal, "body limit"):
            exporter.boundary(0, 0, 15680, 15680, 784)
        self.assertLess(bridge.body_size(7840), 512 * 1024**2)

    # FP16/int8 and unrecognized state dtypes are refused, beside valid BF16/f32 controls.
    def test_unsupported_dtype(self):
        bridge.validate(self.snapshot, self.target)
        for field, i, dtype, reason in [("kv", 3, "float16", "unsupported KV dtype"), ("conv", 0, "int8", "unsupported state dtype"), ("state", 0, "float16", "unsupported state dtype")]:
            bad = copy.deepcopy(self.snapshot)
            bad["layers"][i][field]["dtype"] = dtype
            with self.subTest(field=field), self.assertRaisesRegex(bridge.Refusal, reason):
                bridge.validate(bad, self.target)

    # Source/target layout identity, cache mode, token bounds and page shape cannot be silently substituted.
    def test_identity_and_layout_refusals(self):
        for change, reason in [(lambda s, t: t.update(layout_fingerprint="c" * 64), "fingerprint mismatch"),
                               (lambda s, t: s.update(cache_mode="none"), "execution geometry"),
                               (lambda s, t: s["tokens"].__setitem__(0, -1), "token journal"),
                               (lambda s, t: s["layers"][3]["kv"].update(shape=[2, 4, 16, 512]), "block or head"),
                               (lambda s, t: s["layers"][0]["state"].update(shape=[16, 128, 128]), "recurrent geometry"),
                               (lambda s, t: s["layers"][3].update(scales=[math.nan, 1]), "FP8 scales")]:
            s, t = copy.deepcopy((self.snapshot, self.target))
            change(s, t)
            with self.subTest(reason=reason), self.assertRaisesRegex(bridge.Refusal, reason):
                bridge.validate(s, t)

    # FP8 decoding uses E4M3FN/E5M2 semantics, including subnormals, signed values and nonfinite refusals.
    def test_fp8_and_bf16_values(self):
        self.assertEqual(list(bridge.floats(bytes([0, 56, 184, 126, 1]), "float8_e4m3fn")), [0, 1, -1, 448, 2**-9])
        self.assertEqual(list(bridge.floats(bytes([60, 188, 1]), "float8_e5m2")), [1, -1, 2**-16])
        self.assertEqual(list(bridge.floats(b"\x80\x3f\x00\xc0", "bfloat16")), [1, -2])
        for dtype, value in [("float8_e4m3fn", 127), ("float8_e5m2", 124)]:
            with self.assertRaisesRegex(bridge.Refusal, "nonfinite tensor"):
                bridge.floats(bytes([value]), dtype)

    # q8_0 uses per-32 scales and ties away from zero; a Hadamard impulse spreads with the expected signs.
    def test_q8_and_rotation(self):
        self.assertEqual(bridge.q8_0([0] * 32), bytes(34))
        packed = bridge.q8_0([127, -127, .5, -.5] + [0] * 28)
        self.assertEqual(struct.unpack("<e4b", packed[:6]), (1, 127, -127, 1, -1))
        self.assertEqual(list(bridge.rotate([1, 2, 3, 4], 4)), [5, -1, -2, 0])
        self.assertEqual(list(bridge.rotate([16] + [0] * 255, 256)), [1] * 256)
        with self.assertRaisesRegex(bridge.Refusal, "invalid q8_0"):
            bridge.q8_0([math.nan] * 32)

    # vLLM's [page,head,token,K+V] layout preserves page/token/head order and applies separate FP8 scales.
    def test_paged_kv_layout_and_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = bytearray(2 * 4 * 16 * 512)
            for page in range(2):
                for head in range(4):
                    for token in range(16):
                        at = ((page * 4 + head) * 16 + token) * 512
                        data[at:at + 256] = bytes([56 + page]) * 256
                        data[at + 256:at + 512] = bytes([56 + head]) * 256
            (root / "kv").write_bytes(data)
            layer = dict(kv=dict(file="kv", shape=[2, 4, 16, 512], dtype="float8_e4m3fn"), scales=[2.0, 3.0])
            keys = list(bridge.kv_rows(root, layer, 16, False, 0))
            values = list(bridge.kv_rows(root, layer, 16, True, 0))
            self.assertEqual(len(keys), 32)
            self.assertEqual(keys[0], bridge.q8_0([2] * 1024))
            self.assertEqual(keys[16], bridge.q8_0([2.25] * 1024))
            self.assertEqual(values[0], b"".join(bridge.q8_0([3 * (1 + h / 8)] * 256) for h in range(4)))

    # GDN head order is grouped->tiled; [V,K] stays untransposed, and both conv orientations yield [channel,time].
    def test_recurrent_head_and_conv_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = [float(h * 2 + v % 2) for h in range(48) for v in range(16384)]
            (root / "state.bin").write_bytes(bridge.f32_bytes(state))
            layer = copy.deepcopy(self.snapshot["layers"][0])
            converted = bridge.recurrent_row(root, layer, True, "SD")
            for h in range(48):
                source_head = (h % 16) * 3 + h // 16
                self.assertEqual(struct.unpack_from("<2f", converted, h * 16384 * 4), (source_head * 2, source_head * 2 + 1))
            sd = [float(t * 20000 + c) for t in range(3) for c in range(10240)]
            (root / "conv.bin").write_bytes(bridge.f32_bytes(sd))
            conv = bridge.recurrent_row(root, layer, False, "SD")
            for c in (0, 4095, 4096, 4224, 10112):
                h, d = divmod(c - 4096, 128)
                source = c if c < 4096 else 4096 + ((h % 16) * 3 + h // 16) * 128 + d
                self.assertEqual(struct.unpack_from("<3f", conv, c * 12), tuple(float(t * 20000 + source) for t in range(3)))
            (root / "conv.bin").write_bytes(bridge.f32_bytes(sd[t * 10240 + c] for c in range(10240) for t in range(3)))
            layer["conv"]["shape"] = [10240, 3]
            self.assertEqual(bridge.recurrent_row(root, layer, False, "DS"), conv)

    # Failed conversion preserves the old destination and publishes no temporary delta.
    def test_atomic_refusal_and_file_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "snapshot.json").write_text(json.dumps(self.snapshot))
            (root / "output.delta").write_bytes(b"keep")
            (root / "kv.bin").write_bytes(bytes(12))
            with self.assertRaisesRegex(bridge.Refusal, "truncated or trailing"):
                bridge.convert(root / "snapshot.json", self.target, root / "output.delta")
            self.assertEqual((root / "output.delta").read_bytes(), b"keep")
            self.assertFalse(list(root.glob(".vllm-delta-*")))
            spec = dict(file="../foreign", shape=[1], dtype="float32")
            with self.assertRaisesRegex(bridge.Refusal, "unsafe tensor path"):
                bridge.tensor_bytes(root, spec)

    # The patch is unique, idempotence refuses, and the hook runs after forward and before postprocess.
    def test_patch_location(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Path(directory) / "runner.py"
            runner.write_text("class GPUModelRunner(Base):\n    def execute_model(self):\n        model_output = self._model_forward()\n" + patch_vllm.MARKER + "            pass\n")
            with self.assertRaisesRegex(bridge.Refusal, "differs from inspected"):
                patch_vllm.patch(runner)
            with mock.patch.object(patch_vllm, "RUNNER_SHA256", hashlib.sha256(runner.read_bytes()).hexdigest()):
                patch_vllm.patch(runner)
            text = runner.read_text()
            self.assertLess(text.index("_model_forward()"), text.index("after_prefill(self"))
            self.assertLess(text.index("after_prefill(self"), text.index("gpu_model_runner: postprocess"))
            with self.assertRaisesRegex(bridge.Refusal, "already installed"):
                patch_vllm.patch(runner)

    # The existing C++ envelope and real allocated hybrid caches accept and reproduce a full 64-layer synthetic delta.
    def test_native_roundtrip(self):
        binary = os.environ.get("HYBRID_VLLM_NATIVE_CHECK")
        self.assertTrue(binary, "set HYBRID_VLLM_NATIVE_CHECK to the built native_check; native round-trip must run")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Nonzero sentinels exercise recurrent head/channel permutations through the native reader.
            (root / "kv.bin").write_bytes(b"\x80\x3f" * (4 * 16 * 512))
            (root / "conv.bin").write_bytes(bridge.f32_bytes(float(i % 17) for i in range(30720)))
            (root / "state.bin").write_bytes(bridge.f32_bytes(float(i % 31) / 32 for i in range(786432)))
            (root / "snapshot.json").write_text(json.dumps(self.snapshot))
            (root / "target.json").write_text(json.dumps(self.target))
            header = bridge.convert(root / "snapshot.json", self.target, root / "output.delta")
            self.assertEqual(header["sections"][0]["bytes"], 157452212)
            result = subprocess.run([binary, str(root / "output.delta"), str(root / "target.json")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS native envelope", result.stdout)
            # An append carries only suffix KV and replaces recurrent state at its new endpoint.
            self.snapshot.update(n0=16, n1=32, tokens=list(range(32)))
            (root / "kv.bin").write_bytes(bytes([56]) * (4 * 16 * 512))
            for name in ("conv", "state"):
                path = root / (name + ".bin")
                raw = path.read_bytes()
                path.write_bytes(b"".join(raw[i + 2:i + 4] for i in range(0, len(raw), 4)))
            for layer in self.snapshot["layers"]:
                if layer["layer"] in bridge.ATTENTION:
                    layer["kv"]["dtype"] = "float8_e4m3fn"
                    layer["scales"] = [2.0, 3.0]
                else:
                    layer["conv"]["dtype"] = layer["state"]["dtype"] = "bfloat16"
            (root / "snapshot.json").write_text(json.dumps(self.snapshot))
            bridge.convert(root / "snapshot.json", self.target, root / "append.delta")
            append = subprocess.run([binary, str(root / "output.delta"), str(root / "append.delta"), str(root / "target.json")], capture_output=True, text=True)
            self.assertEqual(append.returncode, 0, append.stdout + append.stderr)
            # A bounded checksum plant proves this reader rejects a changed body.
            with (root / "output.delta").open("r+b") as output:
                length = struct.unpack("<I", output.read(4))[0]
                output.seek(4 + length + 80)
                old = output.read(1)
                output.seek(-1, 1)
                output.write(bytes([old[0] ^ 1]))
            red = subprocess.run([binary, str(root / "output.delta"), str(root / "target.json")], capture_output=True, text=True)
            self.assertEqual(red.returncode, 1)
            self.assertIn("delta checksum mismatch", red.stderr)


if __name__ == "__main__":
    unittest.main()

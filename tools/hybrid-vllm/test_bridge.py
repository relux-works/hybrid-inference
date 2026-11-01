import array
import copy
import hashlib
import json
import math
import os
import random
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
    layers = [dict(layer=i, kv=tensor("%02d-kv.bin" % i, [1, 4, 16, 512], "bfloat16"), scales=[1.0, 1.0]) if i in bridge.ATTENTION else
              dict(layer=i, conv=tensor("%02d-conv.bin" % i, [3, 10240]), state=tensor("%02d-state.bin" % i, [48, 128, 128])) for i in range(64)]
    snapshot = dict(version=1, source_contract=bridge.SOURCE, tp_size=1, pp_size=1, cache_mode="align", block_size=16,
                    n0=0, n1=16, tokens=[37 * p + 5 for p in range(16)], model_layout=layout, conv_layout="SD", rotations=[256, 64], layers=layers)
    return snapshot, target


def write_fixture(root, snapshot):
    for layer in snapshot["layers"]:
        i = layer["layer"]
        if i % 4 == 3:
            rng = random.Random(100 + i)
            values = [rng.randrange(-64, 64) / 16 for _ in range(math.prod(layer["kv"]["shape"]))]
            raw = b"".join(struct.pack("<f", x)[2:] for x in values)
            (root / layer["kv"]["file"]).write_bytes(raw)
        else:
            for name, count, base in (("conv", 30720, 0x3e000000), ("state", 786432, 0x3f000000)):
                bits = array.array("I", range(base + i * count, base + (i + 1) * count))
                if sys.byteorder != "little":
                    bits.byteswap()
                (root / layer[name]["file"]).write_bytes(bits.tobytes())


def reference_hadamard(size):
    # Explicit Sylvester matrix, independent of the converter's in-place butterfly.
    return [[(-1 if (row & col).bit_count() % 2 else 1) / math.sqrt(size)
             for col in range(size)] for row in range(size)]


def reference_q8(values):
    f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
    result = bytearray()
    for start in range(0, len(values), 32):
        block = values[start:start + 32]
        d = f32(max(map(abs, block)) / 127)
        inverse = f32(1 / d) if d else 0
        quant = []
        for value in block:
            scaled = f32(value * inverse)
            quant.append(math.floor(scaled + .5) if scaled >= 0 else math.ceil(scaled - .5))
        result.extend(struct.pack("<e32b", d, *quant))
    return bytes(result)


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

    # Synthetic configuration admits the specified model and refuses parallel, async, speculative and geometric variants.
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

    # A missing layer is refused beside a valid 64-layer control.
    def test_wrong_layer_count(self):
        bridge.validate(self.snapshot, self.target)
        self.snapshot["layers"].pop()
        with self.assertRaisesRegex(bridge.Refusal, "wrong layer count"):
            bridge.validate(self.snapshot, self.target)

    # A manager block below the minimum is refused even with a consistent page shape.
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

    # Distinct FP8 page/head/token/KV rows retain their source order and separate scales without rotation.
    def test_paged_kv_layout_and_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = bytearray(2 * 4 * 16 * 512)
            for page in range(2):
                for head in range(4):
                    for token in range(16):
                        at = ((page * 4 + head) * 16 + token) * 512
                        data[at:at + 256] = bytes([8 + token]) * 128 + bytes([40 + page * 4 + head]) * 128
                        data[at + 256:at + 512] = bytes([80 + token]) * 128 + bytes([64 + page * 4 + head]) * 128
            (root / "kv").write_bytes(data)
            layer = dict(kv=dict(file="kv", shape=[2, 4, 16, 512], dtype="float8_e4m3fn"), scales=[2.0, 3.0])
            keys = list(bridge.kv_rows(root, layer, 16, False, 0))
            values = list(bridge.kv_rows(root, layer, 16, True, 0))
            self.assertEqual(len(keys), 32)
            self.assertEqual(len(values), 32)
            for page in range(2):
                for token in range(16):
                    for value, rows, scale in ((False, keys, 2), (True, values, 3)):
                        expected = []
                        for head in range(4):
                            for byte in ((80 if value else 8) + token, (64 if value else 40) + page * 4 + head):
                                decoded = (1 + (byte & 7) / 8) * 2 ** ((byte >> 3) - 7)
                                expected.extend([decoded * scale] * 128)
                        self.assertEqual(rows[page * 16 + token], reference_q8(expected))

    # Non-unit FP8 K/V scales precede rotation; every q8_0 byte uses an independent decode and matrix.
    def test_fp8_rotated_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = bytes((8 + (i * 17 + i // 512) % 96) for i in range(4 * 16 * 512))
            (root / "kv").write_bytes(raw)
            layer = dict(kv=dict(file="kv", shape=[1, 4, 16, 512], dtype="float8_e4m3fn"), scales=[2.0, 3.0])
            for value, size, scale in ((False, 256, 2), (True, 64, 3)):
                matrix = reference_hadamard(size)
                rows = list(bridge.kv_rows(root, layer, 16, value, size))
                self.assertEqual(len(rows), 16)
                for token in range(16):
                    row = []
                    for head in range(4):
                        at = (head * 16 + token) * 512 + int(value) * 256
                        for start in range(at, at + 256, size):
                            decoded = [(1 + (byte & 7) / 8) * 2 ** ((byte >> 3) - 7) * scale for byte in raw[start:start + size]]
                            row.extend(sum(a * b for a, b in zip(weights, decoded)) for weights in matrix)
                    self.assertEqual(rows[token], reference_q8(row), (value, token))

    # Every serialized BF16 KV row and f32 recurrent byte matches an independent source-layout reference.
    def test_serialized_layout_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root, self.snapshot)
            (root / "snapshot.json").write_text(json.dumps(self.snapshot))
            bridge.convert(root / "snapshot.json", self.target, root / "output.delta")
            with (root / "output.delta").open("rb") as wire:
                length = struct.unpack("<I", wire.read(4))[0]
                wire.seek(length, 1)
                self.assertEqual(wire.read(16), struct.pack("<IiII", 0xAF143CD8, 0, 1, 16))
                for pos, token in enumerate(self.snapshot["tokens"]):
                    self.assertEqual(wire.read(24), struct.pack("<iIiiii", pos, 1, pos, pos, token, 0))
                self.assertEqual(wire.read(16), struct.pack("<4I", 0, 16, 256, 64))
                matrices = {n: reference_hadamard(n) for n in (256, 64)}
                for value, size in ((0, 256), (1, 64)):
                    for i in range(3, 64, 4):
                        raw = (root / self.snapshot["layers"][i]["kv"]["file"]).read_bytes()
                        source = [struct.unpack("<f", b"\0\0" + raw[j:j + 2])[0] for j in range(0, len(raw), 2)]
                        self.assertEqual(wire.read(12), struct.pack("<iQ", 8, 1088))
                        for token in range(16):
                            row = []
                            for head in range(4):
                                at = (head * 16 + token) * 512 + value * 256
                                for start in range(at, at + 256, size):
                                    block = source[start:start + size]
                                    row.extend(sum(a * b for a, b in zip(weights, block)) for weights in matrices[size])
                            self.assertEqual(wire.read(1088), reference_q8(row), ("KV", value, i, token))
                self.assertEqual(wire.read(20), struct.pack("<IiIII", 1, 15, 0, 0, 64))
                # Source [key_head,repeat] becomes [repeat,key_head]; preserve [V,K] within each head.
                heads = [key * 3 + repeat for repeat in range(3) for key in range(16)]
                for name in ("conv", "state"):
                    for i in (j for j in range(64) if j % 4 != 3):
                        raw = (root / self.snapshot["layers"][i][name]["file"]).read_bytes()
                        if name == "state":
                            expected = b"".join(raw[h * 65536:(h + 1) * 65536] for h in heads)
                        else:
                            channels = list(range(4096)) + [4096 + h * 128 + d for h in heads for d in range(128)]
                            expected = b"".join(raw[(t * 10240 + c) * 4:(t * 10240 + c + 1) * 4]
                                                for c in channels for t in range(3))
                        self.assertEqual(wire.read(12), struct.pack("<iQ", 0, len(expected)))
                        self.assertTrue(wire.read(len(expected)) == expected, (name, i))
                self.assertEqual(len(wire.read()), 48)

    # GDN head order is grouped->tiled; [V,K] stays untransposed, and both conv orientations yield [channel,time].
    def test_recurrent_head_and_conv_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = [float(h * 2 + v % 2) for h in range(48) for v in range(16384)]
            (root / "00-state.bin").write_bytes(bridge.f32_bytes(state))
            layer = copy.deepcopy(self.snapshot["layers"][0])
            converted = bridge.recurrent_row(root, layer, True, "SD")
            for h in range(48):
                source_head = (h % 16) * 3 + h // 16
                self.assertEqual(struct.unpack_from("<2f", converted, h * 16384 * 4), (source_head * 2, source_head * 2 + 1))
            sd = [float(t * 20000 + c) for t in range(3) for c in range(10240)]
            (root / "00-conv.bin").write_bytes(bridge.f32_bytes(sd))
            conv = bridge.recurrent_row(root, layer, False, "SD")
            for c in (0, 4095, 4096, 4224, 10112):
                h, d = divmod(c - 4096, 128)
                source = c if c < 4096 else 4096 + ((h % 16) * 3 + h // 16) * 128 + d
                self.assertEqual(struct.unpack_from("<3f", conv, c * 12), tuple(float(t * 20000 + source) for t in range(3)))
            (root / "00-conv.bin").write_bytes(bridge.f32_bytes(sd[t * 10240 + c] for c in range(10240) for t in range(3)))
            layer["conv"]["shape"] = [10240, 3]
            self.assertEqual(bridge.recurrent_row(root, layer, False, "DS"), conv)

    # Failed conversion preserves the old destination and publishes no temporary delta.
    def test_atomic_refusal_and_file_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "snapshot.json").write_text(json.dumps(self.snapshot))
            (root / "output.delta").write_bytes(b"keep")
            (root / "03-kv.bin").write_bytes(bytes(12))
            with self.assertRaisesRegex(bridge.Refusal, "truncated or trailing"):
                bridge.convert(root / "snapshot.json", self.target, root / "output.delta")
            self.assertEqual((root / "output.delta").read_bytes(), b"keep")
            self.assertFalse(list(root.glob(".vllm-delta-*")))
            spec = dict(file="../foreign", shape=[1], dtype="float32")
            with self.assertRaisesRegex(bridge.Refusal, "unsafe tensor path"):
                bridge.tensor_bytes(root, spec)

    # Matching patch reruns preserve bytes; unknown or modified source and misplaced hooks refuse.
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
            original_hash = hashlib.sha256(text.replace(patch_vllm.HOOK, "").encode()).hexdigest()
            with mock.patch.object(patch_vllm, "RUNNER_SHA256", original_hash):
                patch_vllm.patch(runner)
                self.assertEqual(runner.read_text(), text)
                runner.write_text(text + "# changed\n")
                with self.assertRaisesRegex(bridge.Refusal, "differs from inspected"):
                    patch_vllm.patch(runner)
                crlf = text.replace("\n", "\r\n").encode()
                runner.write_bytes(crlf)
                with self.assertRaisesRegex(bridge.Refusal, "differs from inspected"):
                    patch_vllm.patch(runner)
                self.assertEqual(runner.read_bytes(), crlf)
                runner.write_text(text.replace(patch_vllm.HOOK, "") + patch_vllm.HOOK)
                with self.assertRaisesRegex(bridge.Refusal, "hook location changed"):
                    patch_vllm.patch(runner)

    # CPU doubles exercise late-refusal cleanup, retry and publication, not live tensor extraction.
    def test_export_refusal_cleanup(self):
        class MambaSpec:
            block_size = 16
            mamba_cache_mode = "align"
            num_speculative_blocks = num_prefill_checkpoint_blocks = 0

        def cache(shape):
            tensor = mock.MagicMock(shape=shape, dtype="torch.float32")
            tensor.detach.return_value = tensor.cpu.return_value = tensor.contiguous.return_value = tensor
            tensor.view.return_value.numpy.return_value.tobytes.return_value = b"tensor"
            return tensor

        conv = mock.MagicMock(shape=(2, 3, 10240))
        state = mock.MagicMock(shape=(2, 48, 128, 128))
        conv.__getitem__.return_value = cache((3, 10240))
        state.__getitem__.return_value = cache((48, 128, 128))
        groups = [Namespace(kv_cache_spec=MambaSpec(), layer_names=[str(i)]) for i in range(2)]
        req = Namespace(mm_features=[], prompt_embeds=None, lora_request=None, output_token_ids=[],
                        prompt_token_ids=self.snapshot["tokens"], num_computed_tokens=0, block_ids=[[1], [0]])
        runner = Namespace(requests={"request": req}, kv_cache_config=Namespace(kv_cache_groups=groups),
                           vllm_config=Namespace(compilation_config=Namespace(static_forward_context={
                               str(i): Namespace(kv_cache=[conv, state]) for i in range(2)})))
        modules = {"torch": Namespace(uint8="uint8", cuda=Namespace(synchronize=lambda: None)),
                   "vllm": Namespace(__version__="0.32.0"),
                   "vllm.model_executor.layers.mamba.mamba_utils": Namespace(get_conv_state_layout=lambda: "SD"),
                   "vllm.model_executor.models.utils": Namespace(extract_layer_index=int),
                   "vllm.v1.kv_cache_interface": Namespace(FullAttentionSpec=type("FullAttentionSpec", (), {}), MambaSpec=MambaSpec)}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, modules), mock.patch.object(exporter, "check_config"):
            root = Path(directory)
            config = dict(target=self.target, tokens=req.prompt_token_ids, output=directory)
            scheduled = Namespace(num_scheduled_tokens={"request": 16})
            with self.assertRaisesRegex(bridge.Refusal, "invalid recurrent block ownership"):
                exporter.export_snapshot(runner, scheduled, config, 0)
            self.assertEqual(list(root.iterdir()), [])
            req.block_ids[1] = [1]
            with self.assertRaisesRegex(bridge.Refusal, "wrong layer count"):
                exporter.export_snapshot(runner, scheduled, config, 0)
            self.assertEqual(list(root.iterdir()), [])
            with mock.patch.object(exporter, "validate"):
                self.assertEqual(exporter.export_snapshot(runner, scheduled, config, 0), 16)
            destination = root / "capture-00000016"
            self.assertEqual(list(root.iterdir()), [destination])
            self.assertEqual(len(json.loads((destination / "snapshot.json").read_text())["layers"]), 2)
            before = {p.name: p.read_bytes() for p in destination.iterdir()}
            with self.assertRaisesRegex(bridge.Refusal, "capture already exists"):
                exporter.export_snapshot(runner, scheduled, config, 0)
            self.assertEqual({p.name: p.read_bytes() for p in destination.iterdir()}, before)

    # The existing C++ envelope and real allocated hybrid caches accept and reproduce a full 64-layer synthetic delta.
    def test_native_roundtrip(self):
        binary = os.environ.get("HYBRID_VLLM_NATIVE_CHECK")
        self.assertTrue(binary, "set HYBRID_VLLM_NATIVE_CHECK to the built native_check; native round-trip must run")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root, self.snapshot)
            (root / "snapshot.json").write_text(json.dumps(self.snapshot))
            (root / "target.json").write_text(json.dumps(self.target))
            header = bridge.convert(root / "snapshot.json", self.target, root / "output.delta")
            self.assertEqual(header["sections"][0]["bytes"], 157452212)
            result = subprocess.run([binary, str(root / "output.delta"), str(root / "target.json")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS native envelope", result.stdout)
            # An append carries only suffix KV and replaces recurrent state at its new endpoint.
            self.snapshot.update(n0=16, n1=32, tokens=[(37 * p + 5) % 1000 for p in range(32)])
            for layer in self.snapshot["layers"]:
                if layer["layer"] in bridge.ATTENTION:
                    rng = random.Random(200 + layer["layer"])
                    (root / layer["kv"]["file"]).write_bytes(bytes(rng.randrange(8, 112) for _ in range(4 * 16 * 512)))
                    layer["kv"]["dtype"] = "float8_e4m3fn"
                    layer["scales"] = [2.0, 3.0]
                else:
                    for name in ("conv", "state"):
                        path = root / layer[name]["file"]
                        raw = path.read_bytes()
                        path.write_bytes(b"".join(raw[i + 2:i + 4] for i in range(0, len(raw), 4)))
                        layer[name]["dtype"] = "bfloat16"
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

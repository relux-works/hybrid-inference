"""Vectorized codec; the scalar implementation in bridge remains the oracle."""

import math
from pathlib import Path

from bridge import DTYPES, floats, require


class NumpyCodec:
    def __init__(self):
        import numpy as np
        self.np = np
        self.tables = {}

    def tensor(self, root, spec):
        np = self.np
        path = Path(spec["file"])
        require(not path.is_absolute() and len(path.parts) == 1 and path.name not in ("", ".", ".."), "unsafe tensor path")
        path = root / path
        require(not path.is_symlink(), "unsafe tensor symlink")
        require(spec["dtype"] in DTYPES, "unsupported tensor dtype")
        require(path.stat().st_size == math.prod(spec["shape"]) * DTYPES[spec["dtype"]], "truncated or trailing tensor bytes")
        dtype = {"float32": "<f4", "bfloat16": "<u2"}.get(spec["dtype"], "u1")
        return np.memmap(path, dtype=dtype, mode="r", shape=tuple(spec["shape"]))

    def decode(self, raw, dtype):
        np = self.np
        if dtype == "float32":
            data = np.asarray(raw, dtype=np.float32)
        elif dtype == "bfloat16":
            data = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            if dtype not in self.tables:
                # Invalid FP8 encodings remain NaN and are refused after table lookup.
                table = []
                for i in range(256):
                    try:
                        table.append(floats(bytes([i]), dtype)[0])
                    except ValueError:
                        table.append(math.nan)
                self.tables[dtype] = np.array(table, dtype=np.float32)
            data = self.tables[dtype][raw]
        require(bool(np.isfinite(data).all()), "nonfinite tensor value")
        return data

    def quantize(self, data, rotation):
        np = self.np
        data = np.array(data, dtype=np.float32, copy=True, order="C")
        require(bool(np.isfinite(data).all()), "invalid q8_0 input")
        if rotation:
            width = 1
            while width < rotation:
                pairs = data.reshape(-1, 2, width)
                left, right = pairs[:, 0].copy(), pairs[:, 1].copy()
                pairs[:, 0], pairs[:, 1] = left + right, left - right
                width *= 2
            data /= np.float32(math.sqrt(rotation))
        require(bool(np.isfinite(data).all()) and data.size % 32 == 0, "invalid q8_0 input")
        rows = data.reshape(-1, 32)
        scale = (np.max(np.abs(rows), axis=1).astype(np.float64) / 127).astype(np.float32)
        require(bool((scale <= 65504).all()), "q8_0 scale overflow")
        inverse = np.zeros_like(scale)
        np.divide(1., scale.astype(np.float64), out=inverse, where=scale != 0, casting="unsafe")
        require(bool(np.isfinite(inverse).all()), "q8_0 inverse overflow")
        scaled = rows * inverse[:, None]
        absolute = np.abs(scaled)
        integral = np.floor(absolute)
        quant = np.copysign(integral + ((absolute - integral) >= .5), scaled).astype(np.int8)
        out = np.empty((len(rows), 34), dtype=np.uint8)
        out[:, :2] = scale.astype("<f2").view(np.uint8).reshape(-1, 2)
        out[:, 2:] = quant.view(np.uint8)
        return out.tobytes()

    def kv_rows(self, root, layer, block, value, rotation):
        spec = layer["kv"]
        raw = self.tensor(root, spec)
        # Bound working memory independently of the delta and context size.
        pages = max(1, 256 // block)
        for start in range(0, len(raw), pages):
            data = self.decode(raw[start:start + pages], spec["dtype"])
            rows = data[:, :, :, 256 if value else 0:512 if value else 256].transpose(0, 2, 1, 3)
            rows = (rows.astype(self.np.float64) * layer["scales"][int(value)]).astype(self.np.float32)
            yield self.quantize(rows, rotation)

    def recurrent_row(self, root, layer, state, layout):
        np = self.np
        spec = layer["state" if state else "conv"]
        data = self.decode(self.tensor(root, spec), spec["dtype"])
        heads = (np.arange(48) % 16) * 3 + np.arange(48) // 16
        if state:
            return data[heads].astype("<f4").tobytes()
        channels = np.concatenate((np.arange(4096), (4096 + heads[:, None] * 128 + np.arange(128)).reshape(-1)))
        if layout == "SD":
            data = data.T
        return data[channels].astype("<f4").tobytes()


class TorchCodec(NumpyCodec):
    def __init__(self, device="cuda"):
        super().__init__()
        import torch
        require(device == "cpu" or torch.cuda.is_available(), "torch CUDA unavailable")
        self.torch, self.device = torch, device

    def quantize(self, data, rotation):
        torch = self.torch
        data = torch.tensor(data.copy(), dtype=torch.float32, device=self.device)
        require(bool(torch.isfinite(data).all()), "invalid q8_0 input")
        if rotation:
            width = 1
            while width < rotation:
                pairs = data.reshape(-1, 2, width)
                left, right = pairs[:, 0].clone(), pairs[:, 1].clone()
                pairs[:, 0], pairs[:, 1] = left + right, left - right
                width *= 2
            data /= math.sqrt(rotation)
        require(data.numel() % 32 == 0 and bool(torch.isfinite(data).all()), "invalid q8_0 input")
        rows = data.reshape(-1, 32)
        scale = (rows.abs().amax(dim=1).double() / 127).float()
        require(bool((scale <= 65504).all()), "q8_0 scale overflow")
        inverse = torch.where(scale != 0, (1 / scale.double()).float(), 0)
        require(bool(torch.isfinite(inverse).all()), "q8_0 inverse overflow")
        scaled = rows * inverse[:, None]
        absolute = scaled.abs()
        integral = absolute.floor()
        quant = torch.copysign(integral + ((absolute - integral) >= .5), scaled).to(torch.int8)
        out = torch.empty((len(rows), 34), dtype=torch.uint8, device=self.device)
        out[:, :2] = scale.half().view(torch.uint8).reshape(-1, 2)
        out[:, 2:] = quant.view(torch.uint8)
        return out.cpu().numpy().tobytes()


def codec_for(backend):
    if backend == "reference":
        return None
    if backend == "auto":
        try:
            import torch
        except ImportError:
            return NumpyCodec()
        return TorchCodec() if torch.cuda.is_available() else NumpyCodec()
    require(backend in ("numpy", "torch"), "unknown converter backend")
    return TorchCodec() if backend == "torch" else NumpyCodec()

"""CPU-only synthetic vLLM page layout for protocol tests, without model execution."""

import json
from pathlib import Path

import numpy as np

from bridge import ATTENTION, RECURRENT, SOURCE, token_hash


class FakeBackend:
    def __init__(self, target, block_size=16):
        self.target, self.block_size = target, block_size
        self.calls, self.prefixes = [], set()
        self.available = True

    def healthy(self):
        return self.available

    def prefill(self, tokens, n0, output, capture=True):
        block = self.block_size
        reused = max((n for n in range(0, len(tokens), block) if token_hash(tokens[:n]) in self.prefixes), default=0)
        self.calls.append(dict(tokens=list(tokens), n0=n0, capture=capture, reused=reused))
        self.prefixes.update(token_hash(tokens[:n]) for n in range(block, len(tokens) + 1, block))
        if not capture:
            return None
        start = n0 // block * block
        pages = (len(tokens) - start) // block
        shape = (pages, 4, block, 512)
        data = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
        journal = np.array(tokens[start:], dtype=np.uint32).reshape(pages, 1, block, 1)
        data = ((data + journal * 7) % 100 + 8).astype(np.uint8)
        output = Path(output)
        (output / "kv.bin").write_bytes(data.tobytes())
        recurrent = {}
        for name, dims in (("conv", [3, 10240]), ("state", [48, 128, 128])):
            values = (np.arange(np.prod(dims), dtype=np.float32) % 4093 + len(tokens)) / 8192
            (output / (name + ".bin")).write_bytes(values.astype("<f4").tobytes())
            recurrent[name] = dict(file=name + ".bin", shape=dims, dtype="float32")
        layers = [dict(layer=i, kv=dict(file="kv.bin", shape=list(shape), dtype="float8_e4m3fn"), scales=[.125, .375]) if i in ATTENTION
                  else dict(layer=i, **recurrent) for i in range(64)]
        snapshot = dict(version=1, source_contract=SOURCE, tp_size=1, pp_size=1, cache_mode="align", block_size=block,
                        n0=n0, n1=len(tokens), tokens=tokens, model_layout=self.target["model_layout"],
                        conv_layout="SD", rotations=[256, 64], layers=layers)
        path = output / "snapshot.json"
        path.write_text(json.dumps(snapshot))
        return path

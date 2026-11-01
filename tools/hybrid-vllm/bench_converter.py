"""Measure complete bounded delta conversions using synthetic vLLM-layout tensors."""

import argparse
import json
from pathlib import Path
import tempfile
import time

from bridge import convert
from test_fast_codec import synthetic


def benchmark(count, root, backend):
    started, total, chunks = 0., 0, []
    tokens = [i % 1000 for i in range(count)]
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        folder = Path(tmp)
        n0 = 0
        while n0 < count:
            n1 = min(count, n0 + 8000)
            snapshot, target = synthetic(folder, n1 - n0)
            snapshot.update(n0=n0, n1=n1, tokens=tokens[:n1])
            path = folder / "snapshot.json"
            path.write_text(json.dumps(snapshot))
            delta = folder / "out.delta"
            before = time.perf_counter()
            convert(path, target, delta, backend)
            elapsed = time.perf_counter() - before
            started += elapsed
            total += delta.stat().st_size
            chunks.append(dict(n0=n0, n1=n1, seconds=elapsed))
            delta.unlink()
            n0 = n1
    return dict(tokens=count, backend=backend, seconds=started, wire_bytes=total, chunks=chunks,
                includes="decode, rotation, q8_0, recurrent reorder, disk write, XXH128, SHA256, fsync; excludes fixture generation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[4000, 40000])
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--backend", default="numpy", choices=("numpy", "torch"))
    args = parser.parse_args()
    for count in args.tokens:
        if count <= 0 or count % 16:
            parser.error("tokens must be positive multiples of 16")
        print(json.dumps(benchmark(count, args.scratch, args.backend)), flush=True)

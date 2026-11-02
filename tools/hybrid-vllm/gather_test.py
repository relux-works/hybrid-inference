"""Mandatory torch-CPU preflight; no vLLM import, CUDA initialization or model."""

import io
import struct
import unittest

import torch

from bridge import Refusal
from exporter import write_attention_pages


class GatherTests(unittest.TestCase):
    # Scalar source coordinates define bytes independently of the gather's slice/permute/reshape.
    def test_page_bytes(self):
        torch.set_num_threads(1)
        for kernel, ratio in ((16, 1), (8, 2), (16, 3)):
            block = kernel * ratio
            shape = (5 * ratio, 4, kernel, 512)
            count = 5 * ratio * 4 * kernel * 512
            for dtype in (torch.bfloat16, torch.uint8):
                with self.subTest(kernel=kernel, ratio=ratio, dtype=dtype):
                    if dtype == torch.bfloat16:
                        indices = torch.arange(count, device="cpu")
                        bits = ((indices + indices // 16384 * 97) % 16384 + 0x2000).to(torch.int16)
                        cache = bits.view(dtype).reshape(shape)
                    else:
                        cache = (torch.arange(count, device="cpu") % 251).to(dtype).reshape(shape)
                    # Exercise a strided logical cache view as well as contiguous storage.
                    for strided in (False, True):
                        source = cache.transpose(1, 2).contiguous().transpose(1, 2) if strided else cache
                        for n0, n1 in ((0, 3 * block), (block, 3 * block)):
                            blocks = [3, 1, 4]
                            output = io.BytesIO()
                            write_attention_pages(output, source, blocks, block, n0, n1)
                            expected = bytearray()
                            for logical in range(n0 // block, n1 // block):
                                for head in range(4):
                                    for token in range(block):
                                        page = blocks[logical] * ratio + token // kernel
                                        for channel in range(512):
                                            index = ((page * 4 + head) * kernel + token % kernel) * 512 + channel
                                            if dtype == torch.bfloat16:
                                                expected.extend(struct.pack("<H", 0x2000 + (index + index // 16384 * 97) % 16384))
                                            else:
                                                expected.append(index % 251)
                            self.assertEqual(output.getvalue(), expected, (strided, n0, n1))
        self.assertFalse(torch.cuda.is_initialized())

    # Null/out-of-range pages and incompatible kernel sizes refuse beside a valid page control.
    def test_refusals(self):
        cache = torch.zeros((6, 4, 8, 512), dtype=torch.uint8, device="cpu")
        output = io.BytesIO()
        write_attention_pages(output, cache, [1], 16, 0, 16)
        self.assertEqual(len(output.getvalue()), 4 * 16 * 512)
        for blocks in ([0], [-1], [3]):
            output = io.BytesIO()
            with self.assertRaisesRegex(Refusal, "invalid attention block ownership"):
                write_attention_pages(output, cache, blocks, 16, 0, 16)
            self.assertEqual(output.getvalue(), b"")
        with self.assertRaisesRegex(Refusal, "wrong kernel block size"):
            write_attention_pages(io.BytesIO(), cache, [1], 17, 0, 17)
        for shape in ((6, 8, 512), (6, 2, 8, 512), (6, 4, 8, 256)):
            with self.assertRaisesRegex(Refusal, "unsupported attention tensor layout"):
                write_attention_pages(io.BytesIO(), torch.zeros(shape, dtype=torch.uint8), [1], 16, 0, 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)

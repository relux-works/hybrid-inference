#!/usr/bin/env python3
"""Compare native full-file and fragmented SHA-256 with two external references."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class FileHashTests(unittest.TestCase):
    binary = Path('build/bin/llama-hybrid-check-hash').resolve()
    model = None

    def native(self, path, chunk=None):
        command = [str(self.binary), str(path)]
        if chunk is not None:
            command.append(str(chunk))
        process = subprocess.run(command, text=True, capture_output=True, check=True)
        result = json.loads(process.stdout)
        self.assertEqual(result['bytes'], path.stat().st_size)
        return result['sha256']

    def reference(self, path):
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        expected = digest.hexdigest()
        actual = subprocess.run(['sha256sum', str(path)], text=True, capture_output=True, check=True)
        self.assertEqual(actual.stdout.split()[0], expected)
        return expected

    def test_file_boundaries(self):
        # Full-file hashes agree at SHA block/padding and 64 KiB read boundaries, including empty input.
        with tempfile.TemporaryDirectory(prefix='hash-', dir='.temp') as directory:
            path = Path(directory) / 'fixture'
            for size in (0, 1, 55, 56, 63, 64, 65, 119, 120, 127, 128, 129, 65535, 65536, 65537, 131073):
                with self.subTest(size=size):
                    path.write_bytes(bytes((i * 131 + 17) % 256 for i in range(size)))
                    self.assertEqual(self.native(path), self.reference(path))

    def test_fragmented_updates(self):
        # Partial heads, direct blocks, tails and empty updates preserve the digest across chunk sizes.
        with tempfile.TemporaryDirectory(prefix='hash-', dir='.temp') as directory:
            path = Path(directory) / 'fixture'
            for size in (0, 1, 55, 56, 63, 64, 65, 127, 128, 129, 4097):
                path.write_bytes(bytes((i * 131 + 17) % 256 for i in range(size)))
                expected = self.reference(path)
                for chunk in (1, 3, 55, 56, 63, 64, 65, 127, 128, 129, 65536):
                    with self.subTest(size=size, chunk=chunk):
                        self.assertEqual(self.native(path, chunk), expected)

    def test_one_byte_mutations(self):
        # A one-byte change in the first block, a block boundary or the final tail changes the whole-file identity.
        with tempfile.TemporaryDirectory(prefix='hash-', dir='.temp') as directory:
            path = Path(directory) / 'fixture'
            original = bytes((i * 131 + 17) % 256 for i in range(65537))
            path.write_bytes(original)
            before = self.reference(path)
            self.assertEqual(self.native(path), before)
            for position in (0, 63, 64, 65535, 65536):
                with self.subTest(position=position):
                    changed = bytearray(original)
                    changed[position] ^= 1
                    path.write_bytes(changed)
                    expected = self.reference(path)
                    self.assertNotEqual(expected, before)
                    self.assertEqual(self.native(path), expected)
                    self.assertNotEqual(self.native(path, 65), before)
                    self.assertEqual(self.native(path, 65), expected)

    def test_model(self):
        # The real GGUF agrees with sha256sum/hashlib through both the startup function and vendor streaming path.
        if self.model is None:
            self.skipTest('pass --model to include the local GGUF')
        path = Path(self.model)
        expected = self.reference(path)
        self.assertEqual(self.native(path), expected)
        self.assertEqual(self.native(path, 65537), expected)
        print(json.dumps(dict(model_bytes=path.stat().st_size, sha256=expected)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=FileHashTests.binary)
    parser.add_argument('--model', type=Path)
    args, remaining = parser.parse_known_args()
    FileHashTests.binary = args.binary.resolve()
    FileHashTests.model = args.model
    unittest.main(argv=[__file__, *remaining])

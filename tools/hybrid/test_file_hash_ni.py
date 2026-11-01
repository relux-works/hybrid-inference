#!/usr/bin/env python3
"""Check native SHA-NI dispatch and whole-file SHA-256 against hashlib."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import unittest


class FileHashNITests(unittest.TestCase):
    binary = Path('build/bin/llama-hybrid-check-hash').resolve()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='hash-ni-', dir='.temp')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'fixture'

    def invoke(self, portable=False, chunk=None, path=None, switch=None):
        env = os.environ.copy()
        env.pop('HYBRID_HASH_PORTABLE', None)
        if portable or switch is not None:
            env['HYBRID_HASH_PORTABLE'] = '1' if portable else switch
        command = [str(self.binary), str(self.path if path is None else path)]
        if chunk is not None:
            command += ['--file-chunk', str(chunk)]
        return subprocess.run(command, env=env, text=True, capture_output=True)

    def native(self, portable=False, chunk=None, switch=None):
        result = self.invoke(portable, chunk, switch=switch)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record['bytes'], self.path.stat().st_size)
        if portable:
            self.assertEqual(record['backend'], 'portable')
        return record

    def check_both(self, expected, chunk=None):
        for portable in (False, True):
            with self.subTest(portable=portable, chunk=chunk):
                self.assertEqual(self.native(portable, chunk)['sha256'], expected)

    def test_nist_vectors(self):
        # Both file paths match published SHA-256 answers, including one- and two-block padding.
        vectors = (
            (b'', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
            (b'abc', 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'),
            (b'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq',
             '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1'),
            (b'abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno'
             b'ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu',
             'cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1'),
            (b'a' * 1000000, 'cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0'),
        )
        for data, expected in vectors:
            with self.subTest(size=len(data)):
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
                self.path.write_bytes(data)
                self.check_both(expected)

    def test_random_block_boundaries(self):
        # Both paths match hashlib at every 64-byte boundary through 300000 and every tail residue near each end.
        data = random.Random(23).randbytes(300001)
        sizes = set(range(129)) | set(range(299936, 300002))
        for boundary in range(64, 300001, 64):
            sizes.update((boundary - 1, boundary, boundary + 1))
        for size in sorted(sizes):
            with self.subTest(size=size):
                payload = data[:size]
                self.path.write_bytes(payload)
                self.check_both(hashlib.sha256(payload).hexdigest())

    def test_odd_reads(self):
        # Fragmented native reads preserve partial heads and tails across all residues and multi-read files.
        data = random.Random(24).randbytes(300001)
        for size in (*range(129), 65535, 65536, 65537, 300001):
            self.path.write_bytes(data[:size])
            expected = hashlib.sha256(data[:size]).hexdigest()
            for chunk in (1, 3, 7, 55, 63, 65, 127, 4093, 65535):
                with self.subTest(size=size):
                    self.check_both(expected, chunk)

    def test_sparse_above_4gib(self):
        # A 4 GiB + 65 byte sparse stream retains the high byte count and final tail on both paths.
        size = 2**32 + 65
        with self.path.open('wb') as target:
            target.truncate(size)
            for offset, data in ((0, b'a'), (2**31 + 7, b'b'), (size - 1, b'c')):
                target.seek(offset)
                target.write(data)
        digest = hashlib.sha256()
        with self.path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        self.check_both(digest.hexdigest())
        print(json.dumps(dict(sparse_bytes=size, sha256=digest.hexdigest())), flush=True)

    def test_dispatch(self):
        # Observed dispatch uses SHA-NI on a capable Linux x86-64 CPU; only switch value 1 forces portable.
        self.path.write_bytes(random.Random(25).randbytes(129))
        expected = hashlib.sha256(self.path.read_bytes()).hexdigest()
        automatic = self.native()
        forced = self.native(True)
        self.assertEqual(automatic['sha256'], expected)
        self.assertEqual(forced['sha256'], expected)
        if platform.system() == 'Darwin':
            self.assertEqual(automatic['backend'], 'commoncrypto')
        elif platform.machine() in ('x86_64', 'AMD64') and Path('/proc/cpuinfo').exists():
            flags = Path('/proc/cpuinfo').read_text().split('flags', 1)[1].split('\n', 1)[0].split()
            capable = all(flag in flags for flag in ('sha_ni', 'ssse3', 'sse4_1'))
            self.assertEqual(automatic['backend'], 'sha-ni' if capable else 'portable')
        for switch in ('', '0', 'true'):
            self.assertEqual(self.native(switch=switch)['backend'], automatic['backend'])
        print(json.dumps(dict(automatic=automatic['backend'], forced=forced['backend'])), flush=True)

    def test_read_error(self):
        # Open and actual read failures return no digest; the same binary still hashes a valid file.
        self.path.write_bytes(b'abc')
        for portable in (False, True):
            self.assertEqual(self.native(portable)['sha256'], hashlib.sha256(b'abc').hexdigest())
            missing = self.invoke(portable, path=self.path.with_name('missing'))
            self.assertEqual(missing.returncode, 1)
            self.assertEqual(missing.stdout, '')
            self.assertIn('cannot open model', missing.stderr)
            if platform.system() == 'Linux':
                for path in (Path('/proc/self/mem'), Path(self.directory.name)):
                    failed = self.invoke(portable, path=path)
                    self.assertEqual(failed.returncode, 1)
                    self.assertEqual(failed.stdout, '')
                    self.assertIn('cannot read model', failed.stderr)

    def test_one_byte_change(self):
        # A one-byte change in a head, interior block or tail changes both identities and still matches hashlib.
        original = random.Random(26).randbytes(300001)
        before = hashlib.sha256(original).hexdigest()
        self.path.write_bytes(original)
        self.check_both(before)
        for offset in (0, 63, 64, 65535, 65536, len(original) - 1):
            changed = bytearray(original)
            changed[offset] ^= 1
            self.path.write_bytes(changed)
            expected = hashlib.sha256(changed).hexdigest()
            self.assertNotEqual(expected, before)
            self.check_both(expected)

    def test_invalid_read_size(self):
        # Invalid native chunk sizes fail without a digest; adjacent valid sizes retain the file identity.
        self.path.write_bytes(b'abc')
        for chunk in ('0', '65537', '-1', 'x', '18446744073709551616'):
            result = self.invoke(chunk=chunk)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, '')
        for chunk in (1, 65536):
            self.check_both(hashlib.sha256(b'abc').hexdigest(), chunk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=FileHashNITests.binary)
    args, remaining = parser.parse_known_args()
    FileHashNITests.binary = args.binary.resolve()
    unittest.main(argv=[__file__, *remaining])

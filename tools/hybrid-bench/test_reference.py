"""Reference metadata contracts; CPU integration is opt-in via test_reference_native.py."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from bench_common import Refusal
from reference_logits import reference_header, scoring_fixture


class ReferenceTests(unittest.TestCase):
    # A pinned header and identical explicit token positions pass; wrong model/context/N/M refuse.
    def test_reference_binding(self):
        header = dict(format='f16-logsoftmax-le-v1', model_sha256='a' * 64, tokens=[1, 2], continuation=[3], n=2, m=1, score_batch=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.bin'
            data = json.dumps(header).encode()
            path.write_bytes(b'HYBLOG01' + struct.pack('<I', len(data)) + data)
            self.assertEqual(reference_header(path, 'a' * 64), header)
            for digest in ('b' * 64, 'a', ''):
                with self.assertRaisesRegex(Refusal, 'model hash mismatch'):
                    reference_header(path, digest)
            client = dict(model_sha256='b' * 64, context=256)
            fixture = scoring_fixture(header, [1, 2], 1, client, {}, ['node.delta'], path)
            self.assertEqual(fixture['continuation'], [3])
            self.assertEqual(fixture['deltas'], ['node.delta'])
            for tokens, count, message in (([1, 3], 1, 'context tokens'), ([1, 2], 2, 'N/M')):
                with self.assertRaisesRegex(Refusal, message):
                    scoring_fixture(header, tokens, count, client, {}, [], path)
            with self.assertRaisesRegex(Refusal, 'N/M'):
                scoring_fixture(dict(header, n=3), [1, 2], 1, client, {}, [], path)

    # Header lengths are bounded before allocation; truncated and wrong-version files never become fixtures.
    def test_reference_framing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.bin'
            for data, message in ((b'', 'magic'), (b'HYBLOG01', 'truncated'),
                                  (b'HYBLOG01' + struct.pack('<I', 9 * 1024**2), 'exceeds'),
                                  (b'HYBLOG01' + struct.pack('<I', 2) + b'{', 'truncated')):
                path.write_bytes(data)
                with self.assertRaisesRegex(Refusal, message):
                    reference_header(path, 'a' * 64)


if __name__ == '__main__':
    unittest.main()

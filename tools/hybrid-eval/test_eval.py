"""Bounded generator, scorer, routing and bundle contracts; no native model needed."""
import hashlib
import json
from pathlib import Path
import re
import tarfile
import tempfile
import unittest

from hybrid_eval import (FILLER, TASK_TYPES, Refusal, bundle, check_route, comparison, first_difference,
                         generate, score, task_spec)


class EvalTests(unittest.TestCase):
    # Every seeded record set has an independently interpreted oracle, never a filler-derived answer.
    def test_constructed_answers(self):
        for seed in range(12):
            for kind in TASK_TYPES:
                spec = task_spec(kind, seed, 0)
                records = spec['records']
                if kind in ('single_needle', 'multi_key'):
                    facts = dict(line.removeprefix('FACT ').split(' = ') for line in records)
                    key = re.search(r'value of (\w+)', spec['question'])[1]
                    expected = facts[key]
                elif kind == 'multi_value':
                    expected = ','.join(line.split(' += ')[1] for line in records)
                elif kind == 'variable_tracking':
                    state = {}
                    for line in records:
                        op, name, _, value = line.split()
                        state[name] = state[value] if op == 'COPY' else value
                    target = re.search(r'hold (\d+)', spec['question'])[1]
                    expected = ','.join(sorted(name for name, value in state.items() if value == target))
                else:
                    expected = str(sum(int(line.split()[-1]) for line in records))
                self.assertEqual(spec['answer'], expected)
                self.assertTrue(all(part not in FILLER for part in expected.split(',')))
                self.assertEqual(spec, task_spec(kind, seed, 0))
        self.assertNotEqual(task_spec('single_needle', 1, 0), task_spec('single_needle', 2, 0))

    # A stand-in character tokenizer exercises fitting and record placement; native tokenizer checks are separate.
    def test_deterministic_depth_and_position(self):
        for kind in TASK_TYPES:
            for depth in (2048, 4096, 8192):
                early = generate(kind, 17, 0, depth, 0.1, list)
                self.assertEqual(early, generate(kind, 17, 0, depth, 0.1, list))
                self.assertLessEqual(early['material_tokens'], depth)
                self.assertGreaterEqual(early['material_tokens'], depth - 8)
                self.assertEqual(early['material_tokens'], len(early['material']))
                self.assertTrue(all(record in early['material'] for record in early['records']))
        early = generate('single_needle', 17, 0, 2048, 0.1, list)
        late = generate('single_needle', 17, 0, 2048, 0.9, list)
        self.assertLess(early['record_token_offsets'][0], late['record_token_offsets'][0])
        for depth, position in ((0, 0.1), (1048577, 0.1), (2048, -0.1), (2048, 1.1)):
            with self.assertRaises(Refusal):
                generate('single_needle', 17, 0, depth, position, list)

    # Full-answer scoring accepts exact scalars/lists and rejects wrong, partial, duplicate, extra and filler-only output.
    def test_exact_scoring_negative_controls(self):
        self.assertTrue(score('123456', '\n123456\n'))
        self.assertTrue(score('123456,234567', ' 123456, 234567 '))
        for answer in ('123455', 'The answer is 123456', '123456 234567', '', FILLER.strip()):
            self.assertFalse(score('123456', answer))
        for answer in ('123456', '234567,123456', '123456,234567,234567', '123456,234567,345678'):
            self.assertFalse(score('123456,234567', answer))
        self.assertFalse(score('123456', 'quiet'))

    # Actual token comparison detects substitutions and strict prefixes in either direction, independently of answer scoring.
    def test_token_difference(self):
        for left, right, expected in (([1, 2], [1, 2], None), ([1, 2], [1, 3], 1),
                                      ([1], [1, 2], 1), ([1, 2], [1], 1), ([], [1], 0)):
            self.assertEqual(first_difference(left, right), expected)
        local = dict(answer='123456', answer_tokens=[1, 2], answer_prompt_sha256='a')
        other = dict(answer=' 123456 ', answer_tokens=[1, 3], answer_prompt_sha256='b')
        self.assertEqual(comparison(local, other), dict(answer_agreement=True, first_differing_token=1, prompt_identical=False))

    # The routing gate accepts a genuine imported ingest/local continuation and rejects fallback, missing import and lost prefix.
    def test_patchwork_route_controls(self):
        trace = dict(mode='hybrid', path='offload', prefilled=2048, new_tokens=2048, replayed_prefix=0,
                     rollback_n=None, depth_before=0, prompt_depth=2048, depth_after=2050, generated_tokens=2,
                     remote_applied=2047, server_prefill_ms=1, wire_bytes=100, fallback_reason=None)
        check_route(trace, 0, 'hybrid', 'ingest')
        for changes in (dict(remote_applied=0), dict(fallback_reason='http_503'), dict(depth_before=1),
                        dict(prefilled=3000), dict(replayed_prefix=1)):
            with self.assertRaises(Refusal):
                check_route(dict(trace, **changes), 0, 'hybrid', 'ingest')
        answer = dict(trace, mode='local', path='local', depth_before=2050, prompt_depth=2070,
                      depth_after=2072, prefilled=20, new_tokens=20, remote_applied=0)
        check_route(answer, 2050, 'local', 'answer')
        with self.assertRaises(Refusal):
            check_route(dict(answer, remote_applied=1), 2050, 'local', 'answer')

    # Only public allowlisted metadata enters the archive; each member and the archive have verified SHA-256 digests.
    def test_sanitized_bundle(self):
        info = dict(model_sha256='a' * 64, quantization_type='Q4_0', proxy_url='private-address',
                    client_credential='secret-canary', model_path='/private/model', build_record={'path': 'private-build'})
        rows = [dict(stage='summary', status='executed', exit_code=0)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = bundle(root, 'fixture', rows, info, info)
            with tarfile.open(path) as archive:
                content = {name: archive.extractfile(name).read() for name in archive.getnames()}
            self.assertEqual(set(content), {'fixture.jsonl', 'fixture.md', 'client-info.json', 'node-info.json', 'SHA256SUMS'})
            for line in content['SHA256SUMS'].decode().splitlines():
                checksum, name = line.split('  ')
                self.assertEqual(checksum, hashlib.sha256(content[name]).hexdigest())
            public = b''.join(content.values())
            for forbidden in (b'secret-canary', b'private-address', b'/private/model', b'private-build'):
                self.assertNotIn(forbidden, public)
            self.assertEqual((root / (path.name + '.sha256')).read_text().split()[0], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()

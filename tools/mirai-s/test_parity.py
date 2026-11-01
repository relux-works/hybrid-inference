"""CPU contract tests for the parity gate; no model, GPU or remote endpoint."""
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import parity


def fixture(side):
    return dict(side=side, items=[dict(name=name, n_predict=count, prompt=[1] * (int(name[4:]) if name.startswith('long') else 4),
                                     tokens=[2, 3], top=[]) for name, count in parity.CASES])


class ParityTests(unittest.TestCase):
    def compare(self, a, b):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            paths = [Path(directory) / name for name in ('a.json', 'b.json')]
            for path, data in zip(paths, (a, b)):
                path.write_text(json.dumps(data))
            result = parity.compare(argparse.Namespace(a=paths[0], b=paths[1]))
            return result, output.getvalue()

    # Matching nonempty completions pass; token and prompt mismatches fail, including unequal generation lengths.
    def test_compare_tokens(self):
        a = fixture('fork')
        self.assertEqual(self.compare(a, fixture('port'))[0], 0)
        for field, value, diagnostic in [('tokens', [2, 4], 'diverge at generated token 1'),
                                         ('tokens', [2], 'diverge at generated token 1'),
                                         ('prompt', [9], 'PROMPT TOKENS DIFFER')]:
            with self.subTest(field=field, value=value):
                b = fixture('port')
                b['items'][0][field] = value
                result, output = self.compare(a, b)
                self.assertEqual(result, 1)
                self.assertIn(diagnostic, output)

    # Equal empty, missing, reordered or malformed evidence must never qualify parity.
    def test_incomplete_evidence(self):
        cases = [[], fixture('fork')['items'][:-1], fixture('fork')['items'][::-1]]
        for field, value in [('tokens', []), ('tokens', None), ('tokens', [True]), ('tokens', [-1]),
                             ('tokens', [1] * 49), ('prompt', []), ('n_predict', 47), ('name', 'unrelated')]:
            items = copy.deepcopy(fixture('fork')['items'])
            items[0][field] = value
            cases.append(items)
        items = fixture('fork')['items']
        items[-1]['prompt'] = [1]
        cases.append(items)
        for items in cases:
            with self.subTest(items=len(items)), self.assertRaises(ValueError):
                self.compare(dict(side='fork', items=items), dict(side='port', items=items))

    # Reuse sends the fork's exact token IDs and greedy settings, even when the port tokenizer differs.
    def test_collect_reuses_tokens(self):
        data = fixture('fork')
        data['items'][0]['text'] = 'text'
        completions = []

        def http(url, body):
            if url.endswith('/tokenize'):
                return dict(tokens=[99])
            completions.append(body)
            return dict(tokens=[2, 3])

        with tempfile.TemporaryDirectory() as directory, patch.object(parity, 'http', side_effect=http):
            reuse, out = Path(directory) / 'fork.json', Path(directory) / 'port.json'
            reuse.write_text(json.dumps(data))
            with contextlib.redirect_stdout(io.StringIO()):
                parity.collect(argparse.Namespace(reuse=reuse, url='http://fixture', side='port', out=out))
            result = json.loads(out.read_text())
            self.assertFalse(result['items'][0]['own_tokenization_equal'])
            for request, item in zip(completions, data['items'], strict=True):
                self.assertEqual(request, dict(prompt=item['prompt'], n_predict=item['n_predict'], temperature=0, top_k=1,
                                               seed=0, cache_prompt=False, n_probs=5, return_tokens=True))
            self.assertEqual(self.compare(data, result)[0], 0)

    # Fresh collection fixes the eight prompts and both book depths; a short book or missing output cannot pass.
    def test_collect_fresh_and_refuse_missing_tokens(self):
        for book_tokens, completion, valid in [(12288, [7], True), (4095, [7], False), (12288, None, False)]:
            with self.subTest(book_tokens=book_tokens, completion=completion), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'book.txt').write_text('book')
                args = argparse.Namespace(reuse=None, text_dir=root, url='http://fixture', side='fork', out=root / 'out.json')
                with patch.object(parity, 'tokenize', side_effect=lambda url, text: [1] * (book_tokens if text == 'book' else 4)), \
                     patch.object(parity, 'generate', return_value=dict(tokens=completion)), contextlib.redirect_stdout(io.StringIO()):
                    if valid:
                        parity.collect(args)
                        parity.validate(json.loads(args.out.read_text())['items'])
                    else:
                        with self.assertRaises(ValueError):
                            parity.collect(args)
                        self.assertFalse(args.out.exists())

    # Execute only the documented admission block with a CPU stub; unsupported/unknown GPUs stop before a build marker.
    def test_gpu_capability_guard(self):
        report = (Path(__file__).parents[1] / 'hybrid-bench/RESULTS-hybrid-c-r2-261101.md').read_text()
        script = report.split('```bash\n', 1)[1].split('```', 1)[0]
        guard = script[script.index('# Reject unsupported'):script.index('git rev-parse HEAD')]
        self.assertLess(script.index('# Reject unsupported'), script.index('cmake -S'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / 'nvidia-smi'
            stub.write_text('#!/bin/sh\nprintf "%s" "$CAPS"\nexit "$QUERY_EXIT"\n')
            stub.chmod(0o755)
            for caps, query_exit, admitted in [('8.0\n', 0, True), ('8.6\n9.0\n', 0, True), ('7.5\n', 0, False),
                                               ('8.0\n7.5\n', 0, False), ('', 0, False), ('N/A\n', 0, False), ('8.0\n', 1, False)]:
                with self.subTest(caps=caps, query_exit=query_exit):
                    env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'], STUDY_HOME=str(root),
                               CAPS=caps, QUERY_EXIT=str(query_exit))
                    result = subprocess.run(['bash', '-euc', guard + '\necho BUILD_ADMITTED'], env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0 if admitted else 1)
                    self.assertEqual('BUILD_ADMITTED' in result.stdout, admitted)
                    if not admitted:
                        self.assertIn('STOP: ', result.stderr)

    # The real CLI and documented stop block reject divergence/incomplete evidence before an arm marker; valid evidence passes.
    def test_gpu_parity_guard(self):
        report = (Path(__file__).parents[1] / 'hybrid-bench/RESULTS-hybrid-c-r2-261101.md').read_text()
        script = report.split('```bash\n', 1)[1].split('```', 1)[0]
        start = script.index('if ! python3 "$PARITY" compare')
        end = script.index('nice -n 10 "$BUILD/bin/llama-server" -m "$D_MODEL"', start)
        self.assertLess(start, script.index('--label A-Mirai-local'))
        self.assertLess(start, script.index('--label "B-Mirai-$node-imported"'))
        guard = script[start:end]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'parity-fork.json').write_text(json.dumps(fixture('fork')))
            for mode in ('equal', 'diverged', 'empty'):
                with self.subTest(mode=mode):
                    port = fixture('port')
                    if mode == 'diverged':
                        port['items'][0]['tokens'][1] = 99
                    elif mode == 'empty':
                        port['items'] = []
                    (root / 'parity-port.json').write_text(json.dumps(port))
                    env = dict(os.environ, STUDY_HOME=str(root), PARITY=str(Path(parity.__file__).resolve()))
                    result = subprocess.run(['bash', '-euc', guard + '\necho ARM_ADMITTED'], env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0 if mode == 'equal' else 1)
                    self.assertEqual('ARM_ADMITTED' in result.stdout, mode == 'equal')
                    if mode != 'equal':
                        self.assertIn('STOP: Mirai parity failed', result.stderr)


if __name__ == '__main__':
    unittest.main()

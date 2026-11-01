"""CPU-only study contract tests without models or document downloads."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import hetero_study as study


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs = self.root / 'docs'
        self.docs.mkdir(mode=0o700)
        self.book = self.docs / 'book.txt'
        self.book.write_text('Synthetic book. ' * 100)
        self.model = self.root / 'model.gguf'
        self.model.write_bytes(b'unit tokenizer identity')
        self.manifest = self.root / 'manifest.json'
        self.rows = [dict(id='book', url='https://www.gutenberg.org/cache/epub/84/pg84.txt',
                          sha256=study.sha256(self.book), size=self.book.stat().st_size)]
        self.manifest.write_text(json.dumps(dict(documents=self.rows)))
        self.args = argparse.Namespace(manifest=str(self.manifest), documents=str(self.docs), model=str(self.model),
                                       output=str(self.root / 'prepared'), tokenizer_url='http://127.0.0.1:8080', d=1, n=64, m=16, timeout=2)

    def tokenizer(self, base, path, data=None, **kwargs):
        return {'model_path': str(self.model)} if path == '/props' else {'tokens': list(range(100))}

    # Hash and length checks refuse their own invalid inputs; valid local sources produce private exact N/M fixtures.
    def test_prepare_hash_short_and_private_control(self):
        with patch.object(study, 'api', side_effect=self.tokenizer):
            study.prepare(self.args)
        root = Path(self.args.output)
        index, fixtures = study.load_study(root)
        self.assertEqual(fixtures['book'], dict(tokens=list(range(64)), continuation=list(range(64, 80))))
        self.assertEqual((root / 'index.json').stat().st_mode & 0o777, 0o600)
        self.assertEqual(Path(index['documents'][0]['fixture']['path']).stat().st_mode & 0o777, 0o600)
        self.args.output = str(self.root / 'short')
        with patch.object(study, 'api', return_value={'model_path': str(self.model), 'tokens': [1] * 79}):
            with self.assertRaisesRegex(study.Refusal, r'document shorter than N\+M'):
                study.prepare(self.args)
        self.assertFalse((Path(self.args.output) / 'index.json').exists())
        self.book.write_text('tampered')
        self.args.output = str(self.root / 'mismatch')
        with patch.object(study, 'api') as api:
            with self.assertRaisesRegex(study.Refusal, 'manifest hash mismatch'):
                study.prepare(self.args)
            api.assert_not_called()
        self.assertFalse((Path(self.args.output) / 'index.json').exists())

    # A changed fixture cannot be resumed or reported, even when its JSON and N/M remain valid.
    def test_changed_fixture_refuses_before_report(self):
        with patch.object(study, 'api', side_effect=self.tokenizer):
            study.prepare(self.args)
        root = Path(self.args.output)
        index = study.read(root / 'index.json')
        fixture = Path(index['documents'][0]['fixture']['path'])
        value = study.read(fixture)
        value['tokens'][0] += 1
        study.save(fixture, value)
        with self.assertRaisesRegex(study.Refusal, 'stale fixture: hash mismatch'):
            study.prepare(self.args)
        output = self.root / 'report'
        with self.assertRaisesRegex(study.Refusal, 'stale fixture: hash mismatch'):
            study.report(argparse.Namespace(study=str(root), arms=['local'], seed=3, output=str(output)))
        self.assertFalse(output.exists())

    # Percentile interpolation and seeded document resampling match an independent deterministic calculation.
    def test_bootstrap_and_quantiles(self):
        import random
        values = [0, 1, 2, 4, 8, 16, 32]
        rng = random.Random(91)
        samples = sorted(sum(rng.choices(values, k=7)) / 7 for _ in range(10000))
        def percentile(p):
            position = 9999 * p
            i = int(position)
            return samples[i] + (samples[i + 1] - samples[i]) * (position - i)
        expected = (percentile(.025), percentile(.975))
        self.assertEqual(study.bootstrap(values, 91), expected)
        self.assertEqual(study.bootstrap(values, 91), study.bootstrap(values, 91))
        self.assertEqual(study.quantile([0, 10], .9), 9)
        self.assertEqual(study.bootstrap([7], 2), (7, 7))

    # The shared lock excludes another invocation while preserving its owner's ability to release it.
    def test_process_lock(self):
        lock = self.root / 'gpu.lock'
        with study.locked(lock):
            with self.assertRaisesRegex(study.Refusal, 'process lock busy'):
                with study.locked(lock):
                    self.fail('concurrent invocation admitted')
        with study.locked(lock):
            pass

    # An inherited lock survives closure of the parent handle; cancellation reaps only the owned child.
    def test_child_lock_and_cleanup(self):
        lock = self.root / 'inherited.lock'
        with self.assertRaisesRegex(study.Refusal, 'cancel fixture'):
            with study.locked(lock) as fd:
                with patch.object(study, 'ACTIVE_LOCK_FD', fd):
                    manager = study.child([sys.executable, '-c', 'import signal; signal.pause()'], self.root, 'owned')
                    process = manager.__enter__()
            try:
                with self.assertRaisesRegex(study.Refusal, 'process lock busy'):
                    with study.locked(lock):
                        self.fail('orphan reservation released')
                raise study.Refusal('cancel fixture')
            finally:
                manager.__exit__(None, None, None)
        self.assertIsNotNone(process.poll())
        self.assertEqual(study.read(self.root / 'owned.process.json')['pid'], process.pid)
        with study.locked(lock):
            pass

    # Curl pinning commits no new files on a later mismatch; this transport stub performs no downloads.
    def test_fetch_pins_and_refuses_changed_bytes(self):
        sources = self.root / 'urls.json'
        sources.write_text(json.dumps(dict(documents=[dict(id=r['id'], url=r['url']) for r in self.rows])))
        args = argparse.Namespace(manifest=str(sources), output=str(self.root / 'download'))
        def curl(command, *unused):
            Path(command[command.index('--output') + 1]).write_bytes(self.book.read_bytes())
        with patch.object(study, 'run', side_effect=curl):
            study.fetch(args)
            study.fetch(args)
        root = Path(args.output)
        original = (root / 'manifest.json').read_bytes()
        def changed(command, *unused):
            Path(command[command.index('--output') + 1]).write_bytes(b'changed remote bytes')
        with patch.object(study, 'run', side_effect=changed):
            with self.assertRaisesRegex(study.Refusal, 'manifest hash mismatch after fetch'):
                study.fetch(args)
        self.assertEqual((root / 'manifest.json').read_bytes(), original)
        self.assertEqual((root / 'book.txt').read_bytes(), self.book.read_bytes())
        changed_pins = [dict(r, sha256='0' * 64) for r in self.rows]
        sources.write_text(json.dumps(dict(documents=changed_pins)))
        with patch.object(study, 'run', side_effect=curl):
            with self.assertRaisesRegex(study.Refusal, 'fetch input pins changed'):
                study.fetch(args)
        self.assertEqual((root / 'manifest.json').read_bytes(), original)

    # Prepare cannot reach a public host, follow a redirect, or silently bind another tokenizer model.
    def test_tokenizer_identity_and_loopback(self):
        with self.assertRaisesRegex(study.Refusal, 'numeric loopback'):
            study.api('https://www.gutenberg.org', '/tokenize')
        with self.assertRaisesRegex(study.Refusal, 'redirect refused'):
            study.NoRedirect().redirect_request(None)
        with patch.object(study, 'api', return_value={'model_path': '/another/model.gguf'}):
            with self.assertRaisesRegex(study.Refusal, 'tokenizer server model mismatch'):
                study.prepare(self.args)
        self.assertFalse((Path(self.args.output) / 'index.json').exists())


if __name__ == '__main__':
    unittest.main()

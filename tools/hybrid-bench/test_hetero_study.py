"""CPU-only study contract tests without models or document downloads."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import struct
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

    def arm_fixture(self, keep=False):
        with patch.object(study, 'api', side_effect=self.tokenizer):
            study.prepare(self.args)
        root = Path(self.args.output)
        index, fixtures = study.load_study(root)
        fixture = fixtures['book']
        model = study.file_record(self.model)
        refs = study.private_dir(root / 'references')
        header = dict(fixture, model_sha256=model['sha256'], n=64, m=16, token_sha256=study.token_hash(fixture),
                      context_prefill='split-last-v1', schedule_probe='fresh-prefix-spots-v1', score_batch=4, format='f16-logsoftmax-le-v1', layout_fingerprint='a')
        payload = study.encoded(header)
        binary = refs / 'reference.bin'
        binary.write_bytes(b'HYBLOG01' + struct.pack('<I', len(payload)) + payload)
        result = refs / 'result.json'
        study.save(result, {})
        study.commit_receipt(refs / 'book.json', 'reference', dict(study.base_identity(index, index['documents'][0], fixture), model=model),
                             [binary, result], header=header)
        args = study.parser().parse_args(['arm', '--study', str(root), '--kind', 'imported', '--label', 'imported',
                                         '--client-model', str(self.model), '--node-model', str(self.model),
                                         '--reference-model-sha256', model['sha256'], '--score-batch', '4', '--chunk-tokens', '256'])
        args.keep_deltas = keep
        self.addCleanup(patch.stopall)
        patch.object(study, 'engine', return_value=dict(score_batch=4, build=[], helpers=[])).start()
        patch.object(study, 'build_identity', return_value=[]).start()
        patch.object(study, 'deltas', side_effect=self.fake_deltas).start()
        patch.object(study, 'score', side_effect=self.fake_score).start()
        return args

    def fake_deltas(self, args, spec, fixture, attempt, header):
        delta = attempt / '0.delta'
        delta.write_bytes(b'fixture delta')
        return [delta]

    def fake_score(self, args, model, fixture, attempt):
        data = dict(positions=[dict(position=p, hybrid_kl_nats=.125, hybrid_top1_agreed=True) for p in range(64, 80)],
                    schedule_floor=dict(kind='fresh-prefix-spots-v1', positions=[dict(hybrid_kl_nats=0, hybrid_top1_agreed=True)]))
        result = attempt / 'result.json'
        study.save(result, data)
        return result, data

    # Scoring retains hash/size bindings and rows; default deletion and debug retention both permit unchanged resume/report.
    def test_delta_retention_and_metadata_report(self):
        args = self.arm_fixture()
        study.arm(args)
        root = Path(args.study)
        record_path = root / 'arms/imported/book.json'
        record = study.receipt(record_path)
        self.assertEqual(record['deltas'][0]['size'], len(b'fixture delta'))
        self.assertEqual(record['deltas'][0]['sha256'], study.hashlib.sha256(b'fixture delta').hexdigest())
        self.assertFalse(Path(record['deltas'][0]['path']).exists())
        with patch.object(study, 'score') as score:
            study.arm(args)
            score.assert_not_called()
        args.label = 'kept'
        args.keep_deltas = True
        study.arm(args)
        kept = study.receipt(root / 'arms/kept/book.json')
        self.assertTrue(Path(kept['deltas'][0]['path']).is_file())
        allowed = {Path(index['fixture']['path']) for index in study.read(root / 'index.json')['documents']}
        for item in (record, kept):
            allowed.update(Path(out['path']) for out in item['outputs'])
        allowed.add(root / 'references/result.json')
        original_hash = study.sha256
        def bounded_hash(path):
            self.assertIn(Path(path), allowed, 'report read a large/deleted input')
            return original_hash(path)
        self.model.unlink()
        self.book.unlink()
        self.manifest.unlink()
        (root / 'references/reference.bin').unlink()
        with patch.object(study, 'sha256', side_effect=bounded_hash):
            output = self.root / 'report'
            study.report(argparse.Namespace(study=str(root), arms=['imported', 'kept'], seed=3, output=str(output)))
        self.assertIn('no large input revalidation', study.read(output / 'report.json')['settings']['verification'])
        rows = Path(record['outputs'][0]['path'])
        rows.write_bytes(b'[]')
        with self.assertRaisesRegex(study.Refusal, 'stale arm rows/delta'):
            study.report(argparse.Namespace(study=str(root), arms=['imported'], seed=3, output=str(self.root / 'bad-report')))
        self.assertFalse((self.root / 'bad-report').exists())

    # Failed generation/scoring and cancellation remove this attempt's deltas even with --keep-deltas, preserving foreign files.
    def test_failed_attempt_delta_cleanup(self):
        args = self.arm_fixture(keep=True)
        root = Path(args.study)
        foreign = root / 'foreign.delta'
        foreign.write_bytes(b'foreign')
        for where, error in [('generation', study.Refusal('extend fixture')), ('scoring', study.Refusal('score fixture')),
                             ('interruption', KeyboardInterrupt())]:
            args.label = where
            def fail_deltas(*values):
                self.fake_deltas(*values)
                raise error
            target = 'deltas' if where == 'generation' else 'score'
            side_effect = fail_deltas if target == 'deltas' else error
            with patch.object(study, target, side_effect=side_effect):
                with self.assertRaises(type(error)):
                    study.arm(args)
            folder = root / 'arms' / where
            self.assertEqual(list(folder.rglob('*.delta')), [])
            self.assertFalse((folder / 'book.json').exists())
        self.assertEqual(foreign.read_bytes(), b'foreign')

    # Disk admission includes bounded chain bytes, retained documents and margin; one byte below refuses before deltas/score.
    def test_arm_disk_budget(self):
        args = self.arm_fixture()
        fixture = dict(tokens=[0] * 32768)
        pending = [(None, fixture)] * 20
        spec = dict(kind='imported', label='budget')
        args.chunk_tokens = 4096
        usage = study.shutil.disk_usage(self.root)
        with patch.object(study.shutil, 'disk_usage', return_value=usage._replace(free=10**15)):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                budget, margin = study.arm_disk_budget(args, spec, pending, self.root)
            self.assertEqual(budget, 8 * study.API_LIMIT + 20 * 100 * 1024**2)
            self.assertEqual(study.json.loads(output.getvalue())['budget_bytes'], budget)
            args.keep_deltas = True
            kept, _ = study.arm_disk_budget(args, spec, pending, self.root)
            self.assertEqual(kept, 20 * 8 * study.API_LIMIT + 20 * 100 * 1024**2)
        args.keep_deltas = False
        with patch.object(study.shutil, 'disk_usage', return_value=usage._replace(free=budget + margin)):
            study.arm_disk_budget(args, spec, pending, self.root)
        with patch.object(study.shutil, 'disk_usage', return_value=usage._replace(free=budget + margin - 1)):
            with self.assertRaisesRegex(study.Refusal, 'insufficient free disk'):
                study.arm_disk_budget(args, spec, pending, self.root)
        with patch.object(study.shutil, 'disk_usage', return_value=usage._replace(free=0)):
            with patch.object(study, 'deltas') as deltas, patch.object(study, 'score') as score:
                with self.assertRaisesRegex(study.Refusal, 'insufficient free disk'):
                    study.arm(args)
                deltas.assert_not_called()
                score.assert_not_called()

    # Unaligned chunk sizes refuse before any work; aligned boundary sizes reach the arm handler.
    def test_chunk_alignment(self):
        args = ['--lock', str(self.root / 'lock'), 'arm', '--study', str(self.root), '--reference-model-sha256', 'a' * 64]
        for size in (0, 32, 255, 257, 8193):
            with patch.object(study, 'arm') as arm:
                self.assertEqual(study.main(args + ['--chunk-tokens', str(size)]), 2)
                arm.assert_not_called()
        for size in (256, 4096, 8192):
            with patch.object(study, 'arm') as arm:
                self.assertEqual(study.main(args + ['--chunk-tokens', str(size)]), 0)
                arm.assert_called_once()

    # Session-release failures preserve the earlier extend failure; a release-only failure still refuses the attempt.
    def test_delete_preserves_extend_failure(self):
        @contextlib.contextmanager
        def node(*unused):
            yield 'http://127.0.0.1:1', 'fixture', dict(layout_fingerprint='a')
        def api(base, path, *unused, **kwargs):
            if path == '/v1/sessions':
                return dict(session_id='fixture')
            if kwargs.get('method') == 'DELETE':
                raise study.Refusal('release fixture')
            raise study.Refusal('extend fixture')
        with patch.object(study, 'node', node), patch.object(study, 'api', side_effect=api):
            with self.assertRaisesRegex(study.Refusal, 'extend fixture'):
                study.deltas(argparse.Namespace(chunk_tokens=256, timeout=1), dict(node_model=dict(sha256='a')), dict(tokens=[1, 2]), self.root, dict(layout_fingerprint='a'))
            with self.assertRaisesRegex(study.Refusal, 'release fixture'):
                study.deltas(argparse.Namespace(chunk_tokens=256, timeout=1), dict(node_model=dict(sha256='a')), dict(tokens=[1]), self.root, dict(layout_fingerprint='a'))

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

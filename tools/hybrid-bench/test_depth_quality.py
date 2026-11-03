"""Depth quality contract tests; native cases opt in with DEPTH_BINARY and DEPTH_MODEL."""
import copy
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import depth_quality as depth
import hetero_study as study
import stage_depth_quality as stage


class DepthContracts(unittest.TestCase):
    # Exact scoring accepts whitespace but rejects a wrong value, explanatory text and a substring.
    def test_needle_exact_scoring(self):
        self.assertTrue(depth.exact_score(' \nAB1234\n', 'AB1234'))
        for answer in ('AB1235', 'The value is AB1234', 'AB1234X', ''):
            self.assertFalse(depth.exact_score(answer, 'AB1234'))

    # Seeds deterministically bind values; a different seed changes the document's records.
    def test_needle_seed(self):
        self.assertEqual(depth.needle_entries(42, 16), depth.needle_entries(42, 16))
        self.assertNotEqual(depth.needle_entries(42, 16), depth.needle_entries(43, 16))
        self.assertEqual(len({x['answer'] for x in depth.needle_entries(42, 16)}), 16)

    # A planted wrong document or YaRN binding is rejected by the report gate, alongside a valid control.
    def test_mismatched_yarn_setting_refuses(self):
        self.check_binding_mutation('settings')

    def test_changed_document_between_arms_refuses(self):
        self.check_binding_mutation('tokens_sha256')

    def check_binding_mutation(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'logits'
            path.write_bytes(struct.pack('<2f', 0, 1))
            identity = dict(tokens_sha256='a', settings=depth.settings(524288), needles=[])
            record = dict(binding=copy.deepcopy(identity), rows=1, vocab=2, logits_sha256=study.sha256(path), answers=[])
            depth.verify_arm(record, identity, path)
            record['binding'][key] = {} if key == 'settings' else 'changed'
            with self.assertRaisesRegex(study.Refusal, 'document or YaRN mismatch'):
                depth.verify_arm(record, identity, path)

    # Context settings match the model-card depth recommendations, and estimates expose their assumptions.
    def test_yarn_and_estimate_accounting(self):
        self.assertEqual(depth.settings(524288)['yarn_factor'], 2)
        self.assertEqual(depth.settings(1040000)['yarn_factor'], 4)
        for tokens in (524288, 1040000):
            plan = stage.estimates(tokens)
            for row in plan['scenarios']:
                self.assertEqual(row['prefill_seconds'], 6*tokens/row['prefill_tokens_per_second'])
                self.assertAlmostEqual(row['cost_usd'], row['gpu_hours']*1.57)

    # The single-state reservation refuses a pre-existing file and never deletes it.
    def test_stale_state_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root/'context.state'
            state.write_bytes(b'foreign')
            with self.assertRaisesRegex(study.Refusal, 'stale state'):
                with depth.single_state(root):
                    self.fail('entered stale state')
            self.assertEqual(state.read_bytes(), b'foreign')

    # Archival verifies its own member hashes and keeps the sole raw logits outside the tar.
    def test_archive_without_state_or_logit_duplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'report.json').write_text('{}')
            (root/'offload.f32').write_bytes(b'logits')
            stage.archive_case(root)
            with stage.tarfile.open(root/'results.tar.gz') as bundle:
                self.assertEqual(set(bundle.getnames()), {'report.json', 'SHA256SUMS'})
            self.assertEqual(study.read(root/'archive.json')['logits']['offload.f32'], study.sha256(root/'offload.f32'))

    # A deliberately wrong member hash must fail verification and cannot publish an archive seal.
    def test_archive_checksum_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root/'report.json'
            report.write_text('{}')
            original = stage.sha
            with patch.object(stage, 'sha', side_effect=lambda path: '0'*64 if Path(path) == report else original(path)):
                with self.assertRaisesRegex(RuntimeError, 'archive checksum differs'):
                    stage.archive_case(root)
            self.assertFalse((root/'archive.json').exists())


@unittest.skipUnless(os.environ.get('DEPTH_BINARY') and os.environ.get('DEPTH_MODEL'), 'set local native binary and small model; no downloads')
class NativeDepth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ['DEPTH_BINARY']).resolve()
        cls.model = Path(os.environ['DEPTH_MODEL']).resolve()
        cls.root = study.private_dir(os.environ['DEPTH_ARTIFACTS'])
        cls.serial = 0

    def command(self, job, expected=0, reason=None):
        type(self).serial += 1
        prefix = 'native-' + str(self.serial)
        path = self.root/(prefix+'.json')
        study.save(path, job)
        with study.child([self.binary, path], self.root, prefix) as process:
            code = process.wait(timeout=600)
        self.assertEqual(code, expected, (self.root/(prefix+'.log')).read_text()[-3000:])
        if reason:
            self.assertIn(reason, (self.root/(prefix+'.log')).read_text())

    # Independently known distributions prove KL direction, tie handling, raw maxima, shifts and disagreement.
    def test_kl_top1_arithmetic(self):
        cases = [([math.log(.75),math.log(.25)], [math.log(.5),math.log(.5)], .75*math.log(1.5)+.25*math.log(.5), True),
                 ([0,1],[2,-1],None,False), ([0,1],[1000,1001],0,True)]
        for i,(a,b,expected,agrees) in enumerate(cases):
            left, right, output = [self.root/f'arithmetic-{i}-{suffix}' for suffix in ('a','b','out')]
            left.write_bytes(struct.pack('<2f',*a))
            right.write_bytes(struct.pack('<2f',*b))
            self.command(dict(mode='compare', a=str(left), b=str(right), rows=1,vocab=2,output=str(output)))
            result = study.read(output)
            if expected is not None:
                self.assertAlmostEqual(result['a_to_b_kl_nats'], expected, places=6)
            self.assertEqual(result['top1_agreed'], int(agrees))
            self.assertAlmostEqual(result['max_abs_logit_difference'], max(abs(x-y) for x,y in zip(a,b)),places=6)
            if i == 0:
                self.assertAlmostEqual(result['b_to_a_kl_nats'], .5*math.log(2/3)+.5*math.log(2),places=6)

    # Truncation and nonfinite values fail for the intended reason and publish no comparison.
    def test_invalid_logits_refuse(self):
        left, right = self.root/'bad-a', self.root/'bad-b'
        right.write_bytes(struct.pack('<2f',0,1))
        for label, value, reason in [('truncated',b'x','truncated or trailing'),('nan',struct.pack('<2f',float('nan'),0),'nonfinite logit')]:
            left.write_bytes(value)
            output = self.root/('bad-'+label+'.json')
            self.command(dict(mode='compare',a=str(left),b=str(right),rows=1,vocab=2,output=str(output)),2,reason)
            self.assertFalse(output.exists())

    # The optional after-window keeps the needle checkpoint at the stated depth, before the held-out text.
    def test_after_window_saved_state(self):
        root = study.private_dir(self.root/'after-window')
        text = root/'source.txt'
        text.write_text('The river passes a quiet garden and a wooden gate. '*32)
        fixture = depth.prepare(self.binary,self.model,text,dict(id='synthetic-cpu-after',sha256=study.sha256(text)),root,128,8,43,1,600,'after')
        report = depth.run_case(self.binary,self.model,fixture,root,0,8,7,600)
        self.assertEqual(report['position_range'],[128,135])
        for arm in depth.ARMS:
            self.assertEqual(study.read(root/(arm+'.json'))['snapshot_depth'],128)
        self.assertFalse((root/'context.state').exists())

    # Real CPU tokenization, three sequential arms, native file import, tail logits and restored queries execute end to end.
    def test_pipeline_end_to_end_and_native_identity_negatives(self):
        root = study.private_dir(self.root/'pipeline')
        text = root/'public-fixture.txt'
        text.write_text('The river passes the old bridge and a quiet garden. '*128)
        source = dict(id='synthetic-cpu-only',sha256=study.sha256(text),size=text.stat().st_size)
        fixture_path = depth.prepare(self.binary,self.model,text,source,root,512,8,42,2,600)
        fixture = study.read(fixture_path)
        identity = depth.validate_fixture(fixture)
        positions = [n['position'] for n in fixture['needles']]
        self.assertLess(positions[0],256)
        self.assertGreater(positions[1],256)
        mutated = copy.deepcopy(fixture)
        mutated['tokens'][positions[0]] += 1
        with self.assertRaisesRegex(study.Refusal,'needle placement mismatch'):
            depth.validate_fixture(mutated)
        report = depth.run_case(self.binary,self.model,fixture_path,root,0,8,7,600,256)
        self.assertEqual(report['context_depth'],512)
        self.assertEqual(report['position_range'],[504,511])
        self.assertEqual(study.read(root/'offload.json')['snapshot_depth'],512)
        self.assertEqual(study.read(root/'offload.json')['imported_transfer']['chunk_count'],2)
        self.assertEqual(len(report['comparisons']),3)
        self.assertEqual(len(report['needles']),2)
        for comparison in report['comparisons'].values():
            self.assertEqual(len(comparison['positions']),8)
            self.assertTrue(math.isfinite(comparison['a_to_b_kl_nats']))
        self.assertFalse((root/'context.state').exists())
        ledgers = [study.read(p) for p in root.glob('*.process.json')]
        self.assertTrue(ledgers and all(p['exit_code'] == 0 for p in ledgers))
        # This unchanged native state is the positive control for both import rejection plants.
        state = root/'negative.state'
        base = dict(model=str(self.model),model_sha256=fixture['model_sha256'],gpu_layers=0,fixture=str(fixture_path),
                    binding=identity,state=str(state),batch_size=8)
        self.command(dict(base,mode='export'))
        original_state = study.sha256(state)
        original_sidecar = Path(str(state)+'.json').read_bytes()
        try:
            for label in ('mismatched_yarn_setting_refuses','changed_document_between_arms_refuses'):
                changed = copy.deepcopy(fixture)
                if label.startswith('mismatched'):
                    changed['settings']['yarn_factor'] = 4
                else:
                    changed['tokens'][0] += 1
                path = root/(label+'.json')
                study.save(path,changed)
                output, logits = root/(label+'-out.json'),root/(label+'.f32')
                self.command(dict(base,mode='evaluate',fixture=str(path),binding=depth.binding(changed),
                    **{'import':True},output=str(output),logits=str(logits)),2,'state document or YaRN mismatch')
                self.assertFalse(output.exists())
                self.assertFalse(logits.exists())
                self.assertEqual(study.sha256(state),original_state)
                self.assertEqual(Path(str(state)+'.json').read_bytes(),original_sidecar)
        finally:
            state.unlink()
            Path(str(state)+'.json').unlink()


if __name__ == '__main__':
    unittest.main()

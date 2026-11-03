"""Depth quality contract tests; native cases opt in with DEPTH_BINARY and DEPTH_MODEL."""
import copy
import math
import gzip
import hashlib
import json
import argparse
import sys
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import depth_quality as depth
import hetero_study as study
import stage_depth_quality as stage
import depth_corpus as corpus


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

    # Both study depths use the same static factor; estimates expose their assumptions.
    def test_yarn_and_estimate_accounting(self):
        self.assertEqual(depth.settings(524288)['yarn_factor'], 4)
        self.assertEqual(depth.settings(1040000)['yarn_factor'], 4)
        for tokens in (524288, 1040000):
            plan = stage.estimates(tokens)
            for row in plan['scenarios']:
                self.assertEqual(row['prefill_seconds'], plan['documents']*3*row['prefill_seconds_per_arm'])
                self.assertAlmostEqual(row['cost_usd'], row['gpu_hours']*1.57)
            if tokens == 1040000:
                self.assertEqual([r['hours_per_document'] for r in plan['scenarios']], [1,1.5])
            self.assertEqual(plan['needle_restores_per_arm'],1)

    # Static factor four starts immediately above the original context; all other YaRN parameters stay fixed.
    def test_yarn_factor_512k_static_four(self):
        self.assertEqual(depth.settings(262144)['yarn_factor'], 2)
        for tokens in (262145, 524288, 1040000):
            settings = depth.settings(tokens)
            self.assertEqual(settings['yarn_factor'], 4)
            self.assertEqual(settings['original_context'], 262144)
            self.assertEqual(settings['rope_theta'], 10000000)
            self.assertEqual(settings['yarn_beta_fast'], 32)
            self.assertEqual(settings['yarn_beta_slow'], 1)
            self.assertEqual(settings['yarn_ext_factor'], 1)

    # Missing and null engine identities must refuse before fixture access or any native probe.
    def test_handoff_missing_engine_ref_refuses(self):
        self.check_unrecorded_engine({})

    def test_handoff_null_engine_ref_refuses(self):
        self.check_unrecorded_engine(dict(engine_ref=None))

    def check_unrecorded_engine(self, producer):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/'handoff.json'
            study.save(manifest, dict(schema='depth-slot-handoff-1', logits_dtype='float32-le',
                positions='forced inputs at depth..depth+63; logits predict depth+1..depth+64', rows=64, producer=producer))
            args = argparse.Namespace(handoff=manifest, output=root/'check', fixture=root/'absent.json')
            with patch.object(depth, 'invoke') as invoke:
                with self.assertRaisesRegex(study.Refusal, 'requires a recorded engine source revision'):
                    depth.verify_handoff(args)
                invoke.assert_not_called()
            self.assertFalse((args.output/'probe.job.json').exists())

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

    # An all-miss agreement must retain zero local accuracy and explicitly defer retrieval conclusions to logits.
    def test_all_missed_needles_do_not_prove_retrieval(self):
        needles=[dict(correct={a:False for a in depth.ARMS}) for _ in range(2)]
        result=depth.needle_summary(needles)
        self.assertEqual(result['local_needle_accuracy'],0)
        self.assertTrue(result['all_arms_missed_all_needles'])
        self.assertIn('rests on KL',result['needle_conclusion'])
        needles[0]['correct']['local']=True
        control=depth.needle_summary(needles)
        self.assertEqual(control['local_needle_accuracy'],.5)
        self.assertFalse(control['all_arms_missed_all_needles'])

    # Sixteen possible 40-token questions plus 32-token answers cannot fit in the old 512-token reserve.
    def test_sequential_context_budget_refuses_before_arm_work(self):
        fixture=dict(version=2,model_sha256='a'*64,window='final',context_depth=136,needle_mode='sequential',
            tokens=list(range(128)),tail=list(range(128,136)),probe_tokens=[1]*64,source=dict(id='cpu-fixture'),
            settings=dict(n_ctx=2048),needles=[dict(key=str(i),expected=str(i),position=i,inserted_tokens=[i],query=[1]*40,query_next=[1]*40) for i in range(16)])
        depth.validate_fixture(fixture)
        fixture['settings']['n_ctx']=768
        with self.assertRaisesRegex(study.Refusal,'exceeds reserved context'):
            depth.validate_fixture(fixture)

    # Shipped public pins cover three types at 512k and two or three at 1M, with no network access.
    def test_public_corpus_pins_and_document_units(self):
        rows=corpus.documents(Path(corpus.__file__).with_name('depth_corpus_pins.json'))
        self.assertEqual(len(corpus.select(rows,524288)),3)
        self.assertEqual(len(corpus.select(rows,1040000)),2)
        self.assertEqual(len(corpus.select(rows,1040000,3)),3)
        self.assertEqual({r['kind'] for r in rows},{'code','chat','prose'})

    # Fixture bytes exercise retrieval verification and deterministic English chat conversion, not real corpus downloads.
    def test_corpus_positive_and_hash_rejection(self):
        rows=copy.deepcopy(corpus.documents(Path(corpus.__file__).with_name('depth_corpus_pins.json')))
        tree=dict(prompt=dict(role='prompter',lang='en',text='Question',user_id='omit-this-metadata',replies=[
            dict(role='assistant',lang='fr',text='ignored',replies=[]),
            dict(role='assistant',lang='en',text='First English answer',replies=[])]))
        payloads=[b'int main() { return 0; }',gzip.compress((json.dumps(tree)+'\n').encode(),mtime=0),b'A quiet river passed a garden.']
        for row,data in zip(rows,payloads):
            row['files']=row['files'][:1]
            row['files'][0].update(sha256=hashlib.sha256(data).hexdigest(),size=len(data))
        licenses=[b'MIT License\n', b'---\nlicense: apache-2.0\n---\n', b'---\nlicense: cdla-sharing-1.0\n---\n']
        rows[0]['license'].update(sha256=hashlib.sha256(licenses[0]).hexdigest(),size=len(licenses[0]))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'pins.json';study.save(manifest,dict(schema='depth-corpus-1',documents=rows))
            def fetch_fixture(command,directory,name,timeout):
                idx=next(i for i,r in enumerate(rows) if r['id']==directory.name)
                Path(command[command.index('--output')+1]).write_bytes(licenses[idx] if name == 'license' else payloads[idx])
            with patch.object(corpus.study,'run',side_effect=fetch_fixture):
                records=corpus.fetch(manifest,root/'good')
            for record in records:
                evidence=record['license']
                self.assertEqual(study.sha256(Path(evidence['path'])), evidence['sha256'])
                self.assertEqual(study.read(Path(evidence['path']).parent/'license.json'), evidence)
            chat=Path(records[1]['path']).read_text()
            self.assertIn('User: Question\nAssistant: First English answer',chat)
            self.assertNotIn('omit-this-metadata',chat)
            self.assertNotIn('ignored',chat)
            self.assertTrue(all(r['observation_unit']=='one constructed document' for r in records))
            bad=copy.deepcopy(rows);bad[0]['files'][0]['sha256']='0'*64
            study.save(manifest,dict(schema='depth-corpus-1',documents=bad))
            with patch.object(corpus.study,'run',side_effect=fetch_fixture):
                with self.assertRaisesRegex(study.Refusal,'source SHA256/size mismatch'):
                    corpus.fetch(manifest,root/'bad')
            self.assertFalse((root/'bad/manifest.json').exists())

    # Missing provenance refuses before retrieval; a wrong downloaded declaration publishes no license receipt.
    def test_corpus_license_missing_or_changed_refuses(self):
        good=study.read(Path(corpus.__file__).with_name('depth_corpus_pins.json'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'pins.json'
            bad=copy.deepcopy(good); del bad['documents'][1]['license']; study.save(path,bad)
            with self.assertRaisesRegex(study.Refusal,'requires pinned source license declaration'):
                corpus.documents(path)
            def wrong_license(command,directory,name,timeout):
                Path(command[command.index('--output')+1]).write_bytes(b'---\nlicense: wrong-license\n---\n')
            with patch.object(corpus.study,'run',side_effect=wrong_license):
                with self.assertRaisesRegex(study.Refusal,'source license declaration differs'):
                    corpus.fetch_license(good['documents'][1],root)
            self.assertFalse((root/'license.json').exists())

    # Local/private URLs, mutable revisions and missing hashes are refused before any retrieval.
    def test_corpus_private_or_unpinned_source_refuses(self):
        good=study.read(Path(corpus.__file__).with_name('depth_corpus_pins.json'))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pins.json'
            for key,value in [('url','file:///home/owner/private.log'),('revision','main'),('sha256','')]:
                bad=copy.deepcopy(good);bad['documents'][0]['files'][0][key]=value;study.save(path,bad)
                with self.assertRaises(study.Refusal):
                    corpus.documents(path)


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
        fixture = depth.prepare(self.binary,self.model,text,dict(id='synthetic-cpu-after',sha256=study.sha256(text)),root,128,8,43,2,600,'after','independent')
        report = depth.run_case(self.binary,self.model,fixture,root,0,8,7,600)
        self.assertEqual(report['position_range'],[128,135])
        for arm in depth.ARMS:
            self.assertEqual(study.read(root/(arm+'.json'))['snapshot_depth'],128)
            self.assertEqual(study.read(root/(arm+'.json'))['state_restores'],3)
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
        callback=root/'capture-handoff.py'
        callback.write_text('#!'+sys.executable+'\nimport json,sys\nfrom pathlib import Path\np=Path(sys.argv[1]);m=json.loads(p.read_text())\nassert Path(m["state"]["path"]).is_file()\np.with_suffix(".captured.json").write_text("state existed during callback")\n')
        callback.chmod(0o700)
        publish = depth.publish_handoff
        def check_handoff(case, arm, record, fixture_path, state, command, timeout):
            publish(case,arm,record,fixture_path,state,command,timeout)
            if arm != 'local':
                return
            manifest_path = case/(arm+'-handoff.json')
            args = argparse.Namespace(output=study.private_dir(case/'mac-check'),handoff=manifest_path,
                model=self.model,fixture=fixture_path,state=state,reference=case/(arm+'-next64.f32'),
                binary=self.binary,gpu_layers=0,timeout=600)
            original = study.sha256(state)
            depth.verify_handoff(args)
            metrics = study.read(args.output/'comparison.json')
            self.assertEqual(metrics['a_to_b_kl_nats'],0)
            self.assertEqual(metrics['max_abs_logit_difference'],0)
            self.assertEqual(study.sha256(state),original)
            good = study.read(manifest_path)
            # Each rejected binding is checked before a native process or candidate output can be created.
            for label, mutate, reason in [
                ('yarn',lambda x:x['binding']['settings'].update(yarn_factor=4),'document, continuation or YaRN'),
                ('document',lambda x:x['binding'].update(tokens_sha256='0'*64),'document, continuation or YaRN'),
                ('positions',lambda x:x.update(positions='before forced inputs'),'position convention'),
                ('continuation',lambda x:x['continuation'].__setitem__(0,x['continuation'][0]+1),'document, continuation or YaRN'),
                ('state',lambda x:x['state'].update(sha256='0'*64),'state hash/size'),
                ('engine-missing',lambda x:x['producer'].pop('engine_ref'),'requires a recorded engine'),
                ('engine-null',lambda x:x['producer'].update(engine_ref=None),'requires a recorded engine'),
                ('engine',lambda x:x['producer'].update(engine_ref='0'*40),'engine source revision'),
                ('reference',lambda x:x['reference'].update(sha256='0'*64),'reference hash/size')]:
                changed=copy.deepcopy(good); mutate(changed)
                path=case/('bad-handoff-'+label+'.json');study.save(path,changed)
                bad=copy.copy(args);bad.handoff=path;bad.output=case/('bad-mac-'+label)
                with self.assertRaisesRegex(study.Refusal,reason):
                    depth.verify_handoff(bad)
                self.assertFalse((bad.output/'probe.job.json').exists())
                self.assertEqual(study.sha256(state),original)
        with patch.object(depth,'publish_handoff',side_effect=check_handoff):
            engine_ref = subprocess.check_output(['git', '-C', Path(depth.__file__).resolve().parents[2], 'rev-parse', 'HEAD'], text=True).strip()
            report = depth.run_case(self.binary,self.model,fixture_path,root,0,8,7,600,256,callback,engine_ref=engine_ref)
        self.assertEqual(report['context_depth'],512)
        self.assertEqual(report['position_range'],[504,511])
        self.assertEqual(study.read(root/'offload.json')['snapshot_depth'],512)
        self.assertEqual(study.read(root/'offload.json')['imported_transfer']['chunk_count'],2)
        self.assertEqual(len(report['comparisons']),3)
        self.assertEqual(len(report['needles']),2)
        self.assertEqual(report['state_restores'],{a:1 for a in depth.ARMS})
        self.assertTrue(all((root/(a+'-handoff.captured.json')).exists() for a in depth.ARMS))
        self.assertIn('local',report['needle_accuracy'])
        self.assertEqual(report['local_needle_accuracy'],report['needle_accuracy']['local']['accuracy'])
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

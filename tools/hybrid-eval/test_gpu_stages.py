"""CPU contract tests for stage orchestration; no CUDA, downloads or live service stops."""
import copy
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'hybrid-bench'))
import gpu_stage as common
import stage_ruler_bf16 as ruler
import stage_kl_bf16 as kl
from ruler_eval import archive_depth
import test_ruler as fixtures


class StageTests(unittest.TestCase):
    # An active evaluator refuses without signalling it; an idle fixture admits the same checkout.
    def test_active_evaluator_refuses_with_idle_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'; source.mkdir()
            proc = root/'proc'; proc.mkdir()
            common.no_evaluator(source, proc)
            process = proc/'123'; process.mkdir()
            (process/'cwd').symlink_to(source, target_is_directory=True)
            (process/'cmdline').write_bytes(b'python3\0tools/hybrid-eval/ruler_eval.py\0--depth\08192\0')
            with self.assertRaisesRegex(RuntimeError, 'evaluator active'):
                common.no_evaluator(source, proc)
            (process/'cmdline').write_bytes(b'python3\0unrelated.py\0')
            common.no_evaluator(source, proc)

    # Archive verification checks internal checksums even when an attacker updates the outer archive seal.
    def test_archive_internal_corruption_refuses_with_valid_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'depth'
            manifest, _, _, _ = fixtures.Bf16RulerTests().baseline(root)
            archive_depth(root, manifest, 1)
            common.verify_archive(root)
            with tarfile.open(root/'complete.tar.gz') as bundle:
                files = {m.name: bundle.extractfile(m).read() for m in bundle.getmembers()}
            files['analysis.md'] += b'changed'
            with tarfile.open(root/'complete.tar.gz', 'w:gz') as bundle:
                for name, content in files.items():
                    member = tarfile.TarInfo(name); member.size = len(content)
                    bundle.addfile(member, io.BytesIO(content))
            seal = common.read(root/'complete.json'); seal['archive_sha256'] = common.sha(root/'complete.tar.gz')
            common.save(root/'complete.json', seal)
            (root/'complete.tar.gz.sha256').write_text(seal['archive_sha256']+'  complete.tar.gz\n')
            with self.assertRaisesRegex(RuntimeError, 'member checksum differs: analysis.md'):
                common.verify_archive(root)

    # A low-disk refusal has an adjacent admitted boundary; it is not a reject-everything gate.
    def test_disk_gate_boundary(self):
        with patch('gpu_stage.shutil.disk_usage', return_value=SimpleNamespace(free=common.FLOOR+10)):
            self.assertEqual(common.disk('.', 10), common.FLOOR+10)
            with self.assertRaisesRegex(RuntimeError, 'disk gate'):
                common.disk('.', 11)

    # A reusable build must retain all targets and unchanged binary hashes; no build command runs for the valid record.
    def test_cached_build_record_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; source.mkdir()
            directory = root/'build'; (directory/'bin').mkdir(parents=True)
            names = ['llama-server', 'llama-hybrid-prefill', 'llama-hybrid-score', 'llama-hybrid-bench-check']
            for name in names: (directory/'bin'/name).write_bytes(b'fixture binary')
            record = dict(commit='a'*40, flags=common.FLAGS, source_path=str(source),
                          binaries={name: common.sha(directory/'bin'/name) for name in names})
            common.save(directory/'hybrid-bench-build.json', record)
            with patch.object(common, 'clean_source'), patch.object(common, 'run') as execute:
                common.build(source, directory, 'a'*40, root/'build.log')
                execute.assert_not_called()
                (directory/'bin/llama-hybrid-score').write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'build binaries changed'):
                    common.build(source, directory, 'a'*40, root/'build.log')
                del record['binaries']['llama-hybrid-score']
                common.save(directory/'hybrid-bench-build.json', record)
                with self.assertRaisesRegex(RuntimeError, 'lacks required targets'):
                    common.build(source, directory, 'a'*40, root/'build.log')
                execute.assert_not_called()

    # Terminal failure cannot be mistaken for success; the green control produces only the done line.
    def test_terminal_lines_and_child_exit_records(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(common.terminal('fixture', lambda: None), 0)
            def refuse():
                raise RuntimeError('named refusal')
            self.assertEqual(common.terminal('fixture', refuse), 1)
        self.assertIn('done fixture seconds=', out.getvalue())
        self.assertEqual(err.getvalue(), 'failed fixture: named refusal\n')
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'child.log'
            common.run([sys.executable, '-c', 'raise SystemExit(0)'], log)
            with self.assertRaisesRegex(RuntimeError, 'exited 7'):
                common.run([sys.executable, '-c', 'raise SystemExit(7)'], log)
            events = [json.loads(line) for line in Path(str(log)+'.commands.jsonl').read_text().splitlines()]
            self.assertEqual([r['exit_code'] for r in events if r['event'] == 'exit'], [0, 7])

    # GPU admission is explicit: the target card passes, CPU/another card cannot start downloads or builds.
    def test_gpu_identity_admission(self):
        with patch('gpu_stage.subprocess.check_output', return_value='NVIDIA RTX PRO 6000 Blackwell, GPU-fixture, 580, 97887'):
            self.assertIn('RTX PRO 6000', common.gpu_identity())
        with patch('gpu_stage.subprocess.check_output', return_value='NVIDIA A100, GPU-fixture, 580, 81920'):
            with self.assertRaisesRegex(RuntimeError, 'requires one RTX PRO 6000'):
                common.gpu_identity()

    # A later-depth preflight failure prevents every example and still attempts original-service restoration.
    def test_all_depths_preflight_before_examples_and_restore_on_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                original = root/'original'; original.mkdir()
                for role in ruler.ROLES: (original/role).mkdir()
                build = root/'original-build'; build.mkdir()
                common.save(build/'hybrid-bench-build.json', dict(commit='18d3f114'+'0'*32, binaries={}))
                args = SimpleNamespace(stage=root/'stage', original_run=original, original_build=build,
                                       original_source=root/'old-source', source=root/'new-source', ref='a'*40,
                                       build=root/'new-build')
                commands = []
                def invoke(command, log=None):
                    commands.append([str(x) for x in command])
                    if failure and '--preflight-only' in command and '32768' in command:
                        raise RuntimeError('planted manifest mismatch')
                manifests = [dict(depth=d, input_lock_sha256='hash', samples_per_task=30, seed=42, margin=.02, tasks=['qa_1'])
                             for d in (8192, 32768)]
                with patch.object(ruler, 'launch_plan', return_value={r: ['start', r] for r in ruler.ROLES}), \
                     patch.object(ruler, 'clean_source'), patch.object(ruler, 'no_evaluator'), patch.object(ruler, 'build'), \
                     patch.object(ruler, 'live_identity', return_value={}), patch.object(ruler, 'gpu_identity', return_value='fixture'), \
                     patch.object(ruler, 'run', side_effect=invoke), patch.object(ruler, 'restore') as restore, \
                     patch.object(ruler, 'disk'), patch.object(ruler, 'verify_archive'), patch.dict(os.environ):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, 'planted manifest mismatch'):
                            ruler.execute(args, manifests)
                    else:
                        ruler.execute(args, manifests)
                    restore.assert_called_once()
                trials = [c for c in commands if '--paired-depth' in c]
                self.assertEqual([('--preflight-only' in c) for c in trials], [True, True] if failure else [True, True, False, False])

    # Relaunch plans preserve the recorded settings; the ordinary candidate client alone requests bf16.
    def test_launch_plan_preserves_settings_and_refuses_unrepresentable_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = dict(backend='cuda', threads=2, threads_batch=2, context=135168, model_sha256='same')
            (root/'node').mkdir()
            common.save(root/'node/node-info.json', dict(shared, port=8091, sizing=dict(ram_budget_mib=16000)))
            for role, batch, port in [('client', 64, 8092), ('partition', 256, 8094)]:
                (root/role).mkdir()
                common.save(root/role/'client-info.json', dict(shared, model_path='/models/model.gguf', batch=batch, ubatch=batch,
                                                             proxy_url=f'http://127.0.0.1:{port}'))
                common.save(root/role/'proxy-config.json', dict(prefill_url='http://127.0.0.1:8091', decode_url=f'http://127.0.0.1:{port-2}',
                    prefill_timeout=1234, decode_timeout=1234, extend_chunk_tokens=4096))
            old = ruler.launch_plan(root, root/'build-ruler', 'original', root, root, False)
            new = ruler.launch_plan(root/'candidate', root/'build-bf16', 'candidate', root/'new-state', root, True)
            for role in ruler.ROLES:
                self.assertNotIn('--state-dtype', old[role])
                for option in ['--model', '--ctx', '--port']:
                    self.assertEqual(old[role][old[role].index(option)+1], new[role][new[role].index(option)+1])
            self.assertIn('--state-dtype', new['client'])
            self.assertEqual(new['client'][new['client'].index('--state-dtype')+1], 'bf16')
            self.assertNotIn('--state-dtype', new['partition'])
            self.assertNotIn('--state-dtype', new['node'])
            self.assertEqual(new['client'][new['client'].index('--request-timeout')+1], '1234')
            config = common.read(root/'client/proxy-config.json'); config['decode_timeout'] += 1
            common.save(root/'client/proxy-config.json', config)
            with self.assertRaisesRegex(RuntimeError, 'cannot preserve unequal service timeouts'):
                ruler.launch_plan(root, root/'build', 'original', root, root, False)

    # Restoration preserves software/model/settings and refuses a successfully started service with the wrong identity.
    def test_restore_identity_with_positive_control(self):
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for role in ruler.ROLES:
                    folder = root/'original-services'/role; folder.mkdir(parents=True)
                    common.save(folder/('node-info.json' if role == 'node' else 'client-info.json'), dict(commit='original', model_sha256='same'))
                plan = dict(source='candidate', original_source='original', original_run=str(root/'live'), original_ref='original',
                            restore={r: ['start', r] for r in ruler.ROLES})
                def invoke(command, log=None):
                    if command[0] == 'start':
                        role = command[1]; folder = root/'live'/role; folder.mkdir(parents=True)
                        common.save(folder/('node-info.json' if role == 'node' else 'client-info.json'),
                                    dict(commit='changed' if changed else 'original', model_sha256='same'))
                with patch.object(ruler, 'run', side_effect=invoke):
                    if changed:
                        with self.assertRaisesRegex(RuntimeError, 'restored node identity differs'):
                            ruler.restore(root, plan)
                        self.assertFalse((root/'restored.json').exists())
                    else:
                        ruler.restore(root, plan)
                        self.assertTrue((root/'restored.json').exists())


class KlStageTests(unittest.TestCase):
    def fixture(self):
        ids = [f'pg{i}' for i in range(20)]
        previous = dict(settings=dict(seed=42, arms=['D-local', 'C-Q4-local']),
                        rows=[dict(scope='document', arm='D-local', metric='kl_nats', document=i) for i in ids])
        models = common.read(Path(kl.__file__).with_name('state_bf16_model_pins.json'))['models']
        lock = dict(version=1, historical_report_sha256='c'*64, settings=dict(seed=42, d=20, n=32768, m=2048, score_batch=32),
                    documents=[dict(id=i, url='https://www.gutenberg.org/cache/epub/1/pg1.txt', size=10, sha256='d'*64, token_sha256='e'*64) for i in ids],
                    models=models,
                    provenance=dict(kind='newly-pinned-study', source='fixture independent lock'))
        return previous, lock

    # New-study pins admit only explicit immutable model and document/token hashes; absent historical provenance cannot pass reproduction.
    def test_missing_hash_revision_comparator_refusals_with_new_study_control(self):
        previous, lock = self.fixture()
        kl.validate_lock(lock, previous, 'c'*64, 'new-study')
        for mutate, message in [
            (lambda x: x['documents'][0].pop('token_sha256'), 'missing document/token hash'),
            (lambda x: x['models']['q4'].update(revision='main'), 'immutable repository revision'),
            (lambda x: x['documents'].reverse(), 'document order differs'),
        ]:
            bad = copy.deepcopy(lock); mutate(bad)
            with self.assertRaisesRegex(RuntimeError, message):
                kl.validate_lock(bad, previous, 'c'*64, 'new-study')
        lock['provenance']['kind'] = 'recovered-stage2-lock'
        with self.assertRaisesRegex(RuntimeError, 'missing homogeneous stage-2 f32 comparator'):
            kl.validate_lock(lock, previous, 'c'*64, 'reproduce')

    # A real historical homogeneous arm would admit; a heterogeneous row cannot be relabelled into that control.
    def test_reproduction_comparator_identity_with_positive_control(self):
        previous, lock = self.fixture()
        lock['provenance']['kind'] = 'recovered-stage2-lock'
        model_sha = lock['models']['q4']['files'][0]['sha256']
        lock['f32_comparator'] = dict(arm='B-Q4-Q4-imported', node_model_sha256=model_sha, client_model_sha256=model_sha)
        previous['settings']['arms'].append('B-Q4-Q4-imported')
        kl.validate_lock(lock, previous, 'c'*64, 'reproduce')
        previous['settings']['arms'].append('B-Mirai-Q4-imported')
        lock['f32_comparator']['arm'] = 'B-Mirai-Q4-imported'
        with self.assertRaisesRegex(RuntimeError, 'missing homogeneous stage-2 f32 comparator'):
            kl.validate_lock(lock, previous, 'c'*64, 'reproduce')

    # Download admission verifies bytes before publishing the final filename; a wrong digest preserves only its partial.
    def test_model_download_digest_with_positive_control(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                payload = b'fixture GGUF bytes'
                pin = dict(repo='fixture/model', revision='a'*40, primary='model.gguf', files=[dict(name='model.gguf',
                           size=len(payload), sha256='0'*64 if corrupt else hashlib.sha256(payload).hexdigest())])
                def download(command, log=None):
                    Path(command[command.index('--output')+1]).write_bytes(payload)
                with patch.object(kl, 'run', side_effect=download), patch.object(kl, 'disk'):
                    if corrupt:
                        with self.assertRaisesRegex(RuntimeError, 'downloaded model hash differs'):
                            kl.download_models(dict(models=dict(q4=pin)), root)
                        self.assertFalse((root/'q4/model.gguf').exists())
                        self.assertEqual((root/'q4/model.partial').read_bytes(), payload)
                    else:
                        result = kl.download_models(dict(models=dict(q4=pin)), root)
                        self.assertEqual(result['q4'].read_bytes(), payload)

    # Prepared token hashes are compared independently with the lock, including a one-token negative.
    def test_prepared_token_hash_mismatch_with_control(self):
        fixture = dict(tokens=[1, 2], continuation=[3])
        pins = [dict(id='pg1', sha256='source', token_sha256=kl.study.token_hash(fixture))]
        index = dict(documents=[dict(id='pg1', source=dict(sha256='source'))])
        kl.verify_tokens(index, {'pg1': fixture}, pins)
        fixture['continuation'][0] = 4
        with self.assertRaisesRegex(RuntimeError, 'document/token hash differs: pg1'):
            kl.verify_tokens(index, {'pg1': fixture}, pins)

    # The saved interval is the bound, with inclusive endpoints and an explicit outside-interval flag.
    def test_historical_interval_gate(self):
        old = dict(rows=[dict(scope='documents', arm='old', metric=m, count=20, ci_low=.2, ci_high=.4) for m in ('kl_nats', 'top1')])
        new = dict(rows=[dict(scope='documents', arm='new', metric=m, count=20, mean=.2) for m in ('kl_nats', 'top1')])
        self.assertTrue(all(r['passed'] for r in kl.compare_intervals(new, old, 'old', 'new')))
        new['rows'][0]['mean'] = .40001
        self.assertFalse(kl.compare_intervals(new, old, 'old', 'new')[0]['passed'])
        with self.assertRaisesRegex(RuntimeError, 'missing comparator'):
            kl.compare_intervals(new, old, 'absent', 'new')


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""CPU-only contract tests with local corpus fixtures; no downloads."""
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
import random
import re
import tempfile
from types import SimpleNamespace
import unittest
import tarfile
import zipfile
from unittest.mock import patch

from hybrid_eval import Refusal, digest, http
from ruler_analysis import analyze, excess_interval, mcnemar, wilson, write_analysis
from ruler_data import DATA_FILES, PUNKT_FILES, SOURCE, VENDOR, bootstrap, verify_data, verify_source
from ruler_eval import (SLOT_CLEANUP_PREDECESSOR, archive_depth, atomic_json, check_slot_space,
                        example_sessions, main as evaluate_main, recover_sessions, resume_manifest, run_example)
from ruler_generate import Generator, body_for, configurations, score
from ruler_glimmer import CONTEXT, DEPTHS, MODEL, NATIVE_REF, saved_session_bytes, validate_services


def fixtures(root):
    folder = root/'json'
    folder.mkdir()
    (folder/'PaulGrahamEssays.json').write_text(json.dumps(dict(text=('A meadow is quiet. Clouds cross the sky. ')*200)))
    (folder/'english_words.json').write_text(json.dumps({str(i): 'fixtureword'+str(i) for i in range(20)}))
    paragraphs = [dict(context=f'Fixture place {i} has the color fixturegold{i}. '+('Clouds cross the sky. '*30),
                       qas=[dict(question=f'What is the color for fixture place {i}?', is_impossible=False,
                                 answers=[dict(text=f'fixturegold{i}')])]) for i in range(40)]
    (folder/'squad.json').write_text(json.dumps(dict(data=[dict(paragraphs=[p]) for p in paragraphs])))
    hotpot = [dict(question=f'What is the color for fixture place {i}?', answer=f'fixturegold{i}',
                   context=[[f'Place {i}', [p['context']]]]) for i, p in enumerate(paragraphs)]
    (folder/'hotpotqa.json').write_text(json.dumps(hotpot))
    return root


def fixture_sentences(text):
    # The local fixture has no abbreviations; this avoids downloading trained Punkt data on agents1.
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    return PunktSentenceTokenizer().tokenize(text)


def table_row(index, offload=False, partition=False) -> dict:
    return dict(depth=8192, task='niah_single_1', index=index, results={arm: dict(answer=answer, score=float(answer == 'a'),
                prompt_sha256='same') for arm, answer in [('local', 'a'), ('offload', 'b' if offload else 'a'),
                                                         ('local_partition', 'b' if partition else 'a')]})


class RulerTests(unittest.TestCase):
    # These fixtures establish the exact upstream substring rules, including partial lists and QA aliases.
    def test_official_scoring_and_wrong_answer_negative(self):
        namespace = {}
        from ruler_data import VENDOR
        exec((VENDOR/'metrics.py').read_text(), namespace)
        for task in configurations():
            references = ['1234567', '7654321']
            for observed in ('1234567 7654321', '1234567', 'WRONG ANSWER', '', 'X1234567X'):
                family = 'qa' if task.startswith('qa_') else 'niah'
                expected = namespace['TASKS'][family]['metric_fn']([observed], [references])/100
                self.assertAlmostEqual(score(task, references, observed), expected, places=4)
            self.assertEqual(score(task, references, 'WRONG ANSWER'), 0)
            self.assertEqual(score(task, references, '1234567 7654321'), 1)
        with self.assertRaisesRegex(Refusal, 'empty RULER reference'):
            score('qa_1', [''], 'anything')

    # A character tokenizer checks stable contents, exact full-prompt length and seed separation for every configuration.
    def test_all_generators_deterministic_and_too_short_negative(self):
        with tempfile.TemporaryDirectory() as tmp, patch('nltk.tokenize.sent_tokenize', side_effect=fixture_sentences), \
             patch('wonderwords.random_word._get_words_from_text_file', return_value=[f'word{i}' for i in range(300)]):
            data = fixtures(Path(tmp))
            for task in configurations():
                # VT's upstream 500-token example needs a word tokenizer; rendering still counts every character.
                generator = Generator(task, data, str.split if task == 'vt' else list, 17, fixture=True)
                def render(body):
                    return list(''.join(m['content'] for m in body['messages']))
                first = generator.generate(0, 8192, render)
                self.assertEqual(first, generator.generate(0, 8192, render), task)
                self.assertEqual(first['prompt_tokens'], 8192)
                self.assertNotEqual(first['input_sha256'], generator.generate(1, 8192, render)['input_sha256'])
                self.assertEqual(score(task, first['answers'], ' '.join(first['answers'])), 1)
                self.assertEqual(score(task, first['answers'], 'WRONG ANSWER'), 0)
            with self.assertRaisesRegex(Refusal, 'minimum exceeds'):
                Generator('cwe', data, list, 17, fixture=True).generate(0, 512, render)

    # Compare the worked example, answer and insertion to the pinned upstream path, independently of fitting/padding.
    def test_vt_worked_example_matches_upstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = Generator('vt', fixtures(Path(tmp)), str.split, 17, fixture=True)
            module = generator.module
            random.seed(17)
            generator.np.random.seed(17)
            example = module['sys_vartrack_w_noise_random'](
                num_samples=1, max_seq_length=500, incremental=5,
                num_chains=module['args'].num_chains, num_hops=module['args'].num_hops)[0]
            self.assertEqual(generator.icl_example, example)
            self.assertLessEqual(example['length'], 500)
            self.assertTrue(all(re.fullmatch('[A-Z]{3}', name) for name in example['outputs']))
            seed = int(digest([17, 'vt', 0])[:8], 16)
            random.seed(seed)
            generator.np.random.seed(seed)
            text, answers = module['generate_input_output'](20, module['args'].num_chains, module['args'].num_hops)
            worked = module['randomize_icl'](example['input']+' '+' '.join(example['outputs'])+'\n')
            cutoff = text.index(generator.base['template'][:20])
            expected = text[:cutoff]+worked+'\n'+text[cutoff:]
            actual, references = generator.candidate(0, 20, 8192)
            self.assertEqual((actual, references), (expected, answers))
            self.assertEqual(actual.count(generator.base['template'][:20]), 2)
            worked_answers = worked.strip().split()[-len(example['outputs']):]
            self.assertTrue(all(re.fullmatch('[A-Z]{3}', name) for name in worked_answers))
            self.assertTrue(all('VAR '+name in worked for name in worked_answers))
            self.assertEqual((actual, references), generator.candidate(0, 20, 32768))

    # Fixture word/whitespace tokens bound largest requests and total rendering at production depths; no GPU/corpora.
    def test_all_generators_bounded_render_work(self):
        def tokenize(text):
            return re.findall(r'\S+|\s', text)
        with tempfile.TemporaryDirectory() as tmp, patch('nltk.tokenize.sent_tokenize', side_effect=fixture_sentences), \
             patch('wonderwords.random_word._get_words_from_text_file', return_value=[f'word{i}' for i in range(300)]):
            data = fixtures(Path(tmp))
            for depth in (8192, 32768):
                for task in configurations():
                    with self.subTest(task=task, depth=depth):
                        generator = Generator(task, data, tokenize, 17, fixture=True)
                        lengths = []
                        def render(body):
                            tokens = tokenize(''.join(m['content'] for m in body['messages']))
                            lengths.append(len(tokens))
                            self.assertLessEqual(len(tokens), 4*depth, 'oversized tokenizer request')
                            self.assertLessEqual(sum(lengths), 30*depth, 'excessive total render work')
                            return tokens
                        spec = generator.generate(0, depth, render)
                        self.assertEqual(spec['prompt_tokens'], depth)
                        self.assertEqual(score(task, spec['answers'], ' '.join(spec['answers'])), 1)
                        self.assertEqual(score(task, spec['answers'], 'WRONG ANSWER'), 0)

    # The production profile pins all three services; valid neighbors prevent reject-all admission.
    def test_glimmer_pins_services_and_negative_controls(self):
        self.assertEqual(MODEL['repo'], 'meta-models/Muse-Glimmer-30B-GGUF')
        self.assertEqual(MODEL['revision'], 'b44f1072cf75e918a1b82a3a17b45c4389b076e7')
        self.assertEqual(MODEL['file'], 'Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf')
        self.assertEqual(MODEL['bytes'], 16756683904)
        self.assertEqual(MODEL['sha256'], '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e')
        self.assertEqual(NATIVE_REF, '18d3f114e46c4f801c4a09e2c6cf8ba2303ae42d')
        client = dict(model_sha256=MODEL['sha256'], commit=NATIVE_REF, context=131072,
                      batch=64, ubatch=64, backend='cuda')
        services = [client, dict(client, batch=256, ubatch=256), dict(client, batch=256, ubatch=256)]
        for depth in DEPTHS:
            validate_services(*services, depth, saved_session_bytes(depth))
        for index in range(3):
            for key, value, reason in [('model_sha256', 'wrong', 'model pin'), ('commit', 'wrong', 'native pin'),
                                       ('context', 135168, 'context'), ('context', 8192, 'context'),
                                       ('batch', 128, 'batch/ubatch'), ('ubatch', 32, 'batch/ubatch'),
                                       ('backend', 'cpu', 'requires CUDA')]:
                changed = [dict(info) for info in services]
                changed[index][key] = value
                with self.subTest(index=index, key=key, value=value), self.assertRaisesRegex(Refusal, reason):
                    validate_services(*changed, 8192, saved_session_bytes(8192))
        with self.assertRaisesRegex(Refusal, 'bound too small'):
            validate_services(*services, 8192, saved_session_bytes(8192)-1)
        with self.assertRaisesRegex(Refusal, 'invalid Glimmer depth'):
            validate_services(*services, 131072, 10**10)

    # All depths fit native context plus answer reserve; the bound exceeds four measured kit slot sizes.
    def test_glimmer_depth_and_slot_arithmetic(self):
        self.assertEqual(CONTEXT, 131072)
        self.assertEqual(DEPTHS, (8192, 32768, 65536, 122880))
        self.assertEqual(DEPTHS[-1] + 128 + 2, 123010)
        self.assertEqual(CONTEXT - (DEPTHS[-1] + 128 + 2), 8062)
        self.assertGreater(131072 + 128 + 2, CONTEXT)
        for depth, bound in zip(DEPTHS, (152326144, 403984384, 739528704, 1326731264)):
            self.assertEqual(saved_session_bytes(depth), bound)
            self.assertLessEqual(depth + 128 + 2, CONTEXT)
        for depth, measured in ((16000, 156884268), (50000, 397876268),
                                (100000, 752276268), (118000, 879860268)):
            self.assertEqual(7088*depth + 43476268, measured)
            self.assertGreater(10240*(depth+130)+64*1024**2, measured)
        for depth in (2048, 131072, 130943, -1):
            with self.assertRaisesRegex(Refusal, 'invalid Glimmer depth'):
                saved_session_bytes(depth)
        rows = [dict(table_row(0), depth=depth) for depth in DEPTHS]
        self.assertTrue(all('mcnemar_holm_four_depths' in t for t in analyze(rows)[1] if t['task'] == 'ALL'))

    # CLI profile rejection precedes output/session creation; accepted admission records the profile in restart identity.
    def test_glimmer_cli_admission_and_manifest_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'inputs.lock.json').write_text('{}')
            (root/'downloads.json').write_text('[]')
            client = dict(model_sha256='wrong', commit=NATIVE_REF, context=CONTEXT, batch=64, ubatch=64, backend='cuda')
            partition = dict(client, model_sha256=MODEL['sha256'], batch=256, ubatch=256)
            argv = ['ruler_eval', '--model-profile', 'glimmer30', '--client-state', str(root),
                    '--partition-client-state', str(root), '--data', str(root),
                    '--input-lock-sha256', hashlib.sha256(b'{}').hexdigest(), '--output', str(root/'out'),
                    '--depth', '8192', '--saved-session-bytes', str(saved_session_bytes(8192))]
            with patch('sys.argv', argv), patch('ruler_eval.verify_data', return_value={}), \
                 patch('ruler_eval.load_clients', return_value=({}, client, {}, partition, partition)), \
                 patch('ruler_eval.recover_sessions', side_effect=RuntimeError('admitted')) as recover, \
                 patch('ruler_eval.run_example') as execute:
                with self.assertRaisesRegex(Refusal, 'Glimmer model pin differs'):
                    evaluate_main()
                self.assertFalse((root/'out').exists())
                recover.assert_not_called()
                client['model_sha256'] = MODEL['sha256']
                with self.assertRaisesRegex(RuntimeError, '^admitted$'):
                    evaluate_main()
                manifest = json.loads((root/'out'/'depth-8192'/'manifest.json').read_text())
                self.assertEqual(manifest['model_profile'], 'glimmer30')
                self.assertEqual(manifest['seed'], 261102)
                self.assertEqual(manifest['samples_per_task'], 30)
                self.assertEqual(len(manifest['tasks']), 13)
                recover.assert_called_once()
                execute.assert_not_called()

    # Missing settings and a wrong reviewed head fail before GPU probes, downloads, build or state creation.
    def test_glimmer_stage_wrong_head_has_terminal_failure_and_no_writes(self):
        script = Path(__file__).with_name('stage-ruler-glimmer.sh').resolve()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'
            for expected in ('', 'wrong'):
                result = subprocess.run(['bash', str(script)], env=dict(os.environ, RUN=str(run), EXPECTED_HEAD=expected),
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('stage ruler-glimmer failed', result.stderr)
                self.assertNotIn('stage ruler-glimmer done', result.stdout)
                self.assertFalse(run.exists())

    # Partial reports expose Bonferroni, while exactly four depths get joint Holm; excess depths are refused.
    def test_adjustment_labels_partial_and_joint_depths(self):
        rows = [table_row(i, True) for i in range(6)]
        partial = analyze(rows)[1][0]
        self.assertNotIn('mcnemar_holm_four_depths', partial)
        self.assertEqual(partial['mcnemar_bonferroni_four_depths'], 4*partial['mcnemar_p'])
        joint = [dict(row, depth=depth) for depth in (8192, 32768, 65536, 131072) for row in rows]
        primary = [table for table in analyze(joint)[1] if table['task'] == 'ALL']
        self.assertEqual(len(primary), 4)
        for table in primary:
            self.assertNotIn('mcnemar_bonferroni_four_depths', table)
            self.assertEqual(table['mcnemar_holm_four_depths'], 4*table['mcnemar_p'])
        with self.assertRaisesRegex(Refusal, 'more than four'):
            analyze(joint+[dict(row, depth=262144) for row in rows])

    # The independent 11-pair table has b=6,c=0, hence exact p=2/64 and rates 8/11 versus 2/11.
    def test_known_paired_table_and_intervals(self):
        rows = [table_row(i) for i in range(3)] + [table_row(i, True) for i in range(3, 9)] + [table_row(i, True, True) for i in range(9, 11)]
        examples, tables = analyze(rows)
        table = tables[0]
        self.assertEqual((table['n'], table['left_only'], table['right_only'], table['both_true']), (11, 6, 0, 2))
        self.assertEqual(table['mcnemar_p'], .03125)
        self.assertEqual(table['offload_rate'], 8/11)
        self.assertEqual(table['partition_rate'], 2/11)
        self.assertEqual(sum(r['offload_changed'] != r['partition_changed'] for r in examples), 6)
        self.assertEqual(mcnemar(0, 0), 1)
        self.assertEqual(mcnemar(1, 7), 18/256)
        self.assertAlmostEqual(wilson(0, 390)[1], 3.841458820694124/(390+3.841458820694124))
        self.assertAlmostEqual(excess_interval(0, 0, 390)[1], 1-.0125**(1/390), places=12)
        zero = analyze([table_row(i) for i in range(390)])[1][0]
        self.assertFalse(zero['strict_no_more_changes'])
        self.assertTrue(zero['equivalent_at_margin'])
        reversed_rows = [table_row(i, False, True) for i in range(6)]
        reversed_table = analyze(reversed_rows)[1][0]
        self.assertEqual(reversed_table['mcnemar_p'], .03125)
        self.assertEqual(reversed_table['excess'], -1)
        self.assertGreater(reversed_table['excess_conservative95'][1], 0)
        with tempfile.TemporaryDirectory() as tmp:
            write_analysis(Path(tmp), rows)
            self.assertEqual(len((Path(tmp)/'discordant-pairs.jsonl').read_text().splitlines()), 6)
            self.assertTrue((Path(tmp)/'per-example.csv').exists())

    # Repeated rows, incomplete arms and token-mismatched pairs must never enter paired denominators.
    def test_invalid_pairs_rejected(self):
        with self.assertRaisesRegex(Refusal, 'duplicate'):
            analyze([table_row(0), table_row(0)])
        row = table_row(0)
        row['results'].pop('offload')
        with self.assertRaisesRegex(Refusal, 'treatment'):
            analyze([row])
        row = table_row(0)
        row['results']['offload']['prompt_sha256'] = 'different'
        with self.assertRaisesRegex(Refusal, 'prompts differ'):
            analyze([row])

    # Source and input locks catch modifications; interrupted depths cannot receive a completion archive.
    def test_locks_and_incomplete_depth_negative(self):
        verify_source()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'json').mkdir()
            for name in DATA_FILES:
                (root/'json'/name).write_text('fixture')
            (root/'downloads.json').write_text('[]')
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            for name in PUNKT_FILES:
                path = root/'nltk_data'/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            lock = dict(generator_commit=SOURCE['commit'], files={n: sha(root/'json'/n) for n in DATA_FILES},
                        downloads_manifest_sha256=sha(root/'downloads.json'),
                        nltk_files={name: sha(root/'nltk_data'/name) for name in PUNKT_FILES})
            atomic_json(root/'inputs.lock.json', lock)
            verify_data(root)
            (root/'json'/'squad.json').write_text('changed')
            with self.assertRaisesRegex(Refusal, 'hash mismatch'):
                verify_data(root)
            (root/'rows').mkdir()
            with self.assertRaisesRegex(Refusal, 'incomplete'):
                archive_depth(root, {}, 1)
            self.assertFalse((root/'complete.json').exists())

    # A bounded fake HTTP corpus proves audit/hash behavior; a known bad digest must block the production lock.
    def test_bootstrap_audit_and_bad_digest_control(self):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, 'w') as archive:
            for name in ('abbrev_types.txt', 'collocations.tab', 'ortho_context.tab', 'sent_starters.txt'):
                archive.writestr('punkt_tab/english/'+name, 'fixture')
        contents = {name: b'{}' for name in DATA_FILES if name != 'PaulGrahamEssays.json'}
        contents['punkt_tab.zip'] = payload.getvalue()
        pins = {name: ('https://fixture.invalid/'+name, hashlib.sha256(value).hexdigest()) for name, value in contents.items()}
        def fetch(url):
            name = url.rsplit('/', 1)[-1]
            value = contents.get(name, b'<font>A fixture essay.</font>' if '.html' in url else b'A fixture essay.')
            return value, url+'?temporary-credential=REDACT_ME'
        with tempfile.TemporaryDirectory() as tmp, patch('ruler_data.PINNED', pins):
            root = Path(tmp)/'data'
            lock = bootstrap(root, fetch)
            self.assertEqual(verify_data(root), lock)
            records = json.loads((root/'downloads.json').read_text())
            urls = [u for u in (VENDOR/'PaulGrahamEssays_URLs.txt').read_text().splitlines() if u.strip()]
            self.assertEqual(len(records), len(urls)+len(pins))
            self.assertNotIn('REDACT_ME', (root/'downloads.json').read_text())
            for record in records:
                self.assertEqual(hashlib.sha256((root/'downloads'/record['file']).read_bytes()).hexdigest(), record['sha256'])
            pins['squad.json'] = (pins['squad.json'][0], '0'*64)
            with self.assertRaisesRegex(Refusal, 'download hash mismatch: squad.json'):
                bootstrap(Path(tmp)/'bad', fetch)
            self.assertFalse((Path(tmp)/'bad'/'inputs.lock.json').exists())

    # A full paired depth gets an independently verifiable archive; a missing row leaves no success marker.
    def test_complete_archive_and_member_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'rows').mkdir()
            manifest = dict(margin=.02, data_provenance={'inputs.lock.json': {'fixture': True}})
            atomic_json(root/'manifest.json', manifest)
            atomic_json(root/'rows'/'one.json', table_row(0))
            archive_path = archive_depth(root, manifest, 1)
            first_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            self.assertEqual(json.loads((root/'complete.json').read_text())['archive_sha256'], first_hash)
            with tarfile.open(archive_path) as archive:
                member = archive.extractfile('SHA256SUMS')
                assert member is not None
                entries = member.read().decode().splitlines()
                for line in entries:
                    digest, name = line.split('  ')
                    member = archive.extractfile(name)
                    assert member is not None
                    self.assertEqual(hashlib.sha256(member.read()).hexdigest(), digest)
                self.assertNotIn('specs', archive.getnames())

    # Each slot volume gets 2*S+margin admission; one byte below the floor refuses before session writes.
    def test_slot_disk_floor_refuses_before_writing(self):
        configs = [dict(slot_path='client'), dict(slot_path='partition')]
        with patch('ruler_eval.shutil.disk_usage', side_effect=[SimpleNamespace(free=250), SimpleNamespace(free=249)]) as usage:
            with self.assertRaisesRegex(Refusal, 'slot disk floor: free=249 required=250.*no example started'):
                check_slot_space(configs, 100, 50)
            self.assertEqual([call.args[0] for call in usage.call_args_list], ['client', 'partition'])
        with patch('ruler_eval.shutil.disk_usage', return_value=SimpleNamespace(free=250)):
            check_slot_space(configs, 100, 50)
        for size, margin in ((0, 1), (-1, 1), (100, -1)):
            with self.assertRaisesRegex(Refusal, 'invalid slot disk budget'):
                check_slot_space(configs, size, margin)

    # Exceptions, interrupt and commit failure all release every allocated ID, including the arm never started.
    def test_example_sessions_cleanup_on_errors_and_abort(self):
        config = dict(proxy_url='fixture', client_credential='')
        for error in (Refusal('stream failed'), KeyboardInterrupt(), OSError('commit failed')):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'pending.json'
                with patch('ruler_eval.release_session', return_value=True) as release:
                    with self.assertRaises(type(error)):
                        with example_sessions(path, config, config, lambda event: None) as sessions:
                            self.assertEqual(json.loads(path.read_text()), sessions)
                            raise error
                    self.assertEqual(release.call_count, 3)
                    self.assertEqual({call.args[1] for call in release.call_args_list}, set(sessions.values()))
                    self.assertFalse(path.exists())

    # An unsuccessful release keeps the exact recovery IDs; relaunch retries them and allocates fresh IDs.
    def test_pending_sessions_recovered_after_hard_stop(self):
        import uuid
        config = dict(proxy_url='fixture', client_credential='')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pending.json'
            stale = {arm: 'ruler-'+uuid.uuid4().hex for arm in ('local', 'offload', 'local_partition')}
            atomic_json(path, stale)
            with patch('ruler_eval.release_session', return_value=False) as release:
                with self.assertRaisesRegex(Refusal, 'pending RULER sessions could not be released'):
                    recover_sessions(path, config, config, lambda event: None)
                self.assertEqual(release.call_count, 3)
                self.assertEqual(json.loads(path.read_text()), stale)
            with patch('ruler_eval.release_session', return_value=True) as release:
                recover_sessions(path, config, config, lambda event: None)
                self.assertEqual({call.args[1] for call in release.call_args_list}, set(stale.values()))
                with example_sessions(path, config, config, lambda event: None) as fresh:
                    self.assertFalse(set(fresh.values()) & set(stale.values()))
                self.assertFalse(path.exists())

    # An interrupted/erroring DELETE cannot prevent attempts for the other two arms; its ledger survives for relaunch.
    def test_recovery_attempts_all_arms_when_release_raises(self):
        import uuid
        config = dict(proxy_url='fixture', client_credential='')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pending.json'
            sessions = {arm: 'ruler-'+uuid.uuid4().hex for arm in ('local', 'offload', 'local_partition')}
            for error in (KeyboardInterrupt(), OSError('event log unavailable')):
                atomic_json(path, sessions)
                with patch('ruler_eval.release_session', side_effect=[error, True, True]) as release:
                    with self.assertRaises(type(error)):
                        recover_sessions(path, config, config, lambda event: None)
                    self.assertEqual(release.call_count, 3)
                    self.assertEqual(json.loads(path.read_text()), sessions)

    # Only the accepted source repair can migrate an incomplete depth; scoring/data/client identity stays pinned.
    def test_slot_cleanup_manifest_migration_rejects_other_changes(self):
        before: dict = dict(seed=17, source_hashes=dict(SLOT_CLEANUP_PREDECESSOR, **{'hybrid-eval/ruler_generate.py': 'unchanged'}))
        after: dict = dict(before, source_hashes=dict.fromkeys(SLOT_CLEANUP_PREDECESSOR, 'new'))
        after['source_hashes']['hybrid-eval/ruler_generate.py'] = 'unchanged'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root/'manifest.json', before)
            with self.assertRaisesRegex(Refusal, 'resume identity changed'):
                resume_manifest(root, after, False)
            for bad in (dict(after, seed=18), dict(after, source_hashes=dict(after['source_hashes'], **{'hybrid-eval/ruler_generate.py': 'bad'}))):
                with self.assertRaisesRegex(Refusal, 'resume identity changed'):
                    resume_manifest(root, bad, True)
                self.assertEqual(json.loads((root/'manifest.json').read_text()), before)
            resume_manifest(root, after, True)
            self.assertEqual(json.loads((root/'manifest-before-slot-cleanup.json').read_text()), before)
            self.assertEqual(json.loads((root/'manifest.json').read_text()), after)
            resume_manifest(root, after, False)

    # Reproduce the 17-row stop: a source-only migration and fresh services reuse every committed byte and run only row 18.
    def test_resume_slot_cleanup_reuses_17_committed_rows_and_disk_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'
            data.mkdir()
            (data/'inputs.lock.json').write_text('{}')
            (data/'downloads.json').write_text('[]')
            client = dict(model_sha256='fixture', context=4096, batch=64)
            partition = dict(client, batch=256)
            config = dict(decode_url='fixture', decode_token='', proxy_url='fixture', client_credential='', slot_path=str(root))
            argv = ['ruler_eval', '--client-state', str(root), '--partition-client-state', str(root), '--data', str(data),
                    '--input-lock-sha256', hashlib.sha256(b'{}').hexdigest(), '--output', str(root/'out'), '--depth', '512',
                    '--tasks', 'niah_single_1', '--samples-per-task', '18', '--saved-session-bytes', '1024',
                    '--slot-margin-bytes', '256']
            def generate(index, depth, render):
                return dict(depth=depth, task='niah_single_1', index=index, input='x'*511, answer_prefix='y', answers=['a'],
                            prompt_sha256=digest(list('x'*511+'y')), input_sha256=digest(['x'*511, 'y']))
            def fake_http(base, path, token=None, body=None, **kwargs):
                assert body is not None
                if path == '/apply-template':
                    return dict(prompt=''.join(m['content'] for m in body['messages']))
                return dict(tokens=list(body['content']))
            def run(spec, *args):
                if spec['index'] == 17:
                    raise KeyboardInterrupt
                row: dict = dict(table_row(spec['index']), depth=512)
                row.update({k: v for k, v in spec.items() if k != 'input'})
                for arm in row['results'].values():
                    arm['prompt_sha256'] = spec['prompt_sha256']
                return row
            with patch('sys.argv', argv), patch('ruler_eval.verify_data', return_value={}), \
                 patch('ruler_eval.load_clients', return_value=(config, client, config, partition, client)), \
                 patch('ruler_eval.http', side_effect=fake_http), patch('ruler_eval.Generator') as generator, \
                 patch('ruler_eval.run_example', side_effect=run) as execute, \
                 patch('ruler_eval.release_session', return_value=True) as release:
                generator.return_value.generate.side_effect = generate
                with self.assertRaises(KeyboardInterrupt):
                    evaluate_main()
                depth = root/'out'/'depth-512'
                rows = {p.name: p.read_bytes() for p in (depth/'rows').glob('*.json')}
                self.assertEqual(len(rows), 17)
                manifest = json.loads((depth/'manifest.json').read_text())
                manifest['source_hashes'].update(SLOT_CLEANUP_PREDECESSOR)
                atomic_json(depth/'manifest.json', manifest)
                argv.append('--resume-slot-cleanup')
                calls = execute.call_count
                release.reset_mock()
                with patch('ruler_eval.shutil.disk_usage', return_value=SimpleNamespace(free=2303)):
                    with self.assertRaisesRegex(Refusal, 'free=2303 required=2304'):
                        evaluate_main()
                self.assertEqual(execute.call_count, calls)
                self.assertEqual(release.call_count, 0)
                self.assertFalse((depth/'pending-sessions.json').exists())
                execute.side_effect = None
                spec = json.loads((depth/'specs'/'niah_single_1-0017.json').read_text())
                spec.pop('spec_sha256')
                last_row: dict = dict(table_row(17), depth=512)
                execute.return_value = last_row
                for arm in last_row['results'].values():
                    arm['prompt_sha256'] = spec['prompt_sha256']
                self.assertEqual(evaluate_main(), 0)
                self.assertEqual(execute.call_count, calls+1)
                self.assertEqual(execute.call_args.args[0]['index'], 17)
                self.assertEqual(generator.return_value.generate.call_count, 18)
                for name, raw in rows.items():
                    self.assertEqual((depth/'rows'/name).read_bytes(), raw)
                self.assertTrue((depth/'manifest-before-slot-cleanup.json').exists())
                with tarfile.open(depth/'complete.tar.gz') as archive:
                    self.assertIn('slot-cleanup-migration.json', archive.getnames())
                self.assertFalse((depth/'pending-sessions.json').exists())

    # An interrupted pair is rerun once from its frozen input; completed depths skip inference and reject changed identity.
    def test_resume_after_interrupted_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'
            data.mkdir()
            (data/'inputs.lock.json').write_text('{}')
            (data/'downloads.json').write_text('[]')
            lock_sha = hashlib.sha256(b'{}').hexdigest()
            client = dict(model_sha256='fixture', context=4096, batch=64)
            partition = dict(client, batch=256)
            spec = dict(depth=512, task='niah_single_1', index=0, input='x'*511, answer_prefix='y', answers=['a'])
            spec.update(prompt_sha256=digest(list('x'*511+'y')), input_sha256=digest([spec['input'], spec['answer_prefix']]))
            row = table_row(0)
            row.update({k: v for k, v in spec.items() if k != 'input'})
            for result in row['results'].values():
                result['prompt_sha256'] = spec['prompt_sha256']
            argv = ['ruler_eval', '--client-state', str(root), '--partition-client-state', str(root), '--data', str(data),
                    '--input-lock-sha256', lock_sha, '--output', str(root/'out'), '--depth', '512',
                    '--tasks', 'niah_single_1', '--samples-per-task', '1', '--saved-session-bytes', '1024']
            def fake_http(base, path, token=None, body=None, **kwargs):
                assert body is not None
                if path == '/apply-template':
                    return dict(prompt=''.join(m['content'] for m in body['messages']))
                return dict(tokens=list(body['content']))
            config = dict(decode_url='fixture', decode_token='', proxy_url='fixture', client_credential='', slot_path=str(root))
            with patch('sys.argv', argv), patch('ruler_eval.verify_data', return_value={}), \
                 patch('ruler_eval.load_clients', return_value=(config, client, config, partition, client)), \
                 patch('ruler_eval.http', side_effect=fake_http), patch('ruler_eval.Generator') as generator, \
                 patch('ruler_eval.run_example') as run, patch('ruler_eval.release_session', return_value=True):
                generator.return_value.generate.return_value = spec
                run.side_effect = KeyboardInterrupt
                with self.assertRaises(KeyboardInterrupt):
                    evaluate_main()
                depth = root/'out'/'depth-512'
                self.assertFalse((depth/'complete.json').exists())
                self.assertEqual(list((depth/'rows').glob('*.json')), [])
                run.side_effect = None
                run.return_value = row
                self.assertEqual(evaluate_main(), 0)
                self.assertEqual(generator.return_value.generate.call_count, 1)
                frozen = json.loads((depth/'specs'/'niah_single_1-0000.json').read_text())
                self.assertGreaterEqual(frozen['generation_seconds'], 0)
                self.assertEqual(frozen['generation_seconds'], spec['generation_seconds'])
                self.assertEqual(len(list((depth/'rows').glob('*.json'))), 1)
                calls = run.call_count
                self.assertEqual(evaluate_main(), 0)
                self.assertEqual(run.call_count, calls)
                argv += ['--seed', '42']
                with self.assertRaisesRegex(Refusal, 'resume identity changed'):
                    evaluate_main()


def native_checks(root, config, client, pcfg, pclient, node):
    """Called by test_native after its existing routing suite; fixture corpus only."""
    data = root/'ruler-fixtures'
    data.mkdir()
    fixtures(data)
    def tokenize(text):
        return http(config['decode_url'], '/tokenize', data=dict(content=text, add_special=False))['tokens']
    def render(body):
        prompt = http(config['decode_url'], '/apply-template', data=body)['prompt']
        return http(config['decode_url'], '/tokenize', data=dict(content=prompt, add_special=True))['tokens']
    checks = []
    # All configurations use the real GGUF tokenizer; corpus fixtures and untrained Punkt limit this to integration.
    with patch('nltk.tokenize.sent_tokenize', side_effect=fixture_sentences):
        for task in configurations():
            generator = Generator(task, data, tokenize, 17, fixture=True)
            spec = generator.generate(0, 2048 if task != 'cwe' else 4096, render)
            require = unittest.TestCase().assertTrue
            require(spec == generator.generate(0, spec['depth'], render), 'real-tokenizer generation differs')
            require(spec['prompt_tokens'] == spec['depth'], 'real tokenizer depth differs')
            require(score(task, spec['answers'], 'WRONG ANSWER') == 0, 'wrong answer scored correct')
            require(score(task, spec['answers'], ' '.join(spec['answers'])) == 1, 'valid control rejected')
            checks.append({k: spec[k] for k in ('task', 'depth', 'prompt_sha256', 'padding_characters')})
            atomic_json(root/'ruler-generator-progress.json', checks)
            if task == 'niah_single_1':
                # Real three-arm requests verify prefix continuation, observed prompt hashes and state import.
                spec['max_tokens'] = 8
                result = run_example(spec, config, client, pcfg, pclient, node, 17, 300)
                atomic_json(root/'ruler-native-row.json', dict(result, fixture=True))
                write_analysis(root, [result])
    atomic_json(root/'ruler-generator-checks.json', dict(fixture=True, checks=checks,
                limits='Local tiny corpus, real GGUF tokenizer, untrained Punkt for abbreviation-free fixture; no benchmark data downloaded.'))


if __name__ == '__main__':
    unittest.main()

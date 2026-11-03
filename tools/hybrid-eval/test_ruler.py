#!/usr/bin/env python3
"""CPU-only contract tests with local corpus fixtures; no downloads."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import tarfile
import zipfile
from unittest.mock import patch

from hybrid_eval import Refusal, digest, http
from ruler_analysis import analyze, excess_interval, mcnemar, wilson, write_analysis
from ruler_data import DATA_FILES, PUNKT_FILES, SOURCE, VENDOR, bootstrap, verify_data, verify_source
from ruler_eval import archive_depth, atomic_json, main as evaluate_main, run_example
from ruler_generate import Generator, body_for, configurations, score


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


def table_row(index, offload=False, partition=False):
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
        with tempfile.TemporaryDirectory() as tmp, patch('nltk.tokenize.sent_tokenize', side_effect=fixture_sentences):
            data = fixtures(Path(tmp))
            for task in configurations():
                generator = Generator(task, data, list, 17, fixture=True)
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
                    '--tasks', 'niah_single_1', '--samples-per-task', '1']
            def fake_http(base, path, token=None, body=None, **kwargs):
                assert body is not None
                if path == '/apply-template':
                    return dict(prompt=''.join(m['content'] for m in body['messages']))
                return dict(tokens=list(body['content']))
            config = dict(decode_url='fixture', decode_token='')
            with patch('sys.argv', argv), patch('ruler_eval.verify_data', return_value={}), \
                 patch('ruler_eval.load_clients', return_value=(config, client, config, partition, client)), \
                 patch('ruler_eval.http', side_effect=fake_http), patch('ruler_eval.Generator') as generator, \
                 patch('ruler_eval.run_example') as run:
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

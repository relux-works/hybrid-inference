#!/usr/bin/env python3
"""CPU archive admission controls; stage the four owner-pinned archives locally."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from hybrid_eval import digest
from ruler_bf16_adapter import IMPLEMENTATION_SHA256, parse, read_archive, sha256
from ruler_equivalence import load_depth

HASHES = {
    'bf16-8192-complete.tar.gz': 'c0348572c2c1baf50dcfcdd0a73397594ba0d12e45ba157b0ae63980f25799b6',
    'bf16-32768-complete.tar.gz': '5efae77aa8b9902bd31b9b2a1ba336a0d8a090353898fd31fee8c980e73e1fbe',
    'depth-8192-complete.tar.gz': '2be3eae64b783867258a1f22e1c7bbdb4188f257c378f17c07554a8868248aa4',
    'depth-32768-complete.tar.gz': 'a6fb1e7e6ac0fb7d25ed3ed24ef1f47cdf4f6495b54a4866d6486870bfeef176',
}


class Bf16AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get('RULER_BF16_TEST_INPUTS')
        if not root:
            raise RuntimeError('Set RULER_BF16_TEST_INPUTS to the staged, read-only four-archive fixture directory')
        cls.root = Path(root)
        cls.archives = {}
        for name, checksum in HASHES.items():
            cls.archives[name] = read_archive(cls.root/name, checksum)[0]
        cls.adapter = Path(os.environ.get('RULER_BF16_ADAPTER', Path(__file__).with_name('ruler_bf16_adapter.py')))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def run_cli(self, depth=8192, bf16=None, baseline=None, expected=None, output_name='adapted'):
        bf_name, base_name = f'bf16-{depth}-complete.tar.gz', f'depth-{depth}-complete.tar.gz'
        bf_path, bf_sha = bf16 or (self.root/bf_name, HASHES[bf_name])
        base_path, base_sha = baseline or (self.root/base_name, HASHES[base_name])
        output = self.work/output_name
        proc = subprocess.run([sys.executable, str(self.adapter), '--bf16-archive', str(bf_path),
                               '--bf16-sha256', bf_sha, '--baseline-archive', str(base_path),
                               '--baseline-sha256', base_sha, '--output', str(output)], capture_output=True, text=True)
        if expected is not None:
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn('REFUSED:', proc.stderr)
            self.assertIn(expected, proc.stderr)
            self.assertNotIn('Traceback', proc.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(list(self.work.glob('.bf16-adapt-*')))
        else:
            self.assertEqual(proc.returncode, 0, proc.stderr)
        return output

    def write_archive(self, files, label='changed', update_sums=True):
        files = dict(files)
        if update_sums:
            files['SHA256SUMS'] = ''.join(sha256(data)+'  '+name+'\n'
                                        for name, data in sorted(files.items()) if name != 'SHA256SUMS').encode()
        path = self.work/(label+'.tar.gz')
        with tarfile.open(path, 'w:gz') as archive:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return path, sha256(path.read_bytes())

    def changed(self, mutate, label='bf16', reseal=True):
        name = 'bf16-8192-complete.tar.gz' if label == 'bf16' else 'depth-8192-complete.tar.gz'
        files = copy.deepcopy(self.archives[name])
        manifest = parse(files['manifest.json'])
        original_manifest = copy.deepcopy(manifest)
        rows = [parse(line) for line in files['rows.jsonl'].splitlines()]
        mutate(manifest, rows)
        if reseal:
            for row in rows:
                row['row_sha256'] = digest({k: v for k, v in row.items() if k != 'row_sha256'})
        if manifest != original_manifest:
            files['manifest.json'] = json.dumps(manifest, sort_keys=True, indent=2).encode()
        files['rows.jsonl'] = ''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows).encode()
        return self.write_archive(files, label)

    # The real 8k archive maps exactly the bf16 arm, preserves all baseline values and retains both row hashes.
    def test_real_8k_archive_admitted_with_exact_mapping_and_provenance(self):
        before = {p.name: sha256(p.read_bytes()) for p in self.root.glob('*.tar.gz')}
        output = self.run_cli()
        derived, _, _ = load_depth(output/'rows.jsonl')
        source = [parse(l) for l in self.archives['bf16-8192-complete.tar.gz']['rows.jsonl'].splitlines()]
        self.assertEqual(len(derived), 390)
        for original, row in zip(source, derived):
            self.assertEqual(row['results']['bf16'], original['results']['offload_bf16'])
            self.assertNotIn('offload_bf16', row['results'])
            self.assertEqual(row['original_row_sha256'], original['row_sha256'])
            restored = copy.deepcopy(row)
            restored.pop('row_sha256')
            restored['row_sha256'] = restored.pop('original_row_sha256')
            restored['results']['offload_bf16'] = restored['results'].pop('bf16')
            self.assertEqual(restored, original)
            self.assertNotEqual(row['row_sha256'], original['row_sha256'])
        provenance = parse((output/'provenance.json').read_bytes())
        self.assertEqual(provenance['implementation_sha256'], IMPLEMENTATION_SHA256)
        self.assertEqual(provenance['derived_rows_sha256'], sha256((output/'rows.jsonl').read_bytes()))
        self.assertEqual(provenance['rows'], [{k: r[k] for k in ('task', 'index', 'original_row_sha256',
                                                               'row_sha256', 'baseline_row_sha256')} for r in derived])
        self.assertEqual((output/'manifest.json').read_bytes(), self.archives['bf16-8192-complete.tar.gz']['manifest.json'])
        self.assertEqual(before, {p.name: sha256(p.read_bytes()) for p in self.root.glob('*.tar.gz')})

    # The same pinned implementation and baseline contract admit the neighboring real 32k depth.
    def test_real_32k_archive_admitted(self):
        output = self.run_cli(depth=32768)
        rows, manifest, _ = load_depth(output/'rows.jsonl')
        self.assertEqual((len(rows), manifest['depth']), (390, 32768))

    # An unrecognized fourth-arm name cannot be relabelled as the reviewed bf16 measurement.
    def test_unknown_fourth_arm_refuses(self):
        def mutate(m, rows):
            rows[0]['results']['unknown'] = rows[0]['results'].pop('offload_bf16')
        self.run_cli(bf16=self.changed(mutate), expected='unknown or second fourth arm')

    # A second fourth arm is rejected even if the expected offload_bf16 arm is present.
    def test_second_fourth_arm_refuses(self):
        def mutate(m, rows):
            rows[0]['results']['bf16'] = copy.deepcopy(rows[0]['results']['offload_bf16'])
        self.run_cli(bf16=self.changed(mutate), expected='unknown or second fourth arm')

    # A resealed, valid-score change to any frozen baseline arm is refused, not compared as a new baseline.
    def test_baseline_mismatch_refuses(self):
        for arm in ('local', 'offload', 'local_partition'):
            with self.subTest(arm=arm):
                def mutate(m, rows):
                    rows[0]['results'][arm]['answer'] += ' changed'
                self.run_cli(bf16=self.changed(mutate), expected='frozen baseline arms differ',
                             output_name='adapted-'+arm)

    # A resealed change to frozen generation metadata refuses before any output is created.
    def test_frozen_baseline_metadata_mismatch_refuses(self):
        def mutate(m, rows):
            rows[0]['max_tokens'] += 1
        self.run_cli(bf16=self.changed(mutate), expected='frozen baseline metadata differ')

    # Source row checksums are mandatory on both sides; a repaired archive checksum cannot conceal a bad row hash.
    def test_row_hash_failure_refuses(self):
        for side in ('bf16', 'baseline'):
            with self.subTest(side=side):
                def mutate(m, rows):
                    rows[0]['row_sha256'] = '0'*64
                self.run_cli(**{side: self.changed(mutate, label=side, reseal=False)}, expected='row hash failure')

    # Removing one example from either source fails exact task/index coverage before publication.
    def test_missing_task_index_refuses(self):
        for side in ('bf16', 'baseline'):
            with self.subTest(side=side):
                self.run_cli(**{side: self.changed(lambda m, rows: rows.pop(), label=side)},
                             expected='missing (task, index)')

    # Repeating one example in either source cannot silently replace a missing example.
    def test_duplicate_task_index_refuses(self):
        for side in ('bf16', 'baseline'):
            with self.subTest(side=side):
                def mutate(m, rows):
                    rows[1] = copy.deepcopy(rows[0])
                self.run_cli(**{side: self.changed(mutate, label=side)}, expected='duplicate (task, index)')

    # The registered seed is checked in each source manifest independently.
    def test_seed_mismatch_refuses(self):
        for side in ('bf16', 'baseline'):
            with self.subTest(side=side):
                self.run_cli(**{side: self.changed(lambda m, rows: m.update(seed=7), label=side)}, expected='seed mismatch')

    # A resealed individual row cannot override the manifest's registered seed.
    def test_row_seed_mismatch_refuses(self):
        self.run_cli(bf16=self.changed(lambda m, rows: rows[0].update(seed=7)), expected='row seed mismatch')

    # Both input manifests must declare the frozen f42a48cc data lock.
    def test_data_lock_mismatch_refuses(self):
        for side in ('bf16', 'baseline'):
            with self.subTest(side=side):
                self.run_cli(**{side: self.changed(lambda m, rows: m.update(input_lock_sha256='0'*64), label=side)},
                             expected='data lock mismatch')

    # Every service model identity, on either side, is bound to the Qwen weight hash.
    def test_model_mismatch_refuses(self):
        for side in ('bf16', 'baseline'):
            with self.subTest(side=side):
                def mutate(m, rows):
                    m['clients']['node']['model_sha256'] = '0'*64
                self.run_cli(**{side: self.changed(mutate, label=side)}, expected='model mismatch')

    # Editing one declared source hash breaks the pin covering all sources and service binaries.
    def test_declared_implementation_mismatch_refuses(self):
        def mutate(m, rows):
            m['declared_arm']['implementation']['source_hashes']['hybrid-eval/ruler_eval.py'] = '0'*64
        self.run_cli(bf16=self.changed(mutate), expected='bf16 implementation identity mismatch')

    # A transport label declaring f32 cannot be admitted on the strength of a fourth-arm name alone.
    def test_declared_dtype_mismatch_refuses(self):
        def mutate(m, rows):
            m['declared_arm']['state_dtype'] = 'f32'
        self.run_cli(bf16=self.changed(mutate), expected='declared bf16 arm identity mismatch')

    # The manifest must bind the exact baseline archive manifest, not just its depth and model.
    def test_declared_baseline_identity_mismatch_refuses(self):
        def mutate(m, rows):
            m['declared_arm']['baseline']['manifest_sha256'] = '0'*64
        self.run_cli(bf16=self.changed(mutate), expected='declared baseline identity mismatch')

    # A resealed bf16 row must retain its link to the corresponding frozen baseline row hash.
    def test_baseline_row_hash_link_mismatch_refuses(self):
        self.run_cli(bf16=self.changed(lambda m, rows: rows[0].update(baseline_row_sha256='0'*64)),
                     expected='baseline row hash link mismatch')

    # Original inputs cannot supply the reserved derived-provenance field that the adapter adds.
    def test_reserved_provenance_field_refuses(self):
        self.run_cli(bf16=self.changed(lambda m, rows: rows[0].update(original_row_sha256='0'*64)),
                     expected='reserved adapter provenance field')

    # Incorrect bf16 trace evidence is rejected even under an otherwise valid declared implementation.
    def test_bf16_trace_identity_mismatch_refuses(self):
        def mutate(m, rows):
            rows[0]['results']['offload_bf16']['trace']['state_dtype'] = 'f32'
        self.run_cli(bf16=self.changed(mutate), expected='bf16 trace identity mismatch')

    # A wrong expected compressed-file checksum refuses before inspecting archive contents.
    def test_archive_hash_failure_refuses(self):
        self.run_cli(bf16=(self.root/'bf16-8192-complete.tar.gz', '0'*64), expected='archive hash mismatch')

    # Every listed member is checked, including an analysis member not used for the adaptation.
    def test_member_hash_failure_refuses(self):
        files = dict(self.archives['bf16-8192-complete.tar.gz'])
        files['analysis.md'] += b'changed'
        self.run_cli(bf16=self.write_archive(files, update_sums=False), expected='member hash mismatch: analysis.md')

    # An unlisted extra file is refused even if all listed member hashes match.
    def test_unlisted_member_refuses(self):
        files = dict(self.archives['bf16-8192-complete.tar.gz'])
        files['extra.txt'] = b'unlisted'
        self.run_cli(bf16=self.write_archive(files, update_sums=False), expected='unlisted archive member')

    # Tar path traversal is rejected without extracting or creating any output.
    def test_unsafe_member_refuses(self):
        files = dict(self.archives['bf16-8192-complete.tar.gz'])
        files['../escape'] = b'bad'
        self.run_cli(bf16=self.write_archive(files), expected='unsafe archive member')
        self.assertFalse((self.work.parent/'escape').exists())

    # Duplicate JSON keys cannot hide a second arm or conflicting identity behind last-key-wins parsing.
    def test_duplicate_json_key_refuses(self):
        files = dict(self.archives['bf16-8192-complete.tar.gz'])
        files['manifest.json'] = b'{"seed":261102,"seed":7}'
        self.run_cli(bf16=self.write_archive(files), expected='duplicate JSON key: seed')

    # An existing destination is preserved byte-for-byte instead of overwritten by an admitted run.
    def test_existing_output_refuses_without_modification(self):
        output = self.work/'adapted'
        output.mkdir()
        (output/'sentinel').write_bytes(b'keep')
        proc = subprocess.run([sys.executable, str(self.adapter), '--bf16-archive', str(self.root/'bf16-8192-complete.tar.gz'),
                               '--bf16-sha256', HASHES['bf16-8192-complete.tar.gz'], '--baseline-archive',
                               str(self.root/'depth-8192-complete.tar.gz'), '--baseline-sha256', HASHES['depth-8192-complete.tar.gz'],
                               '--output', str(output)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn('output already exists', proc.stderr)
        self.assertEqual(list(output.iterdir()), [output/'sentinel'])
        self.assertEqual((output/'sentinel').read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()

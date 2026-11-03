#!/usr/bin/env python3
"""CPU controls for the preregistered score analysis; no models or downloads."""
import copy
import csv
import json
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys
import tempfile
import unittest

from hybrid_eval import Refusal
from ruler_analysis import ARMS, mcnemar
from ruler_equivalence import (TASKS, SEED, analyze_family, bootstrap_distribution, holm, inside,
                               invert_tost, load_depth, newcombe10, score_pair, validate_rows)


def fixture():
    manifest = dict(depth=8192, tasks=list(TASKS), samples_per_task=30, seed=SEED,
                    clients=dict(client=dict(model_sha256='fixture-model')))
    rows = [dict(depth=8192, task=task, index=i, seed=SEED,
                 results={arm: dict(answer='a', score=1., prompt_sha256=f'{task}-{i}') for arm in ARMS})
            for task in TASKS for i in range(30)]
    return rows, manifest


def write_fixture(root, rows, manifest):
    (root/'manifest.json').write_text(json.dumps(manifest))
    (root/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return root/'rows.jsonl'


class EquivalenceTests(unittest.TestCase):
    # Newcombe (1998), Table III, pp. 2641-2642: e=both+, f=left+, g=right+, h=both-.
    # https://www.site.uottawa.ca/~nat/Courses/csi5388/Newcombe.1998.pdf
    # Values are published to four decimals; these include positive, negative and undefined phi.
    def test_newcombe10_published_reference_cells(self):
        cases = [((36, 12, 2, 0), (.0569, .3404)),
                 ((20, 12, 2, 16), (.0562, .3292)),
                 ((18, 12, 2, 18), (.0562, .3290)),
                 ((35, 14, 0, 1), (.1461, .4175)),
                 ((0, 29, 1, 0), (.6666, .9882)),
                 ((54, 0, 0, 0), (-.0664, .0664)),
                 ((30, 0, 0, 24), (-.0358, .0358)),
                 ((27, 0, 0, 27), (-.0351, .0351))]
        for cells, expected in cases:
            with self.subTest(cells=cells):
                for observed, reference in zip(newcombe10(*cells), expected):
                    self.assertAlmostEqual(observed, reference, delta=.000051)
                reverse = newcombe10(cells[0], cells[2], cells[1], cells[3])
                forward = newcombe10(*cells)
                self.assertAlmostEqual(reverse[0], -forward[1])
                self.assertAlmostEqual(reverse[1], -forward[0])

    # A fixed seed reproduces all 10000 draws; changing it changes the draws on nonconstant data.
    def test_bootstrap_seed_determinism(self):
        differences = [-.5, 0., 0., .25, 1.]
        first = bootstrap_distribution(differences)
        self.assertEqual(len(first), 10000)
        self.assertEqual(first, bootstrap_distribution(differences, SEED))
        self.assertNotEqual(first, bootstrap_distribution(differences, SEED+1))

    # Inversion recovers the known normal one-sided tail; it does not double the TOST p-value.
    def test_tost_inversion_known_normal_tail(self):
        normal = NormalDist()
        def interval(tail):
            z = normal.inv_cdf(1-tail)
            return [.003-z*.005, .003+z*.005]
        expected = 1-normal.cdf((.02-.003)/.005)
        self.assertAlmostEqual(invert_tost(interval), expected, places=12)
        self.assertTrue(inside(interval(.025)))

    # Holm sorts by raw p, keeps adjusted p monotone, restores input order, and uses all four tests.
    def test_holm_ordering_and_pending_depth(self):
        self.assertEqual(holm([.03, .004, .02, .001]), [.04, .012, .04, .004])
        self.assertEqual(holm([.004, .001, 1., 1.]), [.012, .004, 1., 1.])

    # Identical binary arms at n=390 have a nonzero method-10 interval inside the margin.
    def test_identical_arms_equivalent_positive_control(self):
        result = score_pair([1.]*380+[0.]*10, [1.]*380+[0.]*10)
        self.assertTrue(result['equivalent_unadjusted'])
        self.assertLess(4*result['tost_p'], .025)
        self.assertLess(result['interval95'][0], 0)
        self.assertGreater(result['interval95'][1], 0)

    # A 10 pp shift fails equivalence, even though both arms are valid binary inputs.
    def test_shifted_fixture_not_equivalent_positive_control(self):
        result = score_pair([1.]*390, [1.]*351+[0.]*39)
        self.assertFalse(result['equivalent_unadjusted'])
        self.assertEqual(result['tost_p'], 1.)

    # One fractional score selects the paired bootstrap for the whole comparison, including binary pairs.
    def test_mixed_scores_use_whole_set_bootstrap(self):
        result = score_pair([1., .5, 0.], [1., 0., 0.])
        self.assertEqual(result['method'], 'paired_percentile_bootstrap')
        self.assertAlmostEqual(result['difference'], 1/6)
        self.assertEqual(result['interval95'], [0., .5])

    # Resampling whole pairs preserves identical partial scores; finite draws cannot claim p=0 precision.
    def test_identical_partial_pairs_and_bootstrap_resolution(self):
        result = score_pair([1., .5, 0.], [1., .5, 0.])
        self.assertEqual(result['method'], 'paired_percentile_bootstrap')
        self.assertEqual(result['interval95'], [0., 0.])
        self.assertEqual(result['tost_p'], .0001)
        self.assertTrue(result['tost_p_resolution_limited'])
        self.assertTrue(result['equivalent_unadjusted'])

    # A margin touch or crossing fails the strict open interval, with a nearby inside control.
    def test_interval_crossing_margin_is_not_equivalent(self):
        self.assertFalse(inside([-.02, .01]))
        self.assertFalse(inside([-.01, .0201]))
        self.assertFalse(inside([-.01, .02]))
        self.assertTrue(inside([-.0199, .0199]))
        small = score_pair([1.]*30, [1.]*30)
        self.assertFalse(small['equivalent_unadjusted'])
        self.assertGreater(small['tost_p'], .025)

    # McNemar p=1 at n=30 cannot certify +/-2 pp; identical arms at n=390 above can.
    def test_nonsignificant_mcnemar_never_implies_equivalence(self):
        self.assertEqual(mcnemar(0, 0), 1.)
        result = score_pair([1.]*30, [1.]*30)
        self.assertFalse(result['equivalent_unadjusted'])

    # An arm missing even one example is refused, alongside a valid paired positive control.
    def test_mismatched_example_sets_refuse(self):
        rows, manifest = fixture()
        self.assertEqual(len(validate_rows(rows, manifest)), 390)
        del rows[0]['results']['offload']
        with self.assertRaisesRegex(Refusal, 'missing or unknown arms'):
            validate_rows(rows, manifest)

    # Shared truncation and duplicate replacement cannot hide behind matching arm sets or n=390.
    def test_missing_rows_and_duplicate_rows_refuse(self):
        rows, manifest = fixture()
        with self.assertRaisesRegex(Refusal, 'missing or unexpected rows'):
            validate_rows(rows[:-1], manifest)
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaisesRegex(Refusal, 'duplicate paired example'):
            validate_rows(rows, manifest)

    # Row, manifest and explicit per-arm seed changes each refuse before statistical analysis.
    def test_changed_seed_refuses(self):
        for scope in ('row', 'manifest', 'arm'):
            rows, manifest = fixture()
            if scope == 'row':
                rows[0]['seed'] += 1
            elif scope == 'manifest':
                manifest['seed'] += 1
            else:
                rows[0]['results']['offload']['seed'] = SEED+1
            with self.subTest(scope=scope), self.assertRaisesRegex(Refusal, 'changed seed'):
                validate_rows(rows, manifest)

    # Prompt mismatches, nonfinite scores and forged row checksums fail with specific refusals.
    def test_invalid_pair_identity_and_score_refuse(self):
        for change, message in [('prompt', 'paired prompts differ'), ('score', 'invalid response'),
                                ('hash', 'row checksum differs')]:
            rows, manifest = fixture()
            if change == 'prompt':
                rows[0]['results']['offload']['prompt_sha256'] = 'different'
            elif change == 'score':
                rows[0]['results']['offload']['score'] = float('nan')
            else:
                rows[0]['row_sha256'] = 'forged'
            with self.subTest(change=change), self.assertRaisesRegex(Refusal, message):
                validate_rows(rows, manifest)

    # CSV requires verifiable seed/prompt metadata and agrees with JSONL; edited scores refuse.
    def test_csv_companion_metadata_and_score_validation(self):
        rows, manifest = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_fixture(root, rows, manifest)
            path = root/'per-example.csv'
            items = []
            for row in rows:
                item = {k: row[k] for k in ('depth', 'task', 'index')}
                for arm in ARMS:
                    for field in ('score', 'answer'):
                        item[arm+'_'+field] = row['results'][arm][field]
                items.append(item)
            def write_csv():
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(items[0]))
                    writer.writeheader()
                    writer.writerows(items)
            write_csv()
            self.assertEqual(load_depth(path)[:2], load_depth(source)[:2])
            items[0]['offload_score'] = .5
            write_csv()
            with self.assertRaisesRegex(Refusal, 'CSV response differs'):
                load_depth(path)
            source.unlink()
            with self.assertRaisesRegex(Refusal, 'CSV requires companion'):
                load_depth(path)

    # Optional bf16 pairs stay in separate four-depth families; missing depths remain explicit.
    def test_bf16_pairs_and_provisional_holm_family(self):
        rows, manifest = fixture()
        for row in rows:
            row['results']['bf16'] = copy.deepcopy(row['results']['offload'])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_fixture(Path(tmp), rows, manifest)
            result = analyze_family([path], 'fixture', [8192, 32768, 65536, 131072])
            self.assertEqual({r['pair'] for r in result['primary']},
                             {'offload-local', 'local_partition-local', 'bf16-local', 'bf16-offload'})
            for r in result['primary']:
                self.assertTrue(r['provisional'])
                self.assertEqual(r['pending_depths'], [32768, 65536, 131072])
                self.assertAlmostEqual(r['holm_p'], 4*r['tost_p'])
                self.assertTrue(r['equivalent_holm'])

    # Different model identities cannot be pooled, and duplicate depth archives cannot inflate n.
    def test_models_and_duplicate_depths_refuse(self):
        rows, manifest = fixture()
        depths = [8192, 32768, 65536, 131072]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = write_fixture(root, rows, manifest)
            with self.assertRaisesRegex(Refusal, 'unexpected or duplicate depth'):
                analyze_family([first, first], 'fixture', depths)
            other = root/'other'
            other.mkdir()
            manifest['depth'] = 32768
            manifest['clients']['client']['model_sha256'] = 'other-model'
            for row in rows:
                row['depth'] = 32768
            second = write_fixture(other, rows, manifest)
            with self.assertRaisesRegex(Refusal, 'different models'):
                analyze_family([first, second], 'fixture', depths)

    # Four completed depths retain the existing secondary endpoint's family-wide Holm correction.
    def test_completed_family_preserves_secondary_holm(self):
        depths = [8192, 32768, 65536, 131072]
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for depth in depths:
                rows, manifest = fixture()
                manifest['depth'] = depth
                for row in rows:
                    row['depth'] = depth
                root = Path(tmp)/str(depth)
                root.mkdir()
                paths.append(write_fixture(root, rows, manifest))
            result = analyze_family(paths, 'fixture', depths)
            self.assertTrue(all(not r['provisional'] for r in result['primary']))
            for table in result['secondary']:
                if table['task'] == 'ALL':
                    self.assertIn('mcnemar_holm_four_depths', table)
                    self.assertNotIn('mcnemar_bonferroni_four_depths', table)

    # CLI refusal exits 2 and creates no output artifacts, even for malformed input.
    def test_cli_refusal_no_partial_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root/'output'
            proc = subprocess.run([sys.executable, str(Path(__file__).with_name('ruler_equivalence.py')),
                                   str(root/'missing.jsonl'), '--family', 'fixture', '--output', str(output)],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn('manifest.json required', proc.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()

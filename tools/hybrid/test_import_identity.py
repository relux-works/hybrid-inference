"""Compare complete native slot and logit identities against a separate reference build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ImportIdentityTests(unittest.TestCase):
    args = None

    def setUp(self):
        if self.args is None:
            self.skipTest('requires explicit reference/candidate builds and local models')

    def test_cross_build_state_and_full_logits(self):
        # Both builds must match full-state and 64-position full-vocabulary logit identities for fresh/append on each selected model/fixture.
        records = []
        comparisons = []
        for model_index, model in enumerate(self.args.model):
            for fixture_index, golden in enumerate(self.args.golden):
                with self.subTest(model=model, golden=golden):
                    results = []
                    for arm in ('before', 'after'):
                        binary = Path(getattr(self.args, arm)).resolve()
                        name = f'{arm}-{model_index}-{fixture_index}'
                        command = [str(binary), str(Path(model).resolve()), '--identity', str(Path(golden).resolve())]
                        env = dict(os.environ)
                        env['LD_LIBRARY_PATH'] = str(binary.parent) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
                        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
                        path = self.args.artifacts / (name + '.log')
                        with path.open('wb') as log:
                            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
                            code = process.wait()
                        records.append(dict(command=command, pid=process.pid, exit_code=code, binary_sha256=digest))
                        (self.args.artifacts / 'processes.json').write_text(json.dumps(records, indent=2) + '\n')
                        self.assertEqual(code, 0, path.read_text()[-2000:])
                        self.assertEqual(hashlib.sha256(binary.read_bytes()).hexdigest(), digest, 'binary changed during identity check')
                        rows = [json.loads(line.removeprefix('IDENTITY ')) for line in path.read_text().splitlines()
                                if line.startswith('IDENTITY ')]
                        self.assertEqual([row['fresh'] for row in rows], [True, False])
                        for row in rows:
                            self.assertGreater(row['state_bytes'], 0)
                            self.assertGreater(row['logit_bytes'], 0)
                        results.append(rows)
                    comparisons.append(dict(model=model, golden=golden, equal=results[0] == results[1], before=results[0], after=results[1]))
                    (self.args.artifacts / 'result.json').write_text(json.dumps(comparisons, indent=2) + '\n')
                    self.assertEqual(results[0], results[1], 'cross-build state/logit identity differs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True, help='reference llama-hybrid-check-client')
    parser.add_argument('--after', required=True, help='candidate llama-hybrid-check-client')
    parser.add_argument('--model', action='append', required=True)
    parser.add_argument('--golden', action='append', required=True, help='golden filename stem')
    parser.add_argument('--artifacts', type=Path, required=True)
    ImportIdentityTests.args = parser.parse_args()
    ImportIdentityTests.args.artifacts.mkdir(parents=True, exist_ok=False)
    unittest.main(argv=[sys.argv[0]])

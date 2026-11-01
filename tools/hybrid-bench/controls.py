"""Run cleanup mutants from copied source; an unchanged module must pass first."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


def worker(test, result_path):
    import test_bench
    module = test_bench.hybrid_bench
    suite = unittest.defaultTestLoader.loadTestsFromName('test_bench.CleanupTests' + ('.' + test if test else ''))
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    Path(result_path).write_text(json.dumps(dict(
        module=str(Path(module.__file__).resolve()),
        bound_namespace=module.main.__globals__ is module.__dict__ and test_bench.bundle is module.bundle,
        tests=result.testsRun, skipped=result.skipped,
        failures=[dict(test=case.id(), traceback=trace) for case, trace in result.failures],
        errors=[dict(test=case.id(), traceback=trace) for case, trace in result.errors]), indent=2) + '\n')
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    controls = [
        ('identity', 'unchanged source passes the same mocked cleanup tests', '', None, '', '', ''),
        ('equivalence-not-journaled', 'raw checker output alone', 'test_checker_result_survives_final_delete',
         'hybrid-bench/hybrid_bench.py', '        record(result)\n', '', "['checker', 'equivalence']"),
        ('checker-outcome-not-journaled', 'validated metrics alone', 'test_checker_result_survives_final_delete',
         'hybrid-bench/hybrid_bench.py',
         "        record(dict(stage='checker', checker_exit_code=process.returncode,\n"
         "                    check_mode='append_chain' if boundaries is not None else 'fresh', depth=len(tokens),\n"
         "                    result_json=result_path.read_text() if result_path.is_file() else None))\n", '',
         "['checker', 'equivalence']"),
        ('truncate-final-journal', 'partial final write can destroy checker evidence', 'test_bundle_failure_keeps_journal',
         'hybrid-bench/hybrid_bench.py', "    temporary = root / (name + '.jsonl.tmp')", '    temporary = path',
         'durable checker journal changed'),
        ('no-transport-retry', 'first connection error becomes final', 'test_transient_delete_recovers',
         'hybrid-bench/hybrid_bench.py', '        release_retry(delete)', '        delete()', 'AssertionError: 2 != 0'),
        ('retry-401', 'HTTP refusals treated as transient', 'test_unauthorized_delete_not_retried',
         'hybrid-proxy/release.py', 'if status is not None or not isinstance', 'if not isinstance', 'AssertionError: 3 != 1'),
        ('ignore-release-exit', 'failed DELETE can leave overall success', 'test_checker_result_survives_final_delete',
         'hybrid-bench/hybrid_bench.py', '    if release_errors:\n', '    if False and release_errors:\n', '!= 2'),
        ('suppress-active-error', 'cleanup can swallow prior failure', 'test_release_preserves_original_error_and_redacts',
         'hybrid-bench/hybrid_bench.py', '        if not released and not active_error:\n',
         '        if not released and active_error:\n            return\n        if not released and not active_error:\n',
         'Refusal not raised'),
        ('interrupt-not-journaled', 'Ctrl-C preserves checker rows but loses leftover IDs', 'test_interrupt_during_checker_release',
         'hybrid-bench/hybrid_bench.py', '    except (Exception, KeyboardInterrupt) as error:', '    except Exception as error:',
         'AssertionError: False is not true'),
        ('short-mode-timeout', 'mode release inherits the shorter checker timeout', 'test_release_timeout_and_mode_interrupt',
         'hybrid-bench/hybrid_bench.py', 'timeout=30 if scope == \'mode\' else 10', 'timeout=10', 'AssertionError: 10 != 30'),
    ]
    identity = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for directory in ('hybrid-bench', 'hybrid-proxy') for path in (root / 'tools' / directory).glob('*.py')}
    (args.output / 'source-identity.json').write_text(json.dumps(identity, indent=2, sort_keys=True) + '\n')
    rows = []
    for name, bound, test, target, old, new, assertion in controls:
        row = dict(mutant=name, narrows_gate_to=bound, named_test=test or 'CleanupTests', killed=False,
                   survivor_bound=bound, status='not-applied')
        with tempfile.TemporaryDirectory(prefix=name + '-', dir=args.output) as temporary:
            copied = Path(temporary) / 'tools'
            for directory in ('hybrid-bench', 'hybrid-proxy'):
                (copied / directory).mkdir(parents=True)
                for path in (root / 'tools' / directory).glob('*.py'):
                    shutil.copyfile(path, copied / directory / path.name)
            if target:
                path = copied / target
                source = path.read_text()
                if source.count(old) != 1 or old == new:
                    row['reason'] = 'plant must replace exactly one source occurrence'
                    rows.append(row)
                    break
                path.write_text(source.replace(old, new))
            result_path = (args.output / (name + '.json')).resolve()
            command = [sys.executable, '-B', str(copied / 'hybrid-bench/controls.py'), '--worker', test, str(result_path)]
            row.update(command=command, module=str((copied / 'hybrid-bench/hybrid_bench.py').resolve()))
            with (args.output / (name + '.log')).open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                row['pid'] = process.pid
                (args.output / (name + '-process.json')).write_text(json.dumps(row, indent=2) + '\n')
                try:
                    row['exit_code'] = process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    row['exit_code'] = process.returncode
                    row['reason'] = 'control exceeded 60 s'
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                valid = result['module'] == row['module'] and result['bound_namespace'] and result['tests'] > 0 and not result['errors'] and not result['skipped']
                row.update(failures=result['failures'], errors=result['errors'])
                if name == 'identity':
                    row['status'] = 'survived' if valid and row['exit_code'] == 0 and not result['failures'] else 'invalid'
                else:
                    row['killed'] = valid and row['exit_code'] == 1 and bool(result['failures']) and all(
                        test in failure['test'] and assertion in failure['traceback'] for failure in result['failures'])
                    row['status'] = 'killed' if row['killed'] else 'survived' if valid and row['exit_code'] == 0 else 'invalid'
                    if row['killed']:
                        row['survivor_bound'] = None
            else:
                row['status'] = 'invalid'
            rows.append(row)
        if name == 'identity' and row['status'] != 'survived':
            break
    (args.output / 'controls.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        print(row['mutant'] + ': ' + row['status'])
    return 0 if len(rows) == len(controls) and rows[0]['status'] == 'survived' and all(row['killed'] for row in rows[1:]) else 1


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        sys.exit(worker(sys.argv[2], sys.argv[3]))
    sys.exit(main())

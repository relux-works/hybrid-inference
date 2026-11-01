"""Opt-in real 0.8B CPU study test: D=3, N=64, M=16; no downloads."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket
import statistics
import random
import sys
import time
import urllib.error

import hetero_study as study


def test_client_schedule_floor(result):
    # Q4 fresh-prefix probes must measure the known batch-shape variation on this fixed CPU fixture.
    assert result.get('schedule_floor', {}).get('kind') == 'fresh-prefix-spots-v1', 'client schedule floor was omitted'
    assert result['schedule_floor']['local']['max'] > 1e-5, 'client schedule floor was omitted'


def test_homogeneous_row0_control(result):
    # Same-model, same-partition imports must not add the former full-prefill vs scalar row-0 difference.
    assert result['positions'][0]['local_to_hybrid_kl_nats'] < 1e-12, 'row0 schedule asymmetry remains'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--build-dir', default='build')
    parser.add_argument('--artifacts', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root = study.private_dir(args.artifacts)
    driver = Path(__file__).with_name('hetero_study.py').resolve()
    lock = root / 'gpu.lock'
    commands = []

    def cli(name, arguments, expected=0, reason=None):
        command = [sys.executable, driver, '--lock', lock] + arguments
        with study.child(command, root, name) as process:
            code = process.wait(timeout=600)
        commands.append(dict(name=name, command=[str(v) for v in command], exit_code=code, expected=expected))
        study.save(root / 'commands.json', commands)
        log = (root / (name + '.log')).read_text()
        assert code == expected, (name, code, log[-2000:])
        if reason:
            assert reason in log, (name, reason, log[-2000:])
        print('PASS ' + name + ': exit ' + str(code), flush=True)

    # Three synthetic text documents use a real llama-server tokenizer, with no document retrieval.
    docs = study.private_dir(root / 'documents')
    manifest = root / 'manifest.json'
    rows = []
    for i in range(3):
        name = 'doc' + str(i)
        text = ('The river carries water past the old bridge. ' if i == 0 else 'A small observatory records the stars at night. ' if i == 1 else 'The green garden has trees and a wooden gate. ') * 32
        path = docs / (name + '.txt')
        path.write_text(text)
        rows.append(dict(id=name, url='https://www.gutenberg.org/cache/epub/84/pg84.txt', size=path.stat().st_size, sha256=study.sha256(path)))
    study.save(manifest, dict(documents=rows))
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    command = [Path(args.build_dir).resolve() / 'bin/llama-server', '-m', args.reference, '-ngl', '0', '-c', '256', '-t', '2', '-tb', '2',
               '--host', '127.0.0.1', '--port', port, '--no-webui']
    prepared = root / 'study'
    prepare_args = ['prepare', '--manifest', manifest, '--documents', docs, '--model', args.reference, '--tokenizer-url', 'http://127.0.0.1:' + str(port),
                    '--output', prepared, '--d', '3', '--n', '64', '--m', '16']
    with study.child(command, root, 'tokenizer') as process:
        deadline = time.monotonic() + 180
        while True:
            assert process.poll() is None, 'tokenizer startup failed'
            try:
                study.api('http://127.0.0.1:' + str(port), '/health', timeout=1)
                break
            except (urllib.error.URLError, TimeoutError):
                assert time.monotonic() < deadline, 'tokenizer readiness timeout'
                time.sleep(.1)
        cli('prepare', prepare_args)
        cli('prepare-resume', prepare_args)
        bad_rows = [dict(r) for r in rows]
        bad_rows[0]['sha256'] = '0' * 64
        bad = root / 'bad-manifest.json'
        study.save(bad, dict(documents=bad_rows))
        negative = list(prepare_args)
        negative[negative.index('--manifest') + 1] = bad
        negative[negative.index('--output') + 1] = root / 'bad-hash'
        cli('manifest-mismatch', negative, 2, 'manifest hash mismatch')
        negative = list(prepare_args)
        negative[negative.index('--n') + 1] = '4096'
        negative[negative.index('--output') + 1] = root / 'short'
        cli('short-document', negative, 2, 'document shorter than N+M')
    # The tokenizer child is stopped and reaped before any scorer or prefill process starts.
    build = ['--build-dir', args.build_dir, '--gpu-layers', '0', '--score-batch', '4']
    cli('reference', ['reference', '--study', prepared, '--model', args.reference] + build)
    index, fixtures = study.load_study(prepared)
    refhash = study.sha256(args.reference)
    arm_args = ['arm', '--study', prepared, '--reference-model-sha256', refhash] + build
    cli('self', arm_args + ['--kind', 'local', '--label', 'self', '--client-model', args.reference])
    cli('local', arm_args + ['--kind', 'local', '--label', 'local', '--client-model', args.client])
    imported = arm_args + ['--kind', 'imported', '--label', 'imported', '--client-model', args.client,
                           '--node-model', args.reference, '--hetero-quant', '--chunk-tokens', '32', '--ram-budget-mib', '128']
    cli('imported', imported)
    spec_path = root / 'homogeneous.json'
    study.save(spec_path, dict(kind='imported', label='homogeneous', client_model=args.client, node_model=args.client))
    cli('homogeneous', arm_args + ['--spec', spec_path, '--chunk-tokens', '8192', '--ram-budget-mib', '128'])
    before = {str(p): study.sha256(p) for p in prepared.rglob('*.process.json')}
    cli('reference-resume', ['reference', '--study', prepared, '--model', args.reference] + build)
    cli('imported-resume', imported)
    assert {str(p): study.sha256(p) for p in prepared.rglob('*.process.json')} == before

    # D || D stays at its own measured encoding floor; both Q4 arms contain exactly M rows per document.
    for doc in index['documents']:
        name = doc['id'] + '.json'
        reference = study.receipt(prepared / 'references' / name)
        encoding = study.read(reference['outputs'][1]['path'])['encoding']['kl_nats']['max']
        record = study.receipt(prepared / 'arms/self' / name)
        same_rows = study.read(record['outputs'][0]['path'])
        assert max(r['kl_nats'] for r in same_rows) <= max(1e-12, 1.1 * encoding)
        for label in ('local', 'imported'):
            record = study.receipt(prepared / 'arms' / label / name)
            assert len(study.read(record['outputs'][0]['path'])) == 16
        local_record = study.receipt(prepared / 'arms/local' / name)
        test_client_schedule_floor(study.read(local_record['outputs'][1]['path']))
        homogeneous = study.receipt(prepared / 'arms/homogeneous' / name)
        test_homogeneous_row0_control(study.read(homogeneous['outputs'][1]['path']))

    # Report statistics independently recompute from saved rows; a repeated seed gives byte-identical reports.
    output = root / 'report'
    report_args = ['report', '--study', prepared, '--arms', 'self', 'local', 'imported', '--seed', '91', '--output', output]
    cli('report', report_args)
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    cli('report-repeat', report_args)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original
    table = study.read(output / 'report.json')['rows']
    csv_rows = list(csv.DictReader((output / 'report.csv').open()))
    assert len(csv_rows) == len(table)
    for row, csv_row in zip(table, csv_rows):
        assert all(str(value) == csv_row[key] for key, value in row.items())
    doc_means = {}
    def percentile(values, p):
        values = sorted(values)
        position = (len(values) - 1) * p
        low, high = math.floor(position), math.ceil(position)
        return values[low] * (high - position) + values[high] * (position - low) if low != high else values[low]
    def interval(values):
        rng = random.Random(91)
        samples = [sum(rng.choices(values, k=len(values))) / len(values) for _ in range(10000)]
        return percentile(samples, .025), percentile(samples, .975)
    document_metrics = {}
    for label in ('self', 'local', 'imported'):
        groups = [study.read(study.receipt(prepared / 'arms' / label / (doc['id'] + '.json'))['outputs'][0]['path']) for doc in index['documents']]
        doc_means[label] = [sum(r['kl_nats'] for r in group) / 16 for group in groups]
        for metric, key in [('kl_nats', 'kl_nats'), ('top1', 'top1_agreed')]:
            values = [r[key] for group in groups for r in group]
            row = next(r for r in table if r['scope'] == 'positions' and r['arm'] == label and r['metric'] == metric)
            assert math.isclose(row['mean'], sum(values) / len(values), abs_tol=1e-14)
            assert row['max'] == max(values)
            for name, p in [('median', .5), ('p90', .9), ('p99', .99)]:
                assert math.isclose(row[name], percentile(values, p), abs_tol=1e-14)
            per_doc = [sum(r[key] for r in group) / len(group) for group in groups]
            document_metrics[label, metric] = per_doc
            aggregate = next(r for r in table if r['scope'] == 'documents' and r['arm'] == label and r['metric'] == metric)
            low, high = interval(per_doc)
            assert aggregate['count'] == 3 and math.isclose(aggregate['mean'], sum(per_doc) / 3, abs_tol=1e-14)
            assert math.isclose(aggregate['ci_low'], low, abs_tol=1e-14) and math.isclose(aggregate['ci_high'], high, abs_tol=1e-14)
            individual = [r['mean'] for r in table if r['scope'] == 'document' and r['arm'] == label and r['metric'] == metric]
            assert all(math.isclose(a, b, abs_tol=1e-14) for a, b in zip(individual, per_doc))
    pair = next(r for r in table if r['scope'] == 'paired' and r['arm'] == 'imported' and r['other'] == 'local' and r['metric'] == 'kl_nats')
    assert math.isclose(pair['mean'], statistics.fmean(b - a for a, b in zip(doc_means['local'], doc_means['imported'])), abs_tol=1e-14)
    for row in (r for r in table if r['scope'] == 'paired'):
        differences = [b - a for a, b in zip(document_metrics[row['other'], row['metric']], document_metrics[row['arm'], row['metric']])]
        low, high = interval(differences)
        assert row['count'] == 3 and math.isclose(row['mean'], sum(differences) / 3, abs_tol=1e-14)
        assert math.isclose(row['ci_low'], low, abs_tol=1e-14) and math.isclose(row['ci_high'], high, abs_tol=1e-14)

    # Each invalid model/document/delta/fixture input refuses distinctly, with no report files published.
    def rejected_report(name, reason):
        bad_output = root / (name + '-report')
        negative = list(report_args)
        negative[negative.index('--output') + 1] = bad_output
        cli(name, negative, 2, reason)
        assert not bad_output.exists()

    reference_path = prepared / 'references/doc0.json'
    original_reference = reference_path.read_bytes()
    wrong = study.read(prepared / 'references/doc1.json')
    study.save(reference_path, wrong)
    cli('another-document-arm', arm_args + ['--kind', 'local', '--label', 'wrong-document', '--client-model', args.client], 2, 'reference document mismatch')
    rejected_report('another-document', 'reference document mismatch')
    reference_path.write_bytes(original_reference)
    wrong_hash = list(arm_args)
    wrong_hash[wrong_hash.index('--reference-model-sha256') + 1] = study.sha256(args.client)
    cli('another-model', wrong_hash + ['--kind', 'local', '--label', 'wrong-model', '--client-model', args.client], 2, 'reference model hash mismatch')
    config_path = prepared / 'arms/local/arm.json'
    original_config = config_path.read_bytes()
    wrong_config = study.read(config_path)
    wrong_config['reference_model_sha256'] = study.sha256(args.client)
    study.save(config_path, wrong_config)
    rejected_report('another-model-report', 'report reference model mismatch')
    config_path.write_bytes(original_config)
    record = study.receipt(prepared / 'arms/imported/doc0.json')
    delta = Path(record['outputs'][2]['path'])
    backup = delta.with_suffix('.saved')
    delta.rename(backup)
    cli('missing-delta-resume', imported, 2, 'missing arm rows/delta')
    rejected_report('missing-delta', 'missing arm rows/delta')
    backup.rename(delta)
    fixture_path = Path(index['documents'][0]['fixture']['path'])
    original_fixture = fixture_path.read_bytes()
    changed = study.read(fixture_path)
    changed['tokens'][0] += 1
    study.save(fixture_path, changed)
    cli('changed-fixture-resume', imported, 2, 'stale fixture: hash mismatch')
    rejected_report('changed-fixture', 'stale fixture: hash mismatch')
    fixture_path.write_bytes(original_fixture)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original
    study.save(root / 'measurements.json', dict(report=study.read(output / 'report.json'), reference_model_sha256=refhash,
                                              client_model_sha256=study.sha256(args.client), documents=3, n=64, m=16))
    print('ALL CPU STUDY TESTS PASSED', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--schedule-witness':
        test_client_schedule_floor(study.read(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == '--row0-witness':
        test_homogeneous_row0_control(study.read(sys.argv[2]))
    else:
        main()

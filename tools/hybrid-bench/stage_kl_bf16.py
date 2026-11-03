#!/usr/bin/env python3
"""Fresh GPU-host KL stage. Missing historical provenance is a refusal, not a reconstruction."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.parse

import hetero_study as study
from gpu_stage import build, clean_source, disk, exclusive, gpu_identity, interrupted, read, require, run, save, sha, terminal

LABELS = ['state-Q4-local', 'state-f32', 'state-bf16']


def historical_report(archive):
    with tarfile.open(archive) as bundle:
        path = 'report27b-seed42/report.json'
        member = bundle.getmember(path)
        require(member.isfile() and member.size < 16*1024**2, 'invalid historical report')
        stream = bundle.extractfile(member)
        if stream is None:
            raise RuntimeError('missing historical report')
        raw = stream.read()
    import hashlib
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def audit(report):
    ids = [r['document'] for r in report['rows'] if r['scope'] == 'document' and
           r['arm'] == 'D-local' and r['metric'] == 'kl_nats']
    require(len(ids) == len(set(ids)) == 20 and report['settings']['seed'] == 42, 'historical D/seed differ')
    return dict(documents=ids, report_has_document_hashes=False, report_has_token_hashes=False,
                report_has_model_pins=False, same_partition_f32_row=None,
                available_imported_arms=[a for a in report['settings']['arms'] if 'imported' in a],
                constraint='Aggregate rows and an index digest cannot recover document/token hashes or a missing Q4-to-Q4 arm.')


def validate_lock(lock, report, report_sha, mode):
    evidence = audit(report)
    require(lock['version'] == 1 and lock['historical_report_sha256'] == report_sha, 'provenance lock/report differs')
    require(lock['settings'] == dict(seed=42, d=20, n=32768, m=2048, score_batch=32) and all(type(v) is int for v in lock['settings'].values()), 'locked study settings differ')
    docs = lock['documents']
    require([r['id'] for r in docs] == evidence['documents'], 'document order differs from historical report')
    for row in docs:
        require(re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')) and
                re.fullmatch('[0-9a-f]{64}', row.get('token_sha256', '')) and
                type(row.get('size')) is int and 0 < row['size'] <= 64*1024**2,
                'missing document/token hash: ' + row['id'])
    require(set(lock['models']) == {'q4', 'bf16'}, 'exactly q4 and bf16 model pins required')
    for model in ('q4', 'bf16'):
        pin = lock['models'][model]
        require(re.fullmatch('[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', pin['repo']) and
                re.fullmatch('[0-9a-f]{40}', pin['revision']), 'model requires immutable repository revision')
        files = pin['files']
        require(files and pin['primary'] in [f['name'] for f in files] and
                len({f['name'] for f in files}) == len(files), 'invalid model shard list')
        for item in files:
            require(Path(item['name']).name == item['name'] and item['name'].endswith('.gguf') and
                    re.fullmatch('[0-9a-f]{64}', item['sha256']) and type(item['size']) is int and item['size'] > 0,
                    'model requires pinned GGUF size and SHA256')
    require(lock['models'] == read(Path(__file__).with_name('state_bf16_model_pins.json'))['models'], 'model pins differ from recovered Qwen3.8 revisions')
    require(lock['provenance']['kind'] == ('recovered-stage2-lock' if mode == 'reproduce' else 'newly-pinned-study'),
            'provenance kind differs from requested mode')
    require(bool(lock['provenance']['source']), 'provenance source must be stated')
    if mode == 'reproduce':
        comparator = lock.get('f32_comparator')
        require(comparator is not None and comparator['arm'] == 'B-Q4-Q4-imported' and comparator['arm'] in report['settings']['arms'] and
                comparator['node_model_sha256'] == comparator['client_model_sha256'] ==
                next(f['sha256'] for f in lock['models']['q4']['files'] if f['name'] == lock['models']['q4']['primary']), 'missing homogeneous stage-2 f32 comparator')
        require(comparator['arm'].endswith('-imported'), 'f32 comparator is not an imported arm')
    return evidence


def download_models(lock, root):
    result = {}
    for label, pin in lock['models'].items():
        folder = root/label
        folder.mkdir(exist_ok=True)
        for item in pin['files']:
            target = folder/item['name']
            if not target.exists():
                temporary = target.with_suffix('.partial')
                require(not temporary.exists(), 'partial model download exists; preserve/inspect it before retrying')
                disk(root, item['size'] + 8*1024**3)
                url = 'https://huggingface.co/' + pin['repo'] + '/resolve/' + pin['revision'] + '/' + urllib.parse.quote(item['name'])
                run(['curl', '--fail', '--location', '--proto', '=https', '--proto-redir', '=https', '--retry', '3',
                     '--max-filesize', str(item['size']), '--output', temporary, url], root/'download.log')
                require(temporary.stat().st_size == item['size'] and sha(temporary) == item['sha256'], 'downloaded model hash differs')
                temporary.replace(target)
            require(target.stat().st_size == item['size'] and sha(target) == item['sha256'], 'model pin differs')
        result[label] = folder/pin['primary']
    return result


@contextlib.contextmanager
def tokenizer(binary, model, root):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(binary), '-m', str(model), '--host', '127.0.0.1', '--port', str(port), '-ngl', '999',
               '-c', '34816', '-np', '1', '-b', '256', '-ub', '256', '-t', '2', '-tb', '2', '-ctk', 'q8_0', '-ctv', 'q8_0']
    with (root/'tokenizer.log').open('ab') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        save(root/'tokenizer-process.json', dict(pid=process.pid, command=command))
        try:
            url = f'http://127.0.0.1:{port}'
            deadline = time.monotonic()+900
            while True:
                require(process.poll() is None, 'tokenizer exited; inspect tokenizer.log')
                try:
                    study.api(url, '/health', timeout=3)
                    break
                except (OSError, ValueError):
                    require(time.monotonic() < deadline, 'tokenizer startup timeout')
                    time.sleep(1)
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            save(root/'tokenizer-stopped.json', dict(pid=process.pid, returncode=process.returncode))


def verify_tokens(index, fixtures, pins):
    require([d['id'] for d in index['documents']] == [d['id'] for d in pins], 'prepared document order differs')
    for doc, pin in zip(index['documents'], pins):
        require(doc['source']['sha256'] == pin['sha256'] and study.token_hash(fixtures[doc['id']]) == pin['token_sha256'],
                'prepared document/token hash differs: ' + pin['id'])


def compare_intervals(current, previous, old_arm, new_arm):
    checks = []
    for metric in ('kl_nats', 'top1'):
        old = [r for r in previous['rows'] if r['scope'] == 'documents' and r['arm'] == old_arm and r['metric'] == metric]
        new = [r for r in current['rows'] if r['scope'] == 'documents' and r['arm'] == new_arm and r['metric'] == metric]
        require(len(old) == len(new) == 1 and old[0]['count'] == new[0]['count'] == 20, 'missing comparator document interval')
        checks.append(dict(metric=metric, historical_arm=old_arm, current_arm=new_arm, current_mean=new[0]['mean'],
                           historical_low=old[0]['ci_low'], historical_high=old[0]['ci_high'],
                           passed=old[0]['ci_low'] <= new[0]['mean'] <= old[0]['ci_high']))
    return checks


def bootstrap_new_lock(args, previous, report_sha):
    require(args.mode == 'new-study' and args.execute and args.provenance_lock is None,
            'bootstrap-new-lock requires explicit new-study execution and no supplied lock')
    require(args.root is not None and args.build is not None and args.ref is not None, 'bootstrap requires root/build/ref')
    gpu = gpu_identity()
    source = Path(__file__).resolve().parents[2]
    clean_source(source, args.ref)
    root = args.root.resolve()
    require(root.is_dir() and shutil.disk_usage(root).total >= 160_000_000_000, 'create a GPU volume of at least 160 GB')
    target = root/'new-provenance-lock.json'
    require(not target.exists(), 'new provenance lock already exists; use it for the study instead of repinning')
    pins = read(Path(__file__).with_name('state_bf16_model_pins.json'))['models']
    disk(root, sum(f['size'] for p in pins.values() for f in p['files']) + 20_344_176_640 + 14*1024**3)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(root/'tmp'))
    (root/'tmp').mkdir(exist_ok=True)
    with exclusive(root/'stage.lock'):
        with exclusive(root/'build.lock'):
            build(source, args.build.resolve(), args.ref, root/'build.log')
        (root/'models').mkdir(exist_ok=True)
        models = download_models(dict(models=pins), root/'models')
        sources = Path(__file__).with_name('gutenberg_sources.json')
        require([r['id'] for r in read(sources)['documents']] == audit(previous)['documents'], 'source IDs/order differ')
        with exclusive(root/'gpu.lock'):
            study.fetch(argparse.Namespace(manifest=str(sources), output=str(root/'documents')))
            with tokenizer(args.build/'bin/llama-server', models['q4'], root) as url:
                study.prepare(argparse.Namespace(output=str(root/'study'), manifest=str(root/'documents/manifest.json'),
                    documents=str(root/'documents'), model=str(models['q4']), tokenizer_url=url, d=20, n=32768, m=2048, timeout=3600))
        index, fixtures = study.load_study(root/'study')
        documents = [dict(pin, token_sha256=study.token_hash(fixtures[pin['id']])) for pin in read(root/'documents/manifest.json')['documents']]
        lock = dict(version=1, historical_report_sha256=report_sha, settings=dict(seed=42, d=20, n=32768, m=2048, score_batch=32),
                    models=pins, documents=documents, provenance=dict(kind='newly-pinned-study',
                    source='Fresh pinned-model tokenization of Gutenberg bytes; not recovered stage-2 inputs.'))
        validate_lock(lock, previous, report_sha, 'new-study')
        save(target, lock)
        save(root/'bootstrap.json', dict(source_ref=args.ref, gpu_identity=gpu, index_sha256=study.digest(index), provenance_lock_sha256=sha(target)))
        print(f'new-study lock: {target}; historical document/token identity remains unverified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-archive', type=Path, required=True)
    parser.add_argument('--provenance-lock', type=Path, help='independently recovered or newly pinned hashes, never inferred from aggregates')
    parser.add_argument('--mode', choices=['reproduce', 'new-study'], default='reproduce')
    parser.add_argument('--audit-only', action='store_true', help='inspect the small surviving report; no downloads/build/GPU')
    parser.add_argument('--bootstrap-new-lock', action='store_true', help='GPU host: explicitly pin a new corpus/token study, without claiming historical reconstruction')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--ref', help='full reviewed source SHA')
    parser.add_argument('--execute', action='store_true', help='GPU host only: permit pinned downloads, build and GPU work')
    args = parser.parse_args()
    previous, report_sha = historical_report(args.historical_archive)
    evidence = audit(previous)
    if args.bootstrap_new_lock:
        return bootstrap_new_lock(args, previous, report_sha)
    if args.audit_only:
        print(json.dumps(dict(evidence, report_sha256=report_sha), sort_keys=True), flush=True)
        require(args.provenance_lock is not None, 'historical provenance absent; exact reproduction is blocked')
    require(args.provenance_lock is not None, 'supply a provenance lock; saved report has no document/token hashes or model pins')
    lock = read(args.provenance_lock)
    validate_lock(lock, previous, report_sha, args.mode)
    if args.audit_only:
        return
    require(args.execute and args.root is not None and args.build is not None and args.ref is not None,
            'GPU execution requires --execute --root --build --ref')
    gpu = gpu_identity()
    source = Path(__file__).resolve().parents[2]
    clean_source(source, args.ref)
    root = args.root.resolve()
    require(root.is_dir(), 'create the root directory on the GPU data volume first')
    require(shutil.disk_usage(root).total >= 160_000_000_000, 'GPU volume must be at least 160 GB')
    models_bytes = sum(item['size'] for name, pin in lock['models'].items() for item in pin['files']
                       if not (root/'models'/name/item['name']).exists())
    reference_bytes = sum(p.stat().st_size for p in (root/'study/references').glob('*/reference.bin'))
    disk(root, models_bytes + max(0, 20_344_176_640-reference_bytes) + 14*1024**3)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(root/'tmp'))
    (root/'tmp').mkdir(exist_ok=True)
    with exclusive(root/'stage.lock'):
        manifest = dict(version=1, mode=args.mode, source_ref=args.ref, source_sha256=sha(__file__), gpu_identity=gpu,
                        provenance_lock_sha256=sha(args.provenance_lock), historical_archive_sha256=sha(args.historical_archive),
                        historical_report_sha256=report_sha, settings=lock['settings'],
                        historical_limitations=evidence, python=sys.version, models=lock['models'], documents=lock['documents'])
        if (root/'manifest.json').exists():
            require(read(root/'manifest.json') == manifest, 'stage resume manifest differs')
        else:
            save(root/'manifest.json', manifest)
        with exclusive(root/'build.lock'):
            build(source, args.build.resolve(), args.ref, root/'build.log')
        (root/'models').mkdir(exist_ok=True)
        models = download_models(lock, root/'models')
        save(root/'document-pins.json', dict(documents=[{k: r[k] for k in ('id', 'url', 'sha256', 'size')} for r in lock['documents']]))
        driver = source/'tools/hybrid-bench/hetero_study.py'
        def call(*command):
            disk(root, 7*1024**3)
            run([sys.executable, driver, '--lock', root/'gpu.lock', *command], root/'study.log')
        if not (root/'documents/manifest.json').exists():
            call('fetch', '--manifest', root/'document-pins.json', '--output', root/'documents')
        else:
            require(read(root/'documents/manifest.json') == read(root/'document-pins.json'), 'local document pins differ')
        with exclusive(root/'gpu.lock'), tokenizer(args.build/'bin/llama-server', models['q4'], root) as url:
            # prepare is called directly while this process holds the shared GPU lock.
            study.prepare(argparse.Namespace(output=str(root/'study'), manifest=str(root/'documents/manifest.json'),
                documents=str(root/'documents'), model=str(models['q4']), tokenizer_url=url, d=20, n=32768, m=2048, timeout=3600))
        index, fixtures = study.load_study(root/'study')
        verify_tokens(index, fixtures, lock['documents'])
        save(root/'prepared-identity.json', dict(index_sha256=study.digest(index), token_sha256={k: study.token_hash(v) for k, v in fixtures.items()}))
        common = ['--study', root/'study', '--build-dir', args.build, '--gpu-layers', '999', '--score-batch', '32', '--timeout', '14400']
        call('reference', *common, '--model', models['bf16'])
        ref_sha = sha(models['bf16'])
        for label, kind, dtype in zip(LABELS, ['local', 'imported', 'imported'], ['f32', 'f32', 'bf16']):
            options = ['arm', *common, '--kind', kind, '--label', label, '--client-model', models['q4'],
                       '--reference-model-sha256', ref_sha, '--state-dtype', dtype]
            if kind == 'imported':
                options += ['--node-model', models['q4'], '--chunk-tokens', '4096', '--ram-budget-mib', '4096']
            call(*options)
        call('report', '--study', root/'study', '--arms', *LABELS, '--seed', '42', '--output', root/'report')
        current = read(root/'report/report.json')
        comparator = lock.get('f32_comparator')
        checks = compare_intervals(current, previous, comparator['arm'], 'state-f32') if args.mode == 'reproduce' else []
        sanity = compare_intervals(current, previous, 'C-Q4-local', 'state-Q4-local')
        passed = bool(checks) and all(r['passed'] for r in checks)
        save(root/'reproduction.json', dict(status=('passed' if passed else 'new-study-no-exact-reproduction' if args.mode == 'new-study' else 'flagged'),
            f32_checks=checks, local_q4_historical_sanity=sanity,
            note='Historical bootstrap intervals check the new mean; passing is compatibility, not identity or equivalence.'))
        save(root/'outputs.json', dict(files={str(p.relative_to(root)): sha(p) for p in
             [root/'manifest.json', root/'prepared-identity.json', root/'reproduction.json', root/'report/report.json', root/'report/report.csv', root/'report/report.md']}))
        require(args.mode == 'new-study' or passed, 'f32 reproduction outside historical intervals; report retained and flagged')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(terminal('kl-bf16-stage', main))

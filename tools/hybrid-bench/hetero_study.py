#!/usr/bin/env python3
"""Natural-text teacher-forced hetero-quant study. Python 3.10+, standard library only."""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import signal
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from bench_common import Refusal, private_json, require, sha256
from reference_logits import reference_header

ACTIVE_LOCK_FD = None


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('ascii')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def private_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not path.is_symlink() and path.stat().st_mode & 0o077 == 0, 'private directory required: ' + str(path))
    return path.resolve()


def atomic(path, data):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save(path, value):
    atomic(path, encoded(value))


def read(path):
    require(Path(path).is_file(), 'missing file: ' + str(path))
    return json.loads(Path(path).read_text())


def file_record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path), size=path.stat().st_size)


def verify(record, reason):
    path = Path(record['path'])
    require(path.is_file(), 'missing ' + reason + ': ' + str(path))
    require(path.stat().st_size == record['size'] and sha256(path) == record['sha256'], 'stale ' + reason + ': hash mismatch')
    return path


@contextlib.contextmanager
def locked(path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('a+b') as lock:
        os.chmod(path, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refusal('study process lock busy') from None
        # Children inherit this open description, so an orphan keeps the reservation until it exits.
        yield lock.fileno()


def stop(process):
    # Only a live Popen child owned by this invocation may receive a signal.
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


@contextlib.contextmanager
def child(command, directory, name):
    with (directory / (name + '.log')).open('xb') as log:
        process = subprocess.Popen([str(v) for v in command], stdout=log, stderr=log,
                                   pass_fds=() if ACTIVE_LOCK_FD is None else (ACTIVE_LOCK_FD,))
        try:
            save(directory / (name + '.process.json'), dict(pid=process.pid, command=[str(v) for v in command], exit_code=None))
            yield process
        finally:
            stop(process)
            save(directory / (name + '.process.json'), dict(pid=process.pid, command=[str(v) for v in command], exit_code=process.returncode))


def run(command, directory, name, timeout):
    with child(command, directory, name) as process:
        code = process.wait(timeout=timeout)
    require(code == 0, name + ' refused: exit ' + str(code) + '; see ' + str(directory / (name + '.log')))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise Refusal('local API redirect refused')


def api(base, path, data=None, token='', binary=False, timeout=3600):
    url = urllib.parse.urlsplit(base)
    require(url.scheme == 'http' and url.hostname in ('127.0.0.1', '::1') and not url.username
            and not url.query and not url.fragment and url.path in ('', '/'), 'API must use numeric loopback HTTP')
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    request = urllib.request.Request(base.rstrip('/') + path, None if data is None else encoded(data), headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        payload = response.read(514 * 1024**2)
        require(len(payload) < 514 * 1024**2, 'API response exceeds limit')
    return payload if binary else json.loads(payload)


def documents(manifest, pinned=True):
    rows = read(manifest)['documents']
    require(isinstance(rows, list) and rows, 'manifest has no documents')
    seen = set()
    for row in rows:
        name = row['id']
        require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_-]+', name) and name not in seen, 'invalid or duplicate document ID')
        seen.add(name)
        url = urllib.parse.urlsplit(row['url'])
        require(url.scheme == 'https' and url.netloc in ('www.gutenberg.org', 'gutenberg.org')
                and url.path.endswith('.txt') and not url.query and not url.fragment, 'manifest requires Gutenberg HTTPS plain text')
        if pinned:
            require(re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')) and type(row.get('size')) is int
                    and 0 < row['size'] <= 64 * 1024**2, 'manifest requires pinned SHA-256 and size')
    return rows


def fetch(args):
    root = private_dir(args.output)
    rows = documents(args.manifest, pinned=False)
    pin_path = root / 'manifest.json'
    pins = documents(pin_path) if pin_path.exists() else None
    require(not pins or [(r['id'], r['url']) for r in rows] == [(r['id'], r['url']) for r in pins], 'fetch manifest changed')
    attempt = Path(tempfile.mkdtemp(prefix='fetch-', dir=root))
    downloaded = []
    for i, row in enumerate(rows):
        target = root / (row['id'] + '.txt')
        if pins:
            require(target.is_file() and target.stat().st_size == pins[i]['size'] and sha256(target) == pins[i]['sha256'], 'manifest hash mismatch before fetch')
        else:
            require(not target.exists(), 'un-pinned document already exists')
        path = attempt / target.name
        run(['curl', '--fail', '--location', '--proto', '=https', '--proto-redir', '=https', '--max-time', '300',
             '--max-filesize', str(64 * 1024**2), '--output', path, row['url']], attempt, row['id'], 330)
        pin = dict(id=row['id'], url=row['url'], sha256=sha256(path), size=path.stat().st_size)
        expected = pins[i] if pins else row
        require(not pins or 'sha256' not in row or (row['sha256'], row['size']) == (expected['sha256'], expected['size']), 'fetch input pins changed')
        require(('sha256' not in expected or (pin['sha256'], pin['size']) == (expected['sha256'], expected['size'])), 'manifest hash mismatch after fetch')
        downloaded.append((path, target, pin))
    for path, target, _ in downloaded:
        os.replace(path, target)
    save(pin_path, dict(documents=[pin for _, _, pin in downloaded]))


def prepare(args):
    root = private_dir(args.output)
    rows = documents(args.manifest)
    require(0 < args.d <= len(rows) and args.n >= 2 and args.m > 0 and args.n + args.m <= 1048576, 'invalid D/N/M')
    model = file_record(args.model)
    selected = []
    # Verify every selected source before contacting the local tokenizer or writing fixtures.
    for row in rows[:args.d]:
        path = Path(args.documents).resolve() / (row['id'] + '.txt')
        require(path.is_file() and path.stat().st_size == row['size'] and sha256(path) == row['sha256'], 'manifest hash mismatch: ' + row['id'])
        selected.append((row, path))
    specification = dict(version=1, manifest=file_record(args.manifest), tokenizer=model, d=args.d, n=args.n, m=args.m)
    index_path = root / 'index.json'
    if index_path.exists():
        index, _ = load_study(root)
        require(index['specification'] == specification, 'stale prepared index')
        return
    props = api(args.tokenizer_url, '/props', timeout=args.timeout)
    require(Path(props['model_path']).resolve() == Path(model['path']), 'tokenizer server model mismatch')
    attempt = Path(tempfile.mkdtemp(prefix='prepare-', dir=root))
    entries = []
    for row, path in selected:
        result = api(args.tokenizer_url, '/tokenize', dict(content=path.read_text(encoding='utf-8'), add_special=True, parse_special=False), timeout=args.timeout)
        tokens = result['tokens']
        require(isinstance(tokens, list) and all(type(t) is int and 0 <= t <= 2147483647 for t in tokens), 'invalid tokenizer response')
        require(len(tokens) >= args.n + args.m, 'document shorter than N+M: ' + row['id'])
        fixture = attempt / (row['id'] + '.json')
        private_json(fixture, dict(tokens=tokens[:args.n], continuation=tokens[args.n:args.n + args.m]))
        entries.append(dict(id=row['id'], source=file_record(path), fixture=file_record(fixture)))
    save(index_path, dict(specification=specification, documents=entries))


def token_hash(fixture):
    return hashlib.sha256(b''.join(struct.pack('<i', t) for t in fixture['tokens'] + fixture['continuation'])).hexdigest()


def load_study(root):
    index = read(root / 'index.json')
    spec = index['specification']
    verify(spec['manifest'], 'manifest')
    require(len(index['documents']) == spec['d'], 'stale document index')
    fixtures = {}
    for doc in index['documents']:
        verify(doc['source'], 'document')
        fixture = read(verify(doc['fixture'], 'fixture'))
        require(len(fixture['tokens']) == spec['n'] and len(fixture['continuation']) == spec['m'], 'stale fixture N/M')
        fixtures[doc['id']] = fixture
    return index, fixtures


def build_identity(build):
    directory = Path(build).resolve() / 'bin'
    paths = [directory / 'llama-hybrid-score', directory / 'llama-hybrid-prefill']
    paths += sorted(p for p in directory.iterdir() if '.so' in p.name or p.suffix == '.dylib')
    return [file_record(p) for p in paths if p.is_file()]


def receipt(path, identity=None):
    data = read(path)
    require(identity is None or data['identity'] == identity, 'stale resume identity: ' + str(path))
    for output in data['outputs']:
        verify(output, data['kind'])
    return data


def commit_receipt(path, kind, identity, paths, **extra):
    save(path, dict(kind=kind, identity=identity, outputs=[file_record(p) for p in paths], **extra))


def engine(args):
    records = build_identity(args.build_dir)
    binary = Path(args.build_dir).resolve() / 'bin/llama-hybrid-score'
    require(binary.is_file(), 'missing llama-hybrid-score')
    return dict(driver_sha256=sha256(__file__), helpers=[file_record(Path(__file__).with_name(name)) for name in ('bench_common.py', 'reference_logits.py')],
                build=records, gpu_layers=args.gpu_layers, score_batch=args.score_batch)


def base_identity(index, doc, fixture):
    return dict(index_sha256=digest(index), fixture=doc['fixture'], token_sha256=token_hash(fixture))


def reference_check(record, fixture, model_hash):
    path = verify(record['outputs'][0], 'reference')
    header = reference_header(path, model_hash)
    require(header['tokens'] == fixture['tokens'] and header['continuation'] == fixture['continuation'], 'reference document mismatch')
    require((header['n'], header['m']) == (len(fixture['tokens']), len(fixture['continuation'])), 'reference N/M mismatch')
    require(header['token_sha256'] == token_hash(fixture), 'reference token hash mismatch')
    require(header.get('context_prefill') == 'split-last-v1' and header.get('schedule_probe') == 'fresh-prefix-spots-v1', 'reference schedule controls missing; rebuild and regenerate')
    return path, header


def score(args, model, fixture, attempt):
    path = attempt / 'score.json'
    result = attempt / 'result.json'
    private_json(path, fixture)
    run([Path(args.build_dir).resolve() / 'bin/llama-hybrid-score', model, path, result, args.gpu_layers], attempt, 'score', args.timeout)
    data = read(result)
    require(data['identity']['token_sha256'] == token_hash(fixture) and data['identity']['model_sha256'] == fixture['model_sha256'], 'score identity mismatch')
    require(len(data['positions']) == len(fixture['continuation']), 'score row count mismatch')
    require(data['identity'].get('context_prefill') == 'split-last-v1' and data['schedule_floor']['kind'] == 'fresh-prefix-spots-v1', 'score schedule controls missing')
    return result, data


def reference(args):
    root = private_dir(args.study)
    index, fixtures = load_study(root)
    model = file_record(args.model)
    settings = engine(args)
    folder = private_dir(root / 'references')
    for doc in index['documents']:
        fixture = fixtures[doc['id']]
        identity = dict(base_identity(index, doc, fixture), model=model, engine=settings)
        path = folder / (doc['id'] + '.json')
        if path.exists():
            reference_check(receipt(path, identity), fixture, model['sha256'])
            continue
        attempt = Path(tempfile.mkdtemp(prefix=doc['id'] + '-', dir=folder))
        output = attempt / 'reference.bin'
        source = dict(fixture, model_sha256=model['sha256'], max_ctx=index['specification']['n'] + index['specification']['m'],
                      score_batch=args.score_batch, write_reference=str(output))
        result, _ = score(args, model['path'], source, attempt)
        commit_receipt(path, 'reference', identity, [output, result])


def arm_spec(args):
    if args.spec:
        spec = read(args.spec)
    else:
        spec = dict(kind=args.kind, label=args.label, client_model=args.client_model, node_model=args.node_model)
    require(spec.get('kind') in ('local', 'imported') and isinstance(spec.get('label'), str)
            and re.fullmatch(r'[A-Za-z0-9_-]+', spec['label']), 'invalid arm kind/label')
    require(spec.get('client_model'), 'arm requires client model')
    require(spec['kind'] != 'imported' or spec.get('node_model'), 'imported arm requires node model')
    spec['client_model'] = file_record(spec['client_model'])
    spec['node_model'] = file_record(spec['node_model']) if spec.get('node_model') else None
    require(spec['kind'] == 'imported' or spec['node_model'] is None, 'local arm cannot have a node')
    spec['hetero_quant'] = args.hetero_quant
    if spec['kind'] == 'imported':
        require(spec['node_model']['sha256'] == spec['client_model']['sha256'] or args.hetero_quant, 'node/client mismatch requires --hetero-quant')
    return spec


@contextlib.contextmanager
def node(args, spec, attempt, context):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    token = secrets.token_hex(24)
    credentials = attempt / 'clients.json'
    private_json(credentials, {'study': token})
    command = [Path(args.node_build_dir or args.build_dir).resolve() / 'bin/llama-hybrid-prefill', spec['node_model']['path'], credentials, port, context,
               '--max-ctx', context, '--gpu-layers', args.gpu_layers, '--ram-budget-mib', args.ram_budget_mib]
    base = 'http://127.0.0.1:' + str(port)
    try:
        with child(command, attempt, 'node') as process:
            deadline = time.monotonic() + args.timeout
            while True:
                require(process.poll() is None, 'prefill node exited before readiness')
                try:
                    health = api(base, '/v1/health', timeout=1)
                    break
                except (urllib.error.URLError, TimeoutError):
                    require(time.monotonic() < deadline, 'prefill node readiness timeout')
                    time.sleep(.1)
            require(health['model_sha256'] == spec['node_model']['sha256'], 'node model mismatch')
            yield base, token, health
    finally:
        credentials.unlink(missing_ok=True)


def deltas(args, spec, fixture, attempt, header):
    paths = []
    context = max(256, ((len(fixture['tokens']) + 255) // 256) * 256)
    with node(args, spec, attempt, context) as (base, token, health):
        require(health['layout_fingerprint'] == header['layout_fingerprint'], 'node/reference layout mismatch')
        remote = api(base, '/v1/sessions', dict(model_sha256=spec['node_model']['sha256']), token)['session_id']
        route = '/v1/sessions/' + remote
        try:
            for start in range(0, len(fixture['tokens']) - 1, args.chunk_tokens):
                end = min(start + args.chunk_tokens, len(fixture['tokens']) - 1)
                prefix = hashlib.sha256(b''.join(struct.pack('<i', t) for t in fixture['tokens'][:start])).hexdigest()
                body = api(base, route + '/extend', dict(n0=start, prefix_hash_n0=prefix, tokens=fixture['tokens'][start:end], return_delta=True), token, binary=True, timeout=args.timeout)
                path = attempt / (str(start) + '.delta')
                atomic(path, body)
                paths.append(path)
        finally:
            request = urllib.request.Request(base + route, method='DELETE', headers={'Authorization': 'Bearer ' + token})
            with urllib.request.urlopen(request, timeout=30) as response:
                require(response.status == 204, 'node session release failed')
    # Exiting node() waits for the owned process before the client can start.
    return paths


def arm(args):
    root = private_dir(args.study)
    index, fixtures = load_study(root)
    spec = arm_spec(args)
    reference_hash = args.reference_model_sha256
    settings = engine(args)
    settings.update(chunk_tokens=args.chunk_tokens, ram_budget_mib=args.ram_budget_mib,
                    node_build=build_identity(args.node_build_dir or args.build_dir) if spec['kind'] == 'imported' else [])
    folder = private_dir(private_dir(root / 'arms') / spec['label'])
    identity = dict(spec=spec, engine=settings, reference_model_sha256=reference_hash, index_sha256=digest(index))
    config = folder / 'arm.json'
    if config.exists():
        require(read(config) == identity, 'stale arm configuration')
    else:
        save(config, identity)
    pending = []
    for doc in index['documents']:
        fixture = fixtures[doc['id']]
        ref = receipt(root / 'references' / (doc['id'] + '.json'))
        reference_path, header = reference_check(ref, fixture, reference_hash)
        require(ref['identity']['index_sha256'] == digest(index), 'reference study mismatch')
        require(header['score_batch'] == args.score_batch, 'reference scoring schedule mismatch')
        binding = dict(base_identity(index, doc, fixture), arm=identity, reference_sha256=sha256(reference_path))
        path = folder / (doc['id'] + '.json')
        if path.exists():
            receipt(path, binding)
        else:
            pending.append((doc, fixture, path, binding, reference_path, header))
    for doc, fixture, path, binding, reference_path, header in pending:
        attempt = Path(tempfile.mkdtemp(prefix=doc['id'] + '-', dir=folder))
        paths = deltas(args, spec, fixture, attempt, header) if spec['kind'] == 'imported' else []
        require(spec['kind'] != 'imported' or paths, 'missing delta chain')
        source = dict(fixture, model_sha256=spec['client_model']['sha256'], max_ctx=index['specification']['n'] + index['specification']['m'],
                      score_batch=args.score_batch, reference_logits=str(reference_path), reference_model_sha256=reference_hash)
        if paths:
            source.update(deltas=[str(p) for p in paths], hetero_quant=spec['hetero_quant'],
                          node_model_sha256=spec['node_model']['sha256'], node_layout_fingerprint=header['layout_fingerprint'])
        result, data = score(args, spec['client_model']['path'], source, attempt)
        rows = attempt / 'rows.json'
        prefix = 'hybrid' if paths else 'local'
        values = [dict(position=p['position'], kl_nats=p[prefix + '_kl_nats'], top1_agreed=p[prefix + '_top1_agreed']) for p in data['positions']]
        require(all(v['kl_nats'] is not None and v['top1_agreed'] is not None for v in values), 'missing imported score rows')
        private_json(rows, values)
        commit_receipt(path, 'arm rows/delta', binding, [rows, result] + paths)


def quantile(values, probability):
    ordered = sorted(values)
    point = (len(ordered) - 1) * probability
    lower = math.floor(point)
    upper = math.ceil(point)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (point - lower)


def bootstrap(values, seed, resamples=10000):
    rng = random.Random(seed)
    samples = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(resamples)]
    return quantile(samples, .025), quantile(samples, .975)


def report(args):
    root = private_dir(args.study)
    index, fixtures = load_study(root)
    folders = [root / 'arms' / label for label in args.arms]
    require(len(set(args.arms)) == len(args.arms) and all(re.fullmatch(r'[A-Za-z0-9_-]+', label) for label in args.arms), 'invalid report arms')
    all_rows, means, top1_means, floors, batches = {}, {}, {}, {}, {}
    reference_hash = None
    checked_files = set()
    for label, folder in zip(args.arms, folders):
        config = read(folder / 'arm.json')
        require(config['index_sha256'] == digest(index), 'stale report study')
        if reference_hash is None:
            reference_hash = config['reference_model_sha256']
        require(config['reference_model_sha256'] == reference_hash, 'report reference model mismatch')
        dependencies = [config['spec']['client_model']] + config['engine']['build'] + config['engine']['node_build'] + config['engine']['helpers']
        if config['spec']['node_model']:
            dependencies.append(config['spec']['node_model'])
        for item in dependencies:
            key = (item['path'], item['sha256'])
            if key not in checked_files:
                verify(item, 'model/build')
                checked_files.add(key)
        rows_by_doc = {}
        floor_rows = []
        for doc in index['documents']:
            fixture = fixtures[doc['id']]
            ref = receipt(root / 'references' / (doc['id'] + '.json'))
            refpath, _ = reference_check(ref, fixture, reference_hash)
            binding = dict(base_identity(index, doc, fixture), arm=config, reference_sha256=sha256(refpath))
            record = receipt(folder / (doc['id'] + '.json'), binding)
            require(config['spec']['kind'] != 'imported' or len(record['outputs']) > 2, 'missing delta chain')
            rows = read(record['outputs'][0]['path'])
            measured = read(record['outputs'][1]['path'])
            require(measured['schedule_floor']['kind'] == 'fresh-prefix-spots-v1', 'score schedule controls missing')
            prefix = 'hybrid' if config['spec']['kind'] == 'imported' else 'local'
            floor_rows.extend(dict(kl_nats=p[prefix + '_kl_nats'], top1_agreed=p[prefix + '_top1_agreed']) for p in measured['schedule_floor']['positions'])
            require(len(rows) == index['specification']['m'] and [r['position'] for r in rows] == list(range(index['specification']['n'], index['specification']['n'] + len(rows))), 'invalid report positions')
            require(all(type(r['kl_nats']) in (int, float) and math.isfinite(r['kl_nats']) and r['kl_nats'] >= 0
                        and type(r['top1_agreed']) is bool for r in rows), 'invalid report metric')
            rows_by_doc[doc['id']] = rows
        all_rows[label] = rows_by_doc
        floors[label] = floor_rows
        batches[label] = config['engine']['score_batch']
        means[label] = [statistics.fmean(r['kl_nats'] for r in rows) for rows in rows_by_doc.values()]
        top1_means[label] = [statistics.fmean(r['top1_agreed'] for r in rows) for rows in rows_by_doc.values()]
    columns = ['scope', 'arm', 'other', 'document', 'metric', 'score_batch', 'count', 'mean', 'median', 'p90', 'p99', 'max', 'ci_low', 'ci_high']
    table = []
    for label in args.arms:
        for metric, key in [('kl_nats', 'kl_nats'), ('top1', 'top1_agreed')]:
            values = [r[key] for rows in all_rows[label].values() for r in rows]
            table.append(dict(scope='positions', arm=label, metric=metric, count=len(values), mean=statistics.fmean(values),
                              median=quantile(values, .5), p90=quantile(values, .9), p99=quantile(values, .99), max=max(values)))
            doc_values = means[label] if metric == 'kl_nats' else top1_means[label]
            low, high = bootstrap(doc_values, args.seed)
            table.append(dict(scope='documents', arm=label, metric=metric, count=len(doc_values), mean=statistics.fmean(doc_values), ci_low=low, ci_high=high))
            for doc, value in zip(index['documents'], doc_values):
                table.append(dict(scope='document', arm=label, document=doc['id'], metric=metric, count=index['specification']['m'], mean=value))
            values = [r[key] for r in floors[label]]
            require(values and all(type(v) in (int, float, bool) and math.isfinite(v) for v in values), 'invalid schedule floor')
            table.append(dict(scope='schedule_probes', arm=label, metric=metric, count=len(values), mean=statistics.fmean(values),
                              median=quantile(values, .5), p90=quantile(values, .9), p99=quantile(values, .99), max=max(values)))
    for left, right in itertools.combinations(args.arms, 2):
        for metric, source in [('kl_nats', means), ('top1', top1_means)]:
            differences = [b - a for a, b in zip(source[left], source[right])]
            low, high = bootstrap(differences, args.seed)
            table.append(dict(scope='paired', arm=right, other=left, metric=metric, count=len(differences), mean=statistics.fmean(differences), ci_low=low, ci_high=high))
    output = private_dir(args.output)
    for row in table:
        row['score_batch'] = batches[row['arm']]
    settings = dict(seed=args.seed, resamples=10000, index_sha256=digest(index), arms=args.arms)
    csv_text = io.StringIO(newline='')
    writer = csv.DictWriter(csv_text, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(table)
    markdown = '| ' + ' | '.join(columns) + ' |\n|' + '|'.join(['---'] * len(columns)) + '|\n'
    markdown += ''.join('| ' + ' | '.join(str(row.get(key, '')) for key in columns) + ' |\n' for row in table)
    atomic(output / 'report.csv', csv_text.getvalue().encode('ascii'))
    atomic(output / 'report.md', markdown.encode('ascii'))
    save(output / 'report.json', dict(settings=settings, rows=table))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--lock', default=str(Path.home() / '.cache/hybrid-study/gpu.lock'), help='shared by all study invocations on this host')
    commands = result.add_subparsers(dest='command', required=True)
    command = commands.add_parser('fetch', help='GPU-host-only curl download and first-use pinning')
    command.add_argument('--manifest', required=True)
    command.add_argument('--output', required=True)
    command = commands.add_parser('prepare', help='verify local books and use a loopback tokenizer; no downloads')
    for name in ('manifest', 'documents', 'model', 'output', 'tokenizer-url'):
        command.add_argument('--' + name, required=True)
    for name, default in [('d', 20), ('n', 32768), ('m', 2048)]:
        command.add_argument('--' + name, type=int, default=default)
    command.add_argument('--timeout', type=int, default=3600)
    for name in ('reference', 'arm'):
        command = commands.add_parser(name)
        command.add_argument('--study', required=True)
        command.add_argument('--build-dir', default='build')
        command.add_argument('--gpu-layers', type=int, default=0)
        command.add_argument('--score-batch', type=int, default=32)
        command.add_argument('--timeout', type=int, default=3600)
        if name == 'reference':
            command.add_argument('--model', required=True)
        else:
            command.add_argument('--spec', help='JSON: kind, label, client_model, node_model')
            command.add_argument('--kind', choices=['local', 'imported'])
            command.add_argument('--label')
            command.add_argument('--client-model')
            command.add_argument('--node-model')
            command.add_argument('--node-build-dir')
            command.add_argument('--reference-model-sha256', required=True)
            command.add_argument('--hetero-quant', action='store_true')
            command.add_argument('--chunk-tokens', type=int, default=4096)
            command.add_argument('--ram-budget-mib', type=int, default=4096)
    command = commands.add_parser('report')
    command.add_argument('--study', required=True)
    command.add_argument('--arms', nargs='+', required=True)
    command.add_argument('--seed', type=int, default=42)
    command.add_argument('--output', required=True)
    return result


def main(argv=None):
    global ACTIVE_LOCK_FD
    args = parser().parse_args(argv)
    os.umask(0o077)
    try:
        require(getattr(args, 'chunk_tokens', 1) in range(1, 8193), 'chunk tokens out of bounds (1..8192)')
        require(0 <= getattr(args, 'gpu_layers', 0) <= 999 and 1 <= getattr(args, 'score_batch', 1) <= 256, 'invalid GPU layers/score batch')
        with locked(args.lock) as lock_fd:
            ACTIVE_LOCK_FD = lock_fd
            globals()[args.command](args)
        return 0
    except (Refusal, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print('study refused: ' + str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('study interrupted', file=sys.stderr)
        return 130
    finally:
        ACTIVE_LOCK_FD = None


def interrupted(*unused):
    raise KeyboardInterrupt()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())

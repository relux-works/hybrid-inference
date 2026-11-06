#!/usr/bin/env python3
"""Offline admission of the reviewed Qwen bf16 RULER archives; no statistics."""
import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

from hybrid_eval import Refusal, digest, require
from ruler_equivalence import FAMILY_MODELS, SEED, TASKS

BF16_COMMIT = '8d876c71b798651a8c5df768dc374468ced77544'
BASELINE_COMMIT = '18d3f114e46c4f801c4a09e2c6cf8ba2303ae42d'
# Canonical digest of all 55 source hashes and all three services' commits/binaries
# in both owner-pinned, reviewed bf16 archives. This is not learned from a CLI input.
IMPLEMENTATION_SHA256 = 'a0a79998860655482a11add7d8fea80dd61fbd2f5478fabaa5e56a5e5349250e'
INPUT_LOCK = 'f42a48cc9d2ca4a0e6ab0bb986fb7b87990dee3c61cc821da53d4147ed80594b'
BASELINE_ARMS = {'local', 'offload', 'local_partition'}
MAX_ARCHIVE_BYTES = 8*1024**2
MAX_CONTENT_BYTES = 16*1024**2
MAX_MEMBERS = 16


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result


def parse(data):
    return json.loads(data, object_pairs_hook=unique_object)


def read_archive(path, expected):
    require(isinstance(expected, str) and len(expected) == 64 and
            all(c in '0123456789abcdef' for c in expected), 'invalid expected archive hash')
    require(path.stat().st_size <= MAX_ARCHIVE_BYTES, 'archive exceeds size limit')
    raw = path.read_bytes()
    require(sha256(raw) == expected, 'archive hash mismatch: '+path.name)
    files, total = {}, 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        for member in archive:
            require(member.isfile() and member.name == Path(member.name).name and
                    member.name not in ('', '.', '..'), 'unsafe archive member')
            require(member.name not in files, 'duplicate archive member')
            total += member.size
            require(len(files) < MAX_MEMBERS and 0 <= member.size and total <= MAX_CONTENT_BYTES,
                    'archive content exceeds limit')
            files[member.name] = archive.extractfile(member).read()
    require({'manifest.json', 'rows.jsonl', 'SHA256SUMS'} <= files.keys(), 'missing archive inputs')
    hashes = {}
    for line in files['SHA256SUMS'].decode().splitlines():
        checksum, name = line.split('  ', 1)
        require(name in files and name != 'SHA256SUMS' and name not in hashes, 'invalid SHA256SUMS member')
        require(sha256(files[name]) == checksum, 'member hash mismatch: '+name)
        hashes[name] = checksum
    require(hashes.keys() == files.keys()-{'SHA256SUMS'}, 'unlisted archive member')
    return files, dict(archive_sha256=expected, members_sha256=hashes)


def check_manifest(manifest, label):
    require(isinstance(manifest, dict), label+' manifest must be an object')
    require(manifest['depth'] in (8192, 32768), 'unregistered adapter depth')
    require(manifest['seed'] == SEED, label+' seed mismatch')
    require(manifest['input_lock_sha256'] == INPUT_LOCK, label+' data lock mismatch')
    require(manifest['samples_per_task'] == 30 and len(manifest['tasks']) == 13 and
            set(manifest['tasks']) == set(TASKS), label+' task set mismatch')
    clients = manifest['clients']
    require(isinstance(clients, dict) and set(clients) == {'client', 'node', 'partition'},
            label+' client identity missing')
    for info in clients.values():
        require(isinstance(info, dict) and info['model_sha256'] == FAMILY_MODELS['Qwen'], label+' model mismatch')


def check_rows(files, manifest, arms, label):
    rows = [parse(line) for line in files['rows.jsonl'].splitlines()]
    expected = {(task, i) for task in TASKS for i in range(30)}
    by_key = {}
    for row in rows:
        require(isinstance(row, dict), label+' row must be an object')
        require(isinstance(row['task'], str) and type(row['index']) is int, label+' invalid example key')
        key = row['task'], row['index']
        require(key not in by_key, label+' duplicate (task, index)')
        require(key in expected, label+' unexpected (task, index)')
        require(row['depth'] == manifest['depth'], label+' row depth mismatch')
        require(row['seed'] == SEED, label+' row seed mismatch')
        require('original_row_sha256' not in row, 'reserved adapter provenance field')
        require(row.get('row_sha256') == digest({k: v for k, v in row.items() if k != 'row_sha256'}),
                label+' row hash failure')
        results = row['results']
        require(isinstance(results, dict) and set(results) == arms, label+' unknown or second fourth arm')
        for value in results.values():
            require(isinstance(value, dict) and value['prompt_sha256'] == row['prompt_sha256'],
                    label+' prompt mismatch')
            require(value.get('seed', row['seed']) == SEED, label+' arm seed mismatch')
            require(type(value['score']) in (int, float) and math.isfinite(value['score']) and
                    0 <= value['score'] <= 1 and isinstance(value['answer'], str), label+' invalid result')
        by_key[key] = row
    require(by_key.keys() == expected, label+' missing (task, index)')
    return by_key


def adapt(bf16_archive, bf16_sha256, baseline_archive, baseline_sha256, output):
    require(not output.exists() and not output.is_symlink(), 'output already exists')
    require(output.parent.is_dir(), 'output parent must exist')
    bf_files, bf_source = read_archive(bf16_archive, bf16_sha256)
    base_files, base_source = read_archive(baseline_archive, baseline_sha256)
    manifest, baseline = [parse(files['manifest.json']) for files in (bf_files, base_files)]
    check_manifest(manifest, 'bf16')
    check_manifest(baseline, 'baseline')
    require(manifest['depth'] == baseline['depth'], 'baseline depth mismatch')
    declared = manifest['declared_arm']
    require(isinstance(declared, dict) and declared['label'] == 'offload_bf16' and
            declared['state_dtype'] == 'bf16', 'declared bf16 arm identity mismatch')
    implementation = declared['implementation']
    require(digest(implementation) == IMPLEMENTATION_SHA256, 'bf16 implementation identity mismatch')
    require(all(info['commit'] == BF16_COMMIT for info in implementation['services'].values()),
            'bf16 source commit mismatch')
    require(all(info['commit'] == BASELINE_COMMIT for info in baseline['clients'].values()),
            'baseline source commit mismatch')
    baseline_implementation = dict(services={role: {k: info[k] for k in ('commit', 'binaries')}
                                   for role, info in baseline['clients'].items()},
                                   source_hashes=baseline['source_hashes'])
    require(declared['baseline']['implementation'] == baseline_implementation and
            declared['baseline']['manifest_sha256'] == sha256(base_files['manifest.json']),
            'declared baseline identity mismatch')
    require(manifest['clients'] == {role: {k: v for k, v in info.items() if k not in ('commit', 'binaries')}
                                   for role, info in baseline['clients'].items()}, 'bf16 client identity mismatch')
    require(manifest['data_provenance'] == baseline['data_provenance'], 'baseline data provenance mismatch')
    base_rows = check_rows(base_files, baseline, BASELINE_ARMS, 'baseline')
    bf_rows = check_rows(bf_files, manifest, BASELINE_ARMS | {'offload_bf16'}, 'bf16')
    links = {task+f'-{index:04d}.json': row['row_sha256'] for (task, index), row in base_rows.items()}
    require(declared['baseline']['rows'] == links, 'declared baseline row links mismatch')
    derived = []
    for key, row in sorted(bf_rows.items()):
        base = base_rows[key]
        require(row['baseline_row_sha256'] == base['row_sha256'], 'baseline row hash link mismatch')
        require(all(row['results'][arm] == base['results'][arm] for arm in BASELINE_ARMS),
                'frozen baseline arms differ')
        metadata = row.keys()-{'row_sha256', 'baseline_row_sha256', 'results', 'order', 'releases'}
        require(all(row[k] == base[k] for k in metadata), 'frozen baseline metadata differ')
        trace = row['results']['offload_bf16']['trace']
        require(trace['state_dtype'] == 'bf16' and trace['mode'] == 'offload' and
                trace['remote_applied'] > 0 and not trace.get('fallback_reason'), 'bf16 trace identity mismatch')
        item = copy.deepcopy(row)
        item['original_row_sha256'] = item.pop('row_sha256')
        fourth = item['results'].pop('offload_bf16')
        item['results']['bf16'] = fourth
        item['row_sha256'] = digest(item)
        derived.append(item)
    rows_bytes = ''.join(json.dumps(row, sort_keys=True)+'\n' for row in derived).encode()
    provenance = dict(adapter='ruler_bf16_adapter.py', adapter_sha256=sha256(Path(__file__).read_bytes()),
                      depth=manifest['depth'], n=len(derived), bf16_source=bf_source, baseline_source=base_source,
                      seed=SEED, input_lock_sha256=INPUT_LOCK, model_sha256=FAMILY_MODELS['Qwen'],
                      implementation_sha256=IMPLEMENTATION_SHA256, implementation_commit=BF16_COMMIT,
                      mapping={'results.offload_bf16': 'results.bf16'}, derived_rows_sha256=sha256(rows_bytes),
                      derived_manifest_sha256=sha256(bf_files['manifest.json']),
                      row_hash_policy='original_row_sha256 and derived row_sha256 on each row; baseline_row_sha256 retained',
                      rows=[{k: row[k] for k in ('task', 'index', 'original_row_sha256', 'row_sha256',
                                                'baseline_row_sha256')} for row in derived])
    require(shutil.disk_usage(output.parent).free >= 28*1024**3, 'disk below 28 GiB floor')
    # Publish only after both complete archives and every paired row pass admission.
    with tempfile.TemporaryDirectory(prefix='.bf16-adapt-', dir=output.parent) as tmp:
        folder = Path(tmp)
        (folder/'manifest.json').write_bytes(bf_files['manifest.json'])
        (folder/'rows.jsonl').write_bytes(rows_bytes)
        (folder/'provenance.json').write_text(json.dumps(provenance, sort_keys=True, indent=2)+'\n')
        require(not output.exists() and not output.is_symlink(), 'output already exists')
        folder.rename(output)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bf16-archive', type=Path, required=True)
    parser.add_argument('--bf16-sha256', required=True)
    parser.add_argument('--baseline-archive', type=Path, required=True)
    parser.add_argument('--baseline-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True, help='absent directory with an existing parent')
    args = parser.parse_args()
    try:
        adapt(args.bf16_archive, args.bf16_sha256, args.baseline_archive, args.baseline_sha256, args.output)
    except (Refusal, OSError, ValueError, KeyError, TypeError, UnicodeError, EOFError, tarfile.TarError) as error:
        print('REFUSED: '+str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())

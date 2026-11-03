"""Read PRI-31 step-40 rates without extracting archive paths."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

from bench_common import Refusal, require, sha256


def validate_fixture(fixture, model_sha256, architecture):
    require(fixture.get('status') == 'measured' and fixture.get('priority') == 31 and
            fixture.get('normal_priority') is True and fixture.get('override_used') is False,
            'measured PRI-31 fixture required; placeholder/background rates refused')
    require(fixture['model_sha256'] == model_sha256 and fixture['architecture'] == architecture,
            'rate fixture model mismatch')
    require(all(len(fixture['provenance'][key]) == 64 and
                all(c in '0123456789abcdef' for c in fixture['provenance'][key])
                for key in ('archive_sha256', 'raw_sha256', 'priority_sha256')), 'missing fixture provenance')
    rows = fixture['depths']
    require(rows and [r['depth'] for r in rows] == sorted({r['depth'] for r in rows}), 'invalid fixture depths')
    for row in rows:
        require(type(row['depth']) is int and row['depth'] > 0 and set(row['delta_prefill_tok_s']) == {'512', '2000', '10000'},
                'incomplete step-40 depth row')
        require(all(type(v) in (int, float) and math.isfinite(v) and v > 0
                    for v in [row['decode_tok_s'], *row['delta_prefill_tok_s'].values()]), 'invalid fixture rates')
    return [[0, rows[0]['delta_prefill_tok_s']['512']]] + [[r['depth'], r['delta_prefill_tok_s']['512']] for r in rows]


def from_archive(path, expected_sha256):
    require(sha256(path) == expected_sha256, 'step-40 archive checksum mismatch')
    with tarfile.open(path) as archive:
        def read(suffix):
            members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(suffix)]
            require(len(members) == 1 and members[0].size <= 4 * 1024**2, 'ambiguous or oversized archive member: ' + suffix)
            return members[0].name, archive.extractfile(members[0]).read()
        raw_name, raw = read('.jsonl')
        _, priority = read('/OPERATOR-READINESS.json')
        _, qualification = read('/OPERATOR-QUALIFICATION.json')
        _, geometry = read('/model-geometry.json')
    readiness, identity, layout = (json.loads(v) for v in (priority, qualification, geometry))
    require(readiness['normal_priority'] is True and readiness['override_used'] is False and
            readiness['samples'] and all(s['normal_priority'] is True and not s['override_used'] and not s['violations']
                                         for s in readiness['samples']), 'PRI-31 archive readiness refused')
    servers = [p for s in readiness['samples'] for p in s['servers']]
    require(servers and all(p['pri'] == 31 and p['nice'] == 0 and not p['errors'] for p in servers),
            'PRI-31 server evidence missing')
    rows = [json.loads(line) for line in raw.splitlines()]
    turns = [r for r in rows if r['kind'] == 'turn']
    require(turns and any(r['kind'] == 'done' for r in rows) and
            all(not r['reread'] and not r['truncated'] and r['gen_tokens_exact'] for r in turns), 'invalid step-40 run')
    depths = []
    for depth in sorted({r['level'] for r in turns}):
        group = [r for r in turns if r['level'] == depth]
        require(len(group) == 3 and {r['delta'] for r in group} == {512, 2000, 10000}, 'incomplete delta ladder')
        depths.append(dict(depth=depth, delta_prefill_tok_s={str(r['delta']): r['prefill_tok_s'] for r in group},
            decode_tok_s=sum(r['predicted_n'] for r in group) * 1000 / sum(r['predicted_ms'] for r in group),
            decode_samples=[r['decode_tok_s'] for r in group],
            segment_prefill_tok_s=next(r['prefill_tok_s'] for r in rows if r['kind'] == 'segment' and r['level'] == depth)))
    fixture = dict(schema='hybrid-growing-rate-fixture-v1', status='measured', priority=31, normal_priority=True,
        override_used=False, architecture=layout['architecture'], model_sha256=identity['model_sha256'],
        depths=depths, geometry=layout, provenance=dict(archive=Path(path).name, archive_sha256=expected_sha256,
            raw=raw_name, raw_sha256=hashlib.sha256(raw).hexdigest(), priority_sha256=hashlib.sha256(priority).hexdigest(),
            decode_reduction='sum predicted_n / sum predicted_ms; three 128-token delta calls per level',
            routing_selector='delta512 by conditioning level; floor step curve, as Policy.choose',
            limitation='Single PRI-31 ladder, kit engine dde879bc; not a measurement on the r46 prototype'))
    validate_fixture(fixture, fixture['model_sha256'], fixture['architecture'])
    return fixture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    fixture = from_archive(a.archive, a.sha256)
    with a.output.open('x') as target:
        json.dump(fixture, target, indent=2)
        target.write('\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Fit per-call HTTP/native import time, excluding transfer and node snapshot capture."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import unittest


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def fit(points):
    if len(points) < 6 or len({p['tokens'] for p in points}) < 3:
        raise ValueError('need six imports and three distinct transfer sizes')
    x, y = [p['tokens'] for p in points], [p['seconds'] for p in points]
    if any(not math.isfinite(v) or v <= 0 for v in x + y) or max(x) - min(x) < 1000:
        raise ValueError('invalid points or insufficient size span')
    mx, my = statistics.mean(x), statistics.mean(y)
    slope = sum((a-mx)*(b-my) for a,b in zip(x,y)) / sum((a-mx)**2 for a in x)
    intercept = my - slope * mx
    # Nonnegative least squares in two dimensions; retain unconstrained values as diagnostics.
    candidates = [(max(0, intercept), max(0, slope)), (0, sum(a*b for a,b in zip(x,y))/sum(a*a for a in x)), (my, 0)]
    fixed, per_token = min(candidates, key=lambda ab: sum((b-ab[0]-ab[1]*a)**2 for a,b in zip(x,y)))
    residuals = [b-fixed-per_token*a for a,b in zip(x,y)]
    return dict(fixed_s=fixed, seconds_token=per_token, method='nonnegative least squares, per-call import_ms',
                unconstrained=dict(fixed_s=intercept, seconds_token=slope),
                residual_rms_s=math.sqrt(statistics.mean(v*v for v in residuals)), raw_points=points)


def points_from_rows(rows, source):
    points = []
    if not rows or rows[-1].get('stage') != 'summary' or rows[-1].get('status') != 'passed':
        raise ValueError('calibration must have a passed summary')
    for row in rows:
        if row.get('stage') != 'call' or row.get('mode') != 'offload':
            continue
        chunks = row.get('chunks', [])
        if row.get('path') != 'offload' or row.get('fallback_reason') or not chunks:
            raise ValueError('calibration offload fell back or has no chunks')
        if [c['status'] for c in chunks] != ['staged'] * (len(chunks)-1) + ['imported']:
            raise ValueError('incomplete staged transaction')
        n = sum(c['n1'] - c['n0'] for c in chunks)
        seconds = row['import_ms'] / 1000
        if n != row['remote_applied'] or abs(sum(c['import_ms'] for c in chunks)/1000 - seconds) > 1e-6:
            raise ValueError('import token/time accounting mismatch')
        points.append(dict(source=source, schedule_index=row['schedule_index'], tokens=n, seconds=seconds,
            final_commit_s=chunks[-1]['import_ms']/1000, staged_s=sum(c['import_ms'] for c in chunks[:-1])/1000,
            chunks=len(chunks), wire_bytes=row['wire_bytes'], transfer_s=row['transfer_ms']/1000,
            ttft_s=row['ttft_seconds']))
    return points


class Tests(unittest.TestCase):
    # Known coefficients are recovered; too few/degenerate samples and failed imports cannot qualify.
    def test_fit_and_refusals(self):
        points = [dict(tokens=n, seconds=1.25+.0004*n) for n in (100,500,2000,8000,16000,32000)]
        result=fit(points)
        self.assertAlmostEqual(result['fixed_s'],1.25)
        self.assertAlmostEqual(result['seconds_token'],.0004)
        for bad in (points[:2], [dict(tokens=3, seconds=1)]*6, points[:-1]+[dict(tokens=0, seconds=1)]):
            with self.assertRaises(ValueError): fit(bad)
        with self.assertRaises(ValueError): points_from_rows([dict(stage='summary',status='failed')], 'test')
        row=dict(stage='call',mode='offload',path='offload',fallback_reason=None,
                 chunks=[dict(n0=0,n1=2000,status='imported',import_ms=2050)],remote_applied=2000,
                 import_ms=2050,schedule_index=0,wire_bytes=1234,transfer_ms=10,ttft_seconds=3)
        self.assertEqual(points_from_rows([row,dict(stage='summary',status='passed')],'test')[0]['tokens'],2000)
        for fields in (dict(path='local'),dict(remote_applied=1999),dict(import_ms=1),dict(fallback_reason='outage')):
            with self.assertRaises(ValueError): points_from_rows([dict(row,**fields),dict(stage='summary',status='passed')],'test')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--source-root',type=Path,default=Path.cwd())
    p.add_argument('logs',nargs='*',type=Path)
    a=p.parse_args()
    if a.self_test:
        return 0 if unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful() else 1
    if not a.output or not a.logs: p.error('--output and JSONL logs required')
    points=[]
    sources={}
    identities=set()
    for path in a.logs:
        sources[str(path)]=digest(path)
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row.get('stage')=='call':
                client=row.get('client_model_sha256')
                if not isinstance(client,str) or len(client)!=64 or row.get('node_model_sha256')!=client:
                    raise ValueError('missing or different model identities')
                identities.add(client)
        points.extend(points_from_rows(rows,str(path)))
    if len(identities)!=1: raise ValueError('calibration mixes model identities')
    result=fit(points)
    result['model_sha256']=next(iter(identities))
    result['source_head']=subprocess.check_output(['git','-C',str(a.source_root),'rev-parse','HEAD'],text=True).strip()
    result['source_hashes']=sources
    result['code_hashes']={name:digest(a.source_root/name) for name in ('tools/hybrid-bench/hybrid_bench.py','tools/hybrid-proxy/proxy.py')}
    result['fit_script_sha256']=digest(__file__)
    result['limitation']='Per-call fit includes all staged chunks and final commit; geometry/chunk distribution matter. Not native-only profiling.'
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    return 0

if __name__=='__main__': raise SystemExit(main())

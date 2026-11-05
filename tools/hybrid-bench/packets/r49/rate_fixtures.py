#!/usr/bin/env python3
"""Build device-bound PRI31 fixtures from lane-held source artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
from bench_common import require,sha256
from growing_fixture import validate_fixture


def m5_fixture(path,digest):
    require(sha256(path)==digest,'M5 archive hash mismatch')
    with tarfile.open(path) as archive:
        entries=[m for m in archive if m.isfile()]
        def read(suffix):
            candidates=[m for m in entries if m.name.endswith(suffix)]
            require(len(candidates)==1 and candidates[0].size<=4*1024**2,'ambiguous or oversized Mac member')
            return candidates[0].name,archive.extractfile(candidates[0]).read()
        _,manifest_raw=read('/MANIFEST.json');manifest=json.loads(manifest_raw)
        _,priority_raw=read('/OPERATOR-READINESS.json');priority=json.loads(priority_raw)
        _,geometry_raw=read('/model-geometry.json');geometry=json.loads(geometry_raw)
        candidates=[m for m in entries if m.name.endswith('.jsonl') and '-q4km-q8kv-' in m.name]
        require(len(candidates)==1 and candidates[0].size<=4*1024**2,'M5 step40 raw file missing or ambiguous')
        raw_name=candidates[0].name;raw=archive.extractfile(candidates[0]).read()
    require(manifest['normal_priority'] and not manifest['priority_override_used'] and
            priority['normal_priority'] and not priority['override_used'],'M5 normal-priority admission missing')
    servers=[p for sample in priority['samples'] for p in sample['servers']]
    require(servers and all(p['pri']==31 and p['nice']==0 and not p['errors'] for p in servers),'M5 PRI31 server evidence missing')
    rows=[json.loads(line) for line in raw.splitlines()]
    turns=[r for r in rows if r['kind']=='turn']
    require(turns and any(r['kind']=='done' for r in rows) and all(not r['reread'] and not r['truncated'] and r['gen_tokens_exact'] for r in turns),
            'M5 step40 incomplete or replayed')
    depths=[]
    for depth in sorted({r['level'] for r in turns}):
        group=[r for r in turns if r['level']==depth]
        require(len(group)==3 and {r['delta'] for r in group}=={512,2000,10000},'M5 delta ladder incomplete')
        depths.append(dict(depth=depth,delta_prefill_tok_s={str(r['delta']):r['prefill_tok_s'] for r in group},
            decode_tok_s=sum(r['predicted_n'] for r in group)*1000/sum(r['predicted_ms'] for r in group)))
    fixture=dict(schema='hybrid-growing-rate-fixture-v1',status='measured',priority=31,normal_priority=True,
        override_used=False,target_device='m5-max-64gb',architecture=geometry['architecture'],model_sha256=manifest['model_sha256'],
        depths=depths,geometry=geometry,provenance=dict(archive=path.name,archive_sha256=digest,raw=raw_name,
            raw_sha256=hashlib.sha256(raw).hexdigest(),priority_sha256=hashlib.sha256(priority_raw).hexdigest(),
            routing_selector='delta512 by conditioning level, floor step curve',
            limitation='Single PRI31 kit1.6.7 ladder on M5 Max64; prototype import fit measured separately'))
    validate_fixture(fixture,fixture['model_sha256'],fixture['architecture'])
    return fixture


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--m5-archive',required=True,type=Path)
    p.add_argument('--m5-sha256',required=True);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    m3=json.loads((ROOT/'tools/hybrid-bench/fixtures/m3-max-qwen-pri31.json').read_text());m3['target_device']='m3-max-128gb'
    for name,row in [('m3-max-qwen-pri31.json',m3),('m5-max-qwen-pri31.json',m5_fixture(a.m5_archive,a.m5_sha256))]:
        with (a.output/name).open('x') as target:json.dump(row,target,indent=2);target.write('\n')


if __name__=='__main__':main()

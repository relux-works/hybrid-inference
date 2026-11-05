#!/usr/bin/env python3
"""Emit the fixed multi-session arm order before supervisor execution."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
from bench_common import require,sha256,private_json
from session_protocol import prereg_binding,schedule,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session-set',required=True,type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--prereg-commit',required=True);p.add_argument('--group',choices=('main','favourable','public'),required=True)
    p.add_argument('--link-series',choices=('home','fast'),required=True);p.add_argument('--series-id',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--config',type=Path)
    a=p.parse_args();require(not a.output.exists(),'series plan cannot overwrite an earlier attempt')
    require(sha256(a.session_set)==a.sha256,'session set differs from frozen hash')
    selected=json.loads(a.session_set.read_text());prereg=prereg_binding(ROOT,a.prereg_commit)
    require(selected['prereg']==prereg,'session set has another preregistration')
    result=dict(schema='hybrid-e2e-series-plan-v1',prereg=prereg,session_set_sha256=a.sha256,
        frozen_population_digest=selected['frozen_population_digest'],series_id=a.series_id,
        group=a.group,link_series=a.link_series,created_utc=datetime.now(timezone.utc).isoformat(),
        attempts=schedule(selected,a.group,a.link_series),
        admission='Exact accepted r47 kit supervisor/ref and raw cooldown/admission evidence required for every arm. No execution fallback.')
    private_json(a.output,result)
    if a.execute:execute(result,a.session_set,a.config,a.output.parent)
    print(json.dumps(dict(sha256=sha256(a.output),attempts=len(result['attempts']),canonical_digest=digest(result))))


def execute(plan,session_set,config_path,root):
    require(sys.platform=='darwin','series execution requires admitted Mac; no CPU fallback')
    require(config_path is not None,'measurement configuration required')
    config=json.loads(config_path.read_text())
    allowed={'client-state','ca-file','fixture','policy','import-fit','device','link-label','kit-root','kit-ref'}
    require(set(config)==allowed,'measurement config must contain the declared paths/settings only')
    attempts=[];comparison=None;probe_path=None
    for row in plan['attempts']:
        first=row['comparison']!=comparison
        comparison=row['comparison']
        if first:probe_path=root/('probe-'+comparison[0]+'-start.json')
        output=root/('attempt-'+str(row['index']))
        command=[sys.executable,str(ROOT/'tools/hybrid-bench/session_measurement.py')]
        for key,value in config.items():command+=['--'+key,str(value)]
        command+=['--session-set',str(session_set),'--session-set-sha256',plan['session_set_sha256'],
            '--session-digest',row['session_digest'],'--prereg-commit',plan['prereg']['commit'],'--arm',row['arm'],
            '--series-id',plan['series_id'],'--series-probe',str(probe_path),'--comparison',comparison,
            '--group',plan['group'],'--ordinal',str(row['index']),'--output',str(output)]
        if first:command+=['--refresh-series-probe']
        with (root/('attempt-'+str(row['index'])+'.log')).open('wb') as log:
            code=subprocess.run(command,stdout=log,stderr=log).returncode
        record=dict(index=row['index'],exit_code=code,status='measured' if code==0 else 'failed')
        path=output/'trace.json'
        if path.exists():record.update(trace_path=str(path),trace_sha256=sha256(path))
        attempts.append(record);private_json(root/'ATTEMPTS.json',attempts)
        if code:
            # No replacement or hidden retry: persist every unattempted item and stop this series.
            attempts += [dict(index=later['index'],status='not-attempted-after-failure') for later in plan['attempts'][row['index']+1:]]
            private_json(root/'ATTEMPTS.json',attempts);raise RuntimeError('series refused; preserve ATTEMPTS.json')
        last=row['index']==len(plan['attempts'])-1 or plan['attempts'][row['index']+1]['comparison']!=comparison
        if last:
            command=[sys.executable,str(ROOT/'tools/hybrid-bench/packets/r49/probe.py'),'--client-state',config['client-state'],
                '--ca-file',config['ca-file'],'--link-label',config['link-label'],'--series-id',plan['series_id'],
                '--output',str(root/('probe-'+comparison[0]+'-end.json'))]
            with (root/('probe-'+comparison[0]+'-end.log')).open('wb') as log:
                subprocess.run(command,stdout=log,stderr=log,check=True)


if __name__=='__main__':main()

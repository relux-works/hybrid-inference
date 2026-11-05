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
from session_study import amendment_binding,checkpoint_plan,verify_public_rehearsal,verify_qualification,replace_private_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session-set',required=True,type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--prereg-commit',required=True);p.add_argument('--amendment-commit',required=True)
    p.add_argument('--comparison',choices=('both','B/A','C/A'),required=True)
    p.add_argument('--qualification-method',choices=('tc','real-path'),required=True)
    p.add_argument('--public-rehearsal',type=Path);p.add_argument('--public-rehearsal-sha256')
    for key in ('device','model-sha256','link-label'):p.add_argument('--'+key,required=True)
    p.add_argument('--group',choices=('main','favourable','public'),required=True)
    p.add_argument('--link-series',choices=('home','fast'),required=True);p.add_argument('--series-id',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--config',type=Path)
    a=p.parse_args();require(a.output.name=='plan.json','series plan must be named plan.json')
    require(not a.output.exists(),'series plan cannot overwrite an earlier attempt')
    require(sha256(a.session_set)==a.sha256,'session set differs from frozen hash')
    selected=json.loads(a.session_set.read_text());prereg=prereg_binding(ROOT,a.prereg_commit)
    amendment=amendment_binding(ROOT,a.amendment_commit)
    require(selected['prereg']==prereg and selected.get('amendment')==amendment,'session set has another registration')
    source=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    identity=dict(prereg=prereg,session_set_path=str(a.session_set.resolve()),amendment=amendment,source_head=source,device=a.device,model_sha256=a.model_sha256,
        qualification_method=a.qualification_method,link_label=a.link_label,link_series=a.link_series)
    proof=None;budget=30
    if a.group=='public':
        require(a.link_series=='fast' and a.comparison=='both','public first rehearsal requires both comparisons on fast path')
    else:
        require(a.public_rehearsal and sha256(a.public_rehearsal)==a.public_rehearsal_sha256,'verified public rehearsal required before owner execution')
        proof=verify_public_rehearsal(a.public_rehearsal,identity);budget=proof['capture_budget_seconds']
        require(selected.get('freeze_publication'),'owner session set lacks published freeze')
    result=dict(schema='hybrid-e2e-series-plan-v1',prereg=prereg,session_set_sha256=a.sha256,
        session_set_path=str(a.session_set.resolve()),amendment=amendment,source_head=source,device=a.device,model_sha256=a.model_sha256,
        qualification_method=a.qualification_method,link_label=a.link_label,public_rehearsal=proof,
        frozen_population_digest=selected['frozen_population_digest'],series_id=a.series_id,
        group=a.group,link_series=a.link_series,created_utc=datetime.now(timezone.utc).isoformat(),
        attempts=schedule(selected,a.group,a.link_series,a.comparison),
        checkpoints={r['session_digest']:{arm:checkpoint_plan(r,arm,budget) for arm in ('local','hybrid','hybrid-background')} for r in selected['sets'][a.group]['sessions']},
        admission='Exact accepted r47 kit supervisor/ref and raw cooldown/admission evidence required for every arm. No execution fallback.')
    private_json(a.output,result)
    if a.execute:execute(result,a.session_set,a.config,a.output.parent)
    print(json.dumps(dict(sha256=sha256(a.output),attempts=len(result['attempts']),canonical_digest=digest(result))))


def execute(plan,session_set,config_path,root):
    require(sys.platform=='darwin','series execution requires admitted Mac; no CPU fallback')
    require(config_path is not None,'measurement configuration required')
    config=json.loads(config_path.read_text())
    allowed={'client-state','ca-file','fixture','policy','import-fit','device','link-label','kit-root','kit-ref','qualification'}
    require(set(config)==allowed,'measurement config must contain the declared paths/settings only')
    require(config['device']==plan['device'] and config['link-label']==plan['link_label'],'configuration changes declared device/path')
    attempts=[];comparison=None;probe_path=None;qualifications={}
    def qualify(block,side):
        path=root/('qualification-'+block+'-'+side+'.json')
        command=[sys.executable,str(ROOT/'tools/hybrid-bench/packets/r49/qualify_transport.py'),
            '--prereg-commit',plan['prereg']['commit'],'--amendment-commit',plan['amendment']['commit'],
            '--method',plan['qualification_method'],'--series-id',plan['series_id'],'--block',block,'--side',side,
            '--link-series',plan['link_series'],'--link-label',plan['link_label'],'--output',str(path)]
        for key in ('client-state','ca-file'):command+=['--'+key,config[key]]
        extra=config['qualification'];require(isinstance(extra,dict),'qualification configuration required')
        required={'tc-results'} if plan['qualification_method']=='tc' else {'reference-url','reference-header','interface'}
        require(set(extra)==required,'qualification configuration does not match locked method')
        for key,value in extra.items():command+=['--'+key,str(value)]
        with (root/('qualification-'+block+'-'+side+'.log')).open('wb') as log:
            code=subprocess.run(command,stdout=log,stderr=log).returncode
        if code:
            failed=attempts+[dict(index=r['index'],status='not-attempted-after-qualification-failure') for r in plan['attempts'][len(attempts):]]
            replace_private_json(root/'ATTEMPTS.json',failed)
            raise RuntimeError('transport qualification failed; preserve series with no reroll')
        result=json.loads(path.read_text());verify_qualification(result,plan['qualification_method'],plan['link_series'],plan['link_label'])
        qualifications[block+'-'+side]=dict(path=path.name,sha256=sha256(path))
    for row in plan['attempts']:
        first=row['comparison']!=comparison
        comparison=row['comparison']
        if first:
            qualify(comparison[0],'pre')
            probe_path=root/('probe-'+comparison[0]+'-start.json')
        output=root/('attempt-'+str(row['index']))
        command=[sys.executable,str(ROOT/'tools/hybrid-bench/session_measurement.py')]
        for key,value in config.items():
            if key!='qualification':command+=['--'+key,str(value)]
        command+=['--session-set',str(session_set),'--session-set-sha256',plan['session_set_sha256'],
            '--session-digest',row['session_digest'],'--prereg-commit',plan['prereg']['commit'],'--arm',row['arm'],
            '--series-id',plan['series_id'],'--series-probe',str(probe_path),'--comparison',comparison,
            '--amendment-commit',plan['amendment']['commit'],'--series-plan',str(root/'plan.json'),
            '--series-plan-sha256',sha256(root/'plan.json'),'--group',plan['group'],'--ordinal',str(row['index']),'--output',str(output)]
        if first:command+=['--refresh-series-probe']
        with (root/('attempt-'+str(row['index'])+'.log')).open('wb') as log:
            code=subprocess.run(command,stdout=log,stderr=log).returncode
        record=dict(index=row['index'],exit_code=code,status='measured' if code==0 else 'failed')
        path=output/'trace.json'
        if path.exists():record.update(trace_path=str(path),trace_sha256=sha256(path))
        attempts.append(record);replace_private_json(root/'ATTEMPTS.json',attempts)
        if code:
            # No replacement or hidden retry: persist every unattempted item and stop this series.
            attempts += [dict(index=later['index'],status='not-attempted-after-failure') for later in plan['attempts'][row['index']+1:]]
            replace_private_json(root/'ATTEMPTS.json',attempts);raise RuntimeError('series refused; preserve ATTEMPTS.json')
        last=row['index']==len(plan['attempts'])-1 or plan['attempts'][row['index']+1]['comparison']!=comparison
        if last:
            qualify(comparison[0],'post')
            command=[sys.executable,str(ROOT/'tools/hybrid-bench/packets/r49/probe.py'),'--client-state',config['client-state'],
                '--ca-file',config['ca-file'],'--link-label',config['link-label'],'--series-id',plan['series_id'],
                '--output',str(root/('probe-'+comparison[0]+'-end.json'))]
            with (root/('probe-'+comparison[0]+'-end.log')).open('wb') as log:
                subprocess.run(command,stdout=log,stderr=log,check=True)

    private_json(root/'SERIES.json',dict(schema='hybrid-e2e-series-receipt-v1',prereg=plan['prereg'],amendment=plan['amendment'],source_head=plan['source_head'],
        plan=dict(path='plan.json',sha256=sha256(root/'plan.json')),
        attempts=dict(path='ATTEMPTS.json',sha256=sha256(root/'ATTEMPTS.json')),qualification=qualifications))


if __name__=='__main__':main()

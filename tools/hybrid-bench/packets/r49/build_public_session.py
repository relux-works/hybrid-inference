#!/usr/bin/env python3
"""Construct a publishable repository replay using the actual target tokenizer."""
import argparse
from datetime import datetime,timedelta,timezone
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
sys.path.insert(0,str(ROOT/'tools/hybrid-proxy'))
from bench_common import read_private,private_json,require,sha256
from proxy import Node
from session_protocol import recorded_session,digest,prereg_binding,distribution
from session_study import amendment_binding


def build(source,tokenize,model,prereg):
    require(source['schema']=='hybrid-public-repository-source-v1','public source schema differs')
    base=[];base_fragments=[];turns=[];now=datetime(2026,11,5,tzinfo=timezone.utc);base_end=now.isoformat();file_list=[]
    for file in source['files']:
        require(__import__('hashlib').sha256(file['text'].encode()).hexdigest()==file['sha256'],'public file digest differs')
        lines=file['text'].splitlines(keepends=True)
        for start in range(0,len(lines),200):
            text='Tool: git show '+source['commit']+':'+file['path']+'; lines '+str(start+1)+'-'+str(min(start+200,len(lines)))+'\n'+''.join(lines[start:start+200])
            incoming=tokenize(text)
            if len(base)<100000:
                base+=incoming;base_fragments.append(text);continue
            index=len(turns);wait=60 if (index+1)%10==0 else 8
            now+=timedelta(seconds=wait);request=now.isoformat();now+=timedelta(seconds=1)
            output='Reviewed the recorded source block.\n'
            turns.append(dict(source_turn=index,input_ids=incoming,output_ids=tokenize(output),input_text=text,output_text=output,
                request_start_utc=request,response_end_utc=now.isoformat()))
            file_list.append(dict(path=file['path'],sha256=file['sha256'],first_line=start+1,last_line=min(start+200,len(lines))))
            if len(base)+sum(len(r['input_ids'])+len(r['output_ids']) for r in turns)>=200000:break
        else:continue
        break
    session=dict(id='public-llama-source-v1',base_ids=base,base_response_end_utc=base_end,turns=turns)
    workload=recorded_session(session,model,2**31)
    require(workload is not None,'public source does not reach the registered context interval')
    group=dict(label='constructed public-repository workload',sessions=[workload],count=1,status='selected',
        depth_range=[workload['start_depth'],workload['final_depth']],wait_distribution=workload['wait_distribution'])
    return dict(schema='hybrid-e2e-session-set-v1',prereg=prereg,reproducibility_only=True,sets=dict(public=group),
        frozen_population_digest=digest(source),base_text_fragments=base_fragments,public_source=dict(url=source['url'],commit=source['commit'],file_list=file_list),
        wait_origin='Constructed public schedule: 8s, every tenth wait 60s. Not measured owner/tool durations.',
        response_origin='Fixed recorded acknowledgement text; not a live agent or task-success sample.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','client-state','output'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--amendment-commit',required=True);p.add_argument('--source-sha256',required=True);p.add_argument('--prereg-commit',required=True)
    a=p.parse_args();require(not a.output.exists(),'public session must be frozen, never overwritten')
    require(sha256(a.source)==a.source_sha256,'public source artifact hash differs')
    client=read_private(a.client_state/'client-info.json');config=read_private(a.client_state/'proxy-config.json')
    node=Node(config['decode_url'],config['decode_token'],300)
    def tokenize(text):return node.request('POST','/tokenize',dict(content=text,add_special=False))['tokens']
    with gzip.open(a.source,'rt') as stream:source=json.load(stream)
    result=build(source,tokenize,client['model_sha256'],prereg_binding(ROOT,a.prereg_commit))
    result['amendment']=amendment_binding(ROOT,a.amendment_commit)
    result['tokenizer_receipt']=dict(model_sha256=client['model_sha256'],source_sha256=a.source_sha256,
        source_head=client['commit'],backend=client['backend'],tokenization_only=True)
    private_json(a.output,result)
    print(json.dumps(dict(count=1,depth_range=result['sets']['public']['depth_range'],sha256=sha256(a.output),
        wait_distribution=result['sets']['public']['wait_distribution'],wait_origin=result['wait_origin'])))


if __name__=='__main__':main()

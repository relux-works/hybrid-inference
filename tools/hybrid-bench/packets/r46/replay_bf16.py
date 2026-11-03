#!/usr/bin/env python3
"""Replay a captured f32 session with bf16 imports on the same token journal."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
import growing_session as g
from bench_common import Refusal, private_json, require, sha256, start_process


def treatment(source):
    require(source.get('state_dtype','f32')=='f32', 'replay source must be f32')
    require(source.get('acceptance_rule')=='session_append_replay_exact', 'exact baseline controls required')
    chain=next((p for p in source['probes'] if p['name']=='CHAIN'),None)
    require(chain is not None, 'recorded f32 import chain required')
    result=copy.deepcopy(source);result['bf16_replay']=True
    probe=copy.deepcopy(chain);probe['name']='BF16_CHAIN'
    imports=0
    for step in probe['steps']:
        if 'delta' in step:
            step['reencode_state_dtype']='bf16';imports+=1
    require(imports>0,'bf16 replay needs recorded imports')
    result['probes'].append(probe)
    return result


def summarize(row, turn, baseline):
    probe=row['probes']['BF16_CHAIN']
    require(row['status']=='lossy' and row['exit_code']==5 and row['exact'] is False and
            row['gate_controls'].get('state_transport_exact') is False and
            probe['state_transport_exact'] is False and probe['lossy_import'] is True,
            'bf16 replay cannot satisfy an exact control')
    exact={k:v for k,v in row['gate_controls'].items() if k!='state_transport_exact'}
    require(len(exact)==10 and all(exact.values()), 'f32 reference identity control failed')
    require(sha256(row['raw_logits']['files']['H']['path'])==baseline['raw_logits']['files']['H']['sha256'] and
            sha256(row['raw_logits']['files']['L']['path'])==baseline['raw_logits']['files']['L']['sha256'],
            'f32/local reference logits changed from captured arm trace')
    return dict(turn=turn,depth=row['depth'],state_dtype='bf16',lossy=True,exit_code=5,
                imports=probe['import_count'],local_to_bf16=probe['L'],f32_to_bf16=probe['H'],
                local_top1_gate=probe['L']['top1_agreed']>=63,
                exact_transport_gate=False,
                first_free_running_divergence=dict(local=probe['L']['free_running_first_divergence'],
                                                   f32=probe['H']['free_running_first_divergence']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--checker',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--turns',type=int,nargs='+',default=[10,20,30,40])
    a=p.parse_args()
    require(not a.output.exists() and a.output.parent.is_dir(),'new replay output required')
    original=json.loads(a.trace.read_text())
    require(original['status']=='passed' and not original['heterogeneous'], 'passed homogeneous f32 rehearsal required')
    a.output.mkdir(mode=0o700);results=[]
    for turn in a.turns:
        trace,source,fixture,verified=g.verify_recheck(a.trace,turn)
        require(sha256(a.model)==source['model_sha256'],'replay model changed')
        vocab=g.model_identity(a.model)['model_layout']['vocab_size']
        require(type(vocab) is int and 1<=vocab<=1048576,'invalid replay vocabulary')
        require(shutil.disk_usage(a.output).free>28_000_000_000+17*64*vocab*4+100_000_000,
                'insufficient disk for bf16 replay logits above28GB floor')
        run=a.output/str(turn);run.mkdir(mode=0o700)
        converted=treatment(source);private_json(run/'input.json',converted)
        command=[a.checker,a.model,run/'input.json',source['snapshots'][0],run/'result.json',
                 0 if source['backend']=='cpu' else 999]
        private_json(run/'identity.json',dict(source_trace_sha256=sha256(a.trace),source_fixture_sha256=sha256(fixture),
            checker_sha256=sha256(a.checker),model_sha256=source['model_sha256'],preserved_inputs=verified,
            command=[str(x) for x in command]))
        process=start_process(command,run,'checker')
        try:code=process.wait()
        finally:g.stop_process(process)
        private_json(run/'exit.json',dict(pid=process.pid,exit_code=code))
        require((run/'result.json').is_file(),'replay checker produced no result')
        row=json.loads((run/'result.json').read_text());require(code==5,'bf16 checker must report lossy/5')
        g.validate_checkpoint(row,False,len(source['tokens']))
        baseline=json.loads((fixture.parent/'result.json').read_text())
        result=summarize(row,turn,baseline);private_json(run/'summary.json',result);results.append(result)
    result=dict(schema='bf16-growing-replay-v1',status='lossy',exit_code=5,exact=False,
        quality_gate=all(r['local_top1_gate'] for r in results),checkpoints=results,
        interpretation='Same captured token journal and f32 routing boundaries; every critical recurrent envelope is rounded once. Node states remain f32. This isolates accumulated client-side effects, not changed free-running history or bf16 policy decisions.')
    private_json(a.output/'summary.json',result)
    print(json.dumps(dict(status=result['status'],quality_gate=result['quality_gate'],summary=str(a.output/'summary.json'))))
    return 5


if __name__=='__main__':
    try:raise SystemExit(main())
    except (Refusal,OSError,ValueError,KeyError) as error:
        print('refused: '+str(error) if isinstance(error,Refusal) else 'refused: '+type(error).__name__,file=sys.stderr)
        raise SystemExit(2)

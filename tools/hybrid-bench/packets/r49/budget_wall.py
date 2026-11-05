#!/usr/bin/env python3
"""FACTS-rate planning, not a Mac measurement or an owner-session prediction."""
import gzip
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from session_study import checkpoint_plan
from bench_common import sha256
from estimate import estimate


def main():
    fixture=json.loads((HERE/'fixtures/m3-max-qwen-pri31.json').read_text())
    with gzip.open(HERE/'public/cpu-reference-session.json.gz','rt') as stream:public=json.load(stream)
    workload=public['sets']['public']['sessions'][0];depths=fixture['depths']
    def fill(depth):
        bounds=[0]+[r['depth'] for r in depths];total=0
        for i,row in enumerate(depths):
            lo,hi=bounds[i],min(depth,bounds[i+1])
            total+=max(0,hi-lo)/row['segment_prefill_tok_s']
        total+=max(0,depth-bounds[-1])/depths[-1]['segment_prefill_tok_s']
        return total
    scale=59.7*60/fill(249000)
    cps=checkpoint_plan(workload,'hybrid',30)['turns'];depth=workload['start_depth'];quality=[]
    for turn,row in enumerate(workload['turns'],1):
        depth+=len(row['input_ids'])+len(row['output_ids'])
        if turn in cps:quality.append(dict(turn=turn,depth=depth,scaled_seconds=fill(depth)*scale+60,raw_seconds=fill(depth)+60))
    local_cold=fill(workload['start_depth'])*scale
    config=dict(local_rates=[[r['depth'],r['delta_prefill_tok_s']['512']] for r in depths],server_rate=1097,
        fixed=.1,import_fixed=1.498,import_per_token=.0000487,kv_bytes=34816,state_bytes=78448192,eta=.85)
    rows=[]
    for rate in (3.75,6,18.75,27.875,50,100):
        values=estimate(workload,fixture,dict(config,link_rate=rate*1e6))['arms'];arms={}
        for arm,row in values.items():
            remote_cold=.1+1.498+workload['start_depth']*(1/1097+.0000487)+(workload['start_depth']*34816+78448192)/(rate*1e6*.85)
            cold=remote_cold if arm!='local' and remote_cold<local_cold*.85 else local_cold
            checker=sum(q['scaled_seconds'] for q in quality) if arm!='local' else 0
            raw_checker=sum(q['raw_seconds'] for q in quality) if arm!='local' else 0
            fixed=300+90+60
            base=cold+row['active_plus_pause_seconds']+fixed
            # Probe + health overhead is not inference; qualify traffic is included separately below.
            arms[arm]=dict(warm_hours=row['active_plus_pause_seconds']/3600,cold_hours=cold/3600,
                checker_hours=checker/3600,wall_hours=(base+checker)/3600,
                unscaled_fill_wall_hours=(base+raw_checker+(fill(workload['start_depth'])-local_cold if cold==local_cold else 0))/3600,
                offloads=row['turn_offloads'],envelope_GB=row['predicted_envelope_bytes']/1e9)
        # Each block: pre/post T/R/T=2.25GiB; two ordinary 384MiB probes=0.75GiB.
        qualification_hours=3*1024**3/(rate*1e6)/3600
        pair_ca=arms['local']['wall_hours']+arms['hybrid-background']['wall_hours']
        pair_ba=arms['local']['wall_hours']+arms['hybrid']['wall_hours']
        rows.append(dict(link_MB_s=rate,arms=arms,public_four_arms_hours=pair_ca+pair_ba+2*qualification_hours,
            main_CA_ten_arms_hours=5*pair_ca+qualification_hours,
            optional_BA_ten_arms_hours=5*pair_ba+qualification_hours,
            qualification_hours_per_block=qualification_hours))
    depth=workload['start_depth'];extra_decode=0
    for turn in workload['turns']:
        decode=next(r for r in reversed(depths) if r['depth']<=depth)['decode_tok_s']
        extra_decode+=max(0,512-len(turn['output_ids']))/decode/3600
        depth+=len(turn['input_ids'])+len(turn['output_ids'])
    result=dict(schema='hybrid-e2e-wall-budget-v1',fixture_sha256=sha256(HERE/'fixtures/m3-max-qwen-pri31.json'),
        public_sha256=sha256(HERE/'public/cpu-reference-session.json.gz'),session_count=1,turns=len(workload['turns']),
        depths=[workload['start_depth'],workload['final_depth']],checkpoint_turns=cps,checker_components=quality,
        cold_scale=scale,setup_hours=[.5,1],rows=rows,extra_decode_hours_per_arm_512_tokens_at_recorded_depths=extra_decode,
        limits='Public 0.8B token counts used for planning only; target retokenization and owner sample unknown. Background estimate assumes hidden catch-up; rates/surrogate/fit are prior measurements, not current calibration. 512-token sensitivity is decode cost only at the recorded depths (optimistic as it omits added-context slowdown); it is not a selected session or a context-valid alternate. Actual 200k checker during public rehearsal controls the budget decision. Original unscaled local fills retained per arm; slower actual checker can exceed both brackets.')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

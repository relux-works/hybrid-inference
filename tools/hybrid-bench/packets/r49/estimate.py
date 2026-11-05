#!/usr/bin/env python3
"""Planning on the selected owner trace; background overlap is an optimistic bound."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
from policy import Policy
from session_runtime import catchup_decision
from bench_common import require


def estimate(workload,fixture,config):
    require(workload['schema']=='hybrid-owner-workload-v1' and len(workload['turns'])==40,'selected owner window required')
    policy=Policy(**config)
    arms={}
    for arm in ('local','hybrid','hybrid-background'):
        depth=workload['start_depth'];evaluated=depth;remote=depth-1
        total=ttft=wire=offloads=background_gpu=0;previous=False
        for row in workload['turns']:
            n=len(row['input_ids']);output=len(row['output_ids'])
            delta=max(0,depth+n-1-evaluated);pending=max(0,evaluated-remote)
            decision=catchup_decision(policy,delta,evaluated,pending,previous,control_seconds=policy.fixed/2)
            rate=policy.local_rates[0][1];decode=fixture['depths'][0]['decode_tok_s']
            for at,value in policy.local_rates:
                if evaluated>=at:rate=value
            for point in fixture['depths']:
                if depth>=point['depth']:decode=point['decode_tok_s']
            use=arm!='local' and decision['offload']
            prefill=decision['offload_seconds']+1/rate if use else (depth+n-evaluated)/rate
            if use:
                offloads+=1;wire+=delta*policy.kv_bytes+policy.state_bytes;remote=depth+n-1
            depth+=n+output;evaluated=depth-1
            if arm=='hybrid-background' and not use:
                background_gpu+=max(0,evaluated-remote)/policy.server_rate;remote=evaluated
            ttft+=prefill;total+=prefill+output/decode+row['pause_seconds'];previous=use
        arms[arm]=dict(turn_offloads=offloads,predicted_envelope_bytes=wire,ttft_seconds=ttft,
            active_plus_pause_seconds=total,background_prefill_seconds=background_gpu)
    return dict(arms=arms,cold_hybrid_envelope_bytes_each=(workload['start_depth']-1)*policy.kv_bytes+policy.state_bytes,
        limitation='Planning only. Background assumes all previous catch-up is hidden, an optimistic bound. Cold time, evidence and shutdown are separate. Add 384MiB per probe, bounded calibration and TLS framing. Production uses measured pending acknowledgements and link EMA; actual decisions can differ.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workload',required=True,type=Path)
    p.add_argument('--fixture',required=True,type=Path);p.add_argument('--policy',required=True,type=Path)
    a=p.parse_args();print(json.dumps(estimate(*[json.loads(x.read_text()) for x in (a.workload,a.fixture,a.policy)]),indent=2))


if __name__=='__main__':main()

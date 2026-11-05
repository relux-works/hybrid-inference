#!/usr/bin/env python3
"""Extract timing fields and validate nested clocks without mixing host clock offsets."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'tools/hybrid-proxy'))
import timing


def native_rows(path):
    records=[json.loads(line) for line in path.read_text().splitlines()]
    checkpoints=[i for i,r in enumerate(records) if r.get('kind') in ('trace_checkpoint','trace_end')]
    if not checkpoints:
        raise ValueError('native trace has no completed checkpoint: '+str(path))
    end=checkpoints[-1]
    status=records[end]
    if status.get('dropped') != 0 or status.get('writer_ok') is not True:
        raise ValueError('native trace incomplete/dropped: '+str(path))
    spans=[r for r in records[:end] if r.get('kind')=='span']
    if 'records' in status and status['records'] != len(spans):
        raise ValueError('native trace checkpoint count mismatch')
    return spans


def validate_native(spans):
    by_id={(r['clock_domain'],r['id']):r for r in spans}
    if len(by_id)!=len(spans): raise ValueError('duplicate native span id')
    for row in spans:
        if (row['end_ns'] < row['start_ns'] or row['duration_ns'] != row['end_ns']-row['start_ns'] or
                row['bytes'] < 0 or row['tokens'] < 0): raise ValueError('invalid native span interval/count')
        if row['parent_domain']==row['clock_domain'] and row['parent_id']:
            parent=by_id.get((row['parent_domain'],row['parent_id']))
            if parent is None or not parent['start_ns'] <= row['start_ns'] <= row['end_ns'] <= parent['end_ns']:
                raise ValueError('native child outside/missing parent')


def extract(trace, native):
    spans=trace['instrumentation']['spans']
    validate_native(native)
    calls=[]
    for call in trace['calls']:
        meter=call['meter']; bounds=meter['timing_boundaries']; call_id=meter['timing_call_id']
        account=timing.account(spans,bounds['stream_start_ns'],bounds.get('first_nonempty_delta_ns'),call_id,meter['timing_clock_domain'])
        if account != call['timing_accounting']: raise ValueError('persisted TTFT accounting mismatch')
        phases=defaultdict(lambda:dict(duration_ns=0,bytes=0,tokens=0,count=0))
        selected=[r for r in native if r['call_id']==call_id]
        for row in selected:
            target=phases[(row['clock_domain'],row['name'])]
            for key in ('duration_ns','bytes','tokens'): target[key]+=row[key]
            target['count']+=1
        calls.append(dict(arm=call['arm'],turn=call['turn'],stage=call['stage'],
            ttft_seconds=call['ttft_seconds'],local_tail_seconds=call['local_tail_seconds'],
            reconciliation_seconds=call['reconciliation_seconds'],meter=dict(local_prompt_ms=meter.get('local_prompt_ms')),
            import_seconds=call['import_seconds'],import_stage_seconds=call['import_stage_seconds'],
            import_final_save_apply_seconds=call['import_final_save_apply_seconds'],
            boundaries=bounds,accounting=account,
            first_samples=[r for r in selected if r['name']=='decoder.first_sample'],
            native_phase_totals=[dict(clock_domain=k[0],name=k[1],**v) for k,v in phases.items()],
            native_totals_scope='whole call including generation/reconciliation, nested phase totals are NOT additive to TTFT'))
    return dict(schema='hybrid-timing-extract-v2', calls=calls,
        clock_rule='Only clipped Python leaves in the bench/proxy clock domain reconcile to TTFT; native clocks remain separate causal trees.',
        boundary_rule='Receipt means the existing application read/write observation; first socket-byte arrival is bracketed by reads, not inferred from a later SSE line. Native first sample is distinct from first nonempty text.',
        span_counts=dict(python=len(spans),native=len(native)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--native',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    native=[r for path in a.native for r in native_rows(path)]
    result=extract(json.loads(a.trace.read_text()),native)
    with a.output.open('x') as target: json.dump(result,target,indent=2);target.write('\n')


if __name__=='__main__': main()

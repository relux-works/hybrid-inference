#!/usr/bin/env python3
"""Preregistered paired session analysis; retain every planned failure and cost input."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/hybrid-bench'))
from bench_common import require,sha256
from session_protocol import paired_analysis,portion_summary
from session_study import descriptive_pairs,verify_series_transport


def analyze(plan,attempts,gpu_price_per_hour=None,traffic_price_per_GB=None,series_qualified=None):
    require(len(attempts)==len(plan['attempts']),'every planned attempt including failures must be retained')
    seen=set();flow=[];rows={};bins={}
    for expected,attempt in zip(plan['attempts'],attempts):
        require(attempt['index']==expected['index'] and attempt['index'] not in seen,'attempt order/duplicate differs from preregistration')
        seen.add(attempt['index']);trace=attempt.get('trace')
        qualified=bool(trace and trace.get('status')=='measured' and (not plan.get('amendment') or series_qualified is True))
        if trace:
            require(trace['prereg']==plan['prereg'] and trace['session_set_sha256']==plan['session_set_sha256'] and
                trace['series_id']==plan['series_id'] and trace['session_digest']==expected['session_digest'] and
                trace['arms']==[expected['arm']] and trace['comparison']==expected['comparison'], 'attempt changed registered work or arm')
            session=trace.get('sessions',{}).get(expected['arm'],{})
            summary=session.get('summary',{});nodes=summary.get('node',{})
            node_seconds=sum(row['node_compute_seconds'] for row in nodes.values())
            wire=sum(row['tls_sent_bytes']+row['tls_received_bytes'] for row in nodes.values())
            cold_seconds=session.get('cold_base_seconds',0)+session.get('decoder_load_seconds',0)
            warm=session.get('warm_system_seconds')
            flow.append(dict(index=attempt['index'],arm=expected['arm'],qualified=qualified,status=trace.get('status'),
                session_digest=expected['session_digest'],node_service_seconds=node_seconds,tls_bytes=wire,
                node_service_cost=None if gpu_price_per_hour is None else node_seconds*gpu_price_per_hour/3600,
                charged_traffic_cost=None if traffic_price_per_GB is None else wire/1e9*traffic_price_per_GB,
                warm_system_seconds=warm,cold_seconds=cold_seconds,raw_wall_seconds=trace.get('raw_run_wall_seconds')))
            key=(expected['comparison'],expected['session_digest']);rows.setdefault(key,{})[expected['arm']]=(warm,qualified,trace['calls'])
            bins[str(attempt['index'])]=portion_summary(trace['calls']) if qualified else None
        else:flow.append(dict(index=attempt['index'],arm=expected['arm'],qualified=False,status=attempt.get('status','missing')))
    comparisons={};paired_bins={};thresholds={}
    for comparison,hybrid in [('B/A','hybrid'),('C/A','hybrid-background')]:
        pairs=[];groups={label:dict(local_seconds=0,hybrid_seconds=0,count=0) for label in portion_summary([])}
        for (name,sid),arms in rows.items():
            if name!=comparison or 'local' not in arms or hybrid not in arms:continue
            a,b=arms['local'],arms[hybrid]
            if a[0] is None or b[0] is None:continue
            identity=lambda calls:[(r['turn'],r.get('input_ids_sha256'),r.get('recorded_output_sha256'),r['journal_hash']) for r in calls]
            require(identity(a[2])==identity(b[2]),'paired context/output journal differs')
            ba,bb=portion_summary(a[2]),portion_summary(b[2])
            for label,bucket in groups.items():
                require(ba[label]['count']==bb[label]['count'],'paired portion counts differ')
                bucket['local_seconds']+=ba[label]['timed_seconds'];bucket['hybrid_seconds']+=bb[label]['timed_seconds'];bucket['count']+=ba[label]['count']
            pairs.append(dict(session_digest=sid,local_seconds=a[0],hybrid_seconds=b[0],qualified=a[1] and b[1]))
        comparisons[comparison]=descriptive_pairs(pairs)
        if len(pairs)==5 and plan.get('group','main')=='main':comparisons[comparison].update(paired_analysis(pairs),descriptive_only=False)
        for bucket in groups.values():
            bucket['fractional_reduction']=1-bucket['hybrid_seconds']/bucket['local_seconds'] if bucket['local_seconds'] else None
        paired_bins[comparison]=groups
        thresholds[comparison]=next((label for label,bucket in groups.items() if bucket['count'] and bucket['fractional_reduction']>0),None)
    return dict(schema='hybrid-e2e-analysis-v1',prereg=plan['prereg'],session_set_sha256=plan['session_set_sha256'],
        amendment=plan.get('amendment'),series_qualified=series_qualified,flow=flow,comparisons=comparisons,portion_bins=bins,paired_bin_gains=paired_bins,smallest_tested_bin_with_gain=thresholds,
        pricing=dict(gpu_price_per_hour=gpu_price_per_hour,traffic_price_per_GB=traffic_price_per_GB),
        cost_boundary='Native synchronized service wall seconds, not kernel time. TLS TCP payload, not physical billed traffic. Rental wall cost requires the actual rental receipt.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--attempts',required=True,type=Path);p.add_argument('--series-receipt',type=Path);p.add_argument('--gpu-price-per-hour',type=float)
    p.add_argument('--traffic-price-per-GB',type=float);a=p.parse_args()
    attempts=json.loads(a.attempts.read_text())
    for row in attempts:
        if row.get('trace_path'):
            path=Path(row['trace_path']);require(sha256(path)==row['trace_sha256'],'attempt trace hash differs')
            row['trace']=json.loads(path.read_text())
    plan=json.loads(a.plan.read_text());qualified=None
    if plan.get('amendment'):
        receipt=a.series_receipt or a.plan.parent/'SERIES.json'
        qualified=verify_series_transport(receipt,a.plan) if receipt.exists() else False
    print(json.dumps(analyze(plan,attempts,a.gpu_price_per_hour,a.traffic_price_per_GB,qualified),indent=2))


if __name__=='__main__':main()

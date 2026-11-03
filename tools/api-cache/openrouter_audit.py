#!/usr/bin/env python3
"""GET-only cost and price audit; live source and runtime artifacts are read-only."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlencode

import openrouter_cache as api
import openrouter_recovery as recovery
import openrouter_restart as restart

LIVE_ROOT=Path('/home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15d-261103')
MIN_FREE_BYTES=57*1024**3//2
TOLERANCE=Decimal('0.00000001')
PRICE_FIELDS=('prompt','completion','input_cache_read','input_cache_write')


class GetOnlyClient(recovery.Client):
    def request(self,path,body=None):
        if body is not None:raise api.Refusal('audit is GET-only; completion requests refused')
        if not (path=='/key' or path.startswith('/generation?') or (path.startswith('/models/') and path.endswith('/endpoints'))):
            raise api.Refusal('audit GET path refused')
        return super().request(path)


def disk_guard(path):
    if api.shutil.disk_usage(path).free<MIN_FREE_BYTES:
        raise api.Refusal('audit stopped: free disk below28.5 GiB safety threshold (28 GiB floor)')


def output_guard(live,output):
    live,output=live.resolve(),output.resolve()
    if output==live or output.is_relative_to(live):raise api.Refusal('audit output must be outside live runtime and checkout')
    if live.is_relative_to(output):raise api.Refusal('audit output must not contain the live runtime')


def compare_cost(row,data):
    streamed=(row.get('usage') or {}).get('cost')
    if streamed is None:return dict(status='missing-streamed-cost',streamed_cost_usd=None,generation_cost_usd=data.get('total_cost'),difference_usd=None)
    expected=api.money(streamed)
    allowed={row['model']}
    for model in json.loads(api.SNAPSHOT.read_text())['models']:
        if model['data']['id']!=row['model']:continue
        for endpoint in model['data']['endpoints']:
            if endpoint['tag']==row.get('endpoint_tag') and endpoint.get('model_id')==row['model']:
                _,separator,canonical=endpoint['name'].partition(' | ')
                if separator:allowed.add(canonical)
    result=dict(streamed_cost_usd=str(expected),row_cost_usd=row.get('cost_usd'),generation_cost_usd=data.get('total_cost'),difference_usd=None,
                requested_model=row['model'],generation_model=data.get('model'),allowed_model_identities=sorted(allowed),model_identity_basis='exact request alias or exact model name of that endpoint in frozen OpenRouter metadata')
    if data.get('id')!=row.get('generation_id') or data.get('model') not in allowed:
        return dict(result,status='identity-mismatch')
    if data.get('total_cost') is None:return dict(result,status='missing-generation-cost')
    actual=api.money(data['total_cost']);difference=actual-expected
    return dict(result,status='agreement' if abs(difference)<=TOLERANCE else 'mismatch',
                generation_cost_usd=str(actual),difference_usd=str(difference),absolute_difference_usd=str(abs(difference)))


def compare_prices(snapshot,live):
    frozen={(m['data']['id'],e['tag']):e for m in snapshot['models'] for e in m['data']['endpoints']}
    current={(m['data']['id'],e['tag']):e for m in live['models'] for e in m['data']['endpoints']}
    rows=[]
    for model,tag in sorted(set(frozen)|set(current)):
        a,b=frozen.get((model,tag)),current.get((model,tag))
        row=dict(model=model,endpoint_tag=tag,provider=(a or b)['provider_name'],snapshot_pricing=a['pricing'] if a else None,live_pricing=b['pricing'] if b else None,fields={})
        if a is None:row['status']='new-endpoint'
        elif b is None:row['status']='missing-endpoint'
        else:
            for field in PRICE_FIELDS:
                x,y=a['pricing'].get(field),b['pricing'].get(field)
                if x is None and y is None:continue
                drift=(x is None or y is None or api.money(x)!=api.money(y))
                row['fields'][field]=dict(snapshot=x,live=y,drift=drift,delta=None if x is None or y is None else str(api.money(y)-api.money(x)))
            row['status']='drift' if any(f['drift'] for f in row['fields'].values()) else 'agreement'
        rows.append(row)
    return rows


def get_record(client,path,retries=3):
    for attempt in range(retries):
        try:return client.get(path)
        except HTTPError as error:
            status=error.code;error.close()
            if status not in (404,429,502,503,504) or attempt+1==retries:
                return dict(audit_error='HTTP '+str(status),http_status=status,attempts=attempt+1)
        except Exception as error:
            if attempt+1==retries:return dict(audit_error=api.redact(str(error),client.secret),attempts=attempt+1)
        time.sleep((2,5,10)[attempt])


def usage_record(client):
    response=get_record(client,'/key')
    data=response.get('data',{})
    return dict(at=api.utc(),usage_usd=data.get('usage'),limit_remaining_usd=data.get('limit_remaining'),audit_error=response.get('audit_error'))


def capture_rows(live,output):
    # One fixed accepted-row cohort per audit, even while the live writer advances.
    rows=restart.read_rows(live)
    recovery.atomic_json(output/'call-rows.json',rows)
    accepted=[r for r in rows if r.get('accepted')]
    # Pilot/smoke costs are financial evidence only, outside the registered research window.
    for cohort in ('smoke','full','full-v2'):
        for path in sorted((restart.old_root(live)/cohort).glob('call-*.json')):
            row=json.loads(path.read_text())
            if not row.get('accepted'):continue
            row['phase']='legacy-'+cohort
            row.setdefault('provider',row['model']+' / '+row['endpoint_tag'])
            accepted.append(row)
    recovery.atomic_json(output/'accepted-rows.json',accepted)
    return accepted


def source_provenance(live):
    path=live/'workflow.json'
    workflow=json.loads(path.read_text()) if path.exists() else {}
    try:head=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parent),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):head=None
    return dict(source_commit=workflow.get('source_commit'),live_source_sha256=workflow.get('source_sha256'),
                source_commit_basis='live workflow.json at row capture; null if unavailable',audit_source_commit=head)


def audit(live,output,final=False,client_factory=GetOnlyClient):
    output_guard(live,output);disk_guard(output.parent)
    output.mkdir(parents=True,exist_ok=False)
    rows=capture_rows(live,output)
    provenance=source_provenance(live)
    analyzed=restart.analysis(live,final=final,output=output,rows=json.loads((output/'call-rows.json').read_text()))
    started=api.utc();client=client_factory()
    before=usage_record(client)
    snapshot=json.loads(api.SNAPSHOT.read_text());live_prices=dict(at=api.utc(),models=[],errors=[])
    for model in ('qwen/qwen3.8-27b','z-ai/glm-5.3-flash'):
        response=get_record(client,'/models/'+model+'/endpoints')
        if 'data' in response:live_prices['models'].append(response)
        else:live_prices['errors'].append(dict(model=model,**response))
    recovery.atomic_json(output/'endpoint-prices.json',live_prices,client.secret)
    checks=[];records={};limited=False
    for index,row in enumerate(rows):
        disk_guard(output)
        generation=row.get('generation_id')
        base=dict(phase=row['phase'],sequence=row['sequence'],provider=row['provider'],depth=row['depth'],call_index=row['call_index'],generation_id=generation)
        if not generation:
            checks.append(dict(base,status='missing-generation-id'));continue
        if limited:
            checks.append(dict(base,status='not-fetched-endpoint-limit'));continue
        if generation not in records:
            response=get_record(client,'/generation?'+urlencode({'id':generation}))
            records[generation]=response
            recovery.atomic_json(output/('generation-%04d.json'%index),response,client.secret)
        response=records[generation]
        if 'audit_error' in response:
            checks.append(dict(base,status='unavailable',**response))
            if response.get('http_status')==429:limited=True
            continue
        try:checks.append(dict(base,**compare_cost(row,response['data'])))
        except (KeyError,TypeError,api.Refusal) as error:checks.append(dict(base,status='unverifiable',reason=api.redact(str(error),client.secret)))
    after=usage_record(client)
    drift=compare_prices(snapshot,live_prices)
    failed_models={e['model'] for e in live_prices['errors']}
    for row in drift:
        if row['model'] in failed_models:row['status']='unavailable-live-list'
    call_prices=[dict(phase=r['phase'],sequence=r['sequence'],provider=r['provider'],generation_id=r.get('generation_id'),
                     pricing_source='recorded per-call live list' if r.get('pricing') else 'not recorded per-call; historical seed excluded from effective price',pricing=r.get('pricing'),billed_cost_source='streamed usage.cost') for r in rows]
    differences=[api.money(c['absolute_difference_usd']) for c in checks if c.get('absolute_difference_usd') is not None]
    agreements=sum(c['status']=='agreement' for c in checks)
    data=dict(audit='r15e',started_at=started,ended_at=api.utc(),final=final,live_root=str(live),
              **provenance,provider_identity_filter=analyzed['provider_identity_filter'],accepted_calls=len(rows),fetched_unique_generations=len(records),agreement_calls=agreements,
              research_accepted_calls=sum(r['phase'] in ('legacy-v1','v1','amendment') for r in rows),financial_only_accepted_calls=sum(r['phase'] not in ('legacy-v1','v1','amendment') for r in rows),
              mismatch_calls=sum(c['status']=='mismatch' for c in checks),unverified_calls=sum(c['status'] not in ('agreement','mismatch') for c in checks),
              largest_absolute_difference_usd=str(max(differences)) if differences else None,tolerance_usd=str(TOLERANCE),
              sampling='all accepted calls at fixed capture; after a persistent429 the remaining calls are explicitly not fetched' if limited else 'all accepted calls at fixed capture; no sampling',
              cost_source='streamed usage.cost; /generation is a read-only cross-check; cumulative /key and the live ledger bound spend',
              key_usage_before=before,key_usage_after=after,live_counter_warning='Concurrent run and delayed cumulative counters prevent per-call attribution from these two samples; no ledger or charge decision is changed.',
              per_call=checks,price_drift=drift,call_price_sources=call_prices,price_snapshot_sha256=hashlib.sha256(api.SNAPSHOT.read_bytes()).hexdigest(),
              live_source_untouched=True)
    data['audit_source_sha256']={Path(module.__file__).name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in (sys.modules[__name__],api,recovery,restart)}
    ledger=live/'ledger.json'
    data['live_ledger_readback']=json.loads(ledger.read_text()) if ledger.exists() else None
    recovery.atomic_json(output/'audit.json',data,client.secret)
    with (output/'cost-cross-check.csv').open('w',newline='') as handle:
        fields=['phase','sequence','provider','depth','call_index','generation_id','status','streamed_cost_usd','generation_cost_usd','difference_usd']
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for check in checks:writer.writerow({k:api.redact(str(v),client.secret) for k,v in check.items() if k in fields})
    report=output/'API-CACHE-REPORT.md'
    lines=['','## r15e cost source and read-only generation cross-check','',
           'Per-call billed cost is streamed usage.cost; cumulative /key plus the retained-reservation ledger bounds spending. This audit changes no financial or live-run state.',
           'Captured '+str(len(rows))+' accepted calls. Agreements: '+str(agreements)+'; mismatches: '+str(data['mismatch_calls'])+'; unverified: '+str(data['unverified_calls'])+'. Largest absolute difference USD'+str(data['largest_absolute_difference_usd'])+' (toleranceUSD'+str(TOLERANCE)+').',
           'Cost scope includes '+str(data['financial_only_accepted_calls'])+' accepted smoke/pilot calls for financial verification only. They never enter the research availability, cache metrics or effective-price figures above.',
           data['sampling']+'. Per-call agreement and exact differences: cost-cross-check.csv and audit.json. Missing/limited generation records are unverified, never agreement.','',
           '## Frozen snapshot versus live endpoint prices','',
           'Live requests re-estimate reservations and projected effective prices against per-call live list prices. A rise is accepted if the shared cap permits it; the frozen snapshot remains the declared candidate list and historical comparison, not a price-rise refusal. Raw billed costs use usage.cost.',
           'Every accepted call records its price source in audit.json. The historical seed rows lack per-call prices and are excluded from effective-price samples. Current snapshot/live rates, deltas and missing/new endpoints follow.','',
           '| Provider | Model / endpoint | Status | Changed fields: snapshot -> live (delta) |','|---|---|---|---|']
    for item in drift:
        changes=', '.join(f+': '+str(v['snapshot'])+' -> '+str(v['live'])+' ('+str(v['delta'])+')' for f,v in item['fields'].items() if v['drift'])
        lines.append('| '+item['provider']+' | '+item['model']+' / '+item['endpoint_tag']+' | '+item['status']+' | '+(changes or 'none')+' |')
    report.write_text(report.read_text()+api.redact('\n'.join(lines)+'\n',client.secret))
    return data


def observe_end(live,output):
    output_guard(live,output)
    disk_guard(output.parent)
    state=output.parent/'end-observer.json'
    (output.parent/'end-observer.pid').write_text(str(os.getpid())+'\n')
    while True:
        disk_guard(output.parent)
        workflow=json.loads((live/'workflow.json').read_text())
        terminal=live/'terminal.json'
        recovery.atomic_json(state,dict(at=api.utc(),pid=os.getpid(),status='waiting',live_status=workflow['status'],next_check_seconds=30,output=str(output)))
        if terminal.exists() and workflow['status']!='running':break
        if time.time()>restart.epoch('2026-11-05T06:10:00+00:00'):raise api.Refusal('end observer deadline; no terminal workflow observed')
        time.sleep(30)
    result=audit(live,output,final=True)
    refused=result.get('provider_identity_filter',{}).get('rejected_count')
    recovery.atomic_json(state,dict(at=api.utc(),pid=os.getpid(),status='done',accepted_calls=result['accepted_calls'],mismatch_calls=result['mismatch_calls'],unverified_calls=result['unverified_calls'],identity_refused_rows=refused,output=str(output)))
    summary=output.parent/'SUMMARY.md'
    with summary.open('a') as handle:
        handle.write('\nEnd audit '+api.utc()+': '+str(result['accepted_calls'])+' accepted calls; agreements '+str(result['agreement_calls'])+', mismatches '+str(result['mismatch_calls'])+', unverified '+str(result['unverified_calls'])+', largest difference USD'+str(result['largest_absolute_difference_usd'])+'. Identity-refused accepted rows: '+str(refused)+'. Report: '+str(output/'API-CACHE-REPORT.md')+'. Live source/run untouched.\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-root',type=Path,default=LIVE_ROOT)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--watch-end',action='store_true')
    args=parser.parse_args()
    try:
        if os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise api.Refusal('PYTHONDONTWRITEBYTECODE=1 required')
        scratch=Path(os.environ.get('TMPDIR','')).resolve()
        if not scratch.is_dir() or str(scratch)=='/tmp':raise api.Refusal('on-disk task TMPDIR required')
        result=observe_end(args.live_root.resolve(),args.output.resolve()) if args.watch_end else audit(args.live_root.resolve(),args.output.resolve())
        print('done accepted='+str(result['accepted_calls'])+' agreements='+str(result['agreement_calls'])+' mismatches='+str(result['mismatch_calls'])+' unverified='+str(result['unverified_calls'])+' largest_difference_usd='+str(result['largest_absolute_difference_usd']),flush=True)
        return 0
    except Exception as error:
        reason=api.redact(str(error))
        if args.watch_end:
            try:output_guard(args.live_root,args.output)
            except api.Refusal:pass
            else:recovery.atomic_json(args.output.resolve().parent/'end-observer.json',dict(at=api.utc(),pid=os.getpid(),status='failed',reason=reason))
        print('failed '+reason,flush=True)
        return 1


if __name__=='__main__':sys.exit(main())

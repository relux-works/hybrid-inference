#!/usr/bin/env python3
"""Sequential r15d restart with bounded recovery, common-window analysis and heartbeat."""
import argparse
from collections import deque
from datetime import datetime,timezone
from decimal import Decimal
import fcntl
import hashlib
import heapq
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.request import Request,build_opener,ProxyHandler

import openrouter_cache as api
import openrouter_amendment1 as amendment
import openrouter_recovery as recovery

PHASE_ENDS={'v1':'2026-11-04T09:00:00+00:00','amendment':'2026-11-05T05:40:00+00:00'}
WINDOW_START='2026-11-03T14:39:44.389926+00:00'
LOCK_PATH=Path('/home/agent/lanes/hybrid/.temp/openrouter/cache-run.lock')


def epoch(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()


def timestamp(value):
    return datetime.fromtimestamp(value,timezone.utc).isoformat()


def old_root(root):
    return root.parent/'hybrid-d-r15b-261103'


def candidates():
    snapshot=json.loads(api.SNAPSHOT.read_text())
    sessions,_=api.matrix(snapshot)
    result={}
    for s in sessions:
        c={k:s[k] for k in ('model','endpoint','routing_ignore')}
        result[amendment.identity(c)]=c
    return result


def read_rows(root):
    result=[]
    sources=[('legacy-v1',old_root(root)/'full-v3'),('v1',root/'v1'),('amendment',root/'amendment')]
    for phase,directory in sources:
        for path in sorted(directory.glob('call-*.json')):
            row=json.loads(path.read_text())
            row['phase']=phase
            row.setdefault('cohort','v1')
            row.setdefault('provider',row['model']+' / '+row['endpoint_tag'])
            if row.get('started_at','')>=WINDOW_START:result.append(row)
    return result


def price_metrics(rows,candidate,depths):
    warm=recovery.cache_rows(rows)
    measured=amendment.metrics(warm,candidate,depths)
    measured['cache_sample_attempts']=len(warm)
    measured['availability']=recovery.availability(rows)
    valid=[r for r in warm if r.get('accepted')]
    effective={}; weights={}
    for depth in depths:
        subset=[r for r in valid if r['depth']==depth]
        weights[str(depth)]=len(subset)
        projected=[]
        for r in subset:
            if r.get('cached_tokens') is None:break
            p=r.get('pricing',candidate['endpoint']['pricing'])
            retention=min(1,r['cached_tokens']/max(1,r['prompt_tokens']-r['new_tokens_estimate']))
            cache=max(0,depth-1065)*retention
            projected.append(1000*((depth-cache)*(float(p['prompt'])+float(p.get('input_cache_write',0)))+cache*float(p['input_cache_read'])+512*float(p['completion'])))
        effective[str(depth)]=sum(projected)/len(projected) if projected and len(projected)==len(subset) else None
    measured['effective_usd_per_1000_by_depth']=effective
    measured['effective_usd_per_1000_equal_depth_mix']=sum(effective.values())/len(depths) if depths and all(v is not None for v in effective.values()) else None
    measured['measured_depth_mix_counts']=weights
    measured['effective_usd_per_1000_measured_depth_mix']=sum(effective[d]*weights[d] for d in effective if weights[d])/sum(weights.values()) if sum(weights.values()) and all(v is not None for d,v in effective.items() if weights[d]) else None
    zero=[r for r in valid if r['pause_target_seconds']==0]
    known=bool(zero) and all(r.get('cached_tokens') is not None for r in zero)
    measured['pause_levels']['0']=dict(n=len(zero),token_hit_rate=sum(r['cached_tokens'] for r in zero)/sum(r['prompt_tokens'] for r in zero) if known else None,call_hit_rate=sum(r['cached_tokens']>0 for r in zero)/len(zero) if known else None,actual_pauses_seconds=[r['pause_actual_seconds'] for r in zero])
    measured['pricing_basis']='Per-call live list prices; conservative input plus cache-write surcharge; 1065 new input, 512 output. Equal and measured depth mixes disclosed.'
    matched=[r for r in valid if r.get('pause_actual_seconds') is not None and r['pause_target_seconds']-.01<=r['pause_actual_seconds']<=r['pause_target_seconds']+max(2,.1*r['pause_target_seconds'])]
    measured['target_pause_grouping_warning']='pause_levels groups target pauses; actual gaps are retained. Late calls do not establish target TTL.'
    measured['matched_target_pause_levels']={}
    for pause in (0,8,35,60,300,900,3600):
        subset=[r for r in matched if r['pause_target_seconds']==pause]
        reliable=bool(subset) and all(r.get('cached_tokens') is not None for r in subset)
        measured['matched_target_pause_levels'][str(pause)]=dict(n=len(subset),token_hit_rate=sum(r['cached_tokens'] for r in subset)/sum(r['prompt_tokens'] for r in subset) if reliable else None)
    return measured


def analysis(root,final=False,output=None,rows=None):
    rows=read_rows(root) if rows is None else rows
    output=root if output is None else output
    providers=[]
    all_candidates=candidates()
    for name,c in all_candidates.items():
        own=[r for r in rows if r['provider']==name]
        depths=sorted({r['depth'] for r in own})
        phases={}
        for phase in ('v1','amendment'):
            selected=[r for r in own if r['phase']==phase]
            phases[phase]={}
            for cohort in ('v1','core','long'):
                subset=[r for r in selected if r['cohort']==cohort]
                ds=sorted({r['depth'] for r in subset})
                phases[phase][cohort]=price_metrics(subset,c,ds) if ds else None
        million=[r for r in own if r['depth']==1000000 and r['cohort'] not in ('probe','long') and r['phase']!='legacy-v1']
        metrics=price_metrics(million,c,[1000000])
        by_phase={phase:price_metrics([r for r in million if r['phase']==phase],c,[1000000]) for phase in ('v1','amendment')}
        providers.append(dict(provider=name,provider_name=c['endpoint']['provider_name'],model=c['model'],
                              eligibility=recovery.eligible(c,own),eligibility_final=final,
                              whole_window_availability=recovery.availability(own),million=metrics,million_by_phase=by_phase,phases=phases,
                              depths=depths))
    comparisons={}
    for model in ('qwen/qwen3.8-27b','z-ai/glm-5.3-flash'):
        comparisons[model]={}
        for label,without in [('with_alibaba',False),('without_alibaba',True)]:
            selected=[p for p in providers if p['model']==model and (not without or p['provider_name']!='Alibaba')]
            priced=[p for p in selected if p['eligibility']=='eligible' and p['million']['effective_usd_per_1000_by_depth']['1000000'] is not None]
            prices=[p['million']['effective_usd_per_1000_by_depth']['1000000'] for p in priced]
            comparisons[model][label]=dict(providers=[p['provider'] for p in selected],priced_providers=[p['provider'] for p in priced],
                                          mean_effective_usd_per_1000=sum(prices)/len(prices) if prices else None)
    samples=[]
    for directory in (old_root(root)/'full-v3',root/'uptime-window'):
        for path in sorted(directory.glob('uptime-*.json')):
            sample=json.loads(path.read_text()); samples.append(sample)
    for p in providers:
        observations=[]; missing=[]
        for sample in samples:
            try:
                endpoint=api.live_endpoint(sample,all_candidates[p['provider']])
                value=endpoint.get('uptime_last_1d')
                if value is None:raise api.Refusal('uptime absent')
                observations.append(dict(at=sample['at'],uptime_last_1d=value))
            except api.Refusal:missing.append(sample['at'])
        p['remote_daily_uptime']=dict(observations=observations,missing_at=missing,min=min((o['uptime_last_1d'] for o in observations),default=None),max=max((o['uptime_last_1d'] for o in observations),default=None))
        own=[r for r in rows if r['provider']==p['provider']]
        p['cold_probe_latencies']=[dict(depth=r['depth'],call_index=r['call_index'],**amendment.miss_latency(r)) for r in own if r.get('cohort')=='probe']
    _,excluded=api.matrix(json.loads(api.SNAPSHOT.read_text()))
    data=dict(protocol='r15d+r15e-analysis',million_aggregation='pooled restarted-v1 multi-pause cadence plus amendment core cadence; independent probe/long substudy excluded',window_start=WINDOW_START,window_observed_through=api.utc(),window_final=final,excluded_impossible_endpoints=excluded,
              providers=providers,million_token_comparison=comparisons,
              remote_uptime_samples=len(samples),remote_sample_times=[s['at'] for s in samples],
              smoke_used_for_cache=False,seed_calls_used_for_cache=False,
              rules='Only context/no declared cache/whole-window availability exclude. Zero hits are priced. Daily windows overlap.')
    recovery.atomic_json(output/'analysis.json',data)
    lines=['# API cache report - r15d','', 'Whole window: '+WINDOW_START+' through '+data['window_observed_through']+'. Final: '+str(final)+'.','',
           'The whole Chutes smoke and every call_index0 are excluded from cache metrics. All attempted calls, including seeds and charge-resolved timeouts, count toward availability. Old full-v3 attempts join both new phases; pilot/configuration cohorts do not.','',
           'Known zero hits are priced at the measured rate. Missing usage stays unknown. Core and long-pause prices stay separate. Remote rolling daily statistics are metadata, not a substitute for whole-window own availability.','',
           'Headline1M is pooled across restarted-v1 multi-pause cadence and amendment core cadence. Per-phase1M figures are beside it; the separate long-pause substudy and probes are excluded.','',
           '| Endpoint | Attempts | Errors | Timeouts | Availability | Cache analysis calls at1M | Pooled1M USD/1000 | V1-only1M USD/1000 | Amendment-core1M USD/1000 | Eligibility |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    def fmt(v):return 'unknown' if v is None else '%.6g'%v
    for p in providers:
        av=p['whole_window_availability']; m=p['million']
        lines.append('| '+' | '.join([p['provider'],str(av['attempts']),str(av['errors']),str(av['timeouts']),fmt(av['rate']),str(m['cache_sample_attempts']),fmt(m['effective_usd_per_1000_by_depth']['1000000']),fmt(p['million_by_phase']['v1']['effective_usd_per_1000_by_depth']['1000000']),fmt(p['million_by_phase']['amendment']['effective_usd_per_1000_by_depth']['1000000']),p['eligibility']+(' (provisional)' if not final else '')])+' |')
    lines+=['','## Profile and logs comparison','','| Endpoint | Phase/cohort | Warm accepted | Cached fraction | Uncached/call | Delta vs3200 | USD/1000 measured depth mix |','|---|---|---:|---:|---:|---:|---:|']
    for p in providers:
        for phase,cohorts in p['phases'].items():
            for cohort,m in cohorts.items():
                if m is not None:lines.append('| '+' | '.join([p['provider'],phase+'/'+cohort,str(m['accepted']),fmt(m['token_hit_rate']),fmt(m['mean_uncached_tokens']),fmt(m['delta_uncached_vs_logs']),fmt(m['effective_usd_per_1000_measured_depth_mix'])])+' |')
    lines+=['','## Pooled1M with and without Alibaba','']
    for model,result in comparisons.items():
        for label,value in result.items():
            lines.append(model+' / '+label+': '+fmt(value['mean_effective_usd_per_1000'])+' USD/1000 over '+str(len(value['priced_providers']))+' qualified, measured providers; all retained provider identities are in analysis.json.')
    lines+=['','## Cold-probe latency','','| Endpoint | Depth | Classification | TTFT seconds |','|---|---:|---|---:|']
    for p in providers:
        for probe in p['cold_probe_latencies']:lines.append('| '+p['provider']+' | '+str(probe['depth'])+' | '+probe['status']+' | '+fmt(probe['ttft_seconds'])+' |')
    lines+=['','Equal provider weights in these aggregates; per-provider prices retain disclosed depth/profile assumptions. The JSON contains core/long pause hit rates, measured depth mixes and differences from the logs baseline (98.1% cached, 3200 uncached tokens/call). Availability is request-sampled over the whole stated window and does not prove continuous uptime. Rolling daily observations and missing samples are retained per provider in JSON. No surface table or independent review was supplied.','']
    (output/'API-CACHE-REPORT.md').write_text('\n'.join(lines))
    return data


def phase_sessions(phase):
    if phase=='v1':
        source,_=api.matrix(json.loads(api.SNAPSHOT.read_text()))
        result=[]
        for s in source:
            c={k:s[k] for k in ('model','endpoint','routing_ignore')}
            result.append(dict(candidate=c,cohort='v1',depth=s['depth'],pauses=list(api.PAUSES),session_id=s['session_id'].replace('r15b-v3-','r15d-v1-')))
        return result
    manifest=json.loads(amendment.MANIFEST.read_text())
    manifest['run']='hybrid-d-r15d-261103-amendment1'
    result=[]
    for c in manifest['candidates']:
        for s in amendment.sessions(manifest,c):
            s['session_id']=s['session_id'].replace('r15c-amendment-1-','r15d-amendment-1-')
            result.append(s)
    return result

class Phase:
    def __init__(self,root,name,client,until=None,resume=False):
        self.root,self.name,self.client=root,name,client
        self.out=root/name; self.out.mkdir(exist_ok=resume)
        self.end=epoch(PHASE_ENDS[name]) if until is None else until
        self.cap=22 if name=='v1' else 45
        self.budget=recovery.Budget.load(root/'ledger.json')
        self.rows,self.skips=[],[]
        self.preflights=len(list(self.out.glob('uptime-*.json')))
        if resume:
            for path in sorted(self.out.glob('call-*.json')):
                row=json.loads(path.read_text())
                if name+'/%04d'%row['sequence'] not in self.budget.receipts:raise api.Refusal('resume has unresolved charge')
                self.rows.append(row)
            intents={p.name.replace('intent-','call-') for p in self.out.glob('intent-*.json')}
            if intents-{p.name for p in self.out.glob('call-*.json')}:raise api.Refusal('resume has unresolved request intent')
            for filename in ('terminal.json','run.json'):
                path=self.out/filename
                if path.exists():path.replace(self.out/(filename.removesuffix('.json')+'-history-'+str(time.time_ns())+'.json'))
        self.planned=phase_sessions(name)
        self.status,self.reason='running','collecting'
        recovery.atomic_json(self.out/'run.json',dict(phase=name,pid=os.getpid(),at=api.utc(),cutoff=timestamp(self.end),cap_usd=self.cap,planned_sessions=len(self.planned),deadlines=recovery.DEPTH_DEADLINES))

    def csv(self):
        import csv
        fields=list(dict.fromkeys(k for row in self.rows for k in row if k not in ('stream','usage','error_body')))
        with (self.out/'calls.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            for row in self.rows:writer.writerow({k:api.redact(json.dumps(v) if isinstance(v,(dict,list)) else str(v),self.client.secret) for k,v in row.items() if k in fields})

    def progress(self,**changes):
        data=dict(phase=self.name,pid=os.getpid(),at=api.utc(),last_completed_call=self.rows[-1] if self.rows else None,
                  committed_upper_usd=str(self.budget.committed),known_observed_usd=str(self.budget.known),**changes)
        if data['last_completed_call'] is not None:
            data['last_completed_call']={k:data['last_completed_call'].get(k) for k in ('sequence','provider','cohort','depth','call_index','accepted','error','ended_at')}
        recovery.atomic_json(self.root/'progress.json',data,self.client.secret)

    def skip(self,s,index,reason):
        self.skips.append(dict(provider=amendment.identity(s['candidate']),cohort=s['cohort'],depth=s['depth'],session_id=s['session_id'],missing_call_indices=list(range(index,len(s['pauses']))),reason=reason))

    def call(self,s,index,previous_end=None):
        seconds=recovery.deadline_for_depth(s['depth'])
        due=time.monotonic() if previous_end is None else previous_end+s['pauses'][index]
        if time.time()+max(0,due-time.monotonic())+seconds+600>self.end:
            self.skip(s,index,'insufficient phase time for unchanged pause and complete deadline/recovery'); return None
        m=dict(new_tokens=1000 if self.name=='v1' else 1065,framing_reserve=128,max_tokens=16)
        payload=amendment.payload(m,s,index)
        text=payload['messages'][0]['content'][0]['text']
        candidate=s['candidate']; endpoint=candidate['endpoint']
        estimated=api.max_cost(endpoint,min(len(text.encode())+512,endpoint['context_length']))
        try:self.budget.reserve(self.budget.known,estimated,self.cap)
        except api.Refusal as error:
            if 'budget' not in str(error):raise
            self.skip(s,index,str(error)); return None
        self.progress(status='waiting',current_call=None,next_call_time=timestamp(time.time()+max(0,due-time.monotonic())),next_call_reason='registered pause')
        while time.monotonic()<due:
            if (self.root/'STOP').exists():raise api.Refusal('operator STOP')
            time.sleep(min(30,max(0,due-time.monotonic())))
        if (self.root/'STOP').exists():raise api.Refusal('operator STOP')
        if api.shutil.disk_usage(self.root).free<recovery.MIN_FREE_BYTES:raise api.Refusal('free disk below28 GiB')
        sample=api.sample_endpoints(self.client,self.out,self.preflights); self.preflights+=1
        endpoint=api.live_endpoint(sample,candidate)
        if endpoint['context_length']<s['depth'] or 'input_cache_read' not in endpoint['pricing']:
            self.skip(s,index,'live context/cache declaration makes this cell impossible'); return None
        estimated=api.max_cost(endpoint,min(len(text.encode())+512,endpoint['context_length']))
        key=self.client.get('/key')['data']; live=api.money(key['usage'])
        if live>self.budget.committed+recovery.ROUNDING:raise api.Refusal('live spend exceeds committed upper bound')
        self.budget.known=max(self.budget.known,live); self.budget.save()
        try:self.budget.reserve(live,estimated,self.cap,key.get('limit_remaining'))
        except api.Refusal as error:
            if 'budget' not in str(error):raise
            self.skip(s,index,str(error)); return None
        sequence=len(self.rows); call_id=self.name+'/%04d'%sequence
        row=dict(protocol='r15d',phase=self.name,sequence=sequence,provider=amendment.identity(candidate),model=candidate['model'],
                 endpoint_tag=endpoint['tag'],cohort=s['cohort'],depth=s['depth'],session_id=s['session_id'],call_index=index,
                 new_tokens_estimate=m['new_tokens'] if index else None,pause_target_seconds=s['pauses'][index],
                 pause_actual_seconds=None if previous_end is None else time.monotonic()-previous_end,started_at=api.utc(),
                 expected_spend_before_usd=str(self.budget.known),maximum_cost_usd=str(estimated),pricing=endpoint['pricing'],
                 request_deadline_seconds=seconds,provider_request=payload['provider'],prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),accepted=False)
        recovery.atomic_json(self.out/('intent-%04d.json'%sequence),row,self.client.secret)
        self.progress(status='in-flight',current_call={k:row[k] for k in ('sequence','provider','cohort','depth','call_index')},next_call_time=None,
                      next_call_reason='awaiting response',current_request_deadline_at=timestamp(time.time()+seconds))
        row.update(recovery.stream(self.client,payload,endpoint,s['depth'],self.out/('stream-%04d.jsonl'%sequence)))
        ended=time.monotonic(); row['ended_at']=api.utc()
        self.rows.append(row)
        recovery.atomic_json(self.out/('call-%04d.json'%sequence),row,self.client.secret)
        self.csv()
        usage=row.get('usage')
        if row.get('cost_usd') is None and isinstance(usage,dict) and usage.get('cost') is not None:
            row['cost_usd']=float(api.money(usage['cost']))
        if row.get('cost_usd') is not None:
            row['charge_resolution']=self.budget.reported_call(row['cost_usd'],estimated,call_id)
        else:
            samples=[]
            try:
                samples=recovery.settled_samples(self.client,self.budget.known,estimated,self.out/('settlement-%04d.json'%sequence))
                row['charge_resolution']=self.budget.recovered_call(samples,estimated,call_id)
            except api.Refusal:
                recovery.atomic_json(self.out/('unresolved-%04d.json'%sequence),dict(samples=samples,expected_usd=str(self.budget.known),maximum_usd=str(estimated)),self.client.secret)
                raise
        if not row.get('error'):
            target=s['depth']-(len(s['pauses'])-1-index)*m['new_tokens']
            if abs(row['prompt_tokens']-target)>max(512,target*.03):row['error']='measured depth outside tolerance'
            elif row['completion_tokens']>16:row['error']='completion cap exceeded'
            else:row['accepted']=True
        recovery.atomic_json(self.out/('call-%04d.json'%sequence),row,self.client.secret)
        self.csv()
        print(api.redact(json.dumps({k:row.get(k) for k in ('sequence','provider','cohort','depth','call_index','cached_tokens','timed_out','accepted','error')}),self.client.secret),flush=True)
        self.progress(status='between-calls',current_call=None,next_call_time=timestamp(time.time()+s['pauses'][index+1]) if index+1<len(s['pauses']) else None,next_call_reason='next registered call or new series')
        analysis(self.root)
        if row.get('error') and any(term in row['error'] for term in ('provider differs','depth outside','cap exceeded','evidence')):
            raise api.Refusal('instrument/identity violation; no further spending')
        return ended

    def interleave(self,source):
        seeds=deque();warm=[];serial=0
        for s in source:
            previous=[r for r in self.rows if r['session_id']==s['session_id']]
            if not previous:seeds.append(s);continue
            last=max(previous,key=lambda r:r['call_index']);index=last['call_index']+1
            if index<len(s['pauses']):
                ended=time.monotonic()-(time.time()-epoch(last['ended_at']));serial+=1
                heapq.heappush(warm,(ended+s['pauses'][index],serial,s,index,ended))
        while seeds or warm:
            if warm and (warm[0][0]<=time.monotonic() or not seeds):
                _,_,s,index,ended=heapq.heappop(warm)
            else:s,index,ended=seeds.popleft(),0,None
            ended=self.call(s,index,ended)
            if ended is not None and index+1<len(s['pauses']):
                serial+=1
                heapq.heappush(warm,(ended+s['pauses'][index+1],serial,s,index+1,ended))

    def run(self):
        try:
            if self.name=='v1':self.interleave(self.planned)
            else:
                manifest=json.loads(amendment.MANIFEST.read_text())
                observed=analysis(self.root)
                previous={'providers':[]}
                for p in observed['providers']:
                    value=p['million']['effective_usd_per_1000_by_depth']['1000000']
                    previous['providers'].append(dict(provider=p['provider'],admission=p['eligibility'],depths=[1000000],effective_usd_per_1000_by_depth=[value]))
                ordered,ranking=amendment.priority_order(manifest,previous)
                recovery.atomic_json(self.out/'ranking.json',ranking)
                long=[]
                for c in ordered:
                    selected=[s for s in self.planned if amendment.identity(s['candidate'])==amendment.identity(c)]
                    for depth in reversed(manifest['depths']):
                        for cohort in ('probe','core'):
                            s=next(s for s in selected if s['depth']==depth and s['cohort']==cohort)
                            ended=None
                            for index in range(len(s['pauses'])):
                                ended=self.call(s,index,ended)
                                if ended is None:break
                    core=[r for r in self.rows if r['provider']==amendment.identity(c) and r['cohort']=='core']
                    for s in selected:
                        if s['cohort']=='long':
                            if len(core)==20:long.append(s)
                            else:self.skip(s,0,'core coverage incomplete')
                self.interleave(long)
            self.status,self.reason='done','bounded phase ended; explicit skips retained'
        except Exception as error:
            self.status,self.reason='failed',api.redact(str(error),self.client.secret)
            if self.rows and not self.rows[-1].get('accepted'):
                self.rows[-1]['error']=self.rows[-1].get('error') or self.reason
                recovery.atomic_json(self.out/('call-%04d.json'%self.rows[-1]['sequence']),self.rows[-1],self.client.secret)
        finally:
            seen={(r['session_id'],r['call_index']) for r in self.rows}
            known={s['session_id'] for s in self.skips}
            for s in self.planned:
                missing=[i for i in range(len(s['pauses'])) if (s['session_id'],i) not in seen]
                if missing and s['session_id'] not in known:self.skip(s,missing[0],self.reason)
            recovery.atomic_json(self.out/'skipped.json',self.skips)
            try:
                self.csv()
                analysis(self.root)
            except Exception as error:
                self.status,self.reason='failed','output validation: '+api.redact(str(error),self.client.secret)
            recovery.atomic_json(self.out/'terminal.json',dict(status=self.status,reason=self.reason,at=api.utc(),attempts=len(self.rows),skipped_cells=len(self.skips)))
            self.progress(status=self.status,current_call=None,next_call_time=None,next_call_reason=self.reason)
            print(self.status+' phase='+self.name+' '+self.reason,flush=True)
        return 0 if self.status=='done' else 1


def source_identity():
    paths=[Path(__file__),Path(recovery.__file__),Path(api.__file__),Path(amendment.__file__)]
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def public_snapshot(root,index):
    opener=build_opener(ProxyHandler({}),api.NoRedirect())
    models=[]; failures=[]
    at=api.utc()
    for model in ('qwen/qwen3.8-27b','z-ai/glm-5.3-flash'):
        try:
            with opener.open(Request('https://openrouter.ai/api/v1/models/'+model+'/endpoints'),timeout=20) as response:
                models.append(json.load(response))
        except Exception as error:failures.append(dict(model=model,error=api.redact(str(error))))
    directory=root/'uptime-window';directory.mkdir(exist_ok=True)
    recovery.atomic_json(directory/('uptime-%04d.json'%index),dict(at=at,models=models,failures=failures))


def heartbeat(root,client,status='running'):
    def read(name):
        path=root/name
        return json.loads(path.read_text()) if path.exists() else {}
    progress=read('progress.json');ledger=read('ledger.json')
    data=dict(at=api.utc(),status=status,supervisor_pid=os.getpid(),phase=progress.get('phase'),
              last_call=progress.get('last_completed_call'),current_call=progress.get('current_call'),
              known_observed_usd=ledger.get('known_observed_usd'),committed_upper_usd=ledger.get('committed_upper_usd'),
              next_call_time=progress.get('next_call_time'),next_call_reason=progress.get('next_call_reason','phase transition'),
              current_request_deadline_at=progress.get('current_request_deadline_at'),interval_seconds=600)
    data['free_disk_bytes']=api.shutil.disk_usage(root).free
    if data['free_disk_bytes']<recovery.MIN_FREE_BYTES:(root/'STOP').touch()
    if status!='running':data.update(current_call=None,next_call_time=None,next_call_reason='workflow '+status)
    # Persist custody before a GET, so metadata delays cannot hide an in-flight request.
    recovery.atomic_json(root/'heartbeat.json',data)
    try:
        data['live_usage_usd']=str(client.get('/key')['data']['usage']);data['live_usage_sample_at']=api.utc()
    except Exception as error:data['live_usage_error']=api.redact(str(error),client.secret)
    recovery.atomic_json(root/'heartbeat.json',data,client.secret)
    return data


def recover_interrupted(root,phase,client):
    directory=root/phase
    ledger=recovery.Budget.load(root/'ledger.json')
    for path in sorted(directory.glob('intent-*.json')):
        intent=json.loads(path.read_text());sequence=intent['sequence']
        call_id=phase+'/%04d'%sequence
        rowpath=directory/('call-%04d.json'%sequence)
        row=json.loads(rowpath.read_text()) if rowpath.exists() else intent
        receipt=ledger.receipts.get(call_id)
        if receipt is not None:
            if receipt.get('within_maximum') is False:raise api.Refusal('previous reported cost exceeds reservation')
        else:
            # An intent may have reached the server; even a killed writer needs a bounded charge.
            samples=recovery.settled_samples(client,ledger.known,intent['maximum_cost_usd'],directory/('settlement-watchdog-%04d.json'%sequence))
            receipt=ledger.recovered_call(samples,intent['maximum_cost_usd'],call_id)
        if not row.get('charge_resolution'):
            row.update(accepted=False,timed_out=True,error=row.get('error') or 'supervisor watchdog interrupted request',ended_at=row.get('ended_at') or api.utc(),charge_resolution=receipt)
            recovery.atomic_json(rowpath,row,client.secret)


def write_summary(root,status,reason):
    ledger=recovery.Budget.load(root/'ledger.json')
    workflow=json.loads((root/'workflow.json').read_text())
    lines=['# r15d status','','Status: '+status+'. '+reason,'',
           'Historical call34 resolved by three stable cumulative-usage reads: USD2.153047576; no additional observed debit, USD0.420 maximum retained as a safety reservation. Its availability failure and missing usage remain.','',
           'Known observed/accounted cumulative USD'+str(ledger.known)+'; conservative committed upper USD'+str(ledger.committed)+'; cap USD45. Reserved amounts are not reported charges.','',
           'Supervisor PID '+str(workflow['pid'])+'; phase PIDs '+json.dumps(workflow.get('phase_pids',{}))+'.',
           'Log: '+str(root/'supervisor.log')+'. Heartbeat: '+str(root/'heartbeat.json')+'.',
           'Signed protocol: tools/api-cache/RESTART-r15d.md. Source identity and start time: workflow.json.','',
           'V1 cutoff 2026-11-04 09:00Z; amendment cutoff 2026-11-05 05:40Z. No pause shortened. Budget/time skips are explicit in each phase/skipped.json.',
           'Analysis: API-CACHE-REPORT.md and analysis.json; every seed and the whole Chutes smoke excluded from cache results. All research attempts, including timeouts, count toward common-window availability.','']
    for name in ('v1','amendment'):
        path=root/name/'terminal.json'
        if path.exists():lines.append(name+': '+json.dumps(json.loads(path.read_text())))
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


def supervise(root,resume=False):
    previous=None
    if (root/'workflow.json').exists():
        if not resume:raise api.Refusal('workflow already initialized; inspect existing custody before any replay')
        previous=json.loads((root/'workflow.json').read_text())
        if previous['status']!='failed':raise api.Refusal('resume requires failed terminal workflow')
        if list((root/'amendment').glob('call-*.json')):raise api.Refusal('amendment already has data; automatic replay refused')
    elif resume:raise api.Refusal('resume requires existing workflow')
    if api.shutil.disk_usage(root).free<recovery.MIN_FREE_BYTES:raise api.Refusal('free disk below28 GiB')
    lock=LOCK_PATH.open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    client=recovery.Client()
    if resume:
        ledger=recovery.Budget.load(root/'ledger.json')
        # Prove every attempted request settled before replacing failed custody.
        for path in (root/'v1').glob('intent-*.json'):
            row=json.loads(path.read_text());call_id='v1/%04d'%row['sequence']
            if call_id not in ledger.receipts:raise api.Refusal('resume has unresolved charge')
            if ledger.receipts[call_id].get('within_maximum') is False:raise api.Refusal('resume has excessive charge')
        samples=recovery.settled_samples(client,ledger.known,ledger.committed-ledger.known,root/'resume-settlement.json')
        ledger.known=max(ledger.known,api.money(samples[-1]['usage_usd']));ledger.save()
        ledger.reserve(ledger.known,Decimal(0))
    else:
        proof=json.loads((root/'call34-resolution.json').read_text())
        expected=proof['expected_without_call_usd'];maximum=proof['maximum_usd']
        samples=recovery.usage_samples(client)
        resolution=recovery.resolve_delta(samples,expected,maximum)
        recovery.atomic_json(root/'call34-revalidated.json',resolution,client.secret)
        known=max(api.money(expected),api.money(samples[-1]['usage_usd']))
        ledger=recovery.Budget(root/'ledger.json',known,api.money(expected)+api.money(maximum));ledger.save()
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','verify-commit',head],check=True,capture_output=True)
    dirty=subprocess.check_output(['git','status','--porcelain','--','tools/api-cache'],text=True)
    if dirty:raise api.Refusal('cache instrumentation must be committed before new data')
    if previous is not None:
        stamp=str(time.time_ns())
        recovery.atomic_json(root/('workflow-history-'+stamp+'.json'),previous)
        if (root/'terminal.json').exists():(root/'terminal.json').replace(root/('terminal-history-'+stamp+'.json'))
    workflow=dict(pid=os.getpid(),started_at=api.utc(),status='running',resumed=resume,phase_pids={},source_commit=head,source_sha256=source_identity(),
                  phase_cutoffs=PHASE_ENDS,global_cap_usd=45,preregistration_commit='4ab91fcbb',network='HTTPS openrouter.ai only')
    if resume:workflow['resume_registration_commit']=head
    recovery.atomic_json(root/'workflow.json',workflow)
    (root/'supervisor.pid').write_text(str(os.getpid())+'\n')
    stop=threading.Event()
    def monitor():
        index=len(list((root/'uptime-window').glob('uptime-*.json')))
        while not stop.is_set():
            started=time.monotonic()
            try:
                heartbeat(root,client)
                public_snapshot(root,index);index+=1
            except Exception as error:
                recovery.atomic_json(root/'heartbeat-error.json',dict(at=api.utc(),error=api.redact(str(error),client.secret)))
            stop.wait(max(1,600-(time.monotonic()-started)))
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    status,reason='failed','supervisor interrupted'
    write_summary(root,'running','sequential v1 then amendment; guarded restart launched')
    try:
        for phase in ('v1','amendment'):
            if time.time()+600>=epoch(PHASE_ENDS[phase]):raise api.Refusal('phase start too late: '+phase)
            env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['TMPDIR']=str(root/'tmp')
            with (root/(phase+'.log')).open('a' if resume else 'w') as log:
                argv=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--phase',phase,'--lease-fd',str(lock.fileno())]
                if resume:argv.append('--resume')
                child=subprocess.Popen(argv,
                                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=(lock.fileno(),))
                workflow['phase_pids'][phase]=child.pid
                recovery.atomic_json(root/'workflow.json',workflow)
                (root/(phase+'.pid')).write_text(str(child.pid)+'\n')
                print('started phase='+phase+' pid='+str(child.pid)+' log='+str(root/(phase+'.log')),flush=True)
                try:code=child.wait(timeout=max(1,epoch(PHASE_ENDS[phase])-time.time()))
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                    recovery.atomic_json(root/phase/'terminal.json',dict(status='failed',at=api.utc(),reason='supervisor watchdog deadline'))
                    log.write('failed phase='+phase+' supervisor watchdog deadline\n');log.flush()
                    recover_interrupted(root,phase,client)
                    raise api.Refusal('supervisor watchdog ended '+phase)
            recover_interrupted(root,phase,client)
            if code:raise api.Refusal('phase failed: '+phase+'; inspect terminal and charge evidence; no next phase')
        status,reason='done','both bounded phases terminated; budget/time omissions retained'
    except Exception as error:
        status,reason='failed',api.redact(str(error),client.secret)
    finally:
        stop.set();thread.join(timeout=75)
        workflow.update(status=status,reason=reason,ended_at=api.utc())
        recovery.atomic_json(root/'workflow.json',workflow)
        for phase in ('v1','amendment'):
            directory=root/phase
            if not directory.exists():
                directory.mkdir()
            if not (directory/'terminal.json').exists():
                recovery.atomic_json(directory/'terminal.json',dict(status='failed',at=api.utc(),attempts=0,reason='not started: '+reason))
                with (root/(phase+'.log')).open('a') as handle:handle.write('failed phase='+phase+' not started: '+reason+'\n')
        try:analysis(root,final=True)
        except Exception as error:
            status,reason='failed','analysis: '+api.redact(str(error),client.secret)
            recovery.atomic_json(root/'analysis-error.json',dict(at=api.utc(),error=reason))
        heartbeat(root,client,status)
        recovery.atomic_json(root/'terminal.json',dict(status=status,reason=reason,at=api.utc()))
        workflow.update(status=status,reason=reason);recovery.atomic_json(root/'workflow.json',workflow)
        write_summary(root,status,reason)
        print(status+' '+reason,flush=True)
        lock.close()
    return 0 if status=='done' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=('v1','amendment'))
    parser.add_argument('--lease-fd',type=int)
    parser.add_argument('--resume',action='store_true',help='continue a failed v1 only after every prior intent has a settled receipt')
    args=parser.parse_args()
    root=args.root.resolve()
    try:
        if os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise api.Refusal('PYTHONDONTWRITEBYTECODE=1 required')
        scratch=Path(os.environ.get('TMPDIR','')).resolve()
        if scratch!=root/'tmp' or not scratch.is_dir():raise api.Refusal('task-local on-disk TMPDIR required')
        if args.phase:
            if args.lease_fd is None:raise api.Refusal('inherited exclusive lease required')
            lease=os.fdopen(args.lease_fd,'a+',closefd=False)
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return Phase(root,args.phase,recovery.Client(),resume=args.resume).run()
        return supervise(root,resume=args.resume)
    except Exception as error:
        reason=api.redact(str(error))
        recovery.atomic_json(root/'startup-failed.json',dict(at=api.utc(),error=reason))
        recovery.atomic_json(root/'terminal.json',dict(at=api.utc(),status='failed',reason=reason))
        recovery.atomic_json(root/'heartbeat.json',dict(at=api.utc(),status='failed',reason=reason,pid=os.getpid(),next_call_time=None,next_call_reason='startup refused'))
        if not (root/'workflow.json').exists():(root/'SUMMARY.md').write_text('# r15d status\n\nStatus: failed. '+reason+'\nNo new paid request started. Inspect startup-failed.json and supervisor.log.\n')
        print('failed '+reason,flush=True)
        return 1


if __name__=='__main__':
    sys.exit(main())

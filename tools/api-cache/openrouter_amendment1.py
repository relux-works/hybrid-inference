#!/usr/bin/env python3
"""Preregistered Amendment 1; leaves the original runner and its custody untouched."""
import argparse
import csv
import fcntl
import hashlib
import heapq
import json
import os
from pathlib import Path
import time
from decimal import Decimal
from urllib.parse import urlencode

import openrouter_cache as api

MANIFEST = api.HERE / 'snapshots/amendment-1-2026-11-03.json'
PRIOR = api.HERE.parents[1] / '.temp/hybrid-d-r15b-261103'


class BudgetSkip(api.Refusal):
    pass


def identity(candidate):
    return candidate['model'] + ' / ' + candidate['endpoint']['tag']


def nominal_price(candidate):
    p = candidate['endpoint']['pricing']
    uncached = Decimal(19000)
    return (uncached * (api.money(p['prompt']) + api.money(p.get('input_cache_write', 0)))
            + Decimal(981000) * api.money(p['input_cache_read']) + Decimal(512) * api.money(p['completion']))


def priority_order(manifest, previous=None):
    candidates = manifest['candidates']
    qwen = [c for c in candidates if c['model'] == 'qwen/qwen3.8-27b']
    glm = [c for c in candidates if c['model'] == 'z-ai/glm-5.3-flash']
    prices = {}
    for row in (previous or {}).get('providers', []):
        if row.get('admission') != 'eligible' or 1000000 not in row.get('depths', []):
            continue
        value = row['effective_usd_per_1000_by_depth'][row['depths'].index(1000000)]
        if value is not None:
            prices[row['provider']] = api.money(value)
    admitted = sorted([c for c in qwen if identity(c) in prices], key=lambda c:(prices[identity(c)], identity(c)))
    first = []
    labels = {}
    if admitted:
        for label, candidate in [('cheapest admitted Qwen', admitted[0]), ('median admitted Qwen', admitted[len(admitted)//2])]:
            if candidate not in first:
                first.append(candidate)
                labels[identity(candidate)] = label
    for candidate in sorted(qwen, key=lambda c:(nominal_price(c), identity(c))):
        if len(first) >= 2:
            break
        if candidate not in first:
            first.append(candidate)
            labels[identity(candidate)] = 'provisional Qwen; admission not established'
    ordered = first + sorted(glm, key=lambda c:(nominal_price(c), identity(c)))
    ordered += [c for c in sorted(qwen, key=lambda c:(nominal_price(c), identity(c))) if c not in first]
    ranking = [dict(provider=identity(c), priority=i+1, label=labels.get(identity(c), 'GLM' if c in glm else 'remaining Qwen'),
                    nominal_1m_usd=float(nominal_price(c)), prior_admitted_1m_usd_per_1000=float(prices[identity(c)]) if identity(c) in prices else None)
               for i,c in enumerate(ordered)]
    return ordered, ranking


def active_prior(prior):
    active = []
    for name in ('full.pid', 'full-v2.pid', 'full-v3.pid'):
        path = prior/name
        if not path.exists():
            continue
        pid = int(path.read_text().strip())
        process = Path('/proc')/str(pid)
        try:
            command = (process/'cmdline').read_bytes()
            state = (process/'stat').read_text().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            continue
        if state != 'Z' and (b'openrouter_cache.py' in command or b'full_v3_supervisor.py' in command):
            active.append(pid)
    return active


def try_custody(prior, lock_path):
    if active_prior(prior):
        return None
    handle = lock_path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except BlockingIOError:
        handle.close()
        return None


def prior_costs(prior, client, out):
    total = Decimal(0)
    reconciled = []
    for cohort in ('smoke', 'full', 'full-v2', 'full-v3'):
        directory = prior/cohort
        intents = {p.name.replace('intent-', 'call-') for p in directory.glob('intent-*.json')}
        calls = {p.name for p in directory.glob('call-*.json')}
        if intents - calls:
            raise api.Refusal('previous request intent unresolved: '+cohort)
        for path in sorted(directory.glob('call-*.json')):
            row = json.loads(path.read_text())
            cost = row.get('cost_usd')
            if cost is None:
                body = json.loads(row.get('error_body') or '{}')
                metadata = body.get('error', {}).get('metadata', {})
                if row.get('http_status') == 400 and 'provider_name' in metadata and metadata['provider_name'] is None:
                    cost = 0
                else:
                    generation = row.get('generation_id') or next((c.get('id') for c in row.get('stream', []) if c.get('id')), None)
                    if not generation:
                        raise api.Refusal('previous charge unresolved: '+str(path))
                    data = client.get('/generation?'+urlencode({'id':generation}))['data']
                    if not data.get('finish_reason') and not data.get('cancelled'):
                        raise api.Refusal('previous generation not settled')
                    if data.get('id') != generation or data.get('model') != row['model']:
                        raise api.Refusal('previous generation identity differs')
                    cost = data.get('total_cost')
                    api.money(cost)
                    api.write_json(out/('reconcile-'+cohort+'-'+path.name), data, client.secret)
            total += api.money(cost)
            reconciled.append(dict(source=str(path), cost_usd=str(cost)))
    api.write_json(out/'prior-costs.json', dict(total_usd=str(total), calls=reconciled), client.secret)
    return total


def sessions(manifest, candidate):
    result = []
    for cohort in ('probe', 'core', 'long'):
        pauses = [0] if cohort == 'probe' else manifest[cohort+'_pauses']
        for depth in reversed(manifest['depths']):
            seed = manifest['run'] + ':' + identity(candidate) + ':' + cohort + ':' + str(depth)
            result.append(dict(candidate=candidate, cohort=cohort, depth=depth, pauses=pauses,
                               session_id='r15c-amendment-1-'+hashlib.sha256(seed.encode()).hexdigest()[:20]))
    return result


def payload(manifest, session, index):
    candidate = session['candidate']
    count = len(session['pauses'])
    initial = session['depth'] - (count-1)*manifest['new_tokens'] - manifest['framing_reserve']
    text = 'Synthetic public-domain cache experiment. Read the records and reply OK. Session '+session['session_id']+'.\n'
    text += api.synthetic(initial, session['session_id'])
    for turn in range(index):
        text += api.synthetic(manifest['new_tokens'], session['session_id']+':'+str(turn))
    content = dict(type='text', text=text)
    endpoint = candidate['endpoint']
    if not endpoint.get('supports_implicit_caching', False):
        content['cache_control'] = {'type':'ephemeral'}
    routing = dict(order=[endpoint['tag']], only=[endpoint['tag']], allow_fallbacks=False)
    if candidate['routing_ignore']:
        routing['ignore'] = candidate['routing_ignore']
    return dict(model=candidate['model'], messages=[dict(role='user', content=[content])], provider=routing,
                session_id=session['session_id'], max_tokens=manifest['max_tokens'], stream=True,
                stream_options={'include_usage':True}, temperature=0,
                reasoning={'enabled':False} if candidate['model']=='qwen/qwen3.8-27b' else {'effort':'low'})


def miss_latency(row):
    cached = row.get('cached_tokens')
    if not row.get('accepted'):
        status = 'failed-or-unverified'
    elif cached is None:
        status = 'cache-count-unknown'
    elif cached > 0:
        status = 'cache-hit'
    elif row.get('ttft_seconds') is None:
        status = 'no-observed-token'
    else:
        status = 'confirmed-cache-miss'
    return dict(status=status, actual_prompt_tokens=row.get('prompt_tokens'),
                ttft_seconds=row.get('ttft_seconds') if status=='confirmed-cache-miss' else None,
                ttft_minutes=row['ttft_seconds']/60 if status=='confirmed-cache-miss' else None,
                elapsed_seconds=row.get('elapsed_seconds'))


def metrics(rows, candidate, depths):
    valid = [r for r in rows if r.get('accepted')]
    known = bool(valid) and all(r.get('cached_tokens') is not None for r in valid)
    prompt = sum(r['prompt_tokens'] for r in valid)
    cached = sum(r.get('cached_tokens') or 0 for r in valid)
    rates = {}
    for pause in (8,35,60,300,900,3600):
        subset = [r for r in valid if r['call_index'] > 0 and r['pause_target_seconds']==pause]
        count = sum(r['prompt_tokens'] for r in subset)
        reliable = bool(subset) and all(r.get('cached_tokens') is not None for r in subset)
        rates[str(pause)] = dict(n=len(subset), token_hit_rate=sum(r['cached_tokens'] for r in subset)/count if reliable and count else None,
                                call_hit_rate=sum(r['cached_tokens']>0 for r in subset)/len(subset) if reliable else None,
                                actual_pauses_seconds=[r['pause_actual_seconds'] for r in subset])
    prices = candidate['endpoint']['pricing']
    effective = {}
    for depth in depths:
        subset = [r for r in valid if r['depth']==depth and r['call_index']>0]
        cost = None
        if subset and all(r.get('cached_tokens') is not None for r in subset):
            retention = sum(min(1,r['cached_tokens']/max(1,r['prompt_tokens']-r['new_tokens_estimate'])) for r in subset)/len(subset)
            cache = max(0,depth-1065)*retention
            cost = 1000*((depth-cache)*(float(prices['prompt'])+float(prices.get('input_cache_write',0)))
                         +cache*float(prices['input_cache_read'])+512*float(prices['completion']))
        effective[str(depth)] = cost
    fraction = cached/prompt if known and prompt else None
    uncached = (prompt-cached)/len(valid) if known else None
    return dict(attempts=len(rows), accepted=len(valid), errors=len(rows)-len(valid),
                availability=len(valid)/len(rows) if rows else None, token_hit_rate=fraction,
                mean_uncached_tokens=uncached, delta_hit_fraction_vs_logs=None if fraction is None else fraction-.981,
                delta_uncached_vs_logs=None if uncached is None else uncached-3200,
                pause_levels=rates, effective_usd_per_1000_by_depth=effective,
                effective_usd_per_1000_equal_depth_mix=sum(effective.values())/len(depths) if all(v is not None for v in effective.values()) else None)


class Study:
    def __init__(self, manifest, out):
        self.manifest, self.out = manifest, out
        self.client = None
        self.base, self.total = Decimal(0), Decimal(0)
        self.rows, self.samples, self.skips, self.ranking = [], [], {}, []
        self.status, self.reason = 'queued', 'waiting for prior custody'

    @property
    def secret(self):
        return '' if self.client is None else self.client.secret

    def report(self):
        raise api.Refusal('Legacy amendment report retired; live report: '+str(api.LIVE_REPORT_PATH))

    def call(self, session, index, previous_end=None):
        request=payload(self.manifest,session,index)
        pause=session['pauses'][index]
        if previous_end is not None:
            while time.monotonic()<previous_end+pause:
                if (self.out/'STOP').exists():raise api.Refusal('operator STOP')
                time.sleep(min(30,max(0,previous_end+pause-time.monotonic())))
        if (self.out/'STOP').exists():raise api.Refusal('operator STOP')
        if api.shutil.disk_usage(self.out).free<api.MIN_FREE_BYTES:raise api.Refusal('free disk below 28 GiB')
        candidate=session['candidate']
        sequence=len(self.rows)
        sample_sequence=len(self.samples)
        sample=api.sample_endpoints(self.client,self.out,sample_sequence)
        self.samples.append(sample)
        endpoint=api.live_endpoint(sample,candidate)
        api.validate_live_price(candidate['endpoint'],endpoint)
        text=request['messages'][0]['content'][0]['text']
        maximum=api.max_cost(endpoint,min(len(text.encode())+512,endpoint['context_length']),self.manifest['max_tokens'])
        key=self.client.get('/key')['data']
        live=api.money(key['usage'])
        remaining=None if key.get('limit_remaining') is None else api.money(key['limit_remaining'])
        try:api.guard(live,self.base+self.total,maximum,45,remaining)
        except api.Refusal as error:
            if str(error) in ('budget would exceed cap','live remaining budget insufficient'):raise BudgetSkip(str(error)) from None
            raise
        row=dict(amendment=1,sequence=sequence,uptime_snapshot='uptime-%04d.json'%sample_sequence,provider=identity(candidate),model=candidate['model'],endpoint_tag=endpoint['tag'],
                 cohort=session['cohort'],session_id=session['session_id'],depth=session['depth'],call_index=index,
                 started_at=api.utc(),pause_target_seconds=pause,pause_actual_seconds=None if previous_end is None else time.monotonic()-previous_end,
                 new_tokens_estimate=self.manifest['new_tokens'] if index else None,provider_request=request['provider'],
                 prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),prompt_bytes=len(text.encode()),
                 max_cost_usd=str(maximum),live_spend_before_usd=str(live),accepted=False)
        api.write_json(self.out/('intent-%04d.json'%sequence),row,self.secret)
        row.update(api.stream_call(self.client,request,endpoint,event_path=self.out/('stream-%04d.jsonl'%sequence),timeout_seconds=self.manifest['wall_timeout_seconds']))
        ended=time.monotonic()
        row['ended_at']=api.utc()
        if row.get('cost_usd') is None and isinstance(row.get('usage'),dict) and row['usage'].get('cost') is not None:
            row['cost_usd']=float(api.money(row['usage']['cost']))
        if row.get('cost_usd') is not None:self.total+=api.money(row['cost_usd'])
        self.rows.append(row)
        api.write_json(self.out/('call-%04d.json'%sequence),row,self.secret)
        if row.get('cost_usd') is not None and api.money(row['cost_usd'])>maximum+Decimal('.00000001'):
            row['error']='unexpected cost exceeds reservation'
            api.write_json(self.out/('call-%04d.json'%sequence),row,self.secret)
            raise api.Refusal(row['error'])
        after=self.client.get('/key')['data']
        api.write_json(self.out/('spend-%04d.json'%sequence),dict(at=api.utc(),usage=after['usage'],reported_amendment_cost_usd=str(self.total)),self.secret)
        api.guard(after['usage'],self.base+self.total,Decimal(0),45)
        if api.money(after['usage'])>self.base+self.total+Decimal('.000001'):
            row['error']='unexpected unexplained live spend'
            api.write_json(self.out/('call-%04d.json'%sequence),row,self.secret)
            raise api.Refusal(row['error'])
        if not row.get('error'):
            target=session['depth']-(len(session['pauses'])-1-index)*self.manifest['new_tokens']
            if abs(row['prompt_tokens']-target)>max(512,target*.03):row['error']='measured depth outside tolerance'
            elif row['completion_tokens']>self.manifest['max_tokens']:row['error']='completion cap exceeded'
            else:row['accepted']=True
        api.write_json(self.out/('call-%04d.json'%sequence),row,self.secret)
        print(api.redact(json.dumps({k:row.get(k) for k in ('sequence','provider','cohort','depth','call_index','prompt_tokens','cached_tokens','ttft_seconds','cost_usd','error')}),self.secret),flush=True)
        self.report()
        if row.get('error'):raise api.Refusal('call failed or billing/identity unverifiable; no further spending')
        return ended

    def run_matrix(self, ordered):
        self.status,self.reason='running','prior custody released; amendment measuring'
        self.report()
        for candidate in ordered:
            items=sessions(self.manifest,candidate)
            for depth in reversed(self.manifest['depths']):
                for cohort in ('probe','core'):
                    session=next(s for s in items if s['cohort']==cohort and s['depth']==depth)
                    ended=None
                    for index in range(len(session['pauses'])):
                        try:ended=self.call(session,index,ended)
                        except BudgetSkip as error:
                            self.skips[session['session_id']]=str(error); break
        queue=[]
        for rank,candidate in enumerate(ordered):
            core=[r for r in self.rows if r['provider']==identity(candidate) and r['cohort']=='core']
            for session in sessions(self.manifest,candidate):
                if session['cohort']!='long':continue
                if len(core)!=20:
                    self.skips[session['session_id']]='core coverage incomplete; long substudy skipped'
                else:
                    heapq.heappush(queue,(time.monotonic(),rank,session['depth'],0,session,None))
        while queue:
            due,rank,depth,index,session,ended=heapq.heappop(queue)
            try:ended=self.call(session,index,ended)
            except BudgetSkip as error:
                self.skips[session['session_id']]=str(error); continue
            if index+1<len(session['pauses']):
                heapq.heappush(queue,(ended+session['pauses'][index+1],rank,depth,index+1,session,ended))
        self.status='done'
        self.reason='bounded protocol ended; budget/core skips recorded' if self.skips else 'all planned amendment calls recorded'


def run(args):
    Study({},args.output).report()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text())
    study=Study(manifest,args.output)
    lock=None
    try:
        if hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest()!=manifest['engine_sha256']:
            raise api.Refusal('original HTTP engine differs from preregistered source')
        api.write_json(args.output/'run.json',dict(amendment=1,pid=os.getpid(),started_at=api.utc(),initial_status='queued',
                       manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),engine_sha256=manifest['engine_sha256']))
        deadline=time.monotonic()+args.wait_seconds
        study.report()
        while lock is None:
            if (args.output/'STOP').exists():raise api.Refusal('operator STOP while queued')
            lock=try_custody(args.prior,args.key_path.parent/'cache-run.lock')
            if lock is None:
                if time.monotonic()>=deadline:raise api.Refusal('prior custody wait expired; v1 left untouched')
                print('queued amendment=1; waiting for v1 custody; no amendment requests',flush=True)
                time.sleep(min(30,max(0,deadline-time.monotonic())))
        study.client=api.Client(args.key_path)
        known=prior_costs(args.prior,study.client,args.output)
        live=study.client.get('/key')['data']
        study.base=max(known,api.money(live['usage']))
        previous_path=args.prior/'full-v3/summary.json'
        previous=json.loads(previous_path.read_text()) if previous_path.exists() else None
        ordered,study.ranking=priority_order(manifest,previous)
        api.write_json(args.output/'ranking.json',dict(at=api.utc(),order=study.ranking,prior_summary=str(previous_path),prior_summary_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest() if previous_path.exists() else None),study.secret)
        api.write_json(args.output/'budget-start.json',dict(at=api.utc(),live_key_usage=live['usage'],prior_reported_costs_usd=str(known),starting_floor_usd=str(study.base),combined_cap_usd=45),study.secret)
        print('measuring amendment=1; prior_spend_floor='+str(study.base),flush=True)
        study.run_matrix(ordered)
    except Exception as error:
        study.status,study.reason='failed',api.redact(str(error),study.secret)
        if study.rows and not study.rows[-1]['accepted'] and not study.rows[-1].get('error'):
            study.rows[-1]['error']=study.reason
            api.write_json(args.output/('call-%04d.json'%(len(study.rows)-1)),study.rows[-1],study.secret)
    finally:
        api.write_json(args.output/'terminal.json',dict(amendment=1,status=study.status,reason=study.reason,ended_at=api.utc(),attempts=len(study.rows),reported_amendment_cost_usd=str(study.total),prior_spend_floor_usd=str(study.base)),study.secret)
        study.report()
        text=('To agents1_hybrid_orch, #hybrid topic prototype-go: r15c Amendment 1 '+study.status+'; '+study.reason+
              '; attempts '+str(len(study.rows))+'; amendment reported USD'+str(study.total)+
              '; starting combined spend floor USD'+str(study.base)+'. Report '+str(args.output/'API-CACHE-REPORT.md')+'. Matrix delivery pending network exception.\n')
        (args.output/'notification-terminal.txt').write_text(api.redact(text,study.secret))
        if lock is not None:lock.close()
        print(study.status+' amendment=1 '+api.utc()+' '+study.reason,flush=True)
    return 0 if study.status=='done' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=MANIFEST)
    parser.add_argument('--prior',type=Path,default=PRIOR)
    parser.add_argument('--key-path',type=Path,default=api.KEY_PATH)
    parser.add_argument('--wait-seconds',type=int,default=43200)
    args=parser.parse_args()
    if args.wait_seconds<0:parser.error('wait-seconds must be nonnegative')
    return run(args)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print('failed amendment=1 '+(api.redact(str(error)) if isinstance(error,api.Refusal) else type(error).__name__),flush=True)
        raise SystemExit(1)

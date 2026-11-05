"""Frozen selection, replay timing and session-cluster analysis for PREREG-E2E.md."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

from bench_common import Refusal, require, sha256
from session_runtime import token_ids

PROTOCOL='hybrid-e2e-prereg-v1'
BINS=((0,512),(512,2000),(2000,10000),(10000,None))
PAPER_STATEMENT=('We replay recorded sessions with identical message and tool-output tokens, recorded response lengths '
                 'and recorded waiting intervals in every arm. This measures system time and resource use for a fixed '
                 'workload. It does not measure agent task success or the quality of independently chosen actions.')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def prereg_binding(repo,commit):
    require(len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'full preregistration commit required')
    subprocess.run(['git','-C',str(repo),'verify-commit',commit],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',commit,'HEAD'],check=True)
    path='tools/hybrid-bench/PREREG-E2E.md'
    registered=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+path])
    require(registered==(Path(repo)/path).read_bytes(),'preregistration changed after its signed commit')
    require(PROTOCOL.encode() in registered,'unknown preregistration protocol')
    return dict(protocol=PROTOCOL,commit=commit,document_sha256=hashlib.sha256(registered).hexdigest())


def stamp(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(result.tzinfo is not None,'recorded timestamps require a timezone')
    return result.astimezone(timezone.utc)


def distribution(values):
    values=sorted(values)
    if not values:return dict(n=0,min=None,max=None,mean=None,median=None,p90=None,p99=None)
    def percentile(p):return values[max(0,math.ceil(p*len(values))-1)]
    return dict(n=len(values),min=values[0],max=values[-1],mean=sum(values)/len(values),
                median=percentile(.5),p90=percentile(.9),p99=percentile(.99))


def recorded_session(session,model,vocab,observations=None):
    base=token_ids(session['base_ids'],vocab).copy();turns=[]
    previous=stamp(session['base_response_end_utc'])
    source_indices=[r['source_turn'] for r in session['turns']]
    require(all(type(i) is int for i in source_indices) and all(b==a+1 for a,b in zip(source_indices,source_indices[1:])),
            'recorded source turns must be contiguous')
    for row in session['turns']:
        incoming=token_ids(row['input_ids'],vocab);output=token_ids(row['output_ids'],vocab)
        require(len(output)<=262144,'recorded response length outside native bound')
        started,ended=stamp(row['request_start_utc']),stamp(row['response_end_utc'])
        require(previous<=started<=ended,'negative or overlapping recorded response/tool chronology')
        wait=(started-previous).total_seconds()
        turns.append(dict(copy.deepcopy(row),pause_seconds=wait))
        previous=ended
    identity=digest(dict(base_ids=base,turns=[{k:r[k] for k in ('input_ids','output_ids','request_start_utc','response_end_utc')} for r in turns],base_response_end_utc=session['base_response_end_utc']))
    if observations is not None:observations.update(waits=[r['pause_seconds'] for r in turns],identity=identity)
    start=None;window=[]
    for row in turns:
        if start is None and 100000<=len(base)<200000:start=len(base)
        if start is None:
            base+=row['input_ids']+row['output_ids'];continue
        window.append(row)
        final=start+sum(len(t['input_ids'])+len(t['output_ids']) for t in window)
        if final>=200000:
            require(final+2048<=229376,'recorded selected depth exceeds capacity margin')
            return dict(schema='hybrid-owner-workload-v2',protocol=PROTOCOL,model_sha256=model,
                session_digest=identity,source_session=session['id'],base_ids=base,turns=window,
                start_depth=start,final_depth=final,source_turns=[r['source_turn'] for r in window],
                has_request_messages=all('request_messages' in r for r in window),
                wait_distribution=distribution([r['pause_seconds'] for r in window]))
    return None


def freeze_export(export,export_sha256,model,vocab,prereg):
    require(export.get('schema')=='hybrid-owner-session-export-v2' and export.get('model_sha256')==model,
            'timestamped v2 owner export required; aggregate gaps cannot define tool waits')
    population=export.get('population',{})
    require(population.get('scope')=='all-retained-owner-sessions' and population.get('inventory_complete') is True,
            'complete owner population declaration required before selection')
    require(stamp(population['end_utc'])>=stamp(population['start_utc']),'invalid corpus date range')
    source=export.get('source_log_sha256','')
    require(len(source)==64 and all(c in '0123456789abcdef' for c in source),'source log hash required')
    eligible=[];excluded=[];seen=set();observed_waits=[];population_waits=[]
    for index,session in enumerate(export['sessions']):
        try:
            observation={};row=recorded_session(session,model,vocab,observation)
            if observation['identity'] in seen:excluded.append(dict(index=index,reason='duplicate-content'));continue
            seen.add(observation['identity']);population_waits+=observation['waits']
            if row is None:excluded.append(dict(index=index,reason='no-complete-100k-to-200k-interval'));continue
            eligible.append(row)
            observed_waits += [r['pause_seconds'] for r in row['turns']]
        except (Refusal,ValueError,KeyError,TypeError) as error:
            excluded.append(dict(index=index,reason=type(error).__name__,message=str(error)))
    return dict(schema='hybrid-e2e-frozen-population-v1',prereg=prereg,owner_export_sha256=export_sha256,
        source_log_sha256=export['source_log_sha256'],population=population,eligible=eligible,excluded=excluded,
        eligible_count=len(eligible),observed_count=len(export['sessions']),
        eligible_interval_wait_distribution=distribution(observed_waits),
        population_wait_distribution=distribution(population_waits),missing_or_malformed_sessions=sum(r['reason'] not in ('duplicate-content','no-complete-100k-to-200k-interval') for r in excluded))


def select_set(frozen):
    def rank(row,label):return hashlib.sha256((PROTOCOL+'/'+label+'/261105/'+row['session_digest']).encode()).hexdigest(),row['session_digest']
    eligible=frozen['eligible'];require(len(eligible)>=5,'main series requires five distinct eligible sessions')
    main=sorted(eligible,key=lambda r:rank(r,'main'))[:5];chosen={r['session_digest'] for r in main}
    remaining=[r for r in eligible if r['session_digest'] not in chosen and
        sum(len(t['input_ids'])>=2000 for t in r['turns'])>=12 and sum(len(t['input_ids'])>=10000 for t in r['turns'])>=3]
    favourable=sorted(remaining,key=lambda r:rank(r,'favourable'))[:3]
    sets={}
    for label,rows in (('main',main),('favourable',favourable)):
        sets[label]=dict(label=label if label=='main' else 'favourable case',sessions=rows,
            count=len(rows),status='selected' if len(rows)==(5 if label=='main' else 3) else 'insufficient',
            depth_range=[min((r['start_depth'] for r in rows),default=None),max((r['final_depth'] for r in rows),default=None)],
            wait_distribution=distribution([t['pause_seconds'] for r in rows for t in r['turns']]))
    return dict(schema='hybrid-e2e-session-set-v1',prereg=frozen['prereg'],frozen_population_digest=digest(frozen),
        owner_export_sha256=frozen['owner_export_sha256'],sets=sets,
        api_contract='SESSION-REPLAY-FORMAT.md',paper_statement=PAPER_STATEMENT)


def validate_set(selected):
    require(selected['schema']=='hybrid-e2e-session-set-v1','unknown session set')
    if selected.get('reproducibility_only'):
        require(set(selected['sets'])=={'public'} and selected['sets']['public']['count']>=1,'public-only set cannot masquerade as owner sample')
    else:
        main=selected['sets']['main'];require(main['count']==len(main['sessions'])==5,'main series requires five sessions')
    all_rows=[r for group in selected['sets'].values() for r in group['sessions']]
    ids=[r['session_digest'] for r in all_rows]
    require(len(ids)==len(set(ids)),'session set contains duplicate history')
    for row in all_rows:
        require(row['schema']=='hybrid-owner-workload-v2' and row['protocol']==PROTOCOL,'non-replay workload')
        require(row['final_depth']==row['start_depth']+sum(len(t['input_ids'])+len(t['output_ids']) for t in row['turns']),
                'selected context growth differs from recorded work')
    return all_rows


def schedule(selected,group,link_series,comparison='both'):
    validate_set(selected);require(link_series in ('home','fast'),'declared home/fast series required')
    collection=selected['sets'][group];require(collection['status']=='selected','session collection incomplete')
    pairs=[('B/A','hybrid'),('C/A','hybrid-background')]
    if link_series=='fast':pairs.reverse()
    require(comparison in ('both','B/A','C/A'),'unknown comparison')
    pairs=[p for p in pairs if comparison=='both' or p[0]==comparison]
    return [dict(index=i,comparison=comparison,arm=arm,session_digest=row['session_digest'],group=group,link_series=link_series)
        for i,(comparison,row,arm) in enumerate((c,r,a) for c,h in pairs for r in collection['sessions'] for a in ('local',h))]


def finish_wait(deadline_ns,clock=time.monotonic_ns,sleep=time.sleep,evidence_performed=True):
    now=clock()
    require(not evidence_performed or now<=deadline_ns,'evidence overran recorded tool wait; timing contaminated')
    requested=max(0,(deadline_ns-now)/1e9)
    if requested:sleep(requested)
    ready=clock()
    return dict(remaining_sleep_seconds=requested,deadline_ns=deadline_ns,ready_ns=ready,dispatch_lateness_seconds=max(0,(ready-deadline_ns)/1e9))


def bin_label(tokens):
    for low,high in BINS:
        if low<=tokens and (high is None or tokens<high):return str(low)+'-'+(str(high) if high is not None else 'inf')
    raise ValueError('negative portion size')


def portion_summary(calls):
    result={bin_label(low):dict(count=0,input_tokens=0,output_tokens=0,ttft_seconds=0,timed_seconds=0,
        decode_seconds=0,wait_seconds=0,node_seconds=0,tls_bytes=0) for low,_ in BINS}
    for row in calls:
        if not row['turn']:continue
        n=row['incoming_tokens'];bucket=result[bin_label(n)];bucket['count']+=1
        for key,value in dict(input_tokens=n,output_tokens=row['generated_tokens'],ttft_seconds=row['ttft_seconds'] or 0,
            timed_seconds=row['timing']['seconds'],decode_seconds=(row['meter'].get('decode_ms') or 0)/1000,
            wait_seconds=row['recorded_wait_seconds'],node_seconds=row['node_compute_seconds'],
            tls_bytes=row['tls_sent_bytes']+row['tls_received_bytes']).items():bucket[key]+=value
    return result


def paired_analysis(pairs):
    require(len(pairs)==5 and len({r['session_digest'] for r in pairs})==5,'five session clusters required; no turn pseudoreplication')
    gains=[1-r['hybrid_seconds']/r['local_seconds'] for r in pairs]
    def median(xs):return sorted(xs)[len(xs)//2]
    rng=random.Random(261105);samples=sorted(median(rng.choices(gains,k=len(gains))) for _ in range(10000))
    lower,upper=samples[249],samples[9749]
    qualified=all(r['qualified'] for r in pairs)
    return dict(pairs=pairs,session_clusters=5,median_fractional_reduction=median(gains),bootstrap_95=[lower,upper],
        success=qualified and median(gains)>=.1 and lower>0,qualified=qualified)

"""Admission evidence for the signed staged E2E amendment."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import uuid
from bench_common import require, sha256, private_json
from session_protocol import digest,schedule
from store import prefix_hash

AMENDMENT='hybrid-e2e-amendment-r49c-v1'
SENDER='agents1_hybrid_dev-b'
ROOM='#hybrid'
TOPIC='prototype-go'
RECIPIENT='agents1_hybrid_orch'


def amendment_binding(repo,commit):
    require(isinstance(commit,str) and len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'full amendment commit required')
    subprocess.run(['git','-C',str(repo),'verify-commit',commit],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',commit,'HEAD'],check=True)
    path='tools/hybrid-bench/PREREG-E2E-AMENDMENT.md'
    document=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+path])
    require(document==(Path(repo)/path).read_bytes() and AMENDMENT.encode() in document,'amendment changed after its signed commit')
    return dict(protocol=AMENDMENT,commit=commit,document_sha256=hashlib.sha256(document).hexdigest())


def freeze_receipt(frozen,artifact_sha256,study_id):
    require(study_id and len(study_id)<128,'declared study ID required')
    return dict(schema='hybrid-e2e-published-freeze-v1',study_id=study_id,
        artifact_sha256=artifact_sha256,canonical_digest=digest(frozen),
        inventory={k:v for k,v in frozen.items() if k!='eligible'})


def read_events(event=None):
    argv=['a2a','--as',SENDER,'read',ROOM,'--topic',TOPIC,'--json']
    argv+=['--event',event] if event else ['--last','1000']
    output=subprocess.check_output(argv,text=True)
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def verify_publication(receipt,event_id,events=None):
    events=read_events(event_id) if events is None else events
    rows=[r for r in events if r.get('type')=='message' and r.get('event_id')==event_id]
    require(len(rows)==1,'published freeze event missing or ambiguous')
    row=rows[0]
    require(row.get('sender')==SENDER and row.get('room')==ROOM and row.get('topic')==TOPIC and
            RECIPIENT in row.get('to',[]) and row.get('kind')=='notice','freeze event authority/channel differs')
    require(json.loads(row['body'])==receipt,'published freeze receipt differs from frozen inventory')
    return dict(event_id=event_id,receipt=receipt,receipt_sha256=digest(receipt),published_utc=row['at'])


def checkpoint_plan(workload,arm,budget):
    require(math.isfinite(budget) and budget>=30,'capture budget must be at least 30 seconds')
    n=len(workload['turns']);require(n>0,'empty checkpoint workload')
    turns=[]
    if arm!='local':
        eligible=[i for i in range(1,n) if workload['turns'][i]['pause_seconds']>=budget]
        if eligible:turns.append(min(eligible,key=lambda i:(abs(i-n/2),i)))
        turns.append(n)
    return dict(turns=turns,capture_budget_seconds=budget,targets=[.5,1],
        rule='nearest eligible following wait; earlier tie; final always; final-only if no eligible midpoint')


def descriptive_pairs(pairs):
    gains=[1-r['hybrid_seconds']/r['local_seconds'] for r in pairs]
    ordered=sorted(gains);n=len(gains)
    return dict(pairs=pairs,available_pairs=n,session_clusters=n,descriptive_only=True,
        fractional_reductions=gains,median_fractional_reduction=(ordered[n//2] if n%2 else sum(ordered[n//2-1:n//2+1])/2) if n else None,
        min_fractional_reduction=min(gains) if n else None,max_fractional_reduction=max(gains) if n else None,
        qualified=bool(n) and all(r['qualified'] for r in pairs),success=False)


def verify_qualification(result,method,link_series,link_label):
    require(result['method']==method and result['link_series']==link_series and result['link_label']==link_label,'qualification path/method differs')
    if method=='tc':
        rows=result['results']['rows']
        require({(r['rate_MB_s'],r['rtt_ms']) for r in rows}=={(a,b) for a in (15,20) for b in (50,100,150)} and len(rows)==6,'incomplete six-path shaped qualification')
        require(all(r['payload_MB_s']>=.8*r['rate_MB_s'] and r.get('qdisc') for r in rows),'shaped throughput below 80% or missing qdisc evidence')
    else:
        require(method=='real-path','unknown qualification method')
        trials=result['trials'];require([r['tool'] for r in trials]==['T','R','T'],'qualification must interleave T/R/T')
        require(result['reference_implementation']=='curl-ranges/nginx-static' and result['reference_url']!=result['node_url'], 'independent reference server path required')
        require(all(r['payload_bytes']>=384*1024**2 and r['seconds']>0 and r.get('physical_before') and r.get('physical_after') and r.get('raw') for r in trials),'qualification trial too small or missing raw/physical evidence')
        rates=[r['payload_bytes']/r['seconds'] for r in trials]
        require(all(rates[i]>=.8*rates[1] for i in (0,2)),'transport below 80% of adjacent reference')
        require(bool(result.get('interface')),'qualification interface missing')
    return True


def verify_public_rehearsal(path,expected):
    """Reopen all raw arm evidence; a success flag alone never admits a main run."""
    path=Path(path);receipt=json.loads(path.read_text());root=path.parent
    plan_path=root/receipt['plan']['path'];require(sha256(plan_path)==receipt['plan']['sha256'],'public plan hash differs')
    plan=json.loads(plan_path.read_text())
    require(plan['group']=='public' and plan['link_series']=='fast' and len(plan['attempts'])==4,'complete four-arm public fast rehearsal required')
    for key in ('prereg','amendment','source_head','device','model_sha256','qualification_method'):
        require(plan[key]==expected[key],'public rehearsal identity differs: '+key)
    if expected.get('link_series','fast')=='fast':require(plan['link_label']==expected['link_label'],'public fast path differs')
    require({r['comparison'] for r in plan['attempts']}=={'B/A','C/A'},'public comparison missing')
    selected_path=Path(plan['session_set_path']);require(sha256(selected_path)==plan['session_set_sha256'],'public session set hash differs')
    selected=json.loads(selected_path.read_text())
    require(plan['attempts']==schedule(selected,'public','fast','both'),'public full arm schedule differs')
    workload=selected['sets']['public']['sessions'][0]
    journal=workload['base_ids'].copy();journals={0:journal.copy()}
    expected_calls=[(0,prefix_hash([]),prefix_hash([]),prefix_hash(journal))]
    for turn,step in enumerate(workload['turns'],1):
        journal+=step['input_ids']+step['output_ids'];journals[turn]=journal.copy()
        expected_calls.append((turn,prefix_hash(step['input_ids']),prefix_hash(step['output_ids']),prefix_hash(journal)))
    ap=root/receipt['attempts']['path'];require(sha256(ap)==receipt['attempts']['sha256'],'public attempts hash differs')
    attempts=json.loads(ap.read_text());require(len(attempts)==4,'public loop incomplete')
    maximum=0;previous={}
    for expected_row,row in zip(plan['attempts'],attempts):
        require(row['index']==expected_row['index'] and row.get('exit_code')==0 and row['status']=='measured','public arm failed or missing')
        trace_path=Path(row['trace_path']);require(sha256(trace_path)==row['trace_sha256'],'public trace hash differs')
        trace=json.loads(trace_path.read_text())
        require(trace['status']=='measured' and trace['source_head']==plan['source_head'] and trace['series_plan_sha256']==sha256(plan_path), 'public run failed or unbound')
        require(trace['node_accounting_complete'] and trace['admission']['qualified'] and trace['admission'].get('completed_guard_evidence'), 'public admission/accounting incomplete')
        require(trace['checkpoint_plan']==plan['checkpoints'][expected_row['session_digest']][expected_row['arm']], 'public checkpoint plan differs')
        admission=trace['admission'];guard=admission['completed_guard_evidence']
        require(admission['cooldown_seconds']>=300 and sha256(admission['raw_memory_evidence'])==guard['memory_sha256'] and
            sha256(admission['raw_priority_evidence'])==guard['priority_sha256'],'public supervisor raw evidence differs')
        require(json.loads(Path(admission['raw_memory_evidence']).read_text()).get('accepted') is True,'public supervisor did not accept')
        from session_measurement import validate_quality
        for quality in trace['quality']:
            result=Path(quality['result']);input_path=result.parent/'input.json'
            require(sha256(result)==quality['result_sha256'] and sha256(input_path)==quality['input_sha256'],'public checker evidence changed')
            source=json.loads(input_path.read_text())
            extension=[0] if not workload['turns'][quality['turn']-1]['output_ids'] else []
            require(source['tokens']==journals[quality['turn']]+extension and source['model_sha256']==plan['model_sha256'],'public checkpoint journal/model differs')
            require(validate_quality(json.loads(result.read_text()),source),'public native quality gate failed')
        require([q['turn'] for q in trace['quality']]==trace['checkpoint_plan']['turns'] and all(q['gate'] for q in trace['quality']), 'public quality missing or failed')
        require(trace['prereg']==plan['prereg'] and trace['amendment']==plan['amendment'] and trace['device']==plan['device'],'public trace registration/device differs')
        require(trace['session_digest']==expected_row['session_digest'] and trace['arms']==[expected_row['arm']], 'public work/arm differs')
        identity=[(r['turn'],r.get('input_ids_sha256'),r.get('recorded_output_sha256'),r['journal_hash']) for r in trace['calls']]
        require(identity==expected_calls,'public replay loop incomplete or differs from frozen work')
        comparison=expected_row['comparison']
        if expected_row['arm']=='local':previous[comparison]=identity
        else:require(identity==previous.get(comparison),'public paired journal differs')
        for capture in trace.get('evidence',[]):
            if capture.get('capture'):maximum=max(maximum,(capture['end_ns']-capture['start_ns'])/1e9)
    for block in ('B','C'):
        for side in ('pre','post'):
            item=receipt['qualification'][block+'-'+side];qpath=root/item['path']
            require(sha256(qpath)==item['sha256'],'public qualification hash differs')
            q=json.loads(qpath.read_text());require(q['block']==block and q['side']==side and q['series_id']==plan['series_id'] and q['source_head']==plan['source_head'] and q['prereg']==plan['prereg'] and q['amendment']==plan['amendment'],'public qualification block/registration differs')
            require(q['node_url']==trace['link_probe']['node_url'] and q['ca_sha256']==trace['transport']['ca_sha256'],'public qualification node/certificate differs')
            verify_qualification(q,plan['qualification_method'],'fast',plan['link_label'])
    return dict(path=str(path.resolve()),sha256=sha256(path),capture_budget_seconds=max(30,math.ceil(2*maximum)))


def validate_run_plan(plan,selected,args,source_head,model):
    require(plan['source_head']==source_head and plan['session_set_sha256']==args.session_set_sha256 and
        plan['series_id']==args.series_id and plan['group']==args.group and plan['device']==args.device and
        plan['link_label']==args.link_label and plan['model_sha256']==model,'series plan identity differs from measured call')
    rows=plan['attempts'];require(0<=args.ordinal<len(rows),'attempt ordinal outside plan')
    row=rows[args.ordinal]
    require(row['index']==args.ordinal and row['arm']==args.arm and row['session_digest']==args.session_digest and
        row['comparison']==args.comparison,'attempt changed planned arm/order/work')
    budget=30
    if args.group=='public':
        require(plan['link_series']=='fast' and len(rows)==4 and {r['comparison'] for r in rows}=={'B/A','C/A'},'public full loop required')
    else:
        require(plan.get('public_rehearsal'),'verified public rehearsal required before main')
        proof=plan['public_rehearsal'];require(sha256(proof['path'])==proof['sha256'],'public rehearsal receipt changed')
        expected={k:plan[k] for k in ('prereg','amendment','source_head','device','model_sha256','qualification_method','link_label','link_series')}
        verified=verify_public_rehearsal(proof['path'],expected);budget=verified['capture_budget_seconds']
    workload=next(r for r in selected['sets'][args.group]['sessions'] if r['session_digest']==args.session_digest)
    expected=checkpoint_plan(workload,args.arm,budget)
    require(plan['checkpoints'][args.session_digest][args.arm]==expected,'checkpoint plan differs from frozen waits')
    return expected


def verify_series_transport(receipt_path,plan_path):
    receipt_path,plan_path=Path(receipt_path),Path(plan_path)
    receipt=json.loads(receipt_path.read_text());plan=json.loads(plan_path.read_text())
    require(receipt['plan']['sha256']==sha256(plan_path),'series receipt plan differs')
    ap=receipt_path.parent/receipt['attempts']['path']
    require(sha256(ap)==receipt['attempts']['sha256'],'series receipt attempts differ')
    for block in {r['comparison'][0] for r in plan['attempts']}:
        before=None
        for side in ('pre','post'):
            item=receipt['qualification'][block+'-'+side];path=receipt_path.parent/item['path']
            require(sha256(path)==item['sha256'],'series qualification hash differs')
            q=json.loads(path.read_text())
            require(q['series_id']==plan['series_id'] and q['block']==block and q['side']==side and q['source_head']==plan['source_head'] and q['prereg']==plan['prereg'] and q['amendment']==plan['amendment'],'series qualification identity differs')
            if before is not None:require((q['node_url'],q['ca_sha256'])==before,'post qualification node/certificate differs')
            before=(q['node_url'],q['ca_sha256'])
            verify_qualification(q,plan['qualification_method'],plan['link_series'],plan['link_label'])
    return True


def replace_private_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        private_json(temporary,value);temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_capture(start_ns,end_ns,deadline_ns,budget_seconds):
    require(0<=(end_ns-start_ns)/1e9<=budget_seconds,'slot save exceeded declared capture budget; series disqualified')
    require(end_ns<=deadline_ns,'evidence overran recorded tool wait; timing contaminated')

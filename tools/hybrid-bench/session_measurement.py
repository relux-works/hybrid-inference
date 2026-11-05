#!/usr/bin/env python3
"""R49 production calls on a fixed owner workload, with separate post-turn evidence."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from bench_common import (Refusal,private_json,read_private,require,sha256,start_process,wait_health,validate_identity)
import growing_session as g
from growing_fixture import validate_fixture
from session_runtime import SessionProxy,select_workload,run_turn,SELECTION,catchup_decision
from proxy import Node,HTTPError
from policy import Policy
from store import prefix_hash
from parallel_transport import ParallelTransport
from session_protocol import prereg_binding,validate_set,digest,finish_wait,portion_summary
from session_admission import KitAdmission
from session_study import amendment_binding,validate_run_plan,verify_qualification,replace_private_json,validate_capture

ARMS=('local','hybrid','hybrid-background')


def arm_names(dtype):
    require(dtype in ('bf16','f32'),'unsupported live-wire profile')
    return ARMS if dtype=='bf16' else ('local','hybrid-f32','hybrid-background-f32')


def wire_policy(policy,layout,geometry,dtype):
    require(dtype in ('f32','bf16'),'unsupported wire policy profile')
    require(policy.get('kv_bytes')==geometry['kv_bytes'] and
            policy.get('state_bytes')==geometry['state_bytes']+geometry.get('window_bytes',0),
            'baseline policy bytes differ from native geometry')
    return g.bf16_policy(layout,policy) if dtype=='bf16' else dict(policy)


def probe(url,token,ca_file,streams=8,repetitions=3):
    client=ParallelTransport(url,ca_file,streams,300)
    rows=[]
    for _ in range(repetitions):
        before=client.stats();health_start=time.monotonic_ns()
        with client.open('GET','/v1/health',headers={'Authorization':'Bearer '+token}) as response:
            response.read(8*1024**2)
        health_seconds=(time.monotonic_ns()-health_start)/1e9
        start=time.monotonic_ns()
        with client.open('POST','/v1/bench/parallel-link',b'{}',{'Authorization':'Bearer '+token}) as response:
            manifest=json.loads(response.read(1024**2))
        body=client.fetch(manifest,token)
        end=time.monotonic_ns();after=client.stats()
        require(body==b'x'*len(body),'link probe payload mismatch')
        rows.append(dict(health_seconds=health_seconds,start_ns=start,end_ns=end,bytes=len(body),seconds=(end-start)/1e9,
            bytes_per_second=len(body)/((end-start)/1e9),**{k:after[k]-before[k] for k in before}))
    rates=sorted(r['bytes_per_second'] for r in rows)
    return dict(schema='hybrid-https-probe-v1',streams=streams,samples=rows,
        bytes_per_second=rates[len(rates)//2],wire_boundary='encrypted TLS records at TCP payload; TCP/IP headers and retransmissions excluded')


def native_stream(decode,request):
    first_token=first_text=None
    partial=[];final=None
    with decode.open('POST','/completion',dict(request,stream=True)) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]':continue
            row=json.loads(line[6:]);now=time.monotonic_ns()
            if (row.get('tokens') or row.get('content')) and first_token is None:first_token=now
            if row.get('content') and first_text is None:first_text=now
            if row.get('stop'):final=row
            else:partial.extend(row.get('tokens',[]))
    require(final is not None,'native stream did not complete')
    tokens=final.get('tokens') or partial
    return tokens,final.get('timings',{}),first_token,first_text


def execute_turn(app,client,sid,prompt,count,turn,seed,force_base=False,recorded_output=None):
    started=time.monotonic_ns()
    session=app.store.get(sid)
    before=app.decode.request('GET','/slots/0/info')['n']
    session['_turn']={};session['_commit']=False
    mode='local' if app.arm=='local' else 'offload' if force_base else 'hybrid'
    profile=dict(client,benchmark_mode=mode)
    app.origin.value='critical';app.origin.turn=turn;app.origin.call_id=app.arm+'-'+str(turn)
    body=dict(prompt=prompt,n_predict=count,temperature=0,seed=seed,ignore_eos=True,return_tokens=True,cache_prompt=True)
    if recorded_output is not None and count:
        require(len(recorded_output)==count,'recorded replay count differs')
        body.update(replay_mode='recorded-token-replay-v1',replay_tokens=recorded_output)
    try:
        session,_,request=app.prepare_session(profile,body,session,prompt)
        cold_unused_sample=0
        if count:
            generated,timings,first,first_text=native_stream(app.decode,request)
        else:
            native=app.decode.request('POST','/completion',request)
            cold_unused_sample=len(native.get('tokens',[]))
            require(cold_unused_sample in (0,1),'unexpected cold-prefill sample count')
            generated=[];timings=native.get('timings',{});first=first_text=None
        require(len(generated)==count,'native generation length differs from owner workload')
        require(recorded_output is None or generated==recorded_output,'native output differs from recorded replay')
        complete=prompt+generated
        reconciliation=time.monotonic_ns()
        actual=app.decode.request('GET','/slots/0/info')
        require(actual['tokens']==complete[:actual['n']] and len(complete)-actual['n'] in (0,1),
                'completed native state does not match generated journal')
        require(actual['prefix_hash']==prefix_hash(actual['tokens']),'native journal hash mismatch')
        session['tokens']=actual['tokens'];session['_commit']=True
        session['journal']=dict(production_turn=turn)
        app.store.commit_evaluated(session)
        meter=session['_bench']
        meter.update(decode_ms=timings.get('predicted_ms'),cold_unused_sample_tokens=cold_unused_sample,local_prompt_ms=timings.get('prompt_ms'),reconciliation_ms=(time.monotonic_ns()-reconciliation)/1e6)
        require(meter['fallback_reason'] not in ('invalid_delta','http_409'),'production protocol mismatch')
        app.schedule(sid,meter['path'])
        return dict(turn=turn,path=meter['path'],prompt_tokens=len(prompt),depth_before=before,depth_after=len(complete),
            generated_tokens=count,generated_ids=generated,journal_hash=prefix_hash(complete),request_start_ns=started,
            first_token_ns=first,first_text_ns=first_text,
            ttft_seconds=None if first is None else (first-started)/1e9,
            first_text_seconds=None if first_text is None else (first_text-started)/1e9,
            decode_tokens_per_second=(count-1)*1000/timings['predicted_ms'] if count>1 and timings.get('predicted_ms') else None,
            local_tail_seconds=(timings.get('prompt_ms') or 0)/1000 if meter['path']=='offload' else 0,
            reconciliation_seconds=meter['reconciliation_ms']/1000,
            background_join_seconds=meter.get('background_join_seconds',0),meter=meter,complete=complete,
            state_dtype=None if app.arm=='local' else app.state_dtype,lossy_import=meter['lossy_import'])
    finally:
        app.end_turn(sid)


def snapshot(app,root,arm,turn,complete,defer_hash=False):
    info=app.decode.request('GET','/slots/0/info')
    require(info['tokens'] in (complete,complete[:-1]),'quality snapshot does not match recorded journal')
    extension=[0] if info['tokens']==complete else []
    probe_tokens=complete+extension
    name=f'{arm}-{turn}.bin'
    receipt=app.decode.request('POST','/slots/0?action=save',dict(filename=name))
    require(receipt['n_saved']==len(probe_tokens)-1,'quality snapshot token count mismatch')
    path=root/'slots'/name
    binding=dict(arm=arm,turn=turn,path=str(path),sha256=None if defer_hash else sha256(path),tokens=probe_tokens,model_sha256=app.model,recorded_journal_hash=prefix_hash(complete),probe_prefix_extension=extension)
    directory=root/'checkpoint-bindings';directory.mkdir(exist_ok=True,mode=0o700)
    private_json(directory/(arm+'-'+str(turn)+'.json'),binding)
    return binding


def quality_fixture(client,saved,dtype):
    tokens=saved['tokens']
    schedule=[dict(n0=n,n1=min(n+256,len(tokens)-1),logits=False) for n in range(0,len(tokens)-1,256)]
    schedule.append(dict(n0=len(tokens)-1,n1=len(tokens),logits=True))
    return dict(tokens=tokens,k=64,max_ctx=client['context'],model_sha256=client['model_sha256'],backend=client['backend'],
        state_dtype=dtype,acceptance_rule='session_quality_v1',snapshots=[saved['path']],snapshot_sha256=[saved['sha256']],
        replay=dict(A=schedule),pairing=dict(mode=g.COMMON_JOURNAL,journal_hash=prefix_hash(tokens),
            reference_batches=dict(L=dict(batch=256,ubatch=256),P=dict(batch=64,ubatch=32))))


def validate_quality(row,source):
    require(row.get('status_rule')=='session_quality_v1' and row.get('pairing')==source['pairing'] and row.get('k')==64,
            'missing declared production quality pairing')
    require(row['continuations']['H']['teacher_tokens']==row['continuations']['L']['teacher_tokens'] and
            len(row['continuations']['L']['teacher_tokens'])==64,'quality teacher-forced inputs differ')
    agreed=row['top1_agreed']
    require(type(agreed) is int and 0<=agreed<=64 and row['quality_gate']==(agreed>=63),'quality gate contradicts 63/64 rule')
    if source['state_dtype']=='bf16':
        require(row['exact'] is False and row['exit_code']==5 and row['lossy_import'] is True,
                'bf16 checkpoint cannot claim exactness')
    first=next((i for i,(a,b) in enumerate(zip(row['continuations']['L']['free_tokens'],row['continuations']['H']['free_tokens'])) if a!=b),-1)
    require(row['free_running_first_divergence']==first,'quality free divergence contradicts continuation IDs')
    g.validate_raw_positions(row,len(source['tokens']))
    return bool(row['quality_gate'])


def check_quality(client,root,saved,dtype):
    out=root/'quality'/f"{saved['arm']}-{saved['turn']}";out.mkdir(parents=True,mode=0o700)
    source=quality_fixture(client,saved,dtype)
    fixture,result=out/'input.json',out/'result.json'
    private_json(fixture,source)
    command=[Path(client['build_dir'])/'bin/llama-hybrid-bench-check',client['model_path'],fixture,saved['path'],result,
             0 if client['backend']=='cpu' else 999]
    process=start_process(command,out,'checker')
    try:code=process.wait()
    finally:g.stop_process(process)
    require(result.is_file(),'quality checker produced no result')
    row=json.loads(result.read_text())
    require(code==row['exit_code'] and code in (0,3,5),'quality checker failed')
    accepted=validate_quality(row,source)
    Path(saved['path']).unlink()
    return dict(arm=saved['arm'],turn=saved['turn'],gate=accepted,top1_agreed=row['top1_agreed'],
        mean_kl_nats=row['mean_kl_nats'],free_running_first_divergence=row['free_running_first_divergence'],
        checker_exit_code=code,state_dtype=dtype,exact=row['exact'],result=str(result),result_sha256=sha256(result),input_sha256=sha256(fixture))


def summarize(events,calls):
    node=[r for r in events if r['kind']=='node']
    totals={}
    for origin in ('critical','background'):
        selected=[r for r in node if r['origin']==origin]
        totals[origin]=dict(node_compute_seconds=sum(r['node_compute_seconds'] or 0 for r in selected),node_prefill_seconds=sum(r['node_prefill_seconds'] or 0 for r in selected),
            catchup_seconds=sum((r['end_ns']-r['start_ns'])/1e9 for r in selected if r['catchup']),
            catchup_tokens=sum(r['tokens'] for r in selected if r['catchup']),
            tls_sent_bytes=sum(r['tls_sent_bytes'] for r in selected),tls_received_bytes=sum(r['tls_received_bytes'] for r in selected))
    decode_ms=sum(r['meter'].get('decode_ms') or 0 for r in calls if r['turn'])
    envelope=sum(r['meter']['wire_bytes'] for r in calls if r['turn'])
    transfer=sum(r['meter']['transfer_ms'] for r in calls if r['turn'])/1000
    return dict(envelope_bytes=envelope,achieved_payload_MB_s=envelope/transfer/1e6 if transfer else None,
        decode_tokens_per_second=sum(max(r['generated_tokens']-1,0) for r in calls if r['turn'])*1000/decode_ms if decode_ms else None,
        node_compute_unknown_requests=sum(r['status']!='ok' and r['operation'].endswith('/extend') for r in node),
        timed_turn_seconds=sum(r['timing']['seconds'] for r in calls if r['turn']),
        ttft_seconds=sum(r['ttft_seconds'] or 0 for r in calls if r['turn']),node=totals,
        offloaded_turns=sum(r['path']=='offload' for r in calls if r['turn']),
        wire_boundary='Actual encrypted TLS record bytes; excludes TCP/IP headers, retransmissions and decoder loopback',
        node_time_boundary='Native synchronized prefill evaluation wall time, not CUDA kernel-active time')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--client-state',type=Path,required=True);p.add_argument('--owner-export',type=Path)
    p.add_argument('--owner-export-sha256')
    p.add_argument('--session-set',type=Path);p.add_argument('--session-set-sha256');p.add_argument('--session-digest')
    p.add_argument('--prereg-commit');p.add_argument('--amendment-commit')
    p.add_argument('--series-plan',type=Path);p.add_argument('--series-plan-sha256')
    p.add_argument('--arm',choices=ARMS);p.add_argument('--series-probe',type=Path)
    p.add_argument('--refresh-series-probe',action='store_true')
    p.add_argument('--ordinal',type=int)
    p.add_argument('--series-id');p.add_argument('--comparison',choices=('B/A','C/A'));p.add_argument('--group',choices=('main','favourable','public'))
    p.add_argument('--kit-root',type=Path);p.add_argument('--kit-ref')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ca-file',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--import-fit',type=Path,required=True)
    p.add_argument('--state-dtype',choices=('bf16','f32'),default='bf16');p.add_argument('--streams',type=int,default=8)
    p.add_argument('--device',choices=('m3-max-128gb','m5-max-64gb','cpu-qualification'),required=True)
    p.add_argument('--link-label',required=True)
    return p


def run(args):
    state,root=args.client_state.resolve(),args.output.resolve()
    run_arm=dict(zip(ARMS,arm_names(args.state_dtype)))[args.arm]
    require(not root.exists() and root.parent.is_dir(),'new output directory required')
    config=read_private(state/'proxy-config.json');client=read_private(state/'client-info.json')
    node=json.loads((state/'node-info.json').read_text())
    validate_identity(client['model_sha256'],client,node['model_sha256'],node)
    source_head=subprocess.check_output(['git','-C',Path(__file__).resolve().parents[2],'rev-parse','HEAD'],text=True).strip()
    require(client.get('commit')==node.get('commit')==source_head,'node/client build refs differ from reviewed source')
    require(client['model_layout']['architecture']=='qwen35','r49 recurrent transport profile requires Qwen35')
    for name in ('llama-server','llama-hybrid-bench-check'):
        require(sha256(Path(client['build_dir'])/'bin'/name)==client['binaries'][name],'client build identity changed')
    require(sha256(client['model_path'])==client['model_sha256'],'client GGUF identity changed')
    require(args.session_set is not None and args.prereg_commit and args.arm and args.kit_root and args.kit_ref and args.series_probe,
            'preregistered session set, single arm, series probe and kit admission required')
    prereg=prereg_binding(Path(__file__).resolve().parents[2],args.prereg_commit)
    require(args.comparison in ('B/A','C/A') and args.group in ('main','favourable','public') and args.ordinal is not None,'series comparison/group/ordinal required')
    require(args.session_set.stat().st_size<=512*1024**2,'session set exceeds input budget')
    require(sha256(args.session_set)==args.session_set_sha256,'session set differs from frozen delivery hash')
    selected=json.loads(args.session_set.read_text());rows=validate_set(selected)
    amendment=amendment_binding(Path(__file__).resolve().parents[2],args.amendment_commit)
    require(selected['prereg']==prereg and selected.get('amendment')==amendment,'session set registration differs')
    require(args.series_plan and sha256(args.series_plan)==args.series_plan_sha256,'verified series plan required')
    plan=json.loads(args.series_plan.read_text())
    require(plan['prereg']==prereg and plan['amendment']==amendment,'series registration differs')
    checkpoints=validate_run_plan(plan,selected,args,source_head,client['model_sha256'])
    qualification=json.loads((args.series_plan.parent/('qualification-'+args.comparison[0]+'-pre.json')).read_text())
    require(qualification['series_id']==args.series_id and qualification['source_head']==source_head and
        qualification['block']==args.comparison[0] and qualification['side']=='pre' and
        qualification['node_url']==config['prefill_url'] and qualification['ca_sha256']==sha256(args.ca_file),'pre-block qualification identity differs')
    verify_qualification(qualification,plan['qualification_method'],plan['link_series'],args.link_label)
    collection=selected['sets'].get(args.group,{})
    require(collection.get('status')=='selected','selected series group incomplete')
    workload=next((r for r in collection['sessions'] if r['session_digest']==args.session_digest),None)
    require(workload is not None,'session missing from selected group')
    require(workload['model_sha256']==client['model_sha256'],'session tokenizer/model identity differs')
    require(workload['final_depth']+2048<=min(client['context'],node['context'],262144),'owner workload exceeds context with margin')
    fixture=json.loads(args.fixture.read_text())
    rates=validate_fixture(fixture,client['model_sha256'],'qwen35')
    require(fixture.get('target_device')==args.device,'rate fixture belongs to another Mac/device')
    if args.device!='cpu-qualification':
        require(sys.platform=='darwin' and client['backend']=='metal' and node['backend']=='cuda','Mac arm requires Metal client and CUDA node')
        priority=subprocess.check_output(['ps','-p',str(os.getpid()),'-o','pri=,nice='],text=True).split()
        require([int(x) for x in priority]==[31,0],'PRI31 supervisor required')
        chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True)
        require(('M3 Max' if args.device.startswith('m3') else 'M5 Max') in chip,'client SoC differs from device fixture')
    fit=json.loads(args.import_fit.read_text())
    sys.path.insert(0,str(Path(__file__).parent/'packets/r46'))
    from mac_timing import require_fit_identity
    require_fit_identity(fit,client)
    require(fit.get('state_dtype')==args.state_dtype and fit.get('lossy_import')==(args.state_dtype=='bf16'),
            'import fit dtype/lossy profile differs from live wire')
    policy=json.loads(args.policy.read_text())
    require(config.get('state_dtype','f32')=='f32','input policy/config must be the labelled f32 baseline')
    policy=wire_policy(policy,client['model_layout'],config['delta_geometry'],args.state_dtype)
    policy.update(local_rates=rates,import_fixed=fit['fixed_s'],import_per_token=fit['seconds_token'])
    Policy(**policy)
    geometry=config['delta_geometry']
    size=geometry['state_bytes']+(geometry['kv_bytes']+32)*client['context']+1024**2
    disk_floor=40*1024**3 if client['backend']=='metal' else 28.5*1024**3
    require(shutil.disk_usage(root.parent).free>disk_floor+(len(checkpoints['turns'])+3)*size+2*1024**3,'disk budget for checkpoint states and rollback reserve refused')
    root.mkdir(mode=0o700);(root/'slots').mkdir(mode=0o700)
    token_file=root/'decode-token';token_file.write_text(config['decode_token']);token_file.chmod(0o600)
    trace=dict(schema='hybrid-production-session-v2',status='running',arms=[run_arm],state_dtype=args.state_dtype,
        started_utc=datetime.now(timezone.utc).isoformat(),link_label=args.link_label,device=args.device,
        transport=dict(protocol='https-ranges-v1',streams=args.streams,ca_sha256=sha256(args.ca_file)),
        amendment=amendment,series_plan_sha256=args.series_plan_sha256,checkpoint_plan=checkpoints,
        selection_rule='preregistered-random-owner-sessions-v1',prereg=prereg,session_set_sha256=sha256(args.session_set),
        session_digest=args.session_digest,group=args.group,series_id=args.series_id,comparison=args.comparison,ordinal=args.ordinal,
        admission=None,workload=workload,
        calls=[],evidence=[],quality=[],events=[],sessions={},source_head=source_head)
    def flush():
        path=root/'trace.json.tmp';path.write_text(json.dumps(trace,indent=2)+'\n');path.chmod(0o600);path.replace(root/'trace.json')
    flush();decoder=None;app=None;saved=[];kit=None
    raw_start=time.monotonic_ns()
    try:
        remote=Node(config['prefill_url'],next(iter(config['clients'].values()))['prefill_token'],3600,
                    parallel=dict(ca_file=str(args.ca_file.resolve()),streams=args.streams))
        health=remote.request('GET','/v1/health')
        require(health['model_sha256']==client['model_sha256'] and args.state_dtype in health.get('state_dtypes',['f32']),
                'node model or live-wire capability mismatch')
        trace['startup_node_tls_bytes']=remote.last_tls_bytes.copy()
        for arm in [run_arm]:
            events=g.Events(None)
            run_config=dict(config,benchmark=True,slots=1,slot_path=str(root/'slots'),snapshot_disk_reserve_bytes=int(disk_floor),
                decode_url='http://127.0.0.1:'+str(g.free_port()),decode_timeout=3600,prefill_timeout=3600,
                state_dtype=args.state_dtype,policy=policy,control_seconds=0,parallel_transport=dict(ca_file=str(args.ca_file.resolve()),streams=args.streams))
            load_start=time.monotonic_ns()
            kit=KitAdmission(args.kit_root,args.kit_ref,root)
            decoder_args=g.decoder_command(client,run_config,int(run_config['decode_url'].rsplit(':',1)[1]),root/'slots',token_file)
            decoder=start_process(kit.command(decoder_args),root,'decoder-'+arm)
            wait_health(run_config['decode_url'],'/health',decoder,timeout=3600,token=config['decode_token'])
            trace['sessions'][arm]=dict(decoder_load_seconds=(time.monotonic_ns()-load_start)/1e9)
            trace['admission']=kit.cooldown(decoder)
            if args.refresh_series_probe:
                measured=probe(config['prefill_url'],remote.token,str(args.ca_file.resolve()),args.streams)
                measured.update(node_url=config['prefill_url'],ca_sha256=sha256(args.ca_file),series_id=args.series_id,
                    measured_utc=datetime.now(timezone.utc).isoformat())
                require(not args.series_probe.exists(),'series-start probe must not overwrite prior evidence')
                private_json(args.series_probe,measured)
            trace['link_probe']=json.loads(args.series_probe.read_text())
            require(trace['link_probe'].get('node_url')==config['prefill_url'] and
                    trace['link_probe'].get('ca_sha256')==sha256(args.ca_file) and trace['link_probe'].get('series_id')==args.series_id,
                    'series probe belongs to another node/certificate/series')
            control=sorted(r['health_seconds'] for r in trace['link_probe']['samples'])[1]
            policy['fixed']=max(policy.get('fixed',0),2*control)
            policy['link_rate']=trace['link_probe']['bytes_per_second'];trace['policy']=policy
            trace['control_seconds']=control;run_config['control_seconds']=control
            app=SessionProxy(run_config,str(root/(arm+'.sqlite')),events,background=arm in ('hybrid-background','hybrid-background-f32'));app.arm=arm
            client_profile=next(iter(config['clients'].values()))
            sid=uuid.uuid4().hex
            initial=app.store.select(client_profile['id'],[],sid);initial['slot']=0;app.store.save(initial);app.store.set_slot(sid,0)
            journal=workload['base_ids'].copy()
            rows=[];session_start=time.monotonic_ns()
            checkpoint_turns=set(checkpoints['turns'])
            previous_end=None;pending_checkpoint=None
            sequence=[dict(input_ids=[],output_ids=[],pause_seconds=0)]+workload['turns']
            sequence=[dict(row,input_hash=prefix_hash(row['input_ids']),output_hash=prefix_hash(row['output_ids'])) for row in sequence]
            for turn,step in enumerate(sequence):
                if turn:
                    wait_start=previous_end;deadline=wait_start+int(step['pause_seconds']*1e9)
                    evidence_start=time.monotonic_ns()
                    if pending_checkpoint is not None:
                        captured=snapshot(app,root,arm,turn-1,pending_checkpoint,defer_hash=True);saved.append(captured)
                    capture_end=time.monotonic_ns()
                    if pending_checkpoint is not None:
                        validate_capture(evidence_start,capture_end,deadline,checkpoints['capture_budget_seconds'])
                    wait_receipt=finish_wait(deadline,evidence_performed=pending_checkpoint is not None)
                    trace['evidence'].append(dict(arm=arm,turn=turn-1,start_ns=evidence_start,end_ns=capture_end,
                        capture=pending_checkpoint is not None,within_recorded_wait=True))
                    trace['sessions'][arm].setdefault('pauses',[]).append(dict(turn=turn,recorded_seconds=step['pause_seconds'],
                        elapsed_seconds=(wait_receipt['ready_ns']-wait_start)/1e9,**wait_receipt))
                prompt=journal+step['input_ids']
                start_ns=time.monotonic_ns()
                result=execute_turn(app,client_profile,sid,prompt,len(step['output_ids']),turn,261105+turn,
                    force_base=False,recorded_output=step['output_ids'])
                previous_end=time.monotonic_ns()
                journal=result.pop('complete');generated=result.pop('generated_ids')
                result.update(timing=dict(start_ns=start_ns,end_ns=previous_end,seconds=(previous_end-start_ns)/1e9),
                    arm=arm,source_turn=step.get('source_turn'),incoming_tokens=len(step['input_ids']),
                    input_ids_sha256=step['input_hash'],recorded_output_sha256=step['output_hash'],
                    recorded_wait_seconds=step['pause_seconds'])
                trace['calls'].append(result);rows.append(result)
                pending_checkpoint=journal.copy() if turn in checkpoint_turns else None
            active_end=time.monotonic_ns()
            app.close_background(discard_pending=True)
            trace['sessions'][arm]['final_background_join_seconds']=(time.monotonic_ns()-active_end)/1e9
            if pending_checkpoint is not None:
                final_capture_start=time.monotonic_ns()
                saved.append(snapshot(app,root,arm,len(workload['turns']),pending_checkpoint,defer_hash=True))
                trace['evidence'].append(dict(arm=arm,turn=len(workload['turns']),start_ns=final_capture_start,end_ns=time.monotonic_ns(),capture=True,after_session=True))
            for captured in saved:
                captured['sha256']=sha256(captured['path'])
                replace_private_json(root/'checkpoint-bindings'/(captured['arm']+'-'+str(captured['turn'])+'.json'),captured)
            current=app.store.get(sid)
            if current['remote_id']:app.remote(client_profile).request('DELETE','/v1/sessions/'+current['remote_id'])
            complete_events=events.copy()
            for row in rows:
                measured=[r for r in complete_events if r.get('kind')=='node' and r.get('turn')==row['turn']]
                row['node_gpu_eval_seconds']=sum(r['node_prefill_seconds'] or 0 for r in measured)
                row['node_compute_seconds']=sum(r['node_compute_seconds'] or 0 for r in measured)
                row['catchup_on_critical_seconds']=sum((r['end_ns']-r['start_ns'])/1e9 for r in measured if r['catchup'] and r['origin']=='critical')
                row['catchup_off_critical_seconds']=sum((r['end_ns']-r['start_ns'])/1e9 for r in measured if r['catchup'] and r['origin']=='background')
                row['tls_sent_bytes']=sum(r['tls_sent_bytes'] for r in measured)
                row['tls_received_bytes']=sum(r['tls_received_bytes'] for r in measured)
                row['envelope_bytes']=row['meter']['wire_bytes']
                transfer=row['meter']['transfer_ms']/1000
                row['achieved_payload_MB_s']=row['envelope_bytes']/transfer/1e6 if transfer else None
            trace['events']+= [dict(arm=arm,**r) for r in complete_events]
            trace['sessions'][arm].update(raw_window_seconds=(active_end-session_start)/1e9,
                cold_base_seconds=rows[0]['timing']['seconds'],shutdown_seconds=(time.monotonic_ns()-active_end)/1e9,
                summary=summarize(events.copy(),rows),background_errors=app.worker.errors if app.worker else [])
            trace['sessions'][arm]['portion_bins']=portion_summary(rows)
            trace['sessions'][arm]['warm_system_seconds']=(active_end-rows[0]['timing']['end_ns'])/1e9
            app.store.close();app=None
            trace['admission']['completed_guard_evidence']=kit.stop(decoder)
            decoder=None;kit=None;flush()
        # One checker context at a time, after every measured decoder has stopped.
        for item in saved:
            start=time.monotonic_ns()
            row=check_quality(client,root,item,'f32' if item['arm']=='local' else args.state_dtype)
            row['evidence_seconds']=(time.monotonic_ns()-start)/1e9
            trace['quality'].append(row);flush()
        trace['node_accounting_complete']=not any(s['summary']['node_compute_unknown_requests'] for s in trace['sessions'].values())
        trace['background_exercised']=any(r.get('kind')=='node' and r.get('origin')=='background' and r.get('catchup') and r.get('tokens',0)>0 for r in trace['events'])
        background_errors=any(v.get('background_errors') for v in trace['sessions'].values())
        trace['status']='incomplete_node_accounting' if not trace['node_accounting_complete'] else 'background_failed' if background_errors else 'quality_failed' if not all(r['gate'] for r in trace['quality']) else 'measured'
        trace['background_qualification']='exercised' if trace['background_exercised'] else 'unused; no background benefit established'
    except BaseException as error:
        trace.update(status='aborted',error_type=type(error).__name__)
        raise
    finally:
        if app is not None:
            try:
                app.close_background(discard_pending=True)
            except Exception as error:
                trace.setdefault('cleanup_errors',[]).append(type(error).__name__)
            try:
                profile=next(iter(config['clients'].values()))
                for current in app.store.pending():
                    if current['remote_id']:
                        app.remote(profile).request('DELETE','/v1/sessions/'+current['remote_id'])
            except Exception as error:
                trace.setdefault('cleanup_errors',[]).append(type(error).__name__)
            finally:
                trace['events']+=[dict(arm=app.arm,**r) for r in app.events.copy()]
                app.store.close()
        if decoder is not None:
            try:
                if kit is not None:kit.abort(decoder)
                else:g.stop_process(decoder)
            except Exception as error:trace.setdefault('cleanup_errors',[]).append(type(error).__name__)
        trace['cold_start_semantics']='empty context; weights may be OS-cached after identity checks/calibration; no purge'
        trace['raw_run_wall_seconds']=(time.monotonic_ns()-raw_start)/1e9
        trace['capture_timing']='snapshots inside recorded waits; overrun refuses; hashes and checkers after timed session'
        flush()
    return 0 if trace['status']=='measured' else 3


if __name__=='__main__':
    try:raise SystemExit(run(parser().parse_args()))
    except (Refusal,ValueError,OSError,KeyError) as error:
        print('refused: '+str(error) if isinstance(error,Refusal) else 'refused: '+type(error).__name__,file=sys.stderr)
        raise SystemExit(2)

#!/usr/bin/env python3
"""R49 production calls on a fixed owner workload, with separate post-turn evidence."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
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


def execute_turn(app,client,sid,prompt,count,turn,seed,force_base=False):
    started=time.monotonic_ns()
    session=app.store.get(sid)
    before=app.decode.request('GET','/slots/0/info')['n']
    session['_turn']={};session['_commit']=False
    mode='local' if app.arm=='local' else 'offload' if force_base else 'hybrid'
    profile=dict(client,benchmark_mode=mode)
    app.origin.value='critical';app.origin.turn=turn;app.origin.call_id=app.arm+'-'+str(turn)
    body=dict(prompt=prompt,n_predict=count,temperature=0,seed=seed,ignore_eos=True,return_tokens=True,cache_prompt=True)
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


def snapshot(app,root,arm,turn,complete):
    info=app.decode.request('GET','/slots/0/info')
    require(info['tokens']==complete[:-1],'quality snapshot requires one unreconciled sampled token')
    name=f'{arm}-{turn}.bin'
    receipt=app.decode.request('POST','/slots/0?action=save',dict(filename=name))
    require(receipt['n_saved']==len(complete)-1,'quality snapshot token count mismatch')
    path=root/'slots'/name
    binding=dict(arm=arm,turn=turn,path=str(path),sha256=sha256(path),tokens=complete.copy(),model_sha256=app.model)
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
        checker_exit_code=code,state_dtype=dtype,exact=row['exact'],result=str(result),input_sha256=sha256(fixture))


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
    p.add_argument('--client-state',type=Path,required=True);p.add_argument('--owner-export',type=Path,required=True)
    p.add_argument('--owner-export-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ca-file',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--import-fit',type=Path,required=True)
    p.add_argument('--state-dtype',choices=('bf16','f32'),default='bf16');p.add_argument('--streams',type=int,default=8)
    p.add_argument('--device',choices=('m3-max-128gb','m5-max-64gb','cpu-qualification'),required=True)
    p.add_argument('--link-label',required=True)
    return p


def run(args):
    state,root=args.client_state.resolve(),args.output.resolve()
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
    require(args.owner_export.stat().st_size<=256*1024**2,'owner export exceeds input budget')
    require(sha256(args.owner_export)==args.owner_export_sha256,'owner export differs from trusted delivery hash')
    workload=select_workload(json.loads(args.owner_export.read_text()),client['model_sha256'],client['model_layout']['vocab_size'])
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
    require(shutil.disk_usage(root.parent).free>disk_floor+15*size+2*1024**3,'disk budget for 12 checkpoint states and rollback reserve refused')
    root.mkdir(mode=0o700);(root/'slots').mkdir(mode=0o700)
    token_file=root/'decode-token';token_file.write_text(config['decode_token']);token_file.chmod(0o600)
    trace=dict(schema='hybrid-production-session-v1',status='running',arms=list(arm_names(args.state_dtype)),state_dtype=args.state_dtype,
        started_utc=datetime.now(timezone.utc).isoformat(),link_label=args.link_label,device=args.device,
        transport=dict(protocol='https-ranges-v1',streams=args.streams,ca_sha256=sha256(args.ca_file)),
        selection_rule=SELECTION,owner_export_sha256=sha256(args.owner_export),workload=workload,
        calls=[],evidence=[],quality=[],events=[],sessions={},source_head=source_head)
    def flush():
        path=root/'trace.json.tmp';path.write_text(json.dumps(trace,indent=2)+'\n');path.chmod(0o600);path.replace(root/'trace.json')
    flush();decoder=None;app=None;saved=[]
    raw_start=time.monotonic_ns()
    try:
        remote=Node(config['prefill_url'],next(iter(config['clients'].values()))['prefill_token'],3600,
                    parallel=dict(ca_file=str(args.ca_file.resolve()),streams=args.streams))
        health=remote.request('GET','/v1/health')
        require(health['model_sha256']==client['model_sha256'] and args.state_dtype in health.get('state_dtypes',['f32']),
                'node model or live-wire capability mismatch')
        trace['link_probe']=probe(config['prefill_url'],remote.token,str(args.ca_file.resolve()),args.streams)
        control=sorted(r['health_seconds'] for r in trace['link_probe']['samples'])[1]
        policy['fixed']=max(policy.get('fixed',0),2*control)
        policy['link_rate']=trace['link_probe']['bytes_per_second'];trace['policy']=policy
        trace['control_seconds']=control
        depth=workload['start_depth'];remote_depth=depth-1;previous_offload=False
        decisions=[]
        for row in workload['turns']:
            decision=catchup_decision(Policy(**policy),len(row['input_ids']),depth,max(0,depth-remote_depth),previous_offload,control_seconds=control)
            decisions.append(decision);previous_offload=decision['offload']
            if decision['offload']:remote_depth=depth+len(row['input_ids'])-1
            depth+=len(row['input_ids'])+len(row['output_ids'])
        require(any(r['offload'] for r in decisions),'measured-link preflight predicts an empty hybrid arm')
        trace['route_preflight']=decisions
        for arm in arm_names(args.state_dtype):
            events=g.Events(None)
            run_config=dict(config,benchmark=True,slots=1,slot_path=str(root/'slots'),snapshot_disk_reserve_bytes=int(disk_floor),
                decode_url='http://127.0.0.1:'+str(g.free_port()),decode_timeout=3600,prefill_timeout=3600,
                state_dtype=args.state_dtype,policy=policy,control_seconds=control,parallel_transport=dict(ca_file=str(args.ca_file.resolve()),streams=args.streams))
            load_start=time.monotonic_ns()
            decoder=start_process(g.decoder_command(client,run_config,int(run_config['decode_url'].rsplit(':',1)[1]),root/'slots',token_file),root,'decoder-'+arm)
            wait_health(run_config['decode_url'],'/health',decoder,timeout=3600,token=config['decode_token'])
            trace['sessions'][arm]=dict(decoder_load_seconds=(time.monotonic_ns()-load_start)/1e9)
            app=SessionProxy(run_config,str(root/(arm+'.sqlite')),events,background=arm in ('hybrid-background','hybrid-background-f32'));app.arm=arm
            client_profile=next(iter(config['clients'].values()))
            sid=uuid.uuid4().hex
            initial=app.store.select(client_profile['id'],[],sid);initial['slot']=0;app.store.save(initial);app.store.set_slot(sid,0)
            journal=workload['base_ids'].copy()
            rows=[];session_start=time.monotonic_ns()
            for turn,step in enumerate([dict(input_ids=[],output_ids=[],pause_seconds=0)]+workload['turns']):
                if turn:
                    pause_start=time.monotonic_ns();time.sleep(step['pause_seconds'])
                    trace['sessions'][arm].setdefault('pauses',[]).append(dict(turn=turn,seconds=(time.monotonic_ns()-pause_start)/1e9))
                prompt=journal+step['input_ids']
                last={}
                def publish(row,measured):
                    last.update(row);last['timing']=measured
                    public={k:v for k,v in last.items() if k not in ('complete','generated_ids')}
                    public.update(arm=arm,source_turn=step.get('source_turn'),input_ids_sha256=prefix_hash(step['input_ids']))
                    trace['calls'].append(public);rows.append(public)
                def evidence():
                    require(shutil.disk_usage(root).free>disk_floor+size,'checkpoint disk floor reached')
                    journals=root/'journals';journals.mkdir(exist_ok=True,mode=0o700)
                    private_json(journals/(arm+'-'+str(turn)+'.json'),dict(turn=turn,generated_ids=last['generated_ids'],journal_hash=last['journal_hash']))
                    if turn in (10,20,30,40):
                        captured=snapshot(app,root,arm,turn,last['complete']);saved.append(captured)
                        return dict(snapshot_sha256=captured['sha256'],snapshot_bytes=Path(captured['path']).stat().st_size)
                    return dict(snapshot=False)
                _,_,receipt,phase=run_turn(lambda:execute_turn(app,client_profile,sid,prompt,len(step['output_ids']),turn,261105+turn,force_base=turn==0),evidence,publish)
                trace['evidence'].append(dict(arm=arm,turn=turn,receipt=receipt,**phase))
                journal=last['complete'];flush()
            active_end=time.monotonic_ns()
            app.close_background()
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
            require(arm=='local' or trace['sessions'][arm]['summary']['offloaded_turns']>0,'live hybrid arm did not offload')
            app.store.close();app=None;g.stop_process(decoder);decoder=None;flush()
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
                app.close_background()
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
        if decoder is not None:g.stop_process(decoder)
        trace['cold_start_semantics']='empty context; weights may be OS-cached after identity checks/calibration; no purge'
        trace['raw_run_wall_seconds']=(time.monotonic_ns()-raw_start)/1e9
        trace['capture_timing']='post-turn snapshots and deferred checker; raw wall includes evidence; timed turns do not'
        flush()
    return 0 if trace['status']=='measured' else 3


if __name__=='__main__':
    try:raise SystemExit(run(parser().parse_args()))
    except (Refusal,ValueError,OSError,KeyError) as error:
        print('refused: '+str(error) if isinstance(error,Refusal) else 'refused: '+type(error).__name__,file=sys.stderr)
        raise SystemExit(2)

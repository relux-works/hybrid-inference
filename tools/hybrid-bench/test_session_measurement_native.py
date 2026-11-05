#!/usr/bin/env python3
"""Opt-in real CPU r49 mechanics on existing small Qwen weights; no downloads."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

from bench_common import private_json,require,sha256,model_identity
import growing_session as g
from session_runtime import SessionProxy,CatchupPolicy
from session_measurement import execute_turn,snapshot,check_quality,validate_quality
from policy import Policy
from store import prefix_hash
from parallel_transport import Gateway
from native_fixture import native_server
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hybrid'))
from test_limits import Node as NativeNode


@contextmanager
def gateway_server(server):
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield 'https://127.0.0.1:'+str(server.server_port)
    finally:server.shutdown();server.server_close();thread.join()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',required=True,type=Path);p.add_argument('--model',required=True,type=Path)
    p.add_argument('--artifacts',required=True,type=Path)
    args=p.parse_args();build=args.build_dir.resolve();model=args.model.resolve();root=args.artifacts.resolve()
    require(not root.exists(),'new native artifact directory required');root.mkdir(parents=True,mode=0o700)
    client=dict(model_path=str(model),build_dir=str(build),backend='cpu',context=4096,**model_identity(model))
    digest=sha256(model);client['model_sha256']=digest
    cert,key=root/'cert.pem',root/'key.pem'
    with (root/'openssl.log').open('wb') as log:
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=localhost',
            '-addext','subjectAltName=IP:127.0.0.1,DNS:localhost','-keyout',key,'-out',cert],stdout=log,stderr=log,check=True)
    native=SimpleNamespace(binary=str(build/'bin/llama-hybrid-prefill'),model=str(model),artifacts=str(root))
    report=dict(schema='r49-cpu-native-mechanics-v1',model_sha256=digest,arms=[],cpu_only=True,
                limitation='Small Qwen, four controlled turns; not the owner workload, WAN throughput or target-model quality')
    with NativeNode(native,'node',flags=['--bench'],context='4096') as node:
        tls=Gateway(('127.0.0.1',0),node.base,node.clients.values(),cert,key,probe_bytes=1024**2)
        with gateway_server(tls) as url:
            for arm in ('local','hybrid','hybrid-background'):
                work=root/arm
                with native_server(build/'bin/llama-server',model,work,checkpoints=0) as (decoder,slots):
                    config=dict(model_sha256=digest,prefill_url=url,decode_url=decoder.url,decode_token='',
                        slot_path=str(slots),slots=1,state_dtype='bf16',prefill_timeout=120,
                        parallel_transport=dict(ca_file=str(cert),streams=8),benchmark=True,
                        delta_geometry=dict(kv_bytes=6528,state_bytes=20201472),
                        clients={'client':dict(id='a',prefill_token=node.clients['a'])},policy=dict(always=False))
                    events=g.Events(None);app=SessionProxy(config,str(work/'proxy.sqlite'),events,background=arm=='hybrid-background');app.arm=arm
                    owner=config['clients']['client'];sid='native-'+arm
                    app.store.select('a',[],sid);app.store.set_slot(sid,0)
                    journal=decoder.request('POST','/tokenize',dict(content='The clear sky above the meadow. '*12,add_special=True))['tokens']
                    rows=[];race={}
                    if app.worker:
                        entered,released,acknowledged=threading.Event(),threading.Event(),threading.Event()
                        original_ack=app.store.acknowledge_remote
                        original_commit=app.store.commit_evaluated
                        def held_ack(*args):
                            if not entered.is_set():
                                entered.set()
                                require(released.wait(10),'native ACK plant was not released')
                            result=original_ack(*args);acknowledged.set();return result
                        def racing_commit(session):
                            if session['journal']['production_turn']==2:
                                require(entered.wait(10),'native background ACK plant did not apply')
                                race['stale_remote_n']=session['remote_n'];released.set()
                                require(acknowledged.wait(10),'native background ACK did not commit')
                                race['worker_remote_n']=app.store.get(sid)['remote_n']
                                require(race['worker_remote_n']>race['stale_remote_n'],'native ACK plant was not stale')
                            original_commit(session)
                            if session['journal']['production_turn']==2:
                                race['preserved_remote_n']=app.store.get(sid)['remote_n']
                                require(race['preserved_remote_n']>=race['worker_remote_n'],'foreground erased native background ACK')
                        app.store.acknowledge_remote=held_ack
                        app.store.commit_evaluated=racing_commit
                    try:
                        for turn in range(5):
                            # Deliberate mechanics routes exercise both branches; production never uses this per-turn override.
                            app.policy=CatchupPolicy(Policy(always=turn in (0,3,4),local_rates=((0,1e9),)),app.origin,0)
                            prompt=journal+([100,101,102,103]*12 if turn else [])
                            result=execute_turn(app,owner,sid,prompt,8 if turn else 0,turn,1+turn,force_base=turn==0)
                            journal=result.pop('complete');rows.append(result)
                            require(app.store.get(sid)['tokens']==journal[:-1] if turn else app.store.get(sid)['tokens']==journal,
                                    'native durable journal mismatch')
                        app.close_background()
                        require(not (app.worker and app.worker.errors),'native background worker failed')
                        captured=snapshot(app,work,arm,4,journal) if arm=='hybrid-background' else None
                        protocol=events.copy()
                        require(arm!='local' or not protocol,'local arm made node calls')
                        require(arm=='local' or all(r['lossy_import'] for r in rows if r['path']=='offload'),'bf16 import missing lossy label')
                        require(arm=='local' or any(r['path']=='offload' for r in rows if r['turn']),'no real turn import')
                        if arm=='hybrid-background':
                            require(any(r.get('kind')=='background_job' and r['status']=='ok' for r in protocol),'no completed native background job')
                        report['arms'].append(dict(arm=arm,calls=rows,events=protocol,commit_race=race))
                        current=app.store.get(sid)
                        if current['remote_id']:app.remote(owner).request('DELETE','/v1/sessions/'+current['remote_id'])
                    finally:
                        if app.worker:released.set()
                        app.close_background();app.store.close()
                if captured:
                    report['quality']=check_quality(client,root,captured,'bf16')
                    require(report['quality']['gate'],'small-model production quality gate failed')
                    source=json.loads((Path(report['quality']['result']).parent/'input.json').read_text())
                    row=json.loads(Path(report['quality']['result']).read_text())
                    broken=json.loads(json.dumps(row));broken['continuations']['H']['teacher_tokens'][0]^=1
                    try:validate_quality(broken,source)
                    except Exception as error:
                        require('teacher-forced inputs differ' in str(error),'unrelated teacher-pairing refusal')
                    else:raise AssertionError('teacher-pairing mutant survived')
    private_json(root/'summary.json',report)
    print(json.dumps(dict(arms=len(report['arms']),quality=report['quality'],summary=str(root/'summary.json'))))


if __name__=='__main__':main()

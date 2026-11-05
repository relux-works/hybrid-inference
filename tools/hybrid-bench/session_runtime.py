"""R49 committed-journal scheduling and owner-log workload admission."""
from contextlib import contextmanager
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
import urllib.parse
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hybrid-proxy'))
from proxy import Proxy, Node, HTTPError
from store import prefix_hash
from bench_common import require


SELECTION = 'earliest-contiguous-40;start90k-110k;end190k-210k;12inputs>=2k;3inputs>=10k;no-route-filter'


def token_ids(value, vocab):
    require(isinstance(value,list) and all(type(t) is int and 0 <= t < vocab for t in value),'invalid owner token IDs')
    return value


def select_workload(export, model_sha256, vocab):
    require(export.get('schema')=='hybrid-owner-session-export-v1' and export.get('model_sha256')==model_sha256,
            'owner export schema/model mismatch; quantile sampling is not an owner trace')
    source=export.get('source_log_sha256','')
    require(len(source)==64 and all(c in '0123456789abcdef' for c in source),'owner source-log hash required')
    sessions=export.get('sessions')
    require(isinstance(sessions,list) and sessions,'owner ordered session export required')
    def ordered(session):
        stamp=datetime.fromisoformat(session['started_utc'].replace('Z','+00:00'))
        require(stamp.tzinfo is not None,'owner session timestamps require a timezone')
        return stamp.astimezone(timezone.utc),session['id']
    for session in sorted(sessions,key=ordered):
        base=token_ids(session['base_ids'],vocab).copy()
        turns=session['turns']
        require(isinstance(turns,list) and all(type(r.get('source_turn')) is int for r in turns),'owner source indices required')
        require(all(b['source_turn']==a['source_turn']+1 for a,b in zip(turns,turns[1:])),'owner source turns must be contiguous')
        for row in turns:
            token_ids(row['input_ids'],vocab);token_ids(row['output_ids'],vocab)
            require(1<=len(row['output_ids'])<=262144,'invalid observed answer length for native context')
            pause=row['pause_seconds']
            require(type(pause) in (int,float) and math.isfinite(pause) and 0<=pause<=86400,'invalid observed tool pause')
        for start in range(max(0,len(turns)-39)):
            window=turns[start:start+40]
            final=len(base)+sum(len(r['input_ids'])+len(r['output_ids']) for r in window)
            inputs=[len(r['input_ids']) for r in window]
            if (90000<=len(base)<=110000 and 190000<=final<=210000
                    and sum(n>=2000 for n in inputs)>=12 and sum(n>=10000 for n in inputs)>=3):
                return dict(schema='hybrid-owner-workload-v1',selection_rule=SELECTION,model_sha256=model_sha256,
                    source_log_sha256=source,source_session=session['id'],started_utc=session['started_utc'],
                    source_turns=[r['source_turn'] for r in window],base_ids=base.copy(),turns=copy.deepcopy(window),
                    start_depth=len(base),final_depth=final)
            base+=turns[start]['input_ids']+turns[start]['output_ids']
    raise ValueError('no qualifying contiguous owner 40-turn window; samples are never synthesized or enlarged')


class BackgroundCatchup:
    def __init__(self, apply):
        self.apply=apply
        self.condition=threading.Condition()
        self.pending=set()
        self.stopping=False
        self.errors=[]
        self.thread=threading.Thread(target=self.run,name='hybrid-committed-catchup',daemon=True)
        self.thread.start()

    def submit(self, sid):
        with self.condition:
            if self.stopping:raise ValueError('background scheduler closed')
            if sid not in self.pending and len(self.pending)>=16:raise ValueError('background queue budget exceeded')
            self.pending.add(sid);self.condition.notify()

    def run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda:self.pending or self.stopping)
                if not self.pending:return
                sid=self.pending.pop()
            try:self.apply(sid)
            except Exception as error:
                with self.condition:self.errors.append(type(error).__name__)

    def close(self):
        with self.condition:self.stopping=True;self.condition.notify_all()
        self.thread.join()


def catchup_decision(policy,delta,depth,pending,previous=False,link_rate=None,control_seconds=0):
    require(type(pending) is int and pending>=0,'invalid committed catch-up length')
    result=policy.choose(delta,depth,previous,link_rate)
    extra=pending/policy.server_rate+(3*control_seconds if pending else 0)
    result['offload_seconds']+=extra
    threshold=1-policy.margin+(policy.hysteresis if previous else -policy.hysteresis)
    result['offload']=delta>0 and (policy.always or result['offload_seconds']<result['local_seconds']*threshold)
    result.update(pending_catchup_tokens=pending,pending_catchup_predicted_seconds=extra)
    return result


class CatchupPolicy:
    def __init__(self,policy,current,control_seconds):
        self.policy,self.current,self.control_seconds=policy,current,control_seconds

    def choose(self,delta,depth,previous=False,link_rate=None,server_factor=1):
        require(server_factor==1,'production catch-up policy requires the declared server rate')
        result=catchup_decision(self.policy,delta,depth,self.current.pending,previous,link_rate,self.control_seconds)
        self.current.decision=dict(inputs=dict(delta=delta,depth=depth,pending_catchup=self.current.pending),result=result.copy())
        return result


class SessionProxy(Proxy):
    def __init__(self,config,database,events,background=False):
        super().__init__(config,database)
        self.events=events
        self.node_locks={}
        self.origin=threading.local()
        self.policy=CatchupPolicy(self.policy,self.origin,config.get('control_seconds',0))
        self.worker=BackgroundCatchup(self.catchup_committed) if background else None
        self.fatal_background=[]
        self.remote_inflight=set()

    def node_lock(self,sid):
        with self.lock:return self.node_locks.setdefault(sid,threading.Lock())

    def remote(self,client):
        app=self
        class ObservedNode(Node):
            def request(self,method,path,data=None,headers=None,binary=False):
                started=time.monotonic_ns();status='failed'
                try:
                    result=super().request(method,path,data,headers,binary)
                    if path.endswith('/extend'):
                        require(self.last_node_compute_ms is not None and self.last_prefill_ms is not None,'node compute clocks missing')
                    status='ok';return result
                finally:
                    app.events.add(kind='node',id=uuid.uuid4().hex,parent_id=getattr(app.origin,'call_id',''),operation=path,origin=getattr(app.origin,'value','critical'),method=method,
                        start_ns=started,end_ns=time.monotonic_ns(),status=status,turn=getattr(app.origin,'turn',None),
                        catchup=isinstance(data,dict) and data.get('return_delta') is False,
                        tokens=len(data.get('tokens',[])) if isinstance(data,dict) else 0,
                        n0=data.get('n0') if isinstance(data,dict) else None,
                        n1=data['n0']+len(data['tokens']) if isinstance(data,dict) and 'n0' in data and 'tokens' in data else None,
                        payload_bytes=self.last_bytes,upload_body_bytes=self.last_upload_bytes,
                        node_prefill_seconds=float(self.last_prefill_ms or 0)/1000 if status=='ok' else None,
                        node_compute_seconds=float(self.last_node_compute_ms or 0)/1000 if status=='ok' else None,
                        **self.last_tls_bytes)
        return ObservedNode(self.prefill,client['prefill_token'],self.timeout,parallel=self.parallel_transport)

    def prepare_session(self,client,body,session,tokens):
        self.origin.pending=max(0,len(session['tokens'])-session['remote_n'])
        result=super().prepare_session(client,body,session,tokens)
        result[0]['_bench']['decision']=self.origin.decision
        return result

    @contextmanager
    def offload_lock(self,session):
        started=time.monotonic_ns()
        with self.node_lock(session['id']):
            session['_bench']['background_join_seconds']=(time.monotonic_ns()-started)/1e9
            current=self.store.get(session['id'])
            session.update(remote_id=current['remote_id'],remote_n=current['remote_n'])
            with self.lock:self.remote_inflight.add(session['id'])
            try:
                yield
            except BaseException:
                self.end_turn(session['id'])
                raise

    def catchup_committed(self,sid):
        self.origin.value='background'
        started=time.monotonic_ns();status='failed'
        try:
            with self.node_lock(sid):
                with self.lock:
                    if sid in self.remote_inflight:
                        status='skipped_inflight';return
                session=self.store.get(sid)
                journal=session['tokens'].copy()
                self.origin.turn=(session.get('journal') or {}).get('production_turn')
                client=next(c for c in self.clients.values() if c['id']==session['client'])
                node=self.remote(client)
                remote_id=session['remote_id']
                if remote_id is None:
                    remote_id=node.request('POST','/v1/sessions',dict(model_sha256=self.node_model))['session_id']
                    self.store.acknowledge_remote(sid,client['id'],remote_id,0,prefix_hash([]))
                path='/v1/sessions/'+urllib.parse.quote(remote_id,safe='')
                info=node.request('GET',path)
                n=info['n']
                if info.get('model_sha256')!=self.node_model:raise ValueError('background node model mismatch')
                self.validate_background_prefix(info,journal)
                while n<len(journal):
                    end=min(n+self.extend_chunk_tokens,len(journal))
                    node.request('POST',path+'/extend',dict(n0=n,prefix_hash_n0=prefix_hash(journal[:n]),
                        tokens=journal[n:end],return_delta=False),headers={'X-Hybrid-Bench':'1'})
                    deadline=time.monotonic()+self.timeout
                    while True:
                        info=node.request('GET',path)
                        self.validate_background_prefix(info,journal)
                        if info['n']==end:break
                        if not n<=info['n']<end:raise ValueError('background acknowledgement skipped its committed target')
                        if time.monotonic()>=deadline:raise OSError('background acknowledgement timeout')
                        time.sleep(.01)
                    self.store.acknowledge_remote(sid,client['id'],remote_id,end,info['prefix_hash'])
                    n=end
                status='ok'
        except (ValueError,KeyError):
            self.fatal_background.append('committed_journal_contract');raise
        finally:
            self.events.add(kind='background_job',session=sid,start_ns=started,end_ns=time.monotonic_ns(),status=status)
            self.origin.value='critical'

    @staticmethod
    def validate_background_prefix(info,journal):
        n=info['n']
        if type(n) is not int or not 0<=n<=len(journal) or info['prefix_hash']!=prefix_hash(journal[:n]):
            raise ValueError('background node ahead of or divergent from committed journal')

    def end_turn(self,sid):
        with self.lock:self.remote_inflight.discard(sid)

    def schedule(self,sid,path):
        if self.worker is not None and path=='local':self.worker.submit(sid)

    def close_background(self):
        if self.worker is not None:self.worker.close()
        require(not self.fatal_background,'background committed-journal contract failed')


def timed_call(operation,clock=time.monotonic_ns):
    start=clock();result=operation();end=clock()
    return result,dict(start_ns=start,end_ns=end,seconds=(end-start)/1e9)


def post_turn_evidence(operation,clock=time.monotonic_ns):
    # The caller invokes this only after timed_call returns; no evidence future is joined inside the timer.
    start=clock();result=operation();end=clock()
    return result,dict(start_ns=start,end_ns=end,seconds=(end-start)/1e9,excluded_from_turn=True)


def run_turn(operation,evidence,publish,clock=time.monotonic_ns):
    result,measured=timed_call(operation,clock)
    publish(result,measured)
    captured,phase=post_turn_evidence(evidence,clock)
    return result,measured,captured,phase

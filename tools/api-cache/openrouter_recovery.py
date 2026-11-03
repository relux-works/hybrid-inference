"""Charge, stream and analysis contracts for the r15d restart."""
import json
from pathlib import Path
import time
from decimal import Decimal
from urllib.request import Request

import openrouter_cache as api

MIN_FREE_BYTES = 28 * 1024**3
ROUNDING = Decimal('.00000001')
DEPTH_DEADLINES = [(32768,180),(131072,300),(250000,600),(524288,1200),(1000000,2400)]


def deadline_for_depth(depth):
    for bound,seconds in DEPTH_DEADLINES:
        if 0 < depth <= bound:
            return seconds
    raise api.Refusal('depth outside registered deadline table')


def atomic_json(path, data, secret=''):
    text=api.redact(json.dumps(data,indent=2,default=str),secret)+'\n'
    api.assert_no_secret(text,secret)
    temporary=path.with_name(path.name+'.new')
    temporary.write_text(text)
    temporary.replace(path)


def resolve_delta(samples, expected, maximum):
    expected,maximum=api.money(expected),api.money(maximum)
    values=[api.money(s['usage_usd']) for s in samples]
    if len(values)<3 or len(set(values))!=1:
        raise api.Refusal('unresolvable charge: cumulative usage is missing or unstable')
    delta=values[-1]-expected
    if delta < -ROUNDING or delta > maximum:
        raise api.Refusal('unresolvable charge: cumulative delta outside reservation')
    return dict(method='stable cumulative usage delta',expected_usd=str(expected),maximum_usd=str(maximum),
                observed_delta_usd=str(delta),resolved_observed_charge_usd=str(max(Decimal(0),delta)),
                retained_reservation_usd=str(maximum),rounding_tolerance_usd=str(ROUNDING),samples=samples,resolved=True)


def usage_samples(client, count=3, pause=1):
    result=[]
    for index in range(count):
        data=client.get('/key')['data']
        result.append(dict(at=api.utc(),usage_usd=str(data['usage'])))
        if index+1<count:time.sleep(pause)
    return result


def settled_samples(client,expected,maximum,evidence_path,timeout_seconds=600,poll_seconds=15):
    expected,maximum=api.money(expected),api.money(maximum)
    end=time.monotonic()+timeout_seconds
    history=[]
    while time.monotonic()<end:
        client.get_timeout=min(30,max(.1,end-time.monotonic()))
        try:
            item=dict(at=api.utc(),usage_usd=str(client.get('/key')['data']['usage']))
            history.append(item)
        except Exception as error:
            history.append(dict(at=api.utc(),error=api.redact(str(error),client.secret)))
        atomic_json(evidence_path,dict(expected_usd=str(expected),maximum_usd=str(maximum),samples=history),client.secret)
        if 'usage_usd' in history[-1] and api.money(history[-1]['usage_usd'])>expected+maximum:
            client.get_timeout=30
            raise api.Refusal('unresolvable charge: cumulative delta exceeds reservation')
        last=history[-3:]
        if len(last)==3 and all('usage_usd' in s for s in last):
            try:
                resolve_delta(last,expected,maximum)
                client.get_timeout=30
                return last
            except api.Refusal:pass
        remaining=end-time.monotonic()
        if remaining>0:time.sleep(min(poll_seconds,remaining))
    client.get_timeout=30
    raise api.Refusal('unresolvable charge: cumulative counter did not settle within recovery deadline')


class Budget:
    def __init__(self,path,known,committed,reported='0',receipts=None):
        self.path=path
        self.known=api.money(known)
        self.committed=api.money(committed)
        self.reported=api.money(reported)
        self.receipts=receipts or {}

    @classmethod
    def load(cls,path):
        data=json.loads(path.read_text())
        return cls(path,data['known_observed_usd'],data['committed_upper_usd'],data['restart_reported_usd'],data.get('receipts'))

    def save(self):
        atomic_json(self.path,dict(at=api.utc(),known_observed_usd=str(self.known),
                    committed_upper_usd=str(self.committed),restart_reported_usd=str(self.reported),global_cap_usd=45,receipts=self.receipts))

    def reserve(self,live,maximum,cap=45,remaining=None):
        api.guard(live,self.committed,maximum,min(45,cap),remaining)

    def reported_call(self,cost,maximum,call_id):
        if call_id in self.receipts:
            if not self.receipts[call_id].get('within_maximum',True):raise api.Refusal('reported cost exceeds reservation')
            return self.receipts[call_id]
        cost=api.money(cost)
        self.known+=cost; self.committed+=cost; self.reported+=cost
        receipt=dict(method='API reported cost',cost_usd=str(cost),maximum_usd=str(maximum),within_maximum=cost<=api.money(maximum)+ROUNDING)
        self.receipts[call_id]=receipt
        self.save()
        if not receipt['within_maximum']:
            raise api.Refusal('reported cost exceeds reservation')
        return receipt

    def recovered_call(self,samples,maximum,call_id):
        if call_id in self.receipts:return self.receipts[call_id]
        evidence=resolve_delta(samples,self.known,maximum)
        self.known+=api.money(evidence['resolved_observed_charge_usd'])
        self.committed+=api.money(maximum)
        self.receipts[call_id]=evidence
        self.save()
        return evidence


class StreamTrace:
    def __init__(self,path,secret):
        self.path,self.secret=path,secret
        self.start=time.monotonic()
        self.first=None
        self.generation=None
        self.usage=None
        self.providers=set()
        self.handle=path.open('w')

    def record(self,line):
        elapsed=time.monotonic()-self.start
        event={'_client_at':api.utc(),'_client_elapsed_seconds':elapsed}
        if line.startswith(b'data:'):
            raw=line[5:].strip()
            if raw != b'[DONE]':
                obj=json.loads(raw)
                event['event']=obj
                self.generation=obj.get('id') or self.generation
                self.usage=obj.get('usage') or self.usage
                if obj.get('provider'):self.providers.add(obj['provider'])
                for choice in obj.get('choices',[]):
                    delta=choice.get('delta',{})
                    if self.first is None and any(delta.get(k) for k in ('content','reasoning','reasoning_content')):
                        self.first=elapsed
            else:event['done']=True
        else:event['keepalive_or_other_line']=True
        self.handle.write(api.redact(json.dumps(event),self.secret)+'\n')
        self.handle.flush()

    def close(self):
        self.handle.close()


class RecordedResponse:
    def __init__(self,response,trace):
        self.response,self.trace=response,trace
        self.status=response.status

    def __enter__(self):
        self.response.__enter__()
        return self

    def __exit__(self,*args):
        return self.response.__exit__(*args)

    def __iter__(self):
        for line in self.response:
            self.trace.record(line)
            yield line


class Client(api.Client):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.post_timeout=300
        self.get_timeout=30
        self.journal=None
        self.trace=None

    def request(self,path,body=None):
        if not path.startswith('/') or '://' in path:
            raise api.Refusal('invalid API path')
        headers={'Authorization':'Bearer '+self.secret,'Content-Type':'application/json'}
        raw=None if body is None else json.dumps(body).encode()
        if body is not None and self.journal is not None:
            self.trace=StreamTrace(self.journal,self.secret)
        response=self.opener.open(Request(self.base+path,data=raw,headers=headers),timeout=self.get_timeout if body is None else self.post_timeout)
        return RecordedResponse(response,self.trace) if body is not None and self.trace is not None else response


def stream(client,payload,endpoint,depth,journal):
    seconds=deadline_for_depth(depth)
    client.post_timeout=seconds
    client.journal=journal
    client.trace=None
    try:
        result=api.stream_call(client,payload,endpoint,timeout_seconds=seconds)
        trace=client.trace
        if trace is not None:
            if result.get('generation_id') is None:result['generation_id']=trace.generation
            if result.get('usage') is None:result['usage']=trace.usage
            if result.get('ttft_seconds') is None:result['ttft_seconds']=trace.first
            result['observed_providers']=sorted(trace.providers)
        error=result.get('error','').lower()
        result['timed_out']=('deadline' in error or 'timed out' in error or result.get('http_status') in (408,504))
        result['request_deadline_seconds']=seconds
        result['journal']=journal.name
        return result
    finally:
        if client.trace is not None:client.trace.close()
        client.journal=None


def availability(rows):
    count=len(rows)
    success=sum(bool(r.get('accepted')) for r in rows)
    return dict(attempts=count,successes=success,errors=count-success,
                timeouts=sum(bool(r.get('timed_out') or 'deadline' in (r.get('error') or '')) for r in rows),
                rate=success/count if count else None)


def cache_rows(rows):
    return [r for r in rows if r.get('call_index',0)>0 and r.get('cohort')!='probe' and not r.get('smoke')]


def eligible(candidate,rows):
    endpoint=candidate['endpoint']
    if endpoint['context_length'] < (1000000 if candidate['model']=='z-ai/glm-5.3-flash' else 262144):
        return 'excluded-context'
    if 'input_cache_read' not in endpoint['pricing']:return 'excluded-no-declared-cache'
    measured=availability(rows)
    if measured['rate'] is None:return 'unqualified-no-attempts'
    return 'eligible' if measured['rate']>=.95 else 'excluded-availability'

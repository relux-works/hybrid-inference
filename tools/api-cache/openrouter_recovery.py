"""Charge, stream and analysis contracts for the r15d restart."""
import json
from pathlib import Path
import time
from decimal import Decimal
from urllib.request import Request

import openrouter_cache as api

MIN_FREE_BYTES = 28 * 1024**3
DISK_STOP_BYTES = int(28.5 * 1024**3)
DISK_WAIT_SECONDS = 1200
DISK_POLL_SECONDS = 60
REQUEST_OUTPUT_TOKENS = 16
RESERVED_OUTPUT_TOKENS = 1024
ROUNDING = Decimal('.00000001')
# Cumulative API usage and the sum of streamed costs have different precision.
SETTLEMENT_ROUNDING = Decimal('.0000001')
SETTLEMENT_DEADLINE_SECONDS = 1200
MAX_PERMANENT_RESERVATIONS = 3
DEPTH_DEADLINES = [(32768,180),(131072,300),(250000,600),(524288,1200),(1000000,2400)]


class DiskHeadroomTimeout(api.Refusal):
    pass


class SettlementTimeout(api.Refusal):
    pass


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


def wait_for_disk(path,on_event=None,check_stop=None):
    """Wait before admission; callers must finish recording any previous request first."""
    started=None;last_free=None;started_at=None
    def event(kind,free,reason):
        data=dict(at=api.utc(),event=kind,reason=reason,free_bytes=free,wait_started_at=started_at,
                  elapsed_seconds=0 if started is None else time.monotonic()-started,
                  wait_limit_seconds=DISK_WAIT_SECONDS,next_check_seconds=DISK_POLL_SECONDS)
        if on_event is None:print(json.dumps(dict(disk_wait=data)),flush=True)
        else:on_event(data)
    while True:
        if check_stop is not None:
            try:check_stop()
            except api.Refusal:
                if started is not None:event('end',last_free,'stopped')
                raise
        free=api.shutil.disk_usage(path).free;last_free=free;now=time.monotonic()
        if free<MIN_FREE_BYTES:
            event('hard-floor' if started is None else 'end',free,'hard-floor')
            raise api.Refusal('free disk below28 GiB hard floor')
        if free>=DISK_STOP_BYTES:
            if started is not None:event('end',free,'recovered')
            return started is not None
        if started is None:
            started=now;started_at=api.utc();event('start',free,'requires-headroom')
        elif now-started>=DISK_WAIT_SECONDS:
            event('end',free,'timeout')
            raise DiskHeadroomTimeout('free disk requires28.5 GiB after1200s headroom wait')
        else:event('check',free,'requires-headroom')
        time.sleep(min(DISK_POLL_SECONDS,DISK_WAIT_SECONDS-(now-started)))


def resolve_delta(samples, expected, maximum):
    expected,maximum=api.money(expected),api.money(maximum)
    values=[api.money(s['usage_usd']) for s in samples]
    if len(values)<3 or len(set(values))!=1:
        raise api.Refusal('unresolvable charge: cumulative usage is missing or unstable')
    delta=values[-1]-expected
    if delta < -SETTLEMENT_ROUNDING or delta > maximum:
        raise api.Refusal('unresolvable charge: cumulative delta outside reservation')
    return dict(method='stable cumulative usage delta',expected_usd=str(expected),maximum_usd=str(maximum),
                observed_delta_usd=str(delta),resolved_observed_charge_usd=str(max(Decimal(0),delta)),
                retained_reservation_usd=str(maximum),rounding_tolerance_usd=str(SETTLEMENT_ROUNDING),samples=samples,resolved=True)


def usage_samples(client, count=3, pause=1):
    result=[]
    for index in range(count):
        data=client.get('/key')['data']
        result.append(dict(at=api.utc(),usage_usd=str(data['usage'])))
        if index+1<count:time.sleep(pause)
    return result


def settled_samples(client,expected,maximum,evidence_path,timeout_seconds=SETTLEMENT_DEADLINE_SECONDS,poll_seconds=15):
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
    raise SettlementTimeout('unresolvable charge: cumulative counter did not settle within recovery deadline')


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
                    committed_upper_usd=str(self.committed),restart_reported_usd=str(self.reported),global_cap_usd=45,
                    permanent_reservation_count=len(self.permanent_receipts()),receipts=self.receipts))

    def permanent_receipts(self):
        return {key:receipt for key,receipt in self.receipts.items() if receipt.get('permanent_reservation')}

    def check_permanent_limit(self):
        if len(self.permanent_receipts())>MAX_PERMANENT_RESERVATIONS:
            raise api.Refusal('systemic billing problem: fourth permanent reservation; no further spending')

    def reserve(self,live,maximum,cap=45,remaining=None):
        self.check_permanent_limit()
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

    def permanent_call(self,maximum,call_id,evidence):
        if call_id not in self.receipts:
            maximum=api.money(maximum)
            if api.money(evidence['expected_usd'])!=self.known or api.money(evidence['maximum_usd'])!=maximum or not evidence['samples']:
                raise api.Refusal('permanent reservation missing or mismatched settlement evidence')
            if any(api.money(s['usage_usd'])>self.known+maximum for s in evidence['samples'] if 'usage_usd' in s):
                raise api.Refusal('unresolvable charge: cumulative delta exceeds reservation')
            self.committed+=maximum
            self.receipts[call_id]=dict(method='permanent maximum reservation',status='unresolved-reserved',resolved=False,
                permanent_reservation=True,expected_usd=str(self.known),maximum_usd=str(maximum),
                retained_reservation_usd=str(maximum),observed_charge_usd=None,settlement_evidence=evidence,
                reason='no-generation failure; cumulative usage unstable after1200s')
            self.save()
        self.check_permanent_limit()
        return self.receipts[call_id]


def failed_without_generation(row):
    if row.get('generation_id') or any(c.get('id') for c in row.get('stream',[])):return False
    if row.get('timed_out') or row.get('http_status')!=200 or any(c.get('error') for c in row.get('stream',[])):return True
    if row.get('output_guard_triggered') and not row.get('usage_complete'):return True
    try:api.parse_usage(row.get('usage'))
    except api.Refusal:return True
    return False


def reconcile_call(client,budget,row,call_id,evidence_path):
    try:
        samples=settled_samples(client,budget.known,row['maximum_cost_usd'],evidence_path)
        return budget.recovered_call(samples,row['maximum_cost_usd'],call_id)
    except SettlementTimeout:
        if not failed_without_generation(row):raise
        # Missing/corrupt poll evidence is an instrument failure, not a reservation.
        evidence=json.loads(evidence_path.read_text())
        return budget.permanent_call(row['maximum_cost_usd'],call_id,evidence)


class StreamTrace:
    def __init__(self,path,secret):
        self.path,self.secret=path,secret
        self.start=time.monotonic()
        self.first=None
        self.generation=None
        self.usage=None
        self.usage_complete=False
        self.providers=set()
        self.models=set()
        self.output_bytes=0
        self.output_tokens=None
        self.guard_triggered=False
        self.guard_reason=None
        self.evidence_error=None
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
                if obj.get('usage'):
                    self.usage_complete=not obj.get('choices') or any(c.get('finish_reason') is not None for c in obj['choices'])
                if obj.get('provider'):self.providers.add(obj['provider'])
                if obj.get('model'):self.models.add(obj['model'])
                tokens=(obj.get('usage') or {}).get('completion_tokens')
                if isinstance(tokens,int) and not isinstance(tokens,bool) and tokens>=0:self.output_tokens=tokens
                for choice in obj.get('choices',[]):
                    delta=choice.get('delta',{})
                    texts=[delta.get(k) for k in ('content','reasoning','reasoning_content')]
                    if not any(delta.get(k) for k in ('reasoning','reasoning_content')):
                        texts += [part.get('text') for part in delta.get('reasoning_details',[]) if isinstance(part,dict)]
                    self.output_bytes+=sum(len(t.encode('utf-8')) for t in texts if isinstance(t,str))
                    if self.first is None and any(delta.get(k) for k in ('content','reasoning','reasoning_content')):
                        self.first=elapsed
            else:event['done']=True
        else:event['keepalive_or_other_line']=True
        try:
            self.handle.write(api.redact(json.dumps(event),self.secret)+'\n')
            self.handle.flush()
        except OSError:
            self.evidence_error='stream evidence write failed'
            raise
        # SSE chunks are not tokens. Bytes are an earlier, conservative stop for exposed text.
        if self.output_tokens is not None and self.output_tokens>RESERVED_OUTPUT_TOKENS:
            self.guard_reason='reported completion tokens exceed reserved output bound'
        elif self.output_bytes>RESERVED_OUTPUT_TOKENS:
            self.guard_reason='exposed output UTF-8 bytes exceed conservative streaming bound'
        if self.guard_reason:
            self.guard_triggered=True
            raise api.Refusal('output guard canceled stream')

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
        self.instrument_failure=None

    def request(self,path,body=None):
        if not path.startswith('/') or '://' in path:
            raise api.Refusal('invalid API path')
        headers={'Authorization':'Bearer '+self.secret,'Content-Type':'application/json'}
        raw=None if body is None else json.dumps(body).encode()
        if body is not None and self.journal is not None:
            try:self.trace=StreamTrace(self.journal,self.secret)
            except OSError:
                self.instrument_failure='stream evidence could not be opened'
                raise
        response=self.opener.open(Request(self.base+path,data=raw,headers=headers),timeout=self.get_timeout if body is None else self.post_timeout)
        return RecordedResponse(response,self.trace) if body is not None and self.trace is not None else response


def stream(client,payload,endpoint,depth,journal):
    seconds=deadline_for_depth(depth)
    client.post_timeout=seconds
    client.journal=journal
    client.trace=None
    client.instrument_failure=None
    try:
        result=api.stream_call(client,payload,endpoint,timeout_seconds=seconds)
        trace=client.trace
        if trace is not None:
            if result.get('generation_id') is None:result['generation_id']=trace.generation
            if result.get('usage') is None:result['usage']=trace.usage
            if result.get('ttft_seconds') is None:result['ttft_seconds']=trace.first
            result['observed_providers']=sorted(trace.providers)
            result['observed_models']=sorted(trace.models)
            result.update(output_guard_triggered=trace.guard_triggered,output_guard_reason=trace.guard_reason,
                          streamed_output_utf8_bytes=trace.output_bytes,reserved_output_tokens=RESERVED_OUTPUT_TOKENS,
                          usage_complete=trace.usage_complete)
            if trace.guard_triggered:
                result['completion_over_cap']=trace.output_tokens if trace.usage_complete and trace.output_tokens is not None and trace.output_tokens>REQUEST_OUTPUT_TOKENS else None
                result['completion_over_cap_censored']=not trace.usage_complete
            allowed=allowed_models(payload['model'],endpoint)
            result['allowed_model_identities']=sorted(allowed)
            if known_models(trace.models)-allowed:result['error']='serving model differs from pinned model'
            if trace.providers and trace.providers!={endpoint['provider_name']}:result['error']='serving provider differs from pinned provider'
            if trace.evidence_error:result['instrument_failure']=trace.evidence_error
            if result.get('usage') is not None:
                try:result.update(api.parse_usage(result['usage']))
                except api.Refusal:pass
            if trace.guard_triggered and not trace.usage_complete:
                result['partial_usage_cost_usd']=result.pop('cost_usd',None)
                result['observed_completion_tokens_lower_bound']=trace.output_tokens
        error=result.get('error','').lower()
        result['timed_out']=('deadline' in error or 'timed out' in error or result.get('http_status') in (408,504))
        result['request_deadline_seconds']=seconds
        result['journal']=journal.name
        if client.instrument_failure:result['instrument_failure']=client.instrument_failure
        return result
    finally:
        if client.trace is not None:client.trace.close()
        client.journal=None


def allowed_models(model,endpoint):
    allowed={model}
    for item in json.loads(api.SNAPSHOT.read_text())['models']:
        if item['data']['id']!=model:continue
        for frozen in item['data']['endpoints']:
            if frozen['tag']==endpoint['tag'] and frozen.get('model_id')==model:
                _,separator,canonical=frozen['name'].partition(' | ')
                if separator:allowed.add(canonical)
    return allowed


def classify_call(row,target,model,endpoint):
    """Accept complete usage independently of requested output length; retain real failures."""
    row.update(accepted=False,classification_protocol='r15j')
    row['availability_success']=False
    row.pop('halt_class',None)
    tokens=row.get('completion_tokens')
    if not row.get('completion_over_cap_censored') and isinstance(tokens,int) and tokens>REQUEST_OUTPUT_TOKENS:row['completion_over_cap']=tokens
    observed_models=model_observations(row)
    observed_providers=provider_observations(row)
    if row.get('error')=='serving model differs from pinned model' and not observed_models:
        row['previous_classification']=row.pop('error')
    if observed_providers and observed_providers!={endpoint['provider_name']}:row['error']='serving provider differs or is absent'
    if not observed_providers:row['error']='serving provider differs or is absent'
    if observed_models-allowed_models(model,endpoint):
        row.update(error='serving model differs from pinned model',halt_class='identity')
    if row.get('instrument_failure'):row.update(error=row['instrument_failure'],halt_class='instrument')
    row['identity_status']='proven' if observed_models and not observed_models-allowed_models(model,endpoint) and observed_providers=={endpoint['provider_name']} else 'unproven'
    try:parsed=api.parse_usage(row.get('usage'))
    except api.Refusal as error:
        row['usage_error']='usage parsing failed: '+str(error)
        if not row.get('error') or row['error'] in ('completion cap exceeded','output guard canceled stream'):row['error']=row['usage_error']
        return
    if row.get('output_guard_triggered') and not row.get('usage_complete'):return
    row.update(parsed)
    row['availability_success']=row.get('http_status')==200 and not row.get('halt_class') and not any(c.get('error') for c in row.get('stream',[]))
    if parsed['completion_tokens']>REQUEST_OUTPUT_TOKENS:row['completion_over_cap']=parsed['completion_tokens']
    if row.get('error') in ('completion cap exceeded','output guard canceled stream'):
        row['previous_classification']=row.pop('error')
    if row.get('error'):return
    if row.get('http_status')!=200:row['error']='successful HTTP status missing';return
    if not observed_models:row['error']='serving model unknown or absent; identity unproven';return
    if abs(row['prompt_tokens']-target)>max(512,target*.03):row['error']='measured depth outside tolerance';return
    row['accepted']=True


def known_models(values):
    return {value for value in values if isinstance(value,str) and value.strip() and value.strip().lower()!='unknown'}


def model_observations(row):
    return known_models(list(row.get('observed_models') or [])+[c.get('model') for c in row.get('stream',[])])


def provider_observations(row):
    # Explicit empty observations cannot fall back to an older serving-provider label.
    values=row.get('observed_providers',row.get('serving_provider',[]))
    if not isinstance(values,list):return set()
    return {value for value in values if isinstance(value,str) and value.strip()}


def rate_limited_without_generation(row):
    return (row.get('http_status')==429 and row.get('generation_id') is None and row.get('usage') is None
            and not row.get('stream') and not provider_observations(row) and not row.get('observed_models')
            and not row.get('output_guard_triggered'))


def availability(rows):
    count=len(rows)
    success=sum(bool(r.get('availability_success',r.get('accepted'))) for r in rows)
    return dict(attempts=count,successes=success,errors=count-success,
                timeouts=sum(bool(r.get('timed_out') or 'deadline' in (r.get('error') or '')) and not rate_limited_without_generation(r) for r in rows),
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

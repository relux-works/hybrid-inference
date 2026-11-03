#!/usr/bin/env python3
"""Budgeted, provider-pinned OpenRouter cache measurements (standard library only)."""
import argparse
import csv
import fcntl
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import stat
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / 'snapshots/2026-11-03.json'
LIVE_REPORT_PATH = Path('/home/agent/Werkstatt/hybrid-d/llama.cpp/.temp/hybrid-d-r15d-261103/API-CACHE-REPORT.md')
MIN_FREE_BYTES = 28 * 1024**3
KEY_PATH = Path('/home/agent/lanes/hybrid/.temp/openrouter/api_key')
PAUSES = [0, 0, 60, 60, 300, 300, 900, 900, 3600, 3600]
WORDS = list('abcdefghijklmnopqrstuvwxyz')


class Refusal(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def money(value):
    try:
        result = Decimal(str(value))
    except Exception:
        raise Refusal('missing or invalid price/spend') from None
    if not result.is_finite() or result < 0:
        raise Refusal('negative or nonfinite price/spend')
    return result


def redact(value, secret=''):
    text = json.dumps(value, ensure_ascii=True) if not isinstance(value, str) else value
    if secret:
        text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'(?i)(bearer\s+)[^\s"\\]+', r'\1[REDACTED]', text)
    text = re.sub(r'sk-or-v1-[A-Za-z0-9_-]+', '[REDACTED]', text)
    return text


def assert_no_secret(text, secret):
    if secret and secret in text:
        raise Refusal('secret detected in output')


def write_json(path, data, secret=''):
    text = redact(json.dumps(data, indent=2, ensure_ascii=True), secret) + '\n'
    assert_no_secret(text, secret)
    path.write_text(text)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Refusal('redirect refused')


class Client:
    def __init__(self, key_path=KEY_PATH, base='https://openrouter.ai/api/v1', mock=False):
        parsed = urlparse(base)
        if base != 'https://openrouter.ai/api/v1' and not (
                mock and parsed.scheme == 'http' and parsed.hostname == '127.0.0.1'):
            raise Refusal('network destination refused')
        mode = stat.S_IMODE(key_path.stat().st_mode)
        if mode != 0o600:
            raise Refusal('key file must have mode 600')
        self.secret = key_path.read_text().strip()
        if not self.secret or '\n' in self.secret or '\r' in self.secret:
            raise Refusal('invalid key file')
        self.base = base
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(self, path, body=None):
        if not path.startswith('/') or '://' in path:
            raise Refusal('invalid API path')
        headers = {'Authorization': 'Bearer ' + self.secret, 'Content-Type': 'application/json'}
        raw = None if body is None else json.dumps(body).encode()
        return self.opener.open(Request(self.base + path, data=raw, headers=headers), timeout=300)

    def get(self, path):
        with self.request(path) as response:
            return json.load(response)


def parse_usage(usage):
    if not isinstance(usage, dict):
        raise Refusal('usage missing')
    details = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
    prompt = usage.get('prompt_tokens', usage.get('input_tokens'))
    completion = usage.get('completion_tokens', usage.get('output_tokens'))
    cached = details.get('cached_tokens', usage.get('cache_read_input_tokens'))
    written = details.get('cache_write_tokens', usage.get('cache_creation_input_tokens'))
    for name, value in [('prompt', prompt), ('completion', completion), ('cached', cached), ('write', written)]:
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise Refusal('invalid ' + name + ' usage')
    if prompt is None or completion is None:
        raise Refusal('token usage missing')
    if cached is not None and cached > prompt:
        raise Refusal('cache count exceeds prompt')
    if written is not None and written > prompt:
        raise Refusal('write count exceeds prompt')
    return dict(prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached,
                cache_write_tokens=written, cost_usd=float(money(usage.get('cost'))))


def check_provider(observed, endpoint):
    if not observed or observed != endpoint['provider_name']:
        raise Refusal('serving provider differs or is absent')


def max_cost(endpoint, prompt_bound, max_tokens=16):
    price = endpoint['pricing']
    read = money(price['input_cache_read'])
    prompt = money(price['prompt'])
    write = money(price.get('input_cache_write', 0))
    # Sum prompt and write as a conservative bound even if writes replace prompt charges.
    return Decimal(prompt_bound) * max(prompt + write, read) + Decimal(max_tokens) * money(price['completion'])


def guard(live_spend, local_spend, estimate, cap, remaining=None):
    spent = max(money(live_spend), money(local_spend))
    if spent + estimate > money(cap):
        raise Refusal('budget would exceed cap')
    if remaining is not None and estimate > money(remaining):
        raise Refusal('live remaining budget insufficient')
    return spent


def matrix(snapshot):
    sessions, excluded = [], []
    for model in snapshot['models']:
        data = model['data']
        is_qwen = data['id'] == 'qwen/qwen3.8-27b'
        threshold = 262144 if is_qwen else 1000000
        for endpoint in data['endpoints']:
            reasons = []
            if endpoint['context_length'] < threshold:
                reasons.append('context below ' + str(threshold))
            if 'input_cache_read' not in endpoint['pricing']:
                reasons.append('cache read price not declared')
            if not endpoint.get('tag'):
                reasons.append('exact routing tag absent')
            if reasons:
                excluded.append(dict(model=data['id'], endpoint=endpoint['name'], reasons=reasons))
                continue
            depths = [32768, 131072, 250000] if is_qwen else [250000, 1000000]
            if is_qwen and endpoint['context_length'] >= 1000000:
                depths.append(1000000)
            for depth in depths:
                ident = data['id'] + ':' + endpoint['tag'] + ':' + str(depth)
                sessions.append(dict(model=data['id'], endpoint=endpoint, depth=depth,
                                     routing_ignore=[e['tag'] for e in data['endpoints'] if e.get('tag','').startswith(endpoint['tag']+'/')],
                                     session_id='r15b-v3-' + hashlib.sha256(ident.encode()).hexdigest()[:20]))
    random.Random(261103).shuffle(sessions)
    return sessions, excluded


def synthetic(target, seed):
    rng = random.Random(seed)
    # Space-prefixed letters limit proxy character estimates; returned usage checks token depth.
    return ''.join(' ' + rng.choice(WORDS) for _ in range(max(0, target)))


def build_payload(session, index, smoke=False):
    depth = session['depth']
    initial = depth if smoke else depth - 9000 - 64
    endpoint = session['endpoint']
    prefix = 'Synthetic public-domain cache experiment. Read the records and reply OK. Session ' + session['session_id'] + '.\n'
    text = prefix + synthetic(initial, session['session_id'])
    for turn in range(index):
        text += synthetic(1000, session['session_id'] + ':' + str(turn))
    content = [{'type': 'text', 'text': text}]
    if not endpoint.get('supports_implicit_caching', False):
        content[0]['cache_control'] = {'type': 'ephemeral'}
    payload = dict(model=session['model'], messages=[dict(role='user', content=content)], max_tokens=16,
                   stream=True, stream_options={'include_usage': True}, temperature=0,
                   session_id=session['session_id'], provider=dict(order=[endpoint['tag']], only=[endpoint['tag']],
                   allow_fallbacks=False))
    payload['reasoning'] = ({'enabled': False} if session['model']=='qwen/qwen3.8-27b' else {'effort': 'low'})
    if session.get('routing_ignore'):
        payload['provider']['ignore'] = session['routing_ignore']
    return payload


def request_deadline(signum, frame):
    raise Refusal('request exceeded wall-clock deadline')


def stream_call(client, payload, endpoint, event_path=None, timeout_seconds=300):
    start = time.monotonic()
    first, last, usage, providers, generation, chunks = None, None, None, set(), None, []
    status = None
    previous_handler = signal.signal(signal.SIGALRM, request_deadline)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    journal = None if event_path is None else event_path.open('w')
    try:
        with client.request('/chat/completions', payload) as response:
            status = response.status
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                item = line[5:].strip()
                if item == b'[DONE]':
                    break
                obj = json.loads(item)
                chunks.append(obj)
                if journal is not None:
                    journal.write(redact(json.dumps(obj),client.secret)+'\n')
                    journal.flush()
                if obj.get('error'):
                    raise Refusal('stream reported error')
                if obj.get('provider'):
                    providers.add(obj['provider'])
                generation = obj.get('id', generation)
                if obj.get('usage'):
                    usage = obj['usage']
                for choice in obj.get('choices', []):
                    delta = choice.get('delta', {})
                    if delta.get('content') or delta.get('reasoning') or delta.get('reasoning_content'):
                        now = time.monotonic()
                        first = now if first is None else first
                        last = now
        if providers != {endpoint['provider_name']}:
            raise Refusal('serving provider differs or is absent')
        parsed = parse_usage(usage)
        duration = None if first is None or last is None else last - first
        speed = ((parsed['completion_tokens'] - 1) / duration
                 if duration and parsed['completion_tokens'] > 1 else None)
        return dict(http_status=status, generation_id=generation, serving_provider=sorted(providers),
                    ttft_seconds=None if first is None else first - start,
                    decode_tokens_per_second=speed, elapsed_seconds=time.monotonic() - start,
                    usage=usage, stream=chunks, **parsed)
    except HTTPError as error:
        return dict(http_status=error.code, error='HTTP error', error_body=redact(error.read().decode(errors='replace'), client.secret))
    except Exception as error:
        return dict(http_status=status, generation_id=generation, usage=usage, error=redact(str(error), client.secret), stream=chunks,
                    elapsed_seconds=time.monotonic() - start)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if journal is not None:
            journal.close()


def sample_endpoints(client, out, sequence):
    data = {'at': utc(), 'models': []}
    for model in ('qwen/qwen3.8-27b', 'z-ai/glm-5.3-flash'):
        data['models'].append(client.get('/models/' + model + '/endpoints'))
    write_json(out / ('uptime-%04d.json' % sequence), data, client.secret)
    return data


def live_endpoint(sample, session):
    for model in sample['models']:
        if model['data']['id'] == session['model']:
            for endpoint in model['data']['endpoints']:
                if endpoint['tag'] == session['endpoint']['tag']:
                    return endpoint
    raise Refusal('endpoint disappeared')


def validate_live_price(frozen, live):
    for field in ('prompt', 'completion', 'input_cache_read', 'input_cache_write'):
        if money(live['pricing'].get(field, 0)) > money(frozen['pricing'].get(field, 0)):
            raise Refusal('live price increased since preregistration')


def report(out, snapshot, status, reason='', secret=''):
    raise Refusal('Legacy cache report retired; live report: '+str(LIVE_REPORT_PATH))


def run(args):
    out = args.output.resolve()
    report(out,None,'retired')
    out.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads(args.snapshot.read_text())
    sessions, excluded = matrix(snapshot)
    if args.smoke:
        sessions = [s for s in sessions if s['model']=='qwen/qwen3.8-27b' and s['endpoint']['tag']=='chutes/fp8' and s['depth']==32768]
        if len(sessions) != 1:
            raise Refusal('smoke endpoint missing or ambiguous')
        sessions[0]['session_id'] += '-smoke'
    count = 3 if args.smoke else 10
    cap = Decimal('.50') if args.smoke else Decimal('45')
    client = Client(args.key_path)
    status, reason = 'failed', 'run interrupted'
    total, sequence = Decimal(0), 0
    # One process owns this dedicated-key experiment; no automatic resume can duplicate billed calls.
    lock = open(args.key_path.parent/'cache-run.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = utc()
    try:
        initial_key = client.get('/key')['data']
        initial_spend = max(money(initial_key['usage']), money(getattr(args, 'prior_spend_floor', 0)))
        write_json(out/'run.json', dict(started=started, pid=os.getpid(), smoke=args.smoke,
                   snapshot_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   initial_key_usage=float(money(initial_key['usage'])), accounting_spend_floor=float(initial_spend), cap_usd=float(cap), sessions=len(sessions),
                   excluded=excluded), client.secret)
        queue = [(time.monotonic(), i, 0, None) for i in range(len(sessions))]
        heapq.heapify(queue)
        while queue:
            due, number, index, previous_end = heapq.heappop(queue)
            if (out/'STOP').exists():
                raise Refusal('operator STOP file; no further requests')
            if time.monotonic() < due:
                time.sleep(due-time.monotonic())
            if shutil.disk_usage(out).free < MIN_FREE_BYTES:
                raise Refusal('free disk below 28 GiB')
            sample = sample_endpoints(client, out, sequence)
            session = sessions[number]
            endpoint = live_endpoint(sample, session)
            validate_live_price(session['endpoint'], endpoint)
            payload = build_payload(session,index,args.smoke)
            text = payload['messages'][0]['content'][0]['text']
            # UTF-8 bytes bound token count for byte-level tokenizers; context bounds an accepted prompt.
            bound = min(len(text.encode())+512, endpoint['context_length'])
            estimate = max_cost(endpoint,bound)
            key = client.get('/key')['data']
            spent = guard(key['usage'], initial_spend+total, estimate, 45, key.get('limit_remaining'))
            guard(max(Decimal(0),money(key['usage'])-initial_spend),total,estimate,cap,key.get('limit_remaining'))
            before = utc()
            actual_pause = None if previous_end is None else time.monotonic()-previous_end
            row = dict(sequence=sequence, model=session['model'], endpoint_tag=endpoint['tag'],
                       expected_provider=endpoint['provider_name'],provider_request=payload['provider'],depth=session['depth'],session_id=session['session_id'],
                       call_index=index, started_at=before,pause_target_seconds=0 if args.smoke else PAUSES[index],
                       pause_actual_seconds=actual_pause,new_tokens_estimate=1000 if index else None,
                       prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),prompt_bytes=len(text.encode()),
                       max_cost_usd=float(estimate),live_spend_before_usd=float(spent),accepted=False)
            write_json(out/('intent-%04d.json'%sequence),row,client.secret)
            result = stream_call(client,payload,endpoint,event_path=out/('stream-%04d.jsonl'%sequence))
            response_end = time.monotonic()
            row.update(result)
            row['ended_at'] = utc()
            if row.get('cost_usd') is not None:
                total += money(row['cost_usd'])
            write_json(out/('call-%04d.json'%sequence),row,client.secret)
            if row.get('cost_usd') is not None and money(row['cost_usd']) > estimate + Decimal('.00000001'):
                raise Refusal('unexpected billed cost exceeds conservative maximum')
            after = client.get('/key')['data']
            write_json(out/('spend-%04d.json'%sequence),dict(at=utc(),usage=after['usage'],
                       limit_remaining=after.get('limit_remaining'),local_billed_usd=float(total)),client.secret)
            if money(after['usage']) > initial_spend+total+Decimal('.000001'):
                raise Refusal('unexpected live spend not explained by recorded calls')
            guard(after['usage'], initial_spend+total, Decimal(0), 45)
            if 'error' not in row:
                target = session['depth']+index*1000 if args.smoke else session['depth']-9000+index*1000
                if abs(row['prompt_tokens']-target) > max(512,target*.03):
                    row['error'] = 'measured prompt depth outside 3% tolerance'
                elif row['completion_tokens'] > 16:
                    row['error'] = 'completion cap exceeded'
                else:
                    row['accepted'] = True
            write_json(out/('call-%04d.json'%sequence),row,client.secret)
            print(redact(json.dumps({k:row.get(k) for k in ('sequence','endpoint_tag','depth','call_index','prompt_tokens','cached_tokens','cost_usd','http_status','error','accepted')}),client.secret),flush=True)
            sequence += 1
            report(out,snapshot,'running',secret=client.secret)
            if row.get('error'):
                # HTTP availability failures are measured; unknown billing or pin/depth violations stop spending.
                if row.get('http_status') not in (404,408,429,500,502,503,504):
                    raise Refusal('call failed or unverifiable; no further spending')
            if index+1 < count:
                pause = 0 if args.smoke else PAUSES[index+1]
                heapq.heappush(queue,(response_end+pause,number,index+1,response_end))
        status,reason = 'done','all scheduled attempts recorded'
    except Exception as error:
        reason = redact(str(error),client.secret)
        print('failure: '+reason,flush=True)
    finally:
        write_json(out/'terminal.json',dict(status=status,reason=reason,ended_at=utc(),
                   recorded_billed_usd=float(total),attempts=len(list(out.glob('call-*.json')))),client.secret)
        report(out,snapshot,status,reason,client.secret)
        print(status+' '+utc()+' '+reason,flush=True)
        lock.close()
    return 0 if status=='done' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,default=SNAPSHOT)
    parser.add_argument('--key-path',type=Path,default=KEY_PATH)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--prior-spend-floor',default='0',help='Known cumulative prior costs, retained if live spend lags')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:
        report(args.output,json.loads(args.snapshot.read_text()),'report-only')
        return 0
    return run(args)


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as error:
        # Never emit a traceback containing request headers or remote exception payloads.
        print('failed '+type(error).__name__,flush=True)
        sys.exit(1)

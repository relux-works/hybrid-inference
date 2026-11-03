#!/usr/bin/env python3
"""Time the real loopback node -> native CUDA slot path on an exclusive client."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import urllib.parse
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'hybrid-proxy'))
sys.path.insert(0, str(HERE.parent / 'hybrid-bench'))
from proxy import Node, Proxy
from store import prefix_hash
from bench_common import read_private, require, validate_identity

DEPTHS = (16384, 65536, 131072)
DELTAS = (2343, 10000)
PHASES = dict(server_prefill_ms='X-Hybrid-Evaluate-Ms', export_ms='X-Hybrid-Export-Ms',
              conversion_ms='X-Hybrid-Conversion-Ms', backend_ms='X-Hybrid-Backend-Ms')


class TimedNode(Node):
    def request(self, method, path, data=None, headers=None, binary=False):
        self.phase = {}
        with self.open(method, path, data, headers) as response:
            started = time.perf_counter()
            body = response.read(600 * 1024**2 + 1)
            self.last_transfer_seconds = time.perf_counter() - started
            self.last_bytes = len(body)
            require(len(body) <= 600 * 1024**2, 'response too large')
            if binary:
                self.phase = phase_headers(response.headers)
            return body if binary else json.loads(body) if body else None


def phase_headers(headers):
    result = {}
    for name, header in PHASES.items():
        value = float(headers.get(header, 'nan'))
        require(math.isfinite(value) and value >= 0, 'missing/invalid phase header: ' + header)
        result[name] = value
    require(abs(result['backend_ms'] - result['server_prefill_ms'] - result['export_ms']) < .01,
            'backend phase accounting mismatch')
    return result


def chunks(n0, n1, block, maximum=7840):
    step = maximum // block * block
    require(step > 0 and n1 > n0, 'invalid transfer range')
    spans = []
    while n0 < n1:
        end = min(n1, (n0 + step) // block * block)
        require(end > n0, 'chunk made no progress')
        spans.append((n0, end)); n0 = end
    return spans


def first_token(decode, tokens):
    with decode.open('POST', '/completion', dict(prompt=tokens, n_predict=1, id_slot=0, cache_prompt=True,
                                               stream=True, temperature=0, ignore_eos=True)) as response:
        count, first, done = 0, None, False
        for line in response:
            if not line.startswith(b'data: '):
                continue
            if line[6:].strip() == b'[DONE]':
                done = True; continue
            obj = json.loads(line[6:])
            require(not obj.get('error'), 'decoder SSE refusal')
            if obj.get('tokens') or obj.get('content'):
                first = first or time.perf_counter()
                count += 1
            done = done or bool(obj.get('stop'))
        require(first is not None and done and count > 0, 'decoder token/terminal event missing')
        return first


def one(remote, decode, rebinder, health, depth, delta, seed):
    block = health['export_block_size']
    tokens = [100 + seed % 1000] * (depth + delta)
    end = (len(tokens) - 1) // block * block
    require(depth < end < len(tokens), 'no aligned delta endpoint')
    path = '/v1/sessions/' + remote.request('POST', '/v1/sessions', dict(model_sha256=health['model_sha256']))['session_id']
    sid = path.rsplit('/', 1)[1]
    try:
        decode.request('POST', '/slots/0?action=erase', {})
        # Bootstrap both copies to D before starting the measured append.
        for begin in range(0, depth, 7840):
            finish = min(depth, begin + 7840)
            remote.request('POST', path + '/extend', dict(n0=begin, prefix_hash_n0=prefix_hash(tokens[:begin]),
                           tokens=tokens[begin:finish], return_delta=False))
        decode.request('POST', '/completion', dict(prompt=tokens[:depth], id_slot=0, n_predict=1,
                       cache_prompt=True, temperature=0, ignore_eos=True))
        info = decode.request('GET', '/slots/0/info')
        require(info['tokens'] == tokens[:depth], 'decoder bootstrap journal mismatch')
        spans = chunks(depth, end, block)
        transfer = uuid.uuid4().hex
        rows = []
        started = time.perf_counter()
        for index, (begin, finish) in enumerate(spans):
            wire = remote.request('POST', path + '/extend', dict(n0=begin, prefix_hash_n0=prefix_hash(tokens[:begin]),
                tokens=tokens[begin:finish], return_delta=True, delta_version=2, transfer_id=transfer,
                transfer_n0=depth, transfer_n1=end, chunk_index=index, chunk_count=len(spans)), binary=True)
            row = dict(n0=begin, n1=finish, wire_bytes=len(wire), transfer_ms=remote.last_transfer_seconds * 1000, **remote.phase)
            wire = Proxy.client_delta(rebinder, wire)
            import_started = time.perf_counter()
            receipt = decode.request('POST', '/slots/0?action=import_delta', wire,
                        {'Content-Type': 'application/octet-stream', 'X-Hybrid-Session': sid})
            row.update(import_ms=(time.perf_counter() - import_started) * 1000, receipt=receipt)
            rows.append(row)
        info = decode.request('GET', '/slots/0/info')
        require(info['n'] == end and info['tokens'] == tokens[:end] and info['prefix_hash'] == prefix_hash(tokens[:end]),
                'import endpoint/journal mismatch')
        first = first_token(decode, tokens)
        result = dict(depth=depth, delta=delta, exported_endpoint=end, local_tail_tokens=len(tokens) - end,
                      ttft_ms=(first - started) * 1000, chunks=rows, status='measured')
        for key in ('server_prefill_ms', 'export_ms', 'conversion_ms', 'transfer_ms', 'import_ms'):
            result[key] = sum(row[key] for row in rows)
        result['other_and_local_tail_ms'] = result['ttft_ms'] - sum(result[key] for key in
                ('server_prefill_ms', 'export_ms', 'conversion_ms', 'transfer_ms', 'import_ms'))
        require(result['other_and_local_tail_ms'] >= 0, 'phase times exceed TTFT')
        return result
    finally:
        # Exclusive run-owned decoder, with no rollback requirement for its discarded fixture.
        try:
            decode.request('POST', '/slots/0?action=abort_delta', {})
            decode.request('POST', '/slots/0?action=erase', {})
        finally:
            remote.request('DELETE', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client-state', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cooldown', type=float, default=15)
    a = parser.parse_args()
    config = read_private(a.client_state / 'proxy-config.json')
    client = read_private(a.client_state / 'client-info.json')
    require(client['backend'] == 'cuda', 'requires a CUDA decoder')
    for url in (config['prefill_url'], config['decode_url']):
        parsed = urllib.parse.urlsplit(url)
        require(parsed.scheme == 'http' and parsed.hostname == '127.0.0.1' and not parsed.username, 'loopback HTTP required')
    remote = TimedNode(config['prefill_url'], next(iter(config['clients'].values()))['prefill_token'], 1800)
    decode = Node(config['decode_url'], config['decode_token'], 1800)
    health = remote.request('GET', '/v1/health')
    require(health.get('backend') == 'vllm' and health['max_ctx'] >= max(DEPTHS) + max(DELTAS), 'wrong node/capacity')
    validate_identity(config['model_sha256'], client, health['model_sha256'], health, True)
    require(health['model_sha256'] == config['node_model_sha256'], 'node source changed')
    rebinder = SimpleNamespace(hetero_quant=True, model=config['model_sha256'], node_model=health['model_sha256'])
    a.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows = [dict(depth=d, delta=delta, status='pending', rounds=[]) for d in DEPTHS for delta in DELTAS]
    result = dict(schema=1, source_base='fca0fe31', weight_comparison='heterogeneous', client_backend='cuda',
                  transport='loopback', rows=rows)
    def save():
        (a.out / 'bridge-timing.json').write_text(json.dumps(result, indent=2) + '\n')
    save()
    for index, row in enumerate(rows):
        try:
            time.sleep(a.cooldown)
            row['warmup'] = one(remote, decode, rebinder, health, row['depth'], row['delta'], 4 * index)
            for repeat in range(3):
                row['rounds'].append(one(remote, decode, rebinder, health, row['depth'], row['delta'], 4 * index + repeat + 1))
                save()
            row['median'] = {key: statistics.median(r[key] for r in row['rounds']) for key in
                ('ttft_ms', 'server_prefill_ms', 'export_ms', 'conversion_ms', 'transfer_ms', 'import_ms', 'other_and_local_tail_ms')}
            row['status'] = 'measured'
        except Exception as error:
            row.update(status='refused', error=str(error))
            save()
            raise
        finally:
            save()
    return 0


if __name__ == '__main__':
    sys.exit(main())

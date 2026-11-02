#!/usr/bin/env python3
"""A paired growing session through the real proxy, with live-state checkpoints."""
import argparse
from array import array
import asyncio
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import http.server as http_server
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid

from bench_common import (Refusal, http, private_json, read_private, require, sha256,
                          start_process, validate_identity, wait_health)
from hybrid_bench import PROVENANCE, link_probe, sanitized_info, stream_chat, verify_meter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hybrid-proxy'))
from proxy import Handler, Node, Proxy
from policy import Policy
from store import prefix_hash

QUANTILES = [.10, .25, .50, .75, .90, .99]
# The depth bins have only p50 and p90. The flat upper tail is a stated sampling bound.
DEPTH_BINS = [(50000, 406, 6215), (100000, 197, 5344), (200000, 98, 2329), (1048577, 222, 1448)]
OUTPUT = [54, 113, 229, 530, 1135, 4506]
PAUSE = [.2, 3.8, 7.8, 17.9, 34.8, 340]
SYSTEM = 'Read synthetic tool results. Reply with decimal digits only.'
NATIVE_CONTEXT = 262144
DEPTH_MARGIN = 2048


def sample(points, u):
    for (q0, v0), (q1, v1) in zip(points, points[1:]):
        if u <= q1:
            t = (u - q0) / (q1 - q0)
            return math.exp(math.log(v0) + t * math.log(v1 / v0)) if q0 >= .5 and v0 > 0 else v0 + t * (v1 - v0)
    return points[-1][1]


def scenario(depth, turns=40, seed=1, max_new=10875, max_output=4506, pause_cap=340, framing=16):
    rng = random.Random(seed)
    rows = []
    for turn in range(1, turns + 1):
        row = sample_turn(rng, depth, turn, seed, max_new, max_output, pause_cap)
        rows.append(row)
        depth += row['input_tokens'] + row['output_tokens'] + framing
    return rows


def depth_plan(depth, turns, seed, max_new, max_output, pause_cap, framing, capacity=NATIVE_CONTEXT):
    def plan(start):
        rows = scenario(start, turns, seed, max_new, max_output, pause_cap, framing)
        growth = sum(r['input_tokens'] + r['output_tokens'] + framing for r in rows)
        return dict(d0=start, growth=growth, final_depth=start + growth, margin=DEPTH_MARGIN,
                    capacity=capacity, framing_tokens=framing, sequence=rows)
    if depth == 'deep':
        require(plan(100000)['final_depth'] + DEPTH_MARGIN <= capacity,
                'sampled sequence cannot fit at D0=100000 with 2048 margin; no turns truncated')
        for start in range((capacity - DEPTH_MARGIN) // 1024 * 1024, 0, -1024):
            result = plan(start)
            if result['final_depth'] + DEPTH_MARGIN <= capacity:
                return result
        raise Refusal('no positive deep start fits; no turns truncated')
    result = plan(depth)
    require(result['final_depth'] + DEPTH_MARGIN <= capacity,
            'sampled sequence cannot fit at D0=' + str(depth) + ' with 2048 margin; no turns truncated')
    return result


def template_framing(app):
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=' x'),
                dict(role='assistant', content='0'), dict(role='user', content=' x')]
    body = dict(messages=messages, chat_template_kwargs=dict(enable_thinking=False))
    text = app.renderer.render(body)
    ids = app.renderer.encode(text, initial=True)
    tail, _, _, _ = app.renderer.append(body, text, ids, 3, [])
    require(len(app.renderer.encode(' x')) == 1,
            'fixture text must encode to one token per repetition')
    return len(tail) - 1


def sample_turn(rng, depth, turn, seed, max_new, max_output, pause_cap):
    _, median, p90 = next((row for row in DEPTH_BINS if depth < row[0]), DEPTH_BINS[-1])
    new = round(sample([(0, 0), (.5, median), (.9, p90)], rng.random()))
    output = max(1, round(sample([(0, 0)] + list(zip(QUANTILES, OUTPUT)), rng.random())))
    pause = sample([(0, 0)] + list(zip(QUANTILES, PAUSE)), rng.random())
    return dict(turn=turn, conditioning_depth=depth, input_tokens=min(new, max_new),
                output_tokens=min(output, max_output), pause_seconds=min(pause, pause_cap),
                seed=seed + turn, sampled_input_tokens=new, sampled_output_tokens=output,
                sampled_pause_seconds=pause)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


async def relay(port, upstream):
    parsed = urllib.parse.urlsplit(upstream)
    require(parsed.scheme == 'http' and parsed.hostname and parsed.path in ('', '/'), 'relay needs an HTTP origin URL')

    async def connect(reader, writer):
        peer = None
        try:
            incoming, peer = await asyncio.open_connection(parsed.hostname, parsed.port or 80)

            async def copy(source, target):
                while data := await source.read(65536):
                    target.write(data)
                    await target.drain()
                target.close()

            await asyncio.gather(copy(reader, peer), copy(incoming, writer))
        except (OSError, ConnectionError):
            pass
        finally:
            writer.close()
            if peer:
                peer.close()

    async with await asyncio.start_server(connect, '127.0.0.1', port) as server:
        await server.serve_forever()


class Events:
    def __init__(self, state_dir=None):
        self.rows = []
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.state_dir = state_dir

    def add(self, **row):
        with self.lock:
            self.rows.append(dict(at_seconds=time.monotonic() - self.started, **row))

    def copy(self):
        with self.lock:
            return list(self.rows)


class MeasuredNode(Node):
    def __init__(self, *args, events, session, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = events
        self.session = session
        self.source = source

    def request(self, method, path, data=None, headers=None, binary=False):
        started = time.monotonic()
        encoded = json.dumps(data).encode() if isinstance(data, (dict, list)) else data or b''
        row = dict(kind='protocol', session=self.session, source=self.source, method=method, path=path, upload_body_bytes=len(encoded),
                   response_body_bytes=0, status='failed')
        if isinstance(data, dict) and path.endswith('/extend'):
            row.update(n0=data['n0'], n1=data['n0'] + len(data['tokens']),
                       catchup=not data.get('return_delta', False), prefix_hash_n0=data['prefix_hash_n0'])
        try:
            result = super().request(method, path, data, headers, binary)
            row.update(status='ok', response_body_bytes=self.last_bytes)
            if row.get('catchup'):
                require(self.last_prefill_ms is not None, 'node catch-up evaluation timing missing')
                row['node_prefill_seconds'] = float(self.last_prefill_ms) / 1000
            if method == 'GET' and isinstance(result, dict) and 'prefix_hash' in result:
                row.update(n=result['n'], prefix_hash=result['prefix_hash'])
            if self.events.state_dir is not None and path.endswith('/extend') and data.get('return_delta'):
                snapshot = Node(self.url, self.token, self.timeout).request('GET', path[:-6] + 'snapshot', binary=True)
                require(len(snapshot) >= 12 and int.from_bytes(snapshot[8:12], 'little') == row['n1'],
                        'node snapshot moved past the captured import boundary')
                require(shutil.disk_usage(self.events.state_dir).free > 28e9 + len(snapshot) + len(result),
                        'disk floor reached before node state capture')
                stem = self.events.state_dir / uuid.uuid4().hex
                for key, payload in (('node_snapshot', snapshot), ('node_delta', result)):
                    target = stem.with_suffix('.' + key)
                    target.write_bytes(payload)
                    target.chmod(0o600)
                    row[key] = dict(path=str(target), sha256=sha256(target))
            return result
        except Exception as error:
            row['error_type'] = type(error).__name__
            raise
        finally:
            row['seconds'] = time.monotonic() - started
            self.events.add(**row)


class MeasuredPolicy:
    def __init__(self, policy, current):
        self.policy, self.current = policy, current

    def choose(self, delta, depth, previous=False, link_rate=None, server_factor=1):
        result = self.policy.choose(delta, depth, previous, link_rate, server_factor)
        self.current.decision = dict(inputs=dict(delta=delta, depth=depth, previous=previous,
            link_rate=link_rate, server_factor=server_factor), result=result.copy())
        return result


class MeasuredProxy(Proxy):
    def __init__(self, config, database, events, checkpoints):
        super().__init__(config, database)
        self.events, self.checkpoints = events, checkpoints
        self.current = threading.local()
        self.policy_config = asdict(self.policy)
        self.policy = MeasuredPolicy(self.policy, self.current)
        self.phase = {}
        self.saved = {}

    def remote(self, client):
        return MeasuredNode(self.prefill, client['prefill_token'], self.timeout, events=self.events,
                            session=getattr(self.current, 'session', None), source=getattr(self.current, 'origin', 'background'))

    def sync(self, session, client, recreate=True):
        self.current.session = session['id']
        started = time.monotonic()
        n = session['remote_n']
        status = 'failed'
        try:
            result = super().sync(session, client, recreate)
            status = 'ok'
            return result
        finally:
            self.events.add(kind='sync', session=session['id'], remote_id=session['remote_id'],
                acknowledged_before=n, acknowledged_after=session['remote_n'], journal_tokens=len(session['tokens']),
                journal_hash=prefix_hash(session['tokens']), seconds=time.monotonic() - started, status=status)

    def prepare_session(self, client, body, session, tokens):
        self.current.session = session['id']
        self.current.origin = 'critical'
        try:
            result = super().prepare_session(client, body, session, tokens)
        finally:
            self.current.origin = 'background'
        result[0]['_bench']['decision'] = self.current.decision
        return result

    def finish(self, session, prompt):
        phase = self.phase.copy()
        turn = phase.get('turn', 0)
        if turn in self.checkpoints:
            complete = prompt + session['_capture'].native['tokens']
            info = self.decode.request('GET', f"/slots/{session['slot']}/info")
            require(info['tokens'] == complete[:-1], 'checkpoint needs the unreconciled last sampled token')
            filename = phase['arm'] + '-checkpoint-' + str(turn) + '.bin'
            started = time.monotonic()
            result = self.decode.request('POST', f"/slots/{session['slot']}?action=save", dict(filename=filename))
            require(result['n_saved'] == len(complete) - 1, 'checkpoint save token count mismatch')
            self.saved[(phase['arm'], turn)] = dict(path=str(self.slot_path / filename), tokens=complete,
                save_seconds=time.monotonic() - started, sha256=sha256(self.slot_path / filename))
        timings = session['_capture'].native.get('timings') or session['_capture'].timings or {}
        started = time.monotonic()
        super().finish(session, prompt)
        session['_bench'].update(local_prompt_ms=timings.get('prompt_ms'),
                                 reconciliation_ms=(time.monotonic() - started) * 1000)


@contextmanager
def proxy_server(app):
    server = http_server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.app = app
    threads = [threading.Thread(target=server.serve_forever, daemon=True),
               threading.Thread(target=app.replay_loop, daemon=True)]
    for thread in threads:
        thread.start()
    try:
        yield 'http://127.0.0.1:' + str(server.server_port)
    finally:
        app.stop.set()
        server.shutdown()
        server.server_close()
        for thread in threads:
            thread.join()
        remaining = []
        for session in app.store.pending():
            app.store.benchmark_mode(session['id'], 'closed')
            ids = [session['remote_id']] + (session.get('journal') or {}).get('retired_remote_ids', [])
            client = next(c for c in app.clients.values() if c['id'] == session['client'])
            for sid in set(value for value in ids if value):
                try:
                    Node(app.cleanup_prefill, client['prefill_token'], app.timeout).request('DELETE', '/v1/sessions/' + sid)
                except Exception as error:
                    if getattr(error, 'status', None) != 404:
                        remaining.append(sid)
        app.events.add(kind='cleanup', remaining_remote_sessions=remaining)
        app.store.close()
        if remaining and sys.exc_info()[0] is None:
            raise Refusal('remote session cleanup failed; see cleanup event')


def decoder_command(client, config, port, slots, token_file):
    return [Path(client['build_dir']) / 'bin/llama-server', '-m', client['model_path'], '--host', '127.0.0.1',
        '--port', port, '-ngl', 0 if client['backend'] == 'cpu' else 999, '-t', 2, '-tb', 2,
        '-c', client['context'], '-np', 1, '-b', client.get('batch', 256), '-ub', client.get('ubatch', 256),
        '-ctk', 'q8_0', '-ctv', 'q8_0', '-fa', 'on', '--jinja', '--slots', '--spec-type', 'none',
        '--no-context-shift', '--cache-ram', 0, '--no-cache-idle-slots', '--ctx-checkpoints', 0,
        '--checkpoint-min-step', 0, '--slot-save-path', str(slots) + '/', '--api-key-file', token_file]


def base_messages(app, depth):
    body = dict(messages=[dict(role='system', content=SYSTEM), dict(role='user', content='')],
                chat_template_kwargs=dict(enable_thinking=False))
    overhead = len(app.renderer.encode(app.renderer.render(body), initial=True))
    require(depth > overhead + 1, 'base depth too small for the template')
    body['messages'][1]['content'] = ' x' * (depth - overhead - 1)
    require(len(app.renderer.encode(app.renderer.render(body), initial=True)) == depth - 1,
            'existing text fixture does not produce the requested base depth')
    return body['messages']


def route_preflight(policy, plan, disconnect_turn=0, disconnect_turns=1):
    previous = True  # the base is explicitly offloaded
    rows = []
    for step in plan['sequence']:
        delta = step['input_tokens'] + plan['framing_tokens'] - 1
        choice = policy.choose(delta, step['conditioning_depth'], previous, policy.link_rate)
        outage = bool(disconnect_turn and disconnect_turn <= step['turn'] < disconnect_turn + disconnect_turns)
        rows.append(dict(turn=step['turn'], outage=outage, **choice))
        previous = choice['offload'] and not outage
    counts = dict(policy_local=sum(not r['offload'] and not r['outage'] for r in rows),
                  policy_offload=sum(r['offload'] and not r['outage'] for r in rows))
    require(all(counts.values()), 'route preflight refuses degenerate connected policy mix before any arm: ' + str(counts))
    return dict(rows=rows, counts=counts, link_model='constant configured link; live EMA can change routes')


def session_catchup(events, sid):
    return [r for r in events if r.get('session') == sid and r.get('catchup') and r['status'] == 'ok']


def route_counts(calls, events, disconnect):
    rows = [r for r in calls if r['arm'] == 'hybrid' and r['stage'] == 'turn']
    def connected(row):
        return not (disconnect['turn'] and disconnect['turn'] <= row['turn'] < disconnect['turn'] + disconnect['turns'])
    local = [r for r in rows if connected(r) and not r['decision']['result']['offload'] and not r['fallback_reason'] and r['path'] == 'local']
    caught = []
    for row in local:
        ranges = session_catchup(events, row['session_id'])
        # A prompt evaluated locally while connected must later be acknowledged by the node while connected.
        end = row['meter']['prompt_depth']
        if any(e['n0'] < end and e['n1'] > row['depth_before'] and
               not any(d['kind'] == 'disconnect' and row['at_seconds'] <= d['at_seconds'] <= e['at_seconds'] for d in events)
               for e in ranges):
            caught.append(row['turn'])
    fallbacks = {}
    for row in rows:
        if row['fallback_reason']:
            reason = row['fallback_reason']
            fallbacks[reason] = fallbacks.get(reason, 0) + 1
    return dict(connected_policy_local=len(local),
        connected_policy_offload=sum(connected(r) and r['decision']['result']['offload'] and r['path'] == 'offload' and not r['fallback_reason'] for r in rows),
        fallbacks=fallbacks, connected_local_catchup_turns=caught,
        background_catchup_events=sum(e.get('catchup') is True and e['status'] == 'ok' and e.get('source') == 'background' for e in events))


def replay_schedules(trace, turn, depth):
    client = trace.get('client', {})
    require(client.get('batch', 256) == 256 and client.get('ubatch', 256) == 256 and
            client.get('threads', 2) == 2 and client.get('threads_batch', 2) == 2,
            'factorial replay requires primary local 256/256 and two threads')
    calls = [r for r in trace['calls'] if r['arm'] == 'hybrid' and r['turn'] <= turn]
    segments = []
    def add_segment(n0, n1, kind):
        if n1 > n0:
            segments.append(dict(n0=n0, n1=n1, kind=kind))
    n = 0
    for row in calls:
        require(row['depth_before'] == n, 'replay call boundaries do not tile')
        prompt = row['meter']['prompt_depth']
        add_segment(n, prompt, 'prompt')
        add_segment(prompt, row['depth_after'], 'generated')
        n = row['depth_after']
    require(n == depth, 'replay checkpoint depth mismatch')
    imports = [c for row in calls for c in row['meter']['chunks'] if c['status'] == 'imported']
    require(imports, 'replay has no imported boundary')
    last_import = imports[-1]['n1']
    sid = calls[0].get('session_id')
    if sid is None:
        sid = next(r['session'] for r in trace['events'] if r.get('catchup') is False and r.get('n0') == 0 and r['status'] == 'ok')
    node = []
    n = 0
    for event in trace['events']:
        if (event.get('session') != sid or 'n0' not in event or event['status'] != 'ok' or event['n1'] > last_import):
            continue
        require(event['n0'] == n, 'node replay ranges do not tile')
        for start in range(n, event['n1'], 256):
            node.append(dict(n0=start, n1=min(start + 256, event['n1']), logits=True))
        n = event['n1']
    require(n == last_import, 'node replay missing last import')
    def local(segment):
        a, b = segment['n0'], segment['n1']
        chunk = 256 if segment['kind'] == 'prompt' else 1
        return [dict(n0=i, n1=min(i + chunk, b), logits=(min(i + chunk, b) == b or chunk == 1)) for i in range(a, b, chunk)]
    schedules = {'A': [], 'B1': [], 'B2': [], 'B': list(node)}
    for segment in segments:
        schedules['A'].extend(local(segment))
        for name, kind in (('B1', 'generated'), ('B2', 'prompt')):
            if segment['kind'] != kind or segment['n0'] >= last_import:
                schedules[name].extend(local(segment))
                continue
            for batch in node:
                a, b = max(segment['n0'], batch['n0']), min(segment['n1'], batch['n1'])
                if b > a:
                    schedules[name].append(dict(n0=a, n1=b, logits=kind == 'generated' or b == batch['n1']))
            if segment['n1'] > last_import:
                schedules[name].extend(local(dict(segment, n0=last_import)))
        if segment['n1'] > last_import:
            schedules['B'].extend(local(dict(segment, n0=max(last_import, segment['n0']))))
    for schedule in schedules.values():
        n = 0
        for step in schedule:
            require(step['n0'] == n and 0 < step['n1'] - n <= 256, 'factorial replay does not tile')
            n = step['n1']
        require(n == depth and schedule[-1]['logits'], 'incomplete factorial replay')
    return schedules


def checkpoint(app, client, root, turn, heterogeneous, trace):
    saved = [app.saved[(arm, turn)] for arm in ('hybrid', 'local', 'local_partition')]
    hybrid, local, partition = saved
    require(hybrid['tokens'] == local['tokens'] == partition['tokens'], 'checkpoint journal mismatch between arms')
    scratch = root / ('check-' + str(turn))
    scratch.mkdir(mode=0o700)
    fixture = scratch / 'input.json'
    result = scratch / 'result.json'
    source = dict(tokens=local['tokens'], k=64, max_ctx=client['context'], acceptance_rule='session_append_replay_exact',
        model_sha256=client['model_sha256'], backend=client['backend'], heterogeneous=heterogeneous,
        snapshots=[r['path'] for r in saved], snapshot_sha256=[r['sha256'] for r in saved],
        replay=replay_schedules(dict(trace, events=app.events.copy()), turn, len(local['tokens'])))
    if app.events.state_dir is not None:
        source['probes'] = state_probe_plans(source, dict(trace, events=app.events.copy()), turn)
    private_json(fixture, source)
    command = [Path(client['build_dir']) / 'bin/llama-hybrid-bench-check', client['model_path'], fixture,
               hybrid['path'], result, 0 if client['backend'] == 'cpu' else 999]
    process = start_process(command, scratch, 'checker')
    try:
        process.wait()
    finally:
        stop_process(process)
    require(result.is_file(), 'checkpoint checker produced no metrics; inspect its log')
    row = json.loads(result.read_text())
    require(process.returncode == row['exit_code'] and process.returncode in (0, 3, 5), 'checkpoint checker failed')
    require(row.get('status_rule') == 'session_append_replay_exact', 'checkpoint replay gate missing')
    validate_checkpoint(row, heterogeneous, len(local['tokens']))
    row.update(turn=turn, checker_exit_code=process.returncode,
               snapshot_sha256=[r['sha256'] for r in saved], journal_hash=prefix_hash(local['tokens']))
    for item in saved:
        Path(item['path']).unlink()
    return row


def state_probe_plans(source, trace, turn):
    depth = len(source['tokens'])
    calls = [r for r in trace['calls'] if r['arm'] == 'hybrid' and r['turn'] <= turn]
    sid = calls[0]['session_id']
    last = [c['n1'] for r in calls for c in r['meter']['chunks'] if c['status'] == 'imported'][-1]
    events = [e for e in trace['events'] if e.get('session') == sid and e.get('status') == 'ok' and
              'n0' in e and e['n1'] <= last]
    require(events and events[-1]['n1'] == last and 'node_snapshot' in events[-1], 'missing live node state')

    def decode(name, start, end):
        schedule = [dict(s, n0=max(start, s['n0']), n1=min(end, s['n1'])) for s in source['replay'][name]
                    if min(end, s['n1']) > max(start, s['n0'])]
        require(schedule and schedule[0]['n0'] == start and schedule[-1]['n1'] == end, 'state probe schedule gap')
        return dict(decode=schedule)

    node = events[-1]['node_snapshot']
    snapshot = dict(snapshot=node['path'], sha256=node['sha256'])
    compare = dict(compare_snapshot=node['path'], sha256=node['sha256'])
    tail = decode('B', last, depth)
    restore, chain, n, imported = [], [], 0, 0
    for event in events:
        require(event['n0'] == n, 'state probe node events do not tile')
        restore.extend([decode('B', n, event['n1']), dict(roundtrip='native')])
        n = event['n1']
        if not event['catchup']:
            require('node_delta' in event, 'missing live node delta')
            if imported < event['n0']:
                chain.append(decode('A', imported, event['n0']))
            entry = event['node_delta']
            chain.append(dict(delta=entry['path'], sha256=entry['sha256']))
            imported = n
    require(imported == last, 'state probe delta chain incomplete')
    plans = [('NODE', [snapshot, tail]), ('RESTORE', restore + [compare, tail]),
             ('BSTATE', [decode('B', 0, last), compare, tail]),
             ('NODE_RANGE', [snapshot, dict(roundtrip='range'), tail]), ('CHAIN', chain + [tail])]
    for component in ('attention', 'recurrent'):
        plans.append(('CHAIN_' + component.upper(), chain + [dict(snapshot, replace=component), tail]))
    for mode in ('native', 'range'):
        plans.append(('OWN_' + mode.upper(), [decode('A', 0, depth - 1), dict(roundtrip=mode), decode('A', depth - 1, depth)]))
    return [dict(name=name, steps=steps) for name, steps in plans]


def validate_raw_positions(row, depth):
    raw = row['raw_logits']
    vocab = raw['vocab']
    require(raw['dtype'] == 'float32-le' and type(vocab) is int and 1 <= vocab <= 1048576,
            'invalid raw logits shape')
    files = raw['files']
    pairs = {'hybrid': ('L', 'H')}
    if 'partition' in row:
        pairs['partition'] = ('L', 'P')
    if 'factorial' in row:
        for name in row['factorial']:
            left, right = name.split('-')
            pairs[name] = (left, right)
    for name, probe in row.get('probes', {}).items():
        for reference in ('A', 'B', 'H'):
            if reference in probe:
                pairs[name + ':' + reference] = (reference, name)
    handles = {}
    metrics = {name: [] for name in pairs}
    try:
        for name in set(n for pair in pairs.values() for n in pair):
            entry = files[name]
            path = Path(entry['path'])
            require(path.stat().st_size == 64 * vocab * 4 and sha256(path) == entry['sha256'],
                    'raw checkpoint logits size or checksum mismatch')
            handles[name] = path.open('rb')
        for i in range(64):
            vectors, normalizers, argmax = {}, {}, {}
            for name, handle in handles.items():
                values = array('f')
                values.frombytes(handle.read(vocab * 4))
                if sys.byteorder != 'little':
                    values.byteswap()
                require(len(values) == vocab and all(math.isfinite(v) for v in values), 'nonfinite raw logits')
                peak = max(values)
                vectors[name] = values
                normalizers[name] = peak + math.log(math.fsum(math.exp(v - peak) for v in values))
                argmax[name] = values.index(peak)
            for name, (left, right) in pairs.items():
                p, q = vectors[left], vectors[right]
                pz, qz = normalizers[left], normalizers[right]
                exact = p.tobytes() == q.tobytes()
                kl = 0 if exact else max(0, math.fsum(math.exp(x - pz) * (x - pz - y + qz) for x, y in zip(p, q)))
                metrics[name].append(dict(position=depth + i, exact=exact, top1_agreed=argmax[left] == argmax[right],
                    kl_nats=kl, max_abs_logit_diff=max(abs(x - y) for x, y in zip(p, q))))
    finally:
        for handle in handles.values():
            handle.close()
    for name, positions in metrics.items():
        if ':' in name:
            probe, reference = name.split(':')
            measured = row['probes'][probe][reference]
        else:
            measured = row if name == 'hybrid' else row['partition'] if name == 'partition' else row['factorial'][name]
        require(len(measured['positions']) == 64, 'invalid checkpoint position count')
        for got, expected in zip(measured['positions'], positions):
            require(type(got['position']) is int and got['position'] == expected['position'] and
                    type(got['top1_agreed']) is bool and got['top1_agreed'] == expected['top1_agreed'] and
                    type(got['exact']) is bool and got['exact'] == expected['exact'] and
                    math.isclose(got['kl_nats'], expected['kl_nats'], rel_tol=1e-8, abs_tol=1e-12) and
                    got['max_abs_logit_diff'] == expected['max_abs_logit_diff'], 'checkpoint positions contradict raw logits')
        values = [p['kl_nats'] for p in positions]
        require(math.isclose(measured['mean_kl_nats'], math.fsum(values) / 64, rel_tol=1e-8, abs_tol=1e-12) and
                math.isclose(measured['p95_kl_nats'], sorted(values)[60], rel_tol=1e-8, abs_tol=1e-12) and
                measured['top1_agreed'] == sum(p['top1_agreed'] for p in positions) and
                measured['exact'] == all(p['exact'] for p in positions) and
                measured['max_abs_logit_diff'] == max(p['max_abs_logit_diff'] for p in positions),
                'checkpoint summary contradicts raw logits')


def append_replay_controls(row):
    controls = {}
    pairs = [('NODE', 'B'), ('RESTORE', 'B'), ('BSTATE', 'B'), ('NODE_RANGE', 'B'),
             ('OWN_NATIVE', 'A'), ('OWN_RANGE', 'A'), ('CHAIN', 'H'),
             ('CHAIN_ATTENTION', 'B'), ('CHAIN_RECURRENT', 'H')]
    require(set(name for name, _ in pairs) <= row.get('probes', {}).keys() and
            'A-L' in row.get('factorial', {}), 'missing append replay controls')

    def exact(metric):
        return (metric['exact'] is True and metric['max_abs_logit_diff'] == 0 and metric['mean_kl_nats'] == 0 and
                metric['top1_agreed'] == 64 and metric['free_running_first_divergence'] == -1)

    for name, reference in pairs:
        probe = row['probes'][name]
        equal = exact(probe[reference])
        states = probe['roundtrips']
        if name not in ('NODE', 'CHAIN'):
            equal = equal and bool(states)
            for state in states:
                if name in ('RESTORE', 'BSTATE'):
                    equal = equal and state.get('bytes_equal', state.get('snapshot_bytes_equal')) is True
                elif name in ('CHAIN_ATTENTION', 'CHAIN_RECURRENT'):
                    equal = equal and state['recurrent_equal'] is True
                else:
                    equal = equal and state['bytes_equal'] is True
        controls[name + '-' + reference] = equal
    controls['A-L'] = exact(row['factorial']['A-L'])
    return controls


def validate_checkpoint(row, heterogeneous, depth):
    require(row['depth'] == depth and row['k'] == 64 and type(row['exact']) is bool and
            type(row['top1_agreed']) is int and 0 <= row['top1_agreed'] <= 64 and
            math.isfinite(row['mean_kl_nats']) and row['mean_kl_nats'] >= 0 and
            math.isfinite(row['max_abs_logit_diff']) and row['max_abs_logit_diff'] >= 0, 'invalid checkpoint metrics')
    require(not row['exact'] or (row['top1_agreed'] == 64 and row['mean_kl_nats'] == 0 and row['max_abs_logit_diff'] == 0),
            'checkpoint exactness contradicts measured distributions')
    validate_raw_positions(row, depth)
    require(type(row['free_running_first_divergence']) is int and -1 <= row['free_running_first_divergence'] < 64,
            'invalid free-running result')
    accepted = row['exact']
    if row.get('status_rule') in ('session_mean_kl_zero', 'session_append_replay_exact'):
        band = row['partition']['mean_kl_nats']
        p95 = row['partition']['p95_kl_nats']
        require(row['local_partition_mean_kl_nats'] == band and row['local_partition_p95_kl_nats'] == p95 and
                row['kl_limit'] == (0 if row['status_rule'] == 'session_mean_kl_zero' else None),
                'invalid local partition band or KL limit')
        ratio = row['kl_to_partition_ratio']
        ratio_status = 'finite' if band > 0 else 'both_zero' if row['mean_kl_nats'] == 0 else 'zero_noise_band'
        require(row['kl_to_partition_ratio_status'] == ratio_status and
                (math.isclose(ratio, row['mean_kl_nats'] / band, abs_tol=1e-12) if band > 0 else ratio is None),
                'invalid KL to partition ratio')
        if row['status_rule'] == 'session_mean_kl_zero':
            accepted = row['mean_kl_nats'] <= row['kl_limit'] and row['top1_agreed'] >= 63 and row['free_running_first_divergence'] == -1
    if row.get('status_rule') == 'session_append_replay_exact':
        controls = append_replay_controls(row)
        require(row['gate_controls'] == controls and row['kl_limit'] is None, 'checkpoint replay gate contradicts controls')
        accepted = (all(controls.values()) and row['top1_agreed'] >= 63 and
                    (row['free_running_first_divergence'] == -1 or row['free_running_first_divergence'] >= 8))
    status, code = ('heterogeneous', 5) if heterogeneous else ('passed', 0) if accepted else ('failed', 3)
    require((row['status'], row['exit_code']) == (status, code), 'checkpoint verdict contradicts acceptance rule')


def turn_record(arm, step, meter, ttft, total, journal_hash, events, session_id):
    catchup = session_catchup(events, session_id)
    return dict(arm=arm, turn=step['turn'], requested=step, journal_hash=journal_hash, session_id=session_id,
        ttft_seconds=ttft, total_seconds=total, decision=meter['decision'],
        depth_before=meter['depth_before'], depth_after=meter['depth_after'], new_tokens=meter['new_tokens'],
        generated_tokens=meter['generated_tokens'], path=meter['path'], fallback_reason=meter['fallback_reason'],
        server_prefill_seconds=(meter['server_prefill_ms'] or 0) / 1000,
        transfer_seconds=meter['transfer_ms'] / 1000, import_seconds=meter['import_ms'] / 1000,
        remote_prepare_seconds=meter['remote_prepare_ms'] / 1000, wire_bytes=meter['wire_bytes'],
        local_prefill_seconds=(meter['local_prompt_ms'] or 0) / 1000 if meter['path'] == 'local' else 0,
        local_tail_seconds=(meter['local_prompt_ms'] or 0) / 1000 if meter['path'] == 'offload' else 0,
        reconciliation_seconds=meter['reconciliation_ms'] / 1000,
        catchup_tokens=sum(r['n1'] - r['n0'] for r in catchup), catchup_seconds=sum(r['seconds'] for r in catchup),
        catchup_node_seconds=sum(r.get('node_prefill_seconds', 0) for r in catchup),
        catchup_upload_bytes=sum(r['upload_body_bytes'] for r in catchup),
        decode_tokens_per_second=(meter['generated_tokens'] - 1) * 1000 / meter['decode_ms']
            if meter.get('decode_ms') and meter['generated_tokens'] > 1 else None,
        meter=meter)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--client-state', required=True, help='read-only launcher metadata; launches its own decoder')
    p.add_argument('--node-info')
    p.add_argument('--output', required=True, help='new private artifact directory')
    p.add_argument('--depth', type=lambda value: 'deep' if value == 'deep' else int(value), default=100000,
                   help='100000 or deep (derived native-capacity start); small integers for CPU controls')
    p.add_argument('--partition-batch', type=int, default=64)
    p.add_argument('--partition-ubatch', type=int, default=32)
    p.add_argument('--state-probes', action='store_true', help='capture private live node states and isolate serialization/import factors')
    p.add_argument('--turns', type=int, default=40)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--start', choices=['cold', 'pre-staged'], default='pre-staged')
    p.add_argument('--disconnect-turn', type=int, default=20, help='one-based, zero disables')
    p.add_argument('--disconnect-turns', type=int, default=3)
    p.add_argument('--max-new', type=int, default=10875)
    p.add_argument('--max-output', type=int, default=4506)
    p.add_argument('--pause-cap', type=float, default=340, help='zero for CPU smoke; changes the workload')
    p.add_argument('--policy', help='JSON Policy fields from calibration, never per-turn routing overrides')
    p.add_argument('--link-label', required=True, help='physical link description, no credentials or addresses')
    p.add_argument('--request-timeout', type=int, default=3600)
    p.add_argument('--disk-floor-gb', type=float, default=28)
    return p


def run(args):
    require((args.depth == 'deep' or 64 <= args.depth <= NATIVE_CONTEXT) and 1 <= args.turns <= 1000, 'invalid depth or turn count')
    require(1 <= args.partition_ubatch <= args.partition_batch <= 4096, 'invalid partition batch/ubatch')
    require(0 <= args.disconnect_turn <= args.turns and args.disconnect_turns >= 1 and
            (not args.disconnect_turn or args.disconnect_turn + args.disconnect_turns <= args.turns),
            'outage must leave at least one reconnected turn')
    require(0 <= args.pause_cap <= 340 and 0 <= args.max_new <= 10875 and 1 <= args.max_output <= 4506,
            'invalid workload caps')
    require(1 <= args.request_timeout <= 86400 and math.isfinite(args.disk_floor_gb) and args.disk_floor_gb >= 28,
            'invalid timeout or disk floor')
    state, root = Path(args.client_state).resolve(), Path(args.output).resolve()
    config, client = [read_private(state / name) for name in ('proxy-config.json', 'client-info.json')]
    node = json.loads(Path(args.node_info or state / 'node-info.json').read_text())
    require(not config.get('hetero_quant'), 'growing-session requires identical model files')
    validate_identity(client['model_sha256'], client, node['model_sha256'], node)
    require(sha256(client['model_path']) == client['model_sha256'], 'client model file changed')
    for binary in ('llama-server', 'llama-hybrid-bench-check'):
        require(sha256(Path(client['build_dir']) / 'bin' / binary) == client['binaries'][binary], 'client binary changed: ' + binary)
    require(not root.exists() and root.parent.is_dir(), 'output must be a new directory in an existing parent')
    capacity = min(client['context'], node['context'], NATIVE_CONTEXT)
    require(max(client['context'], node['context']) <= NATIVE_CONTEXT, 'native 262144 context only; no YaRN')
    require(args.depth != 'deep' or capacity == NATIVE_CONTEXT, 'deep arm needs native 262144 context on both roles')
    require(args.partition_batch != client.get('batch', 256) and args.partition_ubatch != client.get('ubatch', 256),
            'partition baseline needs different batch and ubatch')
    checkpoints = set(range(10, args.turns + 1, 10)) | {args.turns}
    geometry = config.get('delta_geometry', dict(kv_bytes=34816, state_bytes=157000000))
    size = geometry['state_bytes'] + (geometry['kv_bytes'] + 32) * client['context']
    floor = args.disk_floor_gb * 1e9
    raw_bytes = len(checkpoints) * 16 * 64 * client['model_layout']['vocab_size'] * 4
    require(shutil.disk_usage(root.parent).free > floor + (2 * len(checkpoints) + 6) * size + raw_bytes,
            'insufficient disk for checkpoint and base snapshots above the floor')
    root.mkdir(mode=0o700)
    (root / 'slots').mkdir(mode=0o700)
    state_dir = root / 'node-states'
    if state_dir is not None:
        state_dir.mkdir(mode=0o700)
    events = Events(state_dir)
    trace = dict(schema='hybrid-growing-session-v3', status='running', seed=args.seed, start=args.start,
        depth=args.depth, turns=args.turns, workload=[], calls=[], checkpoints=[],
        provenance=dict(**PROVENANCE, depth_bins=DEPTH_BINS, pause_seconds=PAUSE,
                        sampling='linear below p50, log-linear above; bounded at last published quantile',
                        table_sha256=sha256(Path(__file__).with_name('fixtures') / 'growing-provenance.json'),
                        table_selector='source=agents.all; new_in_by_depth/gap_seconds/output_per_call',
                        lower_half_limit='conditional input p25=median/2; measured global p25/median=0.20; tools differ from hybrid_bench p10 floor'),
        client=sanitized_info(client), node=sanitized_info(node), link=dict(label=args.link_label),
        disconnect=dict(turn=args.disconnect_turn, turns=args.disconnect_turns),
        driver=dict(source_sha256=sha256(__file__), proxy_sha256=sha256(Path(__file__).parents[1] / 'hybrid-proxy/proxy.py')),
        started_utc=datetime.now(timezone.utc).isoformat())

    def flush():
        trace['events'] = events.copy()
        temp = root / 'trace.json.tmp'
        with temp.open('w') as target:
            json.dump(trace, target, indent=2, allow_nan=False)
            target.write('\n')
            target.flush()
            os.fsync(target.fileno())
        temp.replace(root / 'trace.json')

    decoder = bridge = None
    app = None
    base = {}
    try:
        config = dict(config, benchmark=True, slots=1, slot_path=str(root / 'slots'),
                      decode_url='http://127.0.0.1:' + str(free_port()),
                      prefill_timeout=args.request_timeout, decode_timeout=args.request_timeout)
        if args.policy:
            config['policy'] = json.loads(Path(args.policy).read_text())
        token_file = root / 'decode-token'
        token_file.write_text(config['decode_token'])
        token_file.chmod(0o600)
        decoder = start_process(decoder_command(client, config, urllib.parse.urlsplit(config['decode_url']).port,
                                                root / 'slots', token_file), root, 'decode')
        wait_health(config['decode_url'], '/health', decoder, timeout=args.request_timeout, token=config['decode_token'])
        upstream = config['prefill_url']
        bridge_port = free_port()
        bridge_url = 'http://127.0.0.1:' + str(bridge_port)
        generation = 0

        def connect():
            nonlocal generation
            generation += 1
            process = start_process([sys.executable, Path(__file__).resolve(), 'relay', str(bridge_port), upstream],
                                    root, 'relay-' + str(generation))
            try:
                health = wait_health(bridge_url, '/v1/health', process, timeout=30)
                require(health['model_sha256'] == client['model_sha256'], 'live node model changed')
                return process
            except BaseException:
                stop_process(process)
                raise

        bridge = connect()
        config['prefill_url'] = bridge_url
        if node.get('bench_link'):
            trace['link'].update(link_probe(config))
        config['clients'] = {k: dict(v) for k, v in config['clients'].items()}
        app = MeasuredProxy(config, str(root / 'proxy.sqlite'), events, checkpoints)
        app.cleanup_prefill = upstream
        plan = depth_plan(args.depth, args.turns, args.seed, args.max_new, args.max_output, args.pause_cap,
                          template_framing(app), capacity)
        args.depth = plan['d0']
        trace.update(depth=args.depth, depth_plan={k: v for k, v in plan.items() if k != 'sequence'},
                     workload=plan['sequence'], local_partition=dict(batch=args.partition_batch, ubatch=args.partition_ubatch))
        trace['policy'] = app.policy_config
        trace['route_preflight'] = route_preflight(Policy(**app.policy_config), plan, args.disconnect_turn, args.disconnect_turns)
        trace['engine'] = dict(extend_chunk_tokens=app.extend_chunk_tokens, delta_geometry=geometry,
            node=dict(batch=256, ubatch=256, threads=2, threads_batch=2),
            checker=dict(batch=256, ubatch=256, threads=2, threads_batch=2),
            generation=dict(temperature=0, ignore_eos=True, enable_thinking=False))
        trace['engine']['cpu_affinity'] = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
        flush()
        trace['heterogeneous'] = client['backend'] != node['backend']
        credential = config['client_credential']
        with proxy_server(app) as proxy_url:
            previous = {}
            cursors = {}
            def call(arm, sid, messages, step, initial=False):
                require(shutil.disk_usage(root).free > floor + size, 'disk guard reached before call')
                before = cursors.get(arm, len(events.copy()))
                app.phase = dict(arm=arm, turn=step['turn'])
                body = dict(messages=messages, max_tokens=step['output_tokens'], temperature=0,
                            seed=step['seed'], ignore_eos=True, chat_template_kwargs=dict(enable_thinking=False))
                mode = 'offload' if initial and arm == 'hybrid' else 'local' if arm == 'local_partition' else arm
                sid, message, meter, first, total = stream_chat(proxy_url, credential, body, mode, sid, args.request_timeout)
                verify_meter(meter, previous.get(arm, 0), mode)
                previous[arm] = meter['depth_after']
                current = app.store.get(sid)
                require(current['tokens'] == app.decode.request('GET', f"/slots/{current['slot']}/info")['tokens'],
                        'journal mismatch after call')
                require(meter['fallback_reason'] not in ('invalid_delta', 'http_409'), 'protocol mismatch refused offload')
                require(meter['generated_tokens'] == step['output_tokens'], 'generation length differs from scenario')
                captured = events.copy()
                row = turn_record(arm, step, meter, first, total, prefix_hash(current['tokens']), captured[before:], sid)
                cursors[arm] = len(captured)
                row['stage'] = 'start' if initial else 'turn'
                row['at_seconds'] = time.monotonic() - events.started
                row['ttft_other_seconds'] = None if first is None else first - sum(row[k] for k in
                    ('server_prefill_seconds', 'transfer_seconds', 'import_seconds', 'local_prefill_seconds', 'local_tail_seconds'))
                if (arm, step['turn']) in app.saved:
                    row['checkpoint_save_seconds'] = app.saved[(arm, step['turn'])]['save_seconds']
                trace['calls'].append(row)
                if arm != 'hybrid':
                    reference = next(r for r in trace['calls'] if r['arm'] == 'hybrid' and r['turn'] == step['turn'])
                    require(row['journal_hash'] == reference['journal_hash'], 'paired journal hashes differ')
                flush()
                return sid, messages + [message]

            initial = dict(turn=0, output_tokens=1, seed=args.seed)
            def stage_base(arm):
                sid, messages = call(arm, 'growing-' + uuid.uuid4().hex, base_messages(app, args.depth), initial, True)
                current = app.store.get(sid)
                filename = arm + '-base.bin'
                started = time.monotonic()
                app.decode.request('POST', f"/slots/{current['slot']}?action=save", dict(filename=filename))
                base[arm] = (sid, messages, filename)
                events.add(kind='pre_stage', arm=arm, journal_hash=prefix_hash(current['tokens']),
                           bytes=(app.slot_path / filename).stat().st_size, seconds=time.monotonic() - started)

            if args.start == 'pre-staged':
                for arm in ('hybrid', 'local'):
                    stage_base(arm)
            for arm in ('hybrid', 'local', 'local_partition'):
                if arm == 'local_partition':
                    stop_process(decoder)
                    alternate = dict(client, batch=args.partition_batch, ubatch=args.partition_ubatch)
                    decoder = start_process(decoder_command(alternate, config, urllib.parse.urlsplit(config['decode_url']).port,
                        root / 'slots', token_file), root, 'decode-partition')
                    wait_health(config['decode_url'], '/health', decoder, timeout=args.request_timeout, token=config['decode_token'])
                    if args.start == 'pre-staged':
                        stage_base(arm)
                if arm in base:
                    sid, messages, filename = base[arm]
                    with app.reserve(sid):
                        current = app.store.get(sid)
                        with app.slot_lease(current):
                            app.decode.request('POST', f"/slots/{current['slot']}?action=restore", dict(filename=filename))
                            app.verified(sid, True)
                    start = time.monotonic()
                    cursors[arm] = len(events.copy())
                else:
                    start = time.monotonic()
                    sid, messages = call(arm, 'growing-' + uuid.uuid4().hex, base_messages(app, args.depth), initial, True)
                reconnect_events = None
                steps = iter(trace['workload'])
                for step in steps:
                    require(previous[arm] == step['conditioning_depth'], 'actual depth differs from sampled plan')
                    require(previous[arm] + step['input_tokens'] + step['output_tokens'] + plan['framing_tokens'] + DEPTH_MARGIN <=
                            capacity, 'growing context exhausted; no clamping')
                    turn = step['turn']
                    if arm == 'hybrid' and turn == args.disconnect_turn:
                        events.add(kind='disconnect', turn=turn, pid=bridge.pid)
                        stop_process(bridge)
                        bridge = None
                        flush()
                    if arm == 'hybrid' and args.disconnect_turn and turn == args.disconnect_turn + args.disconnect_turns:
                        reconnect_events = len(events.copy())
                        started = time.monotonic()
                        bridge = connect()
                        events.add(kind='reconnect', turn=turn, pid=bridge.pid, seconds=time.monotonic() - started)
                    time.sleep(step['pause_seconds'])
                    sid, messages = call(arm, sid, messages + [dict(role='user', content=' x' * step['input_tokens'])], step)
                    row = trace['calls'][-1]
                    require(row['new_tokens'] == step['input_tokens'] + plan['framing_tokens'],
                            'actual template growth differs from sampled plan; no clamping')
                    if arm == 'hybrid' and args.disconnect_turn <= turn < args.disconnect_turn + args.disconnect_turns and args.disconnect_turn:
                        require(row['path'] == 'local', 'offload accepted during disconnection')
                    if reconnect_events is not None and row['path'] == 'offload':
                        recovered = events.copy()[reconnect_events:]
                        recovered = [r for r in recovered if r.get('session') == sid]
                        catchup = session_catchup(recovered, sid)
                        row['resync'] = dict(tokens=sum(r['n1'] - r['n0'] for r in catchup),
                            seconds=sum(r['seconds'] for r in recovered if r['kind'] == 'protocol' and r.get('catchup') is not False),
                            catchup_seconds=sum(r['seconds'] for r in catchup),
                            upload_body_bytes=sum(r['upload_body_bytes'] for r in catchup))
                        reconnect_events = None
                    if arm == 'local_partition' and turn in checkpoints:
                        result = checkpoint(app, client, root, turn, trace['heterogeneous'], trace)
                        trace['checkpoints'].append(result)
                        flush()
                events.add(kind='window', arm=arm, seconds=time.monotonic() - start)
                require(previous[arm] == plan['final_depth'], 'final depth differs from sampled plan')
                # Closing the replay marker precedes DELETE; no later background work may recreate the session.
                request = urllib.request.Request(proxy_url + '/v1/sessions/' + sid, method='DELETE',
                    headers={'Authorization': 'Bearer ' + credential})
                with urllib.request.urlopen(request, timeout=args.request_timeout):
                    pass
            counts = route_counts(trace['calls'], events.copy(), trace['disconnect'])
            trace['route_counts'] = counts
            trace['final_depth'] = previous
            require(counts['connected_policy_local'] and counts['connected_policy_offload'], 'scenario did not exercise both connected policy routes')
            require(counts['connected_local_catchup_turns'], 'no connected policy-local prompt catch-up')
            require(not args.disconnect_turn or any(r.get('resync') for r in trace['calls']),
                    'no verified offload after reconnection')
            trace['status'] = ('failed' if any(r['status'] == 'failed' for r in trace['checkpoints']) else
                               'heterogeneous' if trace['heterogeneous'] else 'passed')
    except BaseException as error:
        trace.update(status='aborted', error_type=type(error).__name__,
                     reason=str(error) if isinstance(error, Refusal) else 'runtime failure; inspect private logs')
        raise
    finally:
        if bridge is not None:
            stop_process(bridge)
        if decoder is not None:
            stop_process(decoder)
        trace['ended_utc'] = datetime.now(timezone.utc).isoformat()
        flush()
    return trace


def replay_from_trace(argv):
    p = argparse.ArgumentParser(description='Factorial replay from retained private check tokens and public trace boundaries')
    p.add_argument('--trace', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--turn', required=True, type=int)
    p.add_argument('--output', required=True)
    p.add_argument('--checker', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--gpu-layers', type=int, default=0)
    args = p.parse_args(argv)
    trace = json.loads(Path(args.trace).read_text())
    source = read_private(args.input)
    source = {key: source[key] for key in ('tokens', 'k', 'max_ctx', 'model_sha256', 'backend')}
    source['replay'] = replay_schedules(trace, args.turn, len(source['tokens']))
    root = Path(args.output).resolve()
    require(not root.exists() and root.parent.is_dir(), 'replay output must be a new directory')
    require(shutil.disk_usage(root.parent).free > 28e9 + 4 * 64 * trace['client'].get('model_layout', {}).get('vocab_size', 1048576) * 4,
            'insufficient disk for replay logits above floor')
    root.mkdir(mode=0o700)
    private_json(root / 'input.json', source)
    command = [Path(args.checker).resolve(), Path(args.model).resolve(), root / 'input.json', 'unused', root / 'result.json', args.gpu_layers]
    process = start_process(command, root, 'checker')
    try:
        code = process.wait()
    finally:
        stop_process(process)
    private_json(root / 'command.json', dict(command=[str(v) for v in command], pid=process.pid, exit_code=code,
        trace_sha256=sha256(args.trace), retained_input_sha256=sha256(args.input)))
    require(code == 0, 'factorial replay failed; see checker log')
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'replay':
        return replay_from_trace(sys.argv[2:])
    if len(sys.argv) == 4 and sys.argv[1] == 'relay':
        asyncio.run(relay(int(sys.argv[2]), sys.argv[3]))
        return 0
    args = parser().parse_args()
    try:
        trace = run(args)
        print(json.dumps(dict(status=trace['status'], trace=str(Path(args.output) / 'trace.json'))))
        return {'heterogeneous': 5, 'failed': 3, 'passed': 0}[trace['status']]
    except (Refusal, OSError, ValueError, KeyError) as error:
        print('refused: ' + (str(error) if isinstance(error, Refusal) else type(error).__name__), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

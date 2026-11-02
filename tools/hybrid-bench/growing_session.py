#!/usr/bin/env python3
"""A paired growing session through the real proxy, with live-state checkpoints."""
import argparse
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
from store import prefix_hash

QUANTILES = [.10, .25, .50, .75, .90, .99]
# The depth bins have only p50 and p90. The flat upper tail is a stated sampling bound.
DEPTH_BINS = [(50000, 406, 6215), (100000, 197, 5344), (200000, 98, 2329), (1048577, 222, 1448)]
OUTPUT = [54, 113, 229, 530, 1135, 4506]
PAUSE = [.2, 3.8, 7.8, 17.9, 34.8, 340]
SYSTEM = 'Read synthetic tool results. Reply with decimal digits only.'


def sample(points, u):
    for (q0, v0), (q1, v1) in zip(points, points[1:]):
        if u <= q1:
            t = (u - q0) / (q1 - q0)
            return math.exp(math.log(v0) + t * math.log(v1 / v0)) if q0 >= .5 and v0 > 0 else v0 + t * (v1 - v0)
    return points[-1][1]


def scenario(depth, turns=40, seed=1, max_new=10875, max_output=4506, pause_cap=340):
    rng = random.Random(seed)
    rows = []
    for turn in range(1, turns + 1):
        row = sample_turn(rng, depth, turn, seed, max_new, max_output, pause_cap)
        rows.append(row)
        # ChatML framing is measured at runtime; this estimate selects the next depth bin only.
        depth += row['input_tokens'] + row['output_tokens'] + 16
    return rows


def sample_turn(rng, depth, turn, seed, max_new, max_output, pause_cap):
    _, median, p90 = next(row for row in DEPTH_BINS if depth < row[0])
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
    def __init__(self):
        self.rows = []
        self.lock = threading.Lock()
        self.started = time.monotonic()

    def add(self, **row):
        with self.lock:
            self.rows.append(dict(at_seconds=time.monotonic() - self.started, **row))

    def copy(self):
        with self.lock:
            return list(self.rows)


class MeasuredNode(Node):
    def __init__(self, *args, events, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = events

    def request(self, method, path, data=None, headers=None, binary=False):
        started = time.monotonic()
        encoded = json.dumps(data).encode() if isinstance(data, (dict, list)) else data or b''
        row = dict(kind='protocol', method=method, path=path, upload_body_bytes=len(encoded),
                   response_body_bytes=0, status='failed')
        if isinstance(data, dict) and path.endswith('/extend'):
            row.update(n0=data['n0'], n1=data['n0'] + len(data['tokens']),
                       catchup=not data.get('return_delta', False), prefix_hash_n0=data['prefix_hash_n0'])
        try:
            result = super().request(method, path, data, headers, binary)
            row.update(status='ok', response_body_bytes=self.last_bytes)
            if method == 'GET' and isinstance(result, dict) and 'prefix_hash' in result:
                row.update(n=result['n'], prefix_hash=result['prefix_hash'])
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
        return MeasuredNode(self.prefill, client['prefill_token'], self.timeout, events=self.events)

    def sync(self, session, client, recreate=True):
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

    def prepare_session(self, *args):
        result = super().prepare_session(*args)
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


def checkpoint(app, client, root, turn, heterogeneous):
    hybrid, local = [app.saved[(arm, turn)] for arm in ('hybrid', 'local')]
    require(hybrid['tokens'] == local['tokens'], 'checkpoint journal mismatch between arms')
    scratch = root / ('check-' + str(turn))
    scratch.mkdir(mode=0o700)
    fixture = scratch / 'input.json'
    result = scratch / 'result.json'
    private_json(fixture, dict(tokens=local['tokens'], k=64, max_ctx=client['context'],
        model_sha256=client['model_sha256'], backend=client['backend'], heterogeneous=heterogeneous,
        snapshots=[hybrid['path'], local['path']], snapshot_sha256=[hybrid['sha256'], local['sha256']]))
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
    validate_checkpoint(row, heterogeneous, len(local['tokens']))
    row.update(turn=turn, checker_exit_code=process.returncode,
               snapshot_sha256=[hybrid['sha256'], local['sha256']])
    for saved in (hybrid, local):
        Path(saved['path']).unlink()
    return row


def validate_checkpoint(row, heterogeneous, depth):
    require(row['depth'] == depth and row['k'] == 64 and type(row['exact']) is bool and
            0 <= row['top1_agreed'] <= 64 and math.isfinite(row['mean_kl_nats']) and row['mean_kl_nats'] >= 0 and
            math.isfinite(row['max_abs_logit_diff']) and row['max_abs_logit_diff'] >= 0,
            'invalid checkpoint metrics')
    require(not row['exact'] or (row['top1_agreed'] == 64 and row['mean_kl_nats'] == 0 and row['max_abs_logit_diff'] == 0),
            'checkpoint exactness contradicts measured distributions')
    status, code = ('heterogeneous', 5) if heterogeneous else ('passed', 0) if row['exact'] else ('failed', 3)
    require((row['status'], row['exit_code']) == (status, code), 'checkpoint verdict weakened exactness')


def turn_record(arm, step, meter, ttft, total, journal_hash, events):
    catchup = [r for r in events if r.get('catchup') and r['status'] == 'ok']
    return dict(arm=arm, turn=step['turn'], requested=step, journal_hash=journal_hash,
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
        catchup_upload_bytes=sum(r['upload_body_bytes'] for r in catchup),
        decode_tokens_per_second=(meter['generated_tokens'] - 1) * 1000 / meter['decode_ms']
            if meter.get('decode_ms') and meter['generated_tokens'] > 1 else None,
        meter=meter)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--client-state', required=True, help='read-only launcher metadata; launches its own decoder')
    p.add_argument('--node-info')
    p.add_argument('--output', required=True, help='new private artifact directory')
    p.add_argument('--depth', type=int, default=100000)
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
    require(64 <= args.depth <= 1000000 and 1 <= args.turns <= 1000, 'invalid depth or turn count')
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
    workload = scenario(args.depth, args.turns, args.seed, args.max_new, args.max_output, args.pause_cap)
    require(args.depth + sum(r['input_tokens'] + r['output_tokens'] + 32 for r in workload) + 256 <=
            min(client['context'], node['context']), 'context cannot fit the full sequence and checkpoints; no clamping')
    checkpoints = set(range(10, args.turns + 1, 10)) | {args.turns}
    geometry = config.get('delta_geometry', dict(kv_bytes=34816, state_bytes=157000000))
    size = geometry['state_bytes'] + (geometry['kv_bytes'] + 32) * client['context']
    floor = args.disk_floor_gb * 1e9
    require(shutil.disk_usage(root.parent).free > floor + (len(checkpoints) + 4) * size,
            'insufficient disk for checkpoint and base snapshots above the floor')
    root.mkdir(mode=0o700)
    (root / 'slots').mkdir(mode=0o700)
    events = Events()
    trace = dict(schema='hybrid-growing-session-v1', status='running', seed=args.seed, start=args.start,
        depth=args.depth, turns=args.turns, workload=[], calls=[], checkpoints=[],
        provenance=dict(**PROVENANCE, depth_bins=DEPTH_BINS, pause_seconds=PAUSE,
                        sampling='linear below p50, log-linear above; bounded at last published quantile'),
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
        trace['policy'] = app.policy_config
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
                mode = 'offload' if initial and arm == 'hybrid' else arm
                sid, message, meter, first, total = stream_chat(proxy_url, credential, body, mode, sid, args.request_timeout)
                verify_meter(meter, previous.get(arm, 0), mode)
                previous[arm] = meter['depth_after']
                current = app.store.get(sid)
                require(current['tokens'] == app.decode.request('GET', f"/slots/{current['slot']}/info")['tokens'],
                        'journal mismatch after call')
                require(meter['fallback_reason'] not in ('invalid_delta', 'http_409'), 'protocol mismatch refused offload')
                require(meter['generated_tokens'] == step['output_tokens'], 'generation length differs from scenario')
                captured = events.copy()
                row = turn_record(arm, step, meter, first, total, prefix_hash(current['tokens']), captured[before:])
                cursors[arm] = len(captured)
                row['stage'] = 'start' if initial else 'turn'
                row['ttft_other_seconds'] = None if first is None else first - sum(row[k] for k in
                    ('server_prefill_seconds', 'transfer_seconds', 'import_seconds', 'local_prefill_seconds', 'local_tail_seconds'))
                if (arm, step['turn']) in app.saved:
                    row['checkpoint_save_seconds'] = app.saved[(arm, step['turn'])]['save_seconds']
                trace['calls'].append(row)
                flush()
                return sid, messages + [message]

            initial = dict(turn=0, output_tokens=1, seed=args.seed)
            if args.start == 'pre-staged':
                for arm in ('hybrid', 'local'):
                    sid, messages = call(arm, 'growing-' + uuid.uuid4().hex, base_messages(app, args.depth), initial, True)
                    current = app.store.get(sid)
                    filename = arm + '-base.bin'
                    started = time.monotonic()
                    app.decode.request('POST', f"/slots/{current['slot']}?action=save", dict(filename=filename))
                    base[arm] = (sid, messages, filename)
                    events.add(kind='pre_stage', arm=arm, journal_hash=prefix_hash(current['tokens']),
                               bytes=(app.slot_path / filename).stat().st_size, seconds=time.monotonic() - started)
            for arm in ('hybrid', 'local'):
                if arm in base:
                    sid, messages, filename = base[arm]
                    with app.reserve(sid):
                        current = app.store.get(sid)
                        with app.slot_lease(current):
                            app.decode.request('POST', f"/slots/{current['slot']}?action=restore", dict(filename=filename))
                            app.verified(sid, True)
                    start = time.monotonic()
                else:
                    start = time.monotonic()
                    sid, messages = call(arm, 'growing-' + uuid.uuid4().hex, base_messages(app, args.depth), initial, True)
                reconnect_events = None
                rng = random.Random(args.seed)
                steps = (sample_turn(rng, previous[arm], turn, args.seed, args.max_new, args.max_output, args.pause_cap)
                         for turn in range(1, args.turns + 1)) if arm == 'hybrid' else iter(trace['workload'])
                for step in steps:
                    if arm == 'hybrid':
                        trace['workload'].append(step)
                    require(previous[arm] + step['input_tokens'] + step['output_tokens'] + 256 <=
                            min(client['context'], node['context']), 'growing context exhausted; no clamping')
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
                    if arm == 'hybrid' and args.disconnect_turn <= turn < args.disconnect_turn + args.disconnect_turns and args.disconnect_turn:
                        require(row['path'] == 'local', 'offload accepted during disconnection')
                    if reconnect_events is not None and row['path'] == 'offload':
                        recovered = events.copy()[reconnect_events:]
                        catchup = [r for r in recovered if r.get('catchup') and r['status'] == 'ok']
                        row['resync'] = dict(tokens=sum(r['n1'] - r['n0'] for r in catchup),
                            seconds=sum(r['seconds'] for r in recovered if r['kind'] == 'protocol' and r.get('catchup') is not False),
                            catchup_seconds=sum(r['seconds'] for r in catchup),
                            upload_body_bytes=sum(r['upload_body_bytes'] for r in catchup))
                        reconnect_events = None
                    if arm == 'local' and turn in checkpoints:
                        result = checkpoint(app, client, root, turn, trace['heterogeneous'])
                        trace['checkpoints'].append(result)
                        flush()
                events.add(kind='window', arm=arm, seconds=time.monotonic() - start)
                # Closing the replay marker precedes DELETE; no later background work may recreate the session.
                request = urllib.request.Request(proxy_url + '/v1/sessions/' + sid, method='DELETE',
                    headers={'Authorization': 'Bearer ' + credential})
                with urllib.request.urlopen(request, timeout=args.request_timeout):
                    pass
            paths = {r['path'] for r in trace['calls'] if r['stage'] == 'turn' and r['arm'] == 'hybrid'}
            require(paths == {'local', 'offload'}, 'scenario did not exercise both policy routes')
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


def main():
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

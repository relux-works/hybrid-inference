#!/usr/bin/env python3
"""Opt-in CPU growing-session qualification with existing binaries and a local GGUF."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

from bench_common import model_identity, private_json, require, sha256
from growing_session import (Events, MeasuredNode, MeasuredProxy, base_messages, proxy_server, stop_process)
from hybrid_bench import stream_chat
from store import prefix_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hybrid'))
from test_limits import Node
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hybrid-proxy'))
from native_fixture import native_server


def checkpoint_controls(build, model, root):
    controls = root / 'checkpoint-controls'
    controls.mkdir(mode=0o700)
    with native_server(build / 'bin/llama-server', model, controls / 'decoder', checkpoints=0) as (decoder, slots):
        prompt = decoder.request('POST', '/tokenize', dict(content='The clear sky above the quiet meadow.' * 8, add_special=True))['tokens']
        completion = decoder.request('POST', '/completion', dict(prompt=prompt, n_predict=2, id_slot=0,
            cache_prompt=True, temperature=0, ignore_eos=True, return_tokens=True))
        tokens = prompt + completion['tokens']
        snapshot = slots / 'control.bin'
        decoder.request('POST', '/slots/0?action=save', dict(filename=snapshot.name))
        fixture = dict(tokens=tokens, k=64, max_ctx=4096, model_sha256=sha256(model), backend='cpu',
                       snapshots=[str(snapshot), str(snapshot)], snapshot_sha256=[sha256(snapshot)] * 2)
        schedule = [dict(n0=i, n1=min(i + 256, len(prompt)), logits=min(i + 256, len(prompt)) == len(prompt))
                    for i in range(0, len(prompt), 256)]
        schedule += [dict(n0=i, n1=i + 1, logits=True) for i in range(len(prompt), len(tokens))]
        replay = {name: schedule for name in ('A', 'B1', 'B2', 'B')}
        probes = [dict(name='OWN_' + mode.upper(), steps=[dict(decode=schedule[:-1]), dict(roundtrip=mode),
                   dict(decode=schedule[-1:])]) for mode in ('native', 'range')]
        probe_fixture = dict(fixture, replay=replay, probes=probes)
        cases = [('same_backend_checkpoint_exactness', fixture, 0, None),
                 ('own_state_roundtrips', probe_fixture, 0, None),
                 ('test_manifest_above_8MiB', fixture, 0, None),
                 ('test_manifest_above_64MiB_refuses', fixture, 2, 'check input exceeds limit'),
                 ('probe_incomplete_refusal', dict(probe_fixture, probes=[dict(name='INCOMPLETE', steps=[dict(decode=schedule[:-1])])]),
                  2, 'incomplete state probe journal or final logits'),
                 ('probe_duplicate_refusal', dict(probe_fixture, probes=[dict(probes[0], name='A')]), 2, 'duplicate state probe name'),
                 ('checksum_refusal', dict(fixture, snapshot_sha256=['0' * 64, sha256(snapshot)]), 2, 'snapshot checksum mismatch'),
                 ('journal_refusal', dict(fixture, tokens=[(tokens[0] + 1) % 1000] + tokens[1:]), 2, 'snapshot token journal mismatch'),
                 ('context_capacity_refusal', dict(fixture, max_ctx=1048577), 2, 'invalid snapshot context capacity')]
        commands = []
        for name, source, expected, reason in cases:
            source_path, result_path = controls / (name + '.json'), controls / (name + '.result.json')
            private_json(source_path, source)
            # Valid JSON above the old cap still scores; an input above the new bound refuses before parsing.
            if name == 'test_manifest_above_8MiB':
                source_path.write_bytes(b' ' * (9 * 1024 * 1024) + source_path.read_bytes())
            elif name == 'test_manifest_above_64MiB_refuses':
                with source_path.open('wb') as oversized:
                    oversized.truncate(64 * 1024 * 1024 + 1)
            command = [str(build / 'bin/llama-hybrid-bench-check'), str(model), str(source_path), str(snapshot), str(result_path), '0']
            with (controls / (name + '.log')).open('wb') as log:
                process = subprocess.Popen(command, stdout=log, stderr=log)
                (controls / (name + '.pid')).write_text(str(process.pid) + '\n')
                try:
                    code = process.wait()
                finally:
                    stop_process(process)
            commands.append(dict(name=name, command=command, pid=process.pid, exit_code=code, expected=expected))
            (controls / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
            require(code == expected, name + ': wrong exit code')
            if reason:
                require(reason in (controls / (name + '.log')).read_text() and not result_path.exists(), name + ': unrelated refusal')
            else:
                # Two restores of the same live state are an exactness instrument control, not a mixed-route equivalence claim.
                result = json.loads(result_path.read_text())
                require(result['exact'] and result['top1_agreed'] == 64 and result['mean_kl_nats'] == 0, 'exact checkpoint control failed')
                if name == 'own_state_roundtrips':
                    # Native and transfer-format roundtrips preserve bytes and all 64 continuation distributions.
                    require(all(p['A']['exact'] and all(s['bytes_equal'] for s in p['roundtrips'])
                                for p in result['probes'].values()), 'own state roundtrip is not identity')
            print('PASS ' + name + ': exit ' + str(code), flush=True)


def preflight_controls(state, root, config):
    commands = []
    for name, policy in (('all-local', dict(config['policy'], local_rates=[[0,2000]])),
                         ('all-remote', dict(config['policy'], always=True))):
        policy_file = root / (name + '-policy.json')
        private_json(policy_file, policy)
        output = root / name
        command = [sys.executable, str(Path(__file__).with_name('growing_session.py')), '--client-state', str(state),
            '--output', str(output), '--depth', '128', '--turns', '12', '--disconnect-turn', '4',
            '--disconnect-turns', '2', '--max-new', '48', '--max-output', '128', '--pause-cap', '0', '--seed', '281',
            '--policy', str(policy_file), '--link-label', 'CPU preflight negative control']
        with (root / (name + '.log')).open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            (root / (name + '.pid')).write_text(str(process.pid) + '\n')
            try:
                code = process.wait()
            finally:
                stop_process(process)
        commands.append(dict(name=name, command=command, pid=process.pid, exit_code=code))
        private_json(root / ('preflight-' + name + '-command.json'), commands[-1])
        trace = json.loads((output / 'trace.json').read_text())
        # A real decoder/tokenizer is allowed to initialize, but neither base nor measured arm may execute.
        require(code == 2 and trace['calls'] == [] and trace['status'] == 'aborted' and
                'degenerate connected policy mix before any arm' in trace['reason'], 'preflight refused for an unrelated reason')
        print('PASS preflight ' + name + ': exit2, zero arm calls', flush=True)


def multi_chunk_prestage(build, model, root, config):
    output = root / 'multi-chunk'
    output.mkdir(mode=0o700)
    states = output / 'node-states'
    states.mkdir(mode=0o700)
    events = Events(states)
    with native_server(build / 'bin/llama-server', model, output / 'decoder', checkpoints=0) as (decoder, slots):
        app = MeasuredProxy(dict(config, decode_url=decoder.url, decode_token='', slot_path=str(slots), slots=1,
            extend_chunk_tokens=64, prefill_timeout=120, snapshot_disk_reserve_bytes=28_000_000_000),
            str(output / 'proxy.sqlite'), events, set())
        app.cleanup_prefill = config['prefill_url']
        applied = []
        original = app.decode.request
        remote_request = MeasuredNode.request
        def ordered(node, method, path, data=None, headers=None, binary=False):
            if path.endswith('/extend') and data.get('return_delta'):
                info = original('GET', '/slots/0/info')
                require(info['n'] == data['n0'] and info['prefix_hash'] == data['prefix_hash_n0'],
                        'next chunk requested before matching local acknowledgement')
            return remote_request(node, method, path, data, headers, binary)
        def observed(method, path, data=None, headers=None, binary=False):
            result = original(method, path, data, headers, binary)
            if method == 'GET' and path.endswith('/info'):
                applied.append(result.copy())
            return result
        try:
            with mock.patch.object(app.decode, 'request', side_effect=observed), proxy_server(app) as url:
                messages = base_messages(app, 256)
                meters = []
                sid = 'multi-chunk-prestage'
                for turn in (0, 1):
                    # Actual native imports must be acknowledged at n1 before any later chunk is requested.
                    body = dict(messages=messages, max_tokens=1, temperature=0, ignore_eos=True,
                                chat_template_kwargs=dict(enable_thinking=False))
                    start = len(events.copy())
                    with mock.patch.object(MeasuredNode, 'request', new=ordered):
                        sid, answer, meter, _, _ = stream_chat(url, config['client_credential'], body, 'offload', sid, 120)
                    meters.append(meter)
                    chunks = meter['chunks']
                    require(len(chunks) >= (3 if turn == 0 else 1) and all(c['status'] == 'imported' for c in chunks),
                            'multi-chunk pre-stage or subsequent offload did not import every chunk')
                    require(meter['fallback_reason'] is None and meter['local_evaluated'] == 1,
                            'chunk failure was hidden by local fallback')
                    session = app.store.get(sid)
                    prompt = session['tokens']
                    for chunk in chunks:
                        require(any(a['n'] == chunk['n1'] and a['tokens'] == prompt[:chunk['n1']] and
                                    a['prefix_hash'] == prefix_hash(prompt[:chunk['n1']]) for a in applied),
                                'chunk was not acknowledged at its own position')
                    session = app.store.get(sid)
                    require(original('GET', f"/slots/{session['slot']}/info")['tokens'] == session['tokens'],
                            'native journal mismatch')
                    if turn == 0:
                        with app.reserve(sid):
                            original('POST', '/slots/0?action=save', dict(filename='pre-staged.bin'))
                            original('POST', '/slots/0?action=erase', {})
                            original('POST', '/slots/0?action=restore', dict(filename='pre-staged.bin'))
                            require(original('GET', '/slots/0/info')['tokens'] == session['tokens'],
                                    'pre-staged base restore changed its journal')
                    extends = [e for e in events.copy()[start:] if e.get('n1') is not None and not e.get('catchup')]
                    require(len(extends) == len(chunks) and all(e['status'] == 'ok' and 'node_snapshot' in e for e in extends),
                            'missing per-chunk live snapshot capture')
                    messages += [answer, dict(role='user', content=' x' * 80)]
                    print('PASS native multi-chunk turn ' + str(turn) + ': ' + str(len(chunks)) + ' verified imports', flush=True)
                # Force the real ahead-state recovery branch after a lost import, with a matching prompt boundary.
                with app.reserve(sid):
                    session = app.store.get(sid)
                    ahead = session['tokens'] + app.renderer.encode(' x' * 16)
                    remote = app.remote(config['clients'][config['client_credential']])
                    path = app.sync(session, config['clients'][config['client_credential']])
                    remote.request('POST', path + '/extend', dict(n0=len(session['tokens']),
                        prefix_hash_n0=prefix_hash(session['tokens']), tokens=ahead[len(session['tokens']):], return_delta=True), binary=True)
                    app.resync(session, config['clients'][config['client_credential']], ahead + [ahead[-1]])
                    actual = original('GET', f"/slots/{session['slot']}/info")
                    require(actual['tokens'] == ahead and actual['prefix_hash'] == prefix_hash(ahead) and
                            session['tokens'] == ahead and session['remote_n'] == len(ahead), 'native streamed resync failed')
                print('PASS native streamed ahead-state recovery', flush=True)
                private_json(output / 'meters.json', meters)
        finally:
            private_json(output / 'events.json', events.copy())
    # The regression has consumed these states; retain byte identities without using the smoke's disk reservation.
    removed = []
    for path in output.rglob('*'):
        if path.is_file() and path.suffix in ('.node_snapshot', '.node_delta', '.bin'):
            removed.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path)))
            path.unlink()
    private_json(output / 'cleanup.json', removed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--artifacts', required=True)
    p.add_argument('--controls-only', action='store_true')
    p.add_argument('--multi-chunk-only', action='store_true')
    p.add_argument('--state-probes', action='store_true')
    p.add_argument('--start', choices=('both', 'cold', 'pre-staged'), default='both')
    args = p.parse_args()
    build, root, model = Path(args.build_dir).resolve(), Path(args.artifacts).resolve(), Path(args.model).resolve()
    require(root.is_dir(), 'artifacts directory must exist')
    if not args.multi_chunk_only:
        checkpoint_controls(build, model, root)
    if args.controls_only:
        return
    identity = model_identity(model)
    digest = sha256(model)
    native = SimpleNamespace(artifacts=str(root), binary=str(build / 'bin/llama-hybrid-prefill'), model=str(model))
    with Node(native, 'node', flags=['--bench'], context='4096') as node:
        client = dict(model_path=str(model), model_sha256=digest, backend='cpu', context=4096, batch=256, ubatch=256,
                      build_dir=str(build), **identity,
                      binaries={name: sha256(build / 'bin' / name) for name in ('llama-server', 'llama-hybrid-bench-check')})
        config = dict(benchmark=True, model_sha256=digest, decode_token='private-local-decode-fixture',
            prefill_url=node.base, client_credential='private-local-proxy-fixture',
            clients={'private-local-proxy-fixture': dict(id='a', prefill_token=node.clients['a'])},
            delta_geometry=dict(kv_bytes=6528, state_bytes=20201472),
            policy=dict(local_rates=[[0, 10]], server_rate=2000, link_rate=1e12, fixed=4,
                        kv_bytes=6528, state_bytes=20201472))
        state = root / 'metadata'
        state.mkdir(mode=0o700)
        private_json(state / 'client-info.json', client)
        private_json(state / 'proxy-config.json', config)
        private_json(state / 'node-info.json', dict(model_sha256=digest, backend='cpu', context=4096, bench_link=True, **identity))
        multi_chunk_prestage(build, model, root, config)
        if args.multi_chunk_only:
            return
        preflight_controls(state, root, config)
        commands = []
        for start in (('cold', 'pre-staged') if args.start == 'both' else (args.start,)):
            output = root / start
            command = [sys.executable, str(Path(__file__).with_name('growing_session.py')), '--client-state', str(state),
                '--output', str(output), '--depth', '128', '--turns', '12', '--disconnect-turn', '4',
                '--disconnect-turns', '2', '--max-new', '48', '--max-output', '128', '--pause-cap', '1.1', '--seed', '281',
                '--start', start, '--link-label', 'CPU loopback; sampled output capped128/input48 seed281; synthetic policy']
            if args.state_probes:
                command.append('--state-probes')
            with (root / (start + '.log')).open('wb') as log:
                process = subprocess.Popen(command, stdout=log, stderr=log)
                (root / (start + '.pid')).write_text(str(process.pid) + '\n')
                code = process.wait()
            commands.append(dict(command=command, pid=process.pid, exit_code=code))
            (root / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
            require(code in (0, 3), start + ': experiment aborted; inspect the retained trace and log')
            trace = json.loads((output / 'trace.json').read_text())
            calls = [r for r in trace['calls'] if r['stage'] == 'turn']
            # Three live sessions preserve their journals through both routes and the actual relay outage.
            require(len(calls) == 36 and trace['status'] in ('passed', 'failed'), 'incomplete three-arm session')
            hybrid = [r for r in calls if r['arm'] == 'hybrid']
            require(trace['route_counts']['connected_policy_local'] >= 2 and trace['route_counts']['connected_policy_offload'] >= 1, 'missing connected policy routing branch')
            require(trace['route_counts']['connected_local_catchup_turns'], 'connected local catch-up missing')
            require(trace['route_counts']['background_catchup_events'], 'background replay path missing')
            require(all(r['path'] == 'local' for r in hybrid if 4 <= r['turn'] < 6), 'offload during outage')
            require(any(r.get('resync', {}).get('tokens', 0) > 0 for r in hybrid), 'no recovered journal catch-up')
            require([r['turn'] for r in trace['checkpoints']] == [10, 12], 'checkpoint missing')
            require(all(r['k'] == 64 for r in trace['checkpoints']), 'wrong checkpoint width')
            require(node.request('GET', '/v1/sessions') == [], 'remote sessions leaked')
            for arm in ('local', 'local_partition'):
                local = [r for r in calls if r['arm'] == arm]
                require(all(r['journal_hash'] == l['journal_hash'] for r, l in zip(hybrid, local)), 'paired journals differ')
                require(all(r['path'] == 'local' and r['catchup_tokens'] == 0 and r['wire_bytes'] == 0 for r in local),
                        'local baseline used remote work')
            require(trace['depth_plan']['final_depth'] == trace['final_depth']['hybrid'] and
                    len(trace['workload']) == 12, 'depth plan or sequence changed')
            print('OBSERVED ' + start + ': ' + trace['status'] + '; driver exit ' + str(code), flush=True)
        require(sha256(model) == digest, 'model file changed')
    require(all(r['exit_code'] == 0 for r in commands), 'append replay/component/token criterion is not qualified')
    print('PASS all native growing-session checks; CPU only, no downloads', flush=True)


if __name__ == '__main__':
    main()

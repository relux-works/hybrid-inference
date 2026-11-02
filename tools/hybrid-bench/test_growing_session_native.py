#!/usr/bin/env python3
"""Opt-in CPU growing-session qualification with existing binaries and a local GGUF."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from bench_common import model_identity, private_json, require, sha256
from growing_session import stop_process

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
        cases = [('same_backend_checkpoint_exactness', fixture, 0, None),
                 ('checksum_refusal', dict(fixture, snapshot_sha256=['0' * 64, sha256(snapshot)]), 2, 'snapshot checksum mismatch'),
                 ('journal_refusal', dict(fixture, tokens=[(tokens[0] + 1) % 1000] + tokens[1:]), 2, 'snapshot token journal mismatch'),
                 ('context_capacity_refusal', dict(fixture, max_ctx=1048577), 2, 'invalid snapshot context capacity')]
        commands = []
        for name, source, expected, reason in cases:
            source_path, result_path = controls / (name + '.json'), controls / (name + '.result.json')
            private_json(source_path, source)
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--artifacts', required=True)
    p.add_argument('--controls-only', action='store_true')
    args = p.parse_args()
    build, root, model = Path(args.build_dir).resolve(), Path(args.artifacts).resolve(), Path(args.model).resolve()
    require(root.is_dir(), 'artifacts directory must exist')
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
        preflight_controls(state, root, config)
        commands = []
        for start in ('cold', 'pre-staged'):
            output = root / start
            command = [sys.executable, str(Path(__file__).with_name('growing_session.py')), '--client-state', str(state),
                '--output', str(output), '--depth', '128', '--turns', '12', '--disconnect-turn', '4',
                '--disconnect-turns', '2', '--max-new', '48', '--max-output', '128', '--pause-cap', '1.1', '--seed', '281',
                '--start', start, '--link-label', 'CPU loopback; sampled output capped128/input48 seed281; synthetic policy']
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
    require(all(r['exit_code'] == 0 for r in commands), 'same-backend zero mean-KL criterion is not qualified')
    print('PASS all native growing-session checks; CPU only, no downloads', flush=True)


if __name__ == '__main__':
    main()

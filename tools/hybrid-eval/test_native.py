#!/usr/bin/env python3
"""Opt-in CPU integration using existing native binaries; no builds or downloads."""
import argparse
import json
from pathlib import Path
import secrets
import socket
import subprocess
import sys

from hybrid_eval import TASK_TYPES, generate
from bench_common import (context_plan, control, host_info, http, model_identity, private_json,
                          require, sha256, start_process, wait_health)


def port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-binary', required=True)
    parser.add_argument('--decode-binary', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--artifacts', required=True, help='new private directory')
    parser.add_argument('--depths', nargs='+', type=int, default=[2048])
    parser.add_argument('--tasks', choices=TASK_TYPES, nargs='+', default=list(TASK_TYPES))
    args = parser.parse_args()
    root = Path(args.artifacts).resolve()
    root.mkdir(mode=0o700)
    model = Path(args.model).resolve()
    identity = dict(model_sha256=sha256(model), backend='cpu', **model_identity(model))
    context = max(args.depths) + 2048
    plan = context_plan(model, host_info(), context, True)
    node_port, decode_port, proxy_port = port(), port(), port()
    require(len({node_port, decode_port, proxy_port}) == 3, 'allocated ports collided; rerun')
    credential, node_token = secrets.token_hex(24), secrets.token_hex(24)
    private_json(root / 'node-tokens.json', {'eval': node_token})
    (root / 'slots').mkdir(mode=0o700)
    config = dict(benchmark=True, model_sha256=identity['model_sha256'],
                  node_model_sha256=identity['model_sha256'], layout_fingerprint=identity['layout_fingerprint'],
                  prefill_url=f'http://127.0.0.1:{node_port}', decode_url=f'http://127.0.0.1:{decode_port}',
                  decode_token='', client_credential=credential, prefill_timeout=300, decode_timeout=300,
                  slot_path=str(root / 'slots'), slots=1,
                  clients={credential: dict(id='eval', prefill_token=node_token)},
                  delta_geometry=dict(kv_bytes=plan['kv_bytes_per_token'], state_bytes=plan['recurrent_state_bytes']),
                  policy=dict(kv_bytes=plan['kv_bytes_per_token'], state_bytes=plan['recurrent_state_bytes']))
    private_json(root / 'proxy-config.json', config)
    binaries = {name: sha256(path) for name, path in [('node', args.node_binary), ('decode', args.decode_binary)]}
    info = dict(identity, context=context, batch=256, ubatch=256, threads=2, threads_batch=2, binaries=binaries)
    private_json(root / 'client-info.json', dict(info, proxy_url=f'http://127.0.0.1:{proxy_port}', model_path=str(model)))
    private_json(root / 'node-info.json', info)
    node_command = ['nice', '-n', '10', args.node_binary, model, root / 'node-tokens.json', node_port, context,
                    '--max-ctx', context, '--ram-budget-mib', 512, '--max-request-mib', 1]
    decode_command = ['nice', '-n', '10', args.decode_binary, '-m', model, '--host', '127.0.0.1', '--port', decode_port,
                      '-ngl', 0, '-t', 2, '-tb', 2, '-c', context, '-np', 1, '-b', 256, '-ub', 256,
                      '-ctk', 'q8_0', '-ctv', 'q8_0', '-fa', 'on', '--jinja', '--slots', '--no-context-shift',
                      '--cache-ram', 0, '--no-cache-idle-slots', '--ctx-checkpoints', 0,
                      '--checkpoint-min-step', 0, '--slot-save-path', str(root / 'slots') + '/']
    proxy_command = ['nice', '-n', '10', sys.executable, Path(__file__).resolve().parents[1] / 'hybrid-proxy/proxy.py',
                     '--config', root / 'proxy-config.json', '--database', root / 'proxy.sqlite', '--port', proxy_port]
    commands = [list(map(str, command)) for command in (node_command, decode_command, proxy_command)]
    private_json(root / 'commands.json', commands)
    children = []
    try:
        node = start_process(node_command, root, 'prefill')
        children.append(node)
        health = wait_health(config['prefill_url'], '/v1/health', node)
        require(health['model_sha256'] == identity['model_sha256'], 'node hash differs')
        require(health['layout_fingerprint'] == identity['layout_fingerprint'], 'node layout differs')
        decode = start_process(decode_command, root, 'decode')
        children.append(decode)
        wait_health(config['decode_url'], '/health', decode)
        proxy = start_process(proxy_command, root, 'proxy')
        children.append(proxy)
        wait_health(f'http://127.0.0.1:{proxy_port}', '/health', proxy, token=credential)
        def tokenize(text):
            return http(config['decode_url'], '/tokenize', data=dict(content=text, add_special=False))['tokens']
        # The real tokenizer, at every requested small depth, preserves seeded contents, fitting and position order.
        checks = []
        for depth in args.depths:
            for kind in args.tasks:
                spec = generate(kind, 17, 0, depth, 0.1, tokenize)
                require(spec == generate(kind, 17, 0, depth, 0.1, tokenize), 'native generator nondeterministic')
                require(depth - 8 <= spec['material_tokens'] <= depth, 'native token depth differs')
                checks.append({k: spec[k] for k in ('kind', 'requested_depth', 'material_tokens', 'record_token_offsets', 'material_sha256')})
        private_json(root / 'generator-checks.json', checks)
        command = [sys.executable, str(Path(__file__).with_name('hybrid_eval.py')), '--client-state', str(root),
                   '--output', str(root), '--label', 'cpu', '--depths', *map(str, args.depths), '--positions', '0.1',
                   '--tasks', *args.tasks, '--request-timeout', '300']
        with (root / 'evaluation.log').open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            (root / 'evaluation.pid').write_text(str(process.pid) + '\n')
            try:
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)
        private_json(root / 'evaluation-exit.json', dict(command=command, exit_code=code))
        require(code == 0, 'evaluation failed; inspect evaluation.log and bundle')
        # Every task executes all treatments and the noise control, with equal paired prompts and a local answer turn.
        rows = [json.loads(line) for line in next(root.glob('hybrid-eval-*.jsonl')).read_text().splitlines()]
        tasks = [row for row in rows if row['stage'] == 'task']
        require(len(tasks) == len(args.tasks) * len(args.depths), 'missing task results')
        for task in tasks:
            require(set(task['results']) == {'local', 'hybrid', 'offload', 'local_split'}, 'missing mode')
            for mode, result in task['results'].items():
                require(result['turns'][-1]['remote_applied'] == 0, 'answer offloaded')
                if mode in ('hybrid', 'offload'):
                    require(result['turns'][0]['remote_applied'] > 0, 'ingest stayed local')
                    require(result['vs_local']['prompt_identical'], 'paired prompt differs')
        health = http(config['prefill_url'], '/v1/health')
        require(health['sessions'] == health['ram_bytes'] == 0, 'remote session leaked')
        private_json(root / 'result.json', dict(status='passed', tasks=len(tasks), health=health,
                                               limits='CPU routing/scoring execution only; accuracy is measured, not asserted'))
        print(f'PASS {len(tasks)} tasks x four treatments; real tokenizer; imported ingest, local answers, zero retained sessions')
        return 0
    finally:
        state = control(root, stop=True)
        for child in children:
            child.wait(timeout=15)
        private_json(root / 'cleanup.json', state)


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Opt-in live refusal/stream controls. Use only disposable benchmark node/client state."""
import argparse
import http.server as http_server
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import urllib.request

from bench_common import ROOT, control, http, private_json, read_private, require, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client-state', required=True)
    parser.add_argument('--node-state', required=True)
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    root = Path(args.artifacts).resolve()
    require(root.is_dir(), 'artifacts directory must exist')
    output = Path(args.output).resolve()
    client_state, node_state = Path(args.client_state).resolve(), Path(args.node_state).resolve()
    config = read_private(client_state / 'proxy-config.json')
    client_info = read_private(client_state / 'client-info.json')
    node_info = read_private(node_state / 'node-info.json')
    model_hash = sha256(args.model)
    commands = []

    def command(name, argv, expected, text=None, refused_state=None):
        with (root / (name + '.log')).open('wb') as target:
            process = subprocess.run([str(v) for v in argv], stdout=target, stderr=subprocess.STDOUT)
        commands.append(dict(name=name, command=list(map(str, argv)), exit_code=process.returncode))
        (root / 'native-commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        require(process.returncode == expected, f'{name}: expected {expected}, got {process.returncode}')
        if text:
            require(text in (root / (name + '.log')).read_text(), name + ': intended refusal missing')
        if refused_state:
            require(not refused_state.exists(), name + ': partial state directory exists')
            require(not list(refused_state.parent.glob('.' + refused_state.name + '-*')), name + ': staging state leaked')
        print('PASS ' + name, flush=True)

    node_script = ROOT / 'tools/hybrid-bench/deploy-prefill-node.sh'
    client_script = ROOT / 'tools/hybrid-bench/setup-client.sh'
    node_base = [node_script, '--model', args.model, '--port', '18921']
    client_base = [client_script, '--model', args.model, '--prefill-url', config['prefill_url'],
                   '--token-file', node_state / 'client-tokens.json', '--node-sha256', model_hash,
                   '--decode-port', '18920', '--port', '18922']

    # Both roles reject a wrong local checksum before any build/config/process; source bytes are unchanged.
    for role, base in [('node', node_base), ('client', client_base)]:
        state = root / ('bad-hash-' + role)
        command('bad-hash-' + role, [*base, '--state', state, '--sha256', '0' * 64], 2, 'model sha256 mismatch', state)
    # A second GGUF with the same metadata but different bytes refuses before either backend can load it.
    state = root / 'different-model'
    other_model = root / 'different-model.gguf'
    shutil.copyfile(args.model, other_model)
    with other_model.open('ab') as target:
        target.write(b'synthetic model identity control\n')
    changed = list(client_base)
    changed[changed.index('--model') + 1] = other_model
    command('different-model', [*changed, '--state', state], 2, 'client/node model sha256 mismatch', state)
    # Public exposure without a private credential file and impossible context both refuse without partial state.
    for name, options, message in [('public-no-token', ['--public'], '--public requires --token-file'),
                                    ('context-overflow', ['--ctx', '1000000000'], 'cannot fit')]:
        state = root / name
        command(name, [*node_base, '--state', state, *options], 2, message, state)
    # A bounded file:// download mismatch leaves neither final nor temporary downloaded files.
    source = root / 'download-fixture'
    source.write_bytes(b'synthetic checksum-only download fixture\n')
    cache = root / 'cache'
    cache.mkdir()
    state = root / 'download-mismatch'
    command('download-mismatch', [node_script, '--state', state, '--source-url', source.as_uri(),
        '--sha256', '0' * 64, '--model-cache', cache], 2, 'model sha256 mismatch', state)
    require(not list(cache.iterdir()), 'download mismatch left partial cache files')
    # Live missing/wrong node tokens return exactly 401 and do not mutate session state.
    token = next(iter(config['clients'].values()))['prefill_token']
    before = http(config['prefill_url'], '/v1/sessions', token)
    for name, value in [('missing-token', ''), ('wrong-token', 'invalid-token-control')]:
        try:
            http(config['prefill_url'], '/v1/sessions', value)
        except urllib.error.HTTPError as error:
            require(error.code == 401, name + ': expected HTTP 401')
        else:
            raise AssertionError(name + ': unauthorized read accepted')
        require(http(config['prefill_url'], '/v1/sessions', token) == before, name + ': node sessions changed')
        print('PASS ' + name + ' HTTP401 unchanged sessions', flush=True)
    # Client startup cannot silently accept a bad token or an unreachable node.
    wrong = root / 'wrong-token.json'
    private_json(wrong, {'bench': 'wrong-token-control-value'})
    state = root / 'wrong-client-token'
    changed = list(client_base)
    changed[changed.index('--token-file') + 1] = wrong
    command('wrong-client-token', [*changed, '--state', state], 2, 'node authentication refused (HTTP 401)', state)
    state = root / 'missing-client-token'
    changed = [v for i, v in enumerate(client_base) if i not in (client_base.index('--token-file'), client_base.index('--token-file') + 1)]
    command('missing-client-token', [*changed, '--state', state], 2, 'client requires --token-file', state)
    state = root / 'unreachable'
    changed = list(client_base)
    changed[changed.index('--prefill-url') + 1] = 'http://127.0.0.1:1'
    command('unreachable', [*changed, '--state', state], 2, 'prefill node unreachable', state)

    class ControlBridge(http_server.BaseHTTPRequestHandler):
        calls = 0
        trigger = None
        def log_message(self, *_):
            pass
        def do_POST(self):
            type(self).calls += 1
            if self.trigger == 'kill' and self.calls == 2:
                control(node_state, stop=True)
            body = self.rfile.read(int(self.headers['Content-Length']))
            headers = {key: value for key, value in self.headers.items() if key.lower() in ('authorization', 'content-type', 'x-hybrid-bench-mode', 'x-hybrid-session')}
            request = urllib.request.Request(client_info['proxy_url'] + self.path, body, headers)
            with urllib.request.urlopen(request, timeout=180) as upstream:
                self.send_response(upstream.status)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('X-Hybrid-Session', upstream.headers['X-Hybrid-Session'])
                self.end_headers()
                for line in upstream:
                    if self.trigger == 'meter' and line.startswith(b'data: {'):
                        value = json.loads(line[6:])
                        trace = value.get('hybrid_bench')
                        if trace and trace['depth_before'] > 0:
                            trace['prefilled'] += trace['depth_before']
                            line = b'data: ' + json.dumps(value).encode() + b'\n'
                    self.wfile.write(line)
                    self.wfile.flush()

    # A real run must abort on a planted old-prefix meter violation after a valid first call.
    # The bridge changes telemetry only; it does not claim that the native engine actually re-prefilled.
    # Killing the owned node after the second preflight also detects fallback instead of reporting offload success.
    for name, trigger, expected_reason in [('reprefill-control', 'meter', 'full re-prefill detected'),
                                            ('killed-mid-run', 'kill', 'offload unavailable')]:
        ControlBridge.calls, ControlBridge.trigger = 0, trigger
        bridge = http_server.ThreadingHTTPServer(('127.0.0.1', 0), ControlBridge)
        thread = threading.Thread(target=bridge.serve_forever, daemon=True)
        thread.start()
        state = root / name
        state.mkdir(mode=0o700)
        private_json(state / 'proxy-config.json', config)
        info = dict(client_info, proxy_url=f'http://127.0.0.1:{bridge.server_port}')
        private_json(state / 'client-info.json', info)
        private_json(state / 'node-info.json', node_info)
        try:
            command(name, [sys.executable, ROOT / 'tools/hybrid-bench/hybrid_bench.py', '--client-state', state,
                '--output', output, '--scratch', root, '--label', name, '--modes', 'offload', '--smoke'], 2)
            latest = max(output.glob('hybrid-bench-' + name + '-*.jsonl'))
            rows = [json.loads(line) for line in latest.read_text().splitlines()]
            require(any(expected_reason in row.get('reason', '') for row in rows), name + ': unrelated failure')
            require(any(row.get('stage') == 'call' for row in rows), name + ': no preceding positive call')
            require(rows[-1]['status'] == 'aborted', name + ': aborted run reported success')
        finally:
            bridge.shutdown()
            bridge.server_close()
            thread.join()
    require(sha256(args.model) == model_hash, 'negative tests modified source model')
    print('PASS all native refusal and stream controls; disposable prefill node is now stopped', flush=True)


if __name__ == '__main__':
    sys.exit(main())

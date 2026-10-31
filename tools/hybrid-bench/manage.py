"""Private process setup behind the two operator shell entry points."""
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error

from bench_common import (FLOOR, ROOT, Refusal, check_ports, context_plan, control, http, obtain_model,
                          output, preflight, private_json, read_private, require, run, sha256,
                          start_process, wait_health)


def build(args, targets, info):
    commit = output(['git', '-C', ROOT, 'rev-parse', args.ref + '^{commit}'])
    current = output(['git', '-C', ROOT, 'rev-parse', 'HEAD'])
    source = ROOT
    if commit != current:
        source = Path(args.state).expanduser().resolve().parent / ('source-' + commit[:12])
        require(not source.exists(), 'source worktree already exists; inspect it before retrying')
        run(['git', '-C', ROOT, 'worktree', 'add', '--detach', source, commit])
    build_dir = Path(args.build_dir).expanduser().resolve()
    flags = ['-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA=' + ('ON' if args.cuda else 'OFF'),
             '-DGGML_METAL=' + ('ON' if info['os'] == 'Darwin' else 'OFF'), '-DLLAMA_CURL=OFF']
    stamp = build_dir / 'hybrid-bench-build.json'
    if args.skip_build:
        require(stamp.is_file(), '--skip-build requires a previous successful benchmark build record')
        record = json.loads(stamp.read_text())
        require(record['commit'] == commit and record['flags'] == flags, 'build record ref/backend mismatch')
        for name in targets:
            require(record['binaries'].get(name) == sha256(build_dir / 'bin' / name), 'binary hash mismatch: ' + name)
        return build_dir, record
    lock = Path(args.build_lock).expanduser()
    require(lock.parent.is_dir(), 'build lock parent must exist: ' + str(lock.parent))
    if info['os'] == 'Linux':
        import fcntl
        with lock.open('a') as locked:
            fcntl.flock(locked, fcntl.LOCK_EX)
            record = compile_targets(source, build_dir, commit, flags, targets)
    else:
        # macOS standard library provides the same advisory flock operation.
        import fcntl
        with lock.open('a') as locked:
            fcntl.flock(locked, fcntl.LOCK_EX)
            record = compile_targets(source, build_dir, commit, flags, targets)
    stamp.write_text(json.dumps(record, indent=2) + '\n')
    return build_dir, record


def compile_targets(source, build_dir, commit, flags, targets):
    require(shutil.disk_usage(source).free >= FLOOR, 'disk below 25 GiB build admission floor')
    binaries = build_dir / 'bin'
    names = set(targets)
    if binaries.exists():
        names.update(path.name for path in binaries.iterdir() if '.so' in path.name or path.suffix == '.dylib')
    before = {name: sha256(binaries / name) for name in names if (binaries / name).is_file()}
    run(['cmake', '-S', source, '-B', build_dir, *flags])
    require(shutil.disk_usage(source).free >= FLOOR, 'disk below 25 GiB before compilation')
    command = ['cmake', '--build', str(build_dir), '-j2', '--target', *targets]
    process = subprocess.Popen(command)
    minimum_free = shutil.disk_usage(source).free
    while process.poll() is None:
        minimum_free = min(minimum_free, shutil.disk_usage(source).free)
        if minimum_free < FLOOR:
            # Only descendants of the build PID can be stopped by the disk guard.
            rows = [tuple(map(int, row.split())) for row in output(['ps', '-axo', 'pid=,ppid=']).splitlines()]
            owned = {process.pid}
            for _ in range(32):
                expanded = owned | {pid for pid, parent in rows if parent in owned}
                if expanded == owned:
                    break
                owned = expanded
            for pid in sorted(owned, reverse=True):
                try:
                    os.kill(pid, 15)
                except ProcessLookupError:
                    pass
            process.wait()
            raise Refusal('build stopped below 25 GiB disk floor; owned build PIDs: ' + ','.join(map(str, sorted(owned))))
        time.sleep(.2)
    require(process.returncode == 0, f'build exited {process.returncode}; see build log')
    require(shutil.disk_usage(source).free >= FLOOR, 'disk below 25 GiB after compilation')
    names.update(path.name for path in binaries.iterdir() if '.so' in path.name or path.suffix == '.dylib')
    return dict(commit=commit, source_path=str(source), flags=flags, binaries_before=before, minimum_free_bytes=minimum_free, build_pid=process.pid,
                binaries={name: sha256(binaries / name) for name in names if (binaries / name).is_file()},
                source_dirty=bool(output(['git', '-C', source, 'status', '--porcelain', '--untracked-files=no'])))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('role', choices=['node', 'client'])
    p.add_argument('--state', required=True, help='new private state directory')
    action = p.add_mutually_exclusive_group()
    action.add_argument('--status', action='store_true')
    action.add_argument('--stop', action='store_true')
    action.add_argument('--generate-token', metavar='PATH', help='write a private node token JSON and exit')
    p.add_argument('--model')
    p.add_argument('--sha256')
    p.add_argument('--hf-repo')
    p.add_argument('--hf-file')
    p.add_argument('--source-url', help='file:// or loopback HTTP for download tests; HTTPS otherwise')
    p.add_argument('--model-cache', default='.')
    p.add_argument('--ctx', type=int)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--ref', default='HEAD')
    p.add_argument('--build-dir', default=str(ROOT / 'build'))
    p.add_argument('--build-lock', default=str(Path.home() / 'Werkstatt/hybrid-models/.build.lock'))
    p.add_argument('--skip-build', action='store_true')
    p.add_argument('--port', type=int, default=None)
    p.add_argument('--bind', default='127.0.0.1')
    p.add_argument('--public', action='store_true')
    p.add_argument('--token-file', help='0600 id-to-token JSON shared with node operator')
    p.add_argument('--node-info')
    p.add_argument('--node-sha256')
    p.add_argument('--prefill-url', default='http://127.0.0.1:8091')
    p.add_argument('--decode-port', type=int, default=8090)
    return p


def main():
    args = parser().parse_args()
    state = Path(args.state).expanduser().resolve()
    if args.status or args.stop:
        result = control(state, args.stop)
        print(json.dumps(result))
        return 0 if result and (args.stop or all(v == 'running' for v in result.values())) else 3
    if args.generate_token:
        private_json(args.generate_token, {'bench': secrets.token_urlsafe(32)})
        print(str(Path(args.generate_token).resolve()))
        return 0
    info = preflight(args.cuda, args.role == 'node')
    require(not state.exists(), '--state must not exist; preserve or remove a previous stopped run explicitly')
    require(state.parent.is_dir(), 'state parent must already exist')
    port = args.port if args.port is not None else (8091 if args.role == 'node' else 8092)
    require(args.bind in ('127.0.0.1', 'localhost', '::1') or (args.public and args.token_file),
            'non-loopback bind requires --public plus --token-file; HTTP traffic is unencrypted')
    require(not args.public or args.token_file, '--public requires --token-file')
    require(args.role == 'node' or args.bind == '127.0.0.1', 'client proxy must bind to 127.0.0.1')
    tokens = read_private(args.token_file) if args.token_file else None
    if tokens is not None:
        require(isinstance(tokens, dict) and len(tokens) == 1 and
                all(isinstance(k, str) and isinstance(v, str) and 16 <= len(v) <= 256 for k, v in tokens.items()),
                'token file must contain one client id and a 16-256 byte token')
    check_ports(port, args.decode_port) if args.role == 'client' else check_ports(port)
    downloaded = False
    created_state = False
    model = None
    temporary = None
    try:
        model, digest, downloaded = obtain_model(args, args.model_cache)
        plan = context_plan(model, info, args.ctx, args.role == 'client')
        node = json.loads(Path(args.node_info).read_text()) if args.node_info else None
        if args.role == 'client':
            expected = node['model_sha256'] if node else args.node_sha256
            require(expected and digest == expected, 'client/node model sha256 mismatch; no process started')
            require(not args.node_sha256 or args.node_sha256 == digest, '--node-sha256 disagrees with model')
            require(tokens is not None, 'client requires --token-file from node operator')
            try:
                health = http(args.prefill_url, '/v1/health', timeout=3)
            except (OSError, ValueError):
                raise Refusal('prefill node unreachable or unhealthy at configured URL') from None
            require(health['model_sha256'] == digest, 'live node model sha256 mismatch')
            # An authenticated read detects a missing/wrong token before any build or local state.
            try:
                http(args.prefill_url, '/v1/sessions', next(iter(tokens.values())), timeout=3)
            except urllib.error.HTTPError as error:
                raise Refusal(f'node authentication refused (HTTP {error.code})') from None
            if node:
                plan['benchmark_context'] = min(plan['context'], node['context'])
        targets = ['llama-hybrid-prefill'] if args.role == 'node' else ['llama-server', 'llama-hybrid-bench-check']
        build_dir, record = build(args, targets, info)
        temporary = Path(tempfile.mkdtemp(prefix='.' + state.name + '-', dir=state.parent))
        os.chmod(temporary, 0o700)
        # Paths embedded in configs must use the final directory name.
        if args.role == 'node':
            if tokens is None:
                tokens = {'bench': secrets.token_urlsafe(32)}
            private_json(temporary / 'client-tokens.json', tokens)
            info.update(commit=record['commit'], flags=record['flags'], binaries=record['binaries'],
                        model_file=model.name, model_sha256=digest, context=plan['context'], port=port,
                        sizing=plan, build_record=record, driver=info.get('driver'), cuda=info.get('cuda'))
            private_json(temporary / 'node-info.json', info)
            command = [build_dir / 'bin/llama-hybrid-prefill', model, state / 'client-tokens.json', port,
                       plan['context'], '--host', args.bind, '--gpu-layers', 999 if args.cuda else 0, '--bench']
            os.replace(temporary, state)
            created_state = True
            temporary = None
            process = start_process(command, state, 'prefill')
            health_host = '127.0.0.1' if args.bind == '0.0.0.0' else '::1' if args.bind == '::' else args.bind
            if ':' in health_host:
                health_host = '[' + health_host + ']'
            wait_health(f'http://{health_host}:{port}', '/v1/health', process)
            if args.public:
                print('WARNING: HTTP traffic is unencrypted; use an SSH tunnel or a trusted TLS terminator.', file=sys.stderr)
            print(str(state / 'client-tokens.json'))
        else:
            (temporary / 'slots').mkdir(mode=0o700)
            credential, decode_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            (temporary / 'decode-token').write_text(decode_token + '\n')
            os.chmod(temporary / 'decode-token', 0o600)
            config = dict(benchmark=True, model_sha256=digest, prefill_url=args.prefill_url,
                decode_url=f'http://127.0.0.1:{args.decode_port}', decode_token=decode_token,
                node_token_file=str(Path(args.token_file).resolve()), client_credential=credential,
                slot_path=str(state / 'slots'), slots=1, prefill_timeout=180, decode_timeout=180,
                clients={credential: dict(id=next(iter(tokens)), prefill_token=next(iter(tokens.values())))},
                policy=dict(kv_bytes=plan['kv_bytes_per_token'], state_bytes=plan['recurrent_state_bytes']))
            private_json(temporary / 'proxy-config.json', config)
            info.update(commit=record['commit'], flags=record['flags'], binaries=record['binaries'],
                        model_file=model.name, model_sha256=digest, context=plan['context'],
                        benchmark_context=plan.get('benchmark_context', plan['context']), sizing=plan,
                        proxy_url=f'http://127.0.0.1:{port}', model_path=str(model), build_dir=str(build_dir),
                        build_record=record, policy=config['policy'])
            private_json(temporary / 'client-info.json', info)
            if node:
                private_json(temporary / 'node-info.json', node)
            command = [build_dir / 'bin/llama-server', '-m', model, '--host', '127.0.0.1', '--port', args.decode_port,
                '-ngl', 999 if (args.cuda or info['backend'] == 'metal') else 0, '-t', 2, '-tb', 2,
                '-c', plan['context'], '-np', 1, '-b', 256, '-ub', 256, '-ctk', 'q8_0', '-ctv', 'q8_0',
                '-fa', 'on', '--jinja', '--slots', '--no-context-shift', '--cache-ram', 0, '--no-cache-idle-slots',
                '--ctx-checkpoints', 0, '--checkpoint-min-step', 0, '--slot-save-path', str(state / 'slots') + '/',
                '--api-key-file', state / 'decode-token']
            os.replace(temporary, state)
            created_state = True
            temporary = None
            process = start_process(command, state, 'decode')
            wait_health(config['decode_url'], '/health', process, token=decode_token)
            proxy = start_process([sys.executable, Path(record['source_path']) / 'tools/hybrid-proxy/proxy.py', '--config',
                state / 'proxy-config.json', '--database', state / 'proxy.sqlite', '--port', port], state, 'proxy')
            wait_health(info['proxy_url'], '/health', proxy, token=credential)
            print(str(state / 'proxy-config.json'))
        return 0
    except BaseException:
        if created_state:
            control(state, stop=True)
            shutil.rmtree(state)
        if temporary is not None:
            shutil.rmtree(temporary)
        if downloaded:
            model.unlink(missing_ok=True)
        raise


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (Refusal, OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        # HTTP and process failures do not echo response bodies or command arguments containing credentials.
        print('refused: ' + (str(error) if isinstance(error, Refusal) else type(error).__name__), file=sys.stderr)
        sys.exit(2)

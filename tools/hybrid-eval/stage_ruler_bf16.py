#!/usr/bin/env python3
"""Operator stage for an idle RULER node; never stops an evaluator or controller."""
import argparse
import contextlib
import os
from pathlib import Path
import re
import socket
import signal
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'hybrid-bench'))
from gpu_stage import (add_disk_floor_argument, build, clean_source, disk, exclusive, gpu_identity, interrupted, no_evaluator, read, require, run,
                       save, sha, terminal, verify_archive)

ROLES = ('node', 'client', 'partition')


def launch_plan(source, build_dir, ref, states, original, bf16):
    node_info = read(original/'node/node-info.json')
    info = {role: read(original/role/'client-info.json') for role in ROLES[1:]}
    configs = {role: read(original/role/'proxy-config.json') for role in ROLES[1:]}
    model = info['client']['model_path']
    require(all(value['model_sha256'] == node_info['model_sha256'] for value in info.values()), 'homogeneous models required')
    require(all(value['model_path'] == model for value in info.values()), 'service model paths differ')
    require(all(c['prefill_url'] == f"http://127.0.0.1:{node_info['port']}" for c in configs.values()), 'loopback node required')
    result = {}
    for role in ROLES:
        current = node_info if role == 'node' else info[role]
        require(current['backend'] == 'cuda' and current['threads'] == current['threads_batch'] == 2,
                'stage requires original CUDA launcher settings')
        command = [sys.executable, source/'tools/hybrid-bench/manage.py', 'node' if role == 'node' else 'client',
                   '--offline', '--skip-build', '--cuda', '--ref', ref, '--build-dir', build_dir,
                   '--state', states/role, '--model', model, '--ctx', str(current['context']), '--request-timeout', '7200']
        if role == 'node':
            command += ['--port', str(current['port']), '--ram-budget-mib', str(current['sizing']['ram_budget_mib'])]
            if current.get('bench_link'):
                command += ['--bench-link']
        else:
            config = configs[role]
            require(config['prefill_timeout'] == config['decode_timeout'], 'launcher cannot preserve unequal service timeouts')
            command[command.index('--request-timeout')+1] = str(config['prefill_timeout'])
            require(not config.get('hetero_quant') and config.get('state_dtype', 'f32') == 'f32', 'original services must use homogeneous f32')
            command += ['--batch', str(current['batch']), '--ubatch', str(current['ubatch']),
                        '--node-info', states/'node/node-info.json', '--token-file', states/'node/client-tokens.json',
                        '--prefill-url', config['prefill_url'], '--port', current['proxy_url'].rsplit(':', 1)[1],
                        '--decode-port', config['decode_url'].rsplit(':', 1)[1]]
            if 'extend_chunk_tokens' in config:
                command += ['--extend-chunk-tokens', str(config['extend_chunk_tokens'])]
            if bf16 and role == 'client':
                command += ['--state-dtype', 'bf16']
        result[role] = [str(p) for p in command]
    return result


def evaluator_command(args, manifest):
    depth = manifest['depth']
    return [sys.executable, args.source/'tools/hybrid-eval/ruler_eval.py', '--arm', 'offload_bf16',
            '--paired-depth', args.original_run/f'results-r36/depth-{depth}',
            '--client-state', args.stage/'services/client', '--partition-client-state', args.stage/'services/partition',
            '--data', args.original_run/'data', '--input-lock-sha256', manifest['input_lock_sha256'],
            '--output', args.stage/'results', '--depth', str(depth), '--samples-per-task', str(manifest['samples_per_task']),
            '--seed', str(manifest['seed']), '--margin', str(manifest['margin']), '--tasks', *manifest['tasks'],
            '--request-timeout', '7200', '--saved-session-bytes', str((442685280*(depth+130)+8191)//8192)]


def preserve_pidless_state(state, command, role, log):
    require(state.is_dir() and not state.is_symlink(), f'cannot inspect pid-less state directory: {state}')
    require(not list(state.glob('*.pid')), f'PID records present in {state}; use the recorded owner controls')
    ports = []
    for flag in (('--port',) if role == 'node' else ('--port', '--decode-port')):
        require(command.count(flag) == 1, f'cannot determine {flag} for pid-less state {state}')
        try:
            port = int(command[command.index(flag)+1])
        except (ValueError, TypeError, IndexError) as error:
            raise RuntimeError(f'cannot determine {flag} for pid-less state {state}') from error
        require(1 <= port <= 65535 and port not in ports, f'invalid recovery port for {state}')
        ports.append(port)
    # Hold the expected loopback ports while checking for a child spawned before its PID was recorded.
    with contextlib.ExitStack() as stack:
        for port in ports:
            sock = stack.enter_context(socket.socket())
            try:
                sock.bind(('127.0.0.1', port))
            except OSError as error:
                raise RuntimeError(f'pid-less state {state}: port {port} is not free; refusing recovery without signalling its owner') from error
        exact = re.compile(r"(?<![\w./-])" + re.escape(str(state.resolve())) + r"(?=$|[/\s\x00\"'])")
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                command_line = os.fsdecode((entry/'cmdline').read_bytes())
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError as error:
                raise RuntimeError(f'cannot inspect process {entry.name}; refusing pid-less recovery of {state}') from error
            require(not exact.search(command_line),
                    f'process {entry.name} references exact state path {state}; stop it through its owner path before recovery')
        require(not list(state.glob('*.pid')), f'PID record appeared in {state}; refusing pid-less recovery')
        preserved = Path(tempfile.mkdtemp(prefix=state.name+'.pidless-'+time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'-', dir=state.parent))
        state.rename(preserved/state.name)
        with log.open('a') as output:
            output.write('preserved pid-less state '+str(state)+' as '+str(preserved/state.name)+'; ports='+str(ports)+'\n')


def restore(stage, plan):
    errors = []
    # Launcher stop checks recorded PID identity; it does not kill by process name.
    for role in reversed(ROLES):
        state = stage/'services'/role
        if state.exists():
            try:
                run([sys.executable, Path(plan['source'])/'tools/hybrid-bench/manage.py',
                     'node' if role == 'node' else 'client', '--state', state, '--stop'], stage/'restore.log')
            except Exception as error:
                try:
                    preserve_pidless_state(state, plan['candidate'][role], role, stage/'restore.log')
                except Exception as recovery_error:
                    errors.append(f'candidate {role} stop: {error}; {recovery_error}')
    require(not errors, '; '.join(errors))
    for role in ROLES:
        state = Path(plan['original_run'])/role
        if (stage/'original-services'/role).exists() and state.exists() and not list(state.glob('*.pid')):
            preserve_pidless_state(state, plan['restore'][role], role, stage/'restore.log')
    for role in ROLES:
        backup = stage/'original-services'/role
        if not backup.exists():
            continue
        state = Path(plan['original_run'])/role
        try:
            if not state.exists():
                run(plan['restore'][role], stage/'restore.log')
            run([sys.executable, Path(plan['original_source'])/'tools/hybrid-bench/manage.py',
                 'node' if role == 'node' else 'client', '--state', state, '--status'], stage/'restore.log')
            filename = 'node-info.json' if role == 'node' else 'client-info.json'
            before, after = read(backup/filename), read(state/filename)
            keys = ('backend', 'commit', 'model_sha256', 'context', 'batch', 'ubatch', 'threads', 'threads_batch', 'binaries')
            require({k: before.get(k) for k in keys} == {k: after.get(k) for k in keys}, f'restored {role} identity differs')
        except Exception as error:
            errors.append(f'original {role} restore: {error}')
    require(not errors, '; '.join(errors))
    if 'original_live' in plan:
        verify_live(Path(plan['original_run']), plan['original_live'])
    save(stage/'restored.json', dict(original_ref=plan['original_ref'], verified_at=time.time()))


def live_identity(root):
    from ruler_eval import load_clients
    from bench_common import http
    config, client, pcfg, partition, node = load_clients(root/'client', root/'partition')
    for cfg in (config, pcfg):
        http(cfg['proxy_url'], '/health', cfg['client_credential'], timeout=10)
    keys = ('backend', 'commit', 'model_sha256', 'quantization_type', 'layout_fingerprint', 'context',
            'batch', 'ubatch', 'threads', 'threads_batch', 'binaries', 'template_sha256')
    return {role: {k: info[k] for k in keys if k in info} for role, info in
            (('client', client), ('partition', partition), ('node', node))}


def verify_live(root, expected):
    require(live_identity(root) == expected, 'restored live service identity differs')


def execute(args, manifests):
    gpu = gpu_identity()
    args.stage.mkdir(mode=0o700)
    save(args.stage/'gpu.json', dict(identity=gpu))
    disk(args.stage, floor=args.disk_floor_bytes, log=args.stage/'disk.log')
    (args.stage/'services').mkdir(mode=0o700)
    (args.stage/'original-services').mkdir(mode=0o700)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(args.stage/'tmp'))
    (args.stage/'tmp').mkdir(mode=0o700)
    original_ref = read(args.original_build/'hybrid-bench-build.json')['commit']
    require(original_ref.startswith('18d3f114'), 'original native build is not the declared RULER baseline')
    clean_source(args.original_source, original_ref)
    require(all(sha(args.original_build/'bin'/name) == value for name, value in
                read(args.original_build/'hybrid-bench-build.json')['binaries'].items()), 'original build changed')
    plan: dict = dict(source=str(args.source), original_source=str(args.original_source), original_run=str(args.original_run),
                original_ref=original_ref, candidate_ref=args.ref, disk_floor_bytes=args.disk_floor_bytes,
                restore=launch_plan(args.original_source, args.original_build, original_ref, args.original_run, args.original_run, False),
                candidate=launch_plan(args.source, args.build, args.ref, args.stage/'services', args.original_run, True))
    plan['original_live'] = live_identity(args.original_run)
    save(args.stage/'service-plan.json', plan)
    try:
        for role in reversed(ROLES):
            no_evaluator(args.original_source)
            state = args.original_run/role
            run([sys.executable, args.original_source/'tools/hybrid-bench/manage.py', 'node' if role == 'node' else 'client',
                 '--state', state, '--stop'], args.stage/'services.log')
            state.rename(args.stage/'original-services'/role)
        with exclusive(args.original_run/'build.lock'):
            build(args.source, args.build, args.ref, args.stage/'build.log', disk_floor_bytes=args.disk_floor_bytes)
        for role in ROLES:
            run(plan['candidate'][role], args.stage/'services.log')
        # All selected depths must pass the real evaluator's complete manifest/input gate before any example runs.
        for manifest in manifests:
            run([*evaluator_command(args, manifest), '--preflight-only'], args.stage/f"preflight-{manifest['depth']}.log")
        for manifest in manifests:
            depth = manifest['depth']
            disk(args.stage, 2*((442685280*(depth+130)+8191)//8192) + 1024**3,
                 floor=args.disk_floor_bytes, log=args.stage/'disk.log')
            run(evaluator_command(args, manifest), args.stage/f'arm-{depth}.log')
            verify_archive(args.stage/f'results/depth-{depth}')
    finally:
        restore(args.stage, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_disk_floor_argument(parser)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--ref', required=True, help='full reviewed candidate SHA, already installed locally')
    parser.add_argument('--original-source', type=Path, default=Path('/root/proto/src-ruler'))
    parser.add_argument('--original-run', type=Path, default=Path('/root/proto/src-ruler/.temp/ruler-r35'))
    parser.add_argument('--original-build', type=Path, default=Path('/root/proto/src-ruler/build-ruler'))
    parser.add_argument('--build', type=Path, required=True, help='separate candidate build; never build-ruler')
    parser.add_argument('--stage', type=Path, required=True, help='new private stage output directory')
    parser.add_argument('--depths', type=int, nargs='+', choices=[8192, 32768], default=[8192, 32768])
    parser.add_argument('--execute', action='store_true', help='stop/relaunch owned services only in an idle operator window')
    parser.add_argument('--restore-only', action='store_true', help='recover original services after an interrupted stage')
    args = parser.parse_args()
    for name in ('source', 'original_source', 'original_run', 'original_build', 'build', 'stage'):
        setattr(args, name, getattr(args, name).resolve())
    require(args.build != args.original_build and args.original_build not in args.build.parents and
            args.build not in args.original_build.parents, 'candidate build overlaps build-ruler')
    require(args.source != args.original_source, 'candidate must use a separate checkout')
    require(args.source == Path(__file__).resolve().parents[2], 'source must be the checkout containing this stage script')
    no_evaluator(args.original_source)
    with contextlib.ExitStack() as stack:
        stack.enter_context(exclusive(args.original_run/'bf16-stage.lock'))
        if args.restore_only:
            require(args.execute, 'restore-only requires execute')
            plan = read(args.stage/'service-plan.json')
            require(plan['candidate_ref'] == args.ref and Path(plan['source']) == args.source and
                    Path(plan['original_run']) == args.original_run and Path(plan['original_source']) == args.original_source,
                    'restore arguments differ from persisted service plan')
            restore(args.stage, plan)
            return
        clean_source(args.source, args.ref)
        require(not args.stage.exists() and args.stage.parent.is_dir(), 'stage must be absent with an existing parent')
        disk(args.stage.parent, 8*1024**3, floor=args.disk_floor_bytes)
        require(len(set(args.depths)) == len(args.depths) and args.depths == sorted(args.depths), 'depths must be unique and ordered')
        manifests = []
        for depth in args.depths:
            baseline = args.original_run/f'results-r36/depth-{depth}'
            lock = stack.enter_context((baseline/'.lock').open('r'))
            # The evaluator takes a shared lock too; holding it prevents baseline mutation throughout this stage.
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            seal = verify_archive(baseline)
            manifest = read(baseline/'manifest.json')
            require(manifest['depth'] == depth and seal['n'] == len(manifest['tasks'])*manifest['samples_per_task'], 'depth incomplete')
            manifests.append(manifest)
        if not args.execute:
            print('preflight: completed depths and disk passed; no services changed; use --execute in an idle window')
            return
        execute(args, manifests)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(terminal('ruler-bf16-stage' if '--execute' in sys.argv else 'ruler-bf16-preflight', main))

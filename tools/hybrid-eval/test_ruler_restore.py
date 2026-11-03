"""Real-manager restoration tests with owned CPU workers, no GPU or downloads."""
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'hybrid-bench'))
import bench_common as bench
import gpu_stage as common
import stage_ruler_bf16 as ruler

ROOT = Path(__file__).resolve().parents[2]
MANAGER = ROOT/'tools/hybrid-bench/manage.py'
WORKER = '''import argparse, json, shutil, signal, socket, sys, time
from pathlib import Path
sys.path.insert(0, BENCH_PATH)
from bench_common import start_process
p=argparse.ArgumentParser()
p.add_argument('--worker', action='store_true')
p.add_argument('--state', type=Path, required=True)
p.add_argument('--ready', type=Path)
p.add_argument('--role', default='client')
p.add_argument('--info', type=Path)
p.add_argument('--port', type=int)
p.add_argument('--decode-port', type=int)
p.add_argument('--build-dir')
a=p.parse_args()
if a.worker:
    sockets=[]
    for port in (a.port,a.decode_port):
        if port is not None:
            s=socket.socket(); s.bind(('127.0.0.1',port)); s.listen(); sockets.append(s)
    a.ready.write_text('ready')
    while True: signal.pause()
else:
    a.state.mkdir()
    shutil.copyfile(a.info,a.state/('node-info.json' if a.role=='node' else 'client-info.json'))
    name='prefill' if a.role=='node' else 'proxy'
    command=[sys.executable,__file__,'--worker','--state',str(a.state),'--ready',str(a.state/'ready')]
    for flag,value in (('--port',a.port),('--decode-port',a.decode_port)):
        if value is not None: command += [flag,str(value)]
    process=start_process(command,a.state,name)
    deadline=time.monotonic()+10
    while not (a.state/'ready').exists():
        if process.poll() is not None or time.monotonic()>deadline: raise RuntimeError('CPU worker failed')
        time.sleep(.01)
'''


def ports(count):
    with contextlib.ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(count)]
        for sock in sockets:
            sock.bind(('127.0.0.1', 0))
        return [sock.getsockname()[1] for sock in sockets]


def manager(state, action):
    return subprocess.run([sys.executable, str(MANAGER), 'client', '--state', str(state), action],
                          capture_output=True, text=True)


class RestoreRegressionTests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.owned_processes = []
            stage = root/'stage'; (stage/'services').mkdir(parents=True)
            live = root/'live'; live.mkdir()
            original = root/'original-source'; (original/'tools').mkdir(parents=True)
            (original/'tools/hybrid-bench').symlink_to(ROOT/'tools/hybrid-bench', target_is_directory=True)
            helper = root/'worker.py'
            helper.write_text(WORKER.replace('BENCH_PATH', repr(str(ROOT/'tools/hybrid-bench'))))
            values = iter(ports(5))
            plan: dict = dict(source=str(ROOT), original_source=str(original), original_run=str(live),
                        original_ref='18d3f114'+'0'*32, candidate_ref='1f4e96a1d59dadd8e14e5822b7b4110f2216f043',
                        restore={}, candidate={})
            for role in ruler.ROLES:
                backup = stage/'original-services'/role; backup.mkdir(parents=True)
                info = backup/('node-info.json' if role == 'node' else 'client-info.json')
                common.save(info, dict(commit='original', model_sha256='same'))
                options = ['--port', str(next(values))]
                if role != 'node':
                    options += ['--decode-port', str(next(values))]
                plan['restore'][role] = [sys.executable, str(helper), '--role', role, '--state', str(live/role),
                                         '--info', str(info), '--build-dir', str(root/'build-ruler'), *options]
                plan['candidate'][role] = [sys.executable, str(MANAGER), 'node' if role == 'node' else 'client',
                                           '--state', str(stage/'services'/role), *options]
            common.save(stage/'service-plan.json', plan)
            try:
                yield root, stage, plan
            finally:
                for state in {p.parent for p in root.rglob('*.pid') if p.name in ('proxy.pid', 'decode.pid', 'prefill.pid')}:
                    result = manager(state, '--stop')
                    if result.returncode != 0:
                        raise RuntimeError('owned CPU worker cleanup failed: '+result.stderr)
                for process in self.owned_processes:
                    process.wait(timeout=10)

    def assert_restored(self, stage, plan):
        self.assertTrue((stage/'restored.json').is_file())
        for role in ruler.ROLES:
            self.assertEqual(manager(Path(plan['original_run'])/role, '--status').returncode, 0)

    # Reviewer F1: real --stop returns 3 for a renamed pid-less directory; real restore must still relaunch originals.
    def test_restore_pidless_candidate_real_manage_regression(self):
        with self.fixture() as (_, stage, plan):
            state = stage/'services/client'; state.mkdir()
            (state/'interrupted.json').write_text('preserve me')
            stopped = manager(state, '--stop')
            self.assertEqual((stopped.returncode, stopped.stdout.strip()), (3, '{}'))
            ruler.restore(stage, plan)
            self.assert_restored(stage, plan)
            self.assertFalse(state.exists())
            preserved = list(state.parent.glob('client.pidless-*'))
            self.assertEqual(len(preserved), 1)
            self.assertRegex(preserved[0].name, r'client\.pidless-\d{8}T\d{6}Z-')
            self.assertEqual((preserved[0]/'client/interrupted.json').read_text(), 'preserve me')

    # The absent-state control exercises real launch commands and real manager status, without a recovery exception.
    def test_restore_absent_state_real_manage_control(self):
        with self.fixture() as (_, stage, plan):
            ruler.restore(stage, plan)
            self.assert_restored(stage, plan)

    # The symmetric interruption during original relaunch must preserve the pid-less original and retry its launch.
    def test_restore_pidless_original_real_manage_regression(self):
        with self.fixture() as (_, stage, plan):
            state = Path(plan['original_run'])/'node'; state.mkdir()
            (state/'interrupted.json').write_text('preserve me')
            self.assertEqual(manager(state, '--status').returncode, 3)
            ruler.restore(stage, plan)
            self.assert_restored(stage, plan)
            preserved = list(state.parent.glob('node.pidless-*'))
            self.assertEqual(len(preserved), 1)
            self.assertEqual((preserved[0]/'node/interrupted.json').read_text(), 'preserve me')

    def start_owner(self, root, plan, state_argument, port=None, equals=False):
        owner = root/('owner-'+str(time.time_ns())); owner.mkdir()
        state_arg = ['--state='+str(state_argument)] if equals else ['--state', str(state_argument)]
        command = [sys.executable, plan['restore']['client'][1], '--worker', *state_arg, '--ready', str(owner/'ready')]
        if port is not None:
            command += ['--port', str(port)]
        process = bench.start_process(command, owner, 'proxy')
        self.owned_processes.append(process)
        deadline = time.monotonic()+10
        while not (owner/'ready').exists():
            self.assertIsNone(process.poll(), 'owned test worker exited before readiness')
            self.assertLess(time.monotonic(), deadline, 'owned test worker startup timeout')
            time.sleep(.01)
        return owner, process

    def cli(self, root, stage, plan):
        return subprocess.run([sys.executable, str(Path(ruler.__file__)), '--ref', plan['candidate_ref'],
            '--original-source', plan['original_source'], '--original-run', plan['original_run'],
            '--original-build', str(root/'build-ruler'), '--build', str(root/'candidate-build'),
            '--stage', str(stage), '--execute', '--restore-only'], capture_output=True, text=True)

    # Every candidate port is checked. A foreign listener survives the refusal; its owner stop admits recovery.
    def test_restore_pidless_foreign_port_refuses_without_signalling(self):
        for role, option in [('client', '--port'), ('client', '--decode-port'), ('node', '--port')]:
            with self.subTest(role=role, option=option), self.fixture() as (root, stage, plan):
                state = stage/'services'/role; state.mkdir()
                command = plan['candidate'][role]
                port = int(command[command.index(option)+1])
                owner, process = self.start_owner(root, plan, root/'foreign-state', port)
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    self.assertEqual(common.terminal('ruler-bf16-stage', lambda: ruler.restore(stage, plan)), 1)
                self.assertIn(f'port {port} is not free', err.getvalue())
                self.assertIn('failed ruler-bf16-stage:', err.getvalue())
                self.assertNotIn('done ', out.getvalue())
                self.assertIsNone(process.poll(), 'recovery signalled the foreign listener')
                self.assertTrue(state.exists())
                self.assertFalse(list(state.parent.glob(role+'.pidless-*')))
                self.assertFalse((stage/'restored.json').exists())
                self.assertEqual(manager(owner, '--stop').returncode, 0)
                process.wait(timeout=10)
                ruler.restore(stage, plan)
                self.assert_restored(stage, plan)

    # Original-state ambiguity is checked before any original role is relaunched, even when the node path is absent.
    def test_restore_pidless_original_ambiguity_precedes_any_relaunch(self):
        with self.fixture() as (root, stage, plan):
            state = Path(plan['original_run'])/'client'; state.mkdir()
            command = plan['restore']['client']; port = int(command[command.index('--port')+1])
            owner, process = self.start_owner(root, plan, root/'foreign-state', port)
            with self.assertRaisesRegex(RuntimeError, f'port {port} is not free'):
                ruler.restore(stage, plan)
            self.assertTrue(state.exists())
            self.assertFalse((Path(plan['original_run'])/'node').exists())
            self.assertIsNone(process.poll())
            self.assertEqual(manager(owner, '--stop').returncode, 0)
            process.wait(timeout=10)
            ruler.restore(stage, plan)
            self.assert_restored(stage, plan)

    # A child spawned before its PID write is discoverable even with free ports; only its recorded owner may stop it.
    def test_restore_pidless_live_process_refuses_until_owner_stops(self):
        for descendant, equals in [(False, False), (False, True), (True, False)]:
            with self.subTest(descendant=descendant, equals=equals), self.fixture() as (root, stage, plan):
                state = stage/'services/client'; state.mkdir()
                argument = state/'proxy-config.json' if descendant else state
                owner, process = self.start_owner(root, plan, argument, equals=equals)
                self.assertEqual(manager(state, '--stop').returncode, 3)
                with self.assertRaisesRegex(RuntimeError, f'process {process.pid} references exact state path'):
                    ruler.restore(stage, plan)
                self.assertIsNone(process.poll(), 'pid-less recovery signalled a discoverable child')
                self.assertTrue(state.exists())
                self.assertFalse(list(state.parent.glob('client.pidless-*')))
                self.assertFalse((stage/'restored.json').exists())
                self.assertEqual(manager(owner, '--stop').returncode, 0)
                process.wait(timeout=10)
                ruler.restore(stage, plan)
                self.assert_restored(stage, plan)

    # An adjacent path with the same prefix is not ownership of this state; its process must remain alive.
    def test_restore_exact_path_excludes_sibling_prefix(self):
        with self.fixture() as (root, stage, plan):
            state = stage/'services/client'; state.mkdir()
            _, process = self.start_owner(root, plan, state.with_name('client-other'))
            ruler.restore(stage, plan)
            self.assert_restored(stage, plan)
            self.assertIsNone(process.poll())

    # Any PID file or unreadable command line prevents the absence proof, without moving the directory.
    def test_pidless_recovery_refuses_pid_records_and_uninspectable_process(self):
        with self.fixture() as (_, stage, plan):
            state = stage/'services/client'; state.mkdir()
            record = state/'unknown.pid'; record.write_text('untrusted record')
            with self.assertRaisesRegex(RuntimeError, 'PID records present'):
                ruler.preserve_pidless_state(state, plan['candidate']['client'], 'client', stage/'restore.log')
            self.assertEqual(record.read_text(), 'untrusted record')
            record.unlink()
            read_bytes = Path.read_bytes
            def unreadable(path):
                if path == Path('/proc')/str(os.getpid())/'cmdline':
                    raise PermissionError('injected unreadable command line')
                return read_bytes(path)
            with patch.object(Path, 'read_bytes', unreadable), self.assertRaisesRegex(RuntimeError, 'cannot inspect process'):
                ruler.preserve_pidless_state(state, plan['candidate']['client'], 'client', stage/'restore.log')
            self.assertTrue(state.exists())
            self.assertFalse(list(state.parent.glob('client.pidless-*')))
            ruler.restore(stage, plan)
            self.assert_restored(stage, plan)

    # The real --restore-only CLI reaches the same recovery on both interrupted paths and ends with a done line.
    def test_restore_only_pidless_candidate_and_original_real_manage(self):
        with self.fixture() as (root, stage, plan):
            (stage/'services/client').mkdir()
            (Path(plan['original_run'])/'node').mkdir()
            result = self.cli(root, stage, plan)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.rstrip().splitlines()[-1].startswith('done ruler-bf16-stage seconds='))
            self.assert_restored(stage, plan)

    # Ambiguous CLI recovery has a terminal failed line, leaves the foreign owner alive, and admits an owner-stopped retry.
    def test_restore_only_foreign_port_terminal_failed_and_retry(self):
        with self.fixture() as (root, stage, plan):
            state = stage/'services/client'; state.mkdir()
            command = plan['candidate']['client']; port = int(command[command.index('--port')+1])
            owner, process = self.start_owner(root, plan, root/'foreign-state', port)
            result = self.cli(root, stage, plan)
            self.assertEqual(result.returncode, 1)
            self.assertIn('failed ruler-bf16-stage:', result.stderr)
            self.assertIn(f'port {port} is not free', result.stderr)
            self.assertNotIn('done ruler-bf16-stage', result.stdout)
            self.assertTrue(state.exists())
            self.assertIsNone(process.poll())
            self.assertEqual(manager(owner, '--stop').returncode, 0)
            process.wait(timeout=10)
            result = self.cli(root, stage, plan)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assert_restored(stage, plan)


if __name__ == '__main__':
    unittest.main()

"""Bounded benchmark contracts; native integration is a separate opt-in test."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import bench_common as common
import manage
import hybrid_bench
from hybrid_bench import classify_equivalence, PROVENANCE, bundle, quantile_sample, schedule, verify_meter


class BenchmarkTests(unittest.TestCase):
    # An exact-ref build refuses tracked changes before CMake or a worktree mutation can run.
    def test_build_refuses_dirty_source(self):
        args = argparse.Namespace(ref='HEAD')
        with mock.patch.object(manage, 'output', side_effect=['a' * 40, 'a' * 40, ' M tools/hybrid/prefill.cpp']), mock.patch.object(manage, 'run') as run:
            with self.assertRaisesRegex(common.Refusal, 'exact-ref build'):
                manage.build(args, ['llama-hybrid-prefill'], {})
            run.assert_not_called()

    # Offline builds explicitly disable both UI sources; an offline download request never reaches preflight.
    def test_offline_build_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(ref='HEAD', state=str(root / 'state'), build_dir=str(root / 'build'), build_lock=str(root / 'build.lock'), cuda=False, skip_build=False, offline=True)
            (root / 'build').mkdir()
            with mock.patch.object(manage, 'output', side_effect=['a' * 40, 'a' * 40, '']), mock.patch.object(manage, 'compile_targets', return_value={}) as compile:
                manage.build(args, [], dict(os='Linux'))
            self.assertIn('-DLLAMA_BUILD_UI=OFF', compile.call_args.args[3])
            self.assertIn('-DLLAMA_USE_PREBUILT_UI=OFF', compile.call_args.args[3])
        with mock.patch('sys.argv', ['manage.py', 'node', '--state', 'fixture', '--offline', '--hf-repo', 'fixture']), mock.patch.object(manage, 'preflight') as preflight:
            with self.assertRaisesRegex(common.Refusal, 'offline requires --model'):
                manage.main()
            preflight.assert_not_called()

    # A checksum is a prerequisite to publication: wrong local/downloaded bytes leave no cached partial model.
    def test_model_hash_atomicity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'fixture'
            source.write_bytes(b'synthetic download test')
            args = argparse.Namespace(model=None, sha256='0' * 64, hf_repo=None, hf_file=None, source_url=source.as_uri())
            before = set(root.iterdir())
            with self.assertRaisesRegex(common.Refusal, 'sha256 mismatch'):
                common.obtain_model(args, root)
            self.assertEqual(set(root.iterdir()), before)
            args.sha256 = common.sha256(source)
            model, digest, created = common.obtain_model(args, root)
            self.assertTrue(created)
            self.assertEqual(model.read_bytes(), source.read_bytes())
            self.assertEqual(digest, args.sha256)
            args.model, args.source_url, args.sha256 = str(source), None, '0' * 64
            with self.assertRaisesRegex(common.Refusal, 'sha256 mismatch'):
                common.obtain_model(args, root)
            self.assertEqual(source.read_bytes(), b'synthetic download test')

    # A private config is created without an exposure window; existing files and world-readable tokens are rejected.
    def test_private_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config'
            common.private_json(path, {'fixture': 'public test value'})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(common.read_private(path)['fixture'], 'public test value')
            with self.assertRaises(FileExistsError):
                common.private_json(path, {})
            path.chmod(0o644)
            with self.assertRaisesRegex(common.Refusal, 'private file'):
                common.read_private(path)

    # PID reuse never authorizes a signal; matching PID/start identities are stopped individually.
    def test_pid_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            common.private_json(Path(directory) / 'prefill.pid', dict(pid=999999, identity='old'))
            with mock.patch.object(common, 'process_identity', return_value='new'), mock.patch.object(common.os, 'kill') as kill:
                self.assertEqual(common.control(directory, stop=True), {'prefill': 'stopped'})
                kill.assert_not_called()

    # Explicit long-depth ladders keep short tool/output sizes; invalid ordering and bounds cannot run.
    def test_explicit_depth_ladder(self):
        steps = list(schedule(0, smoke=True, depths=[39500, 40000]))
        self.assertEqual([s['depth'] for s in steps if s['kind'] == 'depth'], [39500, 40000])
        self.assertEqual([s['tokens'] for s in steps if s['kind'] == 'tool'], [16, 32, 64] * 2)
        self.assertTrue(all(s['output'] == 4 for s in steps))
        self.assertEqual([s['depth'] for s in schedule(0, smoke=True) if s['kind'] == 'depth'], [128, 512])
        for depths in ([], [1], [1048577], [40000, 39500], [40000, 40000]):
            with self.assertRaisesRegex(common.Refusal, 'depths must increase'):
                list(schedule(0, depths=depths))

    # A long-run timeout is opt-in; zero, negative and excessive values refuse before state changes.
    def test_request_timeout_bounds(self):
        self.assertEqual(manage.parser().parse_args(['client', '--state', 'fixture']).request_timeout, 180)
        for value in ('0', '-1', '86401'):
            with mock.patch('sys.argv', ['manage.py', 'client', '--state', 'fixture', '--request-timeout', value]), mock.patch.object(manage, 'preflight') as preflight:
                with self.assertRaisesRegex(common.Refusal, 'request-timeout'):
                    manage.main()
                preflight.assert_not_called()
            with mock.patch('sys.argv', ['hybrid_bench.py', '--client-state', 'fixture', '--scratch', 'fixture', '--request-timeout', value]), mock.patch.object(hybrid_bench, 'read_private') as read:
                with self.assertRaisesRegex(common.Refusal, 'request-timeout'):
                    hybrid_bench.main()
                read.assert_not_called()

    # The planner records wire caps separately from the node default; RAM and excessive contexts still refuse.
    def test_context_bounds(self):
        metadata = {'general.architecture': 'qwen35'}
        metadata.update({'qwen35.' + k: v for k, v in dict(block_count=24, context_length=262144,
            **{'ssm.inner_size': 2048, 'ssm.state_size': 128, 'ssm.conv_kernel': 4, 'ssm.group_count': 16,
               'attention.key_length': 256, 'attention.value_length': 256, 'attention.head_count_kv': 2}).items()})
        with tempfile.NamedTemporaryFile() as model, mock.patch.object(common, 'model_metadata', return_value=metadata):
            info = dict(available_memory_bytes=16 * common.GIB)
            plan = common.context_plan(model.name, info, 1024)
            self.assertEqual(plan['kv_bytes_per_token'], 6528)
            self.assertEqual(plan['recurrent_state_bytes'], 20201472)
            self.assertEqual(plan['node_default_max_ctx'], 16384)
            self.assertEqual(plan['delta_body_limit_bytes'], 512 * 1024**2)
            self.assertEqual(plan['delta_header_limit_bytes'], 1024**2)
            self.assertNotIn('protocol_limit', plan)
            with self.assertRaisesRegex(common.Refusal, 'cannot fit'):
                common.context_plan(model.name, info, 10**9)
            with self.assertRaisesRegex(common.Refusal, 'cannot fit'):
                common.context_plan(model.name, dict(available_memory_bytes=2 * common.GIB), 1024)

    # The full re-prefill checker must reject the named old-prefix violation, not only malformed input.
    def test_full_reprefill_control(self):
        trace = dict(prefilled=32, new_tokens=32, replayed_prefix=0, rollback_n=None, depth_before=128,
                     depth_after=164, prompt_depth=160, generated_tokens=4, mode='local', path='local', remote_applied=0)
        verify_meter(trace, 128, 'local')
        with self.assertRaisesRegex(common.Refusal, 'full re-prefill detected'):
            verify_meter(dict(trace, prefilled=160), 128, 'local')
        with self.assertRaisesRegex(common.Refusal, 'prior evaluated boundary'):
            verify_meter(dict(trace, depth_before=0), 128, 'local')
        with self.assertRaisesRegex(common.Refusal, 'fell back locally'):
            verify_meter(dict(trace, mode='offload'), 128, 'offload')

    # Budget refusals remain distinguishable from transport fallback, including a partially imported chain.
    def test_budget_refusal_reason(self):
        trace = dict(prefilled=32, new_tokens=32, replayed_prefix=0, rollback_n=None, depth_before=128,
            depth_after=164, prompt_depth=160, generated_tokens=4, mode='offload', path='offload', remote_applied=31,
            server_prefill_ms=1, wire_bytes=123)
        verify_meter(trace, 128, 'offload')
        with self.assertRaisesRegex(common.Refusal, 'ram budget refusal'):
            verify_meter(dict(trace, fallback_reason='ram_budget'), 128, 'offload')
        with self.assertRaisesRegex(common.Refusal, 'http_503'):
            verify_meter(dict(trace, fallback_reason='http_503'), 128, 'offload')

    # Successful mode cleanup records live store counters after DELETE, without requiring an otherwise idle node.
    def test_mode_release_metrics(self):
        rows = []
        config = dict(client_credential='fixture', prefill_url='http://node')
        response = mock.MagicMock()
        with mock.patch.object(hybrid_bench.urllib.request, 'urlopen', return_value=response) as delete, mock.patch.object(hybrid_bench, 'http', return_value=dict(sessions=0, ram_bytes=0, evictions=2)) as health:
            with hybrid_bench.mode_session(config, dict(proxy_url='http://proxy'), 'fixture', rows, 'offload'):
                self.assertFalse(rows)
            self.assertEqual(delete.call_args.args[0].method, 'DELETE')
            health.assert_called_once_with('http://node', '/v1/health', timeout=5)
        self.assertEqual(rows, [dict(stage='release', mode='offload', status='released', node_sessions=0, node_ram_bytes=0, node_evictions=2)])

    # Cross-backend metrics are uncalibrated with exit 5 on both sides of the reference threshold.
    def test_cross_backend_classification(self):
        result = dict(same_partition_exact=False, numerical_within_bounds=True)
        self.assertEqual(classify_equivalence(result, False), ('failed', 3))
        self.assertEqual(classify_equivalence(result, True), ('uncalibrated', 5))
        self.assertEqual(classify_equivalence(dict(result, numerical_within_bounds=False), True), ('uncalibrated_outside_reference', 5))
        self.assertEqual(classify_equivalence(dict(result, same_partition_exact=True), False), ('passed', 0))

    # Startup cleanup preserves readable private logs while credentials stay in the removable state directory.
    def test_startup_log_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / 'state'
            state.mkdir()
            (state / 'prefill.log').write_text('synthetic driver failure')
            common.private_json(state / 'client-tokens.json', {'fixture': 'private'})
            kept = manage.preserve_startup_logs(state)
            self.assertEqual(kept.stat().st_mode & 0o777, 0o700)
            self.assertEqual((kept / 'prefill.log').read_text(), 'synthetic driver failure')
            self.assertEqual([v.name for v in kept.iterdir()], ['prefill.log'])
            self.assertTrue((state / 'client-tokens.json').exists())

    # Hash equality stays the default; hetero explicitly requires the same nonempty layout fingerprint.
    def test_heterogeneous_setup_admission(self):
        identity = dict(layout_fingerprint='1' * 64)
        common.validate_identity('a' * 64, identity, 'a' * 64, identity)
        with self.assertRaisesRegex(common.Refusal, 'sha256 mismatch'):
            common.validate_identity('a' * 64, identity, 'b' * 64, identity)
        common.validate_identity('a' * 64, identity, 'b' * 64, identity, True)
        for other in ({}, dict(layout_fingerprint='2' * 64)):
            with self.assertRaisesRegex(common.Refusal, 'fingerprint mismatch'):
                common.validate_identity('a' * 64, identity, 'b' * 64, other, True)

    # Once run metadata is valid, malformed runtime traces and interrupts still produce honest bundles.
    def test_error_and_interrupt_bundles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(model_sha256='a' * 64)
            info = dict(model_sha256='a' * 64, backend='cpu', bench_link=True)
            common.private_json(root / 'proxy-config.json', config)
            common.private_json(root / 'client-info.json', info)
            common.private_json(root / 'node-info.json', info)
            for error, code, status in ((TypeError('malformed fixture'), 2, 'aborted'), (KeyboardInterrupt(), 130, 'interrupted')):
                argv = ['hybrid_bench.py', '--client-state', str(root), '--output', str(root), '--scratch', str(root)]
                with mock.patch('sys.argv', argv), mock.patch.object(hybrid_bench, 'link_probe', side_effect=error):
                    self.assertEqual(hybrid_bench.main(), code)
                latest = max(root.glob('hybrid-bench-*.jsonl'), key=lambda p: p.stat().st_mtime_ns)
                summary = json.loads(latest.read_text().splitlines()[-1])
                self.assertEqual((summary['status'], summary['exit_code']), (status, code))

    # A non-HEAD source defaults to its own build tree and a lock beside state, avoiding an incompatible CMake cache.
    def test_ref_build_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / ('source-' + 'b' * 12)
            (source / 'build').mkdir(parents=True)
            args = argparse.Namespace(ref='other', state=str(root / 'state'), build_dir=None, build_lock=None,
                                      cuda=False, skip_build=False)
            record = dict(commit='b' * 40)
            with mock.patch.object(manage, 'output', side_effect=['b' * 40, 'a' * 40, 'b' * 40, '']), mock.patch.object(manage, 'compile_targets', return_value=record) as compile:
                selected, _ = manage.build(args, ['fixture'], dict(os='Linux'))
            self.assertEqual(selected, source / 'build')
            self.assertEqual(compile.call_args.args[:2], (source, source / 'build'))
            self.assertTrue((root / 'build.lock').exists())

    # The launcher passes derived limits and exposes the download endpoint only with an explicit opt-in.
    def test_node_limit_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(context=65536, max_ctx=65536, ram_budget_mib=4096, max_request_mib=2, sessions_held=2)
            identity = dict(layout_fingerprint='1' * 64, quantization_type='Q8_0')
            record = dict(commit='a' * 40, flags=[], binaries={})
            for bench_link in (False, True):
                state = root / str(bench_link)
                argv = ['manage.py', 'node', '--state', str(state), '--model', 'fixture', '--ram-budget-mib', '560'] + (['--bench-link'] if bench_link else [])
                with mock.patch('sys.argv', argv), mock.patch.object(manage, 'preflight', return_value={}), mock.patch.object(manage, 'check_ports'), mock.patch.object(manage, 'obtain_model', return_value=(root / 'model', 'a' * 64, False)), mock.patch.object(manage, 'context_plan', return_value=dict(plan)), mock.patch.object(manage, 'model_identity', return_value=identity), mock.patch.object(manage, 'build', return_value=(root / 'build', record)), mock.patch.object(manage, 'start_process') as start, mock.patch.object(manage, 'wait_health', return_value=dict(model_sha256='a' * 64, **identity)):
                    self.assertEqual(manage.main(), 0)
                command = start.call_args.args[0]
                for flag, value in (('--max-ctx', 65536), ('--ram-budget-mib', 560), ('--max-request-mib', 2)):
                    self.assertEqual(command[command.index(flag) + 1], value)
                self.assertEqual('--bench' in command, bench_link)
                info = common.read_private(state / 'node-info.json')
                self.assertEqual(info['sizing']['sessions_held'], 2)

    # Operator store caps cannot exceed host admission or apply to the decoder; rejection creates no state/build.
    def test_node_budget_cap_refusal(self):
        for role, cap in (('node', '0'), ('node', '-1'), ('node', '4097'), ('client', '560')):
            argv = ['manage.py', role, '--state', '/absent-fixture/state', '--model', 'fixture', '--ram-budget-mib', cap]
            with mock.patch('sys.argv', argv), mock.patch.object(manage, 'preflight', return_value={}), mock.patch.object(manage.Path, 'is_dir', return_value=True), mock.patch.object(manage.Path, 'exists', return_value=False), mock.patch.object(manage, 'check_ports'), mock.patch.object(manage, 'obtain_model', return_value=(Path('fixture'), 'a' * 64, False)), mock.patch.object(manage, 'context_plan', return_value=dict(ram_budget_mib=4096)), mock.patch.object(manage, 'build') as build:
                with self.assertRaisesRegex(common.Refusal, 'ram-budget-mib'):
                    manage.main()
                build.assert_not_called()

    # Interrupting a checker stops only its recorded child and rethrows so the run can emit exit 130.
    def test_checker_interrupt_cleanup(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [KeyboardInterrupt(), -15]
        with mock.patch.object(hybrid_bench, 'start_process', return_value=process):
            with self.assertRaises(KeyboardInterrupt):
                hybrid_bench.run_checker(['fixture'], Path('.'))
        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()
        process.wait.side_effect = None
        process.returncode = 0
        with mock.patch.object(hybrid_bench, 'start_process', return_value=process):
            self.assertEqual(hybrid_bench.run_checker(['fixture'], Path('.')).returncode, 0)

    # Seeded schedules reproduce aggregates without reading private conversations; tails are explicitly clamped.
    def test_schedule_and_quantiles(self):
        a = list(schedule(1, measured=True))
        self.assertEqual(a, list(schedule(1, measured=True)))
        self.assertNotEqual(a, list(schedule(2, measured=True)))
        self.assertEqual(len(a), 24)
        self.assertEqual([s['tokens'] for s in schedule(0) if s['kind'] == 'tool'][:3], [512, 2048, 10000])
        with mock.patch('random.Random.random', return_value=1):
            import random
            self.assertEqual(quantile_sample(random.Random(), PROVENANCE['output_per_call']), 4506)

    # Bundle manifests cover every shareable member and exclude credentials, local paths and raw content.
    def test_bundle_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = dict(backend='cpu', model_sha256='0' * 64, model_path='/private/model', token='forbidden')
            archive = bundle(root, 'fixture', [dict(stage='summary', status='passed', skipped=0, aborted=0, exit_code=0)], info, info)
            with tarfile.open(archive) as source:
                data = {entry.name: source.extractfile(entry).read() for entry in source.getmembers()}
            manifest = data.pop('SHA256SUMS').decode().splitlines()
            self.assertEqual(len(manifest), len(data))
            for line in manifest:
                digest, name = line.split('  ')
                self.assertEqual(digest, hashlib.sha256(data[name]).hexdigest())
            self.assertNotIn(b'forbidden', data['client-info.json'])
            self.assertNotIn(b'/private', data['client-info.json'])


if __name__ == '__main__':
    unittest.main()

"""Bounded benchmark contracts; native integration is a separate opt-in test."""
import argparse
import hashlib
import http.server
import io
import socket
import threading
import urllib.error
from types import SimpleNamespace
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
from policy import extend_chunk_limit


class BenchmarkTests(unittest.TestCase):
    # Client partitions are bounded before preflight or state writes; the native node's fixed partition cannot be mislabeled.
    def test_client_partition_bounds(self):
        args = manage.parser().parse_args(['client', '--state', 'fixture', '--batch', '64', '--ubatch', '32'])
        self.assertEqual((args.batch, args.ubatch), (64, 32))
        for role, batch, ubatch in (('client', '0', '0'), ('client', '64', '65'),
                                     ('client', '65537', '1'), ('node', '64', '64')):
            with mock.patch('sys.argv', ['manage.py', role, '--state', 'fixture', '--batch', batch, '--ubatch', ubatch]), mock.patch.object(manage, 'preflight') as preflight:
                with self.assertRaisesRegex(common.Refusal, 'batch/ubatch|batch/--ubatch'):
                    manage.main()
                preflight.assert_not_called()

    # A selected client partition reaches both the actual decoder command and published metadata, with other flags unchanged.
    def test_client_partition_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = root / 'tokens.json'
            common.private_json(tokens, {'eval': 'x' * 24})
            identity = dict(layout_fingerprint='1' * 64, quantization_type='Q4_0')
            plan = dict(context=4096, kv_bytes_per_token=6528, recurrent_state_bytes=20201472)
            record = dict(commit='a' * 40, flags=[], binaries={}, source_path=str(root))
            for batch, ubatch in ((64, 32), (256, 256)):
                state = root / str(batch)
                argv = ['manage.py', 'client', '--state', str(state), '--model', 'fixture', '--token-file', str(tokens),
                        '--node-sha256', 'a' * 64, '--batch', str(batch), '--ubatch', str(ubatch)]
                with mock.patch('sys.argv', argv), mock.patch.object(manage, 'preflight', return_value={'backend': 'cpu'}), mock.patch.object(manage, 'check_ports'), mock.patch.object(manage, 'obtain_model', return_value=(root / 'model', 'a' * 64, False)), mock.patch.object(manage, 'context_plan', return_value=dict(plan)), mock.patch.object(manage, 'model_identity', return_value=identity), mock.patch.object(manage, 'http', return_value=dict(model_sha256='a' * 64)), mock.patch.object(manage, 'build', return_value=(root / 'build', record)), mock.patch.object(manage, 'start_process') as start, mock.patch.object(manage, 'wait_health'):
                    self.assertEqual(manage.main(), 0)
                command = start.call_args_list[0].args[0]
                for flag, expected in (('-b', batch), ('-ub', ubatch), ('-ngl', 0), ('-np', 1), ('-ctk', 'q8_0')):
                    self.assertEqual(command[command.index(flag) + 1], expected)
                info = common.read_private(state / 'client-info.json')
                self.assertEqual((info['batch'], info['ubatch']), (batch, ubatch))

    # An exact-ref build refuses tracked changes before CMake or a worktree mutation can run.
    def test_build_refuses_dirty_source(self):
        args = argparse.Namespace(ref='HEAD')
        with mock.patch.object(manage, 'output', side_effect=['a' * 40, 'a' * 40, ' M tools/hybrid/prefill.cpp']), mock.patch.object(manage, 'run') as run:
            with self.assertRaisesRegex(common.Refusal, 'exact-ref build'):
                manage.build(args, ['llama-hybrid-prefill'], {})
            run.assert_not_called()

    # Both build modes disable UI downloads; offline model-download requests refuse before preflight.
    def test_offline_build_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(ref='HEAD', state=str(root / 'state'), build_dir=str(root / 'build'), build_lock=str(root / 'build.lock'), cuda=False, skip_build=False, offline=True)
            (root / 'build').mkdir()
            for offline in (True, False):
                args.offline = offline
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
        config = dict(client_credential='fixture', prefill_url='http://node', clients={'fixture': dict(prefill_token='fixture')})
        response = mock.MagicMock()
        with mock.patch.object(hybrid_bench.urllib.request, 'urlopen', return_value=response) as delete, mock.patch.object(hybrid_bench, 'http', return_value=dict(sessions=0, ram_bytes=0, evictions=2)) as health:
            with hybrid_bench.mode_session(config, dict(proxy_url='http://proxy'), 'fixture', rows, 'offload'):
                self.assertFalse(rows)
            self.assertEqual(delete.call_args.args[0].method, 'DELETE')
            health.assert_called_once_with('http://node', '/v1/health', 'fixture', timeout=5)
        self.assertEqual(rows, [dict(stage='release', mode='offload', status='released', node_sessions=0, node_ram_bytes=0, node_evictions=2)])

    # Each geometry fits with the 1 MiB reserve; one more token violates that reserved bound. Old divisors exceed even the raw cap.
    def test_native_chunk_boundaries(self):
        cap, reserve = 512 * 1024**2, 1024**2
        for kv, state in ((6528, 20201472), (34816, 157000000)):
            chunk = extend_chunk_limit(kv, state)
            self.assertLessEqual(state + reserve + chunk * (kv + 24), cap)
            self.assertGreater(state + reserve + (chunk + 1) * (kv + 24), cap)
            old_chunk = (cap - state - reserve) // (6528 if kv == 6528 else 32768)
            self.assertGreater(state + old_chunk * (kv + 24), cap)
        self.assertEqual(extend_chunk_limit(), extend_chunk_limit(34816, 157000000))
        for kv, state in ((0, 0), (1, -1), (1.5, 0), (1, cap)):
            with self.assertRaisesRegex(ValueError, 'native delta geometry'):
                extend_chunk_limit(kv, state)

    # Cross-backend metrics are uncalibrated with exit 5 on both sides of the reference threshold.
    def test_cross_backend_classification(self):
        result = dict(same_partition_exact=False, same_partition_within_floor=False, same_partition_agreed=8, k=8, numerical_within_bounds=True, backend='cpu')
        self.assertEqual(classify_equivalence(result, False), ('failed', 3))
        self.assertEqual(classify_equivalence(result, True), ('uncalibrated', 5))
        self.assertEqual(classify_equivalence(dict(result, numerical_within_bounds=False), True), ('uncalibrated', 5))
        self.assertEqual(classify_equivalence(dict(result, same_partition_exact=True, same_partition_within_floor=True), False), ('passed', 0))

    # A logit mismatch fails despite unchanged argmax; measured-floor acceptance and the extra CPU numerical bound remain distinct.
    def test_gpu_noise_classification(self):
        for backend in ('cuda', 'metal'):
            for numerical in (False, True):
                result = dict(same_partition_exact=True, same_partition_within_floor=True, same_partition_agreed=8, k=8, numerical_within_bounds=numerical, backend=backend)
                self.assertEqual(classify_equivalence(result, False, backend), ('passed', 0))
                self.assertEqual(classify_equivalence(dict(result, same_partition_exact=False, same_partition_within_floor=False), False, backend), ('failed', 3))
                self.assertEqual(classify_equivalence(dict(result, same_partition_exact=False), False, backend), ('passed', 0))
                self.assertEqual(classify_equivalence(result, True, backend), ('uncalibrated', 5))
                self.assertEqual(classify_equivalence(dict(result, backend='cpu'), False, 'cpu'), ('passed', 0) if numerical else ('failed', 3))

    # A CPU checker cannot accept a CUDA/Metal label, including uncalibrated measurements.
    def test_checker_backend_mismatch(self):
        for label in ('cuda', 'metal'):
            for cross in (False, True):
                with self.assertRaisesRegex(common.Refusal, 'backend label/device mismatch'):
                    classify_equivalence(dict(backend='cpu', same_partition_within_floor=True, same_partition_agreed=8, k=8, numerical_within_bounds=True), cross, label)

    # Old checker output refuses before any partial JSONL, Markdown or archive is written.
    def test_old_checker_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = dict(stage='equivalence', same_partition_exact=True, numerical_within_bounds=True)
            with self.assertRaisesRegex(common.Refusal, 'incompatible equivalence checker.*rebuild'):
                bundle(root, 'old', [old], {}, {})
            self.assertEqual(list(root.iterdir()), [])

    # M1-shaped checker output is refused after reading, before classification; the isolated remote session is released.
    def test_checker_version_before_classification(self):
        from types import SimpleNamespace
        config = dict(prefill_url='http://fixture', model_sha256='0' * 64, clients={'a': dict(prefill_token='fixture')})
        client = dict(context=2048, build_dir='build', model_path='fixture.gguf', backend='cpu')
        args = SimpleNamespace(check_tokens=8, noise_chunks=[128, 255], request_timeout=60)
        def old_checker(command, scratch):
            Path(command[4]).write_text(json.dumps(dict(same_partition_exact=True, numerical_within_bounds=True)))
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(hybrid_bench, 'http', side_effect=[dict(session_id='fixture'), b'fixture']), \
                mock.patch.object(hybrid_bench, 'run_checker', side_effect=old_checker), \
                mock.patch.object(hybrid_bench, 'classify_equivalence') as classify, \
                mock.patch.object(hybrid_bench.urllib.request, 'urlopen') as release:
            with self.assertRaisesRegex(common.Refusal, 'incompatible equivalence checker.*rebuild'):
                hybrid_bench.equivalence(config, client, [1, 2, 3], args, Path(directory))
            classify.assert_not_called()
            release.assert_called_once()
            self.assertEqual(release.call_args.args[0].method, 'DELETE')

    # Invalid noise plans refuse before loading private runtime state; valid plans still reach state loading.
    def test_noise_chunk_bounds(self):
        base = ['hybrid_bench.py', '--client-state', 'fixture', '--scratch', '.', '--noise-chunks']
        for chunks in (['0'], ['257'], ['1'] * 5):
            with mock.patch('sys.argv', base + chunks), mock.patch.object(hybrid_bench, 'read_private') as read:
                with self.assertRaisesRegex(common.Refusal, 'noise chunks'):
                    hybrid_bench.main()
                read.assert_not_called()
        with mock.patch('sys.argv', base + ['128', '255']), mock.patch.object(hybrid_bench, 'read_private', side_effect=RuntimeError('valid plan')):
            with self.assertRaisesRegex(RuntimeError, 'valid plan'):
                hybrid_bench.main()

    # Public JSONL, Markdown tables and archived members preserve every measured partition, maximum and applied rule.
    def test_noise_bundle(self):
        noise = dict(chunk_tokens=128, noise_kl_nats=.02, noise_top1_agreed=7, noise_free_running_first_divergence=3)
        row = dict(stage='equivalence', check_mode='fresh', depth=132, k=8, status='passed',
                   status_rule='same_backend_same_partition_exact', mean_kl_nats=.03, top1_agreed=7,
                   free_running_first_divergence=2, noise_partitions=[noise], noise_kl_nats=.02,
                   noise_top1_agreed=7, noise_free_running_first_divergence=3, kl_to_noise_ratio=1.5,
                   kl_to_noise_ratio_status='finite', backend='cpu', same_partition_exact=True,
                   same_partition_within_floor=True, same_partition_kl_nats=0, same_partition_max_abs_logit_diff=0,
                   same_partition_repeat_exact=True, same_partition_repeat_kl_nats=0,
                   same_partition_repeat_max_abs_logit_diff=0, same_partition_repeat_seconds=.2,
                   same_partition_positions=[], noise_reference_seconds=.4)
        rows = [row, dict(status='passed', skipped=0, aborted=0, exit_code=0)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle(root, 'noise', rows, {}, {})
            self.assertEqual(json.loads((root / 'noise.jsonl').read_text().splitlines()[0]), row)
            markdown = (root / 'noise.md').read_text()
            self.assertIn('| Local chunks 128 / whole prompt | 0.02 | 7/8 | 3 |', markdown)
            self.assertIn('rule: same_backend_same_partition_exact', markdown)
            self.assertIn('maximum absolute logit difference 0', markdown)
            self.assertIn('repeat 0.200000 s; noise 0.400000 s', markdown)
            self.assertIn('1.5 (finite)', markdown)
            with tarfile.open(root / 'noise.tar.gz') as archive:
                self.assertEqual(archive.extractfile('noise.jsonl').read(), (root / 'noise.jsonl').read_bytes())
                self.assertEqual(archive.extractfile('noise.md').read().decode(), markdown)

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

    # Derived budgets and lower operator caps reach the node; the download endpoint still requires opt-in.
    def test_node_limit_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(context=65536, max_ctx=65536, ram_budget_mib=4096, max_request_mib=2, sessions_held=2)
            identity = dict(layout_fingerprint='1' * 64, quantization_type='Q8_0')
            record = dict(commit='a' * 40, flags=[], binaries={})
            for bench_link, cap in ((False, None), (True, None), (False, 560), (True, 560)):
                state = root / f'{bench_link}-{cap}'
                argv = ['manage.py', 'node', '--state', str(state), '--model', 'fixture']
                argv += ['--ram-budget-mib', str(cap)] if cap is not None else []
                argv += ['--bench-link'] if bench_link else []
                with mock.patch('sys.argv', argv), mock.patch.object(manage, 'preflight', return_value={}), mock.patch.object(manage, 'check_ports'), mock.patch.object(manage, 'obtain_model', return_value=(root / 'model', 'a' * 64, False)), mock.patch.object(manage, 'context_plan', return_value=dict(plan)), mock.patch.object(manage, 'model_identity', return_value=identity), mock.patch.object(manage, 'build', return_value=(root / 'build', record)), mock.patch.object(manage, 'start_process') as start, mock.patch.object(manage, 'wait_health', return_value=dict(model_sha256='a' * 64, **identity)):
                    self.assertEqual(manage.main(), 0)
                command = start.call_args.args[0]
                for flag, value in (('--max-ctx', 65536), ('--ram-budget-mib', 4096 if cap is None else cap), ('--max-request-mib', 2)):
                    self.assertEqual(command[command.index(flag) + 1], value)
                self.assertEqual('--bench' in command, bench_link)
                info = common.read_private(state / 'node-info.json')
                self.assertEqual(info['sizing']['sessions_held'], 2)
                if cap is None:
                    self.assertNotIn('derived_ram_budget_mib', info['sizing'])
                else:
                    self.assertEqual(info['sizing']['derived_ram_budget_mib'], 4096)

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


class CleanupTests(unittest.TestCase):
    @staticmethod
    def checker_result(code=0):
        return dict(stage='equivalence', depth=132, k=8, status_rule='same_backend_same_partition_exact',
            backend='cpu', mean_kl_nats=.001, top1_agreed=8, free_running_first_divergence=-1,
            noise_partitions=[], noise_kl_nats=0, noise_top1_agreed=8, noise_free_running_first_divergence=-1,
            noise_reference_seconds=.1, kl_to_noise_ratio=None, kl_to_noise_ratio_status='zero_noise',
            same_partition_exact=code == 0, same_partition_within_floor=code == 0, same_partition_agreed=8,
            numerical_within_bounds=True, same_partition_kl_nats=0, same_partition_max_abs_logit_diff=0,
            same_partition_repeat_exact=True, same_partition_repeat_kl_nats=0,
            same_partition_repeat_max_abs_logit_diff=0, same_partition_repeat_seconds=.1,
            same_partition_positions=[])

    def run_fake_node(self, failure, checker_code=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions, attempts, before_delete = set(), {}, []
            test = self
            class Handler(http.server.BaseHTTPRequestHandler):
                def log_message(self, *_):
                    pass

                def reply(self, body, code=200):
                    body = json.dumps(body).encode() if not isinstance(body, bytes) else body
                    self.send_response(code)
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def do_GET(self):
                    self.reply(dict(model_sha256='a' * 64, sessions=len(sessions), ram_bytes=0, evictions=0))

                def do_POST(self):
                    self.rfile.read(int(self.headers.get('Content-Length', 0)))
                    if self.path == '/v1/sessions':
                        sid = 'checker-' + str(len(attempts) + 1)
                        sessions.add(sid)
                        self.reply(dict(session_id=sid))
                    else:
                        self.reply(b'fixture delta')

                def do_DELETE(self):
                    sid = self.path.rsplit('/', 1)[-1]
                    if sid.startswith('checker-'):
                        journal = next(root.glob('hybrid-bench-*.jsonl'))
                        saved = [json.loads(line) for line in journal.read_text().splitlines()]
                        # Both rows must already be on disk at the first cleanup call, not only at bundle time.
                        test.assertEqual(saved[-2]['stage'], 'checker')
                        test.assertEqual(saved[-2]['checker_exit_code'], checker_code)
                        test.assertEqual(json.loads(saved[-2]['result_json']), test.checker_result(checker_code))
                        test.assertEqual(saved[-1]['stage'], 'equivalence')
                        test.assertEqual(saved[-1]['checker_exit_code'], checker_code)
                        before_delete.append(sid)
                        attempts[sid] = attempts.get(sid, 0) + 1
                        if sid == 'checker-2' and (failure == 'permanent' or failure == 'transient' and attempts[sid] == 1):
                            self.connection.shutdown(socket.SHUT_RDWR)
                            self.connection.close()
                            return
                        if sid == 'checker-2' and failure == 'lost_ack':
                            if attempts[sid] == 1:
                                sessions.discard(sid)
                                self.connection.shutdown(socket.SHUT_RDWR)
                                self.connection.close()
                            else:
                                self.reply(dict(error='already released'), 404)
                            return
                        if sid == 'checker-2' and failure == 'unauthorized':
                            self.reply(dict(error='unauthorized'), 401)
                            return
                        sessions.discard(sid)
                    self.reply({})

            server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = 'http://127.0.0.1:' + str(server.server_port)
            config = dict(model_sha256='a' * 64, prefill_url=base, decode_url='http://decoder',
                decode_token='decode-secret', client_credential='client-secret', clients={'client-secret': dict(prefill_token='node-secret')})
            info = dict(model_sha256='a' * 64, context=2048, backend='cpu', proxy_url=base,
                        build_dir='fixture', model_path='fixture.gguf')
            for name, value in (('proxy-config', config), ('client-info', info), ('node-info', info)):
                common.private_json(root / (name + '.json'), value)
            real_http = hybrid_bench.http
            def request(url, path, *args, **kwargs):
                if url != 'http://decoder':
                    return real_http(url, path, *args, **kwargs)
                if path == '/apply-template':
                    return dict(prompt='fixture')
                if path == '/tokenize':
                    return dict(tokens=[1] * (len(args[1]['content']) // 2 if args[1]['content'].startswith(' x') else 3))
                return dict(tokens=[1] * 132)
            def checker(command, scratch):
                Path(command[4]).write_text(json.dumps(self.checker_result(checker_code)))
                return SimpleNamespace(returncode=checker_code)
            trace = dict(prefilled=132, new_tokens=132, replayed_prefix=0, rollback_n=None,
                depth_before=0, prompt_depth=132, depth_after=136, generated_tokens=4, mode='offload',
                path='offload', remote_applied=131, server_prefill_ms=1, wire_bytes=10, slot=0, chunks=[])
            argv = ['hybrid_bench.py', '--client-state', str(root), '--scratch', str(root), '--output', str(root),
                    '--modes', 'offload', '--smoke', '--check', '--check-tokens', '8']
            try:
                with mock.patch('sys.argv', argv), mock.patch.object(hybrid_bench, 'http', side_effect=request), \
                        mock.patch.object(hybrid_bench, 'run_checker', side_effect=checker), \
                        mock.patch.object(hybrid_bench, 'schedule', return_value=iter([dict(kind='depth', depth=128, output=4)])), \
                        mock.patch.object(hybrid_bench, 'stream_chat', return_value=('mode-session', {}, trace, .1, .2)), \
                        mock.patch('release.time.sleep') as sleep:
                    code = hybrid_bench.main()
                path = next(root.glob('hybrid-bench-*.jsonl'))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                markdown = path.with_suffix('.md').read_text()
                with tarfile.open(path.with_suffix('.tar.gz')) as archive:
                    archived = archive.extractfile(path.name).read()
                    self.assertEqual(archived, path.read_bytes())
                    self.assertEqual(archive.extractfile(path.with_suffix('.md').name).read().decode(), markdown)
                    for member in archive.getmembers():
                        data = archive.extractfile(member).read()
                        for secret in (b'decode-secret', b'client-secret', b'node-secret', b'fixture delta'):
                            self.assertNotIn(secret, data)
                self.assertFalse(list(root.glob('equivalence-*')))
                return code, rows, markdown, attempts, list(sessions), before_delete, sleep.call_args_list
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    # A real socket drop during the final checker DELETE preserves both verdicts in the journal/archive and reports the leftover ID; passing cleanup has no error.
    def test_checker_result_survives_final_delete(self):
        for failure, checker_code in ((None, 0), ('permanent', 0), ('permanent', 3)):
            with self.subTest(failure=failure, checker_code=checker_code):
                code, rows, markdown, attempts, sessions, before, sleeps = self.run_fake_node(failure, checker_code)
                checks = [row for row in rows if row['stage'] == 'equivalence']
                self.assertEqual(len(checks), 2)
                self.assertEqual([row['checker_exit_code'] for row in checks], [checker_code] * 2)
                self.assertEqual([row['status'] for row in checks], ['failed' if checker_code else 'passed'] * 2)
                self.assertEqual(len([row for row in rows if row['stage'] == 'checker']), 2)
                errors = [row for row in rows if row['stage'] == 'release_error']
                self.assertEqual(code, 2 if failure else 0)
                self.assertEqual(len(errors), 1 if failure else 0)
                self.assertEqual(sessions, ['checker-2'] if failure else [])
                self.assertEqual(attempts['checker-2'], 3 if failure else 1)
                self.assertTrue(before)
                self.assertEqual(rows[-1]['release_failed'], bool(failure))
                self.assertEqual(rows[-1]['checker_exit_codes'], [checker_code] * 2)
                if failure:
                    self.assertEqual(errors[0]['exception_type'], 'RemoteDisconnected')
                    self.assertIn('Remote end closed connection', errors[0]['message'])
                    self.assertEqual(errors[0]['node_session_ids'], ['checker-2'])
                    self.assertIn('Release failed (checker, append_chain)', markdown)
                    self.assertIn('Node sessions that may remain: checker-2', markdown)
                    self.assertEqual(sleeps, [mock.call(.5), mock.call(1.0)])

    # A dropped connection retries once; success or 404 after a lost acknowledgement leaves no release-error row or failure exit.
    def test_transient_delete_recovers(self):
        for failure in ('transient', 'lost_ack'):
            with self.subTest(failure=failure):
                code, rows, _, attempts, sessions, _, sleeps = self.run_fake_node(failure)
                self.assertEqual(code, 0)
                self.assertEqual(attempts['checker-2'], 2)
                self.assertEqual(sessions, [])
                self.assertFalse(any(row['stage'] == 'release_error' for row in rows))
                self.assertEqual(sleeps, [mock.call(.5)])

    # An actual HTTP 401 is terminal after one DELETE and retains the checker metrics plus its unreleased session.
    def test_unauthorized_delete_not_retried(self):
        code, rows, _, attempts, sessions, _, sleeps = self.run_fake_node('unauthorized')
        self.assertEqual(code, 2)
        self.assertEqual(attempts['checker-2'], 1)
        self.assertEqual(sessions, ['checker-2'])
        self.assertEqual(sleeps, [])
        error = next(row for row in rows if row['stage'] == 'release_error')
        self.assertEqual(error['exception_type'], 'HTTPError')
        self.assertIn('401', error['message'])

    # A partial final JSONL write cannot truncate an already durable checker journal.
    def test_bundle_failure_keeps_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'fixture.jsonl'
            saved = json.dumps(dict(stage='checker', checker_exit_code=3, result_json='{}')) + '\n'
            path.write_text(saved)
            original_open = Path.open
            class FailedWrite:
                def __init__(self, target):
                    self.target = target
                def __enter__(self):
                    return self
                def __exit__(self, *_):
                    self.target.close()
                def write(self, data):
                    self.target.write(data[:3])
                    self.target.flush()
                    raise OSError('injected partial write')
            def opened(target, mode='r', *args, **kwargs):
                stream = original_open(target, mode, *args, **kwargs)
                if target.name in ('fixture.jsonl', 'fixture.jsonl.tmp') and mode == 'w':
                    return FailedWrite(stream)
                return stream
            with mock.patch.object(Path, 'open', opened):
                with self.assertRaisesRegex(OSError, 'injected partial write'):
                    bundle(root, 'fixture', [dict(stage='summary', status='passed', skipped=0, aborted=0, exit_code=0)], {}, {})
            self.assertEqual(path.read_text(), saved)
            self.assertFalse((root / 'fixture.tar.gz').exists())

    # Proxy release diagnostics retain known node IDs; malformed responses fall back to the HTTP error without hiding the checker result.
    def test_mode_release_error_details(self):
        config = dict(client_credential='client-secret', clients={'a': dict(prefill_token='node-secret')})
        for body in (dict(node_session_ids=['node-current', 'node-retired'], exception_type='URLError',
                          message='lost node-secret'), [], dict(node_session_ids=1)):
            rows = []
            error = urllib.error.HTTPError('http://proxy', 503, 'Service Unavailable', {}, io.BytesIO(json.dumps(body).encode()))
            with mock.patch.object(hybrid_bench.urllib.request, 'urlopen', side_effect=error) as delete, mock.patch('release.time.sleep') as sleep:
                with self.assertRaisesRegex(common.Refusal, 'mode release failed'):
                    with hybrid_bench.mode_session(config, dict(proxy_url='http://proxy'), 'mode-session', rows, 'offload'):
                        pass
            delete.assert_called_once()
            sleep.assert_not_called()
            row, = rows
            self.assertEqual(row['stage'], 'release_error')
            if isinstance(body, dict) and isinstance(body.get('node_session_ids'), list):
                self.assertEqual(row['node_session_ids'], ['node-current', 'node-retired'])
                self.assertEqual(row['exception_type'], 'URLError')
                self.assertFalse(row['node_sessions_unknown'])
                self.assertNotIn('node-secret', row['message'])
            else:
                self.assertEqual(row['node_session_ids'], [])
                self.assertEqual(row['exception_type'], 'HTTPError')
                self.assertTrue(row['node_sessions_unknown'])

    # Failed mode cleanup cannot suppress an in-flight exception/interrupt, and its diagnostic redacts credentials and URL userinfo.
    def test_release_preserves_original_error_and_redacts(self):
        config = dict(client_credential='client-secret', decode_token='decode-secret',
            clients={'client-secret': dict(prefill_token='node-secret')})
        error = urllib.error.URLError('lost node-secret client-secret decode-secret http://user:password@node/path')
        for original in (common.Refusal('original failure'), KeyboardInterrupt()):
            rows = []
            with mock.patch.object(hybrid_bench.urllib.request, 'urlopen', side_effect=error), mock.patch('release.time.sleep'):
                with self.assertRaises(type(original)) as raised:
                    with hybrid_bench.mode_session(config, dict(proxy_url='http://proxy'), 'mode-session', rows, 'offload'):
                        raise original
            self.assertIs(raised.exception, original)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['stage'], 'release_error')
            self.assertTrue(rows[0]['node_sessions_unknown'])
            for secret in ('node-secret', 'client-secret', 'decode-secret', 'user:password'):
                self.assertNotIn(secret, json.dumps(rows))


class ProcessControlTests(unittest.TestCase):
    # The startup deadline has a 600 s default, can be set per launch and is bounded; bad values refuse before any state change.
    def test_startup_timeout_bounds(self):
        self.assertEqual(manage.parser().parse_args(['client', '--state', 'fixture']).startup_timeout, 600)
        self.assertEqual(manage.parser().parse_args(['node', '--state', 'fixture', '--startup-timeout', '1200']).startup_timeout, 1200)
        for value in ('0', '-1', '86401'):
            with mock.patch('sys.argv', ['manage.py', 'client', '--state', 'fixture', '--startup-timeout', value]), mock.patch.object(manage, 'preflight') as preflight:
                with self.assertRaisesRegex(common.Refusal, 'startup-timeout'):
                    manage.main()
                preflight.assert_not_called()

    # On macOS a zombie keeps its start time in ps; it must count as stopped, a live process keeps its identity.
    def test_macos_zombie_identity(self):
        with mock.patch.object(common.platform, 'system', return_value='Darwin'):
            with mock.patch.object(common, 'output', return_value='Z    Sat Nov  1 10:00:00 2026'):
                self.assertIsNone(common.process_identity(1234))
            with mock.patch.object(common, 'output', return_value='Ss   Sat Nov  1 10:00:00 2026'):
                self.assertEqual(common.process_identity(1234), 'Sat Nov  1 10:00:00 2026')
            with mock.patch.object(common, 'output', side_effect=common.subprocess.CalledProcessError(1, 'ps')):
                self.assertIsNone(common.process_identity(1234))

    # Stopping an owned child reaps it, on the ps path used on macOS (any host with ps) and on the Linux /proc path where
    # /proc exists; before the fix the ps path saw the zombie as still running and refused with "has not stopped".
    def test_control_reaps_owned_child(self):
        # the Linux path needs /proc, the ps path runs on any host with ps
        for system in ('Darwin',) + (('Linux',) if Path('/proc/self/stat').exists() else ()):
            with self.subTest(system=system), tempfile.TemporaryDirectory() as directory, \
                    mock.patch.object(common.platform, 'system', return_value=system):
                child = common.subprocess.Popen(['python3', '-c', 'import time; time.sleep(60)'])
                try:
                    identity = common.process_identity(child.pid)
                    self.assertTrue(identity)
                    common.private_json(Path(directory) / 'decode.pid', dict(pid=child.pid, identity=identity))
                    self.assertEqual(common.control(directory, stop=True), {'decode': 'stopped'})
                    with self.assertRaises(ChildProcessError):
                        common.os.waitpid(child.pid, common.os.WNOHANG)
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.wait()


if __name__ == '__main__':
    unittest.main()

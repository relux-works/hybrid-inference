"""Growing-session sampling and protocol refusal controls, without model downloads."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import growing_session as growing
from bench_common import Refusal
from policy import Policy
from proxy import HTTPError
from store import prefix_hash
import test_proxy


class ScenarioTests(unittest.TestCase):
    # Deep admission chooses the largest fitting 1024 boundary, preserves every turn, and rejects overflow at100k.
    def test_native_depth_plan(self):
        plan = growing.depth_plan('deep', 40, 1, 10875, 4506, 340, 13)
        self.assertEqual(plan['d0'] % 1024, 0)
        self.assertEqual(len(plan['sequence']), 40)
        self.assertLessEqual(plan['final_depth'] + 2048, 262144)
        self.assertEqual(plan['growth'], sum(r['input_tokens'] + r['output_tokens'] + 13 for r in plan['sequence']))
        for depth in range(plan['d0'] + 1024, 262144, 1024):
            rows = growing.scenario(depth, framing=13)
            self.assertGreater(depth + sum(r['input_tokens'] + r['output_tokens'] + 13 for r in rows) + 2048, 262144)
        shallow = growing.depth_plan(100000, 40, 1, 10875, 4506, 340, 13)
        self.assertEqual(shallow['d0'], 100000)
        boundary = growing.depth_plan('deep', 1, 1, 0, 1, 0, 13, 100000 + 14 + 2048)
        self.assertEqual(boundary['d0'], 99328)
        for depth in ('deep', 100000):
            with self.assertRaisesRegex(Refusal, 'D0=100000.*no turns truncated'):
                growing.depth_plan(depth, 1000, 1, 10875, 4506, 340, 13)

    # The session band accepts its boundary, but excess KL, top1<63, or any free-running divergence fails.
    def test_checkpoint_partition_band(self):
        values = [.001] * 60 + [.003] * 4
        mean = sum(values) / 64
        row = dict(depth=1000, k=64, exact=False, top1_agreed=63, mean_kl_nats=.003,
            max_abs_logit_diff=.8, status='passed', exit_code=0, free_running_first_divergence=-1,
            positions=[dict(local_partition_kl_nats=v) for v in values],
            local_partition_mean_kl_nats=mean, local_partition_p95_kl_nats=.003, kl_limit=.003,
            kl_to_partition_ratio=.003 / mean, kl_to_partition_ratio_status='finite',
            status_rule='session_partition_mean_or_position_p95')
        growing.validate_checkpoint(row, False, 1000)
        for fields in (dict(top1_agreed=62), dict(free_running_first_divergence=63),
                       dict(mean_kl_nats=.003001, kl_to_partition_ratio=.003001 / mean)):
            with self.assertRaises(Refusal):
                growing.validate_checkpoint(dict(row, **fields), False, 1000)
            growing.validate_checkpoint(dict(row, **fields, status='failed', exit_code=3), False, 1000)
        for fields in (dict(kl_limit=.01), dict(positions=[]), dict(local_partition_p95_kl_nats=.004),
                       dict(kl_to_partition_ratio=1)):
            with self.assertRaises(Refusal):
                growing.validate_checkpoint(dict(row, **fields), False, 1000)
        zero = dict(row, mean_kl_nats=0, local_partition_mean_kl_nats=0, local_partition_p95_kl_nats=0,
            positions=[dict(local_partition_kl_nats=0)] * 64, kl_limit=0,
            kl_to_partition_ratio=None, kl_to_partition_ratio_status='both_zero')
        growing.validate_checkpoint(zero, False, 1000)
        with self.assertRaises(Refusal):
            growing.validate_checkpoint(dict(zero, mean_kl_nats=.000001, kl_to_partition_ratio_status='zero_noise_band'), False, 1000)

    # Other sessions' background replay cannot be charged to the local baseline or the current hybrid turn.
    def test_catchup_attribution(self):
        meter = dict(decision={}, depth_before=100, depth_after=104, new_tokens=3, generated_tokens=1,
                     path='local', fallback_reason=None, server_prefill_ms=None, transfer_ms=0, import_ms=0,
                     remote_prepare_ms=0, wire_bytes=0, local_prompt_ms=1, reconciliation_ms=1, decode_ms=0)
        events = [dict(session=sid, catchup=True, status='ok', n0=100, n1=105, seconds=2, upload_body_bytes=20)
                  for sid in ('hybrid', 'other')]
        local = growing.turn_record('local', dict(turn=1), meter, .1, .2, 'hash', events, 'local')
        hybrid = growing.turn_record('hybrid', dict(turn=1), meter, .1, .2, 'hash', events, 'hybrid')
        self.assertEqual((local['catchup_tokens'], local['catchup_seconds'], local['catchup_upload_bytes']), (0, 0, 0))
        self.assertEqual((hybrid['catchup_tokens'], hybrid['catchup_seconds'], hybrid['catchup_upload_bytes']), (5, 2, 20))

    # Tiny KL and unchanged top-1 cannot satisfy exactness; malformed or weakened checker verdicts refuse.
    def test_checkpoint_exactness_gate(self):
        row = dict(depth=1000, k=64, exact=True, top1_agreed=64, mean_kl_nats=0,
                   max_abs_logit_diff=0, status='passed', exit_code=0)
        growing.validate_checkpoint(row, False, 1000)
        failed = dict(row, exact=False, mean_kl_nats=.0001, max_abs_logit_diff=.1, status='failed', exit_code=3)
        growing.validate_checkpoint(failed, False, 1000)
        growing.validate_checkpoint(dict(failed, status='heterogeneous', exit_code=5), True, 1000)
        for changed in (dict(failed, status='passed', exit_code=0), dict(row, mean_kl_nats=.0001),
                        dict(row, k=63), dict(row, top1_agreed=63)):
            with self.assertRaises(Refusal):
                growing.validate_checkpoint(changed, False, 1000)

    # Both depth bands use their published quantiles; caps and a fixed seed remain visible and reproducible.
    def test_conditioned_sequence_and_caps(self):
        self.assertEqual(growing.sample([(0, 0), (.5, 98), (.9, 2329)], .5), 98)
        self.assertAlmostEqual(growing.sample([(0, 0), (.5, 98), (.9, 2329)], .9), 2329)
        for depth, maximum in ((100000, 2329), (250000, 1448)):
            rows = growing.scenario(depth, 40, 1)
            self.assertEqual(rows, growing.scenario(depth, 40, 1))
            self.assertEqual(max(r['input_tokens'] for r in rows), maximum)
            self.assertTrue(all(a['conditioning_depth'] < b['conditioning_depth'] for a, b in zip(rows, rows[1:])))
        rows = growing.scenario(250000, 40, 1, max_new=100, max_output=100, pause_cap=0)
        self.assertTrue(all(r['input_tokens'] <= 100 and r['output_tokens'] <= 100 and r['pause_seconds'] == 0 for r in rows))
        self.assertTrue(any(r['sampled_output_tokens'] > r['output_tokens'] for r in rows))

    # The observer preserves the real policy result and records every routing input, including hysteresis.
    def test_policy_observer_preserves_choice(self):
        current = SimpleNamespace()
        policy = Policy()
        observer = growing.MeasuredPolicy(policy, current)
        for delta in (0, 1, 8000):
            expected = policy.choose(delta, 100000, True, 10000000, 1)
            self.assertEqual(observer.choose(delta, 100000, True, 10000000, 1), expected)
            self.assertEqual(current.decision['result'], expected)
            self.assertEqual(current.decision['inputs'], dict(delta=delta, depth=100000, previous=True,
                                                            link_rate=10000000, server_factor=1))

    # Invalid workload/outage controls refuse before metadata reads, process launch or output creation.
    def test_invalid_run_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'absent'
            for fields in ({'disconnect_turn': 40}, {'turns': 0}, {'pause_cap': -1}, {'disk_floor_gb': 27},
                           {'partition_ubatch': 65, 'partition_batch': 64}):
                args = growing.parser().parse_args(['--client-state', 'absent', '--output', str(output), '--link-label', 'CPU'])
                vars(args).update(fields)
                with mock.patch.object(growing, 'start_process') as start, mock.patch.object(growing, 'read_private') as read:
                    with self.assertRaises(Refusal):
                        growing.run(args)
                    start.assert_not_called()
                    read.assert_not_called()
                self.assertFalse(output.exists())


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_proxy.HTTPTests('runTest')
        self.fixture.setUp()
        self.fixture.app.benchmark = True

    def tearDown(self):
        self.fixture.tearDown()

    def call(self, tokens, sid=None, mode='hybrid'):
        headers = {'X-Hybrid-Bench-Mode': mode}
        if sid:
            headers['X-Hybrid-Session'] = sid
        with self.fixture.client.open('POST', '/v1/chat/completions', dict(messages=[], _tokens=tokens), headers) as response:
            sid, data = response.headers['X-Hybrid-Session'], json.loads(response.read())
        with self.fixture.app.reserve(sid):
            pass
        return sid, data['hybrid_bench']

    # A mismatched explicit journal refuses HTTP409 before import and preserves the durable/native valid prefix.
    def test_journal_mismatch_refuses(self):
        f = self.fixture
        sid, _ = self.call([11, 12, 13])
        before = f.app.store.get(sid)
        imports = f.decode.app.imports
        with self.assertRaises(HTTPError) as error:
            self.call([11, 999, 13, 901, 902, 14], sid)
        self.assertEqual(error.exception.status, 409)
        self.assertEqual(f.decode.app.imports, imports)
        self.assertEqual(f.app.store.get(sid)['tokens'], before['tokens'])
        self.assertEqual(f.decode.app.slots[before['slot']], before['tokens'])
        _, meter = self.call(before['tokens'] + [14, 15], sid)
        self.assertEqual(meter['path'], 'offload')

    # Removing catch-up after an offline local turn refuses offload; the real sync then admits a neighboring valid turn.
    def test_disconnect_without_resync_refuses_offload(self):
        f = self.fixture
        sid, _ = self.call([1, 2, 3])
        f.prefill.app.available = False
        tokens = f.app.store.get(sid)['tokens'] + [4, 5]
        _, meter = self.call(tokens, sid)
        self.assertEqual(meter['path'], 'local')
        self.assertEqual(meter['fallback_reason'], 'http_503')
        f.prefill.app.available = True
        current = f.app.store.get(sid)
        imports = f.decode.app.imports
        with mock.patch.object(f.app, 'sync', return_value='/v1/sessions/' + current['remote_id']):
            _, meter = self.call(current['tokens'] + [6, 7], sid)
        self.assertEqual(meter['path'], 'local')
        self.assertEqual(meter['fallback_reason'], 'http_409')
        self.assertEqual(f.decode.app.imports, imports)
        current = f.app.store.get(sid)
        _, meter = self.call(current['tokens'] + [8, 9], sid)
        self.assertEqual(meter['path'], 'offload')
        self.assertEqual(f.app.store.get(sid)['tokens'], current['tokens'] + [8, 9, 901, 902])

    # A checksum-valid wrong catch-up suffix is rejected by the next prefix GET, before any critical import.
    def test_corrupted_catchup_refuses(self):
        f = self.fixture
        sid, _ = self.call([31, 32, 33])
        durable = f.app.store.get(sid)
        imports = f.decode.app.imports
        dispatch = f.prefill.app.handle
        planted = []
        def corrupt(method, path, query, body, client):
            if path.endswith('/extend') and not body['return_delta']:
                body = dict(body, tokens=[777] + body['tokens'][1:])
                planted.append(path)
            return dispatch(method, path, query, body, client)
        with mock.patch.object(f.prefill.app, 'handle', side_effect=corrupt):
            with self.assertRaises(HTTPError) as error:
                self.call(durable['tokens'] + [34, 35], sid)
        self.assertEqual(error.exception.status, 409)
        self.assertTrue(planted)
        self.assertEqual(f.decode.app.imports, imports)
        self.assertEqual(f.app.store.get(sid)['tokens'], durable['tokens'])
        self.assertEqual(f.decode.app.slots[durable['slot']], durable['tokens'])
        with f.prefill.app.db:
            f.prefill.app.db.execute('DELETE FROM sessions')
        _, meter = self.call(durable['tokens'] + [34, 35], sid)
        self.assertEqual(meter['path'], 'offload')


if __name__ == '__main__':
    unittest.main()

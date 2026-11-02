"""Growing-session sampling and protocol refusal controls, without model downloads."""
import copy
from array import array
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
    # Probe plans preserve local prefix KV through append imports; missing or overlapping node evidence refuses.
    def test_state_probe_plans_require_live_boundaries(self):
        source = dict(tokens=list(range(12)), replay={name: [dict(n0=0, n1=12, logits=True)] for name in ('A', 'B')})
        def event(a, b, catchup=False):
            return dict(session='h', status='ok', n0=a, n1=b, catchup=catchup,
                        node_snapshot=dict(path='snapshot', sha256='s'), node_delta=dict(path='delta', sha256='d'))
        trace = dict(calls=[dict(arm='hybrid', turn=1, session_id='h', meter=dict(chunks=[dict(status='imported', n1=9)]))],
                     events=[event(0, 4), event(4, 7, True), event(7, 9)])
        plans = {r['name']: r['steps'] for r in growing.state_probe_plans(source, trace, 1)}
        self.assertEqual(plans['NODE'][0], dict(snapshot='snapshot', sha256='s'))
        self.assertEqual([next(iter(s)) for s in plans['CHAIN']], ['delta', 'decode', 'delta', 'decode'])
        self.assertEqual(plans['CHAIN'][1]['decode'], [dict(n0=4, n1=7, logits=True)])
        self.assertEqual(plans['CHAIN_ATTENTION'][-2]['replace'], 'attention')
        self.assertEqual(plans['CHAIN_RECURRENT'][-2]['replace'], 'recurrent')
        for change, reason in (('snapshot', 'missing live node state'), ('delta', 'missing live node delta'),
                               ('gap', 'node events do not tile')):
            bad = copy.deepcopy(trace)
            if change == 'gap': bad['events'][1]['n0'] = 5
            else: del bad['events'][-1]['node_' + change]
            with self.assertRaisesRegex(Refusal, reason):
                growing.state_probe_plans(source, bad, 1)

    # Captures come from GET snapshot after the live extend; a moved node boundary is rejected before file writes.
    def test_node_snapshot_capture_refuses_boundary_race(self):
        with tempfile.TemporaryDirectory() as directory:
            for count in (4, 5):
                events = growing.Events(Path(directory))
                node = growing.MeasuredNode('http://unused', events=events, session='h', source='critical')
                snapshot = b'12345678' + count.to_bytes(4, 'little')
                with mock.patch.object(growing.Node, 'request', side_effect=[b'delta', snapshot]) as request:
                    if count == 4:
                        self.assertEqual(node.request('POST', '/v1/sessions/s/extend',
                            dict(n0=0, tokens=[1, 2, 3, 4], prefix_hash_n0='h', return_delta=True), binary=True), b'delta')
                        self.assertEqual(Path(events.copy()[0]['node_snapshot']['path']).read_bytes(), snapshot)
                        self.assertEqual(request.call_args.args, ('GET', '/v1/sessions/s/snapshot'))
                    else:
                        before = set(Path(directory).iterdir())
                        with self.assertRaisesRegex(Refusal, 'moved past'):
                            node.request('POST', '/v1/sessions/s/extend',
                                dict(n0=0, tokens=[1, 2, 3, 4], prefix_hash_n0='h', return_delta=True), binary=True)
                        self.assertEqual(set(Path(directory).iterdir()), before)

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

    # A single zero mean-KL limit rejects numerical excess at every checkpoint, irrespective of the partition tail.
    @mock.patch.object(growing, "validate_raw_positions")
    def test_checkpoint_partition_band(self, raw):
        values = [.001] * 60 + [.003] * 4
        mean = sum(values) / 64
        row = dict(depth=1000, k=64, exact=False, top1_agreed=63, mean_kl_nats=0,
            max_abs_logit_diff=.8, status='passed', exit_code=0, free_running_first_divergence=-1,
            positions=[dict(local_partition_kl_nats=v) for v in values],
            local_partition_mean_kl_nats=mean, local_partition_p95_kl_nats=.003, kl_limit=0,
            partition=dict(mean_kl_nats=mean, p95_kl_nats=.003),
            kl_to_partition_ratio=0, kl_to_partition_ratio_status='finite',
            status_rule='session_mean_kl_zero')
        growing.validate_checkpoint(row, False, 1000)
        for fields in (dict(top1_agreed=62), dict(free_running_first_divergence=63),
                       dict(mean_kl_nats=.003001, kl_to_partition_ratio=.003001 / mean),
                       dict(mean_kl_nats=.0001, kl_to_partition_ratio=.0001 / mean)):
            with self.assertRaises(Refusal):
                growing.validate_checkpoint(dict(row, **fields), False, 1000)
            growing.validate_checkpoint(dict(row, **fields, status='failed', exit_code=3), False, 1000)
        for fields in (dict(kl_limit=.01), dict(local_partition_p95_kl_nats=.004),
                       dict(kl_to_partition_ratio=2)):
            with self.assertRaises(Refusal):
                growing.validate_checkpoint(dict(row, **fields), False, 1000)
        zero = dict(row, mean_kl_nats=0, local_partition_mean_kl_nats=0, local_partition_p95_kl_nats=0,
            positions=[dict(local_partition_kl_nats=0)] * 64, kl_limit=0,
            partition=dict(mean_kl_nats=0, p95_kl_nats=0),
            kl_to_partition_ratio=None, kl_to_partition_ratio_status='both_zero')
        growing.validate_checkpoint(zero, False, 1000)
        with self.assertRaises(Refusal):
            growing.validate_checkpoint(dict(zero, mean_kl_nats=.000001, kl_to_partition_ratio_status='zero_noise_band'), False, 1000)

    @staticmethod
    def append_gate_row():
        exact = dict(exact=True, top1_agreed=64, mean_kl_nats=0, max_abs_logit_diff=0,
                     free_running_first_divergence=-1)
        pairs = [('NODE', 'B'), ('RESTORE', 'B'), ('BSTATE', 'B'), ('NODE_RANGE', 'B'),
                 ('OWN_NATIVE', 'A'), ('OWN_RANGE', 'A'), ('CHAIN', 'H'),
                 ('CHAIN_ATTENTION', 'B'), ('CHAIN_RECURRENT', 'H')]
        probes = {name: {reference: dict(exact), 'roundtrips': []} for name, reference in pairs}
        for name in ('RESTORE', 'NODE_RANGE', 'OWN_NATIVE', 'OWN_RANGE'):
            probes[name]['roundtrips'] = [dict(bytes_equal=True)]
        probes['BSTATE']['roundtrips'] = [dict(snapshot_bytes_equal=True)]
        for name in ('CHAIN_ATTENTION', 'CHAIN_RECURRENT'):
            probes[name]['roundtrips'] = [dict(recurrent_equal=True, attention_equal=False)]
        return dict(exact, depth=1000, k=64, status='passed', exit_code=0,
            status_rule='session_append_replay_exact', kl_limit=None, probes=probes,
            partition=dict(mean_kl_nats=0, p95_kl_nats=0), local_partition_mean_kl_nats=0, local_partition_p95_kl_nats=0,
            kl_to_partition_ratio=None, kl_to_partition_ratio_status='both_zero',
            factorial={'A-L': dict(exact)}, gate_controls={name+'-'+ref: True for name, ref in pairs} | {'A-L': True})

    # Isolate the acceptance rule; the separate raw-logit regression validates every measured distribution.
    @mock.patch.object(growing, 'validate_raw_positions')
    def test_checkpoint_append_positive_kl_passes(self, raw):
        row = self.append_gate_row()
        growing.validate_checkpoint(row, False, 1000)
        row.update(exact=False, mean_kl_nats=.25, p95_kl_nats=.9, max_abs_logit_diff=7, kl_to_partition_ratio_status='zero_noise_band')
        growing.validate_checkpoint(row, False, 1000)
        with self.assertRaisesRegex(Refusal, 'verdict'):
            growing.validate_checkpoint(dict(row, status='failed', exit_code=3), False, 1000)
        growing.validate_checkpoint(dict(row, status='heterogeneous', exit_code=5), True, 1000)

    # A real non-exact CHAIN-H result fails even at zero KL; it cannot be hidden by a forged gate summary.
    @mock.patch.object(growing, 'validate_raw_positions')
    def test_checkpoint_append_replay_mismatch_fails(self, raw):
        row = self.append_gate_row()
        row['probes']['CHAIN']['H'].update(exact=False, max_abs_logit_diff=.001)
        with self.assertRaisesRegex(Refusal, 'contradicts controls'):
            growing.validate_checkpoint(row, False, 1000)
        row['gate_controls']['CHAIN-H'] = False
        with self.assertRaisesRegex(Refusal, 'verdict'):
            growing.validate_checkpoint(row, False, 1000)
        growing.validate_checkpoint(dict(row, status='failed', exit_code=3), False, 1000)

    # Every component/roundtrip identity and A-L is required; missing evidence and changed bytes cannot pass.
    @mock.patch.object(growing, 'validate_raw_positions')
    def test_checkpoint_append_component_controls(self, raw):
        base = self.append_gate_row()
        growing.validate_checkpoint(base, False, 1000)
        for name, reference in [('NODE', 'B'), ('RESTORE', 'B'), ('BSTATE', 'B'), ('NODE_RANGE', 'B'),
                                ('OWN_NATIVE', 'A'), ('OWN_RANGE', 'A'), ('CHAIN_ATTENTION', 'B'), ('CHAIN_RECURRENT', 'H')]:
            row = copy.deepcopy(base)
            row['probes'][name][reference]['exact'] = False
            row['gate_controls'][name+'-'+reference] = False
            with self.assertRaisesRegex(Refusal, 'verdict'):
                growing.validate_checkpoint(row, False, 1000)
            growing.validate_checkpoint(dict(row, status='failed', exit_code=3), False, 1000)
        for name, field in [('RESTORE','bytes_equal'), ('BSTATE','snapshot_bytes_equal'), ('NODE_RANGE','bytes_equal'),
                            ('OWN_NATIVE','bytes_equal'), ('OWN_RANGE','bytes_equal'), ('CHAIN_ATTENTION','recurrent_equal'),
                            ('CHAIN_RECURRENT','recurrent_equal')]:
            row = copy.deepcopy(base)
            row['probes'][name]['roundtrips'][0][field] = False
            with self.assertRaisesRegex(Refusal, 'contradicts controls'):
                growing.validate_checkpoint(row, False, 1000)
        row = copy.deepcopy(base)
        row['factorial']['A-L']['exact'] = False
        with self.assertRaisesRegex(Refusal, 'contradicts controls'):
            growing.validate_checkpoint(row, False, 1000)
        row = copy.deepcopy(base)
        del row['probes']['NODE']
        with self.assertRaisesRegex(Refusal, 'missing append replay'):
            growing.validate_checkpoint(row, False, 1000)

    # 63/64 top-1 and divergence exactly at index8 pass; index0..7 or 62/64 fail, with unchanged replay controls.
    @mock.patch.object(growing, 'validate_raw_positions')
    def test_checkpoint_append_token_boundaries(self, raw):
        row = self.append_gate_row()
        row.update(exact=False, mean_kl_nats=.01, max_abs_logit_diff=1, top1_agreed=63, kl_to_partition_ratio_status='zero_noise_band')
        for divergence in (-1, 8, 63):
            growing.validate_checkpoint(dict(row, free_running_first_divergence=divergence), False, 1000)
        for divergence in range(8):
            bad = dict(row, free_running_first_divergence=divergence)
            with self.assertRaisesRegex(Refusal, 'verdict'):
                growing.validate_checkpoint(bad, False, 1000)
            growing.validate_checkpoint(dict(bad, status='failed', exit_code=3), False, 1000)
        with self.assertRaisesRegex(Refusal, 'verdict'):
            growing.validate_checkpoint(dict(row, top1_agreed=62), False, 1000)

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
    @mock.patch.object(growing, "validate_raw_positions")
    def test_checkpoint_exactness_gate(self, raw):
        row = dict(depth=1000, k=64, exact=True, top1_agreed=64, mean_kl_nats=0,
                   max_abs_logit_diff=0, status='passed', exit_code=0, free_running_first_divergence=-1)
        growing.validate_checkpoint(row, False, 1000)
        failed = dict(row, exact=False, mean_kl_nats=.0001, max_abs_logit_diff=.1, status='failed', exit_code=3)
        growing.validate_checkpoint(failed, False, 1000)
        growing.validate_checkpoint(dict(failed, status='heterogeneous', exit_code=5), True, 1000)
        for changed in (dict(failed, status='passed', exit_code=0), dict(row, mean_kl_nats=.0001),
                        dict(row, k=63), dict(row, top1_agreed=63)):
            with self.assertRaises(Refusal):
                growing.validate_checkpoint(changed, False, 1000)

    # The r7r positions/summary mutant, max-diff lie, shifted positions and altered raw bytes cannot pass independent validation.
    def test_checkpoint_raw_logits_reject_r7r_mutant(self):
        from bench_common import sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in ('L', 'H'):
                path = root / (name + '.f32')
                path.write_bytes(array('f', [2, 0] * 64).tobytes())
                files[name] = dict(path=str(path), sha256=sha256(path))
            row = dict(depth=1000, k=64, exact=True, top1_agreed=64, mean_kl_nats=0, p95_kl_nats=0,
                max_abs_logit_diff=0, free_running_first_divergence=-1, status='passed', exit_code=0,
                positions=[dict(position=1000+i, exact=True, top1_agreed=True, kl_nats=0, max_abs_logit_diff=0) for i in range(64)],
                raw_logits=dict(dtype='float32-le', vocab=2, files=files))
            growing.validate_checkpoint(row, False, 1000)
            # Probe comparisons use the same raw-logit audit; a forged exact label must not pass it.
            probe_row = copy.deepcopy(row)
            probe_row['raw_logits']['files'].update(A=files['L'], NODE=files['H'])
            probe_row['probes'] = dict(NODE=dict(A=copy.deepcopy(row)))
            growing.validate_raw_positions(probe_row, 1000)
            probe_row['probes']['NODE']['A']['exact'] = False
            with self.assertRaisesRegex(Refusal, 'contradict'):
                growing.validate_raw_positions(probe_row, 1000)
            mutants = []
            mutant = copy.deepcopy(row)
            for p in mutant['positions']:
                p.update(kl_nats=9.9, top1_agreed=False)
            mutants.append(mutant)
            mutants.append(dict(row, max_abs_logit_diff=12345))
            mutant = copy.deepcopy(row)
            mutant['positions'][0]['position'] += 1
            mutants.append(mutant)
            mutants.append(dict(row, mean_kl_nats=.01))
            mutants.append(dict(row, top1_agreed=63))
            for mutant in mutants:
                with self.assertRaisesRegex(Refusal, 'contradict'):
                    growing.validate_checkpoint(mutant, False, 1000)
            path.write_bytes(array('f', [1, 0] * 64).tobytes())
            with self.assertRaisesRegex(Refusal, 'checksum'):
                growing.validate_checkpoint(row, False, 1000)
            # Non-exact valid logits exercise the KL calculation, rather than an always-rejecting or equality-only validator.
            files['H']['sha256'] = sha256(path)
            # KL(softmax([2,0]) || softmax([1,0])) = 0.0671307544531327 nats.
            kl = 0.0671307544531327
            changed = dict(row, exact=False, mean_kl_nats=kl, p95_kl_nats=kl, max_abs_logit_diff=1, status='failed', exit_code=3,
                positions=[dict(position=1000+i, exact=False, top1_agreed=True, kl_nats=kl, max_abs_logit_diff=1) for i in range(64)])
            growing.validate_checkpoint(changed, False, 1000)

    # The checked-in measured table binds each depth bin and all pause/output quantiles to the source selector.
    def test_sampling_table_provenance(self):
        table = json.loads((Path(__file__).parent / 'fixtures/growing-provenance.json').read_text())
        self.assertEqual(table['source']['sha256'], growing.PROVENANCE['sha256'])
        self.assertEqual([(r[1],r[2]) for r in growing.DEPTH_BINS],
            [(v['p50'],v['p90']) for v in table['new_in_by_depth'].values()])
        keys = ['p10','p25','p50','p75','p90','p99']
        self.assertEqual(growing.PAUSE,[table['gap_seconds'][k] for k in keys])
        self.assertEqual(growing.OUTPUT,[table['output'][k] for k in keys])

    # Equal-rate and always-remote plans refuse before arms; the bounded smoke has connected policy-local and offload decisions.
    def test_route_preflight_rejects_degenerate(self):
        plan = growing.depth_plan(128, 12, 281, 48, 128, 0, 14, 4096)
        for policy in (Policy(local_rates=[[0,2000]], server_rate=2000), Policy(always=True)):
            with self.assertRaisesRegex(Refusal, 'degenerate connected policy mix before any arm'):
                growing.route_preflight(policy, plan, 4, 2)
        result = growing.route_preflight(Policy(local_rates=[[0,10]], fixed=4, link_rate=1e12), plan, 4, 2)
        self.assertEqual(result['counts'], dict(policy_local=2, policy_offload=8))
        self.assertAlmostEqual(sum(r['output_tokens'] for r in plan['sequence']) / plan['growth'], .6386422976501306)

    # Outage fallbacks alone never establish policy-local coverage; only this session's connected catch-up counts.
    def test_route_count_separates_outage_fallback(self):
        def call(turn, local=False, fallback=None):
            return dict(arm='hybrid', stage='turn', turn=turn, path='local' if local or fallback else 'offload',
                decision=dict(result=dict(offload=not local)), fallback_reason=fallback, depth_before=100,
                meter=dict(prompt_depth=120), session_id='hybrid', at_seconds=turn)
        calls = [call(1), call(4, fallback='transport')]
        events = [dict(kind='protocol', session='other', catchup=True, status='ok', n0=100, n1=130, at_seconds=3)]
        counts = growing.route_counts(calls, events, dict(turn=4, turns=2))
        self.assertEqual(counts['connected_policy_local'], 0)
        self.assertEqual(counts['fallbacks'], {'transport':1})
        calls.append(call(2, local=True))
        self.assertEqual(growing.route_counts(calls, events, dict(turn=4, turns=2))['connected_local_catchup_turns'], [])
        events[0]['session'] = 'hybrid'
        self.assertEqual(growing.route_counts(calls, events, dict(turn=4, turns=2))['connected_local_catchup_turns'], [2])
        # A reconnect-only catch-up cannot stand in for the connected path.
        events[0]['at_seconds'] = 6
        events.append(dict(kind='disconnect', at_seconds=4))
        self.assertEqual(growing.route_counts(calls, events, dict(turn=4, turns=2))['connected_local_catchup_turns'], [])

    # Factorial cells preserve whole flat batches only in B and reject missing node ranges before a checker starts.
    def test_factorial_schedule_boundaries(self):
        calls = [dict(arm='hybrid', turn=0, depth_before=0, depth_after=8, session_id='h',
            meter=dict(prompt_depth=4, chunks=[dict(n0=0,n1=3,status='imported')])),
            dict(arm='hybrid', turn=1, depth_before=8, depth_after=12, session_id='h',
            meter=dict(prompt_depth=10, chunks=[dict(n0=8,n1=9,status='imported')]))]
        events = [dict(session='h', status='ok', n0=a, n1=b) for a,b in ((0,3),(3,8),(8,9))]
        trace = dict(calls=calls, events=events)
        cells = growing.replay_schedules(trace,1,12)
        with self.assertRaisesRegex(Refusal, 'primary local 256/256'):
            growing.replay_schedules(dict(trace, client=dict(batch=64, ubatch=32)),1,12)
        self.assertIn(dict(n0=4,n1=8,logits=True), cells['B1'])
        self.assertIn(dict(n0=3,n1=8,logits=True), cells['B'])
        self.assertNotIn(dict(n0=3,n1=8,logits=True), cells['B1'])
        self.assertIn(dict(n0=0,n1=3,logits=True), cells['B2'])
        with self.assertRaisesRegex(Refusal, 'do not tile'):
            growing.replay_schedules(dict(trace,events=events[1:]),1,12)

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

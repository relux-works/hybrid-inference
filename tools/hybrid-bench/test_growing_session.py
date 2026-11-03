"""Growing-session sampling and protocol refusal controls, without model downloads."""
import copy
from array import array
import json
import io
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import growing_session as growing
from bench_common import Refusal
from policy import Policy
from growing_fixture import validate_fixture, from_archive
from proxy import HTTPError
from store import prefix_hash
import test_proxy


class ScenarioTests(unittest.TestCase):
    @staticmethod
    def packet_module(name):
        path = Path(growing.__file__).parent / 'packets/r46' / (name + '.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # Labels cannot alias a transport, and the bf16 policy changes only tensor wire bytes while preserving metadata.
    def test_bf16_arm_labels_and_policy_bytes(self):
        layout=dict(architecture='qwen35',layers=[dict(recurrent=True)]*18,
                    recurrent_layout=dict(state=128,inner=2048,conv=4,groups=16))
        policy=dict(state_bytes=20201472+1600,kv_bytes=6528,local_rates=((0,100),),import_fixed=1.5)
        before=copy.deepcopy(policy);bf16=growing.bf16_policy(layout,policy)
        self.assertEqual(policy,before)
        self.assertEqual(bf16,dict(before,state_bytes=10100736+1600))
        self.assertEqual(growing.arm_names(set(),True),['hybrid','local','hybrid-bf16'])
        self.assertEqual(growing.arm_names(set()),['hybrid','local'])
        self.assertEqual(growing.arm_names({10}),['hybrid','local','local_partition'])
        with self.assertRaisesRegex(Refusal,'common-journal rehearsal'):growing.arm_names({10},True)
        with self.assertRaisesRegex(Refusal,'requires Qwen35'):growing.bf16_policy(dict(layout,architecture='muse-glimmer'),policy)
        module=self.packet_module('mac_timing')
        command=module.command(Path('state'),Path('out'),Path('policy'),Path('fixture'),100000,'5G',Path('bf16'))
        self.assertIn('--include-bf16',command)
        self.assertEqual(command[command.index('--bf16-policy')+1],'bf16')

    # The paired calibration fits both dtypes under one shared cap; fixed metadata stays full width.
    def test_bf16_calibration_shared_budget(self):
        from traffic import TrafficBudget
        module=self.packet_module('calibrate_import')
        self.assertEqual(module.PAIRED_SIZES,[512,1024,2048,512,1024,2048])
        module.validate_plan('qwen35',2_000_000_000,module.PAIRED_SIZES)
        budget=TrafficBudget(2_000_000_000,dict(kv_bytes=34816,state_bytes=156894784,recurrent_tensor_bytes=156893184))
        for dtype in ('f32','bf16'):
            for n in module.PAIRED_SIZES:
                data=dict(tokens=[100]*n,return_delta=True,state_dtype=dtype)
                with budget.reserve('/extend',data,11*n,'POST') as receipt:
                    receipt['actual']=34848*n+156894784-(78446592 if dtype=='bf16' else 0)+4096
        self.assertLess(budget.status()['charged_bytes'],2_000_000_000)
        self.assertEqual(budget.status()['requests'],12)
        self.assertEqual(budget.status()['outstanding_bytes'],0)

    # The rehearsal transforms actual captured delta steps only, leaves f32 inputs untouched and requires imports.
    def test_bf16_common_journal_treatment(self):
        module=self.packet_module('replay_bf16')
        source=dict(acceptance_rule='session_append_replay_exact',probes=[dict(name='CHAIN',steps=[dict(delta='x',sha256='0'*64),dict(decode=[dict(n0=1,n1=2)])])])
        before=copy.deepcopy(source);converted=module.treatment(source)
        self.assertEqual(source,before)
        self.assertTrue(converted['bf16_replay'])
        self.assertEqual(converted['probes'][-1]['name'],'BF16_CHAIN')
        self.assertEqual(converted['probes'][-1]['steps'][0]['reencode_state_dtype'],'bf16')
        with self.assertRaisesRegex(Refusal,'source must be f32'):module.treatment(dict(source,state_dtype='bf16'))
        with self.assertRaisesRegex(Refusal,'recorded imports'):module.treatment(dict(source,probes=[dict(name='CHAIN',steps=[])]))

    # The same source/binary binding accepts an unchanged fit and rejects changed bytes or missing binaries.
    def test_fit_reuse_requires_source_and_binary_identity(self):
        module = self.packet_module('mac_timing')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'bin').mkdir(); binary=root/'bin/llama-server'; binary.write_bytes(b'old')
            client = dict(build_dir=str(root), binaries={'llama-server': 'unused'}, model_sha256='a'*64, build_record={'fixture': True})
            fit = module.import_identity(client)
            module.require_fit_identity(fit, client)
            bad = copy.deepcopy(fit); bad['import_path_sha256']['src/llama-io.cpp']='0'*64
            with self.assertRaisesRegex(Refusal, 'source/binary identity mismatch'):
                module.require_fit_identity(bad, client)
            binary.write_bytes(b'changed')
            with self.assertRaisesRegex(Refusal, 'source/binary identity mismatch'):
                module.require_fit_identity(fit, client)
            binary.unlink()
            with self.assertRaisesRegex(Refusal, 'missing import binary'):
                module.require_fit_identity(fit, client)

    # Metered traffic is reserved before open; rejected or interrupted exchanges cannot dispatch/reclaim credit.
    def test_calibration_budget_refuses_before_dispatch(self):
        from traffic import TrafficBudget, TrafficBudgetExceeded
        data = dict(n0=0, tokens=[1,2], return_delta=True)
        out = len(json.dumps(data).encode())
        limit = TrafficBudget.cleanup_headroom + 8*1024**2 + TrafficBudget.allowance + out
        budget = TrafficBudget(limit, dict(kv_bytes=16, state_bytes=100))
        node = growing.Node('http://unused', budget=budget)
        response = mock.MagicMock(); response.__enter__.return_value=response; response.read.return_value=b'{}'; response.headers={}
        with mock.patch.object(node, 'open', return_value=response) as opened:
            self.assertEqual(node.request('POST','/extend',data,binary=True), b'{}')
            with self.assertRaisesRegex(TrafficBudgetExceeded, 'before dispatch'):
                node.request('POST','/extend',data,binary=True)
            opened.assert_called_once()
        with mock.patch.object(node,'open') as opened:
            with self.assertRaisesRegex(TrafficBudgetExceeded,'snapshot download forbidden before dispatch'):
                node.download('/snapshot',Path('unused.bin'),10**9)
            opened.assert_not_called()
        state=budget.status()
        self.assertLessEqual(state['charged_bytes'],limit)
        self.assertEqual(state['outstanding_bytes'],0)
        interrupted=TrafficBudget(limit,dict(kv_bytes=16,state_bytes=100))
        with self.assertRaises(OSError), interrupted.reserve('/extend',data,out,'POST'):
            raise OSError('transport interrupted')
        with self.assertRaisesRegex(TrafficBudgetExceeded, 'before dispatch'):
            with interrupted.reserve('/extend',data,out,'POST'): pass
        self.assertEqual(interrupted.status()['charged_bytes'],limit-TrafficBudget.cleanup_headroom)

    # The caps apply independently of requested repetition/size flags; local warmup does not waive either cap.
    def test_calibration_plan_caps(self):
        module = self.packet_module('calibrate_import')
        self.assertEqual(module.validate_plan('qwen35',2_000_000_000,module.SIZES),2_000_000_000)
        self.assertEqual(module.validate_plan('muse-glimmer',3_000_000_000,module.SIZES),3_000_000_000)
        for architecture,budget,sizes in [('qwen35',2_000_000_001,module.SIZES),('muse-glimmer',3_000_000_001,module.SIZES),
                                         ('qwen35',0,module.SIZES),('qwen35',2_000_000_000,module.SIZES*2)]:
            with self.assertRaises(Refusal): module.validate_plan(architecture,budget,sizes)

    # An injected 100ns hash delay moves that leaf and TTFT once, while transfer and the explicit residual stay fixed.
    def test_nested_leaf_delay_reconciles_once(self):
        import timing
        def run(delay):
            timing.start(); clock=[0]
            with mock.patch.object(timing.time,'monotonic_ns',side_effect=lambda:clock[0]):
                with timing.span('bench.call',call_id='control'):
                    with timing.span('request'):
                        with timing.span('checksum.sha256'): clock[0]+=delay
                        with timing.span('transfer'): clock[0]+=20
                    clock[0]+=5
            rows=timing.rows(); result=timing.account(rows,0,clock[0],'control'); timing.stop()
            return rows,result
        original,base=run(1); rows,delayed=run(101)
        self.assertEqual(delayed['ttft_ns']-base['ttft_ns'],100)
        self.assertEqual(delayed['leaf_union_ns']-base['leaf_union_ns'],100)
        self.assertEqual(delayed['residual_ns'],5)
        self.assertEqual(next(r['duration_ns'] for r in rows if r['name']=='transfer'),20)
        self.assertEqual(len(delayed['leaf_ids']),2)
        bad=copy.deepcopy(rows); child=next(r for r in bad if r['name']=='transfer'); child['end_ns']+=1000; child['duration_ns']+=1000
        with self.assertRaisesRegex(ValueError,'outside parent'): timing.account(bad,0,126,'control')
        with self.assertRaisesRegex(ValueError,'duplicate'): timing.account(rows+[rows[0]],0,126,'control')
        remote=copy.deepcopy(rows[0]); remote['clock_domain']='remote'; remote['id']='remote'
        self.assertEqual(timing.account(rows+[remote],0,126,'control')['leaf_union_ns'],delayed['leaf_union_ns'])

    # A bounded span collector reports loss instead of claiming a complete trace.
    def test_span_overflow_is_not_success(self):
        import timing
        timing.start(1)
        with timing.span('first'): pass
        with timing.span('second'): pass
        with self.assertRaisesRegex(ValueError,'overflow'): timing.rows()
        timing.stop()

    # Offline extracts retain held-token/reconciliation fields and reject loss, bad nesting or changed accounting.
    def test_timing_extract_preserves_fields_and_refuses_incomplete(self):
        import timing
        module=self.packet_module('extract_timing')
        bounds=dict(stream_start_ns=10,first_nonempty_delta_ns=30)
        account=timing.account([],10,30,'call','domain')
        call=dict(arm='hybrid',turn=1,stage='turn',ttft_seconds=20e-9,local_tail_seconds=.4,
                  reconciliation_seconds=.7,import_seconds=3,import_stage_seconds=1,
                  import_final_save_apply_seconds=2,timing_accounting=account,
                  meter=dict(local_prompt_ms=17,timing_boundaries=bounds,timing_call_id='call',timing_clock_domain='domain'))
        trace=dict(instrumentation=dict(spans=[]),calls=[call])
        result=module.extract(trace,[])['calls'][0]
        self.assertEqual(result['local_tail_seconds'],.4)
        self.assertEqual(result['reconciliation_seconds'],.7)
        self.assertEqual(result['meter']['local_prompt_ms'],17)
        self.assertEqual(result['import_stage_seconds']+result['import_final_save_apply_seconds'],result['import_seconds'])
        changed=copy.deepcopy(trace);changed['calls'][0]['timing_accounting']['residual_ns']+=1
        with self.assertRaisesRegex(ValueError,'accounting mismatch'):module.extract(changed,[])
        row=dict(kind='span',clock_domain='native',id='1',parent_domain='native',parent_id='',
                 start_ns=1,end_ns=3,duration_ns=2,bytes=10,tokens=1)
        module.validate_native([row])
        bad=dict(row,parent_id='missing')
        with self.assertRaisesRegex(ValueError,'missing parent'):module.validate_native([bad])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'native.jsonl'
            marker=dict(kind='trace_checkpoint',dropped=0,writer_ok=True,records=1)
            path.write_text(json.dumps(row)+'\n'+json.dumps(marker)+'\n')
            self.assertEqual(module.native_rows(path),[row])
            for records in ([row],[row,dict(marker,dropped=1)],[row,dict(marker,records=2)]):
                path.write_text('\n'.join(map(json.dumps,records)))
                with self.assertRaises(ValueError):module.native_rows(path)
            path.write_text(json.dumps(row)+'\n{')
            with self.assertRaises(ValueError):module.native_rows(path)

    # Seven requested links, fixed base staging and bf16's tensor-byte reduction are independent table invariants.
    def test_link_table_and_wire_only_projection(self):
        module=self.packet_module('estimate')
        profiles=module.table()
        self.assertEqual(len(profiles),5)
        for profile in profiles:
            self.assertEqual([r['link_MB_s'] for r in profile['rows']],[2,3,5,10,20,30,50])
            for row in profile['rows']:
                self.assertTrue(row['base_offloads'])
                self.assertAlmostEqual(row['hybrid_total_ttft_s'],row['hybrid_base_ttft_s']+row['hybrid_turn_ttft_s'])
                self.assertLessEqual(len(row['policy_offloads']),37)
        f32=module.estimate('qwen',2e6); bf16=module.estimate('qwen',2e6,recurrent_dtype='bf16')
        self.assertEqual(f32['down_body_estimate_bytes']-bf16['down_body_estimate_bytes'],78_446_592)
        self.assertTrue(bf16['wire_projection_only'])
        minimum=module.threshold('qwen')*1e6
        self.assertFalse(module.estimate('qwen',minimum*.999999)['policy_offloads'])
        self.assertTrue(module.estimate('qwen',minimum*1.000001)['policy_offloads'])

    # Ten attention chunks at 100k carry one fixed recurrent state, not ten copies; headers remain an allowance.
    def test_packet_traffic_counts_final_state_once(self):
        path = Path(growing.__file__).parent / 'packets/r46/estimate.py'
        spec = importlib.util.spec_from_file_location('r46_estimate', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.delta_wire_estimate(100000, dict(kv_bytes=34816, state_bytes=157000000)), 3641840960)
        run = module.estimate('qwen', 5.19e6)
        self.assertEqual(run['policy_offloads'], [])
        self.assertLess(run['down_body_estimate_bytes'], 3.65e9)

    # A measured positive control is accepted; placeholders, wrong models, background and incomplete rates refuse.
    def test_pri31_fixture_admission(self):
        fixture = json.loads((Path(growing.__file__).parent / 'fixtures/m3-max-qwen-pri31.json').read_text())
        rates = validate_fixture(fixture, fixture['model_sha256'], 'qwen35')
        self.assertIn([100000, 65.3], rates)
        self.assertEqual(len(rates), 7)
        for fields in (dict(status='placeholder'), dict(priority=4), dict(override_used=True),
                       dict(normal_priority=False), dict(depths=[])):
            with self.subTest(fields=fields), self.assertRaises(Refusal):
                validate_fixture(dict(fixture, **fields), fixture['model_sha256'], 'qwen35')
        with self.assertRaisesRegex(Refusal, 'model mismatch'):
            validate_fixture(fixture, '0' * 64, 'qwen35')
        bad = copy.deepcopy(fixture)
        del bad['depths'][0]['delta_prefill_tok_s']['2000']
        with self.assertRaisesRegex(Refusal, 'incomplete'):
            validate_fixture(bad, fixture['model_sha256'], 'qwen35')

    # Archive pinning happens before tar parsing; a wrong file cannot be relabeled as a new rate source.
    def test_fixture_archive_checksum_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.tar.gz'
            path.write_bytes(b'not a tar archive')
            with self.assertRaisesRegex(Refusal, 'archive checksum mismatch'):
                from_archive(path, '0' * 64)

    # The Mac payload fixes two arms through none, preserves seed/outage, and requires a physical link label.
    def test_mac_packet_command_and_placeholder_refusal(self):
        path = Path(growing.__file__).parent / 'packets/r46/mac_timing.py'
        spec = importlib.util.spec_from_file_location('mac_timing', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        command = module.command(Path('state'), Path('run'), Path('policy'), Path('fixture'), 76000, '5G')
        for flag, value in (('--checkpoint-turns', 'none'), ('--depth', '76000'), ('--seed', '1'), ('--turns', '40'),
                            ('--disconnect-turn', '2'), ('--disconnect-turns', '3'), ('--rate-fixture', 'fixture')):
            self.assertEqual(command[command.index(flag) + 1], value)
        placeholder = json.loads((Path(growing.__file__).parent / 'fixtures/m3-max-glimmer-pri31.pending.json').read_text())
        client = dict(model_sha256='0' * 64, model_layout=dict(architecture='muse-glimmer'))
        with self.assertRaisesRegex(Refusal, 'measured PRI-31 fixture required'):
            module.admission(client, {}, {}, placeholder, {}, 'unused')
        fixture = json.loads((Path(growing.__file__).parent / 'fixtures/m3-max-qwen-pri31.json').read_text())
        client = dict(model_sha256=fixture['model_sha256'], model_layout=dict(architecture='qwen35'), backend='metal', context=196608)
        node = dict(backend='cuda', context=196608)
        fit = dict(model_sha256=fixture['model_sha256'], source_head='head')
        with mock.patch.object(module, 'require_fit_identity') as proof:
            self.assertEqual(module.admission(client, node, {}, fixture, fit, 'head')[1], 100000)
            proof.assert_called_once_with(fit, client)
        for bad_client, bad_node, bad_fit in ((dict(client, context=262144), node, fit),
                                             (client, dict(node, backend='cpu'), fit),
                                             (client, node, dict(fit, source_head='stale'))):
            with mock.patch.object(module, 'require_fit_identity'), self.assertRaises(Refusal):
                module.admission(bad_client, bad_node, {}, fixture, bad_fit, 'head')

    # Timing mode refuses capture options before reading metadata or creating artifacts.
    def test_timing_mode_refuses_capture_options(self):
        self.assertEqual(growing.checkpoint_turns('none', 40), set())
        for option in ('--state-probes', '--keep-checkpoint-states'):
            args = growing.parser().parse_args(['--client-state', 'absent', '--output', 'absent',
                '--link-label', 'CPU', '--checkpoint-turns', 'none', option])
            with mock.patch.object(growing, 'read_private') as read:
                with self.assertRaisesRegex(Refusal, 'timing-only mode forbids'):
                    growing.run(args)
                read.assert_not_called()

    # Timing observes an all-local production policy; the identical plan remains invalid for fidelity qualification.
    def test_timing_all_local_preserves_fidelity_route_gate(self):
        plan = growing.depth_plan(100000, 40, 1, 10875, 4506, 340, 14)
        policy = Policy(local_rates=((0, 2000),), fixed=100)
        result = growing.route_preflight(policy, plan, 2, 3, require_mix=False)
        self.assertEqual(result['counts'], dict(policy_local=37, policy_offload=0))
        with self.assertRaisesRegex(Refusal, 'degenerate connected policy mix'):
            growing.route_preflight(policy, plan, 2, 3)

    # The real request wrapper returns the import envelope without invoking snapshot download when capture is disabled.
    def test_timing_node_request_has_no_snapshot(self):
        events = growing.Events(None)
        node = growing.MeasuredNode('http://unused', 'private-fixture', 1, events=events, session='h', source='critical')
        node.last_bytes = 7
        with mock.patch.object(growing.Node, 'request', return_value=b'envelope'), mock.patch.object(node, 'download') as download:
            result = node.request('POST', '/v1/sessions/h/extend', dict(n0=0, tokens=[1], return_delta=True, prefix_hash_n0='0' * 64))
        self.assertEqual(result, b'envelope')
        download.assert_not_called()
        self.assertFalse(any('node_snapshot' in r or 'node_delta' in r for r in events.copy()))

    # Seed 1 fits Glimmer at 76k with framing up to 32 and a 2048-token margin; 80k must refuse.
    def test_glimmer_native_depth_margin(self):
        plan = growing.depth_plan(76000, 40, 1, 10875, 4506, 340, 32, 131072)
        self.assertEqual(len(plan['sequence']), 40)
        self.assertLessEqual(plan['final_depth'] + 2048, 131072)
        with self.assertRaisesRegex(Refusal, 'no turns truncated'):
            growing.depth_plan(80000, 40, 1, 10875, 4506, 340, 16, 131072)

    # SWA keeps ten controls; changed window bytes and 62/64 cannot pass a forged summary.
    @mock.patch.object(growing, 'validate_raw_positions')
    def test_window_gate_replaces_recurrent_without_weakening(self, raw):
        row = self.append_gate_row()
        row['state_component'] = 'window'
        row['probes']['CHAIN_WINDOW'] = row['probes'].pop('CHAIN_RECURRENT')
        row['gate_controls']['CHAIN_WINDOW-H'] = row['gate_controls'].pop('CHAIN_RECURRENT-H')
        for name in ('CHAIN_ATTENTION', 'CHAIN_WINDOW'):
            state = row['probes'][name]['roundtrips'][0]
            state['window_equal'] = state.pop('recurrent_equal')
        growing.validate_checkpoint(row, False, 1000)
        self.assertEqual(len(growing.append_replay_controls(row)), 10)
        for name in ('CHAIN_ATTENTION', 'CHAIN_WINDOW'):
            bad = copy.deepcopy(row)
            bad['probes'][name]['roundtrips'][0]['window_equal'] = False
            with self.assertRaisesRegex(Refusal, 'contradicts controls'):
                growing.validate_checkpoint(bad, False, 1000)
        bad = dict(row, exact=False, top1_agreed=62)
        with self.assertRaisesRegex(Refusal, 'verdict'):
            growing.validate_checkpoint(bad, False, 1000)

    # Invalid selections fail before private metadata reads, process launch or output writes.
    def test_checkpoint_turns_invalid_is_atomic(self):
        for value in ('', '0', '-1', '41', '20,20', '40,20', '20,', 'x', '1.5', ' 20', '２０'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'absent'
                args = growing.parser().parse_args(['--client-state', 'absent', '--output', str(output),
                    '--link-label', 'CPU', '--checkpoint-turns', value])
                with mock.patch.object(growing, 'read_private') as read, mock.patch.object(growing, 'start_process') as start:
                    with self.assertRaisesRegex(Refusal, 'invalid checkpoint turns'):
                        growing.run(args)
                    read.assert_not_called()
                    start.assert_not_called()
                self.assertFalse(output.exists())
        self.assertEqual(growing.checkpoint_turns('20,40', 40), {20, 40})
        self.assertEqual(growing.checkpoint_turns('7', 40), {7})

    # The default orchestration emits identical plan bytes to r43 for all permitted run lengths.
    # Live trace timestamps, UUIDs and the driver source hash are intentionally outside this control.
    def test_checkpoint_turns_default_run_byte_identical(self):
        args = growing.parser().parse_args(['--client-state', 'unused', '--output', 'unused', '--link-label', 'CPU'])
        self.assertIsNone(args.checkpoint_turns)
        for turns in range(1, 1001):
            legacy = set(range(10, turns + 1, 10)) | {turns}
            current = growing.checkpoint_turns(args.checkpoint_turns, turns)
            def run_plan(selected):
                return json.dumps([(arm, turn, turn in selected) for arm in ('hybrid', 'local', 'local_partition')
                    for turn in range(1, turns + 1)], separators=(',', ':')).encode()
            self.assertEqual(run_plan(current), run_plan(legacy))

    # Probe plans preserve local prefix KV through append imports; missing or overlapping node evidence refuses.
    def test_state_probe_plans_require_live_boundaries(self):
        source = dict(tokens=list(range(12)), replay={name: [dict(n0=0, n1=12, logits=True)] for name in ('A', 'B')})
        def event(a, b, catchup=False):
            return dict(session='h', status='ok', n0=a, n1=b, catchup=catchup,
                        node_snapshot=dict(path='snapshot', sha256='s'), node_delta=dict(path='delta', sha256='d'))
        trace = dict(calls=[dict(arm='hybrid', turn=1, session_id='h', meter=dict(chunks=[dict(status='imported', n1=9)]))],
                     events=[event(0, 4), event(4, 7, True), event(7, 9)])
        plans = {r['name']: r['steps'] for r in growing.state_probe_plans(source, trace, 1)}
        window = {r['name']: r['steps'] for r in growing.state_probe_plans(dict(source, state_component='window'), trace, 1)}
        self.assertIn('CHAIN_WINDOW', window)
        self.assertNotIn('CHAIN_RECURRENT', window)
        self.assertEqual(window['CHAIN_WINDOW'][-2]['replace'], 'window')
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

    # Captures follow the live extend; a moved boundary refuses and removes its temporary files.
    def test_node_snapshot_capture_refuses_boundary_race(self):
        with tempfile.TemporaryDirectory() as directory:
            for count in (4, 5):
                events = growing.Events(Path(directory))
                node = growing.MeasuredNode('http://unused', events=events, session='h', source='critical')
                snapshot = b'12345678' + count.to_bytes(4, 'little')
                response = io.BytesIO(snapshot)
                response.headers = {}
                with mock.patch.object(growing.Node, 'request', return_value=b'delta'), mock.patch.object(growing.Node, 'open', return_value=response) as request:
                    if count == 4:
                        self.assertEqual(node.request('POST', '/v1/sessions/s/extend',
                            dict(n0=0, tokens=[1, 2, 3, 4], prefix_hash_n0='h', return_delta=True), binary=True), b'delta')
                        self.assertEqual(Path(events.copy()[-1]['node_snapshot']['path']).read_bytes(), snapshot)
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

    # 62/64 teacher-forced top-1 fails with unchanged exact controls, including without free divergence.
    @mock.patch.object(growing, 'validate_raw_positions')
    def test_checkpoint_append_top1_62_fails(self, raw):
        row = self.append_gate_row()
        row.update(exact=False, mean_kl_nats=.01, max_abs_logit_diff=1, top1_agreed=63, kl_to_partition_ratio_status='zero_noise_band')
        growing.validate_checkpoint(row, False, 1000)
        for divergence in (-1, 0, 7, 63):
            bad = dict(row, top1_agreed=62, free_running_first_divergence=divergence)
            with self.assertRaisesRegex(Refusal, 'verdict'):
                growing.validate_checkpoint(bad, False, 1000)
            growing.validate_checkpoint(dict(bad, status='failed', exit_code=3), False, 1000)

    # L-H free divergence at 0 or any later position passes at the 63/64 top-1 floor.
    @mock.patch.object(growing, 'validate_raw_positions')
    def test_checkpoint_append_divergence_0_passes(self, raw):
        row = self.append_gate_row()
        row.update(exact=False, mean_kl_nats=.01, max_abs_logit_diff=1, top1_agreed=63, kl_to_partition_ratio_status='zero_noise_band')
        for divergence in (-1, *range(64)):
            growing.validate_checkpoint(dict(row, free_running_first_divergence=divergence), False, 1000)
        with self.assertRaisesRegex(Refusal, 'verdict'):
            growing.validate_checkpoint(dict(row, free_running_first_divergence=0, status='failed', exit_code=3), False, 1000)

    # Exact-control teacher metrics cannot hide a different independent greedy continuation.
    def test_checkpoint_append_control_free_divergence_fails(self):
        row = self.append_gate_row()
        self.assertTrue(growing.append_replay_controls(row)['CHAIN-H'])
        row['probes']['CHAIN']['H']['free_running_first_divergence'] = 3
        self.assertFalse(growing.append_replay_controls(row)['CHAIN-H'])

    # Each flag uses strict comparisons against all three like-for-like cells; no divergence ranks last.
    def test_checkpoint_fidelity_flags(self):
        row = dict(mean_kl_nats=3, p95_kl_nats=5, max_abs_logit_diff=7, top1_agreed=61, free_running_first_divergence=0)
        row['factorial'] = {name: dict(mean_kl_nats=2, p95_kl_nats=4, max_abs_logit_diff=6,
            top1_agreed=62, free_running_first_divergence=-1) for name in ('A-B1', 'A-B2', 'A-B')}
        self.assertTrue(all(growing.fidelity_flags(row).values()))
        for cell in row['factorial'].values():
            for key in growing.fidelity_flags(row):
                saved = cell[key]
                cell[key] = row[key]
                self.assertFalse(growing.fidelity_flags(row)[key])
                cell[key] = saved
        row.update(mean_kl_nats=1, p95_kl_nats=3, max_abs_logit_diff=5, top1_agreed=63, free_running_first_divergence=-1)
        self.assertFalse(any(growing.fidelity_flags(row).values()))
        row['factorial']['A-B1']['free_running_first_divergence'] = 0
        self.assertFalse(growing.fidelity_flags(row)['free_running_first_divergence'])

    # Flags cannot change passing, failing or heterogeneous verdicts, even when every statistic is flagged.
    @mock.patch.object(growing, 'validate_raw_positions')
    def test_checkpoint_fidelity_flag_never_changes_exit_code(self, raw):
        row = self.append_gate_row()
        keys = ('mean_kl_nats', 'p95_kl_nats', 'max_abs_logit_diff', 'top1_agreed', 'free_running_first_divergence')
        for flags in ({key: value for key in keys} for value in (False, True)):
            for top1, heterogeneous, status, code in ((64, False, 'passed', 0), (62, False, 'failed', 3),
                                                     (64, True, 'heterogeneous', 5), (62, True, 'heterogeneous', 5)):
                candidate = dict(row, exact=False, top1_agreed=top1, status=status, exit_code=code, fidelity_flag=flags)
                growing.validate_checkpoint(candidate, heterogeneous, 1000)
                self.assertEqual(candidate['exit_code'], code)

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

    # A lossy-origin snapshot cannot pass the exact gate, even when its measured distributions agree.
    @mock.patch.object(growing, 'validate_raw_positions')
    def test_bf16_import_never_exact(self, raw):
        row = dict(depth=1000, k=64, exact=True, top1_agreed=64, mean_kl_nats=0,
                   max_abs_logit_diff=0, free_running_first_divergence=-1, status='passed', exit_code=0)
        growing.validate_checkpoint(row, False, 1000)
        for status, code, exact in (('passed', 0, True), ('passed', 0, False), ('lossy', 5, True)):
            with self.assertRaises(Refusal):
                growing.validate_checkpoint(dict(row, state_dtype='bf16', lossy_import=True,
                                                 status=status, exit_code=code, exact=exact), False, 1000)
        growing.validate_checkpoint(dict(row, state_dtype='bf16', lossy_import=True,
                                         status='lossy', exit_code=5, exact=False), False, 1000)

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




class RecheckTests(unittest.TestCase):
    @staticmethod
    def fixture(root):
        run = root / 'run'
        (run / 'slots').mkdir(parents=True)
        (run / 'check-1').mkdir()
        tokens = list(range(12))
        saved = []
        for arm in ('hybrid', 'local', 'local_partition'):
            path = run / 'slots' / (arm + '-checkpoint-1.bin')
            path.write_bytes(arm.encode())
            saved.append(dict(path=str(path), sha256=growing.sha256(path), tokens=tokens))
        node, delta = run / 'node.snapshot', run / 'node.delta'
        node.write_bytes(b'node snapshot')
        delta.write_bytes(b'node delta')
        calls = [dict(arm='hybrid', turn=0, depth_before=0, depth_after=8, session_id='h',
                      meter=dict(prompt_depth=4, chunks=[dict(n0=0, n1=3, status='imported')])),
                 dict(arm='hybrid', turn=1, depth_before=8, depth_after=12, session_id='h',
                      meter=dict(prompt_depth=10, chunks=[dict(n0=8, n1=9, status='imported')]), journal_hash=prefix_hash(tokens))]
        calls += [dict(calls[-1], arm=arm) for arm in ('local', 'local_partition')]
        events = [dict(session='h', status='ok', n0=a, n1=b, catchup=catchup,
                       node_snapshot=dict(path=str(node), sha256=growing.sha256(node)),
                       node_delta=dict(path=str(delta), sha256=growing.sha256(delta)))
                  for a, b, catchup in ((0, 3, False), (3, 8, True), (8, 9, False))]
        trace = dict(schema='hybrid-growing-session-v3', status='aborted', turns=1, calls=calls, events=events,
                     client=dict(context=4096, model_sha256='model', backend='cpu'), node=dict(backend='cpu'),
                     heterogeneous=False, engine=dict(checker=dict(batch=256, ubatch=256, threads=2, threads_batch=2)))
        growing.private_json(run / 'trace.json', trace)
        growing.private_json(run / 'check-1/input.json', growing.checkpoint_source(trace['client'], saved, trace, 1))
        return run

    # Reconstruction binds fixture bytes, journal, replay and every referenced state hash before any checker/output side effect.
    def test_recheck_preserved_input_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.fixture(Path(directory))
            before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
            trace, source, fixture, verified = growing.verify_recheck(run / 'trace.json', 1)
            self.assertEqual(source['tokens'], list(range(12)))
            self.assertEqual(len(verified), 5)
            self.assertEqual(before, {p: p.read_bytes() for p in run.rglob('*') if p.is_file()})

    # Only files referenced by this checkpoint are inputs; an unused intermediate capture is not required.
    def test_recheck_unused_intermediate_snapshot_is_not_required(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.fixture(Path(directory))
            path = run / 'trace.json'
            trace = json.loads(path.read_text())
            trace['events'][0]['node_snapshot'] = dict(path=str(run / 'unused-missing.snapshot'), sha256='0' * 64)
            path.write_text(json.dumps(trace))
            _, _, _, verified = growing.verify_recheck(path, 1)
            self.assertNotIn(str(run / 'unused-missing.snapshot'), verified)

    # A running trace is not a stable preserved checkpoint source.
    def test_recheck_running_trace_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.fixture(Path(directory))
            path = run / 'trace.json'
            trace = json.loads(path.read_text())
            trace['status'] = 'running'
            path.write_text(json.dumps(trace))
            with self.assertRaisesRegex(Refusal, 'finished or aborted'):
                growing.verify_recheck(path, 1)

    # Missing snapshots cannot trigger computation or create an output; the positive fixture above uses the same verifier.
    def test_recheck_missing_snapshot_refuses(self):
        self.refusal('node.snapshot', None, 'missing preserved input')

    # A one-byte state change refuses at its original fixture hash, before checker launch and output creation.
    def test_recheck_modified_state_refuses(self):
        self.refusal('slots/hybrid-checkpoint-1.bin', b'changed', 'preserved input checksum mismatch')

    # The trace also anchors deltas; changing a delta cannot be hidden by otherwise valid saved states.
    def test_recheck_modified_delta_refuses(self):
        self.refusal('node.delta', b'changed', 'preserved input checksum mismatch')

    # Even valid JSON with changed serialization is not the preserved input emitted by the driver.
    def test_recheck_modified_input_refuses(self):
        self.refusal('check-1/input.json', b'{}', 'invalid preserved checkpoint input')

    # GPU command construction uses the original driver's 999 layers; this does not execute a GPU backend.
    def test_recheck_command_preserves_gpu_layers(self):
        for backend, expected in (('cpu', 0), ('metal', 999), ('cuda', 999)):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = self.fixture(root)
                trace_path, fixture = run / 'trace.json', run / 'check-1/input.json'
                trace, source = json.loads(trace_path.read_text()), json.loads(fixture.read_text())
                model, checker = root / 'model', root / 'checker'
                model.write_bytes(b'model')
                checker.write_bytes(b'checker')
                checker.chmod(0o700)
                trace['client']['backend'] = source['backend'] = backend
                trace['heterogeneous'] = source['heterogeneous'] = backend != 'cpu'
                trace['client']['model_sha256'] = source['model_sha256'] = growing.sha256(model)
                trace_path.write_text(json.dumps(trace))
                fixture.write_text(json.dumps(source, indent=2) + '\n')
                args = SimpleNamespace(trace=str(trace_path), turn=1, checker=str(checker), model=str(model), output=str(root / 'recheck'))
                process = SimpleNamespace(pid=123, returncode=2, poll=lambda: 2, wait=lambda: 2)
                with mock.patch.object(growing, 'model_identity', return_value=dict(model_layout=dict(vocab_size=1))), \
                     mock.patch.object(growing, 'start_process', return_value=process) as start:
                    with self.assertRaisesRegex(Refusal, 'checker produced no metrics'):
                        growing.recheck(args)
                self.assertEqual(start.call_args.args[0][-1], expected)
                record = json.loads((root / 'recheck/recheck.json').read_text())
                self.assertEqual((record['gpu_layers'], record['checker_exit_code'], record['status']), (expected, 2, 'refused'))

    # Output under the original run and existing output both refuse before reading inputs or launching the checker.
    def test_recheck_output_preserves_original_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root)
            existing = root / 'existing'
            existing.mkdir()
            for output in (run / 'new', existing):
                args = SimpleNamespace(trace=str(run / 'trace.json'), turn=1, output=str(output))
                with mock.patch.object(growing, 'verify_recheck') as verify:
                    with self.assertRaisesRegex(Refusal, 'new directory outside'):
                        growing.recheck(args)
                    verify.assert_not_called()

    def refusal(self, path, content, reason):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root)
            if content is None:
                (run / path).unlink()
            else:
                (run / path).write_bytes(content)
            before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
            args = growing.recheck_parser().parse_args(['--trace', str(run / 'trace.json'), '--turn', '1',
                '--checker', 'unused', '--model', 'unused', '--output', str(root / 'recheck')])
            with mock.patch.object(growing, 'start_process') as start:
                with self.assertRaises((Refusal, KeyError)) as caught:
                    growing.recheck(args)
                self.assertIn(reason, str(caught.exception))
                start.assert_not_called()
            self.assertFalse((root / 'recheck').exists())
            self.assertEqual(before, {p: p.read_bytes() for p in run.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()

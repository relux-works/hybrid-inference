"""CPU loopback controls for recovery and restart; no real provider qualification."""
from decimal import Decimal
from http.server import ThreadingHTTPServer
import json
import itertools
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import openrouter_cache as api
import openrouter_amendment1 as amendment
import openrouter_recovery as recovery
import openrouter_restart as restart
from test_openrouter_amendment1 import Handler


class RecoveryHandler(Handler):
    timeout_mode=False
    output_mode=None

    def do_POST(self):
        if RecoveryHandler.output_mode:
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            Handler.bodies.append(body);Handler.spend+=Handler.cost
            mode=RecoveryHandler.output_mode
            if mode in ('http-error','http-error-503'):
                self.send_response(503 if mode=='http-error-503' else 429);self.end_headers();self.wfile.write(b'{"error":"unavailable"}');return
            count=Handler.seen.get(body['session_id'],0);Handler.seen[body['session_id']]=count+1
            prompt=len(body['messages'][0]['content'][0]['text'].split())
            usage=dict(prompt_tokens=0 if mode=='wrong-depth' else prompt,completion_tokens=1025 if mode=='usage-bound' else 60 if mode=='overcap' else 2,cost=Handler.cost,
                       prompt_tokens_details=dict(cached_tokens=prompt-1065 if count and mode!='wrong-depth' else 0))
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            items=[dict(id='output-control',model='wrong/model' if mode=='wrong-model' else body['model'],provider='Phala',
                        choices=[dict(delta=dict(content='x'*1025 if mode=='guard' else 'OK'))]),dict(usage=usage,choices=[])]
            if mode=='partial-usage':items[0]['usage']=dict(usage,completion_tokens=1025,cost=Handler.cost/2)
            if mode=='guard-no-provider':
                items[0].pop('provider');items[1]['usage']['completion_tokens']=2000
            try:
                for item in items:self.wfile.write(b'data: '+json.dumps(item).encode()+b'\n\n');self.wfile.flush()
                self.wfile.write(b'data: [DONE]\n\n')
            except (BrokenPipeError,ConnectionResetError):pass
            return
        if not RecoveryHandler.timeout_mode:return super().do_POST()
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        Handler.bodies.append(body);Handler.spend+=Handler.cost
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
        item=dict(id='partial-timeout-id',provider='Phala',choices=[dict(delta=dict(content='OK'))])
        self.wfile.write(b'data: '+json.dumps(item).encode()+b'\n\n');self.wfile.flush()
        time.sleep(.15)


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),RecoveryHandler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'run';self.root.mkdir()
        self.key=self.root/'fake-key';self.secret='sk-or-v1-r15d-fake-control'
        self.key.write_text(self.secret);self.key.chmod(0o600)
        self.client=recovery.Client(self.key,'http://127.0.0.1:'+str(self.server.server_port),mock=True)
        self.candidate=next(c for c in json.loads(amendment.MANIFEST.read_text())['candidates'] if c['model']=='qwen/qwen3.8-27b' and c['endpoint']['provider_name']=='Phala')
        self.session=dict(candidate=self.candidate,cohort='v1',depth=32768,pauses=[0,0],session_id='r15d-test')
        recovery.Budget(self.root/'ledger.json',0,0).save()
        Handler.spend=0;Handler.cost=.000001;Handler.provider_override=None
        Handler.seen={};Handler.bodies=[];Handler.cached_known=True;Handler.usage_cost_absent=False
        RecoveryHandler.timeout_mode=False
        RecoveryHandler.output_mode=None

    def phase(self):
        with mock.patch.object(restart,'phase_sessions',return_value=[self.session]):
            return restart.Phase(self.root,'v1',self.client,until=time.time()+5000)

    def samples(self,amount):
        return [dict(at=str(i),usage_usd=str(amount)) for i in range(3)]

    # An out-of-reservation debit stops at one POST; the next scheduled call is never sent.
    def test_unresolvable_charge_stops_before_next_call(self):
        Handler.usage_cost_absent=True;Handler.cost=1
        phase=self.phase()
        with mock.patch.object(recovery.time,'sleep'):
            self.assertEqual(phase.run(),1)
        self.assertEqual(len(Handler.bodies),1)
        terminal=json.loads((phase.out/'terminal.json').read_text())
        self.assertIn('unresolvable charge',terminal['reason'])
        self.assertTrue((phase.out/'unresolved-0000.json').exists())
        unresolved=json.loads((phase.out/'unresolved-0000.json').read_text())
        self.assertEqual(unresolved['sample_count'],1)
        self.assertEqual(unresolved['samples'][0]['usage_usd'],'1')
        self.assertEqual(recovery.Budget.load(self.root/'ledger.json').receipts,{})

    # The same missing-cost path may continue only after stable, bounded cumulative usage.
    def test_charge_resolved_within_maximum_continues(self):
        Handler.usage_cost_absent=True
        phase=self.phase()
        with mock.patch.object(recovery.time,'sleep'):
            self.assertEqual(phase.run(),0)
        self.assertEqual(len(Handler.bodies),2)
        self.assertTrue(all(r['charge_resolution']['resolved'] for r in phase.rows))
        ledger=recovery.Budget.load(self.root/'ledger.json')
        self.assertEqual(ledger.known,Decimal('.000002'))
        self.assertGreater(ledger.committed,ledger.known)
        self.assertTrue(all(not r['accepted'] for r in phase.rows))

    # A real loopback stream timeout preserves its generation/TTFT journal and is an availability error.
    def test_timeout_counts_against_availability_and_keeps_evidence(self):
        RecoveryHandler.timeout_mode=True;phase=self.phase()
        with mock.patch.object(recovery,'deadline_for_depth',return_value=.05),mock.patch.object(recovery,'settled_samples',side_effect=lambda client,*args:self.samples(Handler.spend)):
            phase.call(self.session,0)
        row=phase.rows[0]
        self.assertTrue(row['timed_out']);self.assertFalse(row['accepted'])
        self.assertEqual(row['generation_id'],'partial-timeout-id')
        self.assertIsNotNone(row['ttft_seconds'])
        self.assertIn('partial-timeout-id',(phase.out/'stream-0000.jsonl').read_text())
        self.assertEqual(recovery.availability(phase.rows),dict(attempts=1,successes=0,errors=1,timeouts=1,rate=0))

    # Reported normal usage remains a valid positive control for the recovered-error paths.
    def test_loopback_pinned_success_has_warm_cache_result(self):
        phase=self.phase();self.assertEqual(phase.run(),0)
        self.assertTrue(all(r['accepted'] for r in phase.rows))
        self.assertEqual(phase.rows[0]['cached_tokens'],0)
        self.assertGreater(phase.rows[1]['cached_tokens'],0)
        self.assertEqual(len(recovery.cache_rows(phase.rows)),1)
        for body in Handler.bodies:
            self.assertEqual(body['provider']['only'],[self.candidate['endpoint']['tag']])
            self.assertFalse(body['provider']['allow_fallbacks'])

    # A mismatched provider excludes cache evidence but cannot stop the study after a bounded charge.
    def test_provider_mismatch_excludes_cache_and_continues(self):
        Handler.provider_override='Chutes';phase=self.phase()
        self.assertEqual(phase.run(),0);self.assertEqual(len(Handler.bodies),2)
        self.assertTrue(all(not r['accepted'] for r in phase.rows))
        self.assertIn('provider differs',phase.rows[0]['error'])

    # Full usage above the requested16 remains cache-eligible, successful availability, and permits the next POST.
    def test_over_cap_completion_is_accepted_with_flag_and_continues(self):
        RecoveryHandler.output_mode='overcap';phase=self.phase()
        self.assertEqual(phase.run(),0)
        self.assertEqual(len(Handler.bodies),2)
        self.assertTrue(all(r['accepted'] and r['completion_over_cap']==60 for r in phase.rows))
        self.assertEqual(recovery.availability(phase.rows)['rate'],1)
        self.assertEqual(restart.price_metrics(phase.rows,self.candidate,[32768])['accepted'],1)
        self.assertEqual(phase.rows[0]['cost_usd'],Handler.cost)
        self.assertTrue(all(b['max_tokens']==16 for b in Handler.bodies))

    # The paid path reserves1024 output tokens at the live price while still requesting16.
    def test_reservation_covers_1024_token_completion(self):
        phase=self.phase();phase.call(self.session,0)
        row=phase.rows[0];p=row['pricing']
        prompt=min(len(Handler.bodies[0]['messages'][0]['content'][0]['text'].encode())+512,self.candidate['endpoint']['context_length'])
        expected=Decimal(prompt)*max(Decimal(p['prompt'])+Decimal(p.get('input_cache_write','0')),Decimal(p['input_cache_read']))+Decimal(1024)*Decimal(p['completion'])
        self.assertEqual(Decimal(row['maximum_cost_usd']),expected)
        self.assertEqual(row['reserved_output_tokens'],1024)
        self.assertEqual(Handler.bodies[0]['max_tokens'],16)

    # A loopback byte guard closes the stream, retains partial evidence, settles its charge, and permits a follow-up.
    def test_cancel_guard_past_bound_preserves_unknown_usage_and_continues(self):
        RecoveryHandler.output_mode='guard';phase=self.phase()
        with mock.patch.object(recovery,'settled_samples',side_effect=lambda client,*args:self.samples(Handler.spend)):
            self.assertEqual(phase.run(),0)
        self.assertEqual(len(Handler.bodies),2)
        row=phase.rows[0]
        self.assertTrue(row['output_guard_triggered']);self.assertTrue(row['completion_over_cap_censored'])
        self.assertEqual(row['streamed_output_utf8_bytes'],1025)
        self.assertIsNone(row['completion_over_cap']);self.assertIsNone(row.get('completion_tokens'))
        self.assertFalse(row['accepted']);self.assertTrue(row['charge_resolution']['resolved'])
        journal=(phase.out/'stream-0000.jsonl').read_text()
        self.assertIn('output-control',journal);self.assertNotIn('prompt_tokens',journal)

    # The guard allows the exact byte bound, cancels at1025 reported tokens, and closes its response context.
    def test_output_guard_boundary_and_reported_tokens_close_response(self):
        trace=recovery.StreamTrace(self.root/'guard.jsonl',self.secret);self.addCleanup(trace.close)
        trace.record(b'data: '+json.dumps(dict(choices=[dict(delta=dict(content='x'*1024))])).encode()+b'\n')
        self.assertFalse(trace.guard_triggered)
        response=mock.Mock(status=200)
        response.__enter__=mock.Mock(return_value=response);response.__exit__=mock.Mock(return_value=False)
        response.__iter__=mock.Mock(return_value=iter([b'data: '+json.dumps(dict(usage=dict(completion_tokens=1025))).encode()+b'\n']))
        with self.assertRaisesRegex(api.Refusal,'output guard canceled stream'):
            with recovery.RecordedResponse(response,trace) as recorded:list(recorded)
        response.__exit__.assert_called_once()
        self.assertTrue(trace.guard_triggered);self.assertEqual(trace.output_tokens,1025)
        self.assertIn('1025',(self.root/'guard.jsonl').read_text())

    # A guard triggered by complete final usage still counts HTTP200 as successful availability.
    def test_guard_with_full_usage_keeps_over_cap_success(self):
        RecoveryHandler.output_mode='usage-bound';phase=self.phase()
        self.assertEqual(phase.run(),0);self.assertEqual(len(Handler.bodies),2)
        self.assertTrue(all(r['accepted'] and r['output_guard_triggered'] and r['completion_over_cap']==1025 for r in phase.rows))
        self.assertEqual(recovery.availability(phase.rows)['rate'],1)

    # Reviewer F1 remains excluded from cache; r15j permits the next POST after the bounded charge.
    def test_named_negative_guard_completed_absent_provider_refuses_cache_and_continues(self):
        RecoveryHandler.output_mode='guard-no-provider';phase=self.phase()
        self.assertEqual(phase.run(),0);self.assertEqual(len(Handler.bodies),2)
        row=phase.rows[0]
        self.assertTrue(row['output_guard_triggered']);self.assertTrue(row['usage_complete'])
        self.assertEqual(row['observed_providers'],[]);self.assertFalse(row['accepted'])
        self.assertEqual(row['completion_over_cap'],2000)
        self.assertIn('serving provider differs or is absent',row['error'])
        self.assertEqual(recovery.availability(phase.rows)['rate'],1)

    # Missing identity wins for both poppable presets; the same completed row with Phala remains accepted.
    def test_reviewer_guard_and_legacy_presets_require_observed_provider(self):
        for preset in ('output guard canceled stream','completion cap exceeded','HTTP error',None):
            with self.subTest(preset=preset):
                row=dict(http_status=200,error=preset,output_guard_triggered=True,usage_complete=True,observed_models=[self.candidate['model']],observed_providers=[],
                         usage=dict(prompt_tokens=32768,completion_tokens=2000,cost=.001),completion_tokens=2000)
                recovery.classify_call(row,32768,self.candidate['model'],self.candidate['endpoint'])
                self.assertFalse(row['accepted']);self.assertIn('serving provider differs or is absent',row['error'])
        row.update(observed_models=[self.candidate['model']],observed_providers=['Phala'],error='output guard canceled stream')
        recovery.classify_call(row,32768,self.candidate['model'],self.candidate['endpoint'])
        self.assertTrue(row['accepted']);self.assertEqual(row['completion_over_cap'],2000)

    # A parse failure replaces an obsolete output label while an actual identity refusal remains primary.
    def test_parse_failure_surfaces_usage_error_instead_of_stale_output_label(self):
        for preset in ('completion cap exceeded','output guard canceled stream'):
            row=dict(http_status=200,error=preset,observed_models=[self.candidate['model']],observed_providers=['Phala'],usage=None)
            recovery.classify_call(row,32768,self.candidate['model'],self.candidate['endpoint'])
            self.assertFalse(row['accepted']);self.assertIn('usage parsing failed',row['error'])
            row.update(error=preset,observed_providers=[])
            recovery.classify_call(row,32768,self.candidate['model'],self.candidate['endpoint'])
            self.assertIn('serving provider differs or is absent',row['error'])
            self.assertIn('usage parsing failed',row['usage_error'])

    # Progressive usage in a canceled stream cannot settle its final charge or become a cache result.
    def test_guard_partial_usage_requires_delta_settlement(self):
        RecoveryHandler.output_mode='partial-usage';phase=self.phase()
        with mock.patch.object(recovery,'settled_samples',side_effect=lambda client,*args:self.samples(Handler.spend)):
            self.assertEqual(phase.run(),0)
        row=phase.rows[0]
        self.assertFalse(row['accepted']);self.assertFalse(row['usage_complete'])
        self.assertTrue(row['completion_over_cap_censored']);self.assertIsNone(row['completion_over_cap'])
        self.assertIsNone(row.get('cost_usd'));self.assertEqual(row['partial_usage_cost_usd'],Handler.cost/2)
        self.assertEqual(Decimal(row['charge_resolution']['resolved_observed_charge_usd']),Decimal(str(Handler.cost)))
        self.assertEqual(len(Handler.bodies),2)

    # A complete over-cap receipt can be migrated without another POST, ledger debit, or seed cache sample.
    def test_historical_over_cap_reclassification_preserves_receipt_and_corrects_availability(self):
        RecoveryHandler.output_mode='overcap';phase=self.phase();phase.call(self.session,0)
        row=phase.rows[0];row.update(accepted=False,error='completion cap exceeded');row.pop('completion_over_cap')
        path=phase.out/'call-0000.json';recovery.atomic_json(path,row);original=path.read_bytes()
        recovery.atomic_json(self.root/'workflow.json',dict(status='failed'))
        ledger=(self.root/'ledger.json').read_bytes()
        with mock.patch.object(restart,'phase_sessions',return_value=[self.session]):result=restart.reclassify_over_caps(self.root)
        updated=json.loads(path.read_text())
        self.assertTrue(updated['accepted']);self.assertEqual(updated['completion_over_cap'],60)
        self.assertEqual(result['reclassified'],['v1/call-0000.json'])
        self.assertEqual((self.root/'reclassification-r15f/v1-call-0000.json').read_bytes(),original)
        self.assertEqual((self.root/'ledger.json').read_bytes(),ledger);self.assertEqual(len(Handler.bodies),1)
        own=next(p for p in json.loads((self.root/'analysis.json').read_text())['providers'] if p['provider']==row['provider'])
        self.assertEqual(own['whole_window_availability']['rate'],1)
        self.assertEqual(own['output_noncompliance']['over_cap_calls'],1)
        self.assertEqual(own['phases']['v1']['v1']['cache_sample_attempts'],0)

    # Historical output-only repair cannot bypass a planted wrong model or touch the original row/ledger.
    def test_named_negative_reclassification_refuses_identity_violation(self):
        RecoveryHandler.output_mode='overcap';phase=self.phase();phase.call(self.session,0)
        row=phase.rows[0];row.update(accepted=False,error='completion cap exceeded',observed_models=['wrong/model'])
        path=phase.out/'call-0000.json';recovery.atomic_json(path,row);original=path.read_bytes()
        recovery.atomic_json(self.root/'workflow.json',dict(status='failed'))
        ledger=(self.root/'ledger.json').read_bytes()
        with mock.patch.object(restart,'phase_sessions',return_value=[self.session]):
            with self.assertRaisesRegex(api.Refusal,'failed identity, depth or usage validation'):restart.reclassify_over_caps(self.root)
        self.assertEqual(path.read_bytes(),original);self.assertEqual((self.root/'ledger.json').read_bytes(),ledger)
        self.assertFalse((self.root/'reclassification-r15f').exists());self.assertEqual(len(Handler.bodies),1)

    # A settled legacy cap receipt without observed provider cannot be promoted or rewrite financial/original evidence.
    def test_named_negative_reclassification_absent_provider_refuses(self):
        RecoveryHandler.output_mode='overcap';phase=self.phase();phase.call(self.session,0)
        row=phase.rows[0];row.update(accepted=False,error='completion cap exceeded',observed_providers=[],serving_provider=[])
        path=phase.out/'call-0000.json';recovery.atomic_json(path,row);original=path.read_bytes()
        recovery.atomic_json(self.root/'workflow.json',dict(status='failed'));ledger=(self.root/'ledger.json').read_bytes()
        with mock.patch.object(restart,'phase_sessions',return_value=[self.session]):
            with self.assertRaisesRegex(api.Refusal,'failed identity, depth or usage validation'):restart.reclassify_over_caps(self.root)
        self.assertEqual(path.read_bytes(),original);self.assertEqual((self.root/'ledger.json').read_bytes(),ledger)
        self.assertFalse((self.root/'reclassification-r15f').exists());self.assertEqual(len(Handler.bodies),1)

    # An explicit wrong model still stops after one settled request; output non-compliance cannot mask identity.
    def test_named_negative_wrong_model_identity_halts(self):
        RecoveryHandler.output_mode='wrong-model';phase=self.phase()
        self.assertEqual(phase.run(),1);self.assertEqual(len(Handler.bodies),1)
        self.assertIn('identity violation',phase.reason)
        self.assertIn('model differs',phase.rows[0]['error'])
        self.assertFalse(phase.rows[0]['accepted'])

    # Depth outside tolerance excludes cache evidence, retains its charge and allows the next call.
    def test_measured_depth_outside_tolerance_excludes_cache_and_continues(self):
        RecoveryHandler.output_mode='wrong-depth';phase=self.phase()
        self.assertEqual(phase.run(),0);self.assertEqual(len(Handler.bodies),2)
        self.assertIn('depth outside',phase.rows[0]['error'])

    # Under r15g missing identity is refused regardless of preset error, and remains an availability failure.
    def test_http_failure_without_identity_retains_availability_classification(self):
        row=dict(accepted=False,http_status=429,error='HTTP error')
        recovery.classify_call(row,32768,self.candidate['model'],self.candidate['endpoint'])
        self.assertIn('serving provider differs or is absent',row['error']);self.assertFalse(row['accepted'])
        self.assertEqual(recovery.availability([row])['errors'],1)

    # A settled429 with no generation or usage is an availability failure, not a completed-response identity violation.
    def test_rate_limited_no_generation_zero_debit_continues(self):
        RecoveryHandler.output_mode='http-error';Handler.cost=0;phase=self.phase()
        with mock.patch.object(recovery.time,'sleep'):
            self.assertEqual(phase.run(),0)
        self.assertEqual(len(Handler.bodies),2)
        self.assertTrue(all(not row['accepted'] and row['http_status']==429 and row['generation_id'] is None and row['usage'] is None for row in phase.rows))
        self.assertTrue(all(row['charge_resolution']['resolved_observed_charge_usd']=='0' for row in phase.rows))
        self.assertEqual(recovery.availability(phase.rows)['errors'],2)
        self.assertEqual(recovery.availability(phase.rows)['timeouts'],0)
        self.assertGreater(phase.budget.committed,phase.budget.known)

    # The429 transport exception cannot hide observed identity, usage, generation evidence or a guard-completed response.
    def test_named_negative_rate_limit_exception_cannot_mask_completed_identity(self):
        row=dict(http_status=429,generation_id=None,usage=None,observed_providers=[],observed_models=[],stream=[])
        self.assertTrue(recovery.rate_limited_without_generation(row))
        for changes in (dict(http_status=200),dict(generation_id='gen-wrong'),dict(usage={'cost':0}),
                        dict(observed_providers=['wrong']),dict(observed_models=['wrong/model']),dict(stream=[{}]),dict(output_guard_triggered=True)):
            with self.subTest(changes=changes):self.assertFalse(recovery.rate_limited_without_generation(dict(row,**changes)))

    # The historical supervisor's timeout flag must not turn a completed HTTP429 into a transport timeout.
    def test_historical_rate_limit_watchdog_flag_counts_only_as_availability_error(self):
        row=dict(http_status=429,generation_id=None,usage=None,accepted=False,timed_out=True,error='HTTP error',observed_providers=[],observed_models=[])
        self.assertEqual(recovery.availability([row]),dict(attempts=1,successes=0,errors=1,timeouts=0,rate=0))
        self.assertTrue(row['timed_out'])

    # The stream identity check accepts exactly the frozen endpoint alias, never an arbitrary dated suffix.
    def test_frozen_stream_model_identity_positive_and_wrong_suffix_negative(self):
        allowed=recovery.allowed_models(self.candidate['model'],self.candidate['endpoint'])
        canonical=self.candidate['endpoint']['name'].split(' | ')[1]
        self.assertIn(canonical,allowed);self.assertNotIn(self.candidate['model']+'-wrong-date',allowed)

    # Stable lower/upper edges pass; instability, reset and over-maximum deltas do not.
    def test_usage_delta_boundaries_and_unstable_negative(self):
        self.assertTrue(recovery.resolve_delta(self.samples('2.42'),'2','.42')['resolved'])
        self.assertTrue(recovery.resolve_delta(self.samples('1.999999999'),'2','.42')['resolved'])
        for samples in (self.samples('2.420001'),self.samples('1.9'),self.samples('2')[:2],[dict(usage_usd=x) for x in ('2','2','2.1')]):
            with self.assertRaisesRegex(api.Refusal,'unresolvable charge'):recovery.resolve_delta(samples,2,'.42')

    # Every resolved unknown keeps its full reservation and receipts prevent double-accounting after a crash.
    def test_recovery_receipt_idempotence_and_shared_45_cap(self):
        budget=recovery.Budget(self.root/'ledger.json','44','44')
        first=budget.recovered_call(self.samples('44.1'),'.42','v1/0000')
        loaded=recovery.Budget.load(budget.path)
        self.assertEqual(loaded.recovered_call(self.samples(999),'.42','v1/0000'),first)
        self.assertEqual(loaded.committed,Decimal('44.42'))
        loaded.reserve('44.1',Decimal('.58'))
        with self.assertRaisesRegex(api.Refusal,'budget'):loaded.reserve('44.1',Decimal('.580001'))

    # Known charges are never hidden when they exceed their reservation; continuation is refused.
    def test_reported_cost_over_maximum_stops_and_persists(self):
        budget=recovery.Budget.load(self.root/'ledger.json')
        with self.assertRaisesRegex(api.Refusal,'exceeds reservation'):budget.reported_call('.5','.42','v1/0000')
        loaded=recovery.Budget.load(budget.path)
        self.assertEqual(loaded.known,Decimal('.5'))
        self.assertFalse(loaded.receipts['v1/0000']['within_maximum'])

    # Replaying a watchdog reconciliation with an existing receipt cannot debit the same call twice.
    def test_watchdog_recovery_uses_receipt_without_another_get(self):
        directory=self.root/'v1';directory.mkdir()
        intent=dict(sequence=0,maximum_cost_usd='.42',accepted=False)
        recovery.atomic_json(directory/'intent-0000.json',intent)
        budget=recovery.Budget.load(self.root/'ledger.json');budget.reported_call('.2','.42','v1/0000')
        with mock.patch.object(self.client,'get',side_effect=AssertionError('must not resample')):
            restart.recover_interrupted(self.root,'v1',self.client)
        self.assertEqual(recovery.Budget.load(budget.path).known,Decimal('.2'))
        self.assertFalse(json.loads((directory/'call-0000.json').read_text())['accepted'])

    # This instrument detects a planted secret, while redacted journals contain neither key nor Authorization value.
    def test_named_negative_key_planted_in_log_is_caught(self):
        with self.assertRaisesRegex(api.Refusal,'secret detected'):api.assert_no_secret('Authorization: Bearer '+self.secret,self.secret)
        path=self.root/'trace.jsonl';trace=recovery.StreamTrace(path,self.secret)
        trace.record(('data: '+json.dumps(dict(id=self.secret,Authorization='Bearer '+self.secret,choices=[]))+'\n').encode());trace.close()
        self.assertNotIn(self.secret,path.read_text());self.assertIn('[REDACTED]',path.read_text())

    # The fixed depth table permits minute-scale prefill and rejects depths outside the declared design.
    def test_depth_deadlines_and_28_gib_disk_floor(self):
        self.assertEqual([recovery.deadline_for_depth(d) for d in (32768,131072,250000,524288,1000000)],[180,300,600,1200,2400])
        self.assertEqual(recovery.MIN_FREE_BYTES,30064771072)
        for depth in (0,1000001):
            with self.assertRaisesRegex(api.Refusal,'depth outside'):recovery.deadline_for_depth(depth)

    # A pause that cannot fit is omitted explicitly, never shortened and never POSTed.
    def test_deadline_cutoff_skips_without_compressing_pause(self):
        phase=self.phase();phase.end=time.time()+179+600
        self.assertIsNone(phase.call(self.session,0))
        self.assertEqual(Handler.bodies,[]);self.assertIn('unchanged pause',phase.skips[0]['reason'])

    # A slot that fit the old600s recovery reserve must skip if the longer bounded settlement cannot fit.
    def test_cutoff_reserves_extended_settlement_window(self):
        phase=self.phase();phase.end=time.time()+180+1190
        self.assertIsNone(phase.call(self.session,0))
        self.assertEqual(Handler.bodies,[]);self.assertIn('unchanged pause',phase.skips[0]['reason'])

    # A disk value above 28 decimal GB but below 28 GiB still refuses the paid request.
    def test_28_gib_floor_refuses_before_post(self):
        phase=self.phase()
        with mock.patch.object(api.shutil,'disk_usage',return_value=mock.Mock(free=28500000000)):
            with self.assertRaisesRegex(api.Refusal,'free disk below28 GiB'):phase.call(self.session,0)
        self.assertEqual(Handler.bodies,[])

    # Sustained low headroom skips this cell without any POST after the bounded1200s wait.
    def test_28_5_gib_headroom_refuses_before_post(self):
        phase=self.phase();now=[0]
        with mock.patch.object(api.shutil,'disk_usage',return_value=mock.Mock(free=int(28.4*1024**3))),mock.patch.object(recovery.time,'monotonic',side_effect=lambda:now[0]),mock.patch.object(recovery.time,'sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+seconds)):
            self.assertIsNone(phase.call(self.session,0))
            self.assertIn('requires28.5 GiB',phase.skips[-1]['reason'])
        self.assertEqual(Handler.bodies,[])
        self.assertEqual(now[0],1200)

    # A real loopback POST waits until headroom returns, then sends the original next index exactly once.
    def test_disk_headroom_dip_recovers_and_continues(self):
        phase=self.phase();now=[0];reads=[0];sleeps=[];ledger=(self.root/'ledger.json').read_bytes()
        def disk(path):
            reads[0]+=1
            if reads[0]<=2:
                self.assertEqual(Handler.bodies,[]);self.assertFalse(list(phase.out.glob('intent-*.json')))
                self.assertEqual((self.root/'ledger.json').read_bytes(),ledger)
                return mock.Mock(free=int(28.4*1024**3))
            return mock.Mock(free=30*1024**3)
        def sleep(seconds):sleeps.append(seconds);now[0]+=seconds
        with mock.patch.object(api.shutil,'disk_usage',side_effect=disk),mock.patch.object(recovery.time,'monotonic',side_effect=lambda:now[0]),mock.patch.object(recovery.time,'sleep',side_effect=sleep):
            phase.call(self.session,0)
        self.assertEqual(sleeps,[60,60]);self.assertEqual(len(Handler.bodies),1)
        self.assertTrue(phase.rows[0]['accepted']);self.assertEqual(phase.rows[0]['call_index'],0)
        events=[json.loads(line) for line in (self.root/'disk-waits.jsonl').read_text().splitlines()]
        self.assertEqual((events[0]['event'],events[-1]['event'],events[-1]['reason']),('start','end','recovered'))
        self.assertEqual(events[-1]['elapsed_seconds'],120);self.assertEqual(events[-1]['free_bytes'],30*1024**3)

    # Low headroom for the entire window stops with recorded start/end and no intent, POST or ledger change.
    def test_named_negative_disk_headroom_never_recovers_skips_cell(self):
        phase=self.phase();now=[0];ledger=(self.root/'ledger.json').read_bytes()
        with mock.patch.object(api.shutil,'disk_usage',return_value=mock.Mock(free=int(28.4*1024**3))),mock.patch.object(recovery.time,'monotonic',side_effect=lambda:now[0]),mock.patch.object(recovery.time,'sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+seconds)):
            self.assertIsNone(phase.call(self.session,0))
            self.assertIn('1200s',phase.skips[-1]['reason'])
        self.assertEqual(now[0],1200);self.assertEqual(Handler.bodies,[])
        self.assertFalse(list(phase.out.glob('intent-*.json')));self.assertEqual((self.root/'ledger.json').read_bytes(),ledger)
        events=[json.loads(line) for line in (self.root/'disk-waits.jsonl').read_text().splitlines()]
        self.assertEqual((events[0]['event'],events[-1]['event'],events[-1]['reason']),('start','end','timeout'))

    # Dropping under28GiB during a soft wait stops at that check, without sleeping out the remaining window.
    def test_named_negative_disk_hard_floor_stops_immediately(self):
        now=[0];events=[]
        with mock.patch.object(api.shutil,'disk_usage',side_effect=[mock.Mock(free=int(28.4*1024**3)),mock.Mock(free=28*1024**3-1)]),mock.patch.object(recovery.time,'monotonic',side_effect=lambda:now[0]),mock.patch.object(recovery.time,'sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+seconds)):
            with self.assertRaisesRegex(api.Refusal,'28 GiB hard floor'):recovery.wait_for_disk(self.root,on_event=events.append)
        self.assertEqual(now[0],60);self.assertEqual(events[-1]['reason'],'hard-floor')

    # A heartbeat below soft headroom cannot leave a sticky STOP; a hard-floor observation still sets it.
    def test_heartbeat_soft_dip_waits_but_hard_floor_sets_stop(self):
        with mock.patch.object(api.shutil,'disk_usage',return_value=mock.Mock(free=int(28.4*1024**3))):data=restart.heartbeat(self.root,self.client)
        self.assertFalse((self.root/'STOP').exists());self.assertEqual(data['disk_status'],'waiting-headroom')
        with mock.patch.object(api.shutil,'disk_usage',return_value=mock.Mock(free=28*1024**3-1)):data=restart.heartbeat(self.root,self.client)
        self.assertTrue((self.root/'STOP').exists());self.assertEqual(data['disk_status'],'hard-floor')

    # A wait after a completed call cannot delay its JSON or charge receipt, and resumes at the next index.
    def test_disk_wait_preserves_previous_call_and_resumes_next_index(self):
        phase=self.phase();now=[0];reads=[0]
        def disk(path):
            reads[0]+=1
            if reads[0] in (3,4):
                self.assertTrue(json.loads((phase.out/'call-0000.json').read_text())['accepted'])
                self.assertIn('v1/0000',recovery.Budget.load(self.root/'ledger.json').receipts)
                self.assertEqual(len(Handler.bodies),1);self.assertFalse((phase.out/'intent-0001.json').exists())
                return mock.Mock(free=int(28.4*1024**3))
            return mock.Mock(free=30*1024**3)
        with mock.patch.object(api.shutil,'disk_usage',side_effect=disk),mock.patch.object(recovery.time,'monotonic',side_effect=lambda:now[0]),mock.patch.object(recovery.time,'sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+seconds)):
            self.assertEqual(phase.run(),0)
        self.assertEqual([r['call_index'] for r in phase.rows],[0,1]);self.assertEqual(len(Handler.bodies),2)
        self.assertTrue(all(r['accepted'] for r in phase.rows));self.assertEqual(now[0],120)

    # A late recovery cannot bypass the fixed cutoff; remaining time is checked again before any metadata or POST.
    def test_disk_recovery_rechecks_fixed_phase_cutoff(self):
        phase=self.phase();now=[0];wall=time.time();phase.end=wall+1600;reads=[0]
        def disk(path):
            reads[0]+=1;return mock.Mock(free=int(28.4*1024**3) if reads[0]<=5 else 30*1024**3)
        with mock.patch.object(api.shutil,'disk_usage',side_effect=disk),mock.patch.object(recovery.time,'time',side_effect=lambda:wall+now[0]),mock.patch.object(recovery.time,'monotonic',side_effect=lambda:now[0]),mock.patch.object(recovery.time,'sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+seconds)):
            self.assertIsNone(phase.call(self.session,0))
        self.assertEqual(now[0],300);self.assertEqual(Handler.bodies,[])
        self.assertFalse(list(phase.out.glob('intent-*.json')));self.assertIn('insufficient phase time',phase.skips[0]['reason'])

    # A dip during metadata waits before intent, then repeats live price/spend preflight after recovery.
    def test_disk_dip_during_metadata_refreshes_preflight(self):
        phase=self.phase();now=[0];reads=[0]
        def disk(path):
            reads[0]+=1
            if reads[0]==2:
                self.assertEqual(Handler.bodies,[]);return mock.Mock(free=int(28.4*1024**3))
            return mock.Mock(free=30*1024**3)
        with mock.patch.object(api.shutil,'disk_usage',side_effect=disk),mock.patch.object(recovery.time,'monotonic',side_effect=lambda:now[0]),mock.patch.object(recovery.time,'sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+seconds)):
            phase.call(self.session,0)
        self.assertEqual(now[0],60);self.assertEqual(phase.preflights,2)
        self.assertEqual(len(Handler.bodies),1);self.assertTrue(phase.rows[0]['accepted'])

    # An operator STOP during a headroom wait is not cleared or turned into a recovery.
    def test_named_negative_operator_stop_during_disk_wait_is_preserved(self):
        phase=self.phase();now=[0]
        def sleep(seconds):now[0]+=seconds;(self.root/'STOP').touch()
        with mock.patch.object(api.shutil,'disk_usage',return_value=mock.Mock(free=int(28.4*1024**3))),mock.patch.object(recovery.time,'monotonic',side_effect=lambda:now[0]),mock.patch.object(recovery.time,'sleep',side_effect=sleep):
            with self.assertRaisesRegex(api.Refusal,'operator STOP'):phase.call(self.session,0)
        self.assertTrue((self.root/'STOP').exists());self.assertEqual(Handler.bodies,[]);self.assertEqual(now[0],60)
        events=[json.loads(line) for line in (self.root/'disk-waits.jsonl').read_text().splitlines()]
        self.assertEqual(events[-1]['reason'],'stopped')

    # Ready follow-ups run before another seed, preventing the original all-cold scheduling failure.
    def test_due_warm_call_precedes_next_cold_seed(self):
        phase=self.phase();seen=[]
        def call(s,index,previous_end):seen.append((s['session_id'],index));return time.monotonic()-1
        with mock.patch.object(phase,'call',side_effect=call):phase.interleave([self.session,dict(self.session,session_id='second')])
        self.assertEqual(seen[:3],[('r15d-test',0),('r15d-test',1),('second',0)])

    # The heartbeat records an in-flight deadline and spend without waiting for that request to return.
    def test_heartbeat_during_inflight_call_and_metadata_failure(self):
        recovery.atomic_json(self.root/'progress.json',dict(phase='v1',last_completed_call={'sequence':3},current_call={'sequence':4},next_call_time=None,next_call_reason='awaiting response',current_request_deadline_at='deadline'))
        with mock.patch.object(self.client,'get',side_effect=TimeoutError('metadata unavailable')):
            data=restart.heartbeat(self.root,self.client)
        self.assertEqual(data['last_call']['sequence'],3);self.assertEqual(data['current_call']['sequence'],4)
        self.assertEqual(data['known_observed_usd'],'0');self.assertIsNone(data['next_call_time'])
        self.assertEqual(data['current_request_deadline_at'],'deadline')
        self.assertIn('live_usage_error',data);self.assertNotIn(self.secret,(self.root/'heartbeat.json').read_text())

    # Zero hit rate is a priced result; all smoke and seed observations are excluded from cache inference.
    def test_zero_hit_remains_eligible_and_smoke_seeds_are_excluded(self):
        row=dict(accepted=True,call_index=1,prompt_tokens=32000,cached_tokens=0,new_tokens_estimate=1000,depth=32768,pause_target_seconds=0,pause_actual_seconds=0,cohort='v1')
        self.assertEqual(recovery.eligible(self.candidate,[row]),'eligible')
        rows=[dict(row,call_index=0,cached_tokens=30000),dict(row,smoke=True,cached_tokens=30000),row]
        measured=restart.price_metrics(rows,self.candidate,[32768])
        p=self.candidate['endpoint']['pricing']
        self.assertAlmostEqual(measured['effective_usd_per_1000_by_depth']['32768'],1000*(32768*(float(p['prompt'])+float(p.get('input_cache_write',0)))+512*float(p['completion'])))
        self.assertEqual(measured['cache_sample_attempts'],1);self.assertEqual(measured['token_hit_rate'],0)
        self.assertEqual(measured['pause_levels']['0']['n'],1)

    # Both Alibaba views retain the raw rows; own availability is identical across views and includes failures.
    def test_analysis_with_and_without_alibaba_same_whole_window_rule(self):
        alibaba=json.loads(json.dumps(self.candidate));alibaba['endpoint'].update(provider_name='Alibaba',tag='alibaba')
        names={amendment.identity(c):c for c in (self.candidate,alibaba)}
        rows=[dict(provider=n,session_id=n+'-core',observed_models=[c['model']],observed_providers=[c['endpoint']['provider_name']],phase='amendment',model=c['model'],cohort='core',depth=1000000,call_index=1,accepted=True,prompt_tokens=999000,cached_tokens=0,new_tokens_estimate=1065,pause_target_seconds=8,pause_actual_seconds=8) for n,c in names.items()]
        rows.append(dict(rows[0],accepted=False,timed_out=True,error='deadline',call_index=0))
        with mock.patch.object(restart,'candidates',return_value=names),mock.patch.object(restart,'read_rows',return_value=rows):data=restart.analysis(self.root,final=True)
        comparison=data['million_token_comparison']['qwen/qwen3.8-27b']
        self.assertEqual(len(comparison['with_alibaba']['providers']),2)
        self.assertEqual(len(comparison['without_alibaba']['providers']),1)
        phala=next(p for p in data['providers'] if p['provider_name']=='Phala')
        self.assertEqual(phala['whole_window_availability']['rate'],.5)
        self.assertEqual(phala['eligibility'],'excluded-availability')
        self.assertEqual(len(rows),3)

    # A below-expected counter may settle later; no accounting or POST is allowed during the wait.
    def test_stale_counter_waits_for_three_settled_samples(self):
        clock=mock.Mock();clock.now=0
        def sleep(seconds):clock.now+=seconds
        client=mock.Mock(secret=self.secret)
        client.get.side_effect=[{'data':{'usage':x}} for x in (0,0,0,.2,.2,.2)]
        with mock.patch.object(recovery.time,'monotonic',side_effect=lambda:clock.now),mock.patch.object(recovery.time,'sleep',side_effect=sleep):
            samples=recovery.settled_samples(client,'.2','.1',self.root/'settlement.json')
        self.assertEqual([s['usage_usd'] for s in samples],['0.2']*3)
        self.assertEqual(client.get.call_count,6)
        self.assertEqual(recovery.Budget.load(self.root/'ledger.json').known,0)
        self.assertEqual(Handler.bodies,[])

    def settlement_fixture(self,values,expected,maximum,timeout=None):
        now=[0];client=mock.Mock(secret=self.secret)
        client.get.side_effect=[{'data':{'usage':value}} for value in values]
        args={} if timeout is None else dict(timeout_seconds=timeout)
        with mock.patch.object(recovery.time,'monotonic',side_effect=lambda:now[0]),mock.patch.object(recovery.time,'sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+seconds)):
            samples=recovery.settled_samples(client,expected,maximum,self.root/'poll-evidence.json',**args)
        return samples,client,now[0]

    # Exact0153 phase+watchdog values need690s at15s cadence: the counter catches up, then three readings resolve zero debit.
    def test_exact_0153_lag_then_three_stable_samples_resolves_zero(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/settlement-0153.json').read_text())
        values=[s['usage_usd'] for s in fixture['samples']]
        samples,client,elapsed=self.settlement_fixture(values,fixture['expected_usd'],fixture['maximum_usd'])
        proof=recovery.resolve_delta(samples,fixture['expected_usd'],fixture['maximum_usd'])
        self.assertEqual(client.get.call_count,47);self.assertEqual(elapsed,690)
        self.assertEqual(proof['observed_delta_usd'],'-4.2E-9')
        self.assertEqual(proof['resolved_observed_charge_usd'],'0')
        self.assertEqual(proof['retained_reservation_usd'],'0.0204141')
        self.assertEqual(len(json.loads((self.root/'poll-evidence.json').read_text())['samples']),47)

    # The neighbouring late counter may include a non-zero charge, which must be retained as the measured debit.
    def test_lag_then_stable_positive_debit_within_maximum(self):
        samples,client,elapsed=self.settlement_fixture(['7.212300721']*44+['7.6483417282']*3,'7.6383417282','.0204141')
        proof=recovery.resolve_delta(samples,'7.6383417282','.0204141')
        self.assertEqual(proof['resolved_observed_charge_usd'],'0.0100000000')
        self.assertEqual(client.get.call_count,47);self.assertEqual(elapsed,690)

    # An overshoot after initial lag stops on that first excessive value, without waiting for stability.
    def test_named_negative_lag_then_overshoot_stops_immediately(self):
        with self.assertRaisesRegex(api.Refusal,'cumulative delta exceeds reservation'):
            self.settlement_fixture(['7.212300721']*3+['7.6587558283'],'7.6383417282','.0204141')
        persisted=json.loads((self.root/'poll-evidence.json').read_text())
        self.assertEqual(len(persisted['samples']),4)
        self.assertEqual(persisted['samples'][-1]['usage_usd'],'7.6587558283')

    # A counter inside the charge range that never holds three readings fails at the finite deadline.
    def test_named_negative_counter_never_stabilizes_stops(self):
        with self.assertRaisesRegex(api.Refusal,'did not settle within recovery deadline'):
            self.settlement_fixture(['7.6383417282','7.6383417283']*2,'7.6383417282','.0204141',timeout=60)
        self.assertEqual(len(json.loads((self.root/'poll-evidence.json').read_text())['samples']),4)

    # A permanently stale/reset counter stops at the bounded deadline rather than fabricating a zero charge.
    def test_stale_counter_unresolved_by_deadline_stops(self):
        now=[0]
        client=mock.Mock(secret=self.secret);client.get.return_value={'data':{'usage':0}}
        with mock.patch.object(recovery.time,'monotonic',side_effect=lambda:now[0]),mock.patch.object(recovery.time,'sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+seconds)):
            with self.assertRaisesRegex(api.Refusal,'did not settle within recovery deadline'):
                recovery.settled_samples(client,'.2','.1',self.root/'settlement.json',timeout_seconds=45)
        self.assertEqual(now[0],45);self.assertEqual(Handler.bodies,[])
        self.assertEqual(recovery.Budget.load(self.root/'ledger.json').receipts,{})

    # An excessive observed debit never receives a grace period.
    def test_excessive_delta_stops_on_first_sample(self):
        client=mock.Mock(secret=self.secret);client.get.return_value={'data':{'usage':.31}}
        with mock.patch.object(recovery.time,'sleep') as sleep:
            with self.assertRaisesRegex(api.Refusal,'exceeds reservation'):
                recovery.settled_samples(client,'.2','.1',self.root/'settlement.json')
        self.assertEqual(client.get.call_count,1);sleep.assert_not_called()

    # Resume preserves the original record, continues the next index, and never bills a duplicate index.
    def test_resume_continues_next_index_and_preserves_financial_ledger(self):
        self.session['pauses']=[0,0,0]
        first=self.phase();first.call(self.session,0)
        original=(first.out/'call-0000.json').read_bytes()
        with mock.patch.object(restart,'phase_sessions',return_value=[self.session]):
            resumed=restart.Phase(self.root,'v1',self.client,until=time.time()+5000,resume=True)
        self.assertEqual(resumed.run(),0)
        self.assertEqual([r['call_index'] for r in resumed.rows],[0,1,2])
        self.assertEqual(len(Handler.bodies),3)
        self.assertEqual((first.out/'call-0000.json').read_bytes(),original)
        self.assertEqual(recovery.Budget.load(self.root/'ledger.json').reported,Decimal('.000003'))

    # Missing settled receipts make resume fail before any new POST or ledger write.
    def test_resume_refuses_unsettled_receipt(self):
        phase=self.phase()
        recovery.atomic_json(phase.out/'call-0000.json',dict(sequence=0,session_id='missing',call_index=0))
        with self.assertRaisesRegex(api.Refusal,'resume has unresolved charge'):
            restart.Phase(self.root,'v1',self.client,resume=True)
        self.assertEqual(Handler.bodies,[])

    # Target-group rates retain late observations, while matched rates do not mislabel a long delay as8s.
    def test_late_pause_is_separate_from_log_matched_rates(self):
        row=dict(accepted=True,call_index=1,prompt_tokens=32000,cached_tokens=31000,new_tokens_estimate=1000,depth=32768,pause_target_seconds=8,pause_actual_seconds=600,cohort='core')
        measured=restart.price_metrics([row],self.candidate,[32768])
        self.assertEqual(measured['pause_levels']['8']['n'],1)
        self.assertEqual(measured['matched_target_pause_levels']['8']['n'],0)
        row['pause_actual_seconds']=8.01
        self.assertEqual(restart.price_metrics([row],self.candidate,[32768])['matched_target_pause_levels']['8']['n'],1)

    # Resumed custody retains old signed-era data and ledger, then passes --resume to both sequential children.
    def test_supervisor_resume_keeps_history_and_shared_spend(self):
        first=self.phase();first.call(self.session,0)
        recovery.atomic_json(self.root/'workflow.json',dict(status='failed',pid=123,source_commit='old'))
        recovery.atomic_json(self.root/'terminal.json',dict(status='failed',reason='old guard'))
        children=[]
        def spawn(argv,**kwargs):
            self.assertIn('--resume',argv)
            phase=argv[argv.index('--phase')+1]
            directory=self.root/phase;directory.mkdir(exist_ok=True)
            recovery.atomic_json(directory/'terminal.json',dict(status='done',attempts=0))
            child=mock.Mock(pid=11000+len(children));child.wait.return_value=0;children.append(child)
            return child
        with mock.patch.object(restart,'phase_sessions',return_value=[self.session]),mock.patch.object(restart,'LOCK_PATH',self.root/'mock.lock'),mock.patch.object(recovery,'Client',return_value=self.client),mock.patch.object(recovery,'settled_samples',return_value=self.samples('.000001')),mock.patch.object(restart,'public_snapshot'),mock.patch.object(restart.subprocess,'Popen',side_effect=spawn),mock.patch.object(restart.subprocess,'run'),mock.patch.object(restart.subprocess,'check_output',side_effect=['b'*40,'']):
            self.assertEqual(restart.supervise(self.root,resume=True),0)
        self.assertEqual(len(children),2)
        self.assertEqual(len(Handler.bodies),1)
        self.assertEqual(recovery.Budget.load(self.root/'ledger.json').reported,Decimal('.000001'))
        self.assertEqual(len(list(self.root.glob('workflow-history-*.json'))),1)
        self.assertEqual(len(list(self.root.glob('terminal-history-*.json'))),1)
        self.assertTrue(json.loads((self.root/'workflow.json').read_text())['resumed'])

    # The supervisor sequences both child PIDs under one lease and persists terminal custody and heartbeat.
    def test_supervisor_sequences_children_and_records_terminal_custody(self):
        self.supervisor_fixture(watchdog=False)

    # A fixed phase cutoff settles interrupted intents before the next phase is permitted.
    def test_supervisor_watchdog_settles_then_launches_amendment(self):
        self.supervisor_fixture(watchdog=True)

    # Exact0246 replay keeps its settled zero charge, rejects cache evidence and advances both indices.
    def test_sequence_0246_unknown_model_no_usage_continues_as_availability_failure(self):
        fixture=json.loads((api.HERE/'fixtures/unknown-model-0246.json').read_text())
        result={k:fixture[k] for k in ('http_status','generation_id','usage','error','stream','observed_models','observed_providers','usage_complete','timed_out')}
        phase=self.phase();Handler.cost=0
        with mock.patch.object(recovery,'stream',return_value=result),mock.patch.object(recovery.time,'sleep'):
            self.assertEqual(phase.run(),0)
        self.assertEqual([r['call_index'] for r in phase.rows],[0,1])
        self.assertEqual(recovery.availability(phase.rows)['errors'],2)
        for row in phase.rows:
            self.assertFalse(row['accepted']);self.assertNotIn('halt_class',row)
            self.assertEqual(row['observed_models'],['unknown'])
            self.assertEqual(row['charge_resolution']['resolved_observed_charge_usd'],'0')
        self.assertEqual(len(phase.budget.receipts),2)

    # Missing/unknown model with valid usage is successful availability but cannot be cache-priced.
    def test_unknown_or_absent_model_full_usage_is_excluded_from_cache_only(self):
        for models in ([],['unknown'],[' UNKNOWN ']):
            row=dict(http_status=200,observed_models=models,observed_providers=['Phala'],usage=dict(prompt_tokens=32768,completion_tokens=60,cost=.001))
            recovery.classify_call(row,32768,self.candidate['model'],self.candidate['endpoint'])
            self.assertFalse(row['accepted']);self.assertEqual(row['identity_status'],'unproven')
            self.assertIn('model unknown or absent',row['error']);self.assertNotIn('halt_class',row)
            self.assertEqual(recovery.availability([row])['rate'],1)
            self.assertEqual(row['completion_over_cap'],60)

    # Unknown markers and provider errors must not hide any concrete contradictory model observation.
    def test_named_negative_known_wrong_model_alongside_unknown_still_halts(self):
        phase=self.phase()
        result=dict(http_status=200,observed_models=['unknown','wrong/model'],observed_providers=[],usage=None,error='output guard canceled stream')
        with mock.patch.object(recovery,'stream',return_value=result),mock.patch.object(recovery,'settled_samples',return_value=self.samples(0)):
            self.assertEqual(phase.run(),1)
        self.assertEqual(len(phase.rows),1);self.assertEqual(phase.rows[0]['halt_class'],'identity')
        self.assertIn('proven model identity violation',phase.reason)

    # Excessive reported debit is persisted, halts after one request and cannot be converted to a cell skip.
    def test_named_negative_excessive_reported_debit_halts_study(self):
        Handler.cost=1;phase=self.phase()
        self.assertEqual(phase.run(),1);self.assertEqual(len(Handler.bodies),1)
        self.assertIn('reported cost exceeds reservation',phase.reason)
        ledger=recovery.Budget.load(self.root/'ledger.json')
        self.assertEqual(ledger.known,Decimal(1));self.assertFalse(ledger.receipts['v1/0000']['within_maximum'])

    # A malformed/unavailable remote preflight never admits a POST or changes the ledger; later cells may run.
    def test_preflight_remote_failures_skip_cell_then_next_cell_continues(self):
        for mode in ('get','endpoint','price','spend'):
            with self.subTest(mode=mode):
                registered=[self.session]+[dict(self.session,session_id='after-'+m) for m in ('get','endpoint','price','spend')]
                with mock.patch.object(restart,'phase_sessions',return_value=registered):
                    phase=restart.Phase(self.root,'v1',self.client,until=time.time()+5000,resume=(self.root/'v1').exists())
                before=(self.root/'ledger.json').read_bytes();original=self.client.get
                def get(path):
                    if mode=='get':raise restart.URLError('metadata unavailable')
                    if mode=='spend' and path=='/key':return {'data':{}}
                    data=original(path)
                    if '/models/' in path:
                        if mode=='endpoint':data['data']['endpoints']=[]
                        if mode=='price':
                            for endpoint in data['data']['endpoints']:endpoint['pricing']['completion']='invalid'
                    return data
                count=len(Handler.bodies)
                with mock.patch.object(self.client,'get',side_effect=get):self.assertIsNone(phase.call(self.session,0))
                self.assertEqual(len(Handler.bodies),count);self.assertEqual((self.root/'ledger.json').read_bytes(),before)
                self.assertTrue(phase.skips)
                phase.call(dict(self.session,session_id='after-'+mode),0)
                self.assertEqual(len(Handler.bodies),count+1);self.assertTrue(phase.rows[-1]['accepted'])

    # A soft1200s timeout ends only that cell; restored headroom admits another cell without replay.
    def test_disk_wait_expiry_does_not_halt_next_cell(self):
        phase=self.phase()
        with mock.patch.object(phase,'wait_for_disk',side_effect=recovery.DiskHeadroomTimeout('headroom wait expired')):
            self.assertIsNone(phase.call(self.session,0))
        self.assertEqual(Handler.bodies,[]);self.assertEqual(len(phase.skips),1)
        phase.call(dict(self.session,session_id='next-cell'),0)
        self.assertEqual(len(Handler.bodies),1)

    # Evidence failure is an instrument halt after its charge is settled, even if provider usage is valid.
    def test_named_negative_stream_evidence_failure_halts_after_settlement(self):
        phase=self.phase();result=dict(http_status=200,observed_models=[self.candidate['model']],observed_providers=['Phala'],
                                      usage=dict(prompt_tokens=31768,completion_tokens=2,cost=.000001),instrument_failure='stream evidence write failed')
        with mock.patch.object(recovery,'stream',return_value=result):self.assertEqual(phase.run(),1)
        self.assertEqual(len(phase.rows),1);self.assertFalse(phase.rows[0]['accepted'])
        self.assertIn('instrument violation',phase.reason);self.assertIn('v1/0000',phase.budget.receipts)

    # Trace disk-write faults are distinguished from provider stream failures, rather than swallowed as per-call errors.
    def test_stream_evidence_write_fault_sets_instrument_marker(self):
        trace=recovery.StreamTrace(self.root/'broken.jsonl',self.secret);trace.handle.close();trace.handle=mock.Mock()
        trace.handle.write.side_effect=OSError('disk write failed')
        with self.assertRaisesRegex(OSError,'disk write failed'):trace.record(b'data: {"model":"unknown"}\n')
        self.assertEqual(trace.evidence_error,'stream evidence write failed')

    # The registered number of failed attempts bounds a cell, and cannot prevent an independent cell from running.
    def test_failing_cell_exhausts_registered_limit_without_halting_study(self):
        phase=self.phase();bad=dict(self.session,session_id='failed-cell');good=dict(self.session,session_id='good-cell')
        original=phase.call
        def call(s,index,previous_end=None):
            RecoveryHandler.output_mode='http-error' if s['session_id']=='failed-cell' else None
            return original(s,index,previous_end)
        Handler.cost=0
        with mock.patch.object(phase,'call',side_effect=call),mock.patch.object(recovery.time,'sleep'):phase.interleave([bad,good])
        self.assertEqual([r['call_index'] for r in phase.rows if r['session_id']=='failed-cell'],[0,1])
        self.assertTrue(all(r['accepted'] for r in phase.rows if r['session_id']=='good-cell'))
        self.assertEqual(len(Handler.bodies),4)

    # Starting too late omits v1 explicitly and still starts amendment under its unchanged cutoff.
    def test_late_phase_start_skips_v1_and_launches_amendment(self):
        with mock.patch.dict(restart.PHASE_ENDS,{'v1':restart.timestamp(time.time()-1)}):
            self.supervisor_fixture(watchdog=False,late=True)

    # A child crash remains an instrument failure and prevents amendment spending, even if prior charges settled.
    def test_named_negative_supervisor_child_crash_still_halts(self):
        self.supervisor_fixture(watchdog=False,crash=True)

    # A phase cutoff cannot hand off to amendment when interrupted-request reconciliation is still unresolved.
    def test_named_negative_cutoff_unresolved_charge_prevents_amendment(self):
        self.supervisor_fixture(watchdog=True,unresolved=True)

    # A cutoff between recording a known wrong model and the child's fatal check must not permit phase handoff.
    def test_named_negative_cutoff_cannot_hide_recorded_wrong_model(self):
        self.supervisor_fixture(watchdog=True,wrong_model=True)

    # Low soft headroom at startup cannot halt custody; the child owns the paid-call wait and hard-floor guard.
    def test_supervisor_soft_headroom_startup_does_not_halt(self):
        with mock.patch.object(api.shutil,'disk_usage',return_value=mock.Mock(free=int(28.4*1024**3))):
            self.supervisor_fixture(watchdog=False)

    # The exact80-sample witness settles at its first stable triplet, including the10.4nanodollar shortfall.
    def test_exact_0321_samples_resolve_zero_at_first_stable_triplet(self):
        fixture=json.loads((api.HERE/'fixtures/settlement-0321.json').read_text())
        values=[s['usage_usd'] for s in fixture['samples']]
        samples,client,elapsed=self.settlement_fixture(values,fixture['expected_usd'],fixture['maximum_usd'])
        proof=recovery.resolve_delta(samples,fixture['expected_usd'],fixture['maximum_usd'])
        self.assertEqual(len(values),80);self.assertEqual(client.get.call_count,5);self.assertEqual(elapsed,60)
        self.assertEqual(proof['observed_delta_usd'],'-1.04E-8')
        self.assertEqual(proof['resolved_observed_charge_usd'],'0')
        self.assertEqual(proof['retained_reservation_usd'],'0.12613632')

    # The exact0321 counter sequence resolves the failed call and permits the next registered index.
    def test_exact_0321_resolved_no_generation_failure_continues(self):
        fixture=json.loads((api.HERE/'fixtures/settlement-0321.json').read_text());expected=fixture['expected_usd']
        recovery.Budget(self.root/'ledger.json',expected,expected).save();phase=self.phase()
        values=iter(itertools.chain([expected],[s['usage_usd'] for s in fixture['samples']],itertools.repeat(fixture['samples'][-1]['usage_usd'])))
        original=self.client.get;now=[0]
        def get(path):return {'data':{'usage':next(values)}} if path=='/key' else original(path)
        failed=dict(http_status=429,error='HTTP error',generation_id=None,usage=None,observed_providers=[],observed_models=[])
        with mock.patch.object(self.client,'get',side_effect=get),mock.patch.object(recovery,'stream',return_value=failed),mock.patch.object(recovery.time,'monotonic',side_effect=lambda:now[0]),mock.patch.object(recovery.time,'sleep',side_effect=lambda s:now.__setitem__(0,now[0]+s)):
            self.assertEqual(phase.run(),0)
        self.assertEqual([r['call_index'] for r in phase.rows],[0,1])
        self.assertTrue(all(r['charge_resolution']['resolved_observed_charge_usd']=='0' for r in phase.rows))
        self.assertEqual(recovery.availability(phase.rows)['errors'],2)
        self.assertFalse(phase.budget.permanent_receipts())

    # A real loopback503 without a generation uses the same zero-debit continuation as a429.
    def test_http_503_no_generation_settles_and_continues(self):
        RecoveryHandler.output_mode='http-error-503';Handler.cost=0;phase=self.phase()
        with mock.patch.object(recovery.time,'sleep'):self.assertEqual(phase.run(),0)
        self.assertEqual(len(Handler.bodies),2)
        self.assertTrue(all(r['http_status']==503 and r['charge_resolution']['resolved_observed_charge_usd']=='0' for r in phase.rows))
        self.assertEqual(recovery.availability(phase.rows)['errors'],2)
        self.assertEqual(recovery.availability(phase.rows)['timeouts'],0)

    # A timeout without a generation settles and advances, while retaining its availability and timeout errors.
    def test_timeout_no_generation_settles_and_continues(self):
        phase=self.phase();failed=dict(http_status=200,error='request exceeded wall-clock deadline',generation_id=None,usage=None,timed_out=True)
        with mock.patch.object(recovery,'stream',return_value=failed),mock.patch.object(recovery.time,'sleep'):
            self.assertEqual(phase.run(),0)
        self.assertEqual(len(phase.rows),2)
        self.assertTrue(all(r['charge_resolution']['resolved_observed_charge_usd']=='0' for r in phase.rows))
        self.assertEqual(recovery.availability(phase.rows)['errors'],2)
        self.assertEqual(recovery.availability(phase.rows)['timeouts'],2)

    def permanent_fixture(self,calls):
        self.session['pauses']=[0]*calls;phase=self.phase();now=[0];reads=[0];original=self.client.get
        def get(path):
            if path!='/key':return original(path)
            reads[0]+=1;return {'data':{'usage':'0.000000001' if reads[0]%2 else '0'}}
        failed=dict(http_status=503,error='HTTP error',generation_id=None,usage=None)
        with mock.patch.object(self.client,'get',side_effect=get),mock.patch.object(recovery,'stream',return_value=failed),mock.patch.object(recovery.time,'monotonic',side_effect=lambda:now[0]),mock.patch.object(recovery.time,'sleep',side_effect=lambda s:now.__setitem__(0,now[0]+s)):
            status=phase.run()
        return phase,status,now[0]

    # Two real1200s virtual polling windows retain both full maxima, then continue under both cumulative caps.
    def test_never_stable_no_generation_reserves_permanently_and_continues(self):
        phase,status,elapsed=self.permanent_fixture(2)
        self.assertEqual(status,0);self.assertEqual(elapsed,2400);self.assertEqual(len(phase.rows),2)
        ledger=recovery.Budget.load(self.root/'ledger.json');self.assertEqual(len(ledger.permanent_receipts()),2)
        self.assertEqual(ledger.committed,sum(Decimal(r['maximum_cost_usd']) for r in phase.rows))
        self.assertTrue(all(r['charge_status']=='unresolved-reserved' and not r['accepted'] for r in phase.rows))
        self.assertTrue(all(len(r['charge_resolution']['settlement_evidence']['samples'])==80 for r in phase.rows))
        for cap in (22,45):
            with self.assertRaisesRegex(api.Refusal,'budget'):ledger.reserve(ledger.known,Decimal(cap)-ledger.committed+Decimal('.000001'),cap=cap)
        self.assertEqual(recovery.availability(phase.rows)['errors'],2)
        self.assertIn('Permanent billing reservations',(self.root/'API-CACHE-REPORT.md').read_text())

    # A fourth timeout receipt is persisted with its full hold, then halts before the fifth scheduled attempt.
    def test_named_negative_fourth_permanent_reservation_halts(self):
        phase,status,elapsed=self.permanent_fixture(5)
        self.assertEqual(status,1);self.assertEqual(elapsed,4800);self.assertEqual(len(phase.rows),4)
        self.assertIn('fourth permanent reservation',phase.reason)
        self.assertEqual(phase.rows[-1]['charge_status'],'unresolved-reserved')
        self.assertTrue(json.loads((phase.out/'call-0003.json').read_text())['charge_resolution']['permanent_reservation'])
        ledger=recovery.Budget.load(self.root/'ledger.json');self.assertEqual(len(ledger.permanent_receipts()),4)
        self.assertEqual(ledger.committed,sum(Decimal(r['maximum_cost_usd']) for r in phase.rows))
        with self.assertRaisesRegex(api.Refusal,'fourth permanent reservation'):ledger.reserve(0,0)

    # A counter even1e-12 above the upper bound halts immediately; lower rounding tolerance cannot relax this bound.
    def test_named_negative_no_generation_overshoot_never_becomes_reservation(self):
        client=mock.Mock(secret=self.secret);client.get.return_value={'data':{'usage':'1.100000000001'}}
        budget=recovery.Budget(self.root/'ledger.json',1,1)
        row=dict(http_status=500,error='HTTP error',generation_id=None,usage=None,maximum_cost_usd='.1')
        with self.assertRaisesRegex(api.Refusal,'exceeds reservation'):
            recovery.reconcile_call(client,budget,row,'v1/0000',self.root/'overshoot.json')
        self.assertEqual(client.get.call_count,1);self.assertEqual(budget.receipts,{})
        self.assertEqual(budget.committed,Decimal(1))

    # A known generation, including one preserved only in the stream, cannot use the no-generation timeout exception.
    def test_named_negative_known_generation_never_stable_still_halts(self):
        budget=recovery.Budget.load(self.root/'ledger.json')
        for changes in (dict(generation_id='known'),dict(stream=[{'id':'journal-id'}])):
            row=dict(http_status=504,error='HTTP error',usage=None,maximum_cost_usd='.1',**changes)
            with mock.patch.object(recovery,'settled_samples',side_effect=recovery.SettlementTimeout('deadline')):
                with self.assertRaisesRegex(recovery.SettlementTimeout,'deadline'):
                    recovery.reconcile_call(self.client,budget,row,'v1/0000',self.root/'absent.json')
            self.assertEqual(budget.receipts,{})

    # Tiny lower precision error is zero; a genuinely lagging counter remains unresolved, not an invented zero charge.
    def test_settlement_lower_rounding_boundary_and_real_lag(self):
        self.assertEqual(recovery.resolve_delta(self.samples('.9999999'),'1','.1')['resolved_observed_charge_usd'],'0')
        with self.assertRaisesRegex(api.Refusal,'outside reservation'):
            recovery.resolve_delta(self.samples('.999999899999'),'1','.1')
        self.assertEqual(recovery.ROUNDING,Decimal('.00000001'))

    # Permanent holds survive stable aggregate handoff/end rechecks and idempotent reload without double billing.
    def test_permanent_reservations_rechecked_at_handoff_and_end_without_release(self):
        budget=recovery.Budget.load(self.root/'ledger.json')
        evidence=dict(expected_usd='0',maximum_usd='.2',samples=[{'usage_usd':'0'}])
        budget.permanent_call('.2','v1/0000',evidence)
        with mock.patch.object(recovery,'settled_samples',return_value=self.samples('.05')):
            restart.recheck_reservations(self.root,self.client,'handoff')
            restart.recheck_reservations(self.root,self.client,'end')
        ledger=recovery.Budget.load(self.root/'ledger.json');self.assertEqual(ledger.committed,Decimal('.2'));self.assertEqual(ledger.known,Decimal('.05'))
        receipt=ledger.receipts['v1/0000'];self.assertEqual([r['stage'] for r in receipt['rechecks']],['handoff','end'])
        self.assertEqual(len(ledger.permanent_receipts()),1);self.assertFalse(receipt['resolved'])
        self.assertEqual(receipt['status'],'unresolved-reserved')
        self.assertEqual(ledger.permanent_call('.2','v1/0000',{}),receipt)
        self.assertEqual(ledger.committed,Decimal('.2'))

    # Rechecks that remain unstable retain existing holds without creating another incident or claiming a debit.
    def test_permanent_recheck_still_unstable_retains_hold(self):
        budget=recovery.Budget.load(self.root/'ledger.json');evidence=dict(expected_usd='0',maximum_usd='.2',samples=[{'usage_usd':'0'}])
        budget.permanent_call('.2','v1/0000',evidence)
        def timeout(client,expected,maximum,path):
            recovery.atomic_json(path,evidence);raise recovery.SettlementTimeout('deadline')
        with mock.patch.object(recovery,'settled_samples',side_effect=timeout):restart.recheck_reservations(self.root,self.client,'end')
        ledger=recovery.Budget.load(self.root/'ledger.json');self.assertEqual(ledger.committed,Decimal('.2'));self.assertEqual(ledger.known,0)
        self.assertEqual(ledger.receipts['v1/0000']['rechecks'][-1]['status'],'still-unresolved-reserved')

    # A reservation is never allowed to mask missing/corrupt polling evidence, even after a timeout signal.
    def test_named_negative_missing_reservation_evidence_stops(self):
        budget=recovery.Budget.load(self.root/'ledger.json');row=dict(http_status=503,error='HTTP error',maximum_cost_usd='.1')
        with mock.patch.object(recovery,'settled_samples',side_effect=recovery.SettlementTimeout('deadline')):
            with self.assertRaises(FileNotFoundError):recovery.reconcile_call(self.client,budget,row,'v1/0000',self.root/'missing.json')
        self.assertEqual(budget.receipts,{})

    # An interrupted row cannot claim absent generation when its durable SSE journal already recorded one.
    def test_named_negative_watchdog_journal_generation_disallows_permanent_reservation(self):
        directory=self.root/'v1';directory.mkdir()
        row=dict(sequence=0,http_status=503,error='HTTP error',usage=None,maximum_cost_usd='.1')
        recovery.atomic_json(directory/'intent-0000.json',row);recovery.atomic_json(directory/'call-0000.json',row)
        (directory/'stream-0000.jsonl').write_text(json.dumps(dict(event={'id':'durable-generation'}))+'\n')
        with mock.patch.object(recovery,'settled_samples',side_effect=recovery.SettlementTimeout('deadline')):
            with self.assertRaisesRegex(recovery.SettlementTimeout,'deadline'):restart.recover_interrupted(self.root,'v1',self.client)
        self.assertEqual(recovery.Budget.load(self.root/'ledger.json').receipts,{})

    # A permanent recheck still stops immediately if cumulative spending exceeds the committed bound.
    def test_named_negative_permanent_recheck_overshoot_halts(self):
        budget=recovery.Budget.load(self.root/'ledger.json')
        budget.permanent_call('.2','v1/0000',dict(expected_usd='0',maximum_usd='.2',samples=[{'usage_usd':'0'}]))
        client=mock.Mock(secret=self.secret);client.get.return_value={'data':{'usage':'.200000001'}}
        with self.assertRaisesRegex(api.Refusal,'exceeds reservation'):restart.recheck_reservations(self.root,client,'end')
        self.assertEqual(client.get.call_count,1)
        self.assertEqual(recovery.Budget.load(self.root/'ledger.json').committed,Decimal('.2'))

    # Recovery cannot enlarge an attempted call's bound by trusting a changed response row over its intent.
    def test_named_negative_recovery_reservation_mismatch_stops_before_get(self):
        directory=self.root/'v1';directory.mkdir()
        row=dict(sequence=0,http_status=503,error='HTTP error',usage=None,maximum_cost_usd='.1')
        recovery.atomic_json(directory/'intent-0000.json',row)
        recovery.atomic_json(directory/'call-0000.json',dict(row,maximum_cost_usd='.2'))
        with mock.patch.object(self.client,'get') as get:
            with self.assertRaisesRegex(api.Refusal,'differs from durable intent'):restart.recover_interrupted(self.root,'v1',self.client)
        get.assert_not_called();self.assertEqual(recovery.Budget.load(self.root/'ledger.json').receipts,{})

    def supervisor_fixture(self,watchdog,late=False,crash=False,unresolved=False,wrong_model=False):
        recovery.atomic_json(self.root/'call34-resolution.json',dict(expected_without_call_usd='0',maximum_usd='.420'))
        children=[]
        def spawn(argv,**kwargs):
            phase=argv[argv.index('--phase')+1]
            directory=self.root/phase;directory.mkdir()
            recovery.atomic_json(directory/'terminal.json',dict(status='done',attempts=0))
            if wrong_model:
                recovery.atomic_json(directory/'call-0000.json',dict(session_id=self.session['session_id'],observed_models=['wrong/model'],
                    model=self.candidate['model'],endpoint_tag=self.candidate['endpoint']['tag'],started_at=api.utc(),depth=32768,
                    call_index=0,accepted=False,cohort='v1',sequence=0))
            child=mock.Mock(pid=10000+len(children))
            child.wait.side_effect=[restart.subprocess.TimeoutExpired('fake-child',1),0] if watchdog else [1 if crash else 0]
            children.append(child)
            kwargs['stdout'].write('done phase='+phase+' mock fixture\n')
            return child
        original=restart.recover_interrupted
        def recover(*args):
            if unresolved:raise api.Refusal('unresolvable charge: fixture recovery deadline')
            return original(*args)
        with mock.patch.object(restart,'phase_sessions',return_value=[self.session]),mock.patch.object(restart,'LOCK_PATH',self.root/'mock.lock'),mock.patch.object(recovery,'Client',return_value=self.client),mock.patch.object(recovery,'usage_samples',return_value=self.samples(0)),mock.patch.object(restart,'public_snapshot'),mock.patch.object(restart.subprocess,'Popen',side_effect=spawn),mock.patch.object(restart.subprocess,'run'),mock.patch.object(restart.subprocess,'check_output',side_effect=['a'*40,'']),mock.patch.object(restart,'recover_interrupted',side_effect=recover),mock.patch.object(restart,'recheck_reservations',wraps=restart.recheck_reservations) as rechecks:
            self.assertEqual(restart.supervise(self.root),1 if crash or unresolved or wrong_model else 0)
        self.assertEqual([call.args[2] for call in rechecks.call_args_list],['end'] if crash or unresolved or wrong_model else ['handoff','end'])
        self.assertEqual(len(children),1 if late or crash or unresolved or wrong_model else 2)
        workflow=json.loads((self.root/'workflow.json').read_text())
        self.assertEqual(workflow['status'],'failed' if crash or unresolved or wrong_model else 'done')
        self.assertEqual((self.root/('amendment.pid' if late else 'v1.pid')).read_text().strip(),'10000')
        self.assertTrue((self.root/'SUMMARY.md').exists());self.assertTrue((self.root/'heartbeat.json').exists())
        if watchdog and not (unresolved or wrong_model):
            children[0].terminate.assert_called_once()
            self.assertIn('both bounded',workflow['reason'])
            self.assertEqual(json.loads((self.root/'v1/terminal.json').read_text())['status'],'done')
        elif not (late or crash or unresolved or wrong_model):self.assertEqual((self.root/'amendment.pid').read_text().strip(),'10001')
        if crash or unresolved:
            self.assertFalse((self.root/'amendment.pid').exists())
            self.assertIn('phase failed' if crash else 'unresolvable charge',workflow['reason'])
        if wrong_model:
            self.assertFalse((self.root/'amendment.pid').exists());self.assertIn('proven model identity violation',workflow['reason'])
        if late:
            self.assertFalse((self.root/'v1.pid').exists());self.assertTrue(json.loads((self.root/'v1/skipped.json').read_text()))


if __name__=='__main__':unittest.main()

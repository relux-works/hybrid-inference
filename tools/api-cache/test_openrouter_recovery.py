"""CPU loopback controls for recovery and restart; no real provider qualification."""
from decimal import Decimal
from http.server import ThreadingHTTPServer
import json
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

    def do_POST(self):
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
        with mock.patch.object(recovery,'deadline_for_depth',return_value=.05),mock.patch.object(recovery,'usage_samples',side_effect=lambda client:self.samples(Handler.spend)):
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

    # A mismatched serving provider stops the series even when the charge itself is known.
    def test_provider_mismatch_refuses_followup(self):
        Handler.provider_override='Chutes';phase=self.phase()
        self.assertEqual(phase.run(),1);self.assertEqual(len(Handler.bodies),1)
        self.assertIn('identity violation',phase.reason)

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

    # A disk value above 28 decimal GB but below 28 GiB still refuses the paid request.
    def test_28_gib_floor_refuses_before_post(self):
        phase=self.phase()
        with mock.patch.object(api.shutil,'disk_usage',return_value=mock.Mock(free=28500000000)):
            with self.assertRaisesRegex(api.Refusal,'free disk below28 GiB'):phase.call(self.session,0)
        self.assertEqual(Handler.bodies,[])

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
        rows=[dict(provider=n,phase='amendment',model=c['model'],cohort='core',depth=1000000,call_index=1,accepted=True,prompt_tokens=999000,cached_tokens=0,new_tokens_estimate=1065,pause_target_seconds=8,pause_actual_seconds=8) for n,c in names.items()]
        rows.append(dict(rows[0],accepted=False,timed_out=True,error='deadline',call_index=0))
        with mock.patch.object(restart,'candidates',return_value=names),mock.patch.object(restart,'read_rows',return_value=rows):data=restart.analysis(self.root,final=True)
        comparison=data['million_token_comparison']['qwen/qwen3.8-27b']
        self.assertEqual(len(comparison['with_alibaba']['providers']),2)
        self.assertEqual(len(comparison['without_alibaba']['providers']),1)
        phala=next(p for p in data['providers'] if p['provider_name']=='Phala')
        self.assertEqual(phala['whole_window_availability']['rate'],.5)
        self.assertEqual(phala['eligibility'],'excluded-availability')
        self.assertEqual(len(rows),3)

    # The supervisor sequences both child PIDs under one lease and persists terminal custody and heartbeat.
    def test_supervisor_sequences_children_and_records_terminal_custody(self):
        self.supervisor_fixture(watchdog=False)

    # A watchdog stop produces failed custody and prevents launching the next paid phase.
    def test_supervisor_watchdog_failure_prevents_amendment_launch(self):
        self.supervisor_fixture(watchdog=True)

    def supervisor_fixture(self,watchdog):
        recovery.atomic_json(self.root/'call34-resolution.json',dict(expected_without_call_usd='0',maximum_usd='.420'))
        children=[]
        def spawn(argv,**kwargs):
            phase=argv[argv.index('--phase')+1]
            directory=self.root/phase;directory.mkdir()
            recovery.atomic_json(directory/'terminal.json',dict(status='done',attempts=0))
            child=mock.Mock(pid=10000+len(children))
            child.wait.side_effect=[restart.subprocess.TimeoutExpired('fake-child',1),0] if watchdog else [0]
            children.append(child)
            kwargs['stdout'].write('done phase='+phase+' mock fixture\n')
            return child
        with mock.patch.object(restart,'LOCK_PATH',self.root/'mock.lock'),mock.patch.object(recovery,'Client',return_value=self.client),mock.patch.object(recovery,'usage_samples',return_value=self.samples(0)),mock.patch.object(restart,'public_snapshot'),mock.patch.object(restart.subprocess,'Popen',side_effect=spawn),mock.patch.object(restart.subprocess,'run'),mock.patch.object(restart.subprocess,'check_output',side_effect=['a'*40,'']):
            self.assertEqual(restart.supervise(self.root),1 if watchdog else 0)
        self.assertEqual(len(children),1 if watchdog else 2)
        workflow=json.loads((self.root/'workflow.json').read_text())
        self.assertEqual(workflow['status'],'failed' if watchdog else 'done')
        self.assertEqual((self.root/'v1.pid').read_text().strip(),'10000')
        self.assertTrue((self.root/'SUMMARY.md').exists());self.assertTrue((self.root/'heartbeat.json').exists())
        if watchdog:
            children[0].terminate.assert_called_once()
            self.assertIn('watchdog',workflow['reason'])
            self.assertEqual(json.loads((self.root/'v1/terminal.json').read_text())['status'],'failed')
        else:self.assertEqual((self.root/'amendment.pid').read_text().strip(),'10001')


if __name__=='__main__':unittest.main()

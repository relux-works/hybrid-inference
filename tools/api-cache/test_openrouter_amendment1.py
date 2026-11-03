"""CPU loopback tests of Amendment 1, not provider/tokenizer qualification."""
import argparse
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import itertools
import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from decimal import Decimal

import openrouter_amendment1 as amendment
import openrouter_cache as api
import openrouter_restart as restart
import openrouter_recovery as recovery


class Handler(BaseHTTPRequestHandler):
    spend = 0
    cost = .000001
    provider_override = None
    seen = {}
    bodies = []
    cached_known = True
    usage_cost_absent = False

    def log_message(self,*args):
        pass

    def do_GET(self):
        if '/models/' in self.path:
            model=self.path.split('/models/')[1].removesuffix('/endpoints')
            data=next(m['data'] for m in json.loads(api.SNAPSHOT.read_text())['models'] if m['data']['id']==model)
        elif '/generation?' in self.path:
            data=dict(id='known-generation',model='qwen/qwen3.8-27b',finish_reason='stop',total_cost=.2)
        else:
            data=dict(usage=Handler.spend,limit_remaining=50-Handler.spend)
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({'data':data}).encode())

    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        Handler.bodies.append(body)
        manifest=json.loads(amendment.MANIFEST.read_text())
        endpoint=next(c['endpoint'] for c in manifest['candidates'] if c['model']==body['model'] and c['endpoint']['tag']==body['provider']['only'][0])
        prompt=len(body['messages'][0]['content'][0]['text'].split())
        count=Handler.seen.get(body['session_id'],0)
        Handler.seen[body['session_id']]=count+1
        usage=dict(prompt_tokens=prompt,completion_tokens=2,cost=Handler.cost)
        if Handler.cached_known:
            usage['prompt_tokens_details']=dict(cached_tokens=max(0,prompt-1065) if count else 0,cache_write_tokens=0)
        if Handler.usage_cost_absent:usage.pop('cost')
        Handler.spend+=Handler.cost
        self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
        for item in [dict(id='mock-generation',model=body['model'],provider=Handler.provider_override or endpoint['provider_name'],choices=[dict(delta=dict(content='OK'))]),dict(usage=usage,choices=[])]:
            self.wfile.write(b'data: '+json.dumps(item).encode()+b'\n\n')
        self.wfile.write(b'data: [DONE]\n\n')


class Clock:
    def __init__(self):self.now=0
    def monotonic(self):return self.now
    def sleep(self,seconds):self.now+=seconds


class AmendmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True); cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def setUp(self):
        # These historical engine controls isolate the retired renderer, never the current report or its refusal tests.
        renderer=mock.patch.object(amendment.Study,'report',return_value=None);renderer.start();self.addCleanup(renderer.stop)
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.key=self.root/'fake-key'; self.secret='sk-or-v1-amendment-fake-control'
        self.key.write_text(self.secret); self.key.chmod(0o600)
        self.client=api.Client(self.key,'http://127.0.0.1:'+str(self.server.server_port),mock=True)
        self.manifest=json.loads(amendment.MANIFEST.read_text())
        self.manifest['engine_sha256']=hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest()
        self.manifest_path=self.root/'cpu-manifest.json';self.manifest_path.write_text(json.dumps(self.manifest))
        self.candidate=next(c for c in self.manifest['candidates'] if c['endpoint']['provider_name']=='Phala' and c['model']=='qwen/qwen3.8-27b')
        Handler.spend=0; Handler.cost=.000001; Handler.provider_override=None
        Handler.seen={}; Handler.bodies=[]; Handler.cached_known=True; Handler.usage_cost_absent=False

    def small_manifest(self):
        data=json.loads(json.dumps(self.manifest)); data['depths']=[16384,32768]; data['candidates']=[self.candidate]
        return data

    def study(self):
        out=self.root/'study'; out.mkdir()
        study=amendment.Study(self.small_manifest(),out); study.client=self.client
        return study

    # The registered matrix retains all selected Qwen and >=1M GLM endpoints; low uptime does not remove one.
    def test_frozen_matrix_and_pause_distribution(self):
        self.assertEqual(len(self.manifest['candidates']),42)
        self.assertEqual({c['endpoint']['provider_name'] for c in self.manifest['candidates'] if c['model']=='qwen/qwen3.8-27b'},{'Alibaba','Novita','Phala'})
        self.assertTrue(all(c['endpoint']['context_length']>=1000000 for c in self.manifest['candidates']))
        gaps=sorted(self.manifest['core_pauses'][1:])
        self.assertEqual(gaps,[8]*6+[35]*3)
        self.assertEqual((gaps[4],gaps[8]),(8,35))
        self.assertEqual(self.manifest['long_pauses'],[0,60,300,900,3600])

    # Admission evidence governs the first ranks; absent evidence stays explicitly provisional.
    def test_priority_admitted_and_provisional(self):
        ordered,ranking=amendment.priority_order(self.manifest)
        self.assertEqual([c['endpoint']['provider_name'] for c in ordered[:2]],['Phala','Novita'])
        self.assertTrue(all('provisional' in r['label'] for r in ranking[:2]))
        qwen=[c for c in ordered if c['model']=='qwen/qwen3.8-27b']
        prices={'Alibaba':1,'Phala':2,'Novita':3}
        previous={'providers':[dict(provider=amendment.identity(c),admission='eligible',depths=[1000000],effective_usd_per_1000_by_depth=[prices[c['endpoint']['provider_name']]]) for c in qwen]}
        ordered,ranking=amendment.priority_order(self.manifest,previous)
        self.assertEqual([c['endpoint']['provider_name'] for c in ordered[:2]],['Alibaba','Phala'])
        self.assertEqual([r['label'] for r in ranking[:2]],['cheapest admitted Qwen','median admitted Qwen'])

    # Independent probe/core/TTL prefixes preserve cold-probe identity and exact growing text prefixes.
    def test_sessions_growth_and_probe_depth(self):
        items=amendment.sessions(self.manifest,self.candidate)
        self.assertEqual(len({s['session_id'] for s in items}),6)
        core=next(s for s in items if s['cohort']=='core' and s['depth']==524288)
        a=amendment.payload(self.manifest,core,0); b=amendment.payload(self.manifest,core,1)
        ta=a['messages'][0]['content'][0]['text']; tb=b['messages'][0]['content'][0]['text']
        self.assertTrue(tb.startswith(ta)); self.assertEqual(len(tb.split())-len(ta.split()),1065)
        probe=next(s for s in items if s['cohort']=='probe' and s['depth']==1000000)
        text=amendment.payload(self.manifest,probe,0)['messages'][0]['content'][0]['text']
        self.assertLess(abs(len(text.split())-1000000),256)
        self.assertNotEqual(core['session_id'],probe['session_id'])

    # Only an accepted, explicitly uncached response with observed tokens supplies a miss TTFT.
    def test_miss_hit_unknown_and_failure_classification(self):
        row=dict(accepted=True,cached_tokens=0,ttft_seconds=180,prompt_tokens=999900,elapsed_seconds=181)
        self.assertEqual(amendment.miss_latency(row)['ttft_minutes'],3)
        for changes,status in [({'cached_tokens':1},'cache-hit'),({'cached_tokens':None},'cache-count-unknown'),({'accepted':False},'failed-or-unverified'),({'ttft_seconds':None},'no-observed-token')]:
            updated={**row,**changes}; result=amendment.miss_latency(updated)
            self.assertEqual(result['status'],status); self.assertIsNone(result['ttft_seconds'])

    # A live legacy owner or held budget lock prevents acquisition without instantiating a key-reading client.
    def test_custody_and_no_key_read_while_queued(self):
        lock=self.root/'cache-run.lock'
        with mock.patch.object(amendment,'active_prior',return_value=[123]):
            self.assertIsNone(amendment.try_custody(self.root,lock))
        first=lock.open('a'); fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.addCleanup(first.close)
        with mock.patch.object(amendment,'active_prior',return_value=[]):
            self.assertIsNone(amendment.try_custody(self.root,lock))
        args=argparse.Namespace(output=self.root/'queued',manifest=self.manifest_path,prior=self.root/'prior',key_path=self.key,wait_seconds=0)
        with mock.patch.object(amendment,'try_custody',return_value=None),mock.patch.object(api,'Client') as client:
            self.assertEqual(amendment.run(args),1); client.assert_not_called()
        terminal=json.loads((args.output/'terminal.json').read_text())
        self.assertEqual(terminal['attempts'],0)
        self.assertIn('custody',terminal['reason'])

    # Prior costs survive delayed spend counters; unmatched intents refuse, and finished-generation metadata can reconcile.
    def test_prior_cost_and_uncertain_billing(self):
        prior=self.root/'prior'; d=prior/'full-v2'; d.mkdir(parents=True)
        out=self.root/'reconcile'; out.mkdir()
        (d/'intent-0000.json').write_text('{}')
        with self.assertRaisesRegex(api.Refusal,'intent unresolved'):
            amendment.prior_costs(prior,self.client,out)
        (d/'call-0000.json').write_text(json.dumps(dict(cost_usd=.2)))
        self.assertEqual(amendment.prior_costs(prior,self.client,out),Decimal('.2'))
        (d/'call-0000.json').write_text(json.dumps(dict(model='qwen/qwen3.8-27b',http_status=200)))
        with self.assertRaisesRegex(api.Refusal,'charge unresolved'):
            amendment.prior_costs(prior,self.client,out)
        (d/'call-0000.json').write_text(json.dumps(dict(model='qwen/qwen3.8-27b',generation_id='known-generation')))
        self.assertEqual(amendment.prior_costs(prior,self.client,out),Decimal('.2'))

    # Known earlier charges make the next paid call refuse even when the live key endpoint still says zero.
    def test_combined_budget_refusal_no_post(self):
        study=self.study(); study.base=Decimal(45)
        session=next(s for s in amendment.sessions(study.manifest,self.candidate) if s['cohort']=='probe')
        with self.assertRaises(amendment.BudgetSkip):study.call(session,0)
        self.assertFalse(Handler.bodies); self.assertEqual(study.rows,[])

    # A loopback wrong-provider response is retained but stops spending and never qualifies as accepted.
    def test_pinning_refusal_and_positive_control(self):
        study=self.study(); session=next(s for s in amendment.sessions(study.manifest,self.candidate) if s['cohort']=='probe')
        study.call(session,0)
        self.assertTrue(study.rows[0]['accepted'])
        self.assertEqual(Handler.bodies[0]['provider']['only'],['phala'])
        Handler.provider_override='Wrong Provider'
        with self.assertRaisesRegex(api.Refusal,'unverifiable'):study.call(session,0)
        self.assertFalse(study.rows[1]['accepted'])
        self.assertEqual(len(Handler.bodies),2)

    # Injected bookkeeping stays in the actual response-to-request gap; the target is independently recorded as 8s.
    def test_pause_includes_bookkeeping(self):
        study=self.study(); session=next(s for s in amendment.sessions(study.manifest,self.candidate) if s['cohort']=='core')
        clock=Clock(); original=self.client.get
        def get(path):
            result=original(path); clock.now+=100; return result
        with mock.patch.object(amendment.time,'monotonic',side_effect=clock.monotonic),mock.patch.object(amendment.time,'sleep',side_effect=clock.sleep),mock.patch.object(self.client,'get',side_effect=get):
            ended=study.call(session,0)
            study.call(session,1,ended)
        self.assertEqual(study.rows[1]['pause_target_seconds'],8)
        self.assertEqual(study.rows[1]['pause_actual_seconds'],400)

    # The miniature full matrix exercises real loopback SSE, core/TTL separation, reports and terminal spend.
    def test_end_to_end_mock_amendment(self):
        manifest=self.small_manifest(); path=self.root/'manifest.json'; path.write_text(json.dumps(manifest))
        args=argparse.Namespace(output=self.root/'full',manifest=path,prior=self.root/'prior',key_path=self.key,wait_seconds=0)
        clock=Clock()
        with mock.patch.object(api,'Client',return_value=self.client),mock.patch.object(amendment.time,'monotonic',side_effect=clock.monotonic),mock.patch.object(amendment.time,'sleep',side_effect=clock.sleep):
            self.assertEqual(amendment.run(args),0)
        terminal=json.loads((args.output/'terminal.json').read_text())
        rows=[json.loads(p.read_text()) for p in args.output.glob('call-*.json')]
        self.assertEqual(terminal['attempts'],32);self.assertEqual(terminal['status'],'done')
        core=[r for r in rows if r['cohort']=='core'];long=[r for r in rows if r['cohort']=='long']
        self.assertEqual(len(core),20);self.assertEqual(len(long),10)
        self.assertEqual(len([r for r in rows if r['cohort']=='probe']),2)
        cm=restart.price_metrics(core,self.candidate,manifest['depths'])
        self.assertEqual(cm['pause_levels']['8']['n'],12);self.assertEqual(cm['pause_levels']['35']['n'],6)
        self.assertEqual(recovery.eligible(self.candidate,rows),'eligible')
        self.assertEqual(len(Handler.bodies),32)
        for file in args.output.iterdir():
            if file.is_file():self.assertNotIn(self.secret,file.read_text())

    # TTL hits cannot inflate the short-pause price or cache fraction; absent short-pause evidence remains unknown.
    def test_cohort_separation_and_missing_cache_fields(self):
        study=self.study(); study.status='running'
        for cohort,cached in [('core',0),('long',5000)]:
            session=next(s for s in amendment.sessions(study.manifest,self.candidate) if s['cohort']==cohort)
            study.rows.append(dict(provider=amendment.identity(self.candidate),cohort=cohort,session_id=session['session_id'],depth=session['depth'],call_index=1,
                                   accepted=True,prompt_tokens=10000,cached_tokens=cached,new_tokens_estimate=1065,pause_target_seconds=8 if cohort=='core' else 60,pause_actual_seconds=8 if cohort=='core' else 60))
        core=[r for r in study.rows if r['cohort']=='core'];long=[r for r in study.rows if r['cohort']=='long']
        self.assertEqual(restart.price_metrics(core,self.candidate,study.manifest['depths'])['token_hit_rate'],0)
        self.assertEqual(restart.price_metrics(long,self.candidate,study.manifest['depths'])['token_hit_rate'],.5)
        study.rows[0]['cached_tokens']=None
        cm=restart.price_metrics(core,self.candidate,study.manifest['depths'])
        self.assertIsNone(cm['token_hit_rate']);self.assertIsNone(cm['effective_usd_per_1000_equal_depth_mix'])

    # This planted raw credential must fail the detector; sanitized JSON and echoed streams contain no key bytes.
    def test_negative_planted_key_in_amendment_log_is_caught(self):
        planted='Authorization: Bearer '+self.secret
        with self.assertRaisesRegex(api.Refusal,'secret detected'):api.assert_no_secret(planted,self.secret)
        api.assert_no_secret(api.redact(planted,self.secret),self.secret)
        study=self.study(); Handler.provider_override=self.secret
        session=next(s for s in amendment.sessions(study.manifest,self.candidate) if s['cohort']=='probe')
        with self.assertRaises(api.Refusal):study.call(session,0)
        for file in study.out.iterdir():
            if file.is_file():self.assertNotIn(self.secret,file.read_text())

    # A source-hash mismatch leaves all cells pending and makes no key read or network request.
    def test_engine_identity_refusal(self):
        manifest=self.small_manifest(); manifest['engine_sha256']='bad'
        path=self.root/'bad-manifest.json'; path.write_text(json.dumps(manifest))
        args=argparse.Namespace(output=self.root/'bad',manifest=path,prior=self.root/'prior',key_path=self.key,wait_seconds=0)
        with mock.patch.object(api,'Client') as client:
            self.assertEqual(amendment.run(args),1); client.assert_not_called()
        self.assertIn('engine differs',json.loads((args.output/'terminal.json').read_text())['reason'])


    # Budget skips retain every missing cohort and all distinct preflight snapshots, with no POST or false admission.
    def test_budget_skips_are_reported_with_snapshot_provenance(self):
        study=self.study(); study.base=Decimal(45)
        study.run_matrix([self.candidate])
        self.assertFalse(Handler.bodies)
        self.assertEqual(len(study.skips),6)
        self.assertTrue(any('budget' in reason for reason in study.skips.values()))
        self.assertEqual(len(list(study.out.glob('uptime-*.json'))),4)
        self.assertEqual(recovery.eligible(self.candidate,study.rows),'unqualified-no-attempts')

    # Missing reported cost or a bill above reservation stops after the first actual mock request.
    def test_missing_and_unexpected_cost_stop(self):
        study=self.study(); session=next(s for s in amendment.sessions(study.manifest,self.candidate) if s['cohort']=='probe')
        Handler.usage_cost_absent=True
        with self.assertRaisesRegex(api.Refusal,'unverifiable'):study.call(session,0)
        self.assertEqual(len(Handler.bodies),1); self.assertFalse(study.rows[0]['accepted'])
        Handler.usage_cost_absent=False; Handler.cost=2
        with self.assertRaisesRegex(api.Refusal,'unexpected cost'):study.call(session,0)
        self.assertEqual(len(Handler.bodies),2); self.assertFalse(study.rows[1]['accepted'])
        self.assertEqual(study.rows[1]['error'],'unexpected cost exceeds reservation')

    # Crossing a pause deadline between two clock reads must never pass a negative duration to sleep.
    def test_pause_deadline_crossing_no_negative_sleep(self):
        study=self.study(); session=next(s for s in amendment.sessions(study.manifest,self.candidate) if s['cohort']=='core')
        values=iter(itertools.chain([7,9,9],itertools.repeat(10)))
        def sleep(seconds):self.assertGreaterEqual(seconds,0)
        with mock.patch.object(amendment.time,'monotonic',side_effect=lambda:next(values)),mock.patch.object(amendment.time,'sleep',side_effect=sleep) as sleeper:
            study.call(session,1,0)
        sleeper.assert_called_once_with(0)


if __name__=='__main__':
    unittest.main()

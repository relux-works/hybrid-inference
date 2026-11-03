"""CPU tests of the measurement instrument; no provider qualification is implied."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from decimal import Decimal

import openrouter_cache as cache


class Handler(BaseHTTPRequestHandler):
    provider = 'Chutes'
    usage = dict(prompt_tokens=32768, completion_tokens=3, cost=.001,
                 prompt_tokens_details=dict(cached_tokens=30000, cache_write_tokens=2000))
    payload = None
    calls = 0
    live_spend = 44.99
    dynamic = False
    cost = .001
    reject_disabled_reasoning = False

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        if '/models/' in self.path:
            model = self.path.split('/models/')[1].removesuffix('/endpoints')
            snap = json.loads(cache.SNAPSHOT.read_text())
            data = next(m['data'] for m in snap['models'] if m['data']['id']==model)
            self.wfile.write(json.dumps({'data':data}).encode())
        else:
            self.wfile.write(json.dumps({'data':{'usage':Handler.live_spend,'limit_remaining':50-Handler.live_spend}}).encode())

    def do_POST(self):
        Handler.calls += 1
        if Handler.dynamic:
            Handler.usage = dict(prompt_tokens=32768+(Handler.calls-1)*1000,completion_tokens=3,cost=Handler.cost,
                                 prompt_tokens_details=dict(cached_tokens=0 if Handler.calls==1 else 32000+(Handler.calls-2)*1000,cache_write_tokens=0))
            Handler.live_spend += Handler.cost
        Handler.payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if Handler.reject_disabled_reasoning and Handler.payload.get('reasoning',{}).get('enabled') is False:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled."}}')
            return
        self.send_response(200)
        self.send_header('Content-Type','text/event-stream')
        self.end_headers()
        for obj in [dict(id='mock-1',provider=Handler.provider,choices=[dict(delta=dict(content='OK'))]),
                    dict(id='mock-1',provider=Handler.provider,choices=[],usage=Handler.usage)]:
            self.wfile.write(b'data: '+json.dumps(obj).encode()+b'\n\n')
        self.wfile.write(b'data: [DONE]\n\n')


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.key=self.path/'key'
        self.secret='sk-or-v1-fake-secret-control'
        self.key.write_text(self.secret)
        self.key.chmod(0o600)
        self.client=cache.Client(self.key,'http://127.0.0.1:'+str(self.server.server_port),mock=True)
        self.sessions,_=cache.matrix(json.loads(cache.SNAPSHOT.read_text()))
        self.session=next(s for s in self.sessions if s['model']=='qwen/qwen3.8-27b' and s['endpoint']['tag']=='chutes/fp8' and s['depth']==32768)
        Handler.calls=0
        Handler.dynamic=False
        Handler.cost=.001
        Handler.reject_disabled_reasoning=False
        Handler.live_spend=44.99
        Handler.provider='Chutes'
        Handler.usage=dict(prompt_tokens=32768,completion_tokens=3,cost=.001,
                           prompt_tokens_details=dict(cached_tokens=30000,cache_write_tokens=2000))

    # The local SSE server verifies exact routing, identity rejection and normal usage in the same path.
    def test_provider_pinning_refusal_and_positive_control(self):
        payload=cache.build_payload(self.session,0,True)
        result=cache.stream_call(self.client,payload,self.session['endpoint'])
        self.assertNotIn('error',result)
        self.assertEqual(Handler.payload['provider'],dict(order=['chutes/fp8'],only=['chutes/fp8'],allow_fallbacks=False))
        self.assertNotIn('reasoning',Handler.payload)
        self.assertEqual(result['cached_tokens'],30000)
        self.assertEqual(result['cache_write_tokens'],2000)
        self.assertEqual(result['http_status'],200)
        Handler.provider='Other Provider'
        result=cache.stream_call(self.client,payload,self.session['endpoint'])
        self.assertIn('serving provider differs',result['error'])
        self.assertNotIn('cost_usd',result)
        Handler.provider=None
        self.assertIn('serving provider differs',cache.stream_call(self.client,payload,self.session['endpoint'])['error'])

    # A refused reservation never reaches chat, including lagging live spend and a small remaining balance.
    def test_budget_refusal_before_post_and_positive_control(self):
        data=self.client.get('/key')['data']
        before=Handler.calls
        with self.assertRaisesRegex(cache.Refusal,'budget'):
            cache.guard(data['usage'],0,Decimal('.02'),45,data['limit_remaining'])
        self.assertEqual(Handler.calls,before)
        self.assertEqual(cache.guard(0,0,Decimal('.02'),45,1),0)
        with self.assertRaisesRegex(cache.Refusal,'budget'):
            cache.guard(0,Decimal('44.99'),Decimal('.02'),45)
        with self.assertRaisesRegex(cache.Refusal,'remaining'):
            cache.guard(0,0,Decimal('.02'),45,.01)
        for invalid in ('NaN','Infinity',-1,None):
            with self.assertRaises(cache.Refusal):
                cache.guard(invalid,0,Decimal('.02'),45)

    # Missing cache counters remain unknown; impossible usage and missing cost cannot qualify a response.
    def test_usage_parsing_negative_and_positive(self):
        result=cache.parse_usage(Handler.usage)
        self.assertEqual((result['prompt_tokens'],result['completion_tokens'],result['cached_tokens']),(32768,3,30000))
        absent=cache.parse_usage(dict(prompt_tokens=10,completion_tokens=1,cost=0))
        self.assertIsNone(absent['cached_tokens'])
        for invalid in [dict(prompt_tokens=10,completion_tokens=1),
                        dict(prompt_tokens=10,completion_tokens=1,cost=0,prompt_tokens_details={'cached_tokens':11}),
                        dict(prompt_tokens=-1,completion_tokens=1,cost=0),
                        dict(prompt_tokens=True,completion_tokens=1,cost=0)]:
            with self.assertRaises(cache.Refusal):
                cache.parse_usage(invalid)
        Handler.usage={'prompt_tokens':32768,'completion_tokens':3}
        self.assertIn('error',cache.stream_call(self.client,cache.build_payload(self.session,0),self.session['endpoint']))

    # This planted leaking log must fail the detector; sanitized text must pass with no credential bytes.
    def test_negative_planted_key_in_log_is_caught(self):
        planted='debug Authorization: Bearer '+self.secret+'\n'
        with self.assertRaisesRegex(cache.Refusal,'secret detected'):
            cache.assert_no_secret(planted,self.secret)
        sanitized=cache.redact(planted,self.secret)
        cache.assert_no_secret(sanitized,self.secret)
        self.assertNotIn(self.secret,sanitized)
        cache.write_json(self.path/'log.json',{'error':planted},self.secret)
        self.assertNotIn(self.secret,(self.path/'log.json').read_text())

    # A mock upstream echo containing the key is scrubbed before call artifacts are written.
    def test_response_secret_echo_redaction(self):
        Handler.provider=self.secret
        row=cache.stream_call(self.client,cache.build_payload(self.session,0),self.session['endpoint'])
        cache.write_json(self.path/'call.json',row,self.secret)
        self.assertNotIn(self.secret,(self.path/'call.json').read_text())

    # External destinations and redirect following are rejected before credentials can leave the allowed host.
    def test_network_and_key_permissions_refusal(self):
        with self.assertRaisesRegex(cache.Refusal,'destination'):
            cache.Client(self.key,'https://example.invalid')
        with self.assertRaisesRegex(cache.Refusal,'redirect'):
            cache.NoRedirect().redirect_request(None,None,302,'',{},'https://example.invalid')
        self.key.chmod(0o644)
        with self.assertRaisesRegex(cache.Refusal,'mode 600'):
            cache.Client(self.key)

    # Frozen endpoint exclusion is based on context/caching, never prices or observed hit rate.
    def test_matrix_and_deterministic_growing_prefix(self):
        snapshot=json.loads(cache.SNAPSHOT.read_text())
        sessions,excluded=cache.matrix(snapshot)
        self.assertEqual(sessions,cache.matrix(snapshot)[0])
        self.assertTrue(any('PrimeIntellect' in e['endpoint'] and 'context' in ' '.join(e['reasons']) for e in excluded))
        self.assertTrue(any(s['model']=='z-ai/glm-5.3-flash' and s['depth']==1000000 for s in sessions))
        self.assertTrue(any(s['endpoint']['provider_name']=='Alibaba' for s in sessions))
        a=cache.build_payload(self.session,0)['messages'][0]['content'][0]['text']
        b=cache.build_payload(self.session,1)['messages'][0]['content'][0]['text']
        self.assertTrue(b.startswith(a))
        self.assertEqual(len(b.split())-len(a.split()),1000)
        self.assertEqual(a,cache.build_payload(self.session,0)['messages'][0]['content'][0]['text'])
        self.assertLess(len(a.encode()),2*32768)
        self.assertTrue(self.session['session_id'].startswith('r15b-v2-'))

    # Any live price rise blocks requests; a lower price and cache-write surcharge are handled conservatively.
    def test_live_price_and_max_cost(self):
        endpoint=self.session['endpoint']
        self.assertGreaterEqual(cache.max_cost(endpoint,32768),Decimal(32768)*Decimal(endpoint['pricing']['prompt']))
        live=json.loads(json.dumps(endpoint))
        cache.validate_live_price(endpoint,live)
        live['pricing']['prompt']='1'
        with self.assertRaisesRegex(cache.Refusal,'price increased'):
            cache.validate_live_price(endpoint,live)

    # No observations produces no admission or fabricated zero cost; partial smoke evidence remains partial.
    def test_report_unknown_not_admitted(self):
        snapshot=json.loads(cache.SNAPSHOT.read_text())
        cache.report(self.path,snapshot,'running')
        report=json.loads((self.path/'summary.json').read_text())
        for provider in report['providers']:
            self.assertEqual(provider['admission'],'pending-or-failed')
            self.assertIsNone(provider['effective_usd_per_1000_equal_depth_mix'])
            self.assertIsNone(provider['availability_observed'])


    # The production runner completes three mock calls, preserves spend, and emits reports; a cost breach stops after one.
    def test_runner_smoke_and_unexpected_cost_stop(self):
        args=argparse.Namespace(output=self.path/'smoke',snapshot=cache.SNAPSHOT,smoke=True,key_path=self.key)
        Handler.dynamic=True
        Handler.live_spend=0
        with mock.patch.object(cache,'Client',return_value=self.client):
            self.assertEqual(cache.run(args),0)
        terminal=json.loads((args.output/'terminal.json').read_text())
        self.assertEqual((terminal['status'],terminal['attempts']),('done',3))
        self.assertAlmostEqual(terminal['recorded_billed_usd'],.003)
        rows=[json.loads(p.read_text()) for p in args.output.glob('call-*.json')]
        self.assertTrue(all(r['accepted'] for r in rows))
        self.assertTrue((args.output/'calls.csv').exists())
        for p in args.output.iterdir():
            if p.is_file():
                self.assertNotIn(self.secret,p.read_text())
        Handler.calls=0
        Handler.live_spend=0
        Handler.cost=1
        args.output=self.path/'expensive'
        with mock.patch.object(cache,'Client',return_value=self.client):
            self.assertEqual(cache.run(args),1)
        self.assertEqual(Handler.calls,1)
        terminal=json.loads((args.output/'terminal.json').read_text())
        self.assertEqual(terminal['attempts'],1)
        self.assertIn('unexpected billed cost',terminal['reason'])

    # A real mock-key query at the cumulative cap blocks the complete runner before any paid POST.
    def test_runner_budget_refusal_no_partial_post(self):
        args=argparse.Namespace(output=self.path/'budget',snapshot=cache.SNAPSHOT,smoke=True,key_path=self.key)
        Handler.live_spend=45
        with mock.patch.object(cache,'Client',return_value=self.client):
            self.assertEqual(cache.run(args),1)
        self.assertEqual(Handler.calls,0)
        terminal=json.loads((args.output/'terminal.json').read_text())
        self.assertEqual(terminal['attempts'],0)
        self.assertIn('budget',terminal['reason'])


    # Injected bookkeeping time must count toward the observed gap since the previous response.
    def test_pause_includes_post_response_bookkeeping_regression(self):
        args=argparse.Namespace(output=self.path/'pauses',snapshot=cache.SNAPSHOT,smoke=True,key_path=self.key)
        Handler.dynamic=True
        Handler.live_spend=0
        clock=[0]
        original_get=self.client.get
        def get_with_bookkeeping(path):
            result=original_get(path)
            if Handler.calls:
                clock[0]+=100
            return result
        with mock.patch.object(cache,'Client',return_value=self.client), \
             mock.patch.object(self.client,'get',side_effect=get_with_bookkeeping), \
             mock.patch.object(cache.time,'monotonic',side_effect=lambda:clock[0]):
            self.assertEqual(cache.run(args),0)
        row=json.loads((args.output/'call-0001.json').read_text())
        self.assertEqual(row['pause_actual_seconds'],400)


    # A restart cannot forget known earlier charges just because the live usage endpoint lags.
    def test_prior_spend_floor_blocks_restart(self):
        args=argparse.Namespace(output=self.path/'floor',snapshot=cache.SNAPSHOT,smoke=True,key_path=self.key,prior_spend_floor='45')
        Handler.live_spend=0
        with mock.patch.object(cache,'Client',return_value=self.client):
            self.assertEqual(cache.run(args),1)
        self.assertEqual(Handler.calls,0)
        self.assertIn('budget',json.loads((args.output/'terminal.json').read_text())['reason'])


    # A mandatory-reasoning mock rejects the old flag and accepts the default without raising the output cap.
    def test_mandatory_reasoning_default_regression(self):
        session=next(s for s in self.sessions if s['model']=='z-ai/glm-5.3-flash' and s['endpoint']['tag']=='morph/fp8' and s['depth']==250000)
        Handler.provider='Morph'
        Handler.reject_disabled_reasoning=True
        payload=cache.build_payload(session,0)
        self.assertNotIn('reasoning',payload)
        self.assertEqual(payload['max_tokens'],16)
        self.assertLess(len(payload['messages'][0]['content'][0]['text'].encode())/4+16,262144)
        valid=cache.stream_call(self.client,payload,session['endpoint'])
        self.assertNotIn('error',valid)
        payload['reasoning']={'enabled':False}
        rejected=cache.stream_call(self.client,payload,session['endpoint'])
        self.assertEqual(rejected['http_status'],400)
        self.assertIn('Reasoning is mandatory',rejected['error_body'])


if __name__=='__main__':
    unittest.main()

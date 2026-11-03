"""GET-only loopback controls; fixtures never qualify live API cost or pricing."""
import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
import os
import io
from contextlib import redirect_stdout
from urllib.parse import parse_qs,urlparse

import openrouter_audit as audit
import openrouter_cache as api
import openrouter_amendment1 as amendment
import openrouter_restart as restart
from test_openrouter_amendment1 import Handler


class AuditHandler(Handler):
    generation_cost=.1
    generation_model='qwen/qwen3.8-27b'
    drift=False
    limited=False
    gets=0

    def do_GET(self):
        AuditHandler.gets+=1
        if self.path.startswith('/generation?'):
            if AuditHandler.limited:
                self.send_response(429);self.end_headers();return
            generation=parse_qs(urlparse(self.path).query)['id'][0]
            data=dict(id=generation,model=AuditHandler.generation_model,total_cost=AuditHandler.generation_cost)
        elif '/models/' in self.path:
            model=self.path.split('/models/')[1].removesuffix('/endpoints')
            data=next(m['data'] for m in json.loads(api.SNAPSHOT.read_text())['models'] if m['data']['id']==model)
            if AuditHandler.drift:data['endpoints'][0]['pricing']['prompt']='0.99'
        else:data=dict(usage=2.1)
        self.send_response(200);self.end_headers();self.wfile.write(json.dumps({'data':data}).encode())


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),AuditHandler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.live=self.root/'live';self.live.mkdir()
        self.output=self.root/'audit'
        self.key=self.root/'fake-key';self.secret='sk-or-v1-audit-fake-control';self.key.write_text(self.secret);self.key.chmod(0o600)
        self.client=audit.GetOnlyClient(self.key,'http://127.0.0.1:'+str(self.server.server_port),mock=True)
        self.candidate=next(c for c in restart.candidates().values() if c['endpoint']['provider_name']=='Phala' and c['model']=='qwen/qwen3.8-27b')
        self.row=dict(phase='v1',sequence=0,provider=amendment.identity(self.candidate),model=self.candidate['model'],endpoint_tag='phala',cohort='v1',depth=32768,session_id='fake',call_index=1,accepted=True,started_at=api.utc(),ended_at=api.utc(),generation_id='gen-1',usage={'cost':.1},cost_usd=.1,prompt_tokens=32000,cached_tokens=31000,new_tokens_estimate=1000,pause_target_seconds=0,pause_actual_seconds=.1,pricing=self.candidate['endpoint']['pricing'])
        AuditHandler.generation_cost=.1;AuditHandler.generation_model=self.row['model'];AuditHandler.drift=False;AuditHandler.limited=False;AuditHandler.gets=0;Handler.bodies=[]

    def run_audit(self,rows=None):
        with mock.patch.object(restart,'read_rows',return_value=rows or [self.row]):
            return audit.audit(self.live,self.output,client_factory=lambda:self.client)

    # Both public retired renderers reject before any output; the current isolated report is the positive control below.
    def test_negative_retired_cache_report_names_live_path(self):
        with self.assertRaisesRegex(api.Refusal,'Legacy cache report retired; live report:') as refusal:
            api.report(self.output,{},'running')
        self.assertIn(str(api.LIVE_REPORT_PATH),str(refusal.exception));self.assertFalse(self.output.exists())

    def test_negative_retired_amendment_report_names_live_path(self):
        with self.assertRaisesRegex(api.Refusal,'Legacy amendment report retired; live report:') as refusal:
            amendment.Study({},self.output).report()
        self.assertIn(str(api.LIVE_REPORT_PATH),str(refusal.exception));self.assertFalse(self.output.exists())

    # The standalone routes refuse before constructing a key-reading client or sending even one request.
    def test_negative_retired_runners_make_no_key_read_or_request(self):
        args=argparse.Namespace(output=self.output)
        with mock.patch.object(api,'Client') as client:
            for run in (api.run,amendment.run):
                with self.assertRaisesRegex(api.Refusal,'live report:'):run(args)
            client.assert_not_called()
        self.assertFalse(self.output.exists());self.assertEqual(AuditHandler.gets,0)

    # POST and unrelated GET paths are refused before the HTTP opener runs.
    def test_negative_audit_refuses_completion_and_unrelated_get(self):
        with mock.patch.object(self.client.opener,'open') as opener:
            with self.assertRaisesRegex(api.Refusal,'GET-only'):self.client.request('/chat/completions',{})
            with self.assertRaisesRegex(api.Refusal,'GET path refused'):self.client.request('/chat/completions')
            opener.assert_not_called()
        self.assertEqual(Handler.bodies,[])

    # Output cannot be directed into the live run, including through a symlink, and makes no partial writes.
    def test_negative_audit_refuses_live_output_path(self):
        self.output.symlink_to(self.live,target_is_directory=True)
        factory=mock.Mock()
        with self.assertRaisesRegex(api.Refusal,'outside live'):
            audit.audit(self.live,self.output/'bad',client_factory=factory)
        factory.assert_not_called();self.assertEqual(list(self.live.iterdir()),[])

    # The CLI failure recorder also respects the live write boundary after a refused observer output path.
    def test_negative_observer_cli_failure_does_not_write_live_state(self):
        argv=['audit','--watch-end','--live-root',str(self.live),'--output',str(self.live/'bad')]
        with mock.patch('sys.argv',argv),mock.patch.dict(os.environ,{'PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(self.root)}),redirect_stdout(io.StringIO()) as log:
            self.assertEqual(audit.main(),1)
        self.assertIn('failed audit output must be outside live',log.getvalue())
        self.assertEqual(list(self.live.iterdir()),[])

    # A live-directory symlink to an external output cannot redirect the failure recorder back into live state.
    def test_negative_observer_symlink_failure_recorder_stays_outside_live(self):
        external=self.root/'external';external.mkdir()
        link=self.live/'external-link';link.symlink_to(external,target_is_directory=True)
        argv=['audit','--watch-end','--live-root',str(self.live),'--output',str(link)]
        with mock.patch('sys.argv',argv),mock.patch.dict(os.environ,{'PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(self.root)}),mock.patch.object(audit,'observe_end',side_effect=api.Refusal('planted observer failure')),redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(),1)
        self.assertEqual(list(self.live.iterdir()),[link])
        self.assertEqual(json.loads((self.root/'end-observer.json').read_text())['status'],'failed')

    # This planted20-millidollar mismatch is flagged by the actual loopback generation GET path.
    def test_negative_planted_cost_mismatch_is_flagged(self):
        AuditHandler.generation_cost=.12
        result=self.run_audit()
        self.assertEqual(result['mismatch_calls'],1);self.assertEqual(result['agreement_calls'],0)
        self.assertEqual(result['largest_absolute_difference_usd'],'0.02')
        self.assertEqual(result['per_call'][0]['status'],'mismatch')
        self.assertEqual(Handler.bodies,[])

    # Unchanged costs/prices agree; accepted calls, per-call price provenance and the isolated report are retained.
    def test_generation_cost_agreement_and_read_only_report_positive(self):
        result=self.run_audit()
        self.assertEqual(result['agreement_calls'],1);self.assertEqual(result['largest_absolute_difference_usd'],'0.0')
        self.assertEqual(result['unverified_calls'],0)
        self.assertTrue(all(r['status']=='agreement' for r in result['price_drift']))
        self.assertEqual(result['call_price_sources'][0]['pricing_source'],'recorded per-call live list')
        self.assertTrue((self.output/'API-CACHE-REPORT.md').exists())
        self.assertEqual(list(self.live.iterdir()),[]);self.assertEqual(Handler.bodies,[])

    # A changed provider price is flagged independently of cost agreement, without editing the frozen snapshot.
    def test_negative_planted_snapshot_price_drift_is_flagged(self):
        before=api.SNAPSHOT.read_bytes();AuditHandler.drift=True
        result=self.run_audit()
        changed=[r for r in result['price_drift'] if r['status']=='drift']
        self.assertEqual(len(changed),2)
        self.assertTrue(all(r['fields']['prompt']['drift'] for r in changed))
        self.assertEqual(api.SNAPSHOT.read_bytes(),before)

    # A missing generation identity cannot become an agreement or a fabricated zero-cost record.
    def test_negative_generation_identity_mismatch_and_unknown_cost(self):
        AuditHandler.generation_model='wrong'
        result=self.run_audit();self.assertEqual(result['unverified_calls'],1);self.assertEqual(result['agreement_calls'],0)
        self.assertEqual(result['per_call'][0]['status'],'identity-mismatch')
        record=audit.compare_cost(dict(self.row,usage={}),dict(id='gen-1',model=self.row['model'],total_cost=.1))
        self.assertEqual(record['status'],'missing-streamed-cost')

    # Dated IDs are accepted only through the exact frozen endpoint mapping; arbitrary suffixes still fail.
    def test_frozen_canonical_model_identity_and_wrong_date_negative(self):
        canonical=self.candidate['endpoint']['name'].partition(' | ')[2]
        self.assertTrue(canonical)
        record=dict(id='gen-1',model=canonical,total_cost=.1)
        self.assertEqual(audit.compare_cost(self.row,record)['status'],'agreement')
        record['model']=self.row['model']+'-20990101'
        self.assertEqual(audit.compare_cost(self.row,record)['status'],'identity-mismatch')

    # After a persistent endpoint limit, skipped generation GETs are an explicit ordered sample, never agreements.
    def test_endpoint_limit_declares_unverified_sample(self):
        AuditHandler.limited=True
        with mock.patch.object(audit.time,'sleep'):
            result=self.run_audit([self.row,dict(self.row,sequence=1,generation_id='gen-2')])
        self.assertEqual(result['agreement_calls'],0);self.assertEqual(result['unverified_calls'],2)
        self.assertEqual(result['fetched_unique_generations'],1)
        self.assertEqual(result['per_call'][1]['status'],'not-fetched-endpoint-limit')
        self.assertIn('persistent429',result['sampling'])

    # Echoed fake key bytes are redacted in both raw JSON and the CSV, including the identity-mismatch path.
    def test_generation_echo_redacted_from_json_csv_and_report(self):
        AuditHandler.generation_model='wrong';AuditHandler.generation_cost=self.secret
        self.run_audit()
        for path in self.output.iterdir():
            if path.is_file():self.assertNotIn(self.secret,path.read_text())
        self.assertIn('[REDACTED]',(self.output/'cost-cross-check.csv').read_text())

    # The proactive28.5GiB threshold refuses before client creation; legacy numerical floors are binary28GiB.
    def test_negative_disk_headroom_stops_before_key_read(self):
        factory=mock.Mock()
        with mock.patch.object(api.shutil,'disk_usage',return_value=mock.Mock(free=28*1024**3+1024**3//4)):
            with self.assertRaisesRegex(api.Refusal,'below28.5 GiB'):audit.audit(self.live,self.output,client_factory=factory)
        factory.assert_not_called();self.assertFalse(self.output.exists());self.assertEqual(api.MIN_FREE_BYTES,28*1024**3)

    # The pooled headline is explicitly named and both phase-only1M figures appear beside it.
    def test_pooled_million_label_next_to_phase_figures(self):
        self.run_audit()
        report=(self.output/'API-CACHE-REPORT.md').read_text()
        self.assertIn('Pooled1M USD/1000 | V1-only1M USD/1000 | Amendment-core1M USD/1000',report)
        data=json.loads((self.output/'analysis.json').read_text())
        self.assertIn('pooled',data['million_aggregation'])
        self.assertTrue(all(set(p['million_by_phase'])=={'v1','amendment'} for p in data['providers']))

    # Waiting for a final terminal file performs no HTTP/key read and invokes the final audit once afterward.
    def test_end_observer_waits_then_runs_read_only_audit_once(self):
        workflow=self.live/'workflow.json';workflow.write_text(json.dumps({'status':'running'}))
        def terminate_fixture(seconds):
            workflow.write_text(json.dumps({'status':'done'}));(self.live/'terminal.json').write_text('{"status":"done"}')
        result=dict(accepted_calls=1,agreement_calls=1,mismatch_calls=0,unverified_calls=0,largest_absolute_difference_usd='0')
        with mock.patch.object(audit.time,'sleep',side_effect=terminate_fixture) as sleep,mock.patch.object(audit,'audit',return_value=result) as run,mock.patch.object(api,'Client') as client:
            self.assertEqual(audit.observe_end(self.live,self.output),result)
            sleep.assert_called_once_with(30);run.assert_called_once_with(self.live,self.output,final=True);client.assert_not_called()
        self.assertEqual(json.loads((self.root/'end-observer.json').read_text())['status'],'done')


if __name__=='__main__':unittest.main()

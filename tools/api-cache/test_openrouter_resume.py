"""CPU amendment-resume controls using the exact r15l position/financial projection."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import openrouter_cache as api
import openrouter_amendment1 as amendment
import openrouter_recovery as recovery
import openrouter_restart as restart


class AmendmentResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'run';self.root.mkdir()
        self.fixture=json.loads((api.HERE/'fixtures/resume-r15l.json').read_text())
        self.planned={name:restart.phase_sessions(name) for name in ('v1','amendment')}
        self.sessions={s['session_id']:s for group in self.planned.values() for s in group}
        self.client=mock.Mock(secret='cpu-resume-control')
        data=dict(self.fixture['ledger'],receipts={})
        for entry in self.fixture['attempts']:
            record=dict(zip(self.fixture['columns'],entry))
            data['receipts'][record['phase']+'/%04d'%record['sequence']]=record['receipt']
        recovery.atomic_json(self.root/'ledger.json',data)
        recovery.atomic_json(self.root/'call34-resolution.json',self.fixture['baseline'])
        self.materialize('amendment')
        self.partial=next(s for s in self.planned['amendment'] if s['candidate']['endpoint']['tag']=='coreweave/nvfp4' and s['cohort']=='core' and s['depth']==1000000)

    def materialize(self,name):
        directory=self.root/name;directory.mkdir(exist_ok=True)
        for entry in self.fixture['attempts']:
            record=dict(zip(self.fixture['columns'],entry))
            if record['phase']!=name:continue
            s=self.sessions[record['session_id']]
            row={key:value for key,value in record.items() if key!='receipt'}
            row.update(model=s['candidate']['model'],endpoint_tag=s['candidate']['endpoint']['tag'],provider=amendment.identity(s['candidate']),
                       cohort=s['cohort'],depth=s['depth'],accepted=False,charge_resolution=record['receipt'],started_at=record['ended_at'])
            recovery.atomic_json(directory/('intent-%04d.json'%record['sequence']),{key:value for key,value in row.items() if key!='charge_resolution'})
            recovery.atomic_json(directory/('call-%04d.json'%record['sequence']),row)
        if name=='amendment':
            recovery.atomic_json(directory/'ranking.json',self.fixture['ranking'])
            recovery.atomic_json(directory/'terminal.json',dict(status='failed',attempts=159,reason='free disk below28 GiB hard floor'))
        else:recovery.atomic_json(directory/'terminal.json',self.fixture['v1_terminal'])

    def phase(self):
        return restart.Phase(self.root,'amendment',self.client,resume=True)

    # All159 existing indices are consumed; real scheduling skips complete cells and first selects CoreWeave1M index4.
    def test_exact_r15l_resume_next_indices_without_replay(self):
        before=(self.root/'ledger.json').read_bytes();ranking=(self.root/'amendment/ranking.json').read_bytes()
        phase=self.phase();old={(r['session_id'],r['call_index']) for r in phase.rows};calls=[]
        def call(s,index,previous_end=None):
            self.assertNotIn((s['session_id'],index),old);calls.append((s['session_id'],index,previous_end))
            return None
        with mock.patch.object(phase,'call',side_effect=call),mock.patch.object(amendment,'priority_order',side_effect=AssertionError('ranking must stay frozen')):
            self.assertEqual(phase.run(),0)
        self.assertEqual(len(old),159);self.assertEqual(len(phase.positions),252)
        self.assertEqual(calls[0][:2],(self.partial['session_id'],4))
        self.assertIsNotNone(calls[0][2])
        self.assertTrue(all(index==phase.positions[sid]['next_index'] for sid,index,_ in calls))
        self.assertEqual((self.root/'ledger.json').read_bytes(),before)
        self.assertEqual((self.root/'amendment/ranking.json').read_bytes(),ranking)

    # A replay is refused before metadata, waiting, ledger mutation or any POST; the next index remains a valid control.
    def test_named_negative_replayed_index_refused_before_requests(self):
        phase=self.phase();before=(self.root/'ledger.json').read_bytes()
        with mock.patch.object(restart,'metadata_get') as metadata:
            with self.assertRaisesRegex(api.Refusal,'replayed cell/index refused'):phase.call(self.partial,3)
        metadata.assert_not_called();self.assertEqual((self.root/'ledger.json').read_bytes(),before)
        phase.assert_next_index(self.partial,4)

    # A durable reserved intent added after reconstruction cannot be replayed even if no response row was appended.
    def test_named_negative_reserved_intent_index_refused(self):
        phase=self.phase()
        recovery.atomic_json(phase.out/'intent-0159.json',dict(session_id=self.partial['session_id'],call_index=4))
        with self.assertRaisesRegex(api.Refusal,'durable intent already exists'):phase.call(self.partial,4)

    # Altering the cumulative balance cannot discard old reservations while leaving every receipt looking valid.
    def test_named_negative_inconsistent_committed_ledger_refused(self):
        path=self.root/'ledger.json';data=json.loads(path.read_text());data['committed_upper_usd']='32'
        recovery.atomic_json(path,data);before=path.read_bytes();terminal=(self.root/'amendment/terminal.json').read_bytes()
        with self.assertRaisesRegex(api.Refusal,'inconsistent ledger: committed total'):self.phase()
        self.assertEqual(path.read_bytes(),before);self.assertEqual((self.root/'amendment/terminal.json').read_bytes(),terminal)
        self.assertFalse(list((self.root/'amendment').glob('resume-state-*.json')))

    # A settled row whose receipt disappears is a ledger fault, never an opportunity to reconcile/debit it again.
    def test_named_negative_missing_settled_receipt_refused(self):
        path=self.root/'ledger.json';data=json.loads(path.read_text());data['receipts'].pop('amendment/0158')
        recovery.atomic_json(path,data);before=path.read_bytes()
        with self.assertRaisesRegex(api.Refusal,'resume has unresolved charge'):self.phase()
        with mock.patch.object(recovery,'reconcile_call') as reconcile:
            with self.assertRaisesRegex(api.Refusal,'settled row receipt missing'):restart.recover_interrupted(self.root,'amendment',self.client)
        reconcile.assert_not_called();self.assertEqual(path.read_bytes(),before)

    # A per-call maximum inconsistent with its intent/receipt is rejected independently of sufficient account funds.
    def test_named_negative_receipt_maximum_mismatch_refused(self):
        path=self.root/'amendment/intent-0158.json';data=json.loads(path.read_text());data['maximum_cost_usd']='1'
        recovery.atomic_json(path,data)
        with self.assertRaisesRegex(api.Refusal,'row differs from intent'):self.phase()

    # Existing permanent receipts remain consumed positions, keep their full holds, and survive reconstruction unchanged.
    def test_permanent_hold_carried_forward_without_release(self):
        path=self.root/'ledger.json';data=json.loads(path.read_text());key='amendment/0158';old=data['receipts'][key]
        maximum=old['maximum_usd'];replacement=dict(method='permanent maximum reservation',permanent_reservation=True,
            status='unresolved-reserved',resolved=False,maximum_usd=maximum,retained_reservation_usd=maximum)
        data['receipts'][key]=replacement
        data['committed_upper_usd']=str(Decimal(data['committed_upper_usd'])-Decimal(old['cost_usd'])+Decimal(maximum))
        data['restart_reported_usd']=str(Decimal(data['restart_reported_usd'])-Decimal(old['cost_usd']))
        data['permanent_reservation_count']=1;recovery.atomic_json(path,data);before=path.read_bytes()
        rowpath=self.root/'amendment/call-0158.json';row=json.loads(rowpath.read_text());row.update(charge_resolution=replacement,cost_usd=None)
        recovery.atomic_json(rowpath,row)
        phase=self.phase();self.assertEqual(phase.positions[self.partial['session_id']]['next_index'],4)
        self.assertEqual(len(phase.budget.permanent_receipts()),1);self.assertEqual(path.read_bytes(),before)

    # A receipt-backed intent without a response is recovered as consumed, preserving its hold and skipping its index.
    def test_reserved_intent_without_response_advances_without_replay(self):
        directory=self.root/'amendment';intent=json.loads((directory/'intent-0158.json').read_text())
        intent.update(sequence=159,call_index=4,started_at=api.utc());intent.pop('cost_usd',None);intent.pop('ended_at',None)
        text=amendment.payload(dict(new_tokens=1065,framing_reserve=128,max_tokens=16),self.partial,4)['messages'][0]['content'][0]['text']
        intent['prompt_sha256']=hashlib.sha256(text.encode()).hexdigest()
        recovery.atomic_json(directory/'intent-0159.json',intent)
        budget=recovery.Budget.load(self.root/'ledger.json');maximum=intent['maximum_cost_usd']
        budget.permanent_call(maximum,'amendment/0159',dict(expected_usd=str(budget.known),maximum_usd=maximum,samples=[{'error':'cpu-unstable'}]))
        before=(self.root/'ledger.json').read_bytes()
        with mock.patch.object(self.client,'get',side_effect=AssertionError('receipt already exists')):
            restart.recover_interrupted(self.root,'amendment',self.client)
        phase=self.phase();self.assertEqual(phase.positions[self.partial['session_id']]['next_index'],5)
        self.assertEqual((self.root/'ledger.json').read_bytes(),before)
        with self.assertRaisesRegex(api.Refusal,'replayed cell/index refused'):phase.call(self.partial,4)

    # Real call recording labels every continuation row and only its first boundary; registered pauses and prefix stay intact.
    def test_resumed_cells_labelled_and_registered_pause_preserved(self):
        phase=self.phase();s=self.partial;position=phase.positions[s['session_id']];clock=[10000.]
        wall=time.time()
        def metadata(client,path):
            if path=='/key':return {'data':{'usage':str(phase.budget.known)}}
            model=path.split('/models/')[1].removesuffix('/endpoints')
            return next(m for m in json.loads(api.SNAPSHOT.read_text())['models'] if m['data']['id']==model)
        def stream(client,payload,endpoint,depth,journal):
            index=len(phase.rows)-159+4
            usage=dict(prompt_tokens=depth-(9-index)*1065,completion_tokens=2,cost=0,prompt_tokens_details={'cached_tokens':900000})
            return dict(http_status=200,generation_id='cpu-generation',usage=usage,observed_models=[s['candidate']['model']],
                        observed_providers=[endpoint['provider_name']],usage_complete=True,cost_usd=0)
        with mock.patch.object(restart,'metadata_get',side_effect=metadata),mock.patch.object(recovery,'stream',side_effect=stream),mock.patch.object(restart.time,'time',side_effect=lambda:wall+clock[0]-10000),mock.patch.object(restart.time,'monotonic',side_effect=lambda:clock[0]),mock.patch.object(restart.time,'sleep',side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)):
            index,previous=phase.next_call(s);ended=phase.call(s,index,previous)
            phase.call(s,index+1,ended)
        first,second=phase.rows[-2:]
        self.assertTrue(first['resumed_cell']);self.assertTrue(first['resume_boundary'])
        self.assertTrue(second['resumed_cell']);self.assertFalse(second['resume_boundary'])
        self.assertEqual(first['resume_from_index'],4);self.assertEqual(first['resume_previous_ended_at'],position['previous_ended_at'])
        self.assertEqual(first['pause_target_seconds'],s['pauses'][4])
        self.assertGreater(first['pause_actual_seconds'],s['pauses'][4])
        self.assertAlmostEqual(second['pause_actual_seconds'],s['pauses'][5])
        self.assertTrue(first['accepted']);self.assertTrue(second['accepted'])
        m=dict(new_tokens=1065,framing_reserve=128,max_tokens=16)
        old=amendment.payload(m,s,3)['messages'][0]['content'][0]['text']
        new=amendment.payload(m,s,4)['messages'][0]['content'][0]['text']
        self.assertTrue(new.startswith(old))

    # Supervisor resumes only amendment, preserves v1 custody and money, and records the replacement amendment PID.
    def test_supervisor_exact_r15l_skips_completed_v1(self):
        self.materialize('v1')
        recovery.atomic_json(self.root/'workflow.json',dict(status='failed',pid=158133,phase_pids={'v1':158396,'amendment':263015}))
        recovery.atomic_json(self.root/'terminal.json',dict(status='failed'))
        before=(self.root/'ledger.json').read_bytes();v1=(self.root/'v1/terminal.json').read_bytes();children=[]
        self.client.get.return_value={'data':{'usage':'20.865914938'}}
        def spawn(argv,**kwargs):
            self.assertEqual(argv[argv.index('--phase')+1],'amendment');self.assertIn('--resume',argv)
            recovery.atomic_json(self.root/'amendment/terminal.json',dict(status='done',attempts=159))
            child=mock.Mock(pid=999001);child.wait.return_value=0;children.append(child)
            return child
        samples=[dict(at=str(i),usage_usd='20.865914938') for i in range(3)]
        with mock.patch.object(restart,'LOCK_PATH',self.root/'lease'),mock.patch.object(recovery,'Client',return_value=self.client),mock.patch.object(recovery,'settled_samples',return_value=samples),mock.patch.object(restart,'public_snapshot'),mock.patch.object(restart.subprocess,'Popen',side_effect=spawn),mock.patch.object(restart.subprocess,'run'),mock.patch.object(restart.subprocess,'check_output',side_effect=['c'*40,'']):
            self.assertEqual(restart.supervise(self.root,resume=True),0)
        self.assertEqual(len(children),1);workflow=json.loads((self.root/'workflow.json').read_text())
        self.assertEqual(workflow['phase_pids'],{'amendment':999001});self.assertEqual(workflow['completed_phases'],['v1'])
        self.assertEqual((self.root/'ledger.json').read_bytes(),before);self.assertEqual((self.root/'v1/terminal.json').read_bytes(),v1)

    # Analysis separates entire interrupted cells and their continuations without rewriting the original rows.
    def test_report_separates_resumed_cadence_from_uninterrupted_cells(self):
        s=self.partial;c=s['candidate'];base=dict(model=c['model'],endpoint_tag=c['endpoint']['tag'],provider=amendment.identity(c),
            phase='amendment',cohort='core',depth=1000000,session_id=s['session_id'],call_index=3,accepted=True,
            observed_models=[c['model']],observed_providers=[c['endpoint']['provider_name']],prompt_tokens=990000,cached_tokens=980000,
            new_tokens_estimate=1065,pause_target_seconds=8,pause_actual_seconds=8)
        rows=[base,dict(base,call_index=4,resumed_cell=True,resume_boundary=True,pause_actual_seconds=3600),dict(base,session_id='independent-cell')]
        result=restart.analysis(self.root,rows=rows);own=next(p for p in result['providers'] if p['provider']==base['provider'])
        splits=own['phases']['amendment']['core']['cadence_subsets']
        self.assertEqual(splits['resumed_cells']['accepted'],2);self.assertEqual(splits['continuation_only']['accepted'],1)
        self.assertEqual(splits['uninterrupted']['accepted'],1)
        self.assertEqual(splits['continuation_only']['matched_target_pause_levels']['8']['n'],0)
        self.assertNotIn('resumed_cell',base)
        self.assertIn('Interrupted series',(self.root/'API-CACHE-REPORT.md').read_text())


if __name__=='__main__':unittest.main()

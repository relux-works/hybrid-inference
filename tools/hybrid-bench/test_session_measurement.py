"""R49 scheduling, workload and quality contracts; native logits are checked separately."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from bench_common import Refusal
import growing_session as g
import session_measurement as measurement
import session_runtime as runtime
from store import Store,prefix_hash

MODEL='0'*64


def export_fixture():
    # Synthetic loader control only; never delivered as the owner's measured trace.
    return dict(schema='hybrid-owner-session-export-v1',model_sha256=MODEL,source_log_sha256='1'*64,
        sessions=[dict(id='unit-session',started_utc='2026-11-01T00:00:00Z',base_ids=[1]*100000,
            turns=[dict(source_turn=i,input_ids=[2]*(10000 if i<3 else 2000),output_ids=[3,4],pause_seconds=0) for i in range(40)])])


class WorkloadTests(unittest.TestCase):
    # Admission preserves chronological contiguous IDs and observed lengths, independent of routing choices.
    def test_owner_workload_earliest_contiguous_window(self):
        export=export_fixture()
        later=copy.deepcopy(export['sessions'][0]);later['id']='later';later['started_utc']='2026-11-02T00:00:00Z'
        export['sessions'].insert(0,later)
        row=runtime.select_workload(export,MODEL,10)
        self.assertEqual(row['source_session'],'unit-session')
        self.assertEqual(row['source_turns'],list(range(40)))
        self.assertEqual((row['start_depth'],row['final_depth']),(100000,204080))
        self.assertEqual(row['base_ids'],export['sessions'][1]['base_ids'])
        self.assertEqual(row['turns'],export['sessions'][1]['turns'])

    # Zero-arrival calls and >16k observed answers are preserved if the complete window meets the depth rule.
    def test_owner_workload_does_not_clamp_zero_inputs_or_large_answers(self):
        export=export_fixture()
        rows=export['sessions'][0]['turns']
        for i,row in enumerate(rows):row['input_ids']=[2]*(10000 if i<3 else 2000 if i<12 else 900)
        rows[0]['output_ids']=[3]*20000;rows[-1]['input_ids']=[]
        result=runtime.select_workload(export,MODEL,10)
        self.assertEqual(len(result['turns'][0]['output_ids']),20000)
        self.assertEqual(result['turns'][-1]['input_ids'],[])
        self.assertEqual(result['final_depth'],192378)

    # Aggregate statistics, reordered source indices and insufficient large portions cannot be promoted to an owner trace.
    def test_owner_workload_refuses_aggregate_or_unqualified_trace(self):
        with self.assertRaisesRegex(Refusal,'quantile sampling'):
            runtime.select_workload(dict(schema='quantiles'),MODEL,10)
        broken=export_fixture();broken['sessions'][0]['turns'][2]['source_turn']=99
        with self.assertRaisesRegex(Refusal,'must be contiguous'):runtime.select_workload(broken,MODEL,10)
        broken=export_fixture()
        for row in broken['sessions'][0]['turns']:row['input_ids']=[2]*2000
        with self.assertRaisesRegex(ValueError,'no qualifying contiguous'):runtime.select_workload(broken,MODEL,10)
        broken=export_fixture();broken['sessions'][0]['turns'][0]['input_ids'][0]=10
        with self.assertRaisesRegex(Refusal,'invalid owner token'):runtime.select_workload(broken,MODEL,10)

    # bf16 is the explicit default profile; f32 control labels cannot alias the bf16 arms.
    def test_arm_labels_and_default_wire_profile(self):
        args=measurement.parser().parse_args(['--client-state','s','--owner-export','o','--owner-export-sha256',MODEL,'--output','r','--ca-file','c',
            '--fixture','f','--policy','p','--import-fit','i','--device','m3-max-128gb','--link-label','home'])
        self.assertEqual(args.state_dtype,'bf16')
        self.assertEqual(measurement.arm_names('bf16'),('local','hybrid','hybrid-background'))
        self.assertEqual(measurement.arm_names('f32'),('local','hybrid-f32','hybrid-background-f32'))
        with self.assertRaisesRegex(Refusal,'unsupported live-wire'):measurement.arm_names('q8')
        self.assertEqual(g.arm_names(set(),False),['hybrid','local'])


class WirePolicyTests(unittest.TestCase):
    # The profile changes only 24 f32 tensor bytes to 12 bf16 bytes; eight metadata bytes remain unchanged.
    def test_bf16_policy_uses_matching_geometry_and_f32_control(self):
        layout=dict(architecture='qwen35',layers=[dict(recurrent=True)],
            recurrent_layout=dict(state=1,inner=2,conv=2,groups=1))
        geometry=dict(kv_bytes=10,state_bytes=32)
        policy=dict(kv_bytes=10,state_bytes=32,server_rate=100)
        self.assertEqual(measurement.wire_policy(policy,layout,geometry,'f32'),policy)
        self.assertEqual(measurement.wire_policy(policy,layout,geometry,'bf16'),dict(policy,state_bytes=20))
        self.assertEqual(policy['state_bytes'],32)
        with self.assertRaisesRegex(Refusal,'differ from native geometry'):
            measurement.wire_policy(dict(policy,state_bytes=0),layout,geometry,'bf16')


class CatchupPolicyTests(unittest.TestCase):
    # Pending node work must be priced before offloading; the unchanged r46 policy remains a separate control.
    def test_pending_catchup_changes_route_and_is_not_free(self):
        from policy import Policy
        policy=Policy(local_rates=((0,100),),server_rate=1000,link_rate=1e9,kv_bytes=0,state_bytes=0,fixed=0)
        self.assertTrue(runtime.catchup_decision(policy,100,1000,0)['offload'])
        charged=runtime.catchup_decision(policy,100,1000,2000)
        self.assertFalse(charged['offload']);self.assertAlmostEqual(charged['offload_seconds'],2.1)
        self.assertTrue(policy.choose(100,1000)['offload'])


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(':memory:',MODEL)
        row=self.store.select('client',[],explicit='sid')
        row.update(tokens=[1,2,3],remote_id='remote',slot=0)
        self.store.save(row);self.store.set_slot('sid',0)

    def tearDown(self):
        self.store.close();self.tmp.cleanup()

    # A remote acknowledgement can advance only within the current durable journal and must not replace newer fields.
    def test_background_ack_preserves_newer_journal_and_slot(self):
        newer=self.store.get('sid');newer.update(tokens=[1,2,3,4],slot=1,state_dtype='bf16',journal=dict(production_turn=2))
        self.store.save(newer);self.store.set_slot('sid',1)
        self.store.acknowledge_remote('sid','client','remote',3,prefix_hash([1,2,3]))
        after=self.store.get('sid')
        self.assertEqual(after['tokens'],[1,2,3,4]);self.assertEqual(after['slot'],1)
        self.assertEqual(after['journal'],dict(production_turn=2));self.assertEqual(after['state_dtype'],'bf16')
        self.assertEqual(after['remote_n'],3)

    # A local turn that started before a worker ACK cannot erase that newer remote progress at commit.
    def test_foreground_commit_cannot_erase_background_ack(self):
        stale=self.store.get('sid')
        self.store.acknowledge_remote('sid','client','remote',3,prefix_hash([1,2,3]))
        stale.update(tokens=[1,2,3,4],journal=dict(production_turn=2))
        self.store.commit_evaluated(stale)
        current=self.store.get('sid')
        self.assertEqual(current['remote_n'],3);self.assertEqual(current['tokens'],[1,2,3,4])
        self.assertEqual(current['journal'],dict(production_turn=2))
        invalid=dict(current,tokens=[9])
        with self.assertRaisesRegex(ValueError,'does not extend'):self.store.commit_evaluated(invalid)
        self.assertEqual(self.store.get('sid'),current)

    # A named ahead-of-journal negative cannot partially acknowledge or alter committed state.
    def test_background_ack_never_ahead_of_client_journal(self):
        before=self.store.get('sid')
        with self.assertRaisesRegex(ValueError,'ahead of or divergent'):
            self.store.acknowledge_remote('sid','client','remote',4,prefix_hash([1,2,3,4]))
        self.assertEqual(self.store.get('sid'),before)
        with self.assertRaisesRegex(ValueError,'stale remote'):
            self.store.acknowledge_remote('sid','client','another',3,prefix_hash([1,2,3]))
        self.assertEqual(self.store.get('sid'),before)

    def app(self):
        config=dict(model_sha256=MODEL,prefill_url='http://unused',decode_url='http://unused',slot_path=self.tmp.name,
            clients={'credential':dict(id='client',prefill_token='remote-token')},policy=dict(always=True))
        app=runtime.SessionProxy(config,':memory:',g.Events(None),background=False)
        row=app.store.select('client',[],explicit='sid');row.update(tokens=[1,2,3],remote_id='remote',slot=0)
        app.store.save(row);app.store.set_slot('sid',0)
        return app

    # The worker verifies a committed prefix before dispatch, and rejects an ahead node with zero token uploads.
    def test_background_catchup_never_runs_ahead_named_negative(self):
        app=self.app()
        node=mock.Mock()
        node.request.return_value=dict(n=4,prefix_hash=prefix_hash([1,2,3,4]),model_sha256=MODEL)
        app.remote=mock.Mock(return_value=node)
        try:
            with self.assertRaisesRegex(ValueError,'ahead of or divergent'):app.catchup_committed('sid')
            self.assertEqual([r.args[0] for r in node.request.call_args_list],['GET'])
            self.assertEqual(app.store.get('sid')['tokens'],[1,2,3])
            self.assertEqual(app.store.get('sid')['remote_n'],0)
        finally:app.store.close()

    # A valid worker sends exactly the durable suffix and confirms application with GET, rather than trusting 202.
    def test_background_catchup_positive_committed_suffix(self):
        app=self.app();applied=[];calls=[]
        class Remote:
            def request(self,method,path,data=None,headers=None):
                calls.append((method,path,data))
                if method=='POST':
                    applied.extend(data['tokens']);return dict(queued=True)
                return dict(n=len(applied),prefix_hash=prefix_hash(applied),model_sha256=MODEL)
        app.remote=lambda _:Remote()
        try:
            app.catchup_committed('sid')
            posted=[r[2] for r in calls if r[0]=='POST']
            self.assertEqual([r['tokens'] for r in posted],[[1,2,3]])
            self.assertIs(posted[0]['return_delta'],False)
            self.assertEqual(app.store.get('sid')['remote_n'],3)
            self.assertEqual([r[0] for r in calls],['GET','POST','GET'])
        finally:app.store.close()

    # The worker skips a declared in-flight remote prompt; this is distinct from accepting an unexplained ahead node.
    def test_background_does_not_race_uncommitted_offload(self):
        app=self.app();app.remote=mock.Mock()
        try:
            app.remote_inflight.add('sid');app.catchup_committed('sid')
            app.remote.assert_not_called()
            app.end_turn('sid');self.assertNotIn('sid',app.remote_inflight)
        finally:app.store.close()


class TimingAndQualityTests(unittest.TestCase):
    # A callback blocked after publication cannot move the measured end or prevent the timed result being observed.
    def test_evidence_never_blocks_a_timed_turn(self):
        published=threading.Event();entered=threading.Event();release=threading.Event()
        result={};errors=[];ticks=iter([100,200,1000,9000])
        def publish(row,timing):result.update(row=row,timing=timing);published.set()
        def evidence():
            entered.set()
            if not published.is_set():raise AssertionError('evidence ran before timed publication')
            if not release.wait(5):raise AssertionError('test evidence was not released')
            return 'snapshot'
        def run():
            try:result['all']=runtime.run_turn(lambda:'turn',evidence,publish,lambda:next(ticks))
            except BaseException as error:errors.append(error)
        thread=threading.Thread(target=run);thread.start()
        try:
            self.assertTrue(entered.wait(5));self.assertTrue(published.is_set())
            self.assertEqual(result['timing'],dict(start_ns=100,end_ns=200,seconds=1e-7))
            self.assertTrue(thread.is_alive())
        finally:release.set();thread.join(5)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
        self.assertEqual(result['all'][3]['seconds'],8e-6)

    # Session decode rate is weighted by native decode time, and payload throughput uses its own transfer interval.
    def test_session_aggregates_preserve_native_timing(self):
        calls=[dict(turn=1,timing=dict(seconds=2),ttft_seconds=.5,path='offload',generated_tokens=3,
                    meter=dict(decode_ms=1000,wire_bytes=100,transfer_ms=500)),
               dict(turn=2,timing=dict(seconds=5),ttft_seconds=.7,path='offload',generated_tokens=5,
                    meter=dict(decode_ms=4000,wire_bytes=200,transfer_ms=500))]
        result=measurement.summarize([],calls)
        self.assertAlmostEqual(result['decode_tokens_per_second'],1.2)
        self.assertAlmostEqual(result['achieved_payload_MB_s'],.0003)
        self.assertEqual(result['timed_turn_seconds'],7)
        self.assertEqual(result['node_compute_unknown_requests'],0)

    @staticmethod
    def quality():
        source=dict(state_dtype='bf16',tokens=[1,2],pairing=dict(mode=g.COMMON_JOURNAL,journal_hash=prefix_hash([1,2])))
        row=dict(status_rule='session_quality_v1',pairing=source['pairing'],k=64,top1_agreed=63,quality_gate=True,
            exact=False,exit_code=5,lossy_import=True,free_running_first_divergence=0,
            continuations={n:dict(teacher_tokens=[0]*64,free_tokens=[0]*64) for n in ('L','H')})
        row['continuations']['H']['free_tokens'][0]=1
        return source,row

    # Pairing/verdict controls are isolated here; real raw-logit validation is exercised by the native qualification.
    @mock.patch.object(g,'validate_raw_positions')
    def test_quality_teacher_pairing_and_lossy_negative(self,raw):
        source,row=self.quality()
        self.assertTrue(measurement.validate_quality(row,source))
        raw.assert_called_once_with(row,2)
        divergent=copy.deepcopy(row);divergent['continuations']['H']['teacher_tokens'][0]=1
        with self.assertRaisesRegex(Refusal,'teacher-forced inputs differ'):measurement.validate_quality(divergent,source)
        with self.assertRaisesRegex(Refusal,'cannot claim exactness'):measurement.validate_quality(dict(row,exact=True),source)
        with self.assertRaisesRegex(Refusal,'contradicts 63/64'):measurement.validate_quality(dict(row,top1_agreed=62),source)
        self.assertFalse(measurement.validate_quality(dict(row,top1_agreed=62,quality_gate=False),source))
        self.assertEqual(row['free_running_first_divergence'],0)


if __name__=='__main__':unittest.main()

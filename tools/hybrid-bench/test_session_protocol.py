"""Preregistered selection, common replay, waits and clustered analysis controls."""
import copy
from datetime import datetime,timedelta,timezone
import unittest
from unittest import mock
from bench_common import Refusal
import session_protocol as p
import session_measurement as m

MODEL='0'*64


def session(index,large=False):
    # Synthetic controls cover variable-length natural portions; never delivered as owner evidence.
    now=datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(days=index)
    start=now.isoformat();rows=[]
    for turn in range(60 if not large else 40):
        now+=timedelta(seconds=8 if turn<54 else 60);request=now.isoformat();now+=timedelta(seconds=2)
        n=10000 if large and turn<3 else 2000 if large else 1666
        rows.append(dict(source_turn=turn,input_ids=[2]*n,output_ids=[3,4],request_start_utc=request,response_end_utc=now.isoformat()))
    return dict(id='session-'+str(index),base_ids=[1]*100000,base_response_end_utc=start,turns=rows)


def export():
    return dict(schema='hybrid-owner-session-export-v2',model_sha256=MODEL,source_log_sha256='1'*64,
        population=dict(scope='all-retained-owner-sessions',inventory_complete=True,start_utc='2026-01-01T00:00:00Z',end_utc='2026-12-31T00:00:00Z'),
        sessions=[session(i,large=i>=5) for i in range(10)])


def frozen(data=None):return p.freeze_export(data or export(),'2'*64,MODEL,10,dict(commit='3'*40))


class ProtocolTests(unittest.TestCase):
    # Main selection is deterministic over content, permits naturally small portions and keeps variable turn counts.
    def test_random_main_does_not_filter_for_large_outputs_or_routes(self):
        data=export();f=p.freeze_export(data,'2'*64,MODEL,10,{});selected=p.select_set(f)
        reverse=copy.deepcopy(data);reverse['sessions'].reverse()
        again=p.select_set(p.freeze_export(reverse,'2'*64,MODEL,10,{}))
        self.assertEqual([r['session_digest'] for r in selected['sets']['main']['sessions']],
                         [r['session_digest'] for r in again['sets']['main']['sessions']])
        row=p.recorded_session(session(0),MODEL,10)
        self.assertEqual(len(row['turns']),60);self.assertEqual(row['start_depth'],100000)
        self.assertTrue(all(len(t['input_ids'])<2000 for t in row['turns']))
        self.assertGreaterEqual(row['final_depth'],200000)
        self.assertEqual(selected['sets']['main']['count'],5)

    # Five natural small-portion histories must qualify even when no favourable-case history exists.
    def test_main_accepts_all_small_portions_without_offload_filter(self):
        data=export();data['sessions']=[session(i) for i in range(5)]
        selected=p.select_set(frozen(data))
        self.assertEqual(selected['sets']['main']['count'],5)
        self.assertEqual(selected['sets']['favourable']['status'],'insufficient')
        self.assertTrue(all(all(len(t['input_ids'])<2000 for t in r['turns']) for r in selected['sets']['main']['sessions']))

    # A complete zero-length recorded response is valid replay work, not an undisclosed output-size filter.
    def test_zero_length_recorded_response_is_preserved(self):
        row=session(0);row['turns'][0]['output_ids']=[]
        selected=p.recorded_session(row,MODEL,10)
        self.assertEqual(selected['turns'][0]['output_ids'],[])
        self.assertEqual(len(selected['turns']),60)

    # A fully evaluated zero-output checkpoint uses one fixed probe token in both references, leaving the live journal unchanged.
    def test_zero_output_checkpoint_probe_does_not_change_recorded_journal(self):
        import tempfile
        from pathlib import Path
        from store import prefix_hash
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'slots').mkdir();(root/'slots/local-1.bin').write_bytes(b'state')
            app=mock.Mock();app.model=MODEL
            app.decode.request.side_effect=[dict(tokens=[1,2,3]),dict(n_saved=3)]
            journal=[1,2,3];row=m.snapshot(app,root,'local',1,journal)
            self.assertEqual(journal,[1,2,3]);self.assertEqual(row['tokens'],[1,2,3,0])
            self.assertEqual(row['probe_prefix_extension'],[0]);self.assertEqual(row['recorded_journal_hash'],prefix_hash(journal))

    # A preregistration binding must invoke signature/ancestry verification and reject a changed document.
    def test_prereg_binding_refuses_document_tampering(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'tools/hybrid-bench/PREREG-E2E.md';path.parent.mkdir(parents=True)
            registered=(p.PROTOCOL+' registered').encode();path.write_bytes(registered)
            with mock.patch.object(p.subprocess,'run') as run, mock.patch.object(p.subprocess,'check_output',return_value=registered):
                binding=p.prereg_binding(root,'a'*40)
                self.assertEqual(binding['commit'],'a'*40)
                self.assertIn('verify-commit',run.call_args_list[0].args[0])
                path.write_bytes(registered+b' changed')
                with self.assertRaisesRegex(Refusal,'changed after'):p.prereg_binding(root,'a'*40)

    # Duplicate filenames cannot create extra sample units; fewer than five must block rather than repeat one session.
    def test_duplicate_history_and_small_population_refuse(self):
        data=export();data['sessions']=[session(0)]*5
        f=frozen(data);self.assertEqual(f['eligible_count'],1)
        self.assertEqual(sum(r['reason']=='duplicate-content' for r in f['excluded']),4)
        with self.assertRaisesRegex(Refusal,'five distinct'):p.select_set(f)
        data=export();data['schema']='hybrid-owner-session-export-v1'
        with self.assertRaisesRegex(Refusal,'aggregate gaps'):frozen(data)

    # Waits use response end, not the request-start gap; nearest-rank output is fixed and no 40-turn clamp is applied.
    def test_measured_wait_distribution_and_chronology_negative(self):
        row=p.recorded_session(session(0),MODEL,10)
        self.assertEqual(row['wait_distribution']['median'],8)
        self.assertEqual(row['wait_distribution']['p90'],8)
        self.assertEqual(row['wait_distribution']['p99'],60)
        self.assertEqual(row['turns'][1]['pause_seconds'],8)
        bad=session(0);bad['turns'][1]['request_start_utc']=bad['base_response_end_utc']
        with self.assertRaisesRegex(Refusal,'overlapping'):p.recorded_session(bad,MODEL,10)

    # Natural long waits cannot become an undisclosed duration filter on the main population.
    def test_long_recorded_wait_is_not_filtered(self):
        row=session(0);last=row['turns'][-1]
        start=p.stamp(last['request_start_utc'])+timedelta(days=2)
        last.update(request_start_utc=start.isoformat(),response_end_utc=(start+timedelta(seconds=2)).isoformat())
        selected=p.recorded_session(row,MODEL,10)
        self.assertEqual(selected['turns'][-1]['pause_seconds'],172860)

    # Evidence consumes the existing wait; neither an extra full wait nor an overrun can qualify.
    def test_evidence_uses_existing_tool_deadline_no_free_background_time(self):
        sleeps=[];clock=mock.Mock(side_effect=[3_000_000_000,8_000_000_000])
        result=p.finish_wait(8_000_000_000,clock,sleeps.append)
        self.assertEqual(sleeps,[5]);self.assertEqual(result['remaining_sleep_seconds'],5)
        with self.assertRaisesRegex(Refusal,'timing contaminated'):
            p.finish_wait(8_000_000_000,lambda:9_000_000_000,sleeps.append)
        self.assertEqual(sleeps,[5])
        self.assertEqual(p.finish_wait(0,lambda:1,lambda _:None,evidence_performed=False)['remaining_sleep_seconds'],0)

    # Final closure drops queued catch-up after timing, while joining the already dispatched job.
    def test_final_boundary_drops_queued_catchup_and_joins_inflight(self):
        import threading
        from session_runtime import BackgroundCatchup
        entered,released=threading.Event(),threading.Event();applied=[]
        def apply(sid):
            entered.set();self.assertTrue(released.wait(2));applied.append(sid)
        worker=BackgroundCatchup(apply);worker.submit('inflight');self.assertTrue(entered.wait(2));worker.submit('queued')
        closer=threading.Thread(target=lambda:worker.close(discard_pending=True));closer.start()
        with worker.condition:self.assertTrue(worker.condition.wait_for(lambda:worker.stopping,timeout=2))
        released.set();closer.join(2)
        self.assertFalse(closer.is_alive());self.assertEqual(applied,['inflight']);self.assertFalse(worker.pending)

    # Each comparison alternates its own local baseline and hybrid over five distinct sessions, with reversed block order on fast link.
    def test_paired_schedule_has_distinct_sessions_and_three_declared_arms(self):
        selected=p.select_set(frozen());home=p.schedule(selected,'main','home');fast=p.schedule(selected,'main','fast')
        self.assertEqual(len(home),20);self.assertEqual([r['arm'] for r in home[:4]],['local','hybrid','local','hybrid'])
        self.assertEqual([r['arm'] for r in fast[:4]],['local','hybrid-background','local','hybrid-background'])
        self.assertEqual(len({r['session_digest'] for r in home[:10]}),5)
        self.assertEqual({r['arm'] for r in home},{'local','hybrid','hybrid-background'})

    # An exact-zero-gain/all-local result remains an analyzable failure of the gain criterion, not a routing admission failure.
    def test_success_and_no_offload_nongain_and_cluster_negative(self):
        pairs=[dict(session_digest=str(i),local_seconds=100,hybrid_seconds=75,qualified=True) for i in range(5)]
        self.assertTrue(p.paired_analysis(pairs)['success'])
        for row in pairs:row['hybrid_seconds']=100
        self.assertFalse(p.paired_analysis(pairs)['success'])
        pairs[0]['qualified']=False
        self.assertFalse(p.paired_analysis(pairs)['qualified'])
        with self.assertRaisesRegex(Refusal,'pseudoreplication'):p.paired_analysis([pairs[0]]*5)

    # Recorded incoming size defines bins, including zero/small calls and empty bins; boundaries cannot be grouped by route.
    def test_portion_bins_have_fixed_boundaries(self):
        self.assertEqual([p.bin_label(n) for n in (0,511,512,1999,2000,9999,10000)],
                         ['0-512','0-512','512-2000','512-2000','2000-10000','2000-10000','10000-inf'])
        self.assertEqual(set(p.portion_summary([])),{'0-512','512-2000','2000-10000','10000-inf'})




class AdmissionAdapterTests(unittest.TestCase):
    # CPU call-contract evidence only: the real kit owns priority/VM admission and all cooldown boundaries.
    def test_cooldown_delegates_to_real_kit_interfaces(self):
        from session_admission import KitAdmission
        from pathlib import Path
        kit=KitAdmission.__new__(KitAdmission);kit.output=Path('/fixture');kit.commit='a'*40;kit.hashes={};kit.key='identity';kit.state=Path('/fixture/state')
        kit.memory=mock.Mock();kit.memory.read.return_value=dict(child_pid=42,birth='birth')
        kit.priority=mock.Mock();process=mock.Mock();process.poll.return_value=None
        with mock.patch('session_admission.time.sleep') as sleep, mock.patch('session_admission.time.monotonic_ns',side_effect=[0,300_000_000_000,300_000_000_000]):
            receipt=kit.cooldown(process)
        self.assertEqual(sleep.call_count,60);self.assertTrue(all(call.args==(5,) for call in sleep.call_args_list))
        self.assertEqual(receipt['cooldown_seconds'],300)
        self.assertEqual(kit.client.boundary.call_args_list,[mock.call('end','startup'),mock.call('begin','cooldown'),mock.call('end','cooldown')])
        self.assertEqual(kit.priority.priority_check.call_count,2)
        for call in kit.priority.priority_check.call_args_list:
            self.assertEqual(call.kwargs['server_pids'],[42]);self.assertFalse(call.kwargs['allow_background'])

    # The wrapper gets longer than its own child's 15-second grace period to reap that child on abort.
    def test_abort_allows_owned_supervisor_to_reap_child(self):
        from session_admission import KitAdmission
        kit=KitAdmission.__new__(KitAdmission);process=mock.Mock();process.poll.return_value=None
        kit.abort(process)
        process.terminate.assert_called_once();process.wait.assert_called_once_with(timeout=35)
        process.kill.assert_not_called()

    # No Linux fallback can manufacture a Mac admission receipt, when the exact Mac source is unavailable.
    def test_cpu_cannot_claim_mac_admission(self):
        from session_admission import KitAdmission
        with mock.patch('session_admission.sys.platform','linux'):
            with self.assertRaisesRegex(Refusal,'no CPU fallback'):KitAdmission('/missing','a'*40,'/missing')


class AnalysisTests(unittest.TestCase):
    # Missing/failed attempts stay visible; a changed paired journal cannot be passed off as identical work.
    def test_series_analysis_counts_failures_and_rejects_different_work(self):
        import importlib.util
        from pathlib import Path
        path=Path(__file__).parent/'packets/r49/analyze_series.py'
        spec=importlib.util.spec_from_file_location('analysis',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        selected=p.select_set(frozen());plan=dict(attempts=p.schedule(selected,'main','home'),prereg=selected['prereg'],session_set_sha256='frozen',series_id='home')
        attempts=[dict(index=i,status='not-attempted') for i in range(20)]
        result=module.analyze(plan,attempts)
        self.assertEqual(len(result['flow']),20);self.assertFalse(result['comparisons']['B/A']['success'])
        with self.assertRaisesRegex(Refusal,'including failures'):module.analyze(plan,attempts[:-1])
        def trace(row):
            return dict(status='measured',prereg=plan['prereg'],session_set_sha256='frozen',series_id='home',
                session_digest=row['session_digest'],arms=[row['arm']],comparison=row['comparison'],
                sessions={row['arm']:dict(warm_system_seconds=100,summary=dict(node={}))},
                calls=[dict(turn=0,journal_hash='same')])
        for i in (0,1):attempts[i]['trace']=trace(plan['attempts'][i])
        self.assertEqual(module.analyze(plan,attempts)['comparisons']['B/A']['available_pairs'],1)
        attempts[1]['trace']['calls'][0]['journal_hash']='changed'
        with self.assertRaisesRegex(Refusal,'journal differs'):module.analyze(plan,attempts)


if __name__=='__main__':unittest.main()

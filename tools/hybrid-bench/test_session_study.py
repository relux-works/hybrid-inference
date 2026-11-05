"""CPU controls for study admission; synthetic receipts never qualify target hardware."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from bench_common import Refusal,sha256
from session_study import replace_private_json as private_json
import session_protocol as p
from store import prefix_hash
import session_study as s
from test_session_protocol import export,frozen


def packet(name):
    path=Path(__file__).parent/'packets/r49'/(name+'.py')
    spec=importlib.util.spec_from_file_location('r49_'+name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def tc():
    return dict(rows=[dict(rate_MB_s=a,rtt_ms=b,payload_MB_s=a*.9,qdisc=[dict(kind='netem'),dict(kind='tbf')]) for a in (15,20) for b in (50,100,150)])


def qualify(block='C',side='pre'):
    return dict(prereg={'commit':'c'*40},amendment={'commit':'d'*40},node_url='https://node:9443',ca_sha256='ca',method='tc',link_series='fast',link_label='USB',results=tc(),block=block,side=side,series_id='public',source_head='source')


def public_fixture(root):
    selected=p.select_set(frozen());work=selected['sets']['main']['sessions'][0]
    selected.update(sets=dict(public=dict(status='selected',count=1,sessions=[work])),reproducibility_only=True)
    rows=p.schedule(selected,'public','fast')
    identity=dict(prereg={'commit':'c'*40},amendment={'commit':'d'*40},source_head='source',device='m3',model_sha256='model',qualification_method='tc',link_label='USB')
    private_json(root/'set.json',selected)
    plan=dict(identity,session_set_sha256=sha256(root/'set.json'),session_set_path=str(root/'set.json'),group='public',link_series='fast',series_id='public',attempts=rows,
        checkpoints={work['session_digest']:{a:s.checkpoint_plan(work,a,30) for a in ('local','hybrid','hybrid-background')}})
    private_json(root/'plan.json',plan);attempts=[]
    for row in rows:
        index=row['index'];out=root/str(index);out.mkdir()
        private_json(out/'memory.json',dict(accepted=True));private_json(out/'priority.json',{})
        admission=dict(qualified=True,cooldown_seconds=300,raw_memory_evidence=str(out/'memory.json'),raw_priority_evidence=str(out/'priority.json'),
            completed_guard_evidence=dict(memory_sha256=sha256(out/'memory.json'),priority_sha256=sha256(out/'priority.json')))
        qualities=[]
        for turn in plan['checkpoints'][work['session_digest']][row['arm']]['turns']:
            qp=out/str(turn);qp.mkdir();tokens=work['base_ids'].copy()
            for step in work['turns'][:turn]:tokens+=step['input_ids']+step['output_ids']
            private_json(qp/'input.json',dict(tokens=tokens,model_sha256='model'));private_json(qp/'result.json',{})
            qualities.append(dict(turn=turn,gate=True,result=str(qp/'result.json'),result_sha256=sha256(qp/'result.json'),input_sha256=sha256(qp/'input.json')))
        journal=work['base_ids'].copy();calls=[dict(turn=0,journal_hash=prefix_hash(journal),input_ids_sha256=prefix_hash([]),recorded_output_sha256=prefix_hash([]))]
        for turn,step in enumerate(work['turns'],1):
            journal+=step['input_ids']+step['output_ids']
            calls.append(dict(turn=turn,journal_hash=prefix_hash(journal),input_ids_sha256=prefix_hash(step['input_ids']),recorded_output_sha256=prefix_hash(step['output_ids'])))
        trace=dict(identity,link_probe={'node_url':'https://node:9443'},transport={'ca_sha256':'ca'},status='measured',series_plan_sha256=sha256(root/'plan.json'),node_accounting_complete=True,
            admission=admission,checkpoint_plan=plan['checkpoints'][work['session_digest']][row['arm']],quality=qualities,
            session_digest=work['session_digest'],arms=[row['arm']],calls=calls,evidence=[])
        private_json(out/'trace.json',trace)
        attempts.append(dict(index=index,exit_code=0,status='measured',trace_path=str(out/'trace.json'),trace_sha256=sha256(out/'trace.json')))
    private_json(root/'ATTEMPTS.json',attempts);qualification={}
    for block in ('B','C'):
        for side in ('pre','post'):
            path=root/(block+'-'+side+'.json');private_json(path,qualify(block,side))
            qualification[block+'-'+side]=dict(path=path.name,sha256=sha256(path))
    private_json(root/'SERIES.json',dict(plan=dict(path='plan.json',sha256=sha256(root/'plan.json')),
        attempts=dict(path='ATTEMPTS.json',sha256=sha256(root/'ATTEMPTS.json')),qualification=qualification))
    return identity,plan,attempts


class StudyTests(unittest.TestCase):
    # The actual external event content and routing must match; local assertions of publication cannot admit selection.
    def test_published_freeze_requires_live_event_matching_inventory(self):
        receipt=s.freeze_receipt(frozen(),'a'*64,'study')
        event=dict(type='message',event_id='$event',sender=s.SENDER,room=s.ROOM,topic=s.TOPIC,to=[s.RECIPIENT],kind='notice',at='2026-11-05T00:00:00Z',body=json.dumps(receipt))
        self.assertEqual(s.verify_publication(receipt,'$event',[event])['event_id'],'$event')
        with self.assertRaisesRegex(Refusal,'missing'):s.verify_publication(receipt,'$event',[])
        for key,value in [('sender','other'),('to',[]),('topic','other')]:
            with self.subTest(key=key),self.assertRaisesRegex(Refusal,'authority'):s.verify_publication(receipt,'$event',[dict(event,**{key:value})])
        changed=copy.deepcopy(receipt);changed['inventory']['eligible_count']+=1
        with self.assertRaisesRegex(Refusal,'differs'):s.verify_publication(changed,'$event',[event])

    # Exclusions keep the diagnostic, and API text availability is declared without filtering the population.
    def test_exclusion_message_and_api_availability_are_retained(self):
        data=export();data['sessions'][0]['turns'][1]['request_start_utc']='bad-date'
        result=frozen(data)
        self.assertEqual(result['excluded'][0]['reason'],'ValueError')
        self.assertIn('bad-date',result['excluded'][0]['message'])
        self.assertFalse(result['eligible'][0]['has_request_messages'])

    # Frozen waits alone select the nearest eligible midpoint with earlier tie; no eligible wait gives final-only.
    def test_checkpoint_nearest_wait_tie_and_final_only(self):
        work=dict(turns=[dict(pause_seconds=0) for _ in range(10)])
        work['turns'][3]['pause_seconds']=30;work['turns'][7]['pause_seconds']=30
        self.assertEqual(s.checkpoint_plan(work,'hybrid',30)['turns'],[3,10])
        self.assertEqual(s.checkpoint_plan(work,'local',30)['turns'],[])
        self.assertEqual(s.checkpoint_plan(work,'hybrid',31)['turns'],[10])
        with self.assertRaisesRegex(Refusal,'at least 30'):s.checkpoint_plan(work,'hybrid',29)

    # A sleeper that wakes late must report its actual dispatch lateness.
    def test_wait_lateness_measured_after_sleep(self):
        value=p.finish_wait(8_000_000_000,mock.Mock(side_effect=[3_000_000_000,9_000_000_000]),lambda _:None)
        self.assertEqual(value['dispatch_lateness_seconds'],1)

    # n=1 and n=3 retain their descriptive estimates; they cannot acquire a main-series success claim.
    def test_descriptive_public_and_favourable_pairs_not_dropped(self):
        for n in (1,3):
            pairs=[dict(session_digest=str(i),local_seconds=100,hybrid_seconds=70+i,qualified=True) for i in range(n)]
            row=s.descriptive_pairs(pairs)
            self.assertEqual(row['pairs'],pairs);self.assertEqual(row['available_pairs'],n)
            self.assertAlmostEqual(row['median_fractional_reduction'],.30-(n//2)/100)
            self.assertFalse(row['success']);self.assertNotIn('bootstrap_95',row)

    # The claim still rejects sub-10% medians and any negative session even with a large positive median.
    def test_claim_threshold_and_one_negative_session(self):
        pairs=[dict(session_digest=str(i),local_seconds=100,hybrid_seconds=89.999,qualified=True) for i in range(5)]
        self.assertTrue(p.paired_analysis(pairs)['success'])
        for row in pairs:row['hybrid_seconds']=90.001
        self.assertFalse(p.paired_analysis(pairs)['success'])
        for row in pairs:row['hybrid_seconds']=70
        pairs[0]['hybrid_seconds']=101
        result=p.paired_analysis(pairs);self.assertFalse(result['success']);self.assertLess(result['bootstrap_95'][0],0)

    # Six actual-rate metadata rows are required; this fixture tests the validator, not kernel shaping.
    def test_tc_grid_missing_or_slow_path_refuses(self):
        value=qualify();self.assertTrue(s.verify_qualification(value,'tc','fast','USB'))
        value['results']['rows'][0]['payload_MB_s']=11
        with self.assertRaisesRegex(Refusal,'80%'):s.verify_qualification(value,'tc','fast','USB')
        value=qualify();value['results']['rows'].pop()
        with self.assertRaisesRegex(Refusal,'six-path'):s.verify_qualification(value,'tc','fast','USB')

    # Both adjacent T trials must beat 80%, with large payloads/raw counters and an independent server on the declared path.
    def test_real_path_both_sides_independence_and_fast_binding(self):
        value=dict(method='real-path',link_series='fast',link_label='USB',reference_implementation='curl-ranges/nginx-static',
            reference_url='https://node:9444/payload.bin',node_url='https://node:9443',interface='en7',
            trials=[dict(tool=t,payload_bytes=384*1024**2,seconds=seconds,raw={'fixture':True},physical_before='before',physical_after='after') for t,seconds in zip(('T','R','T'),(11,10,12))])
        self.assertTrue(s.verify_qualification(value,'real-path','fast','USB'))
        for index in (0,2):
            bad=copy.deepcopy(value);bad['trials'][index]['seconds']=13
            with self.subTest(index=index),self.assertRaisesRegex(Refusal,'80%'):s.verify_qualification(bad,'real-path','fast','USB')
        bad=copy.deepcopy(value);bad['reference_url']=bad['node_url']
        with self.assertRaisesRegex(Refusal,'independent'):s.verify_qualification(bad,'real-path','fast','USB')
        with self.assertRaisesRegex(Refusal,'path/method'):s.verify_qualification(value,'real-path','home','WiFi')
        value['trials'][0]['payload_bytes']=128*1024**2
        with self.assertRaisesRegex(Refusal,'too small'):s.verify_qualification(value,'real-path','fast','USB')

    # Raw hashes, four arms, both blocks, native-check invocation and post-block evidence are mandatory. Native logits are mocked here.
    def test_public_full_receipt_positive_missing_arm_and_failed_quality_negative(self):
        with tempfile.TemporaryDirectory() as directory,mock.patch('session_measurement.validate_quality',return_value=True) as quality:
            root=Path(directory);identity,plan,attempts=public_fixture(root)
            receipt=s.verify_public_rehearsal(root/'SERIES.json',identity)
            self.assertEqual(receipt['capture_budget_seconds'],30);self.assertEqual(quality.call_count,4)
            bad=dict(identity,device='m5')
            with self.assertRaisesRegex(Refusal,'identity differs'):s.verify_public_rehearsal(root/'SERIES.json',bad)
            attempts[1]['status']='failed';private_json(root/'ATTEMPTS.json',attempts)
            saved=json.loads((root/'SERIES.json').read_text());saved['attempts']['sha256']=sha256(root/'ATTEMPTS.json');private_json(root/'SERIES.json',saved)
            with self.assertRaisesRegex(Refusal,'arm failed'):s.verify_public_rehearsal(root/'SERIES.json',identity)
            attempts[1]['status']='measured';private_json(root/'ATTEMPTS.json',attempts)
            saved['attempts']['sha256']=sha256(root/'ATTEMPTS.json');private_json(root/'SERIES.json',saved)
            quality.return_value=False
            with self.assertRaisesRegex(Refusal,'native quality gate failed'):s.verify_public_rehearsal(root/'SERIES.json',identity)

    # Post-block failure invalidates analysis admission even after every attempt completed; no unqualified fallback is accepted.
    def test_public_missing_post_qualification_refuses(self):
        with tempfile.TemporaryDirectory() as directory,mock.patch('session_measurement.validate_quality',return_value=True):
            root=Path(directory);identity,_,_=public_fixture(root)
            self.assertTrue(s.verify_series_transport(root/'SERIES.json',root/'plan.json'))
            value=json.loads((root/'C-post.json').read_text());value['results']['rows'][0]['payload_MB_s']=1;private_json(root/'C-post.json',value)
            receipt=json.loads((root/'SERIES.json').read_text());receipt['qualification']['C-post']['sha256']=sha256(root/'C-post.json');private_json(root/'SERIES.json',receipt)
            with self.assertRaisesRegex(Refusal,'80%'):s.verify_public_rehearsal(root/'SERIES.json',identity)

    # A direct arm invocation cannot bypass the public rehearsal or alter wait-selected checkpoint placement.
    def test_direct_main_requires_public_proof_and_frozen_checkpoint_plan(self):
        selected=p.select_set(frozen());work=selected['sets']['main']['sessions'][0]
        args=SimpleNamespace(session_set_sha256='set',series_id='main',group='main',device='m3',link_label='USB',ordinal=0,arm='local',session_digest=work['session_digest'],comparison='C/A')
        plan=dict(source_head='source',session_set_sha256='set',series_id='main',group='main',device='m3',link_label='USB',model_sha256='model',attempts=[dict(index=0,arm='local',session_digest=work['session_digest'],comparison='C/A')])
        with self.assertRaisesRegex(Refusal,'verified public rehearsal'):s.validate_run_plan(plan,selected,args,'source','model')


class FullSeriesAdapterTests(unittest.TestCase):
    # Exercise every scheduler iteration, receipt update, pre/post gate and final receipt. Hardware/driver calls are isolated fakes.
    def test_four_arm_scheduler_loop_and_no_reroll_on_post_failure(self):
        module=packet('session_series')
        for fail_post in (False,True):
            with self.subTest(fail_post=fail_post),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);identity,plan,fixture_attempts=public_fixture(root)
                original=[json.loads(Path(r['trace_path']).read_text()) for r in fixture_attempts]
                (root/'ATTEMPTS.json').unlink();(root/'SERIES.json').unlink()
                config={key:'/fixture/'+key for key in ('client-state','ca-file','fixture','policy','import-fit','kit-root','kit-ref')}
                config.update(device='m3',**{'link-label':'USB','qualification':{'tc-results':'/fixture/tc.json'}})
                private_json(root/'config.json',config);commands=[]
                def run(command,**kwargs):
                    name=Path(command[1]).stem;commands.append(name)
                    def value(flag):return command[command.index(flag)+1]
                    output=Path(value('--output'))
                    if name=='qualify_transport':
                        side=value('--side');block=value('--block')
                        if fail_post and side=='post':return SimpleNamespace(returncode=3)
                        private_json(output,qualify(block,side))
                    elif name=='session_measurement':
                        index=int(value('--ordinal'));output.mkdir()
                        trace=original[index];trace['series_plan_sha256']=sha256(root/'plan.json')
                        private_json(output/'trace.json',trace)
                    return SimpleNamespace(returncode=0)
                with mock.patch.object(module.sys,'platform','darwin'),mock.patch.object(module.subprocess,'run',side_effect=run):
                    if fail_post:
                        with self.assertRaisesRegex(RuntimeError,'no reroll'):module.execute(plan,root/'set.json',root/'config.json',root)
                        attempts=json.loads((root/'ATTEMPTS.json').read_text())
                        self.assertEqual([r['status'] for r in attempts],['measured','measured','not-attempted-after-qualification-failure','not-attempted-after-qualification-failure'])
                        self.assertFalse((root/'SERIES.json').exists())
                    else:
                        module.execute(plan,root/'set.json',root/'config.json',root)
                        self.assertTrue(s.verify_series_transport(root/'SERIES.json',root/'plan.json'))
                        self.assertEqual(len(json.loads((root/'ATTEMPTS.json').read_text())),4)
                        self.assertEqual(commands.count('session_measurement'),4)
                        with mock.patch('session_measurement.validate_quality',return_value=True):
                            s.verify_public_rehearsal(root/'SERIES.json',identity)

    # Declared budget and actual wait are separate bounds: passing either alone is insufficient.
    def test_capture_budget_and_actual_wait_both_bound_save(self):
        s.validate_capture(0,9_000_000_000,60_000_000_000,30)
        with self.assertRaisesRegex(Refusal,'capture budget'):s.validate_capture(0,31_000_000_000,60_000_000_000,30)
        with self.assertRaisesRegex(Refusal,'timing contaminated'):s.validate_capture(0,9_000_000_000,8_000_000_000,30)

    # Mutable progress updates must replace the old file securely; immutable source artifacts still use exclusive create.
    def test_progress_and_deferred_checkpoint_hash_can_update_private_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'receipt.json'
            s.replace_private_json(path,{'hash':None});s.replace_private_json(path,{'hash':'final'})
            self.assertEqual(json.loads(path.read_text()),{'hash':'final'})
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])

class EntryPointTests(unittest.TestCase):
    # Exercise select's real CLI body: a missing publication cannot write a session set; the valid event is actually read.
    def test_select_entrypoint_cannot_skip_publication(self):
        module=packet('prepare_session_set')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);population=frozen();population['amendment']={'commit':'amendment'}
            source=root/'population.json';private_json(source,population);output=root/'selected.json'
            argv=['prepare_session_set','select','--input',str(source),'--sha256',sha256(source),
                '--prereg-commit','prereg','--amendment-commit','amendment','--study-id','study','--output',str(output)]
            with mock.patch.object(module,'prereg_binding',return_value=population['prereg']),mock.patch.object(module,'amendment_binding',return_value=population['amendment']),mock.patch.object(module.sys,'argv',argv):
                with self.assertRaisesRegex(Refusal,'published freeze event required'):module.main()
            self.assertFalse(output.exists())
            receipt=s.freeze_receipt(population,sha256(source),'study')
            event=dict(type='message',event_id='$event',sender=s.SENDER,room=s.ROOM,topic=s.TOPIC,to=[s.RECIPIENT],kind='notice',at='2026-11-05T00:00:00Z',body=json.dumps(receipt))
            with mock.patch.object(module,'prereg_binding',return_value=population['prereg']),mock.patch.object(module,'amendment_binding',return_value=population['amendment']),mock.patch.object(module.sys,'argv',argv+['--published-event','$event']),mock.patch.object(s,'read_events',return_value=[event]) as read,mock.patch('builtins.print'):
                module.main()
            read.assert_called_once_with('$event');self.assertEqual(json.loads(output.read_text())['freeze_publication']['event_id'],'$event')

    # The series analysis itself, not just the summary helper, must emit public and favourable descriptive pairs.
    def test_analyze_entrypoint_preserves_n1_n3_estimates(self):
        module=packet('analyze_series')
        for count,group in ((1,'public'),(3,'favourable')):
            rows=[dict(index=2*i+j,session_digest=str(i),arm=arm,comparison='C/A') for i in range(count) for j,arm in enumerate(('local','hybrid-background'))]
            plan=dict(attempts=rows,prereg={},session_set_sha256='set',series_id='series',group=group)
            attempts=[]
            for row in rows:
                arm=row['arm'];trace=dict(status='measured',prereg={},session_set_sha256='set',series_id='series',session_digest=row['session_digest'],arms=[arm],comparison='C/A',
                    sessions={arm:dict(warm_system_seconds=100 if arm=='local' else 75,summary={'node':{}})},calls=[dict(turn=0,journal_hash='same')])
                attempts.append(dict(index=row['index'],status='measured',trace=trace))
            result=module.analyze(plan,attempts)['comparisons']['C/A']
            self.assertEqual(len(result['pairs']),count);self.assertEqual(result['median_fractional_reduction'],.25);self.assertFalse(result['success'])

class ReferenceAdapterTests(unittest.TestCase):
    # Real curl argv construction uses file-based auth and verified TLS; malformed/truncated and failed-auth responses refuse without printing stderr.
    def test_independent_curl_ranges_and_auth_negative(self):
        module=packet('qualify_transport');calls=[]
        with tempfile.TemporaryDirectory() as directory:
            header=Path(directory)/'auth.hdr';header.write_text('Authorization: Bearer fixture-only\n');header.chmod(0o600)
            def run(command,**kwargs):
                calls.append(command);return SimpleNamespace(returncode=0,stdout=f'206 {384*1024**2//8} 1.0',stderr='')
            with mock.patch.object(module.subprocess,'run',side_effect=run):
                result=module.reference('https://node:9444/payload.bin','ca.pem',header,'en7')
            self.assertEqual(sum(r['bytes'] for r in result),384*1024**2);self.assertEqual(len(calls),8)
            self.assertTrue(all('@'+str(header) in command and 'fixture-only' not in str(command) and '--cacert' in command for command in calls))
            with mock.patch.object(module.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='206 1 1.0',stderr='')):
                with self.assertRaisesRegex(Refusal,'truncated range'):module.reference('https://node:9444/payload.bin','ca.pem',header,'en7')
            with mock.patch.object(module.subprocess,'run',return_value=SimpleNamespace(returncode=22,stdout='',stderr='fixture-secret-must-not-print')):
                with self.assertRaisesRegex(Refusal,'credentials are not logged') as caught:module.reference('https://node:9444/payload.bin','ca.pem',header,'en7')
            self.assertNotIn('fixture-secret',str(caught.exception))

    # Every actual-path trial is at least384MiB and completed trials are persisted before moving to the next independent tool.
    def test_real_trials_interleave_and_preserve_progress(self):
        module=packet('qualify_transport');progress=[]
        probe=dict(samples=[dict(bytes=128*1024**2)]*3)
        with mock.patch.object(module,'physical',return_value='physical fixture'),mock.patch.object(module,'probe',return_value=probe) as transport,mock.patch.object(module,'reference',return_value=[dict(bytes=48*1024**2)]*8) as reference,mock.patch.object(module.time,'monotonic_ns',side_effect=[0,10**9,2*10**9,3*10**9,4*10**9,5*10**9]):
            rows=module.real_trials('https://node:9443','fixture','ca','https://node:9444','header','en7',lambda rows:progress.append(copy.deepcopy(rows)))
        self.assertEqual([r['tool'] for r in rows],['T','R','T']);self.assertEqual([len(r) for r in progress],[1,2,3])
        self.assertEqual([r['payload_bytes'] for r in rows],[384*1024**2]*3)
        self.assertEqual(transport.call_count,2);reference.assert_called_once()

class UnchangedContractTests(unittest.TestCase):
    # The Python boundary must send recorded replay IDs and refuse a different decoder answer before committing it. Native decode is mocked.
    def test_recorded_output_mismatch_never_commits(self):
        import session_measurement as measurement
        for generated in ([3],[4]):
            app=mock.Mock();app.arm='hybrid';app.state_dtype='bf16'
            session={'_bench':dict(fallback_reason=None,path='offload',lossy_import=True)}
            app.store.get.return_value=session
            app.decode.request.side_effect=[dict(n=0),dict(n=2,tokens=[1,2],prefix_hash=prefix_hash([1,2]))]
            requests=[]
            def prepare(profile,body,session,prompt):requests.append(body);return session,None,body
            app.prepare_session.side_effect=prepare
            with mock.patch.object(measurement,'native_stream',return_value=(generated,{'predicted_ms':1},1,1)):
                if generated==[3]:
                    measurement.execute_turn(app,{},'sid',[1,2],1,1,1,recorded_output=[3]);app.store.commit_evaluated.assert_called_once()
                else:
                    with self.assertRaisesRegex(Refusal,'differs from recorded replay'):measurement.execute_turn(app,{},'sid',[1,2],1,1,1,recorded_output=[3])
                    app.store.commit_evaluated.assert_not_called();self.assertFalse(session['_commit'])
            self.assertEqual(requests[0]['replay_mode'],'recorded-token-replay-v1');self.assertEqual(requests[0]['replay_tokens'],[3]);app.end_turn.assert_called_once_with('sid')

    # Public/API replay keeps both the full base text and the recorded next portion/answer; no provider output becomes future context.
    def test_public_api_format_keeps_base_and_recorded_messages(self):
        import hashlib
        module=packet('build_public_session');text='public source line\n'*400
        source=dict(schema='hybrid-public-repository-source-v1',commit='pin',url='https://example.invalid/repo',files=[dict(path='source.txt',text=text,sha256=hashlib.sha256(text.encode()).hexdigest())])
        def tokenize(text):return [2] if text=='Reviewed the recorded source block.\n' else [1]*100000
        result=module.build(source,tokenize,'0'*64,{})
        self.assertEqual(len(result['base_text_fragments']),1)
        turn=result['sets']['public']['sessions'][0]['turns'][0]
        self.assertIn('lines 1-200',result['base_text_fragments'][0]);self.assertIn('lines 201-400',turn['input_text'])
        self.assertEqual(turn['output_text'],'Reviewed the recorded source block.\n');self.assertEqual(turn['output_ids'],[2])


class StandaloneEntryTests(unittest.TestCase):
    # A fresh interpreter must import every runbook command without an inherited proxy path; help performs no inference or network work.
    def test_runbook_entrypoints_import_in_clean_process(self):
        import os,subprocess,sys
        for name in ('prepare_session_set','publish_freeze','qualify_transport','stage_reference','session_series','analyze_series','build_public_session'):
            path=Path(__file__).resolve().parent/'packets/r49'/(name+'.py')
            run=subprocess.run([sys.executable,str(path),'--help'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=''),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            self.assertEqual(run.returncode,0,name+': '+run.stderr)
            self.assertIn('usage:',run.stdout)


class PublicationRecoveryTests(unittest.TestCase):
    # A failed send can reuse only the identical pending body, never replace its inventory before the external event exists.
    def test_pending_freeze_body_retry_keeps_identical_inventory(self):
        module=packet('publish_freeze')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);frozen=dict(prereg={'commit':'p'},amendment={'commit':'a'},eligible=[],eligible_count=0,excluded=[])
            population=root/'population.json';private_json(population,frozen);output=root/'publication.json'
            receipt=s.freeze_receipt(frozen,sha256(population),'study');private_json(output.with_suffix('.body.json'),receipt)
            argv=['publish_freeze','--population',str(population),'--sha256',sha256(population),'--prereg-commit','p','--amendment-commit','a','--study-id','study','--output',str(output)]
            with mock.patch.object(module,'prereg_binding',return_value=frozen['prereg']),mock.patch.object(module,'amendment_binding',return_value=frozen['amendment']),mock.patch.object(module.sys,'argv',argv),mock.patch.object(module,'read_events',return_value=[]),mock.patch.object(module.subprocess,'check_output',return_value='$event\n') as post,mock.patch.object(module,'verify_publication',return_value={'event_id':'$event'}),mock.patch('builtins.print'):
                module.main();post.assert_called_once()
                output.unlink();private_json(output.with_suffix('.body.json'),{'different':'inventory'})
                with self.assertRaisesRegex(Refusal,'no inventory reroll'):module.main()
            self.assertFalse(output.exists())

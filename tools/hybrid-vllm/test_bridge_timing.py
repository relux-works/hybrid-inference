"""CPU controls for timing arithmetic, aligned deltas and strict refusal."""
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import bridge_timing as timing
from node_backend import VllmBackend


class TimingTests(unittest.TestCase):
    # Claim: unaligned D and both delta sizes partition into exact aligned
    # endpoints with a local tail; too-small chunks cannot loop forever.
    def test_depth_delta_chunks(self):
        for depth in timing.DEPTHS:
            for delta in timing.DELTAS:
                end = (depth + delta - 1) // 784 * 784
                rows = timing.chunks(depth, end, 784)
                self.assertEqual(rows[0][0], depth)
                self.assertEqual(rows[-1][1], end)
                self.assertTrue(all(b > a and b % 784 == 0 and b - a <= 7840 for a,b in rows))
                self.assertTrue(all(a[1] == b[0] for a,b in zip(rows,rows[1:])))
                self.assertGreater(depth+delta-end,0)
        with self.assertRaisesRegex(timing.require.__globals__['Refusal'], 'invalid transfer'):
            timing.chunks(10,20,16,maximum=1)

    # Claim: phase arithmetic refuses missing/nonfinite/negative values and
    # backend disagreement beside valid zero and positive phase controls.
    def test_phase_headers(self):
        headers = {'X-Hybrid-Evaluate-Ms':'6', 'X-Hybrid-Export-Ms':'3',
                   'X-Hybrid-Backend-Ms':'9', 'X-Hybrid-Conversion-Ms':'2'}
        self.assertEqual(timing.phase_headers(headers)['server_prefill_ms'],6)
        for name in headers:
            for bad in ('nan','inf','-1'):
                with self.subTest(name=name,bad=bad), self.assertRaises(Exception):
                    timing.phase_headers(dict(headers,**{name:bad}))
            with self.assertRaises(Exception):
                timing.phase_headers({k:v for k,v in headers.items() if k!=name})
        with self.assertRaisesRegex(Exception,'accounting'):
            timing.phase_headers(dict(headers,**{'X-Hybrid-Backend-Ms':'15'}))

    # Claim: the production backend keeps finished.json compatibility, subtracts
    # only measured export from wall time and refuses an impossible export duration.
    def test_backend_export_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            backend=VllmBackend('http://127.0.0.1:1',root/'control',{},root)
            backend.block_size=16
            (root/'finished.json').write_text(json.dumps(dict(n=32,block_size=16)))
            (root/'capture-00000032').mkdir()
            for export in (.25, 0, 2, float('nan')):
                (root/'capture-00000032/timing.json').write_text(json.dumps(dict(export_seconds=export)))
                with mock.patch.object(backend,'request'), mock.patch('node_backend.time.perf_counter',side_effect=[10,11]):
                    if not math.isfinite(export) or export>1:
                        with self.assertRaisesRegex(ValueError,'invalid export'):
                            backend.prefill(list(range(32)),0,root)
                    else:
                        backend.prefill(list(range(32)),0,root)
                        self.assertEqual(backend.last_timing['export_ms'],export*1000)
                        self.assertEqual(backend.last_timing['evaluate_ms'],(1-export)*1000)

    # Claim: the actual adapter emits separate backend/evaluate/export and
    # conversion headers without changing the existing total-prefill header.
    # Synthetic conversion exercises instrumentation, not CUDA arithmetic.
    def test_adapter_phase_headers(self):
        from types import SimpleNamespace
        import node_adapter
        from bridge import token_hash
        app = node_adapter.Adapter.__new__(node_adapter.Adapter)
        app.target = dict(model_layout=dict(vocab_size=100))
        app.context, app.block, app.codec, app.extends = 64, 16, 'reference', 0
        sid = 'a' * 32
        app.sessions = {sid: ('owner', node_adapter.Session([], 0))}
        app.backend = SimpleNamespace(last_timing=dict(backend_ms=9, export_ms=3, evaluate_ms=6),
                                     prefill=lambda *a,**k: Path('unused'))
        def convert(capture, target, delta, *args):
            delta.write_bytes(b'wire')
            return dict(n0=0,n1=16,prefix_hash_n1=token_hash(list(range(16))))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(node_adapter,'convert',side_effect=convert):
            app.scratch=Path(tmp)
            status, wire, headers=app.apply('POST','/v1/sessions/'+sid+'/extend',
                dict(n0=0,prefix_hash_n0=token_hash([]),tokens=list(range(16)),return_delta=True),'owner')
        self.assertEqual((status,wire),(200,b'wire'))
        phases=timing.phase_headers(headers)
        self.assertEqual(phases['server_prefill_ms'],6)
        self.assertEqual(phases['export_ms'],3)
        self.assertGreaterEqual(phases['conversion_ms'],0)
        self.assertIn('X-Hybrid-Prefill-Ms',headers)
        self.assertEqual(app.sessions[sid][1].tokens,list(range(16)))

    # Claim: a complete append imports the aligned prefix before requesting
    # the local tail, records all five phases, and deletes its session on an
    # endpoint mismatch without generating a token. The transport is doubled.
    def test_append_and_mismatch_cleanup(self):
        from types import SimpleNamespace
        from store import prefix_hash
        for mismatch in (False, True):
            state = dict(tokens=[], deleted=False, decode_called=False)
            remote = SimpleNamespace(last_transfer_seconds=.001, phase=dict(server_prefill_ms=2,export_ms=1,conversion_ms=1,backend_ms=3))
            def node_request(method,path,data=None,**kwargs):
                if path=='/v1/sessions': return dict(session_id='a'*32)
                if method=='DELETE': state['deleted']=True; return None
                if data['return_delta']: return b'fixture delta'
                return {}
            remote.request=node_request
            def decoder_request(method,path,data=None,*args):
                if path=='/completion': state['tokens']=data['prompt']; return {}
                if path.endswith('import_delta'):
                    state['tokens']=[100]*48
                    return dict(n=48)
                if path.endswith('/info'):
                    tokens=state['tokens']
                    if mismatch and len(tokens)==48: tokens=tokens[:-1]
                    return dict(n=len(tokens),tokens=tokens,prefix_hash=prefix_hash(tokens))
                return {}
            decoder=SimpleNamespace(request=decoder_request)
            def first(*args):
                state['decode_called']=True
                return timing.time.perf_counter()+1
            with mock.patch.object(timing.Proxy,'client_delta',return_value=b'fixture delta'), \
                    mock.patch.object(timing,'first_token',side_effect=first):
                if mismatch:
                    with self.assertRaisesRegex(Exception,'endpoint/journal'):
                        timing.one(remote,decoder,object(),dict(export_block_size=16,model_sha256='x'),32,17,0)
                    self.assertFalse(state['decode_called'])
                else:
                    row=timing.one(remote,decoder,object(),dict(export_block_size=16,model_sha256='x'),32,17,0)
                    self.assertEqual((row['exported_endpoint'],row['local_tail_tokens']),(48,1))
                    self.assertEqual(row['server_prefill_ms'],2)
                    self.assertEqual(row['transfer_ms'],1)
                    self.assertGreater(row['ttft_ms'],1000)
                self.assertTrue(state['deleted'])

    # Claim: native SSE needs both an actual first token and a terminal marker;
    # a stream ending after content alone cannot qualify the bridge TTFT.
    def test_native_sse(self):
        from types import SimpleNamespace
        for raw, accepted in ((b'data: {"tokens":[42]}\n\ndata: {"stop":true}\n\n',True),
                              (b'data: {"tokens":[42]}\n\n',False),
                              (b'data: {"stop":true}\n\n',False)):
            decode=SimpleNamespace(open=lambda *a,**k: io.BytesIO(raw))
            if accepted:
                self.assertGreater(timing.first_token(decode,[1,2]),0)
            else:
                with self.assertRaisesRegex(Exception,'terminal'):
                    timing.first_token(decode,[1,2])


if __name__=='__main__': unittest.main()

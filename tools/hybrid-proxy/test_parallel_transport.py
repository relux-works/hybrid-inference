"""Real localhost TLS/range controls; no model, network download or shaping claim."""
from contextlib import contextmanager
import hashlib
import http.client
import http.server
import json
from pathlib import Path
import ssl
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error

from parallel_transport import Gateway, GatewayHandler, ParallelTransport, PayloadCache, MAX_PAYLOAD
from proxy import Node, HTTPError


TOKEN = 'transport-test-token-' + '0'*32
OTHER = 'transport-test-token-' + '1'*32
BODY = hashlib.shake_256(b'r49 opaque native envelope identity fixture').digest(4*1024**2+37)


class Upstream(http.server.BaseHTTPRequestHandler):
    calls = 0
    def log_message(self,*_):
        pass
    def do_POST(self):
        type(self).calls += 1
        self.rfile.read(int(self.headers.get('Content-Length','0')))
        self.send_response(200)
        self.send_header('Content-Type','application/octet-stream')
        self.send_header('Content-Length',str(len(BODY)))
        self.send_header('X-Hybrid-Prefill-Ms','12.5')
        self.end_headers();self.wfile.write(BODY)
    def do_GET(self):
        body=json.dumps(dict(delta_version=2,state_dtypes=['f32','bf16'])).encode()
        self.send_response(200);self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)


@contextmanager
def server_thread(server):
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield server
    finally:
        server.shutdown();server.server_close();thread.join()


class ParallelTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.tmp.name)
        cls.cert,cls.key=cls.root/'cert.pem',cls.root/'key.pem'
        with (cls.root/'openssl.log').open('wb') as log:
            subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1',
                '-subj','/CN=localhost','-addext','subjectAltName=IP:127.0.0.1,DNS:localhost',
                '-keyout',str(cls.key),'-out',str(cls.cert)],stdout=log,stderr=log,check=True)
        cls.upstream=http.server.ThreadingHTTPServer(('127.0.0.1',0),Upstream)
        cls.upstream_context=server_thread(cls.upstream);cls.upstream_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.upstream_context.__exit__(None,None,None);cls.tmp.cleanup()

    def setUp(self):
        self.gateway=Gateway(('127.0.0.1',0),'http://127.0.0.1:'+str(self.upstream.server_port),
                             [TOKEN,OTHER],self.cert,self.key,probe_bytes=1024**2)
        self.context=server_thread(self.gateway);self.context.__enter__()
        self.url='https://127.0.0.1:'+str(self.gateway.server_port)
        self.client=ParallelTransport(self.url,self.cert,8)

    def tearDown(self):
        self.context.__exit__(None,None,None)

    def manifest(self):
        with self.client.open('POST','/v1/bench/parallel-link',data=b'{}',headers={'Authorization':'Bearer '+TOKEN}) as response:
            return json.loads(response.read())

    # Parallel transport must preserve opaque f32 bytes; real native import is a separate integration control.
    def test_f32_payload_identity_and_encrypted_byte_accounting(self):
        node=Node(self.url,TOKEN,10,parallel=dict(ca_file=str(self.cert),streams=8))
        result=node.request('POST','/v1/sessions/'+'a'*32+'/extend',dict(tokens=[1],n0=0),binary=True)
        self.assertEqual(result,BODY)
        self.assertEqual(float(node.last_prefill_ms),12.5)
        self.assertEqual(node.last_bytes,len(BODY))
        self.assertGreater(node.last_tls_bytes['tls_received_bytes'],len(BODY))
        self.assertGreater(node.last_tls_bytes['tls_sent_bytes'],0)
        self.assertGreaterEqual(node.parallel.stats()['connections'],10)
        self.assertFalse(self.gateway.cache.objects)

    # A wrong token is rejected before upstream work or cache allocation, alongside a valid control.
    def test_wrong_token_refuses_before_dispatch(self):
        before=Upstream.calls
        node=Node(self.url,OTHER+'bad',10,parallel=dict(ca_file=str(self.cert),streams=8))
        with self.assertRaises(HTTPError) as rejected:
            node.request('POST','/v1/sessions/'+'a'*32+'/extend',{},binary=True)
        self.assertEqual(rejected.exception.status,401)
        self.assertEqual(Upstream.calls,before)
        self.assertFalse(self.gateway.cache.objects)
        self.assertEqual(Node(self.url,TOKEN,10,parallel=dict(ca_file=str(self.cert))).request('GET','/v1/health')['delta_version'],2)

    # A peer withholding its TLS handshake cannot block the accept loop or a valid parallel client.
    def test_stalled_handshake_does_not_serialize_connections(self):
        with socket.create_connection(('127.0.0.1',self.gateway.server_port),timeout=2):
            client=ParallelTransport(self.url,self.cert,8,timeout=2)
            with client.open('GET','/v1/health',headers={'Authorization':'Bearer '+TOKEN}) as response:
                self.assertEqual(json.loads(response.read())['delta_version'],2)

    # Certificate verification cannot be disabled by the transport or satisfied by a valid bearer alone.
    def test_untrusted_certificate_refuses(self):
        untrusted=ParallelTransport(self.url,None)
        with self.assertRaises(ssl.SSLCertVerificationError):
            untrusted.open('GET','/v1/health',headers={'Authorization':'Bearer '+TOKEN})

    # A short range must fail as an incomplete response, never return a successful partial envelope.
    def test_truncated_range_refuses_and_releases_cache(self):
        manifest=self.manifest()
        original=GatewayHandler.reply
        def truncated(handler,status,body=b'',headers=None):
            if status!=206:return original(handler,status,body,headers)
            handler.send_response(status);handler.send_header('Content-Length',str(len(body)))
            for k,v in headers.items():handler.send_header(k,v)
            handler.end_headers();handler.wfile.write(body[:-1]);handler.close_connection=True
        with mock.patch.object(GatewayHandler,'reply',truncated):
            with self.assertRaisesRegex(ValueError,'^truncated payload range$'):self.client.fetch(manifest,TOKEN)
        self.assertFalse(self.gateway.cache.objects)

    # A complete, correctly ranged but corrupted envelope must fail its digest before import.
    def test_hash_mismatch_refuses_and_releases_cache(self):
        manifest=self.manifest();manifest['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'^payload hash mismatch$'):self.client.fetch(manifest,TOKEN)
        self.assertFalse(self.gateway.cache.objects)

    # Another valid client cannot read or delete a different client's immutable payload.
    def test_payload_owner_and_range_bounds(self):
        manifest=self.manifest();path='/payloads/'+manifest['id']
        for method in ('GET','DELETE'):
            with self.assertRaises(urllib.error.HTTPError) as rejected:
                self.client.open(method,path,headers={'Authorization':'Bearer '+OTHER,'Range':'bytes=0-1'})
            self.assertEqual(rejected.exception.code,404);rejected.exception.close()
        with self.assertRaises(urllib.error.HTTPError) as rejected:
            self.client.open('GET',path,headers={'Authorization':'Bearer '+TOKEN,'Range':'bytes=0-999999999'})
        self.assertEqual(rejected.exception.code,416);rejected.exception.close()
        self.assertEqual(self.client.fetch(manifest,TOKEN),b'x'*1024**2)

    # Manifest validation must reject traversal, excessive allocation and non-HTTPS origins before I/O.
    def test_manifest_and_origin_refuse_before_network(self):
        good=dict(schema='hybrid-ranges-v1',id='a'*32,bytes=1,sha256='0'*64)
        for change in (dict(id='../secret'),dict(bytes=MAX_PAYLOAD+1),dict(bytes=True),dict(sha256='bad')):
            with self.assertRaisesRegex(ValueError,'invalid immutable envelope manifest'):
                self.client.fetch(dict(good,**change),TOKEN)
        self.assertEqual(self.client.stats()['connections'],0)
        for origin in ('http://localhost','https://user:secret@localhost','https://localhost/path'):
            with self.assertRaisesRegex(ValueError,'HTTPS origin'):ParallelTransport(origin,self.cert)

    # Leases and reservations bound retained memory; a full cache cannot accept another maximum envelope.
    def test_cache_budget_and_expiry(self):
        cache=PayloadCache(limit=MAX_PAYLOAD,ttl=1)
        cache.reserve()
        with self.assertRaisesRegex(ValueError,'cache full'):cache.reserve()
        cache.reserved=0
        with mock.patch('parallel_transport.time.monotonic',return_value=10):manifest=cache.publish(TOKEN,b'abc')
        with mock.patch('parallel_transport.time.monotonic',return_value=10.5):self.assertEqual(cache.get(manifest['id'],TOKEN),b'abc')
        with mock.patch('parallel_transport.time.monotonic',return_value=11.1):
            with self.assertRaises(KeyError):cache.get(manifest['id'],TOKEN)


if __name__=='__main__':unittest.main()

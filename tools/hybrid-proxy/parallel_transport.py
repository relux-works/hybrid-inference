"""Authenticated HTTPS ranges for immutable native envelopes; standard library only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
import http.client
import http.server
import io
import json
import re
import secrets
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_PAYLOAD = 513 * 1024**2 + 52
MAX_CONTROL = 8 * 1024**2


class TLSStream(io.RawIOBase):
    """Count encrypted TCP payload with SSL's BIO interface, without TLS record parsing."""
    def __init__(self, host, port, context, timeout, record):
        super().__init__()
        self.socket = socket.create_connection((host, port), timeout)
        self.incoming, self.outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
        self.tls = context.wrap_bio(self.incoming, self.outgoing, server_hostname=host)
        self.counts = dict(tls_sent_bytes=0, tls_received_bytes=0)
        self.record = record
        try:
            self.drive(self.tls.do_handshake)
        except BaseException:
            self.close()
            raise

    def flush_tls(self):
        while data := self.outgoing.read():
            view = memoryview(data)
            while view:
                n = self.socket.send(view)
                if not n:
                    raise OSError('TLS socket closed during write')
                self.counts['tls_sent_bytes'] += n
                view = view[n:]

    def drive(self, operation):
        while True:
            try:
                result = operation()
                self.flush_tls()
                return result
            except ssl.SSLWantReadError:
                self.flush_tls()
                data = self.socket.recv(65536)
                if not data:
                    self.incoming.write_eof()
                else:
                    self.counts['tls_received_bytes'] += len(data)
                    self.incoming.write(data)
            except ssl.SSLWantWriteError:
                self.flush_tls()

    def sendall(self, data):
        view = memoryview(data)
        while view:
            n = self.drive(lambda: self.tls.write(view))
            view = view[n:]

    def readable(self):
        return True

    def readinto(self, buffer):
        try:
            data = self.drive(lambda: self.tls.read(len(buffer)))
        except ssl.SSLZeroReturnError:
            return 0
        buffer[:len(data)] = data
        return len(data)

    def makefile(self, *_):
        return io.BufferedReader(self)

    def close(self):
        if not self.closed:
            self.socket.close()
            self.record(self.counts)
            super().close()


class ParallelTransport:
    def __init__(self, url, ca_file, streams=8, timeout=300):
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
            raise ValueError('parallel transport requires an HTTPS origin without credentials or path')
        if type(streams) is not int or not 1 <= streams <= 16:
            raise ValueError('parallel streams must be 1..16')
        self.url, self.host, self.port = url.rstrip('/'), parsed.hostname, parsed.port or 443
        self.streams, self.timeout = streams, timeout
        self.context = ssl.create_default_context(cafile=ca_file)
        self.context.minimum_version = ssl.TLSVersion.TLSv1_2
        self.lock = threading.Lock()
        self.counts = dict(tls_sent_bytes=0, tls_received_bytes=0, connections=0)

    def record(self, counts):
        with self.lock:
            self.counts['connections'] += 1
            for key, value in counts.items():
                self.counts[key] += value

    def stats(self):
        with self.lock:
            return self.counts.copy()

    def open(self, method, path, data=None, headers=None):
        if not path.startswith('/') or any(c in path for c in '\r\n'):
            raise ValueError('invalid transport path')
        fields = dict(headers or {})
        fields.update(Host=self.host + ':' + str(self.port), Connection='close', **{'Content-Length': str(len(data or b''))})
        if any('\r' in str(v) or '\n' in str(v) for v in fields.values()):
            raise ValueError('invalid transport header')
        stream = TLSStream(self.host, self.port, self.context, self.timeout, self.record)
        try:
            request = method + ' ' + path + ' HTTP/1.1\r\n' + ''.join(k + ': ' + str(v) + '\r\n' for k, v in fields.items()) + '\r\n'
            stream.sendall(request.encode('ascii'))
            if data:
                stream.sendall(data)
            response = http.client.HTTPResponse(stream)
            response.begin()
            if response.status >= 400:
                raise urllib.error.HTTPError(self.url + path, response.status, 'transport refused', response.headers, response)
            return response
        except urllib.error.HTTPError:
            raise
        except BaseException:
            stream.close()
            raise

    def fetch(self, manifest, token):
        size, identity = manifest.get('bytes'), manifest.get('id')
        digest = manifest.get('sha256')
        if (manifest.get('schema') != 'hybrid-ranges-v1' or type(size) is not int or not 0 < size <= MAX_PAYLOAD
                or not isinstance(identity, str) or not re.fullmatch('[a-f0-9]{32}', identity)
                or not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest)):
            raise ValueError('invalid immutable envelope manifest')
        path = '/payloads/' + identity
        def part(start, end):
            with self.open('GET', path, headers={'Authorization': 'Bearer ' + token, 'Range': f'bytes={start}-{end-1}'}) as response:
                if (response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end-1}/{size}'
                        or response.headers.get('Content-Length') != str(end-start)):
                    raise ValueError('wrong payload range')
                try:
                    body = response.read(end-start+1)
                except (http.client.IncompleteRead, ssl.SSLEOFError):
                    raise ValueError('truncated payload range') from None
                if len(body) != end-start:
                    raise ValueError('truncated payload range')
                return body
        stride = (size + self.streams - 1) // self.streams
        try:
            with ThreadPoolExecutor(max_workers=self.streams) as pool:
                futures = [pool.submit(part, n, min(n+stride, size)) for n in range(0, size, stride)]
                body = b''.join(f.result() for f in futures)
            if not hmac.compare_digest(hashlib.sha256(body).hexdigest(), digest):
                raise ValueError('payload hash mismatch')
            return body
        finally:
            # A failed response must not leave the gateway's bounded cache occupied.
            try:
                with self.open('DELETE', path, headers={'Authorization': 'Bearer ' + token}) as response:
                    response.read(1024)
            except urllib.error.HTTPError as error:
                error.close()
            except OSError:
                pass  # The lease also expires; never replace the original integrity failure.


class PayloadCache:
    def __init__(self, limit=2*MAX_PAYLOAD, ttl=300):
        if type(limit) is not int or limit < MAX_PAYLOAD or not 1 <= ttl <= 3600:
            raise ValueError('invalid payload cache budget')
        self.limit, self.ttl, self.lock = limit, ttl, threading.Lock()
        self.objects, self.reserved = {}, 0

    def prune(self):
        now = time.monotonic()
        self.objects = {k:v for k,v in self.objects.items() if v[2] > now}

    def reserve(self):
        with self.lock:
            self.prune()
            if self.reserved + sum(len(v[1]) for v in self.objects.values()) + MAX_PAYLOAD > self.limit:
                raise ValueError('payload cache full')
            self.reserved += MAX_PAYLOAD

    def publish(self, owner, body):
        if not 0 < len(body) <= MAX_PAYLOAD:
            raise ValueError('payload size limit')
        identity = secrets.token_hex(16)
        with self.lock:
            self.objects[identity] = owner, body, time.monotonic()+self.ttl
        return dict(schema='hybrid-ranges-v1', id=identity, bytes=len(body), sha256=hashlib.sha256(body).hexdigest())

    def get(self, identity, owner):
        with self.lock:
            self.prune()
            row = self.objects.get(identity)
            if row is None or not hmac.compare_digest(row[0], owner):
                raise KeyError('payload not found')
            return row[1]

    def remove(self, identity, owner):
        with self.lock:
            row = self.objects.get(identity)
            if row is None or not hmac.compare_digest(row[0], owner):
                raise KeyError('payload not found')
            del self.objects[identity]


class Gateway(http.server.ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, upstream, tokens, cert, key, probe_bytes=128*1024**2, **cache):
        parsed = urllib.parse.urlsplit(upstream)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', '::1') or parsed.path not in ('', '/'):
            raise ValueError('gateway upstream must be a loopback HTTP origin')
        if not tokens or any(not isinstance(t,str) or not re.fullmatch('[A-Za-z0-9._~-]{32,256}',t) for t in tokens):
            raise ValueError('strong per-run credentials required')
        if not 1024 <= probe_bytes <= MAX_PAYLOAD:
            raise ValueError('invalid probe size')
        self.upstream, self.tokens, self.probe_bytes = upstream.rstrip('/'), tuple(tokens), probe_bytes
        self.cache, self.admission = PayloadCache(**cache), threading.BoundedSemaphore(32)
        super().__init__(address, GatewayHandler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert, key)
        self.tls_context = context

    def process_request(self, request, address):
        if not self.admission.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request,address)
        except BaseException:
            self.admission.release()
            raise

    def process_request_thread(self, request, address):
        try:
            try:
                wrapped = self.tls_context.wrap_socket(request, server_side=True)
                wrapped.settimeout(300)
            except OSError:
                self.shutdown_request(request)
                return
            super().process_request_thread(wrapped, address)
        finally:
            self.admission.release()

    def get_request(self):
        sock, address = super().get_request()
        sock.settimeout(10)
        return sock, address


class GatewayHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, body=b'', headers=None):
        self.send_response(status)
        self.send_header('Content-Length', str(len(body)))
        for k,v in (headers or {}).items():
            self.send_header(k,v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def handle_request(self):
        reserved = False
        try:
            authorization = self.headers.get('Authorization', '')
            token = authorization.removeprefix('Bearer ')
            if not authorization.startswith('Bearer ') or not re.fullmatch('[A-Za-z0-9._~-]{32,256}',token) or not any(hmac.compare_digest(token, t) for t in self.server.tokens):
                self.reply(401);return
            if self.path.startswith('/payloads/'):
                identity = self.path[len('/payloads/'):]
                if not re.fullmatch('[a-f0-9]{32}', identity):
                    self.reply(404);return
                if self.command == 'DELETE':
                    self.server.cache.remove(identity, token);self.reply(204);return
                if self.command != 'GET':
                    self.reply(405);return
                body = self.server.cache.get(identity, token)
                match = re.fullmatch(r'bytes=(\d+)-(\d+)', self.headers.get('Range', ''))
                if not match:
                    self.reply(416);return
                start,end = map(int,match.groups())
                if not 0 <= start <= end < len(body):
                    self.reply(416);return
                self.reply(206, body[start:end+1], {'Content-Range':f'bytes {start}-{end}/{len(body)}'});return
            length = int(self.headers.get('Content-Length','0'))
            if not 0 <= length <= MAX_CONTROL or self.headers.get('Transfer-Encoding'):
                self.reply(413);return
            data = self.rfile.read(length)
            if len(data) != length:
                self.reply(400);return
            if self.command == 'POST' and self.path == '/v1/bench/parallel-link':
                self.server.cache.reserve();reserved=True
                manifest = self.server.cache.publish(token, b'x'*self.server.probe_bytes)
                self.reply(200,json.dumps(manifest).encode());return
            if not (re.fullmatch(r'/v1/(?:health|sessions(?:/[a-f0-9]{32}(?:/extend)?)?|bench/trace-flush)',self.path)
                    and self.command in ('GET','POST','DELETE')):
                self.reply(404);return
            parallel = self.headers.get('X-Hybrid-Parallel') == '1'
            if self.path.endswith('/extend'):
                self.server.cache.reserve();reserved=True
            headers = {'Authorization':'Bearer '+token,'Content-Type':'application/json'}
            for k in ('X-Hybrid-Bench','X-Hybrid-Trace-Call','X-Hybrid-Trace-Parent','X-Hybrid-Trace-Domain',
                      'X-Hybrid-Trace-Origin','X-Hybrid-Trace-Session','X-Hybrid-Trace-Transfer','X-Hybrid-Trace-Chunk',
                      'X-Hybrid-Trace-N0','X-Hybrid-Trace-N1'):
                if self.headers.get(k):headers[k]=self.headers[k]
            req = urllib.request.Request(self.server.upstream+self.path,data=data if length else None,method=self.command,headers=headers)
            with urllib.request.urlopen(req,timeout=300) as response:
                binary = response.headers.get_content_type() == 'application/octet-stream'
                body = response.read(MAX_PAYLOAD+1 if binary else MAX_CONTROL+1)
                if len(body) > (MAX_PAYLOAD if binary else MAX_CONTROL):
                    raise ValueError('upstream body limit')
                fields = {k:v for k,v in response.headers.items() if k.lower() in ('x-hybrid-prefill-ms','x-hybrid-node-compute-ms','content-type')}
                if parallel:
                    if response.headers.get_content_type() != 'application/octet-stream':
                        raise ValueError('upstream did not return envelope')
                    body=json.dumps(self.server.cache.publish(token,body)).encode()
                    fields['Content-Type']='application/json'
                self.reply(response.status,body,fields)
        except urllib.error.HTTPError as error:
            self.reply(error.code)
        except KeyError:
            self.reply(404)
        except (ValueError,OSError,http.client.HTTPException):
            self.reply(400)
        finally:
            if reserved:
                with self.server.cache.lock:self.server.cache.reserved-=MAX_PAYLOAD

    do_GET = do_POST = do_DELETE = handle_request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind',default='0.0.0.0');parser.add_argument('--port',type=int,default=9443)
    parser.add_argument('--upstream',default='http://127.0.0.1:8091')
    parser.add_argument('--token-file',required=True);parser.add_argument('--cert',required=True);parser.add_argument('--key',required=True)
    args=parser.parse_args()
    from pathlib import Path
    import stat
    for path in (args.token_file,args.key):
        if stat.S_IMODE(Path(path).stat().st_mode)&0o077:raise ValueError('private credential/key file required')
    tokens=json.loads(Path(args.token_file).read_text())
    if not isinstance(tokens,dict):raise ValueError('node id-to-token file required')
    server=Gateway((args.bind,args.port),args.upstream,tokens.values(),args.cert,args.key)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()

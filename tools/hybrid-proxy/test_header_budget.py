"""Native reader maximum-range control and atomic header-budget negatives; small CPU model only."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import time

from proxy import HTTPError, Node
from store import prefix_hash
from test_decode import envelope, require, unpack


def listener_budget(node):
    # Stream the forbidden size in fixed chunks; the listener must reject it without reaching the slot route.
    before = node.request('GET', '/slots/0/info')
    port = int(node.url.rsplit(':', 1)[1])
    limit = 512 * 1024 * 1024 + 1024 * 1024 + 52
    with socket.create_connection(('127.0.0.1', port), timeout=30) as sock:
        sock.sendall((f'POST /slots/0?action=import_delta HTTP/1.1\r\nHost: localhost\r\n'
                      f'Content-Type: application/octet-stream\r\nContent-Length: {limit + 1}\r\nConnection: close\r\n\r\n').encode())
        remaining, chunk = limit + 1, b'x' * 65536
        try:
            while remaining:
                block = chunk[:min(len(chunk), remaining)]
                sock.sendall(block)
                remaining -= len(block)
        except BrokenPipeError:
            pass
        status = sock.recv(4096).split(b'\r\n', 1)[0]
    require(b' 413 ' in status, 'listener did not reject envelope-cap overflow')
    require(node.request('GET', '/slots/0/info') == before, 'listener rejection changed slot')
    print('PASS listener envelope-cap overflow: HTTP 413; slot info unchanged', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--prefill-binary', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--delta-fixture', help='Reuse an actual previously recorded A maximum.delta; skips A generation and its 16385 control')
    args = parser.parse_args()
    directory = Path(args.artifacts).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    slots = directory / 'slots'
    slots.mkdir(exist_ok=True)
    credentials = directory / 'clients.json'
    credentials.write_text(json.dumps({'a': 'header-fixture-only'}))
    credentials.chmod(0o600)
    model_hash = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    processes, logs = [], []

    def start(command, health, name):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        log = open(directory / (name + '.log'), 'wb')
        logs.append(log)
        process = subprocess.Popen(command(port), stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        (directory / (name + '.pid')).write_text(str(process.pid) + '\n')
        binary = Path(command(port)[0]).resolve()
        (directory / (name + '-identity.json')).write_text(json.dumps(dict(
            command=command(port), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            library_sha256=hashlib.sha256((binary.parent / 'libllama.so').read_bytes()).hexdigest()), indent=2))
        node = Node(f'http://127.0.0.1:{port}', 'header-fixture-only' if name == 'prefill' else '', timeout=900)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            require(process.poll() is None, name + ' exited during startup')
            try:
                node.request('GET', health)
                return node
            except (HTTPError, OSError):
                time.sleep(0.1)
        raise AssertionError(name + ' startup timed out')

    try:
        prefill = None if args.delta_fixture else start(lambda p: [args.prefill_binary, args.model, str(credentials), str(p), '16384'], '/v1/health', 'prefill')
        decode = start(lambda p: [args.binary, '-m', args.model, '--host', '127.0.0.1', '--port', str(p),
            '-ngl', '0', '-t', '2', '-tb', '2', '-c', '16640', '-b', '256', '-ub', '256', '-np', '1',
            '-ctk', 'q8_0', '-ctv', 'q8_0', '-fa', 'on', '--slots', '--no-context-shift', '--cache-ram', '0',
            '--no-cache-idle-slots', '--ctx-checkpoints', '0', '--slot-save-path', str(slots) + '/'], '/health', 'decode')
        listener_budget(decode)
        tokens = [100000] * 16384
        # The invalid neighboring range must be rejected before A mutates; the valid maximum is a real native record.
        if prefill:
            sid = prefill.request('POST', '/v1/sessions', {'model_sha256': model_hash})['session_id']
            path = '/v1/sessions/' + sid
            before_remote = prefill.request('GET', path)
            try:
                prefill.request('POST', path + '/extend', dict(n0=0, prefix_hash_n0=prefix_hash([]), tokens=tokens + [100000], return_delta=True))
            except HTTPError as error:
                require(error.status == 400, 'wrong 16385 rejection status')
            else:
                raise AssertionError('16385 accepted')
            require(prefill.request('GET', path) == before_remote, '16385 changed remote state')
            print('PASS A 16385 range rejected unchanged', flush=True)
            wire = prefill.request('POST', path + '/extend', dict(n0=0, prefix_hash_n0=prefix_hash([]), tokens=tokens, return_delta=True), binary=True)
        else:
            wire = Path(args.delta_fixture).read_bytes()
        (directory / 'maximum.delta').write_bytes(wire)
        header, raw = unpack(wire)
        sid = header['session_id']
        require(header['model_sha256'] == model_hash and header['tokens'] == tokens and
                header['n0'] == 0 and header['n1'] == 16384, 'wrong maximum fixture')
        length, = struct.unpack_from('<I', wire)
        require(65536 < length <= 4096 + 11 * 16384, 'maximum header did not cross old 64KiB cap')
        headers = {'Content-Type': 'application/octet-stream', 'X-Hybrid-Session': sid}
        before = decode.request('GET', '/slots/0/info')
        decode.request('POST', '/slots/0?action=save', {'filename': 'before.bin'})
        before_bytes = (slots / 'before.bin').read_bytes()
        # Rejections use valid checksums and real state, so header admission, not a checksum error, is exercised.
        cases = [('range-header-budget', envelope(dict(header, padding='x' * (4096 + 11 * 16384)), raw), 'delta header exceeds range budget'),
                 ('absolute-header-budget', envelope(dict(header, padding='x' * 1048576), raw), 'invalid delta header length')]
        for name, candidate, reason in cases:
            try:
                decode.request('POST', '/slots/0?action=import_delta', candidate, headers)
            except HTTPError as error:
                require(error.status == 400 and reason.encode() in error.body, name + ': wrong rejection')
            else:
                raise AssertionError(name + ': accepted')
            require(decode.request('GET', '/slots/0/info') == before, name + ': metadata changed')
            decode.request('POST', '/slots/0?action=save', {'filename': 'after.bin'})
            require((slots / 'after.bin').read_bytes() == before_bytes, name + ': native state changed')
            print('PASS HTTP ' + name + ': intended 400; slot byte-identical', flush=True)
        decode.request('POST', '/slots/0?action=import_delta', wire, headers)
        require(decode.request('GET', '/slots/0/info') == dict(n=16384, prefix_hash=prefix_hash(tokens), tokens=tokens), 'maximum native import mismatch')
        decode.request('POST', '/slots/0?action=save', {'filename': 'maximum.bin'})
        snapshot = (slots / 'maximum.bin').read_bytes()
        packed_count, = struct.unpack_from('<I', snapshot, 8)
        require(snapshot[12 + 4 * packed_count:] == raw[8:], 'maximum native slot differs from A source')
        print(f'PASS test_native_header_budget_16384: header={length}; maximum range imported; native bytes equal A source', flush=True)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()

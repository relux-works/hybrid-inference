"""Profile real proxy receive and decoder imports; large bodies stay in RAM."""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hybrid-proxy'))
from proxy import HTTPError, Node
from store import prefix_hash
from stub_server import xxh128


def require(condition, message):
    if not condition:
        raise ValueError(message)


def synthetic(golden, count, append=False):
    # This fixture scales real qwen35 Q8_0 KV rows and retains its real recurrent tensors; it is not an inference oracle.
    wire = Path(str(golden) + '.full.delta').read_bytes()
    size, = struct.unpack_from('<I', wire)
    header = json.loads(wire[4:4 + size])
    state = wire[4 + size:-48]
    streams, cells = struct.unpack_from('<II', state, 8)
    n0 = cells if append else 0
    require(streams == 1 and 0 < cells < count <= 65536, 'unsupported synthetic geometry')
    offset = 16 + cells * 24
    transposed, layers, _, _ = struct.unpack_from('<4I', state, offset)
    require(transposed == 0 and layers == 6, 'synthetic fixture requires installed 0.8B qwen35 golden')
    parts = [state[:12], struct.pack('<I', count)]
    parts.append(b''.join(struct.pack('<i', n0 + i) + state[20:40] for i in range(count)))
    parts.append(state[offset:offset + 16])
    offset += 16
    for _ in range(2 * layers):
        _, row = struct.unpack_from('<iQ', state, offset)
        require(row == 544, 'unexpected synthetic KV row')
        parts.extend((state[offset:offset + 12], state[offset + 12:offset + 12 + row] * count))
        offset += 12 + row * cells
    require(struct.unpack_from('<I', state, offset)[0] == 1, 'expected one recurrent cell')
    parts.extend((state[offset:offset + 4], struct.pack('<i', n0 + count - 1), state[offset + 8:]))
    body = b''.join(parts)
    tokens = (header['tokens'] if append else []) + [1] * count
    header.update(n0=n0, n1=n0 + count, transfer_n0=n0, transfer_n1=n0 + count,
                  chunk_index=0, chunk_count=1, fixed_state=True,
                  tokens=[1] * count, prefix_hash_n1=prefix_hash(tokens),
                  sections=[dict(name='llama_seq_state', offset=0, bytes=len(body))])
    text = json.dumps(header, separators=(',', ':')).encode()
    return b''.join((struct.pack('<I', len(text)), text, body, xxh128(body), hashlib.sha256(body).digest()))


@contextmanager
def decoder(binary, model, context, gpu_layers, directory, label):
    slots = directory / (label + '-slots')
    slots.mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(Path(binary).resolve()), '-m', str(Path(model).resolve()), '--host', '127.0.0.1',
               '--port', str(port), '-ngl', str(gpu_layers), '-t', '2', '-tb', '2', '-c', str(context),
               '-b', '256', '-ub', '256', '-np', '1', '-ctk', 'q8_0', '-ctv', 'q8_0', '-fa', 'on',
               '--slots', '--slot-save-path', str(slots.resolve()) + os.sep,
               '--no-context-shift', '--cache-ram', '0', '--no-cache-idle-slots', '--ctx-checkpoints', '0']
    log_path = directory / (label + '-decoder.log')
    with log_path.open('wb') as log:
        env = dict(os.environ, LLAMA_HYBRID_IMPORT_PROFILE='1')
        env['LD_LIBRARY_PATH'] = str(Path(binary).resolve().parent) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
        (directory / (label + '-process.json')).write_text(json.dumps(dict(pid=process.pid, command=command)))
        try:
            node = Node(f'http://127.0.0.1:{port}', timeout=7200)
            deadline = time.monotonic() + 600
            while True:
                require(process.poll() is None, 'decoder exited; see ' + str(log_path))
                try:
                    node.request('GET', '/health')
                    break
                except (HTTPError, OSError):
                    require(time.monotonic() < deadline, 'decoder startup timed out')
                    time.sleep(0.1)
            yield node, log_path
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            (directory / (label + '-exit.json')).write_text(json.dumps(dict(pid=process.pid, exit_code=process.returncode)))


def measure(node, log_path, wire, pair, label, receive_ms, prefix=None):
    size, = struct.unpack_from('<I', wire)
    header = json.loads(wire[4:4 + size])
    node.request('POST', '/slots/0?action=erase', {})
    if prefix:
        node.request('POST', '/slots/0?action=import_delta', prefix,
                     {'Content-Type': 'application/octet-stream', 'X-Hybrid-Session': header['session_id']})
    offset = log_path.stat().st_size
    start = time.monotonic()
    info = node.request('POST', '/slots/0?action=import_delta', wire,
                        {'Content-Type': 'application/octet-stream', 'X-Hybrid-Session': header['session_id']})
    elapsed = 1000 * (time.monotonic() - start)
    require(info['n'] == header['n1'] and info['prefix_hash'] == header['prefix_hash_n1']
            and info['tokens'][header['n0']:] == header['tokens'], 'import endpoint/token identity mismatch')
    with log_path.open('rb') as log:
        log.seek(offset)
        text = log.read().decode(errors='replace')
    stages = {}
    events = []
    for name, count, ms in re.findall(r'hybrid_import stage=(\w+) bytes=(\d+) ms=([\d.]+)', text):
        stages[name] = stages.get(name, 0) + float(ms)
        events.append(dict(stage=name, bytes=int(count), ms=float(ms)))
    require(all(name in stages for name in ('codec', 'checksum', 'native', 'native_validate', 'native_parse', 'tensor_write')),
            'missing import instrumentation')
    copy_ms = sum(stages.get(name, 0) for name in ('http_body_copy', 'http_request_copy', 'task_copy', 'state_copy'))
    parse_ms = stages['codec'] - stages['checksum'] - stages.get('state_copy', 0) + stages['native_validate'] + stages['native_parse']
    sync_ms = stages.get('synchronize', 0)
    # Synchronize is nested in backup; report backup inclusive and keep the residual independent of that split.
    backup_ms = stages.get('backup', 0)
    named_ms = copy_ms + parse_ms + stages['checksum'] + stages['tensor_write'] + backup_ms
    body_bytes = header['sections'][0]['bytes']
    result = dict(pair=pair, label=label, wire_bytes=len(wire), body_bytes=body_bytes, receive_ms=receive_ms,
                  import_ms=elapsed, copy_ms=copy_ms, parse_ms=parse_ms, checksum_ms=stages['checksum'],
                  tensor_write_ms=stages['tensor_write'], synchronize_ms=sync_ms, backup_ms=backup_ms,
                  transport_dispatch_free_ms=elapsed - named_ms, stages=stages, events=events)
    result['MBps'] = {name[:-3]: body_bytes / (1000 * value) for name, value in result.items()
                      if name.endswith('_ms') and name not in ('synchronize_ms', 'backup_ms') and isinstance(value, (float, int)) and value > 0}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', action='append', required=True, help='LABEL=llama-server; two labels produce alternating paired runs')
    parser.add_argument('--model', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--golden', type=Path, help='0.8B golden stem, expanded in memory')
    source.add_argument('--wire', type=Path, help='real fresh HYBD envelope, never modified')
    source.add_argument('--url', help='replay a fresh HYBD envelope over HTTP, e.g. through a loopback SSH tunnel')
    parser.add_argument('--tokens', type=int, default=45056)
    parser.add_argument('--append', action='store_true', help='measure replacement of an occupied recurrent state; requires --golden')
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'output directory already exists')
    require(args.runs >= 1, 'runs must be positive')
    args.output.mkdir(parents=True)
    require(not args.append or args.golden, '--append requires --golden')
    prefix = Path(str(args.golden) + '.full.delta').read_bytes() if args.append else None
    if args.golden:
        wire = synthetic(args.golden, args.tokens, args.append)
    elif args.wire:
        wire = args.wire.read_bytes()
    else:
        wire = Node(args.url, timeout=7200).request('GET', '', binary=True)
    size, = struct.unpack_from('<I', wire)
    header = json.loads(wire[4:4 + size])
    require(header['n0'] == 0 or prefix, 'profile requires fresh delta or --append')

    class Source(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Length', str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Source)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    records = []
    try:
        with ExitStack() as stack:
            decoders = {}
            for entry in args.binary:
                label, binary = entry.split('=', 1)
                require(re.fullmatch(r'[A-Za-z0-9_-]+', label) and label not in decoders, 'invalid/duplicate label')
                decoders[label] = stack.enter_context(decoder(binary, args.model, header['n1'] + 256,
                                                             args.gpu_layers, args.output, label))
            for pair in range(args.runs):
                for label in list(decoders)[::1 if pair % 2 == 0 else -1]:
                    source_node = Node(args.url or 'http://127.0.0.1:' + str(server.server_port), timeout=7200)
                    received = source_node.request('GET', '' if args.url else '/', binary=True)
                    require(received == wire, 'proxy receive changed bytes')
                    node, log_path = decoders[label]
                    record = measure(node, log_path, received, pair, label, source_node.last_transfer_seconds * 1000, prefix)
                    records.append(record)
                    (args.output / 'measurements.json').write_text(json.dumps(records, indent=2) + '\n')
                    print(json.dumps({key: record[key] for key in ('pair', 'label', 'body_bytes', 'receive_ms', 'import_ms', 'MBps')}), flush=True)
                    del received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == '__main__':
    main()

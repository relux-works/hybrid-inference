"""Opt-in CPU overlap/solo equivalence control; uses existing llama-server, never builds."""
import argparse
import concurrent.futures
import hashlib
import http.server
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time


def require(value, message):
    if not value:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--proxy-source', help='Read-only proxy module directory, for the M0 red witness')
    args = parser.parse_args()
    if args.proxy_source:
        sys.path.insert(0, str(Path(args.proxy_source).resolve()))
    from proxy import Handler, HTTPError, Node, Proxy
    from store import prefix_hash

    root = Path(args.artifacts).resolve()
    root.mkdir(parents=True, exist_ok=True)
    slots = root / 'slots'
    slots.mkdir(exist_ok=True)
    with socket.socket() as allocated:
        allocated.bind(('127.0.0.1', 0))
        port = allocated.getsockname()[1]
    command = [str(Path(args.binary).resolve()), '-m', args.model, '--host', '127.0.0.1', '--port', str(port),
               '-ngl', '0', '-t', '2', '-tb', '2', '-c', '8192', '-b', '256', '-ub', '256', '-np', '2',
               '-ctk', 'q8_0', '-ctv', 'q8_0', '-fa', 'on', '--slots', '--no-context-shift', '--cache-ram', '0',
               '--no-cache-idle-slots', '--ctx-checkpoints', '32', '--checkpoint-min-step', '0',
               '--slot-save-path', str(slots) + '/']
    log = (root / 'server.log').open('wb')
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    (root / 'server.pid').write_text(str(process.pid) + '\n')
    (root / 'identity.json').write_text(json.dumps(dict(command=command,
        binary_sha256=hashlib.sha256(Path(args.binary).read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256((Path(args.binary).parent / 'libllama.so').read_bytes()).hexdigest(),
        proxy_source=str(Path(sys.modules['proxy'].__file__).resolve()),
        source_sha256={name: hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest()
                       for name in ('proxy', 'store', 'policy')},
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    app = server = worker = None
    stop = threading.Event()
    monitor = None
    try:
        decode = Node(f'http://127.0.0.1:{port}', timeout=120)
        deadline = time.monotonic() + 120
        while True:
            require(process.poll() is None, 'native server exited during startup')
            try:
                decode.request('GET', '/health')
                break
            except (OSError, HTTPError):
                require(time.monotonic() < deadline, 'native server readiness timeout')
                time.sleep(0.1)
        app = Proxy(dict(model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
            prefill_url='http://127.0.0.1:1', prefill_timeout=0.1, decode_url=decode.url, decode_timeout=120,
            slots=2, slot_path=str(slots), clients={'fixture-native': dict(id='native', prefill_token='unused-fixture')},
            policy=dict(link_rate=1)), str(root / 'proxy.sqlite'))
        records = []
        phase = 'solo'
        opened = app.decode.open

        class Capture:
            def __init__(self, response, request):
                self.response = response
                self.record = dict(phase=phase, question=request['messages'][-1]['content'], slot=request['id_slot'],
                                   first=None, last=None, tokens=[])

            def __enter__(self):
                self.response.__enter__()
                return self

            def __exit__(self, *exc):
                records.append(self.record)
                return self.response.__exit__(*exc)

            def __iter__(self):
                for line in self.response:
                    if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                        chunk = json.loads(line[6:])
                        tokens = [item['id'] for choice in chunk.get('choices', [])
                                  for item in (choice.get('logprobs') or {}).get('content', [])]
                        if tokens:
                            self.record['tokens'].extend(tokens)
                            now = time.monotonic()
                            if self.record['first'] is None:
                                self.record['first'] = now
                            self.record['last'] = now
                    yield line

        def capture(method, path, data=None, headers=None):
            response = opened(method, path, data, headers)
            return Capture(response, data) if path == '/v1/chat/completions' else response

        app.decode.open = capture
        checked = []
        finish = app.finish
        def audit(session, prompt):
            finish(session, prompt)
            # The lease is still pinned: no eviction can invalidate this per-request token/hash assertion.
            info = decode.request('GET', f"/slots/{session['slot']}/info")
            generated = next(r['tokens'] for r in reversed(records) if r['slot'] == session['slot'])
            require(info['tokens'] == prompt + generated, 'native request lost or duplicated reconciled tokens')
            require(info['tokens'] == app.store.get(session['id'])['tokens'] and
                    info['prefix_hash'] == prefix_hash(info['tokens']), 'native slot/journal mismatch')
            with app.lock:
                bindings = [s['slot'] for s in app.store.pending() if s['slot'] is not None]
                require(len(bindings) == len(set(bindings)), 'native slot bound to two sessions')
            checked.append(dict(session=session['id'], slot=session['slot'], n=info['n'], prefix_hash=info['prefix_hash']))
        app.finish = audit
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.app = app
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        client = Node(f'http://127.0.0.1:{server.server_port}', 'fixture-native', timeout=120)
        questions = ['Write integers 1 through 100 as decimal digits, separated by a comma and a space. Begin exactly: 1, 2, 3,',
                     'Write integers 100 through 200 as decimal digits, separated by a comma and a space. Begin exactly: 100, 101, 102,']

        def chat(index, barrier=None):
            body = dict(messages=[dict(role='system', content='Follow the request exactly. ' * 16),
                                  dict(role='user', content=questions[index])], stream=True,
                        max_tokens=64, temperature=0, seed=0, ignore_eos=True, return_tokens=True, verbose=True,
                        logprobs=True, top_logprobs=1,
                        chat_template_kwargs=dict(enable_thinking=False))
            if barrier:
                barrier.wait(timeout=10)
            sid = f'{phase}-{index}'
            with client.open('POST', '/v1/chat/completions', body, {'X-Hybrid-Session': sid}) as response:
                wire = response.read()
                require(b'data: [DONE]' in wire, 'native stream did not complete')
            session = app.store.get(sid)
            info = decode.request('GET', f"/slots/{session['slot']}/info")
            require(session['tokens'] == info['tokens'] and prefix_hash(session['tokens']) == info['prefix_hash'],
                    'native journal and slot diverged')

        # Positive solo controls establish independently generated 64-token outputs before concurrency.
        for index in (0, 1):
            chat(index)
        phase = 'parallel'
        snapshots = []

        def observe():
            while not stop.is_set():
                snapshots.append(dict(at=time.monotonic(), slots=decode.request('GET', '/slots')))
                stop.wait(0.01)

        monitor = threading.Thread(target=observe, daemon=True)
        monitor.start()
        barrier = threading.Barrier(2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(chat, i, barrier) for i in (0, 1)]
            for future in futures:
                future.result(timeout=180)
        stop.set()
        monitor.join(timeout=5)
        (root / 'records.json').write_text(json.dumps(records, indent=2))
        (root / 'slot-windows.json').write_text(json.dumps(snapshots, indent=2))
        solo = {r['question']: r for r in records if r['phase'] == 'solo'}
        parallel = [r for r in records if r['phase'] == 'parallel']
        require(len(solo) == len(parallel) == 2, 'missing native streams')
        for record in parallel:
            require(len(record['tokens']) == 64 and record['tokens'] == solo[record['question']]['tokens'],
                    'parallel/solo greedy token mismatch')
        print('PASS native parallel/solo exact greedy tokens: 64/64 each', flush=True)
        # M0 must fail this claim: windows use actual decoded-token SSEs, plus independently polled server activity.
        overlap = min(r['last'] for r in parallel) - max(r['first'] for r in parallel)
        both_processing = any(sum(s['is_processing'] for s in snap['slots']) == 2 for snap in snapshots)
        print(f'OBSERVED decode_overlap_seconds={overlap:.6f} both_slots_processing={both_processing}', flush=True)
        require(overlap > 0 and both_processing, 'decode windows serialized; expected concurrent sessions')
        print('PASS native overlap: simultaneous decode windows and two processing slots', flush=True)
        phase = 'contention'

        def conversation(index):
            sid = f'contention-{index}'
            messages = [dict(role='user', content=f'Count from {index + 1} to one hundred, separated by commas.')]
            for turn in range(3):
                body = dict(messages=messages, stream=True, max_tokens=8, temperature=0, seed=0,
                            ignore_eos=True, logprobs=True, top_logprobs=1,
                            chat_template_kwargs=dict(enable_thinking=False))
                try:
                    with client.open('POST', '/v1/chat/completions', body, {'X-Hybrid-Session': sid}) as response:
                        wire = response.read()
                except HTTPError as error:
                    (root / f'failure-{index}-{turn}.json').write_text(json.dumps(dict(
                        status=error.status, error=error.body.decode(), body=body, session=app.store.get(sid)), indent=2))
                    raise
                require(b'data: [DONE]' in wire, 'native contention stream incomplete')
                content = []
                for line in wire.splitlines():
                    if line.startswith(b'data: ') and line != b'data: [DONE]':
                        for choice in json.loads(line[6:]).get('choices', []):
                            content.append(choice.get('delta', {}).get('content') or '')
                # M0.2 keeps the native IDs across ordinary user turns; no assistant-continuation workaround is needed.
                messages.append(dict(role='assistant', content=''.join(content)))
                messages.append(dict(role='user', content='Continue counting decimal integers.'))
            return sid

        # Four native conversations contend for two slots, each with three ordinary user turns.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            sessions = list(pool.map(conversation, range(4)))
        require(len(checked) == 16, 'missing per-request native journal checks')
        require(all(sum(c['session'] == sid for c in checked) == 3 for sid in sessions), 'missing contention turns')
        (root / 'checked-journals.json').write_text(json.dumps(checked, indent=2))
        (root / 'records.json').write_text(json.dumps(records, indent=2))
        print('PASS native contention: four threads, four sessions, three requests/session, two slots; all 16 requests checked', flush=True)
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=5)
        if server:
            server.shutdown()
            server.server_close()
        if worker:
            worker.join()
        if app:
            app.store.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


if __name__ == '__main__':
    main()

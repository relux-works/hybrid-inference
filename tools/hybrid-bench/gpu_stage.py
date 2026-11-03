"""Shared, GPU-host-only stage primitives. Importing this module starts no work."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

FLOOR = 28_000_000_000
FLAGS = ['-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA=ON', '-DGGML_METAL=OFF', '-DLLAMA_CURL=OFF',
         '-DLLAMA_BUILD_UI=OFF', '-DLLAMA_USE_PREBUILT_UI=OFF']


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def stream_sha(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024**2), b''):
        digest.update(chunk)
    return digest.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return stream_sha(stream)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def run(command, log=None):
    command = [str(p) for p in command]
    def record(process, event):
        if log is not None:
            with Path(str(log)+'.commands.jsonl').open('a') as ledger:
                ledger.write(json.dumps(dict(event=event, command=command, pid=process.pid, exit_code=process.poll()))+'\n')
    with contextlib.ExitStack() as stack:
        stream = stack.enter_context(Path(log).open('ab')) if log is not None else None
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT if stream else None)
        record(process, 'start')
        try:
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=30)
            raise
        finally:
            record(process, 'exit')
    require(code == 0, f'command PID {process.pid} exited {code}; see {log or "terminal"}')


def disk_floor_bytes(value):
    try:
        floor = int(value)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError('disk floor must be integer bytes, at least 1073741824 (1 GiB)') from error
    if floor < 1024**3:
        raise argparse.ArgumentTypeError('disk floor must be at least 1073741824 bytes (1 GiB)')
    return floor


def add_disk_floor_argument(parser):
    parser.add_argument('--disk-floor-bytes', type=disk_floor_bytes, default=FLOOR,
                        help='GPU-stage free-space floor in bytes, default 28000000000; minimum 1073741824 (1 GiB)')


def disk(path, extra=0, floor=FLOOR, log=None):
    require(type(floor) is int and floor >= 1024**3, 'disk floor must be integer bytes, at least 1073741824 (1 GiB)')
    free = shutil.disk_usage(path).free
    event = json.dumps(dict(event='disk_gate', path=str(path), floor_bytes=floor, extra_bytes=extra,
                            required_bytes=floor + extra, free_bytes=free), sort_keys=True)
    print(event, flush=True)
    if log is not None:
        with Path(log).open('a') as output:
            output.write(event+'\n')
    require(free >= floor + extra, f'disk gate: {free} free bytes; need {floor + extra} (floor {floor} + extra {extra})')
    return free


@contextlib.contextmanager
def exclusive(path):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def interrupted(*_):
    raise KeyboardInterrupt('SIGTERM')


def gpu_identity():
    value = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                     '--format=csv,noheader,nounits'], text=True).strip()
    rows = value.splitlines()
    require(len(rows) == 1 and 'RTX PRO 6000' in rows[0] and float(rows[0].rsplit(',', 1)[1]) >= 90000,
            'stage requires one RTX PRO 6000 with at least 90000 MiB VRAM')
    return value


def clean_source(source, ref):
    require(re.fullmatch('[0-9a-f]{40}', ref), 'candidate ref must be a full commit SHA')
    require(subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == ref,
            'candidate source HEAD differs')
    require(not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip(),
            'candidate source must be clean')


def build(source, directory, ref, log, disk_floor_bytes=FLOOR):
    clean_source(source, ref)
    targets = ['llama-server', 'llama-hybrid-prefill', 'llama-hybrid-score', 'llama-hybrid-bench-check']
    stamp = directory/'hybrid-bench-build.json'
    if stamp.exists():
        record = read(stamp)
        require(record['commit'] == ref and record['flags'] == FLAGS and record['source_path'] == str(source),
                'candidate build identity differs')
        require(set(targets) <= set(record['binaries']), 'candidate build record lacks required targets')
        require(all(sha(directory/'bin'/name) == value for name, value in record['binaries'].items()),
                'candidate build binaries changed')
        return
    require(not directory.exists(), 'unrecorded build directory exists; preserve and inspect it')
    disk(directory.parent, 6 * 1024**3, floor=disk_floor_bytes, log=log)
    run(['uptime'], log)
    check = subprocess.run(['pgrep', '-u', str(os.getuid()), 'cc1plus'], capture_output=True, text=True)
    require(check.returncode == 1, 'native compiler already active, or pgrep failed')
    run(['nice', '-n', '10', 'cmake', '-S', source, '-B', directory, *FLAGS], log)
    disk(directory.parent, 4 * 1024**3, floor=disk_floor_bytes, log=log)
    run(['nice', '-n', '10', 'cmake', '--build', directory, '-j4', '--target', *targets], log)
    disk(directory.parent, floor=disk_floor_bytes, log=log)
    names = sorted(p for p in (directory/'bin').iterdir() if p.name in targets or '.so' in p.name)
    require(all((directory/'bin'/name).is_file() for name in targets), 'missing build target')
    save(stamp, dict(commit=ref, source_path=str(source), flags=FLAGS, source_dirty=False,
                     binaries={p.name: sha(p) for p in names}))


def no_evaluator(source, proc=Path('/proc')):
    # Inspect only; never signal or modify an evaluation/controller process.
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry/'cmdline').read_bytes().split(b'\0')
            cwd = (entry/'cwd').resolve()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(os.fsdecode(arg)).name == 'ruler_eval.py' for arg in args if arg):
            require(not (cwd == source or source in cwd.parents or any((cwd/os.fsdecode(arg)).resolve().is_relative_to(source) for arg in args if arg and
                         Path(os.fsdecode(arg)).name == 'ruler_eval.py')),
                    f'RULER evaluator active (PID {entry.name}); leave it untouched and wait for an idle window')


def verify_archive(depth):
    seal = read(depth/'complete.json')
    require(seal['manifest_sha256'] == sha(depth/'manifest.json') and seal['archive_sha256'] == sha(depth/'complete.tar.gz'),
            'depth completion seal differs')
    require((depth/'complete.tar.gz.sha256').read_text().strip() == seal['archive_sha256'] + '  complete.tar.gz',
            'archive sidecar differs')
    with tarfile.open(depth/'complete.tar.gz') as archive:
        members = archive.getmembers()
        require(len({m.name for m in members}) == len(members) and all(m.isfile() and m.size <= 256*1024**2 for m in members),
                'archive has duplicate, non-file or oversized member')
        sums = archive.extractfile('SHA256SUMS')
        if sums is None:
            raise RuntimeError('archive lacks member checksums')
        expected = dict(line.decode().rstrip('\n').split('  ', 1)[::-1] for line in sums)
        require(set(expected) == {m.name for m in members} - {'SHA256SUMS'}, 'archive checksum member set differs')
        for name, value in expected.items():
            stream = archive.extractfile(name)
            require(stream is not None and stream_sha(stream) == value,
                    'archive member checksum differs: ' + name)
        stream = archive.extractfile('manifest.json')
        require(stream is not None and stream.read() == (depth/'manifest.json').read_bytes(), 'archived manifest differs')
    return seal


def terminal(name, function):
    started = time.monotonic()
    try:
        result = function()
        require(result in (None, 0), 'stage returned nonzero')
    except (Exception, KeyboardInterrupt) as error:
        print(f'failed {name}: {error}', file=sys.stderr, flush=True)
        return 1
    print(f'done {name} seconds={time.monotonic()-started:.1f}', flush=True)
    return 0

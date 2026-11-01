"""Shared deployment and benchmark utilities. No third-party Python packages."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import stat
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

GIB = 1024 ** 3
FLOOR = 25 * GIB
ROOT = Path(__file__).resolve().parents[2]


class Refusal(Exception):
    pass


def require(condition, message):
    if not condition:
        raise Refusal(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(command, **kwargs):
    return subprocess.run([str(v) for v in command], check=True, **kwargs)


def output(command):
    return run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()


def private_json(path, value):
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as target:
        json.dump(value, target, indent=2)
        target.write('\n')


def read_private(path):
    path = Path(path)
    require(path.is_file() and not stat.S_IMODE(path.stat().st_mode) & 0o077,
            f'private file required (chmod 600 {path})')
    return json.loads(path.read_text())


def http(url, path, token='', data=None, timeout=120, binary=False, headers=None):
    fields = dict(headers or {})
    if token:
        fields['Authorization'] = 'Bearer ' + token
    if data is not None and not isinstance(data, bytes):
        data = json.dumps(data).encode()
        fields['Content-Type'] = 'application/json'
    request = urllib.request.Request(url.rstrip('/') + path, data=data, headers=fields)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = response.read(600 * 1024 * 1024 + 1)
        require(len(value) <= 600 * 1024 * 1024, 'HTTP response exceeds limit')
        return value if binary else json.loads(value)


def host_info(cuda=False):
    total = available = 0
    if platform.system() == 'Linux':
        values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        total = int(values['MemTotal'].split()[0]) * 1024
        available = int(values['MemAvailable'].split()[0]) * 1024
        chip = platform.machine()
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                chip = line.split(':', 1)[1].strip()
                break
    else:
        total = int(output(['sysctl', '-n', 'hw.memsize']))
        # Unified memory is shared with the OS; a further 20% margin is applied by sizing.
        available = total
        chip = output(['sysctl', '-n', 'machdep.cpu.brand_string'])
    info = dict(os=platform.system(), os_version=platform.release(), arch=platform.machine(), chip=chip,
                memory_bytes=total, available_memory_bytes=available, backend='cpu')
    if cuda:
        row = output(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits']).splitlines()
        require(len(row) == 1, 'select exactly one GPU using a dedicated host; multi-GPU sizing is unsupported')
        name, memory, free, driver = [v.strip() for v in row[0].split(',')]
        info.update(backend='cuda', gpu=name, vram_bytes=int(memory) * 1024**2,
                    available_vram_bytes=int(free) * 1024**2, driver=driver,
                    cuda=output(['nvcc', '--version']).splitlines()[-1])
    elif platform.system() == 'Darwin':
        info['backend'] = 'metal'
    return info


def preflight(cuda=False, node=False):
    system = platform.system()
    require(system in ('Linux', 'Darwin') and not (node and system != 'Linux'), 'prefill deployment requires Linux; client requires Linux or macOS')
    hints = {'git': 'sudo apt-get install git' if system == 'Linux' else 'xcode-select --install',
             'cmake': 'sudo apt-get install cmake' if system == 'Linux' else 'brew install cmake',
             'c++': 'sudo apt-get install build-essential' if system == 'Linux' else 'xcode-select --install',
             'python3': 'sudo apt-get install python3' if system == 'Linux' else 'brew install python@3.12'}
    if system == 'Linux':
        hints['flock'] = 'sudo apt-get install util-linux'
    if cuda:
        hints.update({'nvidia-smi': 'install the GPU vendor driver for this distribution, then reboot',
                      'nvcc': 'install the CUDA Toolkit matching the installed driver; add its bin directory to PATH'})
    for command, hint in hints.items():
        require(shutil.which(command), f'missing {command}; install: {hint}')
        try:
            run([command, '--version'] if command != 'nvidia-smi' else [command, '-L'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            raise Refusal(f'{command} failed readiness check; install/repair: {hint}') from None
    import sys
    require(sys.version_info >= (3, 10), 'Python 3.10+ required; install: brew install python@3.12 (macOS), sudo apt-get install python3 (Linux)')
    free = shutil.disk_usage(ROOT).free
    require(free >= FLOOR, f'free disk {free} bytes below required {FLOOR}; free disk space before retrying')
    info = host_info(cuda)
    require(info['available_memory_bytes'] >= 2 * GIB, 'at least 2 GiB available RAM required; close applications or use a larger host')
    if system == 'Darwin':
        require(shutil.which('xcrun'), 'Metal compiler required; install: xcode-select --install')
        output(['xcrun', '--find', 'metal'])
    return info


def model_metadata(path):
    """Read bounded GGUF metadata, skipping tokenizer arrays without retaining their content."""
    formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    size = Path(path).stat().st_size
    with open(path, 'rb') as source:
        def raw(count):
            require(0 <= count <= 16 * 1024 * 1024, 'invalid GGUF metadata size')
            value = source.read(count)
            require(len(value) == count, 'truncated GGUF metadata')
            return value
        def scalar(fmt):
            return struct.unpack('<' + fmt, raw(struct.calcsize('<' + fmt)))[0]
        def string(keep=True):
            count = scalar('Q')
            require(count <= 16 * 1024 * 1024 and source.tell() + count <= size, 'invalid GGUF string size')
            if keep:
                return raw(count).decode('utf-8')
            source.seek(count, 1)
        def value(kind, keep, depth=0):
            require(depth <= 4, 'nested GGUF metadata exceeds limit')
            if kind in formats:
                return scalar(formats[kind])
            if kind == 8:
                return string(keep)
            require(kind == 9, 'unknown GGUF metadata type')
            subtype, count = scalar('I'), scalar('Q')
            require(count <= 2_000_000, 'GGUF metadata array too large')
            if subtype in formats and not keep:
                length = struct.calcsize('<' + formats[subtype]) * count
                require(source.tell() + length <= size, 'truncated GGUF array')
                source.seek(length, 1)
                return None
            require(not keep or count <= 4096, 'GGUF sizing array too large')
            result = []
            for _ in range(count):
                item = value(subtype, keep, depth + 1)
                if keep:
                    result.append(item)
            return result if keep else None
        require(raw(4) == b'GGUF' and scalar('I') == 3, 'GGUF v3 model required')
        scalar('Q')
        count = scalar('Q')
        require(count <= 100_000, 'GGUF metadata count too large')
        metadata = {}
        for _ in range(count):
            key, kind = string(), scalar('I')
            keep = key in ('general.architecture', 'general.file_type') or key.startswith(('qwen35.', 'qwen35moe.'))
            if key == 'tokenizer.ggml.tokens':
                require(kind == 9 and scalar('I') == 8, 'invalid GGUF token array')
                token_count = scalar('Q')
                require(0 < token_count <= 2_000_000, 'invalid vocabulary size')
                digest = hashlib.sha256()
                for _ in range(token_count):
                    token = string().encode('utf-8')
                    digest.update(struct.pack('<Q', len(token)))
                    digest.update(token)
                metadata['vocab_size'] = token_count
                metadata['vocab_sha256'] = digest.hexdigest()
                continue
            item = value(kind, keep)
            if keep:
                metadata[key] = item
    return metadata


def model_identity(path):
    meta = model_metadata(path)
    arch = meta['general.architecture']
    require(arch == 'qwen35', 'layout identity supports dense qwen35 only')
    def get(key):
        return meta[arch + '.' + key]
    count = get('block_count')
    require(type(count) is int and 0 < count <= 1024, 'invalid layout layer count')
    recurrent = meta.get(arch + '.attention.recurrent_layers')
    if recurrent is None:
        interval = meta.get(arch + '.full_attention_interval', 4)
        require(type(interval) is int and interval > 0, 'invalid attention interval')
        recurrent = [(i + 1) % interval != 0 for i in range(count)]
    require(isinstance(recurrent, list) and len(recurrent) == count, 'invalid recurrent layout')
    def layer_value(key, i):
        value = get(key)
        return value[i] if isinstance(value, list) else value
    layout = dict(version=1, architecture=arch, layer_count=count, embedding_size=get('embedding_length'),
        layers=[dict(heads=layer_value('attention.head_count', i), kv_heads=layer_value('attention.head_count_kv', i),
                     key_dim=layer_value('attention.key_length', i), value_dim=layer_value('attention.value_length', i),
                     recurrent=bool(recurrent[i])) for i in range(count)],
        recurrent_layout={name: get('ssm.' + key) for name, key in
            [('conv', 'conv_kernel'), ('inner', 'inner_size'), ('state', 'state_size'), ('rank', 'time_step_rank'), ('groups', 'group_count')]},
        vocab_size=meta['vocab_size'], vocab_sha256=meta['vocab_sha256'])
    fingerprint = hashlib.sha256(json.dumps(layout, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    kind = meta.get('general.file_type')
    quant = {0: 'F32', 1: 'F16', 2: 'Q4_0', 7: 'Q8_0', 15: 'Q4_K_M', 32: 'BF16'}.get(kind, str(kind))
    return dict(layout_fingerprint=fingerprint, model_layout=layout, quantization_type=quant)


def validate_identity(digest, identity, expected, node_identity, hetero=False):
    require(bool(expected), 'node model identity required')
    if not hetero:
        require(digest == expected, 'client/node model sha256 mismatch; no process started')
        return
    fingerprint = identity.get('layout_fingerprint')
    require(isinstance(fingerprint, str) and len(fingerprint) == 64 and
            fingerprint == node_identity.get('layout_fingerprint'), 'client/node layout fingerprint mismatch')


def context_plan(model, info, requested=None, client=False):
    meta = model_metadata(model)
    arch = meta.get('general.architecture')
    require(arch == 'qwen35', 'automatic sizing currently supports qwen35 dense hybrid GGUF only')
    def get(name):
        value = meta.get(arch + '.' + name)
        require(isinstance(value, int) and value > 0, 'missing or invalid GGUF sizing key: ' + name)
        return value
    layers, inner, state = get('block_count'), get('ssm.inner_size'), get('ssm.state_size')
    recurrent = meta.get(arch + '.attention.recurrent_layers')
    if recurrent is None:
        interval = meta.get(arch + '.full_attention_interval', 4)
        require(isinstance(interval, int) and interval > 0, 'invalid full attention interval')
        attn = layers // interval
    else:
        require(isinstance(recurrent, list) and len(recurrent) == layers, 'invalid recurrent layer table')
        attn = sum(not v for v in recurrent)
    head_k, head_v, heads = get('attention.key_length'), get('attention.value_length'), get('attention.head_count_kv')
    require(head_k % 32 == 0 and head_v % 32 == 0 and attn > 0, 'unsupported q8_0 cache geometry')
    kv = attn * heads * (head_k + head_v) * 34 // 32
    recurrent_bytes = (layers - attn) * 4 * (state * inner + (get('ssm.conv_kernel') - 1) * (inner + 2 * get('ssm.group_count') * state))
    weights = Path(model).stat().st_size
    capacity = info.get('available_vram_bytes', info['available_memory_bytes'])
    # Host RAM holds saved sessions and wire buffers even with a GPU backend.
    budget = int(capacity * 0.80) - weights - 2 * GIB - recurrent_bytes
    memory_ctx = (budget // kv // 256) * 256
    # One mode session plus one sequential checker session can coexist.
    sessions_held = 2
    host_available = int(info['available_memory_bytes'] * .8) - 2 * GIB
    store_budget = max(0, host_available // 3)
    store_ctx = ((store_budget // sessions_held - recurrent_bytes - 1024**2) // (kv + 32) // 256) * 256
    host_ctx = ((int(info['available_memory_bytes'] * 0.8) - 2 * GIB - 4 * recurrent_bytes) // (4 * kv) // 256) * 256
    limit = min(memory_ctx, host_ctx, store_ctx, get('context_length'))
    ctx = requested if requested is not None else limit
    require(isinstance(ctx, int) and 256 <= ctx <= limit,
            f'context {ctx} cannot fit: maximum {limit}; memory={memory_ctx}, host staging={host_ctx}, session store={store_ctx}')
    ram_budget = sessions_held * (recurrent_bytes + ctx * (kv + 32) + 1024**2)
    return dict(context=ctx, maximum_context=limit, sessions_held=sessions_held,
                max_ctx=ctx, ram_budget_mib=(ram_budget + 1024**2 - 1) // 1024**2,
                max_request_mib=max(1, (4096 + 11 * ctx + 1024**2 - 1) // 1024**2), kv_bytes_per_token=kv, recurrent_state_bytes=recurrent_bytes,
                weights_bytes=weights, margin_fraction=0.20, compute_reserve_bytes=2 * GIB,
                memory_limit=memory_ctx, store_limit=store_ctx, protocol_limit=16384,
                checkpoint_count=0 if client else None)


def obtain_model(args, cache):
    expected = args.sha256
    if expected:
        require(len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), '--sha256 must be lowercase hexadecimal')
    if args.model:
        require(not args.hf_repo and not args.source_url, '--model cannot be combined with a download source')
        path = Path(args.model).expanduser().resolve()
        require(path.is_file(), 'model file does not exist')
        digest = sha256(path)
        require(not expected or digest == expected, 'model sha256 mismatch; no state changed')
        return path, digest, False
    require(expected, '--sha256 is required for downloaded models')
    url = args.source_url
    if not url:
        require(args.hf_repo and args.hf_file, 'supply --model or --hf-repo/--hf-file with --sha256')
        url = 'https://huggingface.co/' + urllib.parse.quote(args.hf_repo, safe='/') + '/resolve/main/' + urllib.parse.quote(args.hf_file, safe='/')
    require(urllib.parse.urlsplit(url).scheme in ('https', 'file', 'http'), 'unsupported model source URL')
    if url.startswith('http:'):
        require(urllib.parse.urlsplit(url).hostname in ('127.0.0.1', 'localhost', '::1'), 'plain HTTP downloads require loopback')
    cache = Path(cache)
    require(cache.is_dir(), 'download cache must be an existing directory')
    destination = cache / (expected + '.gguf')
    if destination.exists():
        require(sha256(destination) == expected, 'cached model sha256 mismatch')
        return destination, expected, False
    fd, temporary = tempfile.mkstemp(prefix='.model-', dir=cache)
    try:
        with os.fdopen(fd, 'wb') as target, urllib.request.urlopen(url, timeout=120) as source:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                require(shutil.disk_usage(cache).free - len(block) >= FLOOR, 'download would cross 25 GiB disk floor')
                target.write(block)
        require(sha256(temporary) == expected, 'model sha256 mismatch; temporary download removed')
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination, expected, True


def check_ports(*ports):
    require(len(set(ports)) == len(ports), 'ports must be distinct')
    for port in ports:
        require(1 <= port <= 65535, 'port must be in 1..65535')
        with socket.socket() as sock:
            try:
                sock.bind(('127.0.0.1', port))
            except OSError:
                raise Refusal(f'port {port} is already in use') from None


def process_identity(pid):
    try:
        # Start time prevents a stale PID file from stopping a reused process ID.
        if platform.system() == 'Linux':
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            return None if fields[0] == 'Z' else fields[19]
        return output(['ps', '-p', str(pid), '-o', 'lstart=']) or None
    except (OSError, subprocess.CalledProcessError):
        return None


def start_process(command, state, name):
    with (state / (name + '.log')).open('wb') as log:
        process = subprocess.Popen([str(v) for v in command], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(process.pid)
    require(identity, name + ' exited at startup')
    private_json(state / (name + '.pid'), dict(pid=process.pid, identity=identity))
    return process


def control(state, stop=False):
    results = {}
    for name in ('proxy', 'decode', 'prefill'):
        path = Path(state) / (name + '.pid')
        if not path.exists():
            continue
        record = read_private(path)
        pid = record['pid']
        require(isinstance(pid, int) and pid > 1 and pid != os.getpid(), 'invalid PID file')
        same = process_identity(pid) == record['identity']
        results[name] = 'running' if same else 'stopped'
        if stop and same:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 15
            while process_identity(pid) == record['identity'] and time.monotonic() < deadline:
                time.sleep(.1)
            if process_identity(pid) == record['identity']:
                os.kill(pid, signal.SIGKILL)
                deadline = time.monotonic() + 5
                while process_identity(pid) == record['identity'] and time.monotonic() < deadline:
                    time.sleep(.1)
                require(process_identity(pid) != record['identity'], f'PID {pid} has not stopped')
            results[name] = 'stopped'
    return results


def wait_health(url, path, process, timeout=180, token=''):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require(process.poll() is None, 'process exited during startup; inspect its private log')
        try:
            return http(url, path, token, timeout=1)
        except (OSError, ValueError):
            time.sleep(.2)
    raise Refusal('health check timed out')

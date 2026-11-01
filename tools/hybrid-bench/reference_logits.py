"""Pinned reference metadata and local reference production. Python standard library only."""
import argparse
import json
from pathlib import Path
import struct
import subprocess

from bench_common import Refusal, private_json, require, sha256


def reference_header(path, expected_model):
    with Path(path).open('rb') as source:
        require(source.read(8) == b'HYBLOG01', 'invalid reference magic')
        size_bytes = source.read(4)
        require(len(size_bytes) == 4, 'truncated reference header')
        size, = struct.unpack('<I', size_bytes)
        require(0 < size <= 8 * 1024**2, 'reference header exceeds limit')
        data = source.read(size)
        require(len(data) == size, 'truncated reference header')
        header = json.loads(data)
    require(header['format'] == 'f16-logsoftmax-le-v1', 'unsupported reference format')
    require(len(expected_model) == 64 and all(c in '0123456789abcdef' for c in expected_model)
            and header['model_sha256'] == expected_model, 'reference model hash mismatch')
    return header


def scoring_fixture(header, tokens, count, client, config, paths, reference):
    require(header['tokens'] == tokens, 'reference context tokens mismatch')
    require(header['n'] == len(tokens) and header['m'] == count and len(header['continuation']) == count, 'reference N/M mismatch')
    require(len(tokens) + count <= client['context'], 'reference continuation exceeds client context')
    return dict(tokens=tokens, continuation=header['continuation'], model_sha256=client['model_sha256'],
                max_ctx=client['context'], score_batch=header['score_batch'], deltas=paths,
                node_model_sha256=config.get('node_model_sha256', client['model_sha256']),
                hetero_quant=config.get('hetero_quant', False),
                node_layout_fingerprint=config.get('node_layout_fingerprint'),
                reference_logits=str(Path(reference).resolve()), reference_model_sha256=header['model_sha256'])


def produce(argv):
    parser = argparse.ArgumentParser(description='Produce reference logits locally from explicit context and continuation token IDs.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokens', required=True, help='JSON with tokens (context) and continuation arrays; no sampling')
    parser.add_argument('--output', required=True, help='new reference binary path')
    parser.add_argument('--result', required=True, help='new JSON measurement path')
    parser.add_argument('--scratch', required=True, help='existing private directory on disk')
    parser.add_argument('--build-dir', default='build')
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--score-batch', type=int, default=32)
    parser.add_argument('--max-ctx', type=int, default=16384)
    args = parser.parse_args(argv)
    scratch = Path(args.scratch)
    require(scratch.is_dir(), 'scratch must already exist')
    fixture = json.loads(Path(args.tokens).read_text())
    fixture.update(model_sha256=sha256(args.model), write_reference=str(Path(args.output).resolve()),
                   score_batch=args.score_batch, max_ctx=args.max_ctx)
    require(not any(k in fixture for k in ('reference_logits', 'deltas')), 'reference input must be local tokens only')
    path = scratch / 'reference-fixture.json'
    private_json(path, fixture)
    try:
        command = [str(Path(args.build_dir) / 'bin/llama-hybrid-score'), args.model, str(path), args.result, str(args.gpu_layers)]
        with (scratch / 'reference-score.log').open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            (scratch / 'reference-score.pid').write_text(str(process.pid) + '\n')
            try:
                code = process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)
                raise
        print(json.dumps(dict(command=command, exit_code=code)))
        return code
    finally:
        path.unlink()

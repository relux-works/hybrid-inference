#!/usr/bin/env python3
"""Restart-safe three-treatment RULER evaluation on existing hybrid services."""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tarfile
import uuid

from hybrid_eval import (Refusal, digest, http, model_identity, public_trace, read_private, require,
                         sha256, validate_identity, validate_partition)
from hybrid_bench import mode_session, stream_chat, verify_meter
from ruler_analysis import ARMS, write_analysis
from ruler_data import SOURCE, verify_data, verify_source
from ruler_generate import Generator, body_for, configurations, runtime_identity, score


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as output:
        output.write(json.dumps(value, sort_keys=True, indent=2)+'\n')
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def load_clients(state, partition):
    config, client = read_private(state/'proxy-config.json'), read_private(state/'client-info.json')
    pcfg, pclient = read_private(partition/'proxy-config.json'), read_private(partition/'client-info.json')
    node = json.loads((state/'node-info.json').read_text())
    require(config.get('benchmark') and pcfg.get('benchmark'), 'benchmark traces required')
    require(not config.get('hetero_quant') and not pcfg.get('hetero_quant'), 'RULER design requires homogeneous models')
    validate_identity(client['model_sha256'], client, node['model_sha256'], node, False)
    validate_partition(client, pclient, node)
    require(config['model_sha256'] == pcfg['model_sha256'] == client['model_sha256'], 'proxy model mismatch')
    require(config.get('node_model_sha256', config['model_sha256']) == node['model_sha256'], 'node model mismatch')
    require(client['backend'] == node['backend'], 'node and client backend differ')
    require(pcfg['prefill_url'] == config['prefill_url'], 'partition uses a different node')
    require(pcfg['decode_url'] != config['decode_url'] and pclient['proxy_url'] != client['proxy_url'], 'partition endpoints shared')
    for cfg, info in ((config, client), (pcfg, pclient)):
        props = http(cfg['decode_url'], '/props', cfg['decode_token'], timeout=10)
        require(sha256(props['model_path']) == info['model_sha256'], 'live decoder model mismatch')
        require(model_identity(props['model_path'])['quantization_type'] == info['quantization_type'], 'live quantization mismatch')
        info['template_sha256'] = digest({k: props.get(k) for k in ('chat_template', 'chat_template_tool_use', 'bos_token', 'eos_token')})
    require(client['template_sha256'] == pclient['template_sha256'], 'partition template differs')
    return config, client, pcfg, pclient, node


def run_example(spec, config, client, pcfg, pclient, node, seed, timeout, record=None):
    results = {}
    order = list(ARMS)
    random.Random(digest([seed, spec['task'], spec['index'], spec['depth']])).shuffle(order)
    releases = []
    def event(row):
        releases.append(row)
        if record:
            record(row)
    for arm in order:
        cfg, info = (pcfg, pclient) if arm == 'local_partition' else (config, client)
        mode = 'local' if arm == 'local_partition' else arm
        health = http(cfg['prefill_url'], '/v1/health', timeout=10)
        require(health['model_sha256'] == node['model_sha256'], 'live node model changed')
        body = dict(body_for(spec['input'], spec['answer_prefix']), temperature=0, seed=seed,
                    max_tokens=spec['max_tokens'])
        sid = 'ruler-'+uuid.uuid4().hex
        with mode_session(cfg, info, sid, releases, arm, record=event):
            returned, message, trace, _, seconds = stream_chat(info['proxy_url'], cfg['client_credential'],
                                                               body, mode, sid, timeout=timeout)
            require(returned == sid, 'proxy changed session identity')
            verify_meter(trace, 0, mode)
            require(not trace.get('fallback_reason'), 'RULER fallback invalidates comparison')
            require(mode != 'offload' or trace['remote_applied'] > 0, 'offload did not import state')
            slot = http(cfg['decode_url'], f"/slots/{trace['slot']}/info", cfg['decode_token'], timeout=timeout)
            tokens = slot['tokens']
            require(len(tokens) == trace['depth_after'] and trace['prompt_depth'] == spec['depth'], 'observed depth differs')
            prompt_hash = digest(tokens[:trace['prompt_depth']])
            require(prompt_hash == spec['prompt_sha256'], 'actual paired prompt differs from frozen input')
            generated = tokens[trace['prompt_depth']:]
            require(len(generated) == trace['generated_tokens'], 'sampled token count differs')
            require(not message.get('reasoning_content'), 'unexpected reasoning output')
            results[arm] = dict(answer=message['content'], score=score(spec['task'], spec['answers'], message['content']),
                                prompt_sha256=prompt_hash, sampled_sha256=digest(generated),
                                output_budget_reached=len(generated) >= spec['max_tokens'],
                                seconds=seconds, trace=public_trace(trace))
    return dict(**{k: v for k, v in spec.items() if k != 'input'}, results=results, order=order, releases=releases)


def archive_depth(root, manifest, expected):
    paths = sorted((root/'rows').glob('*.json'))
    require(len(paths) == expected, 'incomplete depth cannot be archived as complete')
    rows = [json.loads(p.read_text()) for p in paths]
    require(len({(r['task'], r['index']) for r in rows}) == expected, 'duplicate depth rows')
    (root/'rows.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows))
    write_analysis(root, rows, manifest['margin'])
    files = {name: (root/name).read_bytes() for name in
             ('rows.jsonl', 'analysis.json', 'analysis.md', 'per-example.csv', 'discordant-pairs.jsonl', 'manifest.json')}
    for name, value in manifest['data_provenance'].items():
        files[name] = (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
    files['SHA256SUMS'] = ''.join(hashlib.sha256(data).hexdigest()+'  '+name+'\n' for name, data in sorted(files.items())).encode()
    target = root/'complete.tar.gz'
    with tarfile.open(root/'complete.tar.gz.tmp', 'w:gz') as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o600
            archive.addfile(member, io.BytesIO(data))
    (root/'complete.tar.gz.tmp').replace(target)
    (root/'complete.tar.gz.sha256').write_text(sha256(target)+'  complete.tar.gz\n')
    atomic_json(root/'complete.json', dict(n=expected, manifest_sha256=sha256(root/'manifest.json'), archive_sha256=sha256(target)))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client-state', type=Path, required=True)
    parser.add_argument('--partition-client-state', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--input-lock-sha256', required=True, help='frozen GPU bootstrap lock hash, shared by every depth')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--samples-per-task', type=int, default=30)
    parser.add_argument('--tasks', nargs='+', choices=list(configurations()), default=list(configurations()))
    parser.add_argument('--seed', type=int, default=261102)
    parser.add_argument('--margin', type=float, default=.02)
    parser.add_argument('--request-timeout', type=int, default=7200)
    args = parser.parse_args()
    require(512 <= args.depth <= 1048576 and 1 <= args.samples_per_task <= 1000, 'invalid run size')
    require(len(set(args.tasks)) == len(args.tasks) and 0 <= args.margin < 1, 'duplicate tasks or invalid margin')
    require(1 <= args.request_timeout <= 86400, 'invalid request timeout')
    verify_source()
    require(sha256(args.data/'inputs.lock.json') == args.input_lock_sha256, 'pinned input lock differs')
    lock = verify_data(args.data)
    import nltk
    nltk.data.path.insert(0, str(args.data/'nltk_data'))
    config, client, pcfg, pclient, node = load_clients(args.client_state, args.partition_client_state)
    require(args.depth > max(client['batch'], pclient['batch']), 'prompt does not exercise both partitions')
    require(args.depth+128+2 <= min(client['context'], node['context']), 'insufficient service context')
    keys = ('backend', 'commit', 'model_sha256', 'quantization_type', 'layout_fingerprint', 'context',
            'batch', 'ubatch', 'threads', 'threads_batch', 'binaries', 'template_sha256')
    tools_root = Path(__file__).resolve().parents[1]
    source_paths = [p for folder in ('hybrid-eval', 'hybrid-proxy', 'hybrid-bench')
                    for p in (tools_root/folder).glob('*.py')]
    manifest = dict(depth=args.depth, samples_per_task=args.samples_per_task, tasks=args.tasks, seed=args.seed,
                    python_version=sys.version.split()[0],
                    margin=args.margin, generator=SOURCE, input_lock_sha256=sha256(args.data/'inputs.lock.json'),
                    protocol='one-turn RULER user input plus assistant answer-prefix continuation; exact rendered prompt depth',
                    source_hashes={str(p.relative_to(tools_root)): sha256(p) for p in source_paths},
                    packages=runtime_identity(),
                    clients={name: {k: info[k] for k in keys if k in info} for name, info in
                             (('client', client), ('partition', pclient), ('node', node))},
                    data_provenance={'inputs.lock.json': lock, 'downloads.json': json.loads((args.data/'downloads.json').read_text())})
    root = args.output/f'depth-{args.depth}'
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_lock = (root/'.lock').open('a')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        run_lock.close()
        require(False, 'another evaluator owns this depth directory')
    try:
        if (root/'manifest.json').exists():
            require(json.loads((root/'manifest.json').read_text()) == manifest, 'resume identity changed; use a new output directory')
        else:
            atomic_json(root/'manifest.json', manifest)
        if (root/'complete.json').exists():
            completed = json.loads((root/'complete.json').read_text())
            require(completed['manifest_sha256'] == sha256(root/'manifest.json') and
                    completed['archive_sha256'] == sha256(root/'complete.tar.gz'), 'completed depth archive changed')
            print(root/'complete.tar.gz')
            return 0
        for name in ('specs', 'rows'):
            (root/name).mkdir(exist_ok=True, mode=0o700)
        def tokenize(text):
            return http(config['decode_url'], '/tokenize', config['decode_token'], dict(content=text, add_special=False),
                        timeout=args.request_timeout)['tokens']
        def render_tokens(body):
            prompt = http(config['decode_url'], '/apply-template', config['decode_token'], body, timeout=args.request_timeout)['prompt']
            return http(config['decode_url'], '/tokenize', config['decode_token'], dict(content=prompt, add_special=True),
                        timeout=args.request_timeout)['tokens']
        for task in args.tasks:
            generator = None
            for index in range(args.samples_per_task):
                name = task+f'-{index:04d}.json'
                path = root/'specs'/name
                if path.exists():
                    spec = json.loads(path.read_text())
                    checksum = spec.pop('spec_sha256')
                    require(checksum == digest(spec), 'frozen example metadata changed')
                    require(spec['input_sha256'] == digest([spec['input'], spec['answer_prefix']]) and
                            spec['prompt_sha256'] == digest(render_tokens(body_for(spec['input'], spec['answer_prefix']))), 'frozen input changed')
                else:
                    if generator is None:
                        generator = Generator(task, args.data, tokenize, args.seed)
                    spec = generator.generate(index, args.depth, render_tokens)
                    atomic_json(path, dict(spec, spec_sha256=digest(spec)))
                result_path = root/'rows'/name
                if result_path.exists():
                    saved = json.loads(result_path.read_text())
                    checksum = saved.pop('row_sha256')
                    require(checksum == digest(saved) and saved['spec_sha256'] == digest(spec), 'saved pair differs')
                    continue
                def record(event):
                    with (root/'events.jsonl').open('a') as output:
                        output.write(json.dumps(event, sort_keys=True)+'\n')
                result = run_example(spec, config, client, pcfg, pclient, node, args.seed, args.request_timeout, record)
                result['spec_sha256'] = digest(spec)
                atomic_json(result_path, dict(result, row_sha256=digest(result)))
                print(f'{args.depth} {task} {index+1}/{args.samples_per_task}', flush=True)
        print(archive_depth(root, manifest, len(args.tasks)*args.samples_per_task))
        return 0
    finally:
        run_lock.close()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (Refusal, OSError, ValueError, KeyError) as error:
        print('refused: '+(str(error) if isinstance(error, Refusal) else type(error).__name__), file=sys.stderr)
        sys.exit(2)

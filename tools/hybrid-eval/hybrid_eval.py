#!/usr/bin/env python3
"""Seeded task-level patchwork cache evaluation, Python 3.10+ standard library only."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys
import tarfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hybrid-bench'))
from bench_common import Refusal, http, model_identity, read_private, require, sha256, validate_identity
from hybrid_bench import mode_session, stream_chat, verify_meter

TASK_TYPES = ('single_needle', 'multi_key', 'multi_value', 'variable_tracking', 'aggregation')
FILLER = 'The meadow is quiet. Smooth stones rest beside the path. Clouds cross the pale sky.\n'
SYSTEM = ('Read synthetic records supplied by the archive tool. Ignore neutral landscape filler. '
          'Assignments copy the current value, not a reference. Reply OK while loading material. '
          'For the final question give only the requested answer, without explanation or formatting.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def task_spec(kind, seed, index):
    require(kind in TASK_TYPES, 'unknown task type')
    rng = random.Random(f'patchwork-task-v1:{seed}:{index}:{kind}')
    values = rng.sample(range(100000, 999999), 12)
    key = 'item_' + str(rng.randrange(1000, 9999))
    if kind == 'single_needle':
        records = [f'FACT {key} = {values[0]}']
        question, answer = f'What is the value of {key}? Reply with its six digits only.', str(values[0])
    elif kind == 'multi_key':
        records = [f'FACT {key}_{i} = {values[i]}' for i in range(6)]
        target = rng.randrange(6)
        question, answer = f'What is the value of {key}_{target}? Reply with its six digits only.', str(values[target])
    elif kind == 'multi_value':
        records = [f'VALUE {key} += {values[i]}' for i in range(4)]
        question = f'List all values added to {key}, in record order. Reply with comma-separated six-digit values only.'
        answer = ','.join(str(v) for v in values[:4])
    elif kind == 'variable_tracking':
        names = ['name_a', 'name_b', 'name_c', 'name_d', 'name_e']
        rng.shuffle(names)
        a, b, c, d, e = names
        records = [f'SET {a} = {values[0]}', f'SET {b} = {values[1]}', f'COPY {c} = {a}',
                   f'COPY {b} = {c}', f'SET {a} = {values[2]}', f'COPY {d} = {b}',
                   f'SET {c} = {values[3]}', f'COPY {e} = {a}']
        question = f'After all assignments, which names hold {values[0]}? Reply with all names in alphabetical order, comma-separated.'
        answer = ','.join(sorted([b, d]))
    else:
        amounts = rng.sample(range(11, 90), 6)
        records = [f'ADD {key} {amount}' for amount in amounts]
        question = f'What is the sum of every ADD amount for {key}? Reply with decimal digits only.'
        answer = str(sum(amounts))
    return dict(kind=kind, seed=seed, index=index, records=records, question=question, answer=answer)


def generate(kind, seed, index, depth, position, tokenize):
    """Fit tool material to depth +/- 8 decoder tokens; record actual record offsets."""
    require(128 <= depth <= 1048576, 'depth must be in 128..1048576')
    require(0 <= position <= 1, 'position must be in 0..1')
    spec = task_spec(kind, seed, index)
    records = spec['records']
    # Rotate a set of spread positions, preserving the order needed by copy/aggregation tasks.
    fractions = [position] if len(records) == 1 else sorted((position + i / len(records)) % 1 for i in range(len(records)))
    def material(repeats):
        slots = [round(f * repeats) for f in fractions]
        parts, offsets, cursor = [], [], 0
        for record, slot in zip(records, slots):
            parts.append(FILLER * (slot - cursor))
            offsets.append(sum(map(len, parts)))
            parts.append(record + '\n')
            cursor = slot
        parts.append(FILLER * (repeats - cursor))
        return ''.join(parts), offsets
    low, high = 0, max(1, depth // max(1, len(tokenize(FILLER))) + 64)
    while len(tokenize(material(high)[0])) <= depth:
        high *= 2
    while low < high:
        mid = (low + high + 1) // 2
        text, _ = material(mid)
        if len(tokenize(text)) <= depth:
            low = mid
        else:
            high = mid - 1
    text, offsets = material(low)
    require(len(tokenize(text)) <= depth, 'records exceed requested depth')
    # A whitespace word is a small pad on supported tokenizers; never assume one token per repeat.
    while len(tokenize(text + ' meadow')) <= depth:
        text += ' meadow'
    count = len(tokenize(text))
    require(depth - 8 <= count <= depth, 'tokenizer cannot fit requested depth within eight tokens')
    spec.update(material=text, requested_depth=depth, material_tokens=count, position=position,
                record_token_offsets=[len(tokenize(text[:offset])) for offset in offsets],
                material_sha256=hashlib.sha256(text.encode()).hexdigest())
    return spec


def canonical(answer):
    return ','.join(part.strip() for part in answer.strip().split(','))


def score(expected, observed):
    # Full output equality rejects substring matches, missing values, duplicates and extra prose.
    return canonical(expected) == canonical(observed)


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def tool_messages(text, index):
    call = 'archive_' + str(index)
    return [dict(role='user', content='Load this archive part. Reply OK.'),
            dict(role='assistant', content='', tool_calls=[dict(id=call, type='function',
                 function=dict(name='archive', arguments='{}'))]),
            dict(role='tool', tool_call_id=call, content=text)]


def check_route(trace, previous, mode, stage):
    verify_meter(trace, previous, mode)
    require(not trace.get('fallback_reason'), 'remote fallback invalidates task comparison')
    if stage == 'ingest' and mode in ('hybrid', 'offload'):
        require(trace['remote_applied'] > 0, 'hybrid ingest did not import remote state; calibrate proxy policy')
    if stage == 'answer':
        require(trace['mode'] == 'local' and trace['remote_applied'] == 0, 'question must execute locally')


def public_trace(trace):
    keys = ('mode', 'path', 'depth_before', 'prompt_depth', 'depth_after', 'new_tokens', 'generated_tokens',
            'local_evaluated', 'remote_applied', 'prefilled', 'replayed_prefix', 'rollback_n',
            'wire_bytes', 'server_prefill_ms', 'decode_ms', 'fallback_reason')
    return {key: trace[key] for key in keys if key in trace}


def run_task(spec, mode, config, client, node, max_tokens, seed, timeout, release_rows):
    control = mode == 'local_split'
    wire_mode = 'local' if control else mode
    text = spec['material']
    split = text.rfind('\n', 0, len(text) // 2) + 1
    pieces = [text[:split], text[split:]] if control else [text]
    require(all(pieces), 'empty tool split')
    history = [dict(role='system', content=SYSTEM)]
    previous, turns = 0, []
    sid = 'eval-' + uuid.uuid4().hex
    def decode(path, data=None):
        return http(config['decode_url'], path, config['decode_token'], data, timeout=timeout)
    def prompt_tokens(messages):
        prompt = decode('/apply-template', dict(messages=messages, chat_template_kwargs=dict(enable_thinking=False)))['prompt']
        return decode('/tokenize', dict(content=prompt, add_special=True))['tokens']
    with mode_session(config, client, sid, release_rows, mode):
        for index in range(len(pieces) + 1):
            answering = index == len(pieces)
            stage, requested_mode = ('answer', 'local') if answering else ('ingest', wire_mode)
            messages = history + ([dict(role='user', content=spec['question'])] if answering else tool_messages(pieces[index], index))
            prompt = prompt_tokens(messages)
            budget = max_tokens if answering else 4
            require(len(prompt) + budget + 2 <= min(client['context'], node['context']), 'task exceeds configured context; increase --ctx')
            health = http(config['prefill_url'], '/v1/health', timeout=5)
            require(health['model_sha256'] == node['model_sha256'], 'live node model identity changed')
            if config.get('hetero_quant'):
                require(health.get('layout_fingerprint') == node['layout_fingerprint'], 'live node layout changed')
            body = dict(messages=messages, max_tokens=budget, temperature=0, seed=seed,
                        chat_template_kwargs=dict(enable_thinking=False))
            if not answering:
                body['grammar'] = 'root ::= "OK"'
            returned_sid, message, trace, _, total = stream_chat(client['proxy_url'], config['client_credential'],
                                                               body, requested_mode, sid, timeout=timeout)
            require(returned_sid == sid, 'proxy changed explicit session identity')
            check_route(trace, previous, requested_mode, stage)
            info = decode(f"/slots/{trace['slot']}/info")
            require(len(info['tokens']) == trace['depth_after'], 'evaluated sequence length differs from trace')
            actual_prompt = info['tokens'][:trace['prompt_depth']]
            generated = info['tokens'][trace['prompt_depth']:trace['depth_after']]
            require(len(generated) == trace['generated_tokens'], 'sampled token sequence unavailable')
            turns.append(dict(stage=stage, seconds=total, rendered_prompt_tokens=len(prompt), **public_trace(trace)))
            if not answering:
                require(message['content'] == 'OK' and not message.get('reasoning_content'), 'ingest acknowledgement differs from fixed OK')
            history, previous = messages + [message], trace['depth_after']
    return dict(answer=message['content'], correct=score(spec['answer'], message['content']), turns=turns,
                answer_tokens=generated, answer_prompt_sha256=digest(actual_prompt),
                output_budget_reached=len(generated) >= max_tokens,
                seconds=sum(turn['seconds'] for turn in turns))


def comparison(local, other):
    return dict(answer_agreement=canonical(local['answer']) == canonical(other['answer']),
                first_differing_token=first_difference(local['answer_tokens'], other['answer_tokens']),
                prompt_identical=local['answer_prompt_sha256'] == other['answer_prompt_sha256'])


def summary(rows):
    lines = ['# Patchwork task quality', '',
             'Accuracy is full-answer exact match after outer/comma whitespace normalization.',
             'Token difference is zero-based on actual sampled IDs (including EOS); null means identical.',
             'local_split is a framing-confounded local noise reference, not a calibrated numerical threshold.', '',
             '| Depth | Task | Mode | N | Correct | Accuracy | Agree/local | Token disagreements | Seconds |',
             '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    results = [row for row in rows if row.get('stage') == 'task']
    for depth, kind, mode in sorted({(r['requested_depth'], r['kind'], m) for r in results for m in r['results']}):
        group = [r['results'][mode] for r in results if r['requested_depth'] == depth and r['kind'] == kind]
        correct = sum(r['correct'] for r in group)
        comparisons = [r['vs_local'] for r in group if 'vs_local' in r]
        agreement = str(sum(r['answer_agreement'] for r in comparisons)) + '/' + str(len(comparisons)) if comparisons else '-'
        differences = str(sum(r['first_differing_token'] is not None for r in comparisons)) if comparisons else '-'
        lines.append(f'| {depth} | {kind} | {mode} | {len(group)} | {correct} | {correct / len(group):.3f} | {agreement} | {differences} | {sum(r["seconds"] for r in group):.2f} |')
    lines += ['', 'Execution: ' + rows[-1]['status'] + '; exit ' + str(rows[-1]['exit_code']) + '.',
              'Execution success does not assert task accuracy or noise equivalence. Inspect each denominator and disagreement.']
    return '\n'.join(lines) + '\n'


def bundle(root, name, rows, client, node):
    # Explicit allowlist excludes paths, addresses, credentials, session IDs, inputs and native state.
    info_keys = ('backend', 'commit', 'model_sha256', 'quantization_type', 'layout_fingerprint',
                 'context', 'batch', 'ubatch', 'threads', 'threads_batch', 'binaries')
    public = {name + '.jsonl': ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows).encode(),
              name + '.md': summary(rows).encode()}
    for label, info in [('client', client), ('node', node)]:
        public[label + '-info.json'] = (json.dumps({k: info[k] for k in info_keys if k in info}, sort_keys=True, indent=2) + '\n').encode()
    public['SHA256SUMS'] = ''.join(hashlib.sha256(data).hexdigest() + '  ' + filename + '\n'
                                   for filename, data in sorted(public.items())).encode()
    archive = root / (name + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as target:
        for filename, data in public.items():
            member = tarfile.TarInfo(filename)
            member.size, member.mode = len(data), 0o600
            target.addfile(member, io.BytesIO(data))
    for filename, data in public.items():
        if filename != 'SHA256SUMS':
            (root / filename).write_bytes(data)
    (root / (archive.name + '.sha256')).write_text(sha256(archive) + '  ' + archive.name + '\n')
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client-state', required=True)
    parser.add_argument('--node-info')
    parser.add_argument('--output', required=True)
    parser.add_argument('--label', default='quality')
    parser.add_argument('--depths', type=int, nargs='+', default=[8192, 32768, 65536, 131072])
    parser.add_argument('--positions', type=float, nargs='+', default=[0.1, 0.5, 0.9])
    parser.add_argument('--tasks', choices=TASK_TYPES, nargs='+', default=list(TASK_TYPES))
    parser.add_argument('--modes', choices=['local', 'hybrid', 'offload'], nargs='+', default=['local', 'hybrid', 'offload'])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--max-tokens', type=int, default=48)
    parser.add_argument('--request-timeout', type=int, default=7200)
    args = parser.parse_args()
    require(re.fullmatch('[a-zA-Z0-9_-]{1,64}', args.label), 'invalid label')
    require(all(128 <= n <= 1048576 for n in args.depths), 'depth must be in 128..1048576')
    require(all(0 <= p <= 1 for p in args.positions), 'position must be in 0..1')
    require(1 <= args.repeats <= 100 and 1 <= args.max_tokens <= 256 and 1 <= args.request_timeout <= 86400, 'invalid run bounds')
    require('local' in args.modes and len(set(args.modes)) == len(args.modes), 'include local exactly once as the reference')
    state, root = Path(args.client_state), Path(args.output)
    require(root.is_dir(), 'output directory must exist')
    config = read_private(state / 'proxy-config.json')
    client = read_private(state / 'client-info.json')
    node = json.loads(Path(args.node_info or state / 'node-info.json').read_text())
    require(config.get('benchmark'), 'proxy must enable benchmark traces')
    require(client['model_sha256'] == config['model_sha256'], 'client metadata model identity mismatch')
    require(node['model_sha256'] == config.get('node_model_sha256', config['model_sha256']), 'node metadata model identity mismatch')
    validate_identity(client['model_sha256'], client, node['model_sha256'], node, config.get('hetero_quant', False))
    require(client.get('quantization_type') and node.get('quantization_type'), 'quantization metadata required')
    name = 'hybrid-eval-' + args.label + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    rows = [dict(stage='run', generator_version=1, evaluator_sha256=sha256(__file__), seed=args.seed,
                 depths=args.depths, positions=args.positions, tasks=args.tasks, repeats=args.repeats,
                 modes=args.modes + ['local_split'], max_tokens=args.max_tokens,
                 noise_control='identical material in two tool turns; framing differs',
                 comparison_calibrated=False, question_mode='local')]
    code = 0
    try:
        def tokenize(text):
            return http(config['decode_url'], '/tokenize', config['decode_token'],
                        dict(content=text, add_special=False), timeout=args.request_timeout)['tokens']
        http(config['decode_url'], '/health', config['decode_token'], timeout=5)
        props = http(config['decode_url'], '/props', config['decode_token'], timeout=5)
        require(sha256(props['model_path']) == client['model_sha256'], 'live decoder model file hash differs from client metadata')
        live_client = model_identity(props['model_path'])
        require(live_client['quantization_type'] == client['quantization_type'], 'live decoder quantization differs from client metadata')
        rows[0].update(client_identity_source='client-info plus live decoder props and model-file SHA-256',
                       node_identity_source='node-info quantization bound to live health model SHA-256',
                       live_client_quantization_type=props['model_ftype'])
        for depth in args.depths:
            for kind in args.tasks:
                for repeat in range(args.repeats):
                    for position in args.positions:
                        spec = generate(kind, args.seed, repeat, depth, position, tokenize)
                        results = {}
                        releases = []
                        try:
                            for mode in ['local'] + [m for m in args.modes if m != 'local'] + ['local_split']:
                                results[mode] = run_task(spec, mode, config, client, node, args.max_tokens,
                                                         args.seed, args.request_timeout, releases)
                                if mode != 'local':
                                    results[mode]['vs_local'] = comparison(results['local'], results[mode])
                                    if mode != 'local_split':
                                        require(results[mode]['vs_local']['prompt_identical'], 'mode prompts differ; invalid paired comparison')
                            for result in results.values():
                                result['answer_tokens_sha256'] = digest(result.pop('answer_tokens'))
                            rows.append(dict(stage='task', **{k: v for k, v in spec.items() if k not in ('material', 'records')},
                                             results=results))
                        finally:
                            rows.extend(releases)
    except KeyboardInterrupt:
        code = 130
        rows.append(dict(stage='error', reason='KeyboardInterrupt'))
    except Exception as error:
        code = 2
        rows.append(dict(stage='error', reason=str(error) if isinstance(error, Refusal) else type(error).__name__))
    rows.append(dict(stage='summary', status='interrupted' if code == 130 else 'aborted' if code else 'executed',
                     exit_code=code, tasks_executed=sum(r['stage'] == 'task' for r in rows)))
    for row in rows:
        row.update(client_model_sha256=client['model_sha256'], node_model_sha256=node['model_sha256'],
                   client_quantization_type=client['quantization_type'], node_quantization_type=node['quantization_type'],
                   hetero_quant=config.get('hetero_quant', False))
    archive = bundle(root, name, rows, client, node)
    print(str(archive))
    print('sha256 ' + sha256(archive))
    return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (Refusal, OSError, ValueError, KeyError) as error:
        print('refused: ' + (str(error) if isinstance(error, Refusal) else type(error).__name__), file=sys.stderr)
        sys.exit(2)

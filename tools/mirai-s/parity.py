#!/usr/bin/env python3
"""Greedy token parity between the source fork and this port; collect JSON, then compare."""
# Adapted from the orchestrator's mirai_parity.py; reject incomplete evidence before admitting study arms.
import argparse
import json
import sys
import urllib.request
from pathlib import Path

SHORT = [
    'The history of the printing press begins',
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    '\u0420\u0430\u0441\u0441\u043a\u0430\u0436\u0438 \u043a\u0440\u0430\u0442\u043a\u043e, \u043f\u043e\u0447\u0435\u043c\u0443 \u043d\u0435\u0431\u043e \u0433\u043e\u043b\u0443\u0431\u043e\u0435.',
    '-- Find the ten customers with the highest total order value\nSELECT',
    'Q: What is 17 * 23 + 144 / 12?\nA:',
    '{"name": "Alice", "skills": [',
    'User: Write a haiku about autumn rain.\nAssistant:',
    'Prove that the sum of the first n odd numbers is n squared. Proof:',
]
LONG = [4096, 12288]
CASES = [(f'short{i}', 48) for i in range(len(SHORT))] + [(f'long{n}', 32) for n in LONG]


def http(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.loads(response.read())


def token_ids(value):
    return isinstance(value, list) and bool(value) and all(type(t) is int and t >= 0 for t in value)


def validate(items, generated=True):
    if not isinstance(items, list) or len(items) != len(CASES):
        raise ValueError('expected all 10 parity cases')
    for item, (name, count) in zip(items, CASES, strict=True):
        if item['name'] != name or item['n_predict'] != count:
            raise ValueError(f'wrong parity case or generation budget: {name}')
        if not token_ids(item['prompt']):
            raise ValueError(f'missing or invalid prompt tokens: {name}')
        if name.startswith('long') and len(item['prompt']) != int(name[4:]):
            raise ValueError(f'wrong book prompt length: {name}')
        if generated and (not token_ids(item.get('tokens')) or len(item['tokens']) > count):
            raise ValueError(f'missing or invalid generated tokens: {name}')


def tokenize(url, text):
    return http(url + '/tokenize', {'content': text, 'add_special': True})['tokens']


def generate(url, tokens, n_predict):
    response = http(url + '/completion', dict(prompt=tokens, n_predict=n_predict, temperature=0, top_k=1, seed=0,
                                             cache_prompt=False, n_probs=5, return_tokens=True))
    probs = response.get('completion_probabilities') or []
    top = [[(c.get('id'), c.get('logprob')) for c in (p.get('top_logprobs') or [])] for p in probs]
    return dict(tokens=response.get('tokens'), top=top, prompt_n=response.get('timings', {}).get('prompt_n'))


def collect(args):
    items = []
    if args.reuse:
        reuse = json.loads(Path(args.reuse).read_text())
        validate(reuse['items'])
        for item in reuse['items']:
            own = tokenize(args.url, item['text']) if item.get('text') is not None else None
            items.append(dict(name=item['name'], text=item.get('text'), prompt=item['prompt'], n_predict=item['n_predict'],
                              own_tokenization_equal=(own == item['prompt']) if own is not None else None))
    else:
        for i, text in enumerate(SHORT):
            items.append(dict(name=f'short{i}', text=text, prompt=tokenize(args.url, text), n_predict=48))
        books = sorted(Path(args.text_dir).glob('*.txt'), key=lambda p: p.stat().st_size, reverse=True)
        if not books:
            raise ValueError('no local book text found')
        book = books[0]
        body = tokenize(args.url, book.read_text(errors='replace')[:400000])
        for n in LONG:
            if len(body) < n:
                raise ValueError(f'book {book.name} has only {len(body)} tokens; need {n}')
            items.append(dict(name=f'long{n}', text=None, prompt=body[:n], n_predict=32, source=book.name))
    validate(items, generated=False)
    for item in items:
        item.update(generate(args.url, item['prompt'], item['n_predict']))
        print(item['name'], len(item['prompt']), 'prompt tokens,', len(item['tokens'] or []), 'generated', flush=True)
    validate(items)
    Path(args.out).write_text(json.dumps(dict(side=args.side, items=items), ensure_ascii=True) + '\n')


def compare(args):
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    validate(a['items'])
    validate(b['items'])
    ok = True
    for x, y in zip(a['items'], b['items'], strict=True):
        if x['prompt'] != y['prompt']:
            print(x['name'], 'PROMPT TOKENS DIFFER')
            ok = False
            continue
        if y.get('own_tokenization_equal') is False:
            print(x['name'], 'note: tokenizers differ; the shared token IDs were used')
        if x['tokens'] == y['tokens']:
            print(x['name'], 'identical', len(x['tokens']), 'tokens')
            continue
        ok = False
        k = next((i for i, (s, t) in enumerate(zip(x['tokens'], y['tokens'])) if s != t), min(len(x['tokens']), len(y['tokens'])))

        def margin(side):
            top = side.get('top', [])
            row = top[k] if k < len(top) else []
            return round(row[0][1] - row[1][1], 5) if len(row) > 1 and None not in (row[0][1], row[1][1]) else None

        print(x['name'], f'diverge at generated token {k}: {a["side"]} {x["tokens"][k:k + 3]} vs '
              f'{b["side"]} {y["tokens"][k:k + 3]}; top-2 logprob margin {a["side"]} {margin(x)}, {b["side"]} {margin(y)}')
    print('ALL IDENTICAL' if ok else 'NOT ALL IDENTICAL')
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--url', required=True)
    c.add_argument('--side', required=True)
    c.add_argument('--out', required=True)
    source = c.add_mutually_exclusive_group(required=True)
    source.add_argument('--reuse')
    source.add_argument('--text-dir')
    d = sub.add_parser('compare')
    d.add_argument('--a', required=True)
    d.add_argument('--b', required=True)
    args = parser.parse_args()
    try:
        if args.cmd == 'collect':
            collect(args)
            return 0
        return compare(args)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f'PARITY FAILED: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

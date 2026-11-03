#!/usr/bin/env python3
"""Preregistered paired RULER score equivalence, CPU and standard library only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import NormalDist
import sys

from hybrid_eval import Refusal, digest, require
from ruler_analysis import ARMS, analyze as analyze_text, paired_cells, wilson

SEED = 261102
RESAMPLES = 10000
MARGIN = .02
ALPHA = .025
TASKS = tuple(f'niah_{kind}_{i}' for kind in ('single', 'multikey') for i in range(1, 4)) + (
    'niah_multivalue', 'niah_multiquery', 'vt', 'cwe', 'fwe', 'qa_1', 'qa_2')
PAIRS = (('offload', 'local'), ('local_partition', 'local'), ('bf16', 'local'), ('bf16', 'offload'))


def newcombe10(e, f, g, h, tail=ALPHA):
    """Cells: both correct, left only, right only, both incorrect; tail is one-sided."""
    # Newcombe (1998), Statistics in Medicine 17:2635-2650, section 5, method 10.
    require(all(type(x) is int and x >= 0 for x in (e, f, g, h)), 'invalid paired counts')
    n = e+f+g+h
    require(n > 0 and 0 < tail <= .5, 'invalid size or tail')
    z = NormalDist().inv_cdf(1-tail)
    p, q = (e+f)/n, (e+g)/n
    lp, up = wilson(e+f, n, z)
    lq, uq = wilson(e+g, n, z)
    denominator = math.sqrt((e+f)*(g+h)*(e+g)*(f+h))
    covariance = e*h-f*g
    # Method 10 corrects positive phi, not the marginal Wilson limits (method 9).
    if covariance > 0:
        covariance = max(covariance-n/2, 0)
    phi = covariance/denominator if denominator else 0
    def radius(a, b):
        return math.sqrt(max(0, a*a-2*phi*a*b+b*b))
    return [max(-1, p-q-radius(p-lp, uq-q)), min(1, p-q+radius(up-p, q-lq))]


def quantile(values, probability):
    position = (len(values)-1)*probability
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high]-values[low])*(position-low)


def bootstrap_distribution(differences, seed=SEED):
    require(bool(differences), 'empty paired bootstrap')
    rng = random.Random(seed)
    n = len(differences)
    return sorted(math.fsum(rng.choices(differences, k=n))/n for _ in range(RESAMPLES))


def inside(interval, margin=MARGIN):
    return interval[0] > -margin and interval[1] < margin


def invert_tost(interval, margin=MARGIN):
    """Invert nested central intervals at coverage 1-2*tail; never use McNemar."""
    if not inside(interval(.5), margin):
        return 1.0
    low, high = 1e-15, .5
    if inside(interval(low), margin):
        return low
    for _ in range(60):
        mid = (low+high)/2
        if inside(interval(mid), margin):
            high = mid
        else:
            low = mid
    return high


def score_pair(left, right):
    require(len(left) == len(right) and bool(left), 'mismatched example sets')
    require(all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1
                for x in left+right), 'invalid score')
    differences = [a-b for a, b in zip(left, right)]
    result = dict(n=len(left), difference=math.fsum(differences)/len(left))
    if all(x in (0, 1) for x in left+right):
        cells = paired_cells([bool(x) for x in left], [bool(x) for x in right])
        counts = [cells[k] for k in ('both_true', 'left_only', 'right_only', 'both_false')]
        interval = lambda tail: newcombe10(*counts, tail=tail)
        result.update(method='newcombe10', cells=cells)
    else:
        distribution = bootstrap_distribution(differences)
        interval = lambda tail: [quantile(distribution, tail), quantile(distribution, 1-tail)]
        result.update(method='paired_percentile_bootstrap', resamples=RESAMPLES, bootstrap_seed=SEED,
                      bootstrap_tail_resolution=1/RESAMPLES)
    ci = interval(ALPHA)
    inverted = invert_tost(interval)
    limited = result['method'] == 'paired_percentile_bootstrap' and inverted < 1/RESAMPLES
    # Do not report more tail precision than 10000 bootstrap resamples can support.
    result.update(interval95=ci, tost_p=max(inverted, 1/RESAMPLES) if limited else inverted,
                  tost_p_inverted=inverted, tost_p_resolution_limited=limited,
                  equivalent_unadjusted=inside(ci))
    return result


def holm(pvalues):
    require(len(pvalues) == 4 and all(0 <= p <= 1 for p in pvalues), 'Holm requires four p-values')
    adjusted = [1.0]*4
    running = 0.0
    for rank, index in enumerate(sorted(range(4), key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (4-rank)*pvalues[index]))
        adjusted[index] = running
    return adjusted


def validate_rows(rows, manifest):
    require(set(manifest['tasks']) == set(TASKS) and len(manifest['tasks']) == 13 and
            manifest['samples_per_task'] == 30, 'expected 13 tasks x 30 examples')
    require(manifest['seed'] == SEED, 'changed seed in manifest')
    expected = {(task, i) for task in TASKS for i in range(30)}
    seen = set()
    arms = None
    for row in rows:
        key = (row['task'], row['index'])
        require(key not in seen, 'duplicate paired example')
        seen.add(key)
        require(row['depth'] == manifest['depth'], 'depth differs from manifest')
        require(row['seed'] == manifest['seed'], 'changed seed in row')
        results = row['results']
        current = set(results)
        require(current in (set(ARMS), set(ARMS) | {'bf16'}), 'missing or unknown arms')
        if arms is None:
            arms = current
        require(current == arms, 'mismatched example sets across arms')
        prompts = [v['prompt_sha256'] for v in results.values()]
        require(all(isinstance(p, str) and p for p in prompts) and len(set(prompts)) == 1,
                'paired prompts differ')
        for value in results.values():
            require(value.get('seed', row['seed']) == row['seed'], 'changed seed in arm')
            score = value['score']
            require(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1
                    and isinstance(value['answer'], str), 'invalid response')
        if 'row_sha256' in row:
            require(row['row_sha256'] == digest({k: v for k, v in row.items() if k != 'row_sha256'}),
                    'row checksum differs')
    require(seen == expected and len(rows) == 390, 'missing or unexpected rows')
    return sorted(rows, key=lambda r: (r['task'], r['index']))


def load_depth(path):
    manifest_path = path.with_name('manifest.json')
    require(manifest_path.is_file(), 'manifest.json required to verify seeds and completeness')
    manifest = json.loads(manifest_path.read_text())
    source = path
    if path.suffix == '.csv':
        # The evaluator CSV omits seed and prompt identities; require its source rows.
        source = path.with_name('rows.jsonl')
        require(source.is_file(), 'CSV requires companion rows.jsonl for seed and prompt verification')
    rows = validate_rows([json.loads(line) for line in source.read_text().splitlines()], manifest)
    if path.suffix == '.csv':
        with path.open(newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
        require(len(csv_rows) == len(rows), 'missing CSV rows')
        by_key = {(r['task'], r['index']): r for r in rows}
        seen = set()
        for item in csv_rows:
            key = (item['task'], int(item['index']))
            require(key in by_key and key not in seen, 'mismatched example sets in CSV')
            seen.add(key)
            row = by_key[key]
            require(int(item['depth']) == row['depth'], 'CSV depth differs')
            for arm, value in row['results'].items():
                require(float(item[arm+'_score']) == value['score'] and item[arm+'_answer'] == value['answer'],
                        'CSV response differs from source rows')
    evidence = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in {path, source, manifest_path}}
    return rows, manifest, dict(sorted(evidence.items()))


def analyze_family(paths, family, depths):
    require(len(depths) == len(set(depths)) == 4 and all(d > 0 for d in depths), 'declare four distinct depths')
    primary, text_rows, descriptives, sources = [], [], [], []
    seen = set()
    model = None
    for path in paths:
        rows, manifest, hashes = load_depth(path)
        depth = manifest['depth']
        require(depth in depths and depth not in seen, 'unexpected or duplicate depth')
        seen.add(depth)
        identity = manifest['clients']['client']['model_sha256']
        require(isinstance(identity, str) and bool(identity), 'missing model identity')
        if model is None:
            model = identity
        require(identity == model, 'different models must use separate families')
        sources.append(dict(depth=depth, sha256=hashes))
        text_rows.extend(dict(row, results={a: row['results'][a] for a in ARMS}) for row in rows)
        for task in ['ALL'] + sorted(TASKS):
            group = [r for r in rows if task == 'ALL' or r['task'] == task]
            descriptives.append(dict(depth=depth, task=task, n=len(group),
                means={arm: math.fsum(r['results'][arm]['score'] for r in group)/len(group)
                       for arm in rows[0]['results']}))
        for left, right in PAIRS:
            if left not in rows[0]['results']:
                continue
            result = score_pair([r['results'][left]['score'] for r in rows],
                                [r['results'][right]['score'] for r in rows])
            result.update(depth=depth, pair=left+'-'+right,
                          registration='observed before registration' if depth in (8192, 32768) and left != 'bf16'
                          else 'registered before data')
            primary.append(result)
    require(bool(primary), 'no completed depths')
    _, secondary = analyze_text(text_rows)
    for left, right in PAIRS:
        pair = left+'-'+right
        group = {r['depth']: r for r in primary if r['pair'] == pair}
        adjusted = holm([group[d]['tost_p'] if d in group else 1.0 for d in depths])
        missing = [d for d in depths if d not in group]
        for depth, p in zip(depths, adjusted):
            if depth in group:
                group[depth].update(holm_p=p, equivalent_holm=p < ALPHA and group[depth]['equivalent_unadjusted'],
                                    provisional=bool(missing), pending_depths=missing)
    code = Path(__file__)
    return dict(family=family, model_sha256=model, depths=list(depths), margin=MARGIN, alpha_per_side=ALPHA,
                analysis_sha256=hashlib.sha256(code.read_bytes()).hexdigest(), python_version=sys.version,
                secondary_analysis_sha256=hashlib.sha256(code.with_name('ruler_analysis.py').read_bytes()).hexdigest(),
                interval_rule='newcombe10 if all compared scores binary; paired percentile bootstrap otherwise',
                pending_policy='p=1 for missing depths in each four-depth arm-pair family; provisional decisions',
                sources=sorted(sources, key=lambda r: r['depth']),
                primary=sorted(primary, key=lambda r: (r['depth'], r['pair'])),
                secondary=secondary, descriptives=descriptives)


def markdown(result):
    lines = [f"# {result['family']}: paired RULER equivalence", '',
             'Differences and intervals are percentage points. Margin: +/-2 pp; alpha per side: 0.025.',
             'Holm is separate for each arm pair across four depths. Pending depths enter as p=1; decisions are provisional.', '',
             '| Depth | Pair | Difference | 95% interval | Method | TOST p | Holm p | Equivalent (raw / Holm) | Registration |',
             '|---:|---|---:|---|---|---:|---:|---|---|']
    for r in result['primary']:
        ci = ', '.join(f'{100*x:+.4f}' for x in r['interval95'])
        verdict = f"{r['equivalent_unadjusted']} / {r['equivalent_holm']}"
        if r['provisional']:
            verdict += ' (provisional)'
        lines.append(f"| {r['depth']} | {r['pair']} | {100*r['difference']:+.4f} | [{ci}] | {r['method']} | {r['tost_p']:.6g} | {r['holm_p']:.6g} | {verdict} | {r['registration']} |")
    lines += ['', '## Secondary: answer text (unchanged ruler_analysis.py)', '',
              '| Depth | Offload changed / n | Partition changed / n | McNemar p | Excess 95% (pp) | Text equivalent +/-2 pp |',
              '|---:|---:|---:|---:|---|---|']
    for r in result['secondary']:
        if r['task'] == 'ALL':
            ci = ', '.join(f'{100*x:+.4f}' for x in r['excess_conservative95'])
            lines.append(f"| {r['depth']} | {r['offload_changed']}/{r['n']} | {r['partition_changed']}/{r['n']} | {r['mcnemar_p']:.6g} | [{ci}] | {r['equivalent_at_margin']} |")
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rows', type=Path, nargs='+')
    parser.add_argument('--family', required=True)
    parser.add_argument('--depths', type=int, nargs=4, default=[8192, 32768, 65536, 131072])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze_family(args.rows, args.family, args.depths)
    except (Refusal, OSError, ValueError, KeyError, TypeError) as error:
        print('REFUSED: '+str(error), file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'equivalence.json').write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n')
    (args.output/'equivalence.md').write_text(markdown(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())

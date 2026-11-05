#!/usr/bin/env python3
"""Paired RULER analysis; Python standard library only."""
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from hybrid_eval import require

ARMS = ('local', 'offload', 'local_partition')


def wilson(k, n, z=1.959963984540054):
    require(n > 0 and 0 <= k <= n, 'invalid binomial counts')
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0.0, center-radius), min(1.0, center+radius)]


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c)+1)) / 2**n)


def binomial_cdf(k, n, p):
    if p == 0:
        return 1.0
    if p == 1:
        return float(k == n)
    return min(1.0, math.fsum(math.exp(math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)
                        + i*math.log(p)+(n-i)*math.log1p(-p)) for i in range(k+1)))


def upper_exact(k, n, alpha=.025):
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(70):
        mid = (lo+hi)/2
        if binomial_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def excess_interval(b, c, n):
    # Four one-sided tails at .0125 give a conservative 95% two-sided interval for the cell difference.
    alpha = .0125
    lower_b = 0.0 if b == 0 else 1-upper_exact(n-b, n, alpha)
    lower_c = 0.0 if c == 0 else 1-upper_exact(n-c, n, alpha)
    return [lower_b-upper_exact(c, n, alpha), upper_exact(b, n, alpha)-lower_c]


def paired_cells(left, right):
    return dict(both_false=sum(not a and not b for a, b in zip(left, right)),
                left_only=sum(a and not b for a, b in zip(left, right)),
                right_only=sum(not a and b for a, b in zip(left, right)),
                both_true=sum(a and b for a, b in zip(left, right)))


def analyze(rows, margin=.02):
    require(rows and 0 <= margin < 1, 'empty data or invalid equivalence margin')
    seen = set()
    examples = []
    for row in rows:
        key = (row['depth'], row['task'], row['index'])
        require(key not in seen, 'duplicate paired example')
        seen.add(key)
        require(set(row['results']) == set(ARMS), 'missing or extra treatment')
        results = row['results']
        require(len({v['prompt_sha256'] for v in results.values()}) == 1, 'paired prompts differ')
        require(all(isinstance(v['answer'], str) and 0 <= v['score'] <= 1 for v in results.values()), 'invalid response')
        item = {k: row[k] for k in ('depth', 'task', 'index')}
        for arm in ARMS:
            item[arm+'_answer'] = results[arm]['answer']
            item[arm+'_score'] = results[arm]['score']
        # Answer identity is exact after outer whitespace; RULER substring scoring is a separate endpoint.
        item['offload_changed'] = results['offload']['answer'].strip() != results['local']['answer'].strip()
        item['partition_changed'] = results['local_partition']['answer'].strip() != results['local']['answer'].strip()
        examples.append(item)
    tables: list[dict[str, Any]] = []
    for depth in sorted({r['depth'] for r in examples}):
        depth_rows = [r for r in examples if r['depth'] == depth]
        for task in ['ALL'] + sorted({r['task'] for r in depth_rows}):
            group = [r for r in depth_rows if task == 'ALL' or r['task'] == task]
            n = len(group)
            left = [r['offload_changed'] for r in group]
            right = [r['partition_changed'] for r in group]
            cells = paired_cells(left, right)
            b, c = cells['left_only'], cells['right_only']
            interval = excess_interval(b, c, n)
            table = dict(depth=depth, task=task, n=n, **cells, mcnemar_p=mcnemar(b, c),
                         offload_changed=sum(left), partition_changed=sum(right),
                         offload_rate=sum(left)/n, partition_rate=sum(right)/n,
                         offload_wilson95=wilson(sum(left), n), partition_wilson95=wilson(sum(right), n),
                         excess=(b-c)/n, excess_conservative95=interval, margin=margin,
                         strict_no_more_changes=interval[1] <= 0,
                         noninferior_at_margin=interval[1] < margin,
                         equivalent_at_margin=interval[0] > -margin and interval[1] < margin,
                         score_means={arm: sum(r[arm+'_score'] for r in group)/n for arm in ARMS})
            table['accuracy_pairs'] = {}
            for arm in ARMS[1:]:
                acc = paired_cells([r[arm+'_score'] == 1 for r in group], [r['local_score'] == 1 for r in group])
                table['accuracy_pairs'][arm] = dict(**acc, mcnemar_p=mcnemar(acc['left_only'], acc['right_only']))
            tables.append(table)
    running = 0.0
    primary = [t for t in tables if t['task'] == 'ALL']
    require(len(primary) <= 4, 'more than four primary depths')
    adjustment = 'mcnemar_holm_four_depths' if len(primary) == 4 else 'mcnemar_bonferroni_four_depths'
    for rank, table in enumerate(sorted(primary, key=lambda t: t['mcnemar_p'])):
        if len(primary) < 4:
            table[adjustment] = min(1.0, 4*table['mcnemar_p'])
            continue
        running = max(running, min(1.0, (4-rank)*table['mcnemar_p']))
        table[adjustment] = running
    return examples, tables


def write_analysis(root, rows, margin=.02):
    examples, tables = analyze(rows, margin)
    (root/'analysis.json').write_text(json.dumps(tables, indent=2)+'\n')
    with (root/'per-example.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(examples[0]))
        writer.writeheader()
        writer.writerows(examples)
    (root/'discordant-pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples
                                      if r['offload_changed'] != r['partition_changed']))
    lines = ['# Paired RULER', '',
             'Changed answers use outer-whitespace equality. Scores use NVIDIA substring rules.',
             'McNemar p is two-sided exact, unadjusted. Primary endpoints are ALL at each depth; use Holm across four depths.',
             'Excess is offload changed rate minus partition changed rate. Its conservative 95% interval uses Bonferroni exact binomial cell bounds.',
             'Failure to reject McNemar is not equivalence. Strict no-more-changes requires upper <= 0; practical equivalence uses the declared margin on both sides.', '',
             '| Depth | Task | n | Offload changes (95% Wilson) | Partition changes (95% Wilson) | O only | P only | McNemar p | Excess 95% | Strict <=0 | Equivalent at margin |',
             '|---:|---|---:|---|---|---:|---:|---:|---|---|---|']
    for t in tables:
        def rate(name):
            return f"{t[name+'_rate']:.4f} {t[name+'_wilson95']}"
        lines.append(f"| {t['depth']} | {t['task']} | {t['n']} | {rate('offload')} | {rate('partition')} | {t['left_only']} | {t['right_only']} | {t['mcnemar_p']:.6g} | {t['excess_conservative95']} | {t['strict_no_more_changes']} | {t['equivalent_at_margin']} (margin {margin}) |")
    lines += ['', '| Depth | Task | n | Local RULER score (%) | Offload score (%) | Partition score (%) | Offload/local accuracy McNemar p |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for t in tables:
        scores = t['score_means']
        lines.append(f"| {t['depth']} | {t['task']} | {t['n']} | {100*scores['local']:.2f} | {100*scores['offload']:.2f} | {100*scores['local_partition']:.2f} | {t['accuracy_pairs']['offload']['mcnemar_p']:.6g} |")
    (root/'analysis.md').write_text('\n'.join(lines)+'\n')
    return tables


def write_bf16_analysis(root, rows, margin=.02):
    require(rows, 'empty bf16 data')
    examples = []
    seen = set()
    for row in rows:
        results = row['results']
        key = (row['depth'], row['task'], row['index'])
        require(key not in seen and set(results) == set(ARMS) | {'offload_bf16'}, 'invalid bf16 paired row')
        seen.add(key)
        require(len({v['prompt_sha256'] for v in results.values()}) == 1 and
                all(isinstance(v['answer'], str) and 0 <= v['score'] <= 1 for v in results.values()), 'bf16 paired input/score differs')
        item = {k: row[k] for k in ('depth', 'task', 'index')}
        for arm, value in results.items():
            item[arm+'_answer'], item[arm+'_score'] = value['answer'], value['score']
        for arm in ('offload', 'offload_bf16', 'local_partition'):
            item[arm+'_changed'] = results[arm]['answer'].strip() != results['local']['answer'].strip()
            item[arm+'_score_disagrees_local'] = results[arm]['score'] != results['local']['score']
        item['bf16_score_disagrees_f32'] = results['offload_bf16']['score'] != results['offload']['score']
        examples.append(item)
    tables: list[dict[str, Any]] = []
    for depth in sorted({r['depth'] for r in rows}):
        for task in ['ALL'] + sorted({r['task'] for r in rows if r['depth'] == depth}):
            group = [r for r in examples if r['depth'] == depth and (task == 'ALL' or r['task'] == task)]
            changes = paired_cells([r['offload_bf16_changed'] for r in group], [r['offload_changed'] for r in group])
            accuracy = paired_cells([r['offload_bf16_score'] == 1 for r in group], [r['offload_score'] == 1 for r in group])
            tables.append(dict(depth=depth, task=task, n=len(group),
                bf16_answer_changes=sum(r['offload_bf16_changed'] for r in group),
                f32_answer_changes=sum(r['offload_changed'] for r in group),
                partition_answer_changes=sum(r['local_partition_changed'] for r in group),
                bf16_local_score_disagreements=sum(r['offload_bf16_score_disagrees_local'] for r in group),
                bf16_f32_score_disagreements=sum(r['bf16_score_disagrees_f32'] for r in group),
                answer_change_pairs=dict(**changes, mcnemar_p=mcnemar(changes['left_only'], changes['right_only'])),
                accuracy_pairs=dict(**accuracy, mcnemar_p=mcnemar(accuracy['left_only'], accuracy['right_only'])),
                score_means={arm: sum(r[arm+'_score'] for r in group)/len(group) for arm in (*ARMS, 'offload_bf16')}))
    (root/'analysis.json').write_text(json.dumps(tables, indent=2)+'\n')
    with (root/'per-example.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(examples[0]))
        writer.writeheader(); writer.writerows(examples)
    (root/'discordant-pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples
        if r['offload_bf16_changed'] != r['offload_changed'] or r['bf16_score_disagrees_f32']))
    lines = ['# RULER bf16 state arm', '', 'Only offload_bf16 was run. Local, f32 offload and partition rows are frozen baseline observations.',
        'Answer changes use outer-whitespace equality; score disagreements compare full fractional RULER scores. Accuracy is score == 1.',
        'McNemar is two-sided exact, unadjusted, bf16 versus f32. Non-significance does not establish equivalence. Apply Holm across four primary depths.', '',
        '| Depth | Task | n | bf16 changes | f32 changes | bf16/local score disagreements | bf16/f32 score disagreements | Change McNemar p | Accuracy McNemar p |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in tables:
        lines.append(f"| {t['depth']} | {t['task']} | {t['n']} | {t['bf16_answer_changes']} | {t['f32_answer_changes']} | {t['bf16_local_score_disagreements']} | {t['bf16_f32_score_disagreements']} | {t['answer_change_pairs']['mcnemar_p']:.6g} | {t['accuracy_pairs']['mcnemar_p']:.6g} |")
    (root/'analysis.md').write_text('\n'.join(lines)+'\n')
    return tables


def exact_power(n, p_left=.15, p_right=.05, alpha=.05):
    # Condition on total discordants, then average the exact test's rejection probability.
    q = p_left+p_right
    total = 0.0
    for d in range(n+1):
        pd = math.comb(n, d)*q**d*(1-q)**(n-d)
        conditional = p_left/q
        tail, critical = 0, -1
        for b in range(d//2+1):
            tail += math.comb(d, b)
            if 2*tail / 2**d > alpha:
                break
            critical = b
        if critical >= 0:
            total += pd*(binomial_cdf(critical, d, conditional)+binomial_cdf(critical, d, 1-conditional))
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rows', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--margin', type=float, default=.02)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_analysis(args.output, [json.loads(s) for s in args.rows.read_text().splitlines()], args.margin)

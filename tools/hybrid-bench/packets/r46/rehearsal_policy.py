#!/usr/bin/env python3
"""Generate illustrative GPU rehearsal routing from the r46b planning scenario."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

PACKET = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKET))
import estimate
from bench_common import Refusal, private_json, read_private, require


def prepare(kind, geometry):
    require(kind in ('qwen', 'glimmer'), 'unknown rehearsal model')
    qwen = kind == 'qwen'
    require(geometry['kv_bytes'] > 0 and geometry['state_bytes'] >= 0, 'invalid rehearsal geometry')
    require((geometry['state_bytes'] > 0 and not geometry.get('window_bytes', 0)) if qwen else
            (geometry['state_bytes'] == 0 and geometry.get('window_bytes', 0) > 0), 'wrong rehearsal state component')
    policy = estimate.planning_policy(kind, geometry, 20000000)
    plan = estimate.g.depth_plan(100000 if qwen else 76000, 40, 1, 10875, 4506, 340,
                                 14 if qwen else 32, 196608 if qwen else 131072)
    outage = 2 if qwen else 20

    def routes(selected):
        result = estimate.g.route_preflight(selected, plan, outage, 3)
        return dict(counts=result['counts'], offload_turns=[r['turn'] for r in result['rows'] if r['offload'] and not r['outage']])

    report = dict(model=kind, illustrative=True, calibration=False, geometry=geometry,
        source='r46b estimate.py / LINK-PLANNING.md, 20 decimal MB/s; prior Metal import fit',
        local_source='measured PRI31 Qwen delta512 floor curve' if qwen else 'interim hypothetical Glimmer 100 tok/s',
        seed=1, turns=40, depth=plan['d0'], final_depth=plan['final_depth'], framing=plan['framing_tokens'],
        outage_turns=list(range(outage, outage + 3)), f32=routes(policy),
        limitation='Configured-link preflight predictions. Live transfer-rate EMA and actual framing may change routes; inspect the captured trace.')
    if qwen:
        bf16 = estimate.planning_policy(kind, dict(geometry, state_bytes=geometry['state_bytes'] // 2), 20000000)
        report['bf16_independent_policy'] = routes(bf16)
        imports = report['f32']['offload_turns']
        report['bf16_chain_turn_imports'] = imports
        report['bf16_chain_imports_at_checkpoints'] = {str(t): 1 + sum(n <= t for n in imports) for t in (10,20,30,40)}
    return asdict(policy), report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=['qwen', 'glimmer'])
    p.add_argument('--client-state', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    report_path = a.output.with_suffix('.planning.json')
    require(not a.output.exists() and not report_path.exists(), 'new rehearsal policy output required')
    geometry = read_private(a.client_state / 'proxy-config.json')['delta_geometry']
    policy, report = prepare(a.model, geometry)
    private_json(a.output, policy)
    private_json(report_path, report)
    print(json.dumps(report))


if __name__ == '__main__':
    try:
        main()
    except (Refusal, OSError, ValueError, KeyError) as error:
        print('refused: ' + str(error) if isinstance(error, Refusal) else 'refused: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(2)

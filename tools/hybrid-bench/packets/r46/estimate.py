#!/usr/bin/env python3
"""Planning arithmetic, not production timings or physical network counters."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools/hybrid-bench'))
import growing_session as g
from policy import Policy, extend_chunk_limit


def curve(depth, rows, key):
    rate = rows[0][key]
    for row in rows:
        if row['depth'] <= depth:
            rate = row[key]
    return rate


def estimate(kind, link, local=100, decode=18):
    qwen = kind == 'qwen'
    fixture = json.loads((ROOT / 'tools/hybrid-bench/fixtures/m3-max-qwen-pri31.json').read_text())
    rows = [dict(r, prefill=r['delta_prefill_tok_s']['512']) for r in fixture['depths']]
    prefill = lambda d: curve(d, rows, 'prefill') if qwen else local
    decoded = lambda d: curve(d, rows, 'decode_tok_s') if qwen else decode
    depth, framing, context = (100000, 14, 196608) if qwen else (76000, 32, 131072)
    geometry = dict(kv_bytes=34816 if qwen else 7072, state_bytes=157000000 if qwen else 0)
    if not qwen:
        geometry['window_bytes'] = 43450368 + 2304 * 28 + 16
    plan = g.depth_plan(depth, 40, 1, 10875, 4506, 340, framing, context)
    state = geometry['state_bytes'] + geometry.get('window_bytes', 0)
    policy = Policy(local_rates=tuple([(0, prefill(0))] + [(r['depth'], r['prefill']) for r in rows]) if qwen else ((0, local),),
        server_rate=1767 if qwen else 2823, link_rate=link, fixed=.1406, import_fixed=1.498,
        import_per_token=.0000487, kv_bytes=geometry['kv_bytes'], state_bytes=state)
    chunk = extend_chunk_limit(**geometry)
    def wire(n):
        return (geometry['kv_bytes'] + 32) * n + (state + 4096) * ((n + chunk - 1) // chunk)
    down = wire(depth - 1)
    uploads = 11 * depth
    previous, offloads = True, []
    hybrid_prefill = local_prefill = 0
    for r in plan['sequence']:
        n = r['input_tokens'] + framing - 1
        choice = policy.choose(n, r['conditioning_depth'], previous)
        remote = choice['offload'] and r['turn'] not in (2, 3, 4)
        previous = remote
        if remote:
            offloads.append(r['turn'])
            down += wire(n)
        hybrid_prefill += choice['offload_seconds'] if remote else n / prefill(r['conditioning_depth'])
        local_prefill += n / prefill(r['conditioning_depth'])
        uploads += 11 * (r['input_tokens'] + framing + r['output_tokens'])
    generated = sum(r['output_tokens'] / decoded(r['conditioning_depth']) for r in plan['sequence'])
    pauses = sum(r['pause_seconds'] for r in plan['sequence'])
    local_base = sum(min(256, depth - d) / prefill(d) for d in range(0, depth, 256))
    hybrid_base = policy.fixed + depth / policy.server_rate + down / (link * policy.eta)
    timing = dict(hybrid_turn_h=(generated + pauses + hybrid_prefill) / 3600,
                  local_turn_h=(generated + pauses + local_prefill) / 3600,
                  base_h=(local_base + hybrid_base) / 3600)
    return dict(model=kind, assumptions=dict(link_bytes_s=link, local_prefill=local if not qwen else 'PRI31 delta512 floor curve',
        decode=decode if not qwen else 'PRI31 weighted depth curve', framing=framing,
        import_fit='Qwen prior Metal fit, illustrative for Glimmer; must recalibrate on Mac'),
        final_depth=plan['final_depth'], policy_offloads=offloads, mixed_routes=bool(offloads) and len(offloads) < 37,
        seconds_model=timing, subtotal_h=sum(timing.values()),
        down_body_estimate_bytes=down, up_token_json_upper_estimate_bytes=uploads,
        link_probe_both_directions_bytes=6 * 64 * 1024**2,
        planning_traffic_bytes=down + uploads + 6 * 64 * 1024**2,
        limits='Delta payload/header allowance plus token JSON; excludes SSH/TCP retransmission, HTTP framing and calibration. No captures/checker. Time omits save/restore, background serialization, catch-up waits and thermal tails; add 25-50 percent operational margin.')


if __name__ == '__main__':
    print(json.dumps(dict(qwen=[estimate('qwen', x) for x in (5.19e6, 9.75e6, 16e6, 25e6)],
        glimmer_hypothetical=[estimate('glimmer', link, local, decode) for link in (5.19e6, 9.75e6, 25e6)
                              for local, decode in ((50, 12), (100, 18), (150, 24))]), indent=2))

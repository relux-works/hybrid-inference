#!/usr/bin/env python3
"""Planning arithmetic, not production timings or physical network counters."""
import json
import argparse
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


def delta_wire_estimate(tokens, geometry, chunk_tokens=None):
    chunk = extend_chunk_limit(**geometry) if chunk_tokens is None else chunk_tokens
    # Protocol v2 carries fixed state only in the final chunk; reserve 4096 header bytes per chunk.
    return ((geometry['kv_bytes'] + 32) * tokens + geometry['state_bytes'] + geometry.get('window_bytes', 0) +
            4096 * ((tokens + chunk - 1) // chunk))


def planning_policy(kind, geometry, link, local=100):
    fixture = json.loads((ROOT / 'tools/hybrid-bench/fixtures/m3-max-qwen-pri31.json').read_text())
    rates = g.validate_fixture(fixture, fixture['model_sha256'], fixture['architecture']) if kind == 'qwen' else [[0, local]]
    return Policy(local_rates=rates, server_rate=1767 if kind == 'qwen' else 2823,
        link_rate=link, fixed=.1406, import_fixed=1.498, import_per_token=.0000487,
        kv_bytes=geometry['kv_bytes'], state_bytes=geometry['state_bytes'] + geometry.get('window_bytes', 0))


def estimate(kind, link, local=100, decode=18, recurrent_dtype="f32"):
    qwen = kind == 'qwen'
    fixture = json.loads((ROOT / 'tools/hybrid-bench/fixtures/m3-max-qwen-pri31.json').read_text())
    rows = [dict(r, prefill=r['delta_prefill_tok_s']['512']) for r in fixture['depths']]
    prefill = lambda d: curve(d, rows, 'prefill') if qwen else local
    decoded = lambda d: curve(d, rows, 'decode_tok_s') if qwen else decode
    depth, framing, context = (100000, 14, 196608) if qwen else (76000, 32, 131072)
    geometry = dict(kv_bytes=34816 if qwen else 7072,
                    state_bytes=fixture['geometry']['state_bytes'] if qwen else 0)
    if not qwen:
        geometry['window_bytes'] = 43450368 + 2304 * 28 + 16
    original_chunk = extend_chunk_limit(**geometry)
    if recurrent_dtype == 'bf16':
        if not qwen:
            raise ValueError('bf16 projection is recurrent-Qwen wire only')
        geometry['state_bytes'] -= fixture['geometry']['recurrent_tensor_bytes'] // 2
    elif recurrent_dtype != 'f32':
        raise ValueError('unknown recurrent wire dtype')
    plan = g.depth_plan(depth, 40, 1, 10875, 4506, 340, framing, context)
    policy = planning_policy(kind, geometry, link, local)
    base_down = delta_wire_estimate(depth - 1, geometry, original_chunk)
    down = base_down
    uploads = 11 * depth
    previous, offloads = True, []
    hybrid_prefill = local_prefill = 0
    hybrid_turn_ttft = local_turn_ttft = 0
    for r in plan['sequence']:
        n = r['input_tokens'] + framing - 1
        choice = policy.choose(n, r['conditioning_depth'], previous)
        remote = choice['offload'] and r['turn'] not in (2, 3, 4)
        previous = remote
        if remote:
            offloads.append(r['turn'])
            down += delta_wire_estimate(n, geometry, original_chunk)
        hybrid_prefill += choice['offload_seconds'] if remote else n / prefill(r['conditioning_depth'])
        local_prefill += n / prefill(r['conditioning_depth'])
        common = 1 / prefill(r['conditioning_depth']) + 1 / decoded(r['conditioning_depth'])
        hybrid_turn_ttft += (choice['offload_seconds'] if remote else n / prefill(r['conditioning_depth'])) + common
        local_turn_ttft += n / prefill(r['conditioning_depth']) + common
        uploads += 11 * (r['input_tokens'] + framing + r['output_tokens'])
    generated = sum(r['output_tokens'] / decoded(r['conditioning_depth']) for r in plan['sequence'])
    pauses = sum(r['pause_seconds'] for r in plan['sequence'])
    local_base = sum(min(256, depth - d) / prefill(d) for d in range(0, depth, 256))
    hybrid_base = (policy.fixed + (depth-1) / policy.server_rate + base_down / (link * policy.eta) +
                   policy.import_fixed + policy.import_per_token*(depth-1) + 1/prefill(depth))
    timing = dict(hybrid_turn_h=(generated + pauses + hybrid_prefill) / 3600,
                  local_turn_h=(generated + pauses + local_prefill) / 3600,
                  base_h=(local_base + hybrid_base) / 3600,local_base_h=local_base/3600,hybrid_base_h=hybrid_base/3600)
    return dict(model=kind, assumptions=dict(link_bytes_s=link, local_prefill=local if not qwen else 'PRI31 delta512 floor curve',
        decode=decode if not qwen else 'PRI31 weighted depth curve', framing=framing,
        import_fit='Qwen prior Metal fit, illustrative for Glimmer; must recalibrate on Mac'),
        final_depth=plan['final_depth'], policy_offloads=offloads, mixed_routes=bool(offloads) and len(offloads) < 37,
        recurrent_dtype=recurrent_dtype, wire_projection_only=recurrent_dtype == 'bf16',
        base_tokens=depth, base_offloads=True, base_mode='forced pre-stage, unchanged driver',
        hybrid_turn_ttft_s=hybrid_turn_ttft, local_turn_ttft_s=local_turn_ttft,
        hybrid_base_ttft_s=hybrid_base + 1/decoded(depth), local_base_ttft_s=local_base + 1/decoded(depth),
        hybrid_total_ttft_s=hybrid_turn_ttft+hybrid_base+1/decoded(depth),
        local_total_ttft_s=local_turn_ttft+local_base+1/decoded(depth),
        ttft_model='prompt/policy cost plus one local held prompt token and 1/decode_rate first-sample proxy; excludes queue/serialization/unmodeled HTTP jitter; no pauses or post-first-token decode',
        seconds_model=timing, subtotal_h=timing['hybrid_turn_h']+timing['local_turn_h']+timing['base_h'],
        down_body_estimate_bytes=down, up_token_json_upper_estimate_bytes=uploads,
        link_probe_both_directions_bytes=6 * 64 * 1024**2,
        planning_traffic_bytes=down + uploads + 6 * 64 * 1024**2,
        limits='Delta payload/header allowance plus token JSON; excludes SSH/TCP retransmission, HTTP framing and calibration. No captures/checker. Time omits save/restore, background serialization, catch-up waits and thermal tails; add 25-50 percent operational margin.')


def qwen_three_arm(link):
    f=estimate('qwen',link);b=estimate('qwen',link,recurrent_dtype='bf16')
    return dict(body_bytes=sum(x['down_body_estimate_bytes']+x['up_token_json_upper_estimate_bytes'] for x in (f,b)),
                hours=f['subtotal_h']+b['seconds_model']['hybrid_turn_h']+b['seconds_model']['hybrid_base_h'])


LINKS = [2, 3, 5, 10, 20, 30, 50]


def threshold(kind, dtype='f32', local=100, decode=18):
    # This is the infimum for at least one connected turn, excluding the forced pre-stage.
    low, high = 1., 1e9
    for _ in range(55):
        middle = (low+high)/2
        if estimate(kind, middle, local, decode, dtype)['policy_offloads']:
            high = middle
        else:
            low = middle
    return high/1e6


def table():
    profiles = [('Qwen f32 PRI31', 'qwen', 'f32', 100, 18),
                ('Qwen hybrid-bf16 wire estimate', 'qwen', 'bf16', 100, 18)]
    profiles += [('Glimmer interim '+str(p)+'/'+str(d), 'glimmer', 'f32', p, d)
                 for p, d in ((50, 12), (100, 18), (150, 24))]
    result = []
    for label, kind, dtype, prefill, decode in profiles:
        result.append(dict(profile=label, turn_offload_threshold_MB_s=threshold(kind, dtype, prefill, decode),
            rows=[dict(link_MB_s=link, **estimate(kind, link*1e6, prefill, decode, dtype)) for link in LINKS]))
    return result


def markdown(profiles):
    lines = ['# r46b link-rate planning', '',
        'Decimal MB/s. Seed1,40 turns, outage2-4. Base is forced offload by the current pre-staged driver at every positive link rate:100k Qwen,76k Glimmer. It is outside the measured turn window. A non-empty *turn* offload arm has the profile-specific threshold below; including the forced base, every arm is non-empty for any positive link.', '',
        'Wire is estimated request/delta application bodies, excluding calibration, link probes, HTTP/SSH/TCP overhead/retransmissions and captures. Fixed state occurs only in the last chunk. The hybrid-bf16 arm is enabled with accepted transport; this planning model changes recurrent tensor bytes only. BF16 import costs and quality remain unmeasured and unchanged in this calculation; the Mac run requires its own bounded fit. Glimmer curves are interim hypothetical sensitivity cases (prefill/decode tok/s), not PRI31 fixtures.', '',
        'TTFT totals include base plus40 turns; turn-only totals are also shown. They model prompt/policy costs, one held local prompt token and1/decode_rate first-sample proxy; queue, node load/save/serialization, readback and other residual costs remain unmodeled. No pauses or remaining output decode are added. RTT0.1406s, eta0.85, prior Qwen import1.498s+0.0000487s/token; Glimmer uses that fit as an explicitly interim portability assumption.', '']
    for profile in profiles:
        lines += ['## '+profile['profile'], '',
            'First connected turn offload above approximately '+format(profile['turn_offload_threshold_MB_s'], '.6f')+' MB/s (strict policy inequality; not a measured link).', '',
            '| MB/s | Turn offloads /40 | Base offload | Wire GB | Hybrid total TTFT s | Local total TTFT s | Hybrid turn TTFT s | Local turn TTFT s |',
            '|---:|---:|:---:|---:|---:|---:|---:|---:|']
        for row in profile['rows']:
            lines.append('| '+str(row['link_MB_s'])+' | '+str(len(row['policy_offloads']))+' | yes, forced | '+
                format((row['down_body_estimate_bytes']+row['up_token_json_upper_estimate_bytes'])/1e9, '.3f')+' | '+
                ' | '.join(format(row[key], '.2f') for key in ('hybrid_total_ttft_s','local_total_ttft_s','hybrid_turn_ttft_s','local_turn_ttft_s'))+' |')
        lines.append('')
    lines += ['## Qwen three-arm total', '', 'Run order: hybrid, local, hybrid-bf16. Body bytes include both forced bases; one403MB link probe and the shared <=2GB calibration are additional. Hours include generated output, sampled pauses and bases, with25-50% operational allowance still to add. The bf16 row assumes the f32 import fit only for planning.', '', '| MB/s | Total body GB | Three-arm hours before allowance |', '|---:|---:|---:|']
    for link in LINKS:
        row=qwen_three_arm(link*1e6)
        lines.append('| '+str(link)+' | '+format(row['body_bytes']/1e9,'.3f')+' | '+format(row['hours'],'.2f')+' |')
    return '\n'.join(lines).rstrip()+'\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table', action='store_true')
    p.add_argument('--markdown', action='store_true')
    a = p.parse_args()
    if a.table or a.markdown:
        profiles = table()
        print(markdown(profiles) if a.markdown else json.dumps(profiles, indent=2), end='' if a.markdown else '\n')
    else:
        print(json.dumps(dict(qwen=[estimate('qwen', x) for x in (5.19e6, 9.75e6, 16e6, 25e6)],
            glimmer_hypothetical=[estimate('glimmer', link, local, decode) for link in (5.19e6, 9.75e6, 25e6)
                                  for local, decode in ((50, 12), (100, 18), (150, 24))]), indent=2))

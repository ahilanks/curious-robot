"""Aggregate eval JSONL -> table vs. the paper's LeWM columns (Tables 1, 3, 4).
Mean over eval seeds (and RP1 planner seeds); Cube also reports the paper's
'hard' score (s - f) / (100 - f) using the measured no-op floor f at the same h."""
import json, sys, collections
import numpy as np

PAPER = {  # LeWM columns, success %
    ('tworoom', 25): {'cem/latent': 84.0, 'cem/value': 100.0, 'mppi/latent': 70.7, 'mppi/value': 87.3, 'adam/latent': 94.7, 'adam/value': 96.7, 'rp1': 100.0},
    ('tworoom', 100): {'cem/latent': 13.3, 'cem/value': 94.7, 'mppi/latent': 20.0, 'mppi/value': 64.0, 'adam/latent': 24.0, 'adam/value': 83.3, 'rp1': 94.2},
    ('reacher', 25, 0.1): {'cem/latent': 98.7, 'cem/value': 97.3, 'mppi/latent': 63.7, 'mppi/value': 74.0, 'adam/latent': 94.0, 'adam/value': 88.0, 'rp1': 98.7},
    ('reacher', 25, 0.05): {'cem/latent': 80.3, 'cem/value': 82.0, 'mppi/latent': 39.3, 'mppi/value': 42.0, 'adam/latent': 66.0, 'adam/value': 64.7, 'rp1': 88.7},
    ('cube', 25): {'noop': (56.0, None), 'cem/latent': (74.0, 40.9), 'cem/value': (84.0, 63.6), 'mppi/latent': (56.7, 1.6), 'mppi/value': (63.3, 16.6), 'adam/latent': (74.0, 40.9), 'adam/value': (74.7, 42.5), 'rp1': (89.1, 75.2)},
    ('cube', 100): {'noop': (45.3, None), 'cem/latent': (58.0, 23.2), 'cem/value': (76.7, 57.4), 'mppi/latent': (46.7, 2.5), 'mppi/value': (52.0, 12.2), 'adam/latent': (57.3, 21.9), 'adam/value': (68.7, 42.7), 'rp1': (82.4, 67.8)},
}
ORDER = ['noop', 'random', 'replay', 'cem/latent', 'cem/value', 'mppi/latent', 'mppi/value', 'adam/latent', 'adam/value', 'rp1']

def read_jsonl(fn):
    """Robust JSONL reader: tolerates concatenated objects / literal '\\n' separators on one line."""
    dec, out = json.JSONDecoder(), []
    for line in open(fn):
        line = line.replace('\\n', '\n')
        for chunk in line.split('\n'):
            chunk = chunk.strip()
            while chunk:
                obj, end = dec.raw_decode(chunk)
                out.append(obj); chunk = chunk[end:].strip()
    return out

recs = [r for fn in sys.argv[1:] for r in read_jsonl(fn)]
groups = collections.defaultdict(list)
for r in recs:
    key = r['planner'] + ('/' + r['objective'] if r.get('objective') else '')
    groups[(r['domain'], r['h'], r.get('reacher_tau'), key)].append(r)

def fmt(x):
    return '   -  ' if x is None else f'{x:6.1f}'

cells = sorted(set((d, h, t) for d, h, t, _ in groups), key=lambda x: (x[0], x[1], -(x[2] or 0)))
for dom, h, tau in cells:
    hdr = f'{dom} h={h}' + (f' tau={tau}' if tau else '')
    floor = None
    if dom == 'cube' and (dom, h, tau, 'noop') in groups:
        floor = np.mean([r['success_rate'] for r in groups[(dom, h, tau, 'noop')]])
    ref = PAPER.get((dom, h, tau) if tau else (dom, h), {})
    print(f'\n== {hdr}   (runs = eval seeds x planner seeds; eps = episodes)')
    print(f"   {'planner':12s} {'ours':>6s} {'sd':>5s} {'runs':>4s} {'eps':>5s} | {'paper':>6s}" + (f" | {'hard':>6s} {'paper':>6s}" if dom == 'cube' else ''))
    keys = sorted({k for (d, hh, t, k) in groups if (d, hh, t) == (dom, h, tau)}, key=lambda k: ORDER.index(k) if k in ORDER else 99)
    for key in keys:
        rs = groups[(dom, h, tau, key)]
        sr = np.array([r['success_rate'] for r in rs])
        p = ref.get(key)
        line = f'   {key:12s} {sr.mean():6.1f} {sr.std():5.1f} {len(rs):4d} {sum(r["n"] for r in rs):5d} | '
        if dom == 'cube':
            pe, ph = p if p else (None, None)
            hard = None if floor is None or key == 'noop' else max(0.0, (sr.mean() - floor) / (100 - floor) * 100)
            line += fmt(pe) + f' | {fmt(hard)} {fmt(ph)}'
        else:
            line += fmt(p)
        print(line)

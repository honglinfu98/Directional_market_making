#!/usr/bin/env python3
"""Emit a typed-kick table from REGRESSION-identified magnitudes (Stage A of c(x)Q).

The shipped kirchner_typed_*.json builds w from a per-event covariance proxy:
    w_k  ~  E[ future activity | event of channel k ]
with nothing controlled for.  Market orders OCCUR INSIDE bursts, so they
inherit the burst's future activity without causing it -- the estimate is
confounded, and it is why w[MO] came out at 122-458x while every other group
sat below 1.3.  Doc 4.15 then deployed those magnitudes and dispersion blew up
8-37x.

The joint regression in typed_matrix_probe.py conditions on ALL groups' lagged
counts, so its column sums

    c_h = sum_g M[g,h]        (net offspring of one type-h event)

are the causally identified magnitude.  Measured:

    SOL  LO 0.87/1.02  CO 0.98/0.88  MO 1.04/0.25  IS 1.19/1.05
    ETH  LO 1.01/1.09  CO 0.88/0.74  MO 2.36/2.13  IS 0.95/0.94
    BTC  LO 1.00/1.00  CO 0.82/0.92  MO 3.16/6.20  IS 0.99/0.96

i.e. real MO ignition is 2-5x on the small-tick assets (and ~1x on SOL), not
100x+.  Because MO is only ~0.1-0.3% of events, the marked-cluster variance
multiplier E[w^2]/E[w]^2 stays at 1.01-1.03, so n is left at the shipped value
and the predicted Fano perturbation is a few percent -- which is the hypothesis
this arm tests.

Output matches the kirchner_typed_*.json schema so scripts/assemble_typed_ckpt.py
consumes it unchanged (use --variant group).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

GROUPS = ['LO_b', 'LO_a', 'CO_b', 'CO_a', 'MO_b', 'MO_a', 'IS_b', 'IS_a']
SIDES = ('b', 'a')


def fixed_names():
    names = []
    for et in ('LO', 'CO'):
        for s in SIDES:
            for l in range(1, 11):
                names.append(f'{et}_{s}_L{l}')
    for s in SIDES:
        names.append(f'MO_{s}_L1')
    for s in SIDES:
        for l in range(1, 11):
            names.append(f'IS_{s}_L{l}')
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comp', required=True, help='comp_<coin>.json from typed_matrix_probe')
    ap.add_argument('--typed', required=True, help='kirchner_typed_<coin>.json (for p_channel)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--floor', type=float, default=0.05,
                    help='floor on net offspring; guards against a group whose '
                         'net mass is ~0 collapsing its kick to zero')
    args = ap.parse_args()

    comp = json.loads(open(args.comp).read())
    typed = json.loads(open(args.typed).read())
    if comp['groups'] != GROUPS or typed['groups'] != GROUPS:
        raise SystemExit('group ordering mismatch')

    M = np.asarray(comp['M_raw'], float)          # [target, source]
    c = M.sum(axis=0)                              # net offspring per source
    c_floored = np.maximum(c, args.floor)

    names = fixed_names()
    gidx = {g: i for i, g in enumerate(GROUPS)}
    w = np.array([c_floored[gidx['_'.join(n.split('_')[:2])]] for n in names])

    p = np.asarray(typed['p_channel'], float)
    if len(p) != len(names):
        raise SystemExit(f'p_channel has {len(p)} entries, expected {len(names)}')

    w_norm = w / float(p @ w)                      # E[w] = 1 under empirical p
    Ew2 = float(p @ (w_norm ** 2))
    print(f'net offspring c: ' + ' '.join(f'{GROUPS[i]}={c[i]:.2f}' for i in range(8)))
    print(f'E[w]=1 by construction; E[w^2]/E[w]^2 = {Ew2:.4f} '
          f'(shipped confounded table: ~20)')
    print(f'w[MO_b]={w_norm[names.index("MO_b_L1")]:.3f}  '
          f'w[MO_a]={w_norm[names.index("MO_a_L1")]:.3f}  '
          f'w[LO_b_L1]={w_norm[0]:.3f}')

    out = {
        'groups': GROUPS,
        'w_group': [float(c_floored[i]) for i in range(8)],
        'w_channel': w.tolist(),
        'p_channel': typed['p_channel'],
        'source': 'regression column sums of M_raw (typed_matrix_probe joint fit)',
        'Ew2_over_Ew2': Ew2,
        'net_offspring_raw': c.tolist(),
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

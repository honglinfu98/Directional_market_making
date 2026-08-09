"""Typed Kirchner fit: per-source-group (8) and per-channel (62, shrunk) kicks.

Stage 1 (grouped, joint): OLS of total bin counts on lagged PER-GROUP counts
over the lin-log cell grid -> group excitation rows Phi^g_c. Rank-1 check
(separability), group kick weights w_g from row masses, shared kernel shape
from the aggregate row.

Stage 2 (ungrouped, shrunk): within each group, allocate the group's kick
mass across member channels proportional to each channel's short-lag
covariance contribution, shrunk toward uniform by channel support:
  w_k  proportional to  (N_k * cov_k + tau * mean) / (N_k + tau)
i.e. empirical-Bayes toward the group mean; channels with little data
collapse onto their group.

Outputs a JSON with: shared bank betas, aggregate a (from the type-blind
fit, rescaled to target n), w_group[8], w_channel[62], p_bar[62].
Kicks normalized so E[w] = sum_k p_k w_k = 1 (then n = sum a/beta exactly).
"""
import argparse
import glob
import gzip
import json
import os
import re

import numpy as np

TRAIN_FRAC = 0.70
TYPES = ['LO', 'CO', 'MO', 'IS']
SIDES = ['b', 'a']
GROUPS = [f'{t}_{s}' for t in TYPES for s in SIDES]          # 8
EV_RE = re.compile(r'"event_type": "([A-Z]+)", "side": "([ab])", "level": (\d+)')


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


NAMES = fixed_names()
NAME_IDX = {n: i for i, n in enumerate(NAMES)}
GROUP_OF = {n: f"{n.split('_')[0]}_{n.split('_')[1]}" for n in NAMES}
GIDX = {g: i for i, g in enumerate(GROUPS)}


def load_day(path, frac=TRAIN_FRAC):
    """Per-event (time_s, channel_idx) for the train zone of one file."""
    ts, ch = [], []
    with gzip.open(path, 'rt') as f:
        lines = f.readlines()
    n = int(len(lines) * frac)
    for line in lines[:n]:
        i = line.index(':') + 1
        t = int(line[i:line.index(',', i)]) / 1e9
        for et, sd, lv in EV_RE.findall(line):
            name = f'{et}_{sd}_L{lv}'
            k = NAME_IDX.get(name)
            if k is not None:
                ts.append(t); ch.append(k)
    ts = np.asarray(ts); ch = np.asarray(ch)
    return ts - ts[0], ch


def cells(delta, lag_max, n_cells):
    edges_s = np.unique(np.concatenate([
        np.arange(delta, 10 * delta, delta),
        np.geomspace(10 * delta, lag_max, n_cells)]))
    return np.round(edges_s / delta).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--delta', type=float, default=0.25)
    ap.add_argument('--lag-max', type=float, default=240.0)
    ap.add_argument('--n-cells', type=int, default=14)
    ap.add_argument('--tau', type=float, default=5000.0, help='shrinkage strength (events)')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, '*.jsonl.gz')))
    edges = cells(args.delta, args.lag_max, args.n_cells)
    K = len(edges) - 1
    G = len(GROUPS)
    burn = int(edges[-1]) + 1
    P = G * K + 1
    xtx = np.zeros((P, P)); xty = np.zeros(P)
    ch_events = np.zeros(62)
    ch_cov = np.zeros(62)            # short-lag covariance proxy per channel
    total_events, total_secs = 0, 0.0

    for fpath in files:
        t, ch = load_day(fpath)
        nb = int(t[-1] / args.delta)
        keep = t < nb * args.delta
        tb = (t[keep] / args.delta).astype(int)
        chk = ch[keep]
        y = np.bincount(tb, minlength=nb).astype(np.float64)
        gcounts = np.zeros((G, nb))
        ccounts = {}
        for g in range(G):
            mask = np.array([GROUP_OF[NAMES[c]] == GROUPS[g] for c in range(62)])
            sel = mask[chk]
            gcounts[g] = np.bincount(tb[sel], minlength=nb)
        rows = np.arange(burn, nb)
        X = np.empty((len(rows), P)); X[:, 0] = 1.0
        for g in range(G):
            C = np.concatenate([[0.0], np.cumsum(gcounts[g])])
            for k in range(K):
                X[:, 1 + g * K + k] = C[rows - edges[k]] - C[rows - edges[k + 1]]
        xtx += X.T @ X; xty += X.T @ y[rows]
        # per-channel short-lag (<= 2.5 s) covariance with future total counts
        short = int(round(2.5 / args.delta))
        ycs = np.concatenate([[0.0], np.cumsum(y)])
        fut = ycs[np.minimum(np.arange(nb) + 1 + short, nb)] - ycs[np.minimum(np.arange(nb) + 1, nb)]
        for c in range(62):
            sel = chk == c
            ne = int(sel.sum())
            if ne == 0: continue
            ch_events[c] += ne
            ch_cov[c] += fut[tb[sel]].sum()          # sum over events of future activity
        total_events += len(t); total_secs += t[-1]
        print(f'  {os.path.basename(fpath)}: {len(t):,} ev', flush=True)

    R = total_events / total_secs
    sol = np.linalg.solve(xtx + 1e-6 * np.eye(P), xty)
    widths_s = (edges[1:] - edges[:-1]) * args.delta
    PhiG = sol[1:].reshape(G, K) * widths_s[None, :] / args.delta   # [G,K] group rows
    print('\ngroup row masses (int phi_g):')
    for g in range(G):
        print(f'  {GROUPS[g]:<6} {PhiG[g].sum():+.4f}')
    # separability check
    U, S, Vt = np.linalg.svd(np.maximum(PhiG, 0.0))
    print(f'separability: sigma1/sigma2 = {S[0]/max(S[1],1e-12):.1f} '
          f'(rank-1 energy {S[0]**2/np.sum(S**2)*100:.1f}%)')

    p_ch = ch_events / ch_events.sum()
    # group kicks from row masses (clip negatives), normalized E[w]=1
    gm = np.maximum(PhiG.sum(axis=1), 0.0)
    p_g = np.array([p_ch[[i for i in range(62) if GROUP_OF[NAMES[i]] == GROUPS[g]]].sum() for g in range(G)])
    w_g = gm / np.maximum(p_g, 1e-12)
    w_g = w_g / (p_g * w_g).sum()
    # channel kicks: per-event future-activity, shrunk to group mean
    per_ev = np.where(ch_events > 0, ch_cov / np.maximum(ch_events, 1), 0.0)
    w_c = np.zeros(62)
    for g in range(G):
        idx = [i for i in range(62) if GROUP_OF[NAMES[i]] == GROUPS[g]]
        gmean = np.average(per_ev[idx], weights=np.maximum(ch_events[idx], 1))
        raw = (ch_events[idx] * per_ev[idx] + args.tau * gmean) / (ch_events[idx] + args.tau)
        # scale channel profile so the group's frequency-weighted mean kick = w_g
        prof = raw / max(np.average(raw, weights=np.maximum(p_ch[idx], 1e-12)), 1e-12)
        w_c[idx] = w_g[GIDX[GROUPS[g]]] * prof
    w_c = w_c / (p_ch * w_c).sum()

    print('\ngroup kicks w_g (E[w]=1):')
    for g in range(G):
        print(f'  {GROUPS[g]:<6} {w_g[g]:.3f}')
    top = np.argsort(-w_c)[:8]
    print('top channel kicks:', [(NAMES[i], round(w_c[i], 2)) for i in top])

    out = {'groups': GROUPS, 'w_group': w_g.tolist(), 'w_channel': w_c.tolist(),
           'p_channel': p_ch.tolist(), 'PhiG': PhiG.tolist(),
           'edges_s': (edges * args.delta).tolist(), 'rate': R,
           'separability_s1_over_s2': float(S[0] / max(S[1], 1e-12))}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

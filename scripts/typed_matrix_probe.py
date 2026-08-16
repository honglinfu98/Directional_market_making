#!/usr/bin/env python3
"""RISK-1 PROBE: does a typed (8-group) linear Hawkes ground stand up on its own?

Background.  scripts/kirchner_fit_typed.py regresses TOTAL bin counts on lagged
PER-GROUP counts, so it recovers only each group's OUTGOING excitation mass
(PhiG is [8 groups x 22 lag cells]) -- never which groups that excitation lands
on.  Its "separability" number is an SVD in (group x lag) space, i.e. do all
groups share one kernel SHAPE; it says nothing about source->target structure.
So the 8x8 needed for a genuinely typed ground does not exist yet.

This script fits it: same lin-log cell design matrix, but with EIGHT responses
(one per target group) instead of one.  From the resulting Phi[target, source,
cell] it reports

  M[i,j] = int phi_{j->i}      kernel-norm matrix (8x8)
  rho    = spectral radius of M                    <- stability certificate
  mu     = (I - M) R_g                             <- vector rate pin (Jain)

and then answers the actual question: simulate the ground ALONE as an 8-dim
linear Hawkes cluster process (no neural mark head anywhere) and compare its
Fano curve to the real one measured on the same data.

Why this is the gate.  Doc 4.15 blew up at 8-37x real dispersion, but that arm
was typed kicks bolted onto a SCALAR ground whose types came from a neural mark
head -- the documented mechanism is reflexive (bursts -> p(MO|u) up -> 122x
kicks -> bigger bursts).  In a true multivariate linear Hawkes the offspring
type is drawn from the kernel, a FIXED law, so that amplifier does not exist.
If this probe is stable and near-real, that reading is confirmed and a typed
ground is viable.  If it explodes anyway, typed rates are unstable at these
magnitudes and both Option 1 and Option 3 die together.

Caveat recorded in the output: the branching simulation cannot represent
negative (inhibitory) kernel mass, so M is clipped at 0 and the clipped
fraction is reported.  A large clipped fraction means the linear-branching
reading understates inhibition and rho is optimistic.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os

import numpy as np

# Reuse the shipped fitter's data reader and lag grid verbatim -- the grid
# starts at `delta` (not 0) and the reader regexes raw lines with ns
# timestamps, several events per line.  Reimplementing either silently
# changes the estimand.
from kirchner_fit_typed import (  # noqa: E402
    GROUPS, GROUP_OF, NAMES, GIDX, cells, load_day,
)

CH_GROUP = np.array([GIDX[GROUP_OF[n]] for n in NAMES])


def fano_curve(times, duration, scales, bucket=1.0):
    """Fano factor Var(N)/E[N] at each scale, equal-duration segments."""
    out = []
    for s in scales:
        nb = int(duration // s)
        if nb < 8:
            out.append(float('nan'))
            continue
        c = np.histogram(times, bins=nb, range=(0.0, nb * s))[0].astype(float)
        m = c.mean()
        out.append(float(c.var() / m) if m > 0 else float('nan'))
    return out


def simulate_typed(M, Phi, edges_s, mu, T, rng, hard_cap=8_000_000):
    """8-dim linear Hawkes by its branching (immigration-birth) representation.

    Immigrants of type i at rate mu[i]; every event of type j spawns
    Poisson(M[i,j]) children of type i, each at a lag drawn from the
    normalised cell profile Phi[i,j,:].  Returns all event times.
    """
    G = len(mu)
    lo, hi = edges_s[:-1], edges_s[1:]
    prof = np.maximum(Phi, 0.0)
    tot = prof.sum(axis=2, keepdims=True)
    prof = np.divide(prof, np.where(tot > 0, tot, 1.0))   # [i,j,K] lag pmf

    times, types = [], []
    for i in range(G):
        n = rng.poisson(mu[i] * T)
        if n:
            times.append(rng.uniform(0.0, T, n))
            types.append(np.full(n, i))
    if not times:
        return np.zeros(0)
    ft, fk = np.concatenate(times), np.concatenate(types)
    all_t = [ft]
    total = len(ft)

    while len(ft):
        nt, nk = [], []
        for i in range(G):
            lam = M[i, fk]                                  # expected children of type i
            nc = rng.poisson(lam)
            tot_c = int(nc.sum())
            if tot_c == 0:
                continue
            parent = np.repeat(np.arange(len(ft)), nc)
            pj = fk[parent]
            # sample a cell per child, then a uniform lag inside it
            cum = np.cumsum(prof[i, pj, :], axis=1)
            u = rng.random(tot_c)
            cell = (u[:, None] > cum).sum(axis=1).clip(0, prof.shape[2] - 1)
            lag = lo[cell] + rng.random(tot_c) * (hi[cell] - lo[cell])
            ct = ft[parent] + lag
            keep = ct < T
            if keep.any():
                nt.append(ct[keep]); nk.append(np.full(int(keep.sum()), i))
        if not nt:
            break
        ft, fk = np.concatenate(nt), np.concatenate(nk)
        all_t.append(ft)
        total += len(ft)
        if total > hard_cap:
            print(f'  HARD CAP hit at {total:,} events -- treating as explosive', flush=True)
            return None
    return np.sort(np.concatenate(all_t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--delta', type=float, default=0.25)
    ap.add_argument('--lag-max', type=float, default=240.0)
    ap.add_argument('--n-cells', type=int, default=14)
    ap.add_argument('--train-frac', type=float, default=0.7)
    ap.add_argument('--sim-duration', type=float, default=3600.0)
    ap.add_argument('--sim-reps', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fit-only', action='store_true',
                    help='fit and save Phi/M/mu only; skip the branching '
                         'simulation. Used for the Option-B composition fit, '
                         'which needs the INHIBITORY (OLS) matrix and never '
                         'runs a branching sim, so subcriticality of the '
                         'clipped matrix is irrelevant.')
    ap.add_argument('--rho-target', type=float, default=None,
                    help='rescale M to this spectral radius (shipped recipe: '
                         'raw Kirchner branching under-disperses and is capped '
                         'to ~0.99 before use)')
    ap.add_argument('--nonneg', action='store_true',
                    help='constrain all kernel cells >= 0 (NNLS per target row). '
                         'Unconstrained OLS is not a valid Hawkes estimator: it '
                         'returns inhibitory mass that the linear-branching '
                         'representation cannot express.')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, '*.jsonl.gz')))
    if not files:
        raise SystemExit(f'no data in {args.data_dir}')
    edges = cells(args.delta, args.lag_max, args.n_cells)
    K = len(edges) - 1
    G = len(GROUPS)
    burn = int(edges[-1]) + 1
    P = G * K + 1

    xtx = np.zeros((P, P))
    xty = np.zeros((P, G))                 # EIGHT responses -- the whole point
    grp_events = np.zeros(G)
    total_secs = 0.0
    real_times, real_dur = [], 0.0

    for fpath in files:
        t, ch = load_day(fpath, args.train_frac)
        nb = int(t[-1] / args.delta)
        keep = t < nb * args.delta
        tb = (t[keep] / args.delta).astype(int)
        gk = CH_GROUP[ch[keep]]
        gcounts = np.zeros((G, nb))
        for g in range(G):
            gcounts[g] = np.bincount(tb[gk == g], minlength=nb)
        rows = np.arange(burn, nb)
        X = np.empty((len(rows), P)); X[:, 0] = 1.0
        for g in range(G):
            C = np.concatenate([[0.0], np.cumsum(gcounts[g])])
            for k in range(K):
                X[:, 1 + g * K + k] = C[rows - edges[k]] - C[rows - edges[k + 1]]
        xtx += X.T @ X
        xty += X.T @ gcounts[:, rows].T
        grp_events += gcounts.sum(axis=1)
        total_secs += t[-1]
        real_times.append(t + real_dur); real_dur += t[-1]
        print(f'  {os.path.basename(fpath)}: {len(t):,} ev', flush=True)

    if args.nonneg:
        # min ||Xb - y||^2 s.t. b >= 0, from the normal equations only:
        # X'X = L'L  =>  ||Xb-y||^2 = ||Lb - L^-T X'y||^2 + const
        from scipy.optimize import nnls
        L = np.linalg.cholesky(xtx + 1e-6 * np.eye(P)).T          # upper, L'L = xtx
        sol = np.empty((P, G))
        for g in range(G):
            rhs = np.linalg.solve(L.T, xty[:, g])
            sol[:, g], _ = nnls(L, rhs)
        print('(non-negative constrained fit)')
    else:
        sol = np.linalg.solve(xtx + 1e-6 * np.eye(P), xty)      # [P, G]
    widths_s = (edges[1:] - edges[:-1]) * args.delta
    # Phi[target i, source j, cell k]
    Phi = np.transpose(sol[1:].reshape(G, K, G), (2, 0, 1)) * widths_s[None, None, :] / args.delta
    M_raw = Phi.sum(axis=2)
    neg = float(np.minimum(M_raw, 0.0).sum() / max(np.abs(M_raw).sum(), 1e-12))
    M = np.maximum(M_raw, 0.0)

    rho_raw = float(np.max(np.abs(np.linalg.eigvals(M_raw))))
    rho = float(np.max(np.abs(np.linalg.eigvals(M))))
    R_g = grp_events / total_secs

    # The RAW Kirchner branching estimate systematically under-disperses: the
    # shipped blind ground measured 0.9635 and was capped to n=0.99 (doc 4.9),
    # then retuned per asset (4.10).  Rescaling M is the multivariate analogue
    # of that projection -- and of Jain's eigenvalue cap.
    if args.rho_target:
        s = args.rho_target / rho
        M = M * s
        Phi = Phi * s
        rho = float(np.max(np.abs(np.linalg.eigvals(M))))
        print(f'rescaled M by {s:.4f} -> rho={rho:.4f}')

    mu = (np.eye(G) - M) @ R_g

    print(f'\n=== {args.label} ===')
    print(f'total rate {R_g.sum():.3f} ev/s   clipped(negative) mass fraction {neg:.3f}')
    print('\nM[target<-source]  (kernel norms)')
    print('        ' + ''.join(f'{g:>9}' for g in GROUPS))
    for i, g in enumerate(GROUPS):
        print(f'{g:>7} ' + ''.join(f'{M_raw[i, j]:9.3f}' for j in range(G)))
    print(f'\ncolumn sums (outgoing mass per source): ' +
          ' '.join(f'{GROUPS[j]}={M_raw[:, j].sum():.2f}' for j in range(G)))
    print(f'spectral radius: rho_raw={rho_raw:.4f}  rho_clipped={rho:.4f}')
    print(f'mu = (I-M)R  -> ' + ' '.join(f'{GROUPS[i]}={mu[i]:+.4f}' for i in range(G)))

    res = {'label': args.label, 'groups': GROUPS, 'M_raw': M_raw.tolist(),
           'rho_raw': rho_raw, 'rho_clipped': rho, 'neg_mass_frac': neg,
           'R_group': R_g.tolist(), 'mu': mu.tolist(), 'edges_s': (edges * args.delta).tolist(),
           'Phi': Phi.tolist(), 'real_rate': float(R_g.sum())}

    scales = [1, 2, 5, 10, 20, 50]
    rt = np.concatenate(real_times)
    # Duration-MATCHED real baseline: the simulation is a stationary run of
    # sim_duration seconds, so the real curve must be measured on segments of
    # the same length and averaged.  Measuring real Fano over all 7 days lets
    # intraday seasonality and multi-day drift inflate the long scales (~1.6x
    # at 50 s here), which a stationary Hawkes cannot and should not match.
    seg = args.sim_duration
    nseg = int(real_dur // seg)
    segc = []
    for i in range(nseg):
        w = rt[(rt >= i * seg) & (rt < (i + 1) * seg)] - i * seg
        if len(w) > 100:
            segc.append(fano_curve(w, seg, scales))
    res['fano_real'] = list(np.nanmean(np.array(segc), axis=0)) if segc else fano_curve(rt, real_dur, scales)
    res['fano_real_unmatched'] = fano_curve(rt, real_dur, scales)
    res['n_real_segments'] = len(segc)
    print(f'\nFano real (matched, {len(segc)} x {seg:.0f}s) @{scales}: '
          + ' '.join(f'{x:8.1f}' for x in res['fano_real']))
    print(f'Fano real (unmatched, full {real_dur/86400:.1f}d)      : '
          + ' '.join(f'{x:8.1f}' for x in res['fano_real_unmatched']))

    # A slightly negative mu component is estimation noise, not a violation:
    # judge it relative to that group's own rate, then clip for the simulation.
    mu_frac = mu / np.maximum(R_g, 1e-12)
    bad_mu = bool((mu_frac < -0.05).any())
    print(f'mu as fraction of own group rate: ' +
          ' '.join(f'{GROUPS[i]}={mu_frac[i]:+.3f}' for i in range(G)))
    res['mu_frac_of_rate'] = mu_frac.tolist()

    if args.fit_only:
        print('\nFIT-ONLY: skipping simulation (composition fit)')
        res['verdict'] = 'FIT_ONLY'
    elif rho >= 1.0 or bad_mu:
        print('\nVERDICT: FAIL_UNSTABLE  (rho>=1, or immigration below -5% of a '
              "group's rate -- typed ground is not a valid subcritical Hawkes "
              'as fitted; no simulation run)')
        res['verdict'] = 'FAIL_UNSTABLE'
    else:
        rng = np.random.default_rng(args.seed)
        curves, rates = [], []
        for r in range(args.sim_reps):
            st = simulate_typed(M, Phi, np.asarray(edges * args.delta, float),
                                np.maximum(mu, 0.0), args.sim_duration, rng)
            if st is None:
                curves = None
                break
            rates.append(len(st) / args.sim_duration)
            curves.append(fano_curve(st, args.sim_duration, scales))
            print(f'  rep {r+1}: {len(st):,} ev  rate={rates[-1]:.2f}/s  '
                  f'Fano=' + ' '.join(f'{x:7.1f}' for x in curves[-1]), flush=True)
        if curves is None:
            print('\nVERDICT: FAIL_EXPLOSIVE (hard cap hit)')
            res['verdict'] = 'FAIL_EXPLOSIVE'
        else:
            C = np.array(curves)
            res['fano_sim_mean'] = C.mean(axis=0).tolist()
            res['fano_sim_sd'] = C.std(axis=0).tolist()
            res['sim_rate'] = float(np.mean(rates))
            ratio = C.mean(axis=0) / np.array(res['fano_real'])
            res['fano_ratio'] = ratio.tolist()
            print(f'\nFano sim   @{scales}: ' + ' '.join(f'{x:8.1f}' for x in res['fano_sim_mean']))
            print(f'ratio sim/real       : ' + ' '.join(f'{x:8.2f}' for x in ratio))
            print(f'rate sim={res["sim_rate"]:.2f}/s vs real={R_g.sum():.2f}/s')
            ok_rate = abs(res['sim_rate'] / R_g.sum() - 1) < 0.15
            ok_fano = bool(np.all((ratio > 0.4) & (ratio < 2.5)))
            res['verdict'] = ('PASS' if (ok_rate and ok_fano) else
                              'FAIL_DISPERSION' if ok_rate else 'FAIL_RATE')
            print(f'\nVERDICT: {res["verdict"]}  (rate_ok={ok_rate} fano_ok={ok_fano})')

    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

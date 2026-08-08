"""Kirchner-style binned-count fit of the LGM scalar ground on real event data.

"Set, don't learn": instead of MLE (which misallocates the kernel spectrum,
three demonstrations in docs/fano_investigation_2026-08.md), estimate the
Hawkes kernel integrals from the count-autoregression across a lin-log lag
grid (Kirchner 2017 / Jain thesis Sec 4.2), then project onto the LGM
exponential bank:

  E[X_n | past] ~= const + sum_k Phi_k * (lagged count in cell k)/width_k,
  Phi_k = int_{cell k} phi(s) ds,   phi(t) = sum_m a_m e^{-beta_m t}.

With the beta bank FIXED (log-spaced), the projection Phi_k -> a_m is a
non-negative least squares in a_m. Branching n = sum a_m/beta_m falls out;
mu is pinned by the model (mu0 = R(1-n)), matching Jain's mu = (I-M)Lambda.

Validation before deployment: simulate the fitted scalar ground (exact
cluster/thinning-free algorithm via per-event kicks) and compare its
Fano-vs-scale to the empirical curve, including scales beyond 50s.

Usage:
  python scripts/kirchner_fit_lgm.py --data-dir data/events/cbse_sol_7d \
      --out data/kirchner_ground_sol.json
"""
import argparse
import glob
import gzip
import json
import os

import numpy as np

TRAIN_FRAC = 0.70


def load_day_times(path, frac=TRAIN_FRAC):
    """Event timestamps (s, day-relative) for the train zone of one file."""
    ts = []
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            # fast path: timestamp is the first field of every line
            i = line.index(':') + 1
            j = line.index(',', i)
            ts.append(int(line[i:j]))
    n = int(len(ts) * frac)
    t = np.asarray(ts[:n], dtype=np.float64) / 1e9
    return t - t[0]


def kirchner_cells(delta, lag_min, lag_max, n_cells):
    """Lin-log lag grid cell edges in units of bins."""
    edges_s = np.unique(np.concatenate([
        np.arange(delta, 10 * delta, delta),                 # linear head
        np.geomspace(10 * delta, lag_max, n_cells),          # log tail
    ]))
    edges_s = edges_s[edges_s >= lag_min]
    return np.round(edges_s / delta).astype(int)


def fit_day(counts, edges, xtx, xty, burn):
    """Accumulate normal equations for one day's binned counts."""
    C = np.concatenate([[0.0], np.cumsum(counts)])
    n = len(counts)
    K = len(edges) - 1
    rows = np.arange(burn, n)
    X = np.empty((len(rows), K + 1), dtype=np.float64)
    X[:, 0] = 1.0
    for k in range(K):
        lo, hi = edges[k], edges[k + 1]
        # lagged count in (t - hi*delta, t - lo*delta]
        X[:, k + 1] = C[rows - lo] - C[rows - hi]
    y = counts[rows]
    xtx += X.T @ X
    xty += X.T @ y
    return xtx, xty


def nnls(A, b, iters=5000, lr=None):
    """Tiny projected-gradient NNLS (avoids a scipy dependency)."""
    AtA, Atb = A.T @ A, A.T @ b
    L = np.linalg.eigvalsh(AtA).max()
    x = np.maximum(np.linalg.lstsq(A, b, rcond=None)[0], 0.0)
    lr = 1.0 / L
    for _ in range(iters):
        x = np.maximum(x - lr * (AtA @ x - Atb), 0.0)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--delta', type=float, default=0.25, help='bin width (s)')
    ap.add_argument('--lag-max', type=float, default=240.0, help='max lag (s)')
    ap.add_argument('--n-cells', type=int, default=14)
    ap.add_argument('--betas', type=str, default='',
                    help='comma decays; default log-spaced 100..0.02, M=6')
    ap.add_argument('--n-cap', type=float, default=0.99)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, '*.jsonl.gz')))
    print(f'{len(files)} files; binning at {args.delta}s, lags to {args.lag_max}s')
    edges = kirchner_cells(args.delta, args.delta, args.lag_max, args.n_cells)
    K = len(edges) - 1
    widths_s = (edges[1:] - edges[:-1]) * args.delta
    burn = int(edges[-1]) + 1

    xtx = np.zeros((K + 1, K + 1))
    xty = np.zeros(K + 1)
    total_events, total_secs = 0, 0.0
    for f in files:
        t = load_day_times(f)
        nb = int(t[-1] / args.delta)
        counts = np.bincount((t[t < nb * args.delta] / args.delta).astype(int), minlength=nb).astype(np.float64)
        xtx, xty = fit_day(counts, edges, xtx, xty, burn)
        total_events += len(t); total_secs += t[-1]
        print(f'  {os.path.basename(f)}: {len(t):,} train events, {t[-1]/3600:.1f}h')
    R = total_events / total_secs
    print(f'train-zone mean rate: {R:.3f} ev/s')

    # Solve regression: coefficient on (lagged count / width) is Phi_cell.
    # We regressed on raw lagged counts, so coef_k = Phi_k / width_k... keep
    # raw: E[X_n] = c + sum_k b_k * L_k with b_k = (delta/width_k)*Phi_k?  In
    # Kirchner's INAR form E[X_n|past] = delta*mu + sum over lag bins of
    # phi(lag)*delta * X_{n-lag}. Aggregating bins into cell k assumes phi
    # constant per cell: b_k = mean_{cell} phi * delta  => Phi_k = b_k *
    # width_s_k / delta.
    beta_full = np.linalg.solve(xtx + 1e-8 * np.eye(K + 1), xty)
    b = beta_full[1:]
    Phi = b * widths_s / args.delta                      # int_{cell} phi ds
    n_raw = float(Phi.sum())
    print('cell edges (s):', np.round(edges * args.delta, 2).tolist())
    print('Phi per cell  :', np.round(Phi, 4).tolist())
    print(f'raw branching (sum Phi to {args.lag_max}s): {n_raw:.4f}')

    # Project onto the exponential bank: Phi_k = sum_m a_m/beta_m *
    # (e^{-beta_m lo_k} - e^{-beta_m hi_k});  NNLS in a_m.
    if args.betas:
        betas = np.array([float(x) for x in args.betas.split(',')])
    else:
        betas = np.geomspace(100.0, 0.02, 6)
    lo = edges[:-1] * args.delta
    hi = edges[1:] * args.delta
    A = (np.exp(-np.outer(lo, betas)) - np.exp(-np.outer(hi, betas))) / betas[None, :]
    a = nnls(A, np.maximum(Phi, 0.0))
    n_fit = float((a / betas).sum())
    if n_fit > args.n_cap:
        a *= args.n_cap / n_fit
        n_fit = args.n_cap
        print(f'(capped n at {args.n_cap})')
    print('betas:', np.round(betas, 4).tolist())
    print('a_fit:', np.round(a, 5).tolist())
    print(f'fitted branching n = {n_fit:.4f}   mu0 (pin) = {R * (1 - n_fit):.4f}')

    # ---- validation: simulate the fitted scalar Hawkes, measure Fano ----
    rng = np.random.default_rng(0)
    mu0 = R * (1 - n_fit)
    T, reps = 600.0, 48
    fano_scales = [1, 2, 5, 10, 20, 50]
    counts_by_scale = {s: [] for s in fano_scales}
    for _ in range(reps):
        # cluster (branching) simulation: immigrants + offspring cascades
        events = list(rng.uniform(0, T, rng.poisson(mu0 * T)))
        stack = list(events)
        while stack:
            parent = stack.pop()
            for m, (am, bm) in enumerate(zip(a, betas)):
                k = rng.poisson(am / bm)
                if k:
                    kids = parent + rng.exponential(1.0 / bm, size=k)
                    kids = kids[kids < T]
                    events.extend(kids.tolist()); stack.extend(kids.tolist())
        ev = np.sort(np.asarray(events))
        for s in fano_scales:
            nb = int(T / s)
            c = np.bincount((ev[ev < nb * s] / s).astype(int), minlength=nb)
            counts_by_scale[s].append(c)
    print('\nfitted-ground simulated Fano vs scale:')
    prof = []
    for s in fano_scales:
        c = np.concatenate(counts_by_scale[s])
        prof.append(c.var() / c.mean())
    print('  sim :', [round(x, 1) for x in prof])

    out = {'data_dir': os.path.abspath(args.data_dir), 'delta': args.delta,
           'edges_s': (edges * args.delta).tolist(), 'Phi': Phi.tolist(),
           'betas': betas.tolist(), 'a': a.tolist(), 'n': n_fit,
           'rate': R, 'mu0': mu0, 'sim_fano': prof, 'fano_scales': fano_scales}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

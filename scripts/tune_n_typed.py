#!/usr/bin/env python3
"""Marked-cluster n-tuner for TYPED-KICK LGM grounds.

With typed kicks the count process is a marked Hawkes: an event of channel k
kicks the ground by w_k (E[w]=1 under the empirical mark law), so its expected
offspring is w_k * n and the offspring-count variance across marks is
sigma^2 = n + n^2 (E[w^2] - 1). Cluster-size dispersion amplifies by
~E[w^2]/E[w]^2 (measured 30-150x on these fits), so the type-blind n*=0.99 is
far too hot -- this script re-tunes n by simulating the marked branching
representation and matching the pipeline-real Fano curve in log-MSE, exactly
mirroring the type-blind tuner protocol.

Branching representation (exact for linear marked Hawkes):
  immigrants ~ Poisson(mu0 = R(1-n)) with i.i.d. marks ~ p;
  an event with mark k spawns Poisson(w_k * n) children, delays ~ mixture of
  Exp(beta_m) with weights (a_m/beta_m)/sum(a_m/beta_m), marks i.i.d. ~ p.
(Marks i.i.d. ~ empirical p is the tuning approximation; the pipeline
 estimator on real rollouts remains the final arbiter, as with kf2.)

Usage: python3 scripts/tune_n_typed.py --asset sol --variant channel
"""
import argparse
import json

import numpy as np

SCALES = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
# Equal-duration-matched real Fano from the ma_cbse eval protocol (the yardstick
# every simulator arm is scored against).
REAL_FANO = {
    'sol': [42.3, 58.9, 94.8, 142.4, 233.1, 433.1],
    'btc': [55.4, 85.2, 157.1, 258.4, 432.9, 899.5],
    'eth': [78.7, 125.1, 245.1, 417.5, 748.3, 1571.2],
}

TYPES = ['LO', 'CO', 'MO', 'IS']
SIDES = ['b', 'a']


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


def kick_table(typed, variant):
    """w[62], renormalized so E[w]=1 under p_channel."""
    p = np.asarray(typed['p_channel'], float)
    if variant == 'channel':
        w = np.asarray(typed['w_channel'], float)
    else:
        names = fixed_names()
        gidx = {g: i for i, g in enumerate(typed['groups'])}
        w = np.array([typed['w_group'][gidx[f"{n.split('_')[0]}_{n.split('_')[1]}"]]
                      for n in names])
    w = np.clip(w, 1e-3, None)
    w = w / float(p @ w)                       # exact E[w]=1
    return w, p


def simulate_marked(mu0, n, betas, q_m, w, p, T, rng, max_events=6_000_000):
    """One realization of the marked branching process on [0, T]; returns times."""
    K = len(w)
    n_imm = rng.poisson(mu0 * T)
    t_imm = rng.uniform(0.0, T, n_imm)
    k_imm = rng.choice(K, size=n_imm, p=p)
    times = [t_imm]
    frontier_t, frontier_k = t_imm, k_imm
    total = n_imm
    while len(frontier_t):
        n_child = rng.poisson(w[frontier_k] * n)
        tot_c = int(n_child.sum())
        if tot_c == 0:
            break
        total += tot_c
        if total > max_events:
            return None                        # supercritical-in-practice guard
        parent_t = np.repeat(frontier_t, n_child)
        comp = rng.choice(len(betas), size=tot_c, p=q_m)
        child_t = parent_t + rng.exponential(1.0 / betas[comp])
        keep = child_t <= T
        child_t = child_t[keep]
        child_k = rng.choice(K, size=len(child_t), p=p)
        times.append(child_t)
        frontier_t, frontier_k = child_t, child_k
    return np.sort(np.concatenate(times))


def fano_curve(times, T, burn=500.0):
    t = times[(times >= burn) & (times <= T)] - burn
    span = T - burn
    out = []
    for s in SCALES:
        nb = int(span // s)
        c = np.bincount((t[t < nb * s] / s).astype(np.int64), minlength=nb)
        out.append(c.var() / max(c.mean(), 1e-9))
    return np.array(out)


def score(n, ground, w, p, reps, T, seed0):
    a = np.asarray(ground['a'], float)
    betas = np.asarray(ground['betas'], float)
    contrib = a / betas
    q_m = contrib / contrib.sum()
    R = float(ground['rate'])
    mu0 = R * (1.0 - n)
    curves = []
    for r in range(reps):
        rng = np.random.default_rng(seed0 + 7919 * r)
        times = simulate_marked(mu0, n, betas, q_m, w, p, T, rng)
        if times is None:
            return None, None
        curves.append(fano_curve(times, T))
    f = np.mean(curves, axis=0)
    return f, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', default='sol')
    ap.add_argument('--variant', choices=['channel', 'group'], default='channel')
    ap.add_argument('--typed-json', default=None)
    ap.add_argument('--ground-json', default=None)
    ap.add_argument('--reps', type=int, default=4)
    ap.add_argument('--duration', type=float, default=40000.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    typed = json.load(open(args.typed_json or f'data/kirchner_typed_{args.asset}.json'))
    ground = json.load(open(args.ground_json or f'data/kirchner_ground_{args.asset}.json'))
    w, p = kick_table(typed, args.variant)
    Ew2 = float(p @ w**2)
    real = np.array(REAL_FANO[args.asset])
    print(f'{args.asset}/{args.variant}: E[w]={p @ w:.4f} E[w^2]={Ew2:.1f} '
          f'(amplification vs blind ~{Ew2:.0f}x)  w[MO]={w[40]:.1f}/{w[41]:.1f}')

    def logmse(f):
        return float(np.mean((np.log(f) - np.log(real))**2))

    # coarse -> fine grid
    results = {}
    grid = [0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98]
    for phase in range(2):
        for n in grid:
            if n in results:
                continue
            f, _ = score(n, ground, w, p, args.reps, args.duration, args.seed)
            if f is None:
                print(f'  n={n:.4f}  EXPLODED (event cap)')
                results[n] = (np.inf, None)
                continue
            results[n] = (logmse(f), f)
            print(f'  n={n:.4f}  logMSE={results[n][0]:.4f}  fano={[round(x,1) for x in f]}')
        best = min((k for k in results if np.isfinite(results[k][0])),
                   key=lambda k: results[k][0])
        if phase == 0:
            lo = best - 0.02
            hi = min(best + 0.02, 0.995)
            grid = [round(x, 4) for x in np.linspace(lo, hi, 5)]
    best = min((k for k in results if np.isfinite(results[k][0])),
               key=lambda k: results[k][0])
    f = results[best][1]
    print(f'\nBEST n*={best:.4f}  logMSE={results[best][0]:.4f}')
    print(f'  sim  {[round(x,1) for x in f]}')
    print(f'  real {[round(x,1) for x in real]}')
    out = {'asset': args.asset, 'variant': args.variant, 'n_star': best,
           'Ew2': Ew2, 'sim_fano': f.tolist(), 'real_fano': real.tolist(),
           'grid': {str(k): results[k][0] for k in sorted(results)}}
    path = f'data/tuned_n_typed_{args.asset}_{args.variant}.json'
    json.dump(out, open(path, 'w'), indent=1)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()

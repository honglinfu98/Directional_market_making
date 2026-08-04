"""Fit per-channel order-volume distributions (Coletta-style, adapted to crypto).

For each of the 62 canonical channels (event_type x side x level) collect the
volumes of TRAIN-split events (first 70% of each file's lines, mirroring
BFNXDataset's per-file zones) and fit a spike+tail mixture:

  - spike: the top-M exact volume values with relative frequency >= spike-min-frac
    (crypto's analogue of Coletta's round-lot negative-binomial component:
    round sizes like 0.01 / 0.1 / 1 BTC and algo-typical repeats)
  - tail:  lognormal (closed-form MLE on logs) or gamma (moment-matched) on the
    remaining volumes

Thin channels (< min-count events) back off: channel -> type_side pool ->
type pool -> global. Output is a JSON consumed by dmm_sim.volume_sampler.

Usage:
  python scripts/fit_volumes.py --data-dir data/events/cbse_btc \
      --out data/volume_fits_cbse_btc.json --val-check
"""
import argparse
import glob
import gzip
import json
import math
import os
import sys
from array import array
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmm_sim.training.data_loader import _fixed_bfnx_event_names  # noqa: E402

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _count_lines(path):
    with _open(path) as f:
        return sum(1 for _ in f)


def collect(files, zone, max_lines=None):
    """zone: (frac_lo, frac_hi) of each file's lines. Returns name -> array('d')."""
    vols = defaultdict(lambda: array("d"))
    schema = set(_fixed_bfnx_event_names())
    for path in files:
        n = _count_lines(path)
        lo, hi = int(n * zone[0]), int(n * zone[1])
        if max_lines:
            hi = min(hi, lo + max_lines)
        with _open(path) as f:
            for i, line in enumerate(f):
                if i < lo:
                    continue
                if i >= hi:
                    break
                d = json.loads(line)
                for e in d["events"]:
                    name = f"{e['event_type']}_{e['side']}_L{e['level']}"
                    if name in schema and e["volume"] > 0:
                        vols[name].append(e["volume"])
        print(f"  {os.path.basename(path)}: lines [{lo},{hi}) done", flush=True)
    return {k: np.asarray(v, dtype=np.float64) for k, v in vols.items()}


def fit_mixture(v, spike_top_m, spike_min_frac, tail_family):
    """Spike+tail fit for one volume sample. Returns a JSON-able dict."""
    n = len(v)
    vals, counts = np.unique(v, return_counts=True)
    order = np.argsort(counts)[::-1]
    spike_idx = [i for i in order[:spike_top_m] if counts[i] / n >= spike_min_frac]
    spike_vals = vals[spike_idx]
    spike_counts = counts[spike_idx]
    p_spike = float(spike_counts.sum() / n)
    tail = v[~np.isin(v, spike_vals)] if len(spike_vals) else v
    fit = {
        "n": int(n),
        "p_spike": p_spike,
        "spike_values": [float(x) for x in spike_vals],
        "spike_probs": [float(c / spike_counts.sum()) for c in spike_counts] if len(spike_vals) else [],
        "tail_family": tail_family,
    }
    if len(tail) >= 10:
        if tail_family == "lognormal":
            lv = np.log(tail)
            fit["mu"], fit["sigma"] = float(lv.mean()), float(max(lv.std(), 1e-6))
        else:  # gamma, moment-matched
            m, s2 = float(tail.mean()), float(tail.var())
            fit["shape"] = m * m / max(s2, 1e-18)
            fit["scale"] = max(s2, 1e-18) / m
    else:
        # nearly-degenerate channel: everything is spike
        fit["p_spike"] = 1.0
        if not len(spike_vals):
            fit["spike_values"] = [float(np.median(v))]
            fit["spike_probs"] = [1.0]
    return fit


def ks_2samp(a, b):
    """Two-sample KS statistic (no scipy dependency)."""
    a, b = np.sort(a), np.sort(b)
    both = np.concatenate([a, b])
    ca = np.searchsorted(a, both, side="right") / len(a)
    cb = np.searchsorted(b, both, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-lines-per-file", type=int, default=None)
    ap.add_argument("--spike-top-m", type=int, default=16)
    ap.add_argument("--spike-min-frac", type=float, default=0.005)
    ap.add_argument("--min-count", type=int, default=500)
    ap.add_argument("--tail", choices=["lognormal", "gamma"], default="lognormal")
    ap.add_argument("--val-check", action="store_true",
                    help="report per-channel KS of sampler draws vs val-split volumes")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.jsonl.gz")) +
                   glob.glob(os.path.join(args.data_dir, "*.jsonl")))
    if args.max_files:
        files = files[: args.max_files]
    print(f"{len(files)} files; collecting TRAIN zone [0,{TRAIN_FRAC})")
    vols = collect(files, (0.0, TRAIN_FRAC), args.max_lines_per_file)

    channels = _fixed_bfnx_event_names()
    out = {"meta": {"data_dir": os.path.abspath(args.data_dir), "files": [os.path.basename(f) for f in files],
                    "train_frac": TRAIN_FRAC, "tail_family": args.tail,
                    "spike_top_m": args.spike_top_m, "spike_min_frac": args.spike_min_frac,
                    "min_count": args.min_count},
           "channels": {}, "pools": {}}

    # pooled fits for backoff
    pools = defaultdict(list)
    for name, v in vols.items():
        etype, side, _ = name.split("_")
        pools[f"{etype}_{side}"].append(v)
        pools[etype].append(v)
        pools["__global__"].append(v)
    for pname, arrs in pools.items():
        pv = np.concatenate(arrs)
        if len(pv):
            out["pools"][pname] = fit_mixture(pv, args.spike_top_m, args.spike_min_frac, args.tail)

    n_direct = n_backoff = 0
    for name in channels:
        v = vols.get(name, np.empty(0))
        etype, side, _ = name.split("_")
        if len(v) >= args.min_count:
            out["channels"][name] = {"count": int(len(v)), "backoff": None,
                                     "fit": fit_mixture(v, args.spike_top_m, args.spike_min_frac, args.tail)}
            n_direct += 1
        else:
            for pool in (f"{etype}_{side}", etype, "__global__"):
                if pool in out["pools"]:
                    out["channels"][name] = {"count": int(len(v)), "backoff": pool, "fit": None}
                    break
            n_backoff += 1
    print(f"fitted {n_direct} channels directly, {n_backoff} back off to pools")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")

    if args.val_check:
        from dmm_sim.volume_sampler import VolumeSampler
        sampler = VolumeSampler(args.out)
        rng = np.random.default_rng(0)
        print(f"\nVAL check (zone [{TRAIN_FRAC},{TRAIN_FRAC + VAL_FRAC})): KS sampler-vs-real")
        val = collect(files, (TRAIN_FRAC, TRAIN_FRAC + VAL_FRAC), args.max_lines_per_file)
        rows = []
        for name, v in sorted(val.items(), key=lambda kv: -len(kv[1])):
            if len(v) < 200:
                continue
            v = v[:200_000]
            draws = sampler.sample(name, len(v), rng)
            rows.append((name, len(v), ks_2samp(v, draws)))
        print(f"{'channel':<12}{'val n':>10}{'KS':>8}")
        for name, n, ks in rows[:15]:
            print(f"{name:<12}{n:>10,}{ks:>8.3f}")
        med = float(np.median([r[2] for r in rows]))
        print(f"... {len(rows)} channels checked, median KS = {med:.3f}")


if __name__ == "__main__":
    main()

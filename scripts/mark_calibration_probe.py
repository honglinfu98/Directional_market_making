#!/usr/bin/env python3
"""Mark-calibration diagnostic (markcal).

Decomposes the E_p_model[w] ~= 0.77 deficit recorded in the typed-kick probe
(doc 4.15) into its two candidate causes, per asset/seed, on the SHIPPED
lgm-kf2 mark checkpoints.

The typed assembly normalises the kick table so that E[w] = 1 under the
EMPIRICAL mark law p_channel (tune_n_typed.kick_table) and freezes that law
into the p_bar buffer (assemble_typed_ckpt).  But the ground actually fires
according to the MODEL's realised mark law.  If the model under-produces the
heavily-weighted MO channels, the realised mean kick is < 1 and the ground
under-fires at baseline -- which is exactly the kappa ~= 1.3 seen in every
typed arm.

Measured over the 62 channels:
  p_emp  empirical frequencies on the eval (test) stream
  p_tf   mean softmax(item_logits) over genuine test events  (REAL states)
  p_roll realised frequencies in free rollout                 (MODEL states)
and for each, E[w] = p @ w under the fitted Kirchner kick tables (channel and
group variants) plus the aggregate MO mass, which carries most of the
excitation (p_MO * w_MO is ~3/4 of E[w] on SOL).

Reading the result:
  E_tf ~= 1  and  E_roll < 1   -> free-rollout drift (exposure / compounding);
                                  a post-hoc reweighting will NOT fix it.
  E_tf < 1   (~= E_roll)       -> head miscalibration on rare classes; a
                                  post-hoc logit adjustment on the mark head
                                  is sufficient, stays a softmax, and so
                                  preserves rate-neutrality and the transplant.
Both off by different amounts -> report the split; fix the head first, then
                                  re-measure the residual drift.

Rate-neutrality note: this probe does NOT calibrate kappa.  The mark head is a
softmax over the backbone state, so a uniform rate rescaling only perturbs the
mark law through timing; kf2's own kappa is 1.014-1.019, i.e. within 2% of 1,
so the uncalibrated rollout is the right measurement for mark PROPORTIONS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from volume_set_mtpp.training.data_loader import create_bfnx_dataloaders
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
from volume_set_mtpp.evaluation.stylized_facts import real_stream, simulate_stream
from volume_set_mtpp.evaluation.world_model_diagnostics import get_device, move_batch

SIDES = ("b", "a")


def fixed_names():
    """Canonical 62-channel order (matches bfnx_loader and the typed fits)."""
    names = []
    for et in ("LO", "CO"):
        for s in SIDES:
            for l in range(1, 11):
                names.append(f"{et}_{s}_L{l}")
    for s in SIDES:
        names.append(f"MO_{s}_L1")
    for s in SIDES:
        for l in range(1, 11):
            names.append(f"IS_{s}_L{l}")
    return names


def kick_table(typed, variant):
    """w[62] renormalised so E[w] = 1 under the fit's own p_channel.

    Identical to tune_n_typed.kick_table -- kept inline so the probe has no
    cross-repo import.
    """
    p = np.asarray(typed["p_channel"], float)
    names = fixed_names()
    if variant == "channel":
        w = np.asarray(typed["w_channel"], float)
    else:
        gidx = {g: i for i, g in enumerate(typed["groups"])}
        w = np.array(
            [
                typed["w_group"][gidx[f"{n.split('_')[0]}_{n.split('_')[1]}"]]
                for n in names
            ]
        )
    w = np.clip(w, 1e-3, None)
    w = w / float(p @ w)
    return w, p


def _norm(counts):
    tot = float(counts.sum())
    return counts / tot if tot > 0 else counts


def empirical_p(loader, stride, max_windows):
    marks, _dt = real_stream(loader, stride, max_windows)          # [N, K] bool
    return _norm(marks.sum(axis=0).astype(float)), int(marks.shape[0])


@torch.no_grad()
def teacher_forced_p(model, loader, device, max_batches):
    """Mean softmax over genuine events, evaluated on REAL states."""
    acc = None
    n_ev = 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = move_batch(batch, device)
        im = batch["input_marks"].float()
        ts = torch.cumsum(batch["input_times"].float().clamp_min(0.0), dim=1)
        _states, left = model.decoder.get_states_and_event_left_states(im, ts)
        d = model.get_total_intensity_and_items(left)
        logits = d["item_logits"]                                   # [B,N,K]
        ev = im.sum(dim=-1) > 0                                     # genuine only
        if not bool(ev.any()):
            continue
        p = torch.softmax(logits[ev].float(), dim=-1).sum(dim=0)    # [K]
        acc = p if acc is None else acc + p
        n_ev += int(ev.sum())
    if acc is None:
        raise RuntimeError("no genuine events found for teacher-forced pass")
    return (acc / n_ev).cpu().numpy().astype(float), n_ev


def rollout_p(model, batch, device, duration, n_seq, seed):
    marks, _dt, cum = simulate_stream(
        model, batch, device,
        steps=0, n_seq=n_seq, horizon=60.0, n_grid=32, seed=seed,
        duration=duration, carried=True,
    )
    # truncate each sequence at `duration` -- simulate_stream keeps stepping
    # past it for sequences that finish early.
    keep = cum <= duration                                          # [n, S]
    counts = (marks & keep[:, :, None]).sum(axis=(0, 1)).astype(float)
    return _norm(counts), int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--typed-json", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--max-files", type=int, default=7)
    ap.add_argument("--seq-length", type=int, default=4096)
    ap.add_argument("--stride", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tf-batches", type=int, default=200)
    ap.add_argument("--max-real-windows", type=int, default=4096)
    ap.add_argument("--rollout-duration", type=float, default=600.0)
    ap.add_argument("--rollout-sequences", type=int, default=32)
    ap.add_argument("--rollout-seed", type=int, default=1)
    args = ap.parse_args()

    device = get_device(args.device)
    dl_kwargs = {}
    if args.cache_dir:
        dl_kwargs["cache_dir"] = args.cache_dir
    _tr, _va, test_loader, em = create_bfnx_dataloaders(
        args.data_dir, args.batch_size, args.seq_length, args.stride,
        args.max_files, **dl_kwargs
    )

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = create_volume_set_mtpp(
        em.num_events, cfg, device,
        use_volume=cfg.get("use_volume", True),
        intensity_type=cfg.get("intensity_type", "dynamic"),
    )
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()

    names = fixed_names()
    if em.num_events != len(names):
        raise SystemExit(
            f"channel count {em.num_events} != canonical 62; "
            "the fixed_names() order would be misaligned with the kick table"
        )
    mo_idx = [i for i, n in enumerate(names) if n.startswith("MO_")]

    typed = json.loads(Path(args.typed_json).read_text())

    print(f"[{args.label}] empirical ...", flush=True)
    p_emp, n_real = empirical_p(test_loader, args.stride, args.max_real_windows)
    print(f"[{args.label}] teacher-forced ...", flush=True)
    p_tf, n_tf = teacher_forced_p(model, test_loader, device, args.tf_batches)
    print(f"[{args.label}] rollout ...", flush=True)
    first_batch = move_batch(next(iter(test_loader)), device)
    p_roll, n_roll = rollout_p(
        model, first_batch, device,
        args.rollout_duration, args.rollout_sequences, args.rollout_seed,
    )

    res = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "counts": {"real_events": n_real, "tf_events": n_tf, "rollout_events": n_roll},
        "p_MO": {
            "emp": float(p_emp[mo_idx].sum()),
            "tf": float(p_tf[mo_idx].sum()),
            "roll": float(p_roll[mo_idx].sum()),
        },
        "variants": {},
        "p_emp": p_emp.tolist(),
        "p_tf": p_tf.tolist(),
        "p_roll": p_roll.tolist(),
        "channel_names": names,
    }

    for variant in ("group", "channel"):
        w, p_fit = kick_table(typed, variant)
        e_fit = float(p_fit @ w)          # 1.0 by construction -- sanity check
        e_emp = float(p_emp @ w)
        e_tf = float(p_tf @ w)
        e_roll = float(p_roll @ w)
        res["variants"][variant] = {
            "E_w_fit_pchannel": e_fit,
            "E_w_emp_test": e_emp,
            "E_w_tf": e_tf,
            "E_w_roll": e_roll,
            "mo_share_of_Ew_emp": float(p_emp[mo_idx] @ w[mo_idx] / max(e_emp, 1e-12)),
            "implied_kappa_tf": 1.0 / e_tf if e_tf > 0 else None,
            "implied_kappa_roll": 1.0 / e_roll if e_roll > 0 else None,
        }
        print(
            f"[{args.label}] {variant:>7}  E[w]  fit={e_fit:.4f}  emp_test={e_emp:.4f}  "
            f"tf={e_tf:.4f}  roll={e_roll:.4f}   "
            f"kappa_implied roll={1.0/e_roll if e_roll>0 else float('nan'):.4f}",
            flush=True,
        )

    print(
        f"[{args.label}] p(MO)  emp={res['p_MO']['emp']:.5f}  "
        f"tf={res['p_MO']['tf']:.5f}  roll={res['p_MO']['roll']:.5f}",
        flush=True,
    )

    # Verdict: attribute the deficit. Uses the group variant (the shrunk fit
    # that survived calibration in 4.15).
    g = res["variants"]["group"]
    d_tf = 1.0 - g["E_w_tf"]
    d_roll = 1.0 - g["E_w_roll"]
    if abs(d_roll) < 0.05:
        verdict = "NO_DEFICIT (realised E[w] within 5% of 1; kappa!=1 came from elsewhere)"
    elif abs(d_tf) < 0.05:
        verdict = "ROLLOUT_DRIFT (head is calibrated on real states; deficit is closed-loop)"
    elif abs(d_tf - d_roll) < 0.05 * max(abs(d_roll), 1e-9):
        verdict = "HEAD_MISCALIBRATION (deficit already present teacher-forced)"
    else:
        frac = d_tf / d_roll if d_roll else float("nan")
        verdict = f"MIXED (teacher-forced accounts for {frac:.0%} of the rollout deficit)"
    res["verdict"] = verdict
    print(f"[{args.label}] VERDICT {verdict}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(res, indent=2))
    print(f"[{args.label}] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

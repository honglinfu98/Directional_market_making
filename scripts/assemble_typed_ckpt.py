#!/usr/bin/env python3
"""Assemble a TYPED-KICK LGM checkpoint from a trained mark donor.

Takes a trained lgm mark checkpoint (rate-neutral marks untouched), enables
typed kicks in the config, and transplants:
  - the Kirchner ground (a scaled to the marked-cluster n*, betas, mu0 pin),
  - the fitted kick table w (E[w]=1 under empirical p), and
  - the empirical mark frequencies into the p_bar buffer (frozen in eval).
Mark training is teacher-forced and reads only the backbone slice, so the
transplant is exact for the marks; only the time law changes.

Run on the cluster (needs torch + repo on PYTHONPATH):
  python3 assemble_typed_ckpt.py --donor .../lgm-w4k48-s1/train/best_model.pt \
      --ground kirchner_ground_sol.json --typed kirchner_typed_sol.json \
      --variant channel --n 0.80 --out .../lgm-tkc80-s1/train/best_model.pt
"""
import argparse
import json
import os

import numpy as np
import torch

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
    p = np.asarray(typed['p_channel'], float)
    if variant == 'channel':
        w = np.asarray(typed['w_channel'], float)
    else:
        gidx = {g: i for i, g in enumerate(typed['groups'])}
        w = np.array([typed['w_group'][gidx[f"{n.split('_')[0]}_{n.split('_')[1]}"]]
                      for n in fixed_names()])
    w = np.clip(w, 1e-3, None)
    w = w / float(p @ w)
    return w, p


def inv_softplus(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return torch.log(torch.expm1(x.clamp_min(1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--donor', required=True)
    ap.add_argument('--ground', required=True)
    ap.add_argument('--typed', required=True)
    ap.add_argument('--variant', choices=['channel', 'group'], required=True)
    ap.add_argument('--n', type=float, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ground = json.load(open(args.ground))
    typed = json.load(open(args.typed))
    w, p = kick_table(typed, args.variant)

    ck = torch.load(args.donor, map_location='cpu', weights_only=False)
    cfg = ck['config']
    assert cfg.get('decoder_type') == 'lgm', cfg.get('decoder_type')
    cfg['lgm_typed_kicks'] = True
    state = ck['model_state_dict']

    # ground transplant: scale a to the marked-cluster n* (E[w]=1 keeps
    # n = sum a/beta exact), betas via the stable inverse-softplus form
    a_fit = np.asarray(ground['a'], float)
    betas = np.asarray(ground['betas'], float)
    n_fit = float((a_fit / betas).sum())
    a = a_fit * (args.n / n_fit)
    from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp
    model = create_volume_set_mtpp(cfg.get('num_channels', 62), cfg,
                                   torch.device('cpu'),
                                   use_volume=cfg.get('use_volume', False))
    d = model.decoder
    assert d.M == len(betas)
    with torch.no_grad():
        dd = (torch.tensor(betas) - d.min_decay).clamp_min(1e-3)
        state['decoder.log_delta_g'] = (dd + torch.log(-torch.expm1(-dd))).to(
            state['decoder.log_delta_g'].dtype)
        state['decoder.a_raw'] = inv_softplus(a).to(state['decoder.a_raw'].dtype)
        state['decoder.kick_raw'] = inv_softplus(w).float()
        state['decoder.p_bar'] = torch.tensor(p, dtype=torch.float32)

    model.load_state_dict(state)          # strict: verifies key sets match
    d = model.decoder
    ew = float((d.p_bar * torch.nn.functional.softplus(d.kick_raw)).sum())
    print(f'ASSEMBLED variant={args.variant} n_target={args.n} '
          f'closed_form_rho={d.closed_form_rho():.4f} E[w]={ew:.4f} '
          f'w[MO]={float(torch.nn.functional.softplus(d.kick_raw[40])):.1f}/'
          f'{float(torch.nn.functional.softplus(d.kick_raw[41])):.1f} '
          f'target_rate={float(d.target_rate):.4f}')
    assert abs(d.closed_form_rho() - args.n) < 5e-3

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ck['model_state_dict'] = state
    ck['config'] = cfg
    torch.save(ck, args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

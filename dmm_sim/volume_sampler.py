"""Per-channel order-volume sampler (spike + tail mixture, fitted offline).

Companion to scripts/fit_volumes.py. Once the MTPP has sampled an event
channel k, draw its volume:

    sampler = VolumeSampler("data/volume_fits_cbse_btc.json")
    v = sampler.sample("LO_b_L1", 1, rng)          # by name
    v = sampler.sample_idx(k, 1, rng)              # by channel index (0..61)

With probability p_spike the draw is one of the channel's high-frequency exact
sizes (multinomial); otherwise it comes from the fitted lognormal/gamma tail.
Thin channels resolve through their recorded backoff pool automatically.
"""
import json
from typing import Union

import numpy as np

from .training.data_loader import _fixed_bfnx_event_names


class VolumeSampler:
    def __init__(self, fits_path: str):
        with open(fits_path) as f:
            self._doc = json.load(f)
        self._names = _fixed_bfnx_event_names()
        self._fits = {}
        for name, entry in self._doc["channels"].items():
            fit = entry["fit"] if entry["fit"] is not None else self._doc["pools"][entry["backoff"]]
            self._fits[name] = fit

    def fit_for(self, channel: Union[int, str]) -> dict:
        name = self._names[channel] if isinstance(channel, int) else channel
        return self._fits[name]

    def sample(self, channel: Union[int, str], size: int = 1,
               rng: np.random.Generator = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        fit = self.fit_for(channel)
        out = np.empty(size, dtype=np.float64)
        spike = rng.random(size) < fit["p_spike"]
        n_spike = int(spike.sum())
        if n_spike and fit["spike_values"]:
            out[spike] = rng.choice(fit["spike_values"], size=n_spike, p=fit["spike_probs"])
        elif n_spike:
            spike[:] = False
        n_tail = int((~spike).sum())
        if n_tail:
            if fit["tail_family"] == "lognormal" and "mu" in fit:
                out[~spike] = rng.lognormal(fit["mu"], fit["sigma"], size=n_tail)
            elif fit["tail_family"] == "gamma" and "shape" in fit:
                out[~spike] = rng.gamma(fit["shape"], fit["scale"], size=n_tail)
            elif fit["spike_values"]:  # degenerate: no tail fit stored
                out[~spike] = rng.choice(fit["spike_values"], size=n_tail, p=fit["spike_probs"])
            else:
                raise ValueError(f"channel {channel}: no tail fit and no spikes")
        return out

    def sample_idx(self, k: int, size: int = 1, rng: np.random.Generator = None) -> np.ndarray:
        return self.sample(self._names[k], size, rng)

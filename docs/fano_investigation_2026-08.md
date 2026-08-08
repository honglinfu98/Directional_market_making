# The Fano Investigation: Closing the Dispersion Gap in Neural LOB Simulators

**Lab log / paper source material — 2026-08-04 → 2026-08-07.**
All experiments on the UCL cluster under `~/simulation/experiments/ma_cbse/<coin>/<tag>/`;
code in `honglinfu98/simulation` (commits `7b97b21..88d22cc`) mirrored in this repo's `dmm_sim/`.

---

## 1. Problem statement

A neural MTPP world model of Coinbase order flow (62-channel LOB events) must not only
predict the next event but *simulate* order flow whose count dispersion matches reality.
The headline statistic is the **Fano factor** F(Δ) = Var(N_Δ)/E[N_Δ] of bucketed event
counts across scales Δ ∈ [1, 50] s. Two identities organize everything:

- **Law of total variance (Cox form):** F(Δ) = 1 + Var(Λ_Δ)/E[Λ_Δ], where Λ_Δ = ∫λ dt.
  All excess dispersion is variance of the *integrated intensity*; post-hoc rate
  rescaling (λ→κλ) scales F−1 linearly and cannot repair relative fluctuation structure.
- **Branching form (linear Hawkes):** F(∞) = 1/(1−n)², with n the branching ratio.
  Cluster sizes S satisfy E[S] = 1/(1−n), E[S²] = 1/(1−n)³ ⇒ compound-Poisson Fano
  E[S²]/E[S]. The rising-with-Δ profile is the fingerprint of long-memory (power-law-like)
  intensity autocovariance; single-exponential kernels plateau.

Real Coinbase (7 days, Jan 2026): F(1s) ≈ 40–70 rising to **F(50s) ≈ 370 (SOL),
1250 (ETH), 1460 (BTC)**, still rising at the window edge. Implied n: SOL ≈ 0.948,
ETH ≈ 0.971, BTC ≈ 0.973 (lower bounds).

## 2. Data and asset heterogeneity (the tick-regime contrast)

7 days (2026-01-01..07), Coinbase spot, `cbse_{btc,eth,sol}_7d` (the corrected builds:
the home-dir `cbse/` events predate the trade-windowing fix and are MO-free — 3 vs 884
MO per 500k lines; do not use).

| | trades (7d) | book update rows | updates/trade | rate (train) | 1024-ev window |
|---|---|---|---|---|---|
| BTC | 41,675 | 52.9M | **1268:1** | 40.6 ev/s | ~25 s |
| ETH | 94,048 | 55.5M | 590:1 | 42.8 ev/s | ~24 s |
| SOL | 40,048 | 20.1M | 501:1 | 22.3 ev/s | ~46 s |

Event-type mix (same day, 400k events): BTC **52% IS** / 38% LO / 9% CO; ETH 34% IS;
SOL **7% IS / 43% CO**. BTC/ETH are small-tick assets (price-priority undercutting wars
inside a multi-tick spread — the most self-exciting flow); SOL is effectively large-tick
(1-tick spread, queue-priority churn). This microstructural contrast, not just rate,
explains the criticality ordering and why every simulator finds SOL easiest. Rate-
normalized dispersion (F(50)/rate): BTC ≈ 35, SOL ≈ 15.

## 3. Why SS2P2 under-disperses: the mechanism, measured

SS2P2 = shared S2P2 SSM backbone → two decoupled heads: softmin-bounded scalar rate
λ(t) = s·softplus(c − softplus(c − wᵀh(u))) ≤ s·softplus(c), and rate-neutral softmax
marks. Stability is a bound on the rate *level*; nothing constrains the event→intensity
feedback.

**Impulse-response measurement** (inject one extra event into real histories, integrate
the extra intensity mass over 20 s; `_empirical_branching`): across six trained Coinbase
checkpoints the median response is **≈ 0 to negative** (−8.7…+0.2), positive in only
27–56% of states, sign flips across seeds. Post-event intensity is pinned at the ceiling
(~390 ev/s at seq-1024 configs) in every window — the marginal response saturates exactly
where bursts live. Conclusion: *SS2P2 as trained is not reliably self-exciting*; its
residual dispersion (F(1s) ≈ 15–27) comes from the stereotyped impulse-decay envelope,
not compounding cascades.

## 4. Arms and results (all: ma_cbse protocol — 12 epochs unless noted, seq/stride 1024,
batch 64, TBPTT, categorical marks, streaming genuine eval on every test event,
val-calibrated 600 s rollouts × 3 seeds × 3 rollout seeds)

### 4.1 ss2p2-lob (top-10 book conditioning; 6 ladder-derived features)
Prediction: statistically identical to baseline (BTC acc 0.4484 vs 0.4478). Simulation:
*worse* on volatility-structure facts (F6 0.162 vs 0.276; kurtosis 9.1 vs 21.3) — extra
state conditioning becomes a smoothing feedback path in closed loop. Book-summary
conditioning is not the Fano lever. (Also matches Coletta et al.: the world-agent
literature likewise uses low-dim summaries, cutoffs ≤ 5 levels.)

### 4.2 LGM (linear Hawkes ground × SS2P2 marks, shared backbone)
λ_k = (μ₀ + Σ_m a_m S^m(t))·softmax(z_k(u)); n = Σ a_m/β_m exact and projected;
**pin μ₀ = R(1−n)** makes the free-rollout mean rate exact by construction (validated:
κ = 1.01 in calibration vs SS2P2's κ ≈ 0.52). Two porting incidents worth recording:
1. A log-domain clamp capped the ground accumulators at 1.0 — silently erasing burst
   accumulation (n stalled at 0.30; after fix MLE hit the 0.95 projection boundary in
   one epoch).
2. **Cold-start bias / epoch-1 trap:** ground accumulators cold-started at 0, but their
   stationary mean is E[S] = R/β (≈ 600 for slow kernels). Windowed validation therefore
   *punished* slow-memory solutions increasingly as training improved; val loss rose
   monotonically after epoch 1 and best-model selection froze every run at epoch 1.
   Fix: stationary-mean cold start S₀ = R/β wherever no carried state exists
   (`stationary_ground()`, simulation@2f4dd00). After the fix: 6–12 best-saves, val
   improving through epoch 12.

Fixed-LGM results (SOL): marks near-parity (0.186 vs 0.192), timing worse (−2.23 vs
−2.91 nats; KS 0.46 — the 8-parameter clock's real cost), Fano **shape right, level low**
(2.3→59.8, rise ×26, seed sd ±2 vs SS2P2's ±217 — dispersion as a *property of the
certified parameters*, not seed luck).

### 4.3 lgm-k, "Konark arm" (typed kicks w_k per channel, M=6 log-spaced kernels
100→0.02 /s, n projected 0.97)
Prediction improved to best-in-class on SOL (0.1967 > SS2P2 0.1921). Fano **regressed**
(1.7→34.1): with more slow-kernel capacity available, MLE moved the branching mass to
β ≈ 0.008 (~2-min memory), pushing variance beyond the 50 s observation window. Third
consecutive demonstration (with 4.2's β-drift and 4.4's grid limit) that **one-step
likelihood systematically misallocates the kernel spectrum from dispersion's viewpoint**.

### 4.4 ss2p2-disp (dispersion-matching aux loss @ {0.5,2,8} s, frozen log-spaced
spectrum, cap 12)
Teacher-forced loss: F_model(Δ) = 1 + Var(Λ_Δ)/E[Λ_Δ] matched (log-space) to the batch-
empirical Var(N_Δ)/E[N_Δ]. Result: level pinned reproducibly at trained scales
(F(1s) = 15.7, sd ±14 at 50 s vs baseline ±217), **flat beyond the grid** (×2.2 rise);
zero prediction cost. Also: cap 12 produced one calibration failure of 3 (bisection
bracket [0.5357, 0.5365] non-verifiable — the documented near-critical rescaling
fragility of the loosely-capped family). Lesson: teacher-forced variance matching ≠
closed-loop compounding; and the loss's gradient is dead on saturated events.

### 4.5 ss2p2-w4k — the window-coverage result (headline)
Hypothesis: at seq 1024 the TBPTT gradient truncates at ~24–25 s on BTC/ETH — *before*
the 20–50 s scales where their simulated profiles flatten; SOL (~46 s windows) covers
the range and produced real-like rises in 2/3 seeds. Test: seq/stride 4096 (~100 s BTC/
ETH, ~184 s SOL), recipe otherwise identical.

Per-seed Fano, 1s → 50s (real BTC 70 → 1463, ×20.9):

| BTC | seq1024 | seq4096 |
|---|---|---|
| s1 | 11.7→28.9 (×2.5) | **28.2→900.6 (×31.9)** |
| s2 | 13.6→49.8 (×3.7) | 18.2→269.8 (×14.8) |
| s3 | 25.9→361.5 (×13.9) | 26.6→744.4 (×28.0) |

**All three BTC seeds now rise steeply** (best: 1.6× under real at 50 s); ETH 2/3
transformed (best 477 vs real 1247); SOL unchanged (negative control — its windows
already covered the scales). The seed lottery on BTC was gradient starvation at
unobserved scales, not irreducible criticality-avoidance. Cost: ~3pp accuracy on every
asset (stride-matched seq quarters the update count → undertraining; equal-updates
control below). Figures: `docs/figs/fano_w4k.png`, `docs/figs/fano_three_assets.png`.

### 4.6 Diagnostics on the w4k checkpoint (BTC s1) — what the residual ~2× offset is
1. **Ceiling saturation:** dominating rate 359 ev/s; λ at events p50 = 224,
   **p90 = 357, p99 = 358**; 32.8% of events above 0.9×ceiling. A massive atom AT the
   cap: bursts are clipped, and gradients (incl. any dispersion loss) are dead there.
2. **Teacher-forced level:** model-implied F is only **1.2–1.4× below real** at every
   scale (e.g. 50 s: 240 vs 290). Conditioned on real history the model already carries
   ~75% of real dispersion; the closed-loop gap (1.6–2.3×) adds modest exposure drift.

⇒ The remaining level offset is **primarily the cap**, secondarily closed-loop drift;
not an amplitude-learning failure. Next lever: cap 9 on the w4k48 recipe (12 was
calibration-fragile; 6 is clipping a third of events).

### 4.7 The equal-updates controls (completed 2026-08-07/08) — the training-path
trade-off

`ss2p2-w4k48` (seq 4096, 48 epochs = the baseline's exact total update count) and
`ss2p2-w4kd` (same + dispersion loss, grid {0.5,2,8,32} s). BTC four-way (per-seed
F(50s); real 70 → 1463):

| arm | acc | F(1s) | F(50s) per seed |
|---|---|---|---|
| base 1024/12ep | 0.4478 | 17.1 | 29, 50, 361 |
| w4k 4096/12ep | 0.4183 | 24.3 | **901, 270, 744** |
| w4k48 4096/48ep | **0.4514** | 16.6 | 232, 87, 230 |
| w4kd +disp/48ep | **0.4519** | 11.1 | 52, 43, 43 |

Findings (same pattern on ETH/SOL with their usual seed lottery; one ETH w4kd
outlier at 852):
1. **Accuracy fully recovers and exceeds baseline** at equal updates on every asset
   (BTC 0.4514, ETH 0.3451, SOL 0.1950) — the w4k drop was pure undertraining, and
   long windows are strictly better predictors at matched budget.
2. **The near-real w4k dispersion was partly an early-stopping artifact**: 4x more
   updates pulled BTC back from [901,270,744] to [232,87,230] (still ~3x the seq-1024
   baseline). Sharpest form of the arc's core lesson: **along one model's MLE
   trajectory, dispersion peaks early and accuracy peaks late — no NLL-selected
   checkpoint has both.** The prediction/simulation tension exists within a single
   training run, not just across architectures. (The w4k 12-epoch checkpoints are
   the de-facto "dispersion-optimal early stop"; a paper can present the frontier.)
3. **The dispersion loss under cap 6 suppresses long-scale Fano** (BTC w4kd: flat
   43–52, tightly converged) — as predicted by the saturation diagnostic (§4.6): its
   gradient is dead on the ~33% of events at the ceiling, and its teacher-forced
   targets prefer the smooth solution beyond genuinely-compounding scales. Best
   accuracy of all arms; the "loss alone" path to dispersion is closed.

### 4.8 The cap-9 control (completed 2026-08-08) — hypothesis refuted, branch closed

`ss2p2-w4kc9` (cap 9 + w4k48 recipe, 9 tasks, all STATUS=0, all calibrations
verified — no fragility at 9). F(50s) per seed vs the cap-6 w4k48 control:

| | cap 6 (w4k48) | cap 9 (w4kc9) | real |
|---|---|---|---|
| BTC | 232, 87, 230 | 47, 59, 112 | 1463 |
| ETH | 544, 51, 123 | 160, 14, 60 | 1247 |
| SOL | 301, 68, 81 | 228, 10, 59 | 394 |

Accuracy identical (BTC 0.451, ETH 0.346, SOL 0.195). **The saturation hypothesis
for the training-path re-smoothing is refuted**: extra headroom does not keep
48-epoch MLE at the burst-faithful solution — if anything the fully-trained optimum
is smoother with a looser cap. The drift is intrinsic to bounded-head likelihood
training. (Ceiling saturation remains the correct explanation for the *rollout
clipping* of the early-stopped w4k model — the two findings coexist.)

**Per the pre-registered decision rule, the SS2P2 branch closes here.** Its final
exhibits: (a) the window-coverage result (§4.5) — gradient horizon must cover the
measured scales; (b) the training-path dispersion/accuracy frontier (§4.7) — the
dispersion-optimal SS2P2 is the seq-4096/12-epoch early stop (BTC F(50s) up to 901,
1.6x under real), and no NLL-selected checkpoint matches it; (c) the certified
LGM/Kirchner route is the paper's answer for *controlled* dispersion.

## 5. The Jain (Konark) reference point

Sources: impulse-control paper (arXiv 2510.26438), Compound-Hawkes FRL paper
(arXiv 2312.08927), PhD thesis (UCL, 215 pp — `reference/Jain_10221263_Thesis-4.pdf`).

- **He never uses the Fano factor by name** (0 mentions in the thesis). Dispersion
  realism is scored via volatility **signature plots** and |returns|-ACF. His own
  Fig 4.7b shows the simulated signature plot **2–3× below** empirical — the published
  state of the art *also* undershoots dispersion level; it matches shape.
- **How his simulator earns its dispersion:** (i) kernels estimated by **Kirchner-style
  binned-count least squares on a lin-log lag grid** — the estimator's target *is* the
  cross-scale count-autocovariance (moments, not MLE; recovers simulated ground truth to
  ~6%); (ii) **power-law kernels** selected by AIC 100% of the time, straight lines over
  six decades of lag (10⁻⁴–10² s); (iii) 12-D mutual excitation with inhibitory kernels,
  65 kernels post-threshold, kernel-norm matrix **eigenvalue capped at 1** by rescaling
  (his `project_subcritical`); (iv) baselines derived, not fitted: **μ = (I−M)Λ** — the
  matrix form of LGM's pin; (v) compound order sizes (Dirac spikes at round numbers +
  geometric; constant-size ablation wrong by ×100 on price variance); (vi) U-shaped
  time-of-day multiplier.
- **Impulse-control paper adds consequences, not measurements:** removing Hawkes
  intensities from the RL state collapses Sharpe +31.5 → −20/−51 (the burst state is the
  MM's most valuable observation); exponential kernels admit dynamic arbitrage that his
  own solver exploits ("pump & dump"), rarely under power-law — mis-specified memory
  manufactures fake alpha. Direct warning for training agents inside our world model.

## 6. Standing conclusions

1. **Prediction and simulation are different objectives**; one-step MLE is structurally
   blind to (and misallocates) the spectrum that controls F(Δ). Three independent
   demonstrations (§4.2, 4.3, 4.4).
2. **The gradient horizon must cover the measured scales** — the cheapest large win
   found (§4.5): cover the scale, get the shape, reproducibly (BTC 3/3 seeds).
3. **The rate ceiling is the current level bottleneck** (§4.6): a third of events sit at
   the cap; raise moderately (9) with re-verified calibration.
4. **Certified dispersion needs linear-in-history structure**: only LGM has settable
   n / pinned rate / closed-form F(Δ) at any horizon; SS2P2's dispersion can only be
   regularized toward, never set. The Kirchner regression is the proven recipe for
   fitting the LGM ground (and typed kicks) to cross-scale count covariances — post hoc,
   marks untouched, no window limit.
5. **Per-asset targets**: n and spectrum must be fitted per asset (0.948 vs 0.973 is a
   4× Fano difference); tick regime (IS share) is the microstructural driver and should
   shape typed-kick priors (IS channels dominant on BTC/ETH).
6. Volume/price-domain dispersion (signature plots) will additionally need the compound
   size layer — already built (`dmm_sim/volume_sampler.py`, spike+lognormal per channel,
   median KS 0.16 on held-out).

## 7. Ranked roadmap

1. `ss2p2-w4kc9`: w4k48 + cap 9 (evidence-backed level lever; RUNNING, job 7148914).
2. Kirchner binned-regression fit of the LGM ground (typed, per asset) — "set, don't
   learn"; closed-form F(Δ) checked against real at all horizons before any GPU.
3. GMH-lite: LGM ground × bounded neural gate g(u) — restores timing NLL/backbone
   gradient without breaking the certificate (ρ_eff ≤ n·G_max).
4. Time-of-day multiplier (certificate-compatible; long-scale variance).
5. Agent phase guardrail: score exploitability (Jain's pump-and-dump lesson) before
   trusting any policy trained in the simulator.

## 8. Provenance

- Experiments: `peacock:~/simulation/experiments/ma_cbse/{btc,eth,sol}/` — tags
  `ss2p2-full-s*` (baseline), `ss2p2-lob-s*`, `lgm-s*` (+ archived `lgm-cold0-s*`),
  `lgm-k-s*`, `ss2p2-disp-s*`, `ss2p2-w4k-s*`, `ss2p2-w4k48-s*` (7142407),
  `ss2p2-w4kd-s*` (7142411).
- Code: `simulation` commits — LGM port 7b97b21; cold-start fix 2f4dd00; Konark arm
  2950094; disp arm 11a8fb7; w4k e8b48a3; w4k48 88d22cc. Mirrors in `dmm_sim/`.
- Diagnostics scripts (session-local, re-derivable): impulse-response probe
  (§3), saturation/teacher-forced-level probe (§4.6).
- Figures: `docs/figs/fano_three_assets.png` (all arms), `docs/figs/fano_w4k.png`
  (window-coverage result, per-seed).

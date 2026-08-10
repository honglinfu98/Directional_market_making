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

### 4.9 The closing result: Kirchner-fitted LGM (`lgm-kf`, SOL, 2026-08-08)

"Set, don't learn," executed: the LGM ground's kernel weights were fitted by
Kirchner-style binned-count regression on the real SOL train zones
(`scripts/kirchner_fit_lgm.py`: 0.25 s bins, lin-log lag grid to 240 s → cell
integrals Φ_k by OLS → NNLS projection onto a fixed log-spaced 6-exponential bank;
measured raw branching 0.9635, capped n = 0.99; μ₀ by the pin). The fitted spectrum
is *balanced* — 0.80 of the branching on β = 18.2 s⁻¹ plus a genuine slow tail —
exactly the allocation MLE never found. The fitted ground was transplanted into the
three trained SOL mark checkpoints (rate-neutral marks untouched) and run through
the full eval pipeline (job 7149220, all verified, κ = 1.014):

| SOL | Fano [1,2,5,10,20,50]s | sd@50s | acc | time-NLL | time-KS |
|---|---|---|---|---|---|
| ss2p2-full | 15.0 → 238.4 | ±217 | 0.1921 | −2.911 | 0.173 |
| lgm (MLE ground) | 2.3 → 59.8 | ±2 | 0.1862 | −2.227 | 0.464 |
| **lgm-kf** | **31.1, 46.5, 80.3, 125.7, 207.7, 422.9** | **±0.0** | 0.1860 | **−3.035** | 0.228 |
| real | 41.3, 54.7, 85.6, 125.3, 198.7, 393.6 | | | | |

Findings:
1. **Near-exact dispersion in full closed loop**: exact at 10 s, within 5–7% at
   20/50 s, within 6% at 5 s; only 1–2 s modestly low. The ground-only validation
   survived marks + calibration + rollout intact.
2. **Zero seed variance**: dispersion is a deterministic function of six fitted
   parameters; the marks cannot perturb it (rate-neutrality confirmed operationally).
3. **Moment fitting beat MLE on likelihood itself**: time-NLL −3.035 is the best in
   the zoo — better than SS2P2's neural rate head (−2.911) and 0.8 nats better than
   the MLE-trained ground (−2.227); time-KS 0.464 → 0.228. Windowed MLE was stuck in
   a bad spectral optimum that the count regression simply computes past.
4. Cost: a 3-minute CPU fit. No GPU training for the ground at all.

This closes the investigation's central question.

**BTC/ETH replication (job 7149357, 2026-08-08):** grounds fitted per asset and
n tuned to each empirical Fano curve (BTC 0.98, ETH 0.985):

| | Fano [1,2,5,10,20,50]s | sd@50s | vs real |
|---|---|---|---|
| ETH lgm-kf | 51.9, 81.0, 141.8, 219.9, 346.8, 629.8 | 0.0 | real 60.7 → 1247 |
| BTC lgm-kf | 77.0, 118.2, 188.3, 258.6, 344.5, 526.1 | 1.5 | real 70.0 → 1463 |

Replicated: short-scale match unprecedented (BTC 1–2 s essentially exact — no prior
arm within 2.5×), determinism (sd ≤ 1.5), best-in-zoo time-NLL (−3.64/−3.43), and
ETH's 50 s value beats every prior ETH arm with zero variance. Not replicated: at
20–50 s both assets deliver ~50–55% of the standalone ground simulation's prediction
at the same parameters (SOL matched its prediction exactly). Since the transplanted
law is identical, this is a closed-loop/estimator protocol discrepancy specific to
the near-critical high-rate regime (leading suspect: the tuner's pooled-replica
count variance vs the pipeline's equal-duration segment matching, which partially
absorbs the slow-cluster component). **Defined fix: tune n against the pipeline's
own estimator** (run tuner candidates through stylized_facts directly) — expected
to push BTC/ETH to n ≈ 0.985–0.99 under the pipeline's measure and recover the
50 s tail. Open residuals: that retune, and the short-scale type-dependence
(typed kicks) refinement.

### 4.10 Pipeline-estimator retune + typed excitation (2026-08-09, in progress)

**Retune probes (jobs 7149568/7149621):** running n candidates through the
pipeline's own stylized-facts estimator removes the standalone-sim mismatch.
First grid (SF-only, 1 rollout seed):

| n | BTC F(50s) | ETH F(50s) |
|---|---|---|
| 0.9875 | 714 | 694 |
| 0.99 | 848 | 772 |
| 0.9925 | 1009 | 928 |
| real | 1463 | 1247 |

Monotone, smooth (~+20% at 50 s per +0.0025 of n); ETH at 0.9925 is already exact
through 10 s (61/98/183/292 vs 61/95/183/316). Final rung (BTC 0.995/0.9975,
ETH 0.995) in flight.

**Typed excitation measured (scripts/kirchner_fit_typed.py):** joint 8-group
binned-count regression + shrunk per-channel kicks (E[w]=1 normalized):

| group | BTC | ETH | SOL |
|---|---|---|---|
| MO_b / MO_a | **286 / 458** | **131 / 124** | **122 / 23** |
| IS_b / IS_a | 0.39 / 0.48 | 0.54 / 0.62 | 4.6 / 3.5 |
| CO_b / CO_a | 0.61 / 0.97 | 1.18 / 1.15 | 0.63 / 0.61 |
| LO_b / LO_a | 0.22 / 0.28 | 0.35 / 0.44 | 0.47 / 0.56 |

Universal law: **market orders are the ignition events** (~100–450x the average
event's excitation; one BTC trade begets ~hundreds of quote events). Honest
correction to §2's prediction: per-event IS kicks on BTC are LOW (0.4) precisely
because IS is 52% of flow — undercutting is the diluted routine; the rare MOs
carry the ignition. Separability (rank-1 energy): BTC 87%, ETH 92%, SOL 73% —
the separable typed decoder suffices for the first pass. Caveat for transplants:
kick heterogeneity implies large E[w^2]/E[w]^2 variance multipliers, so typed
grounds need substantially lower n (marked-cluster tuning next).

**Accuracy track:** LGM mark retrains at seq 4096 / 48 epochs / stationary
cold-start (job 7149607, tags lgm-w4k48-s*) — targeting mark parity or better vs
SS2P2 (its own w4k48 control gained +0.4pp), onto which fitted grounds transplant
freely. Final assembly: retrained marks x per-asset tuned ground (typed or
untyped, whichever wins the probes).

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

### 4.11 Final assembly + the two-lane SSM (2026-08-09, in flight)

**Retune final:** BTC n=0.9925 (0.995/0.9975 overshoot short scales; logMSE
arbitrates), ETH n=0.995 (logMSE 0.0084 — within ~10% at every scale). Mark
retrains (lgm-w4k48, seq 4096/48ep/stationary fix) landed: SOL 0.195 (**beats**
ss2p2 0.192), BTC 0.428 (+3.3pp vs stale donors), ETH 0.326 (+3.6pp); the
residual ~2pp on BTC/ETH is the structural time-gradient gap. Final assembly
`lgm-kf2` (tuned grounds x retrained marks, 9 eval tasks, job 7150066) in flight.

**Two-lane SSM (lgm-tl, job 7150155):** the unified architecture — the ground as
a certified lane (linear, positive, fixed decays, frozen at the Kirchner fit)
inside one state machine, plus a bounded mean-one gate g(u) =
exp(log(Gmax)(tanh(v'u) − EMA)) coupling the free lane to the rate. Gateless it
is law-identical to lgm-kf; the gate restores the time-likelihood gradient to
the backbone (the underemployment fix) and adds bounded state-dependent
dispersion (kurtosis target), at the cost of certificate ≤ Gmax²·n (bound, not
equality) and expectation-level (not exact) pin. Pre-launch verification passed:
neutral start exact, bounds held, backbone time-gradients confirmed, frozen
ground intact (n=0.99), geometric-mean-one gate. SOL 3 seeds, seq 4096/48ep,
Gmax=2, vs lgm-kf reads: mark accuracy, kappa≈1 retention, Fano retention +
kurtosis gain.

### 4.12 Results: lgm-kf2 final assembly + lgm-tl (2026-08-09, jobs 7150066/7150155)

**lgm-kf2 (tuned grounds × lgm-w4k48 retrained marks; 3 seeds × 3 rollouts per
asset, all 9 eval tasks rc=0).** Fano scales {1,2,5,10,20,50}s, equal-duration
matched real in parens; seed-sd across the 9 rollouts in brackets:

| asset | acc (3 seeds) | ppl | time-NLL | time-KS | κ | Fano model vs real |
|---|---|---|---|---|---|---|
| BTC | 0.4280/0.4277/0.4293 | 9.2 | −3.4254 | 0.111 | 1.019 | [101, 162, 283, 422, 618, 1007] vs [55, 85, 157, 258, 433, 900] |
| ETH | 0.3277/0.3252/0.3252 | 14.1 | −3.6450 | 0.150 | 1.016 | [67, 110, 210, 349, 585, 1170] vs [79, 125, 245, 417, 748, 1571] |
| SOL | 0.1944/0.1961/0.1952 | 23.9 | −3.0350 | 0.228 | 1.014 | [31, 46, 80, 126, 208, 423] vs [42, 59, 95, 142, 233, 433] |

- **SOL: the clean win.** Accuracy 0.194–0.196 (≥ ss2p2-full 0.192), Fano within
  ~25% at 1 s converging to ~2% at 50 s, κ=1.014. F6 0.113 (real 0.150).
- **ETH: level and shape right, modest undershoot** (ratio 0.85 at 1 s → 0.74 at
  50 s). F6 0.512 vs real 0.489 — volatility clustering essentially matched.
- **BTC: now OVER-disperses at short scales** (×1.8 at 1 s, converging to ×1.1
  at 50 s). The n=0.9925 retune was arbitrated by logMSE across scales; against
  this eval's equal-duration real curve ([55…900], lower at short scales than
  the tuning reference) the short-scale weight is too hot. A half-step retune
  (n≈0.99) would likely center it; deferred — the miss is 2× where ss2p2's was
  15–30× under.
- Time-NLL/KS identical across seeds within an asset (frozen shared ground owns
  the time law — marks are rate-neutral). Seed-sd of dispersion remains ~0 at
  short scales (2–8%), confirming transplant determinism.
- Kurtosis remains the open residual everywhere: BTC 47 (real 88), ETH 7 (35),
  SOL 22 (163). This is the TL gate's target, not the linear ground's.

**lgm-tl (gated two-lane SSM, SOL).** Genuine (all 3 seeds): acc
0.1974/0.1960/0.1966, time-NLL −3.086/−3.163/−3.090, time-KS 0.190/0.199/0.199.
Stylized facts: s2 completed — κ=1.044, Fano [44, 64, 110, 168, 249, 400] vs
real [42, 59, 95, 142, 233, 433], F6 0.110, excess kurtosis 62.1; s1/s3 sf
FAILED κ-calibration (5% tolerance, 10 bisection steps; brackets
[0.977,0.979] and [0.989,0.991] — the gated rate is non-monotone/noisy in κ
near criticality). Reads:

- **(a) accuracy — PASS.** All three seeds ≥0.196: beats lgm-kf2 SOL
  (0.194–0.196) and ss2p2-full (0.192). Time-NLL −3.09 to −3.16 vs kf2's −3.035
  and time-KS 0.19 vs 0.228 — the gate demonstrably restored the time-likelihood
  gradient to the backbone (the underemployment fix worked).
- **(b) κ≈1 retention — FAIL/partial.** s2 needed κ=1.044 (kf2: 1.014) and 2/3
  seeds could not be calibrated to 5% at all. The expectation-level pin is
  measurably looser than the exact pin; the gate's rate perturbation makes
  bisection fragile near n=0.99.
- **(c) dispersion — PASS on s2 alone** (Fano [44,64,110,168,249,400] vs real
  [42,59,95,142,233,433]; kurt 62 vs kf2 22). SUPERSEDED by the 3-seed rerun
  below — the single completed seed was not representative.

**Interim verdict (superseded by §4.13):** the gate buys prediction + kurtosis
at the cost of pin exactness. Next lever: recover calibration robustness.

### 4.13 Calibrator fix + full 3-seed lgm-tl simulation (2026-08-10, jobs 7150548/7150559)

**The fragility, diagnosed:** κ-bisection assumes the probe rate is a low-noise
monotone function of κ. Near criticality the probe SE ~ sqrt(Fano·R/T_total)
exceeds the 5% tolerance (probes jumped 21.7→25.7→21.7 over a 0.3% κ interval),
so brackets collapse on noise. **Fix (simulation@aae89cb, 44700c6; converged
fast path untouched):** (1) *deep-accept confirm* — a within-tol probe at
bisection step ≥4 must survive 2 replicate seeds at 2× sequences, pooled;
(2) *regression fallback* on bracket exhaustion — pool all probes near the
target and fit the κ-scaled Hawkes law 1/rate = a/κ + c (linear in 1/κ!),
solve for κ, confirm at 12× probe fidelity; accept ≤5%, relaxed-accept ≤10%
(the confirmation is low-noise, so a 6–10% miss is the model's razor-steep
rate response, not estimator noise — the full-scale 15% verify arbitrates).
Synthetic harness (law fitted to the real failed pool + 12% noise): 7/8 trials
within 5.9% true error, 1/8 raises loudly. Replaying the two real failed pools:
κ=0.979/0.985, fit-predicted rates within 1% of target.

**Rerun (RESUME=1, SF-only): all 3 seeds STATUS=0.** Calibration paths:
s1 fallback κ=0.9817 (relaxed 6.4%, verify 7.0%), s2 escalated bisection
κ=1.0457 (verify 12.7%), s3 fallback κ=0.9857 (3.6%, verify 7.8%). The
calibrator is fixed — every seed now completes.

**But the full 3-seed dispersion overturns the s2-only read:**

| seed | κ | Fano @{1,2,5,10,20,50}s | kurt |
|---|---|---|---|
| s1 | 0.9817 | [17.8, 22.7, 32.7, 44.8, 64.8, 118.8] | 12.8 |
| s2 | 1.0457 | [45.5, 65.9, 110.5, 167.0, 246.8, 401.2] | 61.0 |
| s3 | 0.9857 | [53.9, 91.7, 191.8, 326.5, 550.3, 1128.7] | 40.5 |
| real | — | [42.3, 58.9, 94.8, 142.4, 233.1, 433.1] | 163.1 |
| kf2 (ref) | 1.0136 | [31.1, 46.5, 80.3, 125.7, 207.7, 422.9] sd≈[2,3,9,16,29,76] | 21.8 |

Pooled TL mean [39, 60, 112, 179, 287, 550] looks fine; the seed spread does
not: sd at 1 s is 16.9 (kf2: 2.0), and 50 s Fano spans 119→1129 across seeds
(×9.5). **The learned gate converts dispersion from a set constant into a seed
lottery** — each seed's gate learns a different coupling to the frozen ground,
and Var(g) compounds over 600 s rollouts. Rate-neutrality's zero-seed-variance
transplant guarantee is exactly what the gate spends.

**Final TL verdict (all reads, 3 seeds):**
- (a) prediction — PASS (unchanged: acc 0.196–0.197, tNLL −3.09..−3.16, KS 0.19).
- (b) κ≈1 — FAIL as a pin (κ ∈ [0.982, 1.046], rate residuals −7%..+13%) but
  now operationally robust (calibrator completes every seed).
- (c) dispersion — FAIL on reliability: right on average, ×9.5 seed spread.
  Kurtosis mean 38 vs kf2 22 (real 163) — directionally right, seed-unstable.

**Standing:** lgm-kf2 remains the shipped simulator (deterministic dispersion,
exact pin). TL is the research direction for prediction + kurtosis; before
promotion it needs gate variance control — smaller Gmax (1.25–1.5), a gate
log-variance penalty, or dispersion-loss-matched gate training (§ss2p2-disp
machinery exists) so the gate's dispersion contribution is *targeted* rather
than free.

### 4.14 Synthesis: why the decoupling is sound — the two-valve theory (2026-08-11)

The soundness of the lgm-kf2 split (Hawkes owns *when*, S2P2 owns *what*)
rests on three exact facts:

1. **Factorization is an identity.** Any MTPP satisfies λ_k = Λ·(λ_k/Λ);
   choosing to model the two factors is not an assumption. This is the form of
   Chang, Boyd & Smyth (AISTATS 2024, "Probabilistic Modeling for Sequences of
   Sets in Continuous-Time" — the lineage of this codebase, eq. 6 there).
2. **The likelihood separates additively with disjoint blocks:**
   ℓ = [Σ log Λ(t_i) − ∫Λ] + Σ log p(k_i|u(t_i⁻)) = ℓ_time(θ_g) + ℓ_mark(θ_m).
   No cross-terms, so block-wise estimation (Kirchner moments for the ground,
   MLE for the marks) IS the joint MLE restricted to each block. ℓ_mark is
   teacher-forced on real times and contains no ground parameter ⇒ post-hoc
   ground transplantation is exact, not approximate.
3. **The count process is autonomous.** Type-blind kicks make Λ a function of
   event *times* only: the times are a self-contained scalar Hawkes; marks are
   a coloring of its ticks. Coloring cannot change the number of ticks ⇒ every
   count statistic (Fano curve, branching) is a set-once constant of (a, β, n)
   for any mark parameters/seed/training path — the transplant *theorem*, and
   the reason measured dispersion seed-variance is ~0.

**Two one-way valves.** Rate-neutrality (softmax simplex) closes the
marks→rate direction: training decouples (fact 2). Mark-blindness (unit kicks)
closes the rate→marks direction: simulation decouples (fact 3). The
experimental arms are exactly the four valve states:

| variant | marks→rate | rate→marks | consequence |
|---|---|---|---|
| lgm-kf2 | closed | closed | exact pin, transplantable ground, F(Δ) deterministic |
| typed kicks w_k | closed | OPEN | count law mark-dependent → n retune on marked clusters |
| lgm-tl (gate) | OPEN | closed | ℓ non-separable → pin loosens, F(Δ) a seed lottery (§4.13) |
| ss2p2 / Chang et al. | OPEN | OPEN | one shared state drives both; nothing certified/settable |

Each closed valve also blocks something useful: marks→rate closed blocks the
time-gradient the backbone wants (~2pp accuracy, the TL motivation);
rate→marks closed blocks MO-ignition physics (kicks 122–458×, §typed fits).
kf2 is the both-closed corner; the open problem is opening either valve *by a
controlled amount* without losing the theorems.

**Relation to Chang et al. (2024):** they supply the factorized form (their
eq. 6) and the L_Time/L_Set likelihood split (their eq. 2), and even observe
the capacity tension (static set models beat dynamic on L_Time alone). But
both their factors read the same hidden state and are trained jointly — the
factorization there is a parameterization device for 2^K set marks, not a
separation. The parametric split, mark-blind autonomous ground, moment-based
setting of the time block, and the invariance guarantees are this project's
addition. One-line positioning: *we adopt the total-intensity × conditional-
mark factorization of Chang et al. and harden it into a parametric separation,
replacing the shared-state neural total intensity with a frozen convexly-
fitted linear Hawkes ground so that all count statistics become set-once
constants.*

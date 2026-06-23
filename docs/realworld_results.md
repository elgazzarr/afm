# Real-World Benchmark Results (AFM)

Generated on the lightweight CPU pipeline (no GluonTS/GPU). Single seed unless noted.
Scripts: `examples/realworld_battery.py`, `examples/realworld_cmapss.py`, `examples/realworld_compare.py`.
Sample plots: `results/realworld/*.png`. Published baselines + protocols: see
`docs/superpowers/specs/2026-06-18-real-world-degradation-benchmarks-design.md` §7.

> **Headline honesty:** On these two *smooth real-world degradation* datasets the AFM
> advantage over the FM baseline does **not** appear — FM is clearly better on battery
> capacity-fade and roughly tied on C-MAPSS sensor forecasting. This is consistent with
> the paper's own finding that AFM's edge is in **chaotic/stochastic extrapolation**
> (synthetic systems), not smooth monotonic decay (cf. the Van der Pol CRPS result).
> Frame accordingly; do not claim AFM > FM on these datasets.

## 1. Battery — capacity-fade forecasting (AFM, leave-one-battery-out, ctx=66, forecast ~66 cycles, 50 epochs)

| Holdout | CRPS | NRMSE | capacity RMSE (Ah) | MAE (Ah) | ≈ norm. SOH RMSE* |
|---|---|---|---|---|---|
| B0005 | 0.297 | 0.395 | 0.0731 | 0.0577 | ~3.7% |
| B0006 | 0.329 | 0.710 | 0.0947 | 0.0767 | ~4.7% |
| B0007 | 0.305 | 0.740 | 0.0967 | 0.0873 | ~4.8% |
| B0018 | 0.323 | 2.830 | 0.1240 | 0.1203 | ~6.2% |
| **mean (B0005/6/7)** | 0.310 | 0.615 | **0.0882** | 0.0739 | ~4.4% |
| mean (all 4) | 0.314 | 1.169 | 0.0971 | 0.0855 | ~4.9% |

\*capacity RMSE ÷ ~2 Ah rated. B0018 is the known hard cell (capacity-regeneration / atypical
discharge profile); the literature also reports it as an outlier.

**vs published (matching protocol):** Richardson, Osborne & Howey 2017 (GPR forecasting of
future capacity) reports normalized RMSE ≈ 0.06–0.08 at a ~40-cycle horizon. Our ~0.044 norm.
over a ~66-cycle horizon (leave-one-battery-out) is **competitive** — and AFM additionally
provides a calibrated predictive distribution (CRPS). Not comparable to same-cell decomposition
hybrids (CEEMDAN ~0.004–0.009 Ah) which use 100-cycle context + pre-smoothed signal.

## 2. AFM vs FM head-to-head (the paper's core ablation, on real data)

FM trained 2× epochs (per the paper's protocol); single seed/holdout — treat as indicative.

| Dataset (protocol) | Model | CRPS | NRMSE | RMSE (orig units) | MAE |
|---|---|---|---|---|---|
| Battery B0007 (ctx=66, H=66) | AFM | 0.305 | 0.740 | 0.0967 Ah | 0.0873 |
| | **FM** | **0.229** | **0.467** | **0.0227 Ah** | **0.0161** |
| C-MAPSS FD001 (ctx=20, H=108, sensor s2) | AFM | 0.187 | 0.731 | 0.338 | 0.268 |
| | FM | **0.182** | **0.700** | 0.348 | 0.275 |

- **Battery: FM wins decisively** (smooth, near-monotonic decay favors FM's whole-window
  Brownian-prior global fit; AFM's step-wise rollout drifts over the 66-step horizon — see
  `results/realworld/battery_B0007_{AFM,FM}_capacity.png`).
- **C-MAPSS: effectively tied** (FM marginally better CRPS/NRMSE; AFM marginally better on the
  de-normalized sensor). Within run-to-run noise.

## 3. C-MAPSS — RUL (proof-of-concept only; NOT competitive)

Health-index → threshold-crossing derivation, FD001, cap=125:

| | RMSE (cycles) | Score |
|---|---|---|
| AFM-derived RUL | 64.5 | 1.5×10⁶ |
| Published (FD001): DCNN | 12.61 | 273.7 |
| Published (FD001): DAST | 11.43 | 203 |

**Do not put this head-to-head.** The HI-forecast rarely reaches the failure threshold within the
horizon, so RUL saturates at the cap and the asymmetric Score explodes. AFM is a forecaster, not a
RUL regressor — present RUL strictly as proof-of-concept, or omit it.

## 4. Caveats / to strengthen before the manuscript

- Single seed; battery has only 3 training cells (high variance). Report mean ± std over seeds
  (and over holdouts) before quoting numbers.
- FM got 2× training epochs — note the compute asymmetry (paper's protocol) or equalize.
- C-MAPSS forecasting has **no external baseline** (the literature is RUL); report as internal
  AFM-vs-FM-vs-naive only.

## 5. Diagnostics — where AFM actually wins (C-MAPSS FD001, ctx=20, H=108)

Horizon-resolved analysis (the aggregate CRPS tie hides this):

| diagnostic | AFM | FM | nominal/best |
|---|---|---|---|
| coverage@50% (overall) | 0.417 | 0.421 | 0.50 |
| coverage@90% (overall) | 0.794 | 0.830 | 0.90 |
| cross-sensor corr-matrix distance | **1.747** | 2.558 | 0 |

- **Near/mid horizon (steps 1–~60): AFM wins** — lower CRPS and 90% coverage closer to nominal.
- **Long horizon (>~60): FM wins** — AFM rollout drifts and turns overconfident (coverage → ~0.6).
- **AFM reproduces the multivariate dependence structure better** (corr distance 1.75 vs 2.56) —
  the reviewer-requested "joint distribution" diagnostic.

Plots: `results/realworld/cmapss_FD001_{calibration,crps}_vs_horizon.png`.

**Defensible AFM argument:** AFM is the stronger *near-to-mid-horizon multivariate probabilistic*
forecaster (lower CRPS + better-calibrated PIs to ~step 60) and better captures cross-channel
dependence; whole-window FM only overtakes at very long horizons where AR error accumulates.
Compete on horizon-resolved calibration + dependence, not aggregate point accuracy.

## 6. Multi-seed robustness (3 seeds, mean ± std)

Each seed = independent AFM + FM training (model init, training noise, sampling).
Data window-sampling order is fixed across seeds; variation reflects optimization +
sampling. Plots: `results/realworld/*_{calibration,crps}_vs_horizon_ms.png` (seed-averaged,
±std band); raw per-seed numbers in `results/realworld/*_multiseed.json`.

**C-MAPSS FD001 (ctx=20, H=108)**

| metric | AFM | FM |
|---|---|---|
| CRPS | 0.1825 ± 0.0047 | 0.1792 ± 0.0032 |
| NRMSE | 0.718 ± 0.016 | 0.685 ± 0.008 |
| coverage@50% | 0.435 ± 0.020 | 0.435 ± 0.013 |
| coverage@90% | 0.810 ± 0.019 | 0.842 ± 0.013 |
| **corr-matrix dist** | **1.841 ± 0.227** | 2.410 ± 0.258 |

- **Stable AFM wins:** (i) near/mid-horizon CRPS (the seed-averaged CRPS-vs-horizon bands are
  separated for steps ~1–55, AFM below), and (ii) cross-sensor dependence (corr distance
  1.84 vs 2.41, gap exceeds the ±std). Both survive multi-seed scrutiny.
- Aggregate CRPS/NRMSE: tie within noise (horizon profile averages out). Long horizon: FM wins
  (AFM rollout drift). Aggregate cov90 marginally favors FM.

**Battery B0007 (ctx=66, H=66)**

| metric | AFM | FM |
|---|---|---|
| CRPS | 0.312 ± 0.007 | **0.230 ± 0.008** |
| NRMSE | 0.773 ± 0.065 | **0.498 ± 0.007** |
| capacity RMSE (Ah) | 0.101 ± 0.005 | **0.034 ± 0.002** |
| coverage@90% | 0.135 ± 0.032 | **0.533 ± 0.050** |
| corr-matrix dist | 0.716 ± 0.004 | 1.457 ± 0.913 |

- **FM dominates battery on accuracy AND calibration, robustly.** AFM is severely overconfident
  on smooth monotonic decay (cov90 ≈ 0.14 vs nominal 0.90, across the whole horizon — no crossover).
  The only AFM-lower number is corr distance, but battery's 4 channels co-vary trivially and FM's
  value has huge variance (±0.91), so it is not a meaningful win.

### Consolidated verdict
- **Use C-MAPSS as the AFM showcase:** AFM is the better *near-to-mid-horizon multivariate
  probabilistic* forecaster and better captures the *joint dependence* — both seed-stable, and
  both are reviewer-requested diagnostics.
- **Do not feature battery as an AFM win.** It robustly favors FM and exposes AFM over-confidence
  on smooth low-noise series. Present it as the FM-favorable / known-limitation case (consistent
  with the synthetic Van der Pol result), ideally in an appendix.

## 7. Root-cause fix: sliding covariate window in the AFM rollout (single-seed verification)

**Bug:** `AutoregressiveFlowMatching.sample` froze the covariate/time window at the initial
context while the state context slid forward during rollout, feeding the model
`(state, stale-time)` pairs never seen in training -> long-horizon drift + overconfidence.
**Fix:** slide the covariate window in lockstep using the full context+horizon covariates
(backward-compatible: context-length-only covariates keep the old frozen behavior, so the
synthetic-systems path is unchanged). See commit on `fix-afm-covariate-rollout`.

Single-seed before -> after (same protocol):

| dataset | metric | AFM pre-fix | AFM post-fix | FM |
|---|---|---|---|---|
| Battery B0007 | NRMSE | 0.740 | **0.555** | 0.467 |
| | capacity RMSE (Ah) | 0.097 | **0.073** | 0.023 |
| | cov90 | 0.159 | 0.250 | 0.561 |
| C-MAPSS FD001 | CRPS | 0.187 | 0.184 | 0.182 |
| | corr-matrix dist | 1.747 | **1.344** | 2.558 |

- Battery: AFM point accuracy ~25% better; FM still wins smooth monotonic decay; AFM still
  overconfident (residual exposure-bias issue, not addressed by this fix).
- C-MAPSS: AFM's cross-sensor dependence advantage strengthened (1.34 vs 2.56).
- **Residual issue (separate):** multi-step overconfidence — the rollout never sees its own
  errors in training (single-step teacher forcing), so predictive bands understate accumulated
  uncertainty. Needs a training-side change (scheduled sampling / multi-step rollout training).

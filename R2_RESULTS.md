# R2 results log (IJF-D-26-00120R1, minor revision)

Deadline: 6 November 2026. Brief: `AFM_R2_revision/CLAUDE_CODE_R2_BRIEF.md`.
Note: the brief refers to a `paper/` folder. The files actually live in `AFM_R2_revision/`.

Citation shorthand: `P:` = git revision `f4131fe^` (the last tree before the "restructre repo" commit that
deleted the GluonTS path); `HEAD:` = current `main` (721e209).

---

## Task 0: Recon (2026-09-28, no experiments run)

### 0.1 GluonTS datasets and test windows (recovered from git history)

The real-world GluonTS pipeline is no longer on `main` but is intact at `P`
(`data/gluonts_loaders.py`, `trainers/eval_gluonts.py`, `main.py`, `models/flow.py`).
The working entry point was `P:main.py`; `P:main_gluonts.py` is broken at that revision
(imports a missing `trainers.flow_train_stf`).

- Datasets: `get_dataset(name, regenerate=False)` (`P:data/gluonts_loaders.py:124`), choices
  `traffic_nips, electricity_nips, solar_nips, exchange_rate_nips, wiki-rolling_nips` (`P:main.py:27-28`).
  The paper's dataset table (2000 series, daily, length 792, horizon 30) matches `wiki2000_nips`
  rather than `wiki-rolling_nips` (`P:notebooks/test_gluonts.ipynb:49-111`). Not relevant to the R2 rows.
- Frequency and prediction length come from dataset metadata (`P:data/gluonts_loaders.py:125-126`):
  24 for the hourly sets, 30 for Exchange and Wikipedia.
  Exchange is business-daily (`B`) in GluonTS; the paper's dataset table says "daily".
- Context length = `prediction_length * context_ratio` (`:127`), default `context_ratio=2.0` (`P:main.py:29`).
  **The per-dataset `context_ratio` used for the published AFM runs is not recorded anywhere in the repo**
  (no configs or results; runs were logged to the remote wandb project `flowtime`).
- Test windows: `MultivariateGrouper(num_test_dates=len(test)/len(train), max_target_dim=min(2000, card))`
  (`P:data/gluonts_loaders.py:131-132`). This gives 7 rolling windows for Electricity/Solar/Traffic and 5 for
  Exchange/Wikipedia (consistent with TSFlow's test sizes 2590/959/6741 and 40/10000 series-windows).
  Each window is `split(test, offset=-prediction_length)` + `generate_instances(prediction_length)` (`:198-200`).
- Scaling: per-window mean scaling over the context (`P:data/scalers.py:15-16`), inverted before scoring,
  so **metrics are computed in the original scale**.

### 0.2 Evaluation harness

- `MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:], target_agg_funcs={'sum': np.sum})`
  (`P:trainers/eval_gluonts.py:93-94`), i.e. the 19-level grid 0.05, 0.10, ..., 0.95.
- Forecasts are `SampleForecast` (`:81`), 100 samples (`P:main.py:64`).
- Reported: `CRPS = agg["mean_wQuantileLoss"]`, `CRPS-Sum = agg["m_sum_mean_wQuantileLoss"]` (`P:main.py:147,150`).
- No versions are pinned: `requirements.txt`/`setup.py` are empty at every relevant commit. The only
  environment evidence is Python 3.12.4 on a cluster conda env (notebook metadata).

### 0.3 Provenance of Table 4 statistical baselines

- **No code for SN / AutoARIMA / AutoETS has ever existed in this repo** (pickaxe search on all refs for
  statsforecast, SeasonalNaive, AutoETS, AutoARIMA, RForecastPredictor, rpy2, sktime: no hits in code).
- They are quoted from TSFlow, Kollovieh et al. (ICLR 2025, arXiv 2410.03024), Table 3. The TSFlow row
  in our Table 4 (0.045/0.009/0.343/0.083/0.227) matches **arXiv v1** exactly.
- TSFlow's protocol differs from ours: series are treated as **univariate**, CRPS is approximated with
  **9 quantiles {0.1, ..., 0.9}**, 100 samples; ARIMA/ETS were fitted with **AutoGluon**, SN and the other
  baselines with GluonTS implementations (TSFlow App. A.4; Sec. 4 "Evaluation Metrics").
  The seasonal period `m` is not stated in the paper (GluonTS default seasonality would be 24 for hourly,
  5 for business-daily, 7 for daily; still to be confirmed from TSFlow's code).
- **Transcription mismatches against the cited source** (present since the original submission,
  `rebuttal /main_old.tex:547-555`):

  | Cell | Our Table 4 | TSFlow v1 Table 3 |
  |---|---|---|
  | SN, Electricity | 0.060 | 0.069 |
  | AutoARIMA, Exchange | 0.013 | 0.008 |
  | AutoETS, Electricity | 0.056 | 0.055 |
  | SSSD, Solar | 0.354 ± 0.024 | 0.384 ± 0.023 |
  | SSSD, Traffic | 0.107 ± 0.002 | 0.097 ± 0.002 |
  | several ± values (DeepAR, WaveNet, CSDI, SSSD) | ± 0.000 | ± 0.001 to 0.055 |

  The TimeGrad row is attributed to Rasul et al. (2021), but that paper reports CRPS-Sum; the source of
  the marginal-CRPS TimeGrad numbers is not yet identified.
  Consequence for Exchange: the correct AutoARIMA value (0.008) ties AutoETS for best.
  The R1 response letter explained these mismatches as pipeline differences; since the rows are quoted,
  they are transcription errors.

### 0.4 Figure 2 (synthetic systems)

- `figures/fig2_updated.pdf` is an Adobe Illustrator composite (created 2025-03-12; source
  `~/Downloads/flowtime_figures/Untitled-7.ai`). It has 2×4 panels (Lorenz, FitzHugh-Nagumo, Lotka-Volterra,
  Brusselator), with legend "observed / prediction / extrapolation" and **no Van der Pol panel**.
- The per-system panels came from `bd699df:dynamical_systems/main.py:157-159` → `plot_trajectories`
  (`bd699df:dynamical_systems/visualization.py:27-109`), committed as
  `bd699df:dynamical_systems/results/<sys>/{ar,noar}_predictions_{0..9}.pdf`. The FHN and Brusselator PDFs
  were overwritten after the figure was assembled. Van der Pol panels exist from the same run.
- **The plotted trajectories are held-out test trajectories.** Train and test sets are generated separately
  with different subkeys and fresh initial conditions (`bd699df:dynamical_systems/training.py:156-160`).
  Plots use `next(iter(test_loader))` with `shuffle=False` (`bd699df:dynamical_systems/main.py:107,157`),
  and no code path plots the training array. Training targets are steps 76-150 only (the extrapolation
  segment is dropped, `bd699df:training.py:56`).
- The band is the empirical 2.5-97.5% interval over 50 samples and the dashed line is the **mean**
  (`visualization.py:34,47-51,82-86`). The caption is correct on both.
- Table 2 metrics are computed on the test loader only (`bd699df:main.py:126-146`;
  `P:dynamical_systems/evaluation/evaluator.py:72,96-97`).
- **Discrepancies with the paper's Section 4.1 description:**
  - The Fig. 2 run used 1024 train / 204 test trajectories (`bd699df:main.py:58`), not 2000/400. It was
    also **deterministic**: `stochastic` was never passed, so the default `False` applied.
  - Every system simulator uses `VirtualBrownianTree(..., key=PRNGKey(0))`
    (`src/afm/systems/systems.py:38,67,97,126,...`, same at `bd699df`). **All trajectories share one Brownian
    path**, contradicting "randomly sampled ... Brownian motion realizations" (`main.tex:382`).
  - **The run that produced Table 2 cannot be identified.** No synthetic results or checkpoints exist, and
    no config in history matches the paper (2400 initial conditions, 2000/400, 200 steps).
- The current `main` synthetic pipeline is broken. `src/afm/experiments/base.py:244` passes
  `(train_loader, test_loader)` to the evaluator, which expects 4-tuples, and `multi_system.py:104`
  references an undefined `test_data`.
- CPU cost to retrain a system is unknown. The paper quotes about 7 min per system on an A100.

### 0.5 C-MAPSS FM and Fig. B.4

- Figures: `examples/make_appendix.py` (`fig_calibration` 89-101, `fig_crps` 104-114) from
  `results/realworld/cmapss_FD001_multiseed.json`; Table 6 from `tbl_multiseed` (144-163).
  Per-seed curves come from `examples/realworld_multiseed.py` (`run_seed` 74-107).
  Saved models exist: `results/realworld/models/cmapss_FD001_{AFM,FM}_seed{0,1,2}.pkl` (post-RNG-fix, Jun 23).
  Raw samples (`cmapss_FD001_figdata.npz`) exist for seed 0 only.
- Test windows: first 20 engines of `train_FD001`, truncated to 128 cycles; context 20, horizon 108.
  **No padding, no masking: every horizon step averages over the same 20 engines × 14 sensors.**
  No off-by-one between FM output and target was found.
- FM is **trained on 40-step windows** (`_fm_window`, `realworld_multiseed.py:35-37`) but **sampled on
  108 steps**. Brownian prior variance at step k is k, so steps 41-108 are beyond the prior scale seen in
  training. Its bidirectional LSTM starts the backward pass from zero state at the final index.
- Numbers from saved artefacts (read-only):
  - FM per-step CRPS (seed mean), steps 101-108: .170 .176 .168 .168 .166 .163 .150 .131 (mean steps 1-100: .181).
  - FM 90% interval width (seed 0): 3.72 at step 101, rising to 7.58 at step 108, vs a mean of 2.08 over
    steps 1-100. AFM's end-to-average width ratio is only 1.18×.
  - FM 90% coverage, steps 105-108: .930 .951 .969 .985.
  - FM MAE does not improve at the end (0.96 at step 108 vs 0.57 on average).
  - Aggregate CRPS (mean ± std over 3 seeds): FM .1792 ± .0032 with all steps, .1803 ± .0032 without the
    last 4 steps; AFM .1802 ± .0040 and .1797 ± .0037.

### 0.6 CRITICAL: `compute_crps` is not the CRPS

`src/afm/evaluation/metrics.py:4-31` (unchanged since before the restructure) computes
`mean_i (1{x_(i) >= y} - i/S)^2` over the sorted samples, **without the spacing between samples**
(the dy in the CRPS integral). The result depends only on the rank of y among the samples, is bounded in
about [1/12, 1/3] regardless of units, and is **not proper**: it rewards arbitrarily wide ensembles.

Check (200 samples, truth y = 0):

| Ensemble | repo `compute_crps` | proper CRPS (energy form) |
|---|---|---|
| sharp (sd 0.1), centred | 0.083 | 0.023 |
| very wide (sd 100), centred | 0.088 | 24.9 |
| sharp, off by 1 | 0.331 | 0.944 |
| very wide, off by 1 | **0.083** | 24.9 |

Callers: `src/afm/evaluation/evaluator.py:44` (synthetic systems, **Table 2**),
`src/afm/datasets/eval.py:27` and `diagnostics.py:36,46` (**Table 6**, CRPS-Sum, Figs. B.x horizon curves).
All published values in those tables fall inside [0.083, 0.333], e.g. Lorenz AFM NRMSE 0.017 with CRPS
0.090 (the rank floor).
The paper's Eq. (CRPS integral, `main.tex:448`) defines the proper CRPS, so code and paper disagree.
**Table 4 is unaffected**: it uses the GluonTS evaluator.

This explains the Fig. B.4 drop. FM's ensemble balloons over the last steps, so the truth sits nearer
the middle of the samples, which this score rewards. Under a proper CRPS (seed 0), FM's step 108 is
*worse* (0.75 vs 0.41 average).

---

## Decisions (Ahmed, 2026-09-28)

- Keep changes minimal because the paper is close to acceptance. **No new numbers beyond what the reviewers
  explicitly asked for.**
- Correct the Table 4 transcription errors (D3: yes).
- Keep the quoted single-season SN / AutoARIMA / AutoETS rows (D4: keep quoted).
- No re-runs of Table 2 or Table 6 (D1/D2: no recomputation).
- Table 4 corrections are explained in the R2.2 response, with no reference to the R1 explanation.
- DHR-ARIMA with the smaller K grid (option B).
- Code link omitted from the manuscript for this round (to be added at acceptance). No anonymous mirror.
- TimeGrad row left as is: source unknown, and not in TSFlow v1/v2 or TSDiff (arXiv 2307.11494).
- No Section 4.1 wording change about the shared Brownian path.

Checked while discussing how to explain the Table 4 corrections:
- The SN / ARIMA / ETS / SSSD rows are **identical in TSFlow arXiv v1, v2 and the ICLR version**, so a
  source update does not explain the mismatches. Only TSFlow's own row changed between versions: ours is v1.
- The repo score equals **(p³ + (1-p)³)/3, where p is the PIT of y among the samples**
  (verified numerically to 4 decimals at p = 0, .1, .3, .5, .8, 1). It is minimised (1/12) when y is at the
  forecast median, whatever the spread.

---

## Tasks 1-2: statistical baselines

**Environment:** `baselines/.venv` (uv, separate from the JAX env). Python 3.12.7, gluonts 0.17.0,
statsforecast 2.1.1, utilsforecast 0.2.16, numpy 2.5.3, pandas 2.3.3. Apple M1 Pro, 10 CPUs.
**Script:** `baselines/r2_statistical_baselines.py`. Outputs go to `results/r2/*.json` (gitignored).

### Harness validation (2026-09-28)

Command: `baselines/.venv/bin/python baselines/r2_statistical_baselines.py validate`.
The windows are rebuilt as in AFM (`MultivariateGrouper(num_test_dates=W)`, then `split(offset=-H)`).
The script asserts that grouped labels equal the per-series futures and that the forecast origins agree.

| Check | afm mode (19 q) | tsflow mode (9 q) | Quoted (TSFlow v1) | Our Table 4 |
|---|---|---|---|---|
| SN, Electricity (m=24) | 0.0693 | 0.0693 | 0.069 | 0.060 (typo) |
| SN, Solar (m=24) | 0.5121 | 0.5121 | 0.512 | 0.512 |
| SN, Traffic (m=24) | 0.2209 | 0.2209 | 0.221 | 0.221 |
| SN, Exchange (m=30) | 0.0130 | 0.0130 | 0.013 | removed in R2 |
| AutoETS, Exchange (m=5, last 2500) | 0.0076 | 0.0079 | 0.008 | 0.008 |
| AutoETS, Exchange (m=5, full history) | 0.0077 | 0.0080 | | |
| AutoARIMA, Exchange (m=5, last 2500) | 0.0077 | 0.0079 | 0.008 | 0.013 (typo) |
| AutoARIMA, Exchange (m=5, full history) | 0.0073 | 0.0076 | | |

- **The windows are validated.** Deterministic SN reproduces every quoted value to 4 decimals.
- **SN season length:** m=30 on Exchange reproduces 0.013. Other periods give m=1: .0093, 5: .0107,
  7: .0112, 20: .0124, 60: .0169, 260: .0227. This matches TSFlow's code
  (`tsflow/utils/variables.py`: `season_lengths = {"H": 24, "D": 30, "B": 30}`, repo commit ff89317).
  So the quoted single-season rows use **m = 24 (hourly) and m = 30 (daily)**.
- **Model-based rows:** AutoETS and AutoARIMA on Exchange reproduce the quoted 0.008. Our recomputation
  independently confirms that 0.060 (SN-Electricity) and 0.013 (AutoARIMA-Exchange) are transcription errors.
- **Comparability:** `MultivariateEvaluator.mean_wQuantileLoss` pools all dimensions (sum of quantile loss
  over sum of |y|). This equals the univariate `Evaluator` over all series, and the SN values agree to
  1e-9 between modes. The only protocol difference between our harness and the quoted rows is the
  quantile grid (19 vs 9 levels), worth roughly 0.0003 on Exchange. **The quoted rows are comparable to AFM.**
  Side note: the Table 4 caption's "averaged over dimensions" is really a |y|-weighted pooled average.

### Task 1: Naive on Exchange

Command: `... r2_statistical_baselines.py naive`. statsforecast `Naive()` gives Gaussian intervals
σ√h, with σ from in-sample one-step residuals. It was fitted per series and window on **all
observations before the forecast origin**.

| History | afm mode (Table 4 metric) | tsflow mode |
|---|---|---|
| all before origin (as specified) | **0.0074** | 0.0077 |
| last 2500 | 0.0078 | 0.0081 |

Under the Table 4 metric, Naive (0.007) is below AutoETS/AutoARIMA (0.008, quoted) and AFM/TSFlow (0.009).
Caveat: on TSFlow's 9-quantile grid it is 0.0077, which rounds to 0.008 (a tie). The margin is at the
third decimal.

### Task 2: multi-seasonal baselines

History: last **1344 h (8 weeks)** before each origin, the same for all multi-seasonal models.
Forecasts are clipped to Ω.

Timing (10 series × 1 window unless noted):
- MSTL-ETS: 0.30 s/cell with 10 cells (dominated by pool start-up); **0.062 s/cell with 100 cells** (Traffic).
  Full run is about 11 min in total, so it was started without asking (under the 30-min rule).
- DHR-ARIMA: 10 cells take 8.0 s at K₁=K₂=2 and 16.0 s at K₁=K₂=10, i.e. 8-16 s CPU per fit.
  The full run is 10,290 fits (2590 + 959 + 6741), about **3-6 h** on 8 performance cores, plus about 40 min
  for K selection (5×5 grid × 20 series × 3 datasets). **Awaiting Ahmed's go-ahead.**

**MSTL-ETS full run** (2026-09-28). Command: `... run --model mstl --dataset <ds>`, history 1344,
`MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"))`, clipped to Ω.
Wall time: Solar 42 s, Electricity 113 s, Traffic 271 s.

| Dataset | afm mode (Table 4 metric) | tsflow mode | ND | Quoted single-season SN / AutoARIMA / AutoETS |
|---|---|---|---|---|
| Electricity | **0.0579** | 0.0603 | 0.0754 | 0.069 / 0.344 / 0.055 |
| Solar | **0.5198** | 0.5401 | 0.6039 | 0.512 / 0.558 / 0.550 |
| Traffic | **0.2475** | 0.2569 | 0.2912 | 0.221 / 0.486 / 0.492 |

Read-out (corrected 2026-09-28; an earlier version compared against the typo SN-Elec 0.060):
- Electricity: MSTL-ETS beats SN (0.069) and AutoARIMA, but not AutoETS (0.055).
- Solar and Traffic: it beats AutoARIMA and AutoETS, but not SN.
- It is far behind the deep models (e.g. AFM 0.042 / 0.284 / 0.089).

**DHR-ARIMA** (option B, chosen by Ahmed): K grid {2, 4, 6}², 20 random series (seed 0), first test window,
1344 h history. `AutoARIMA(season_length=1)` with self-computed Fourier regressors for periods 24 and 168,
phased on absolute time (Period ordinal in hours). Driver: `bash baselines/run_r2_dhr.sh`
(run under `caffeinate -i`), log in `results/r2/dhr_run.log`.
| Dataset | (K₁, K₂) | afm mode (Table 4) | tsflow mode | ND | Selection | Full run |
|---|---|---|---|---|---|---|
| Electricity | (6, 4) | **0.0656** | 0.0683 | 0.0827 | 62 s | 1029 s |
| Solar | (6, 2) | **0.4604** | 0.4764 | 0.5357 | 81 s | 380 s |
| Traffic | (6, 6) | **0.2018** | 0.2093 | 0.2444 | 89 s | 5551 s |

- Comparisons, derived programmatically from the table values:
  - DHR-ARIMA is the best statistical method on Solar and Traffic (it beats SN, AutoARIMA, AutoETS and MSTL).
  - On Electricity it beats SN and AutoARIMA but not AutoETS (0.055) or MSTL (0.058).
  - It stays far behind the deep models.
- Caveats, recorded here and not in the paper:
  - K₁ = 6 was selected on all three datasets, which is the edge of the reduced grid (option B); Traffic's K₂ = 6 is also at the edge.
  - AutoARIMA chose different differencing orders d across fits (d=0/1 counts: Elec 34/146, Solar 171/9,
    Traffic 108/72), so summed AICc compares some models with different d. This is the usual fpp-style
    practice, but not strictly like-for-like.
  - Traffic took 93 min instead of the extrapolated 45-70. The load average reached about 50-100, which
    suggests BLAS thread oversubscription inside the 10 workers. Setting `OMP_NUM_THREADS=1` would likely
    speed up a re-run.

---

## Task 3: Figure 2 (no re-run)

- Verified from code at `bd699df`: the plotted trajectories are held-out test trajectories, the band is the
  2.5-97.5% interval, and the dashed line is the mean (see 0.4).
- `examples/r2_fig2_relabel.py` produces `AFM_R2_revision/figures/fig2_r2.pdf` from `fig2_updated.pdf`.
  It replaces only the legend ("observed / within training horizon / beyond training horizon"; Helvetica
  14 pt; patch colours sampled from the rendered original). Panels are unchanged, and the old legend text
  is removed (redacted), not just covered.
- No Van der Pol column (no re-run; decision: minimal changes). "four" stays in Section 4.1.
- Response R2.1: dropped "400". The Fig. 2 run had 204 test trajectories (0.4), and the claim now reads
  "held-out test set".

## Task 4: Fig. B.4 (text only, no new runs)

`python examples/r2_b4_diagnostic.py` is read-only and uses the saved artefacts. Output of 2026-09-28:
- 20 engines × 14 sensors at all 108 steps; no NaN, no padding.
- FM (3-seed mean): score at steps 101-108 = .170 .176 .168 .168 .166 .163 .150 .131 (mean over 1-100: .181).
  cov90 = .919 .881 .901 .912 .930 .951 .969 .985 (mean over 1-100: .835).
- FM aggregate 0.1792 ± 0.0032 → 0.1803 ± 0.0032 without the last 4 steps.
  AFM 0.1802 ± 0.0040 → 0.1797 ± 0.0037.
- FM (seed 0): 90% width 7.58 at step 108 vs 2.077 on average (×3.65). MAE of mean 0.959 vs 0.574.
  Share of observations in the central 50%: 0.675 vs 0.426.
- AFM (seed 0): width ratio ×1.18; share in the central 50% 0.34-0.39 at the end.
- Reference only (not in the paper): energy-form CRPS for FM is 0.748 at step 108 vs 0.414 on average.
- Recomputed repo scores match the JSON for seed 0 exactly.
- FM training window: `_fm_window(108) = 40` (`examples/realworld_multiseed.py:35-37`), sampled over 108 steps.

Conclusion: the drop is not skill. FM's predictive spread inflates over its final sampled steps, and the
sample-rank score (0.6) does not penalise over-dispersion, so it improves as the truth moves to the
centre of the ensemble. It is not padding or misalignment. The effect on Table 6 is 0.001.

## Text changes made (2026-09-28)

`AFM_R2_revision/main.tex`:
- **Table 4:** transcription fixes to the TSFlow v1 source, marked `\rev`:
  - SN-Elec 0.069; AutoARIMA-Exch 0.008; AutoETS-Elec 0.055; SSSD Solar 0.384 ± 0.023, Traffic 0.097 ± 0.002.
  - Standard errors: DeepAR Solar/Traffic, WaveNet Solar/Traffic/Wiki, CSDI Exch/Solar/Traffic,
    SSSD Elec/Exch/Wiki.
  - New rows: Naive 0.007 (bold); MSTL-ETS 0.058 / 0.520 / 0.248.
  - Exchange underline moved to AutoARIMA and AutoETS (0.008).
  - Caption: TSFlow numbers are from arXiv v1. TimeGrad row untouched (source unverified).
- **Baselines paragraph:** m = 24 (hourly), SN m = 30 (daily); statsforecast 2.1.1; "same forecast
  origins and horizons" (not "context windows"); history rules; K selection procedure.
- **Results paragraph:** Exchange sentence (Naive best, stated plainly) and the MSTL sentence. Also
  **restored the Solar/Traffic/Wikipedia sentences**, which the delivered draft had accidentally commented
  out by prefixing a `%% TODO-R2` to that line.
- **Section 4.1:** added a `\rev` paragraph describing the sample-rank CRPS estimator and its properties.
- **Fig. B.4 caption:** explanation with numbers. **Fig. 2:** `\includegraphics` now points to `fig2_r2.pdf`.
- Build: `\rev` is now conditional on `\cleanmode`, and `main_clean.tex` is the clean wrapper.

`AFM_R2_revision/response_R2.tex`: AE 1 (Naive), AE 2 (m, MSTL table and sentence), R2.1 ("400" removed,
code check stated), R2.2 (bolding and the transcription corrections, without reference to the R1
explanation), R2.3 (full explanation).

`src/afm/evaluation/metrics.py`: docstring of `compute_crps` only. The code is unchanged, so published
numbers still reproduce.

Compiled: `latexmk -pdf main.tex` and `latexmk -pdf response_R2.tex` give no undefined references or
citations. Label numbers: Table 6 = `tab:realworld_degradation`, Fig. B.4 = `fig:app_cmapss_diag`.

## Task 5: final fill, checks and builds (2026-09-28)

- All `\TBD` and `%% TODO-R2` resolved. The unused `\TBD` macro definitions were removed, and `grep -c TBD`
  gives 0 for both `main.tex` and `response_R2.tex`.
- Bold and underline re-checked per column:
  - Electricity: AFM / TSFlow. Exchange: Naive / AutoARIMA + AutoETS. Solar: AFM / TSFlow.
  - Traffic: TSFlow / AFM. Wikipedia: WaveNet / SSSD.
  - No new row changes any column's best or second best except Exchange (Naive).
- Claims check: "best CRPS on two of the five datasets (Electricity and Solar)" still holds. The Results
  paragraph states the Exchange and multi-seasonal findings without softening.
- Code-availability sentence removed for this round (the comment in source marks where to add the GitHub
  link at acceptance).
- Table 4 caption: the dagger rows are "from quantile forecasts" (the old caption text says SampleForecast,
  which applies to AFM).
- Builds, all with 0 errors, 0 undefined references/citations and 0 TBD in the PDF text:
  - `latexmk -pdf -jobname=main_marked main.tex` → `main_marked.pdf` (33 pp).
  - `latexmk -pdf main_clean.tex` → `main_clean.pdf` (33 pp, no blue).
  - `latexmk -pdf response_R2.tex` → `response_R2.pdf`.
- Page check: Table 4 on p. 16, the CRPS paragraph on p. 13, Fig. B.4 on p. 32. Table 6 and Fig. B.4
  numbering match the response letter.

## Original proposed plan (superseded in part by the decisions above)

**Setup**
- Work on a new branch `r2-revision` and edit `AFM_R2_revision/` in place.
- Use a separate env for the baselines (`gluonts`, `statsforecast`, `utilsforecast`, all pinned) so the
  JAX pins in `pyproject.toml` are untouched.

**Tasks 1-2: statistical baselines (CPU, no AFM runs)**
- Write `baselines/r2_statistical_baselines.py`. It rebuilds the recovered harness: `*_nips` test windows
  via `MultivariateGrouper`, `MultivariateEvaluator` with 19 quantiles, and a `QuantileForecast` wrapper.
- Validate in two modes:
  - A = AFM harness.
  - B = TSFlow protocol (univariate `Evaluator`, 9 quantiles).
  - Checks: SN is deterministic, so it is a pure windowing/metric test against TSFlow's
    0.069/0.013/0.512/0.221/0.423. Then AutoETS on Exchange against 0.008.
  - Stop and report if mode B does not reproduce these, or if A and B differ materially.
- Run Naive on Exchange, then MSTL-ETS and DHR-ARIMA (K₁, K₂ chosen by AICc on 20 series) on
  Electricity, Solar and Traffic. Each window uses 1344 h of history.
- Time 10 series × 1 window before any full run.
- The Baselines paragraph must say "same forecast windows", not "same context windows": AFM's context
  (24-48 steps) is far too short for MSTL.

**Task 3: Fig. 2**
- Verification is done: the plotted trajectories are test trajectories, and the dashed line is the mean.
- Regeneration depends on D2 below. Either relabel the existing composite (legend text only), or
  regenerate all 5 systems, including Van der Pol, from a new run.

**Task 4: Fig. B.4 (no retraining needed)**
- Run an evaluation-only pass for seeds 1-2 to get raw samples.
- Sample FM with `n_steps_pred=112` (and 40) from the saved pkls, to separate the boundary effect from the
  40-vs-108 train/sample length mismatch.
- Recompute B.4 and Table 6 with a proper CRPS (see D1).

**Decisions needed from Ahmed**
- D1: Fix the CRPS metric?
  - Recommended scope: Table 6 and the B figures, recomputed from saved models in minutes.
  - Table 2 has no checkpoints, so it needs a full synthetic re-run after fixing the broken pipeline.
    CPU cost to be timed.
- D2: If Table 2 is re-run, also use independent Brownian paths per trajectory, a 2000/400 split and
  `stochastic=True`, to match the paper's description?
- D3: Correct the Table 4 transcription mismatches to the TSFlow source? No compute needed.
- D4: Keep the quoted single-season rows, or recompute them in-house under the same harness? Decide
  after the validation step.

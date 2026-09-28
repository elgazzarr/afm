# Brief for Claude Code: AFM paper, IJF second revision (R2)

**Manuscript:** IJF-D-26-00120R1, *Probabilistic Forecasting via Autoregressive Flow Matching*
**Journal:** International Journal of Forecasting
**Decision:** minor revision. **Resubmission deadline: 6 November 2026.**
**Owner:** Ahmed El-Gazzar

This file tells you (Claude Code, working in the AFM code repository) what to do for this revision round.
The LaTeX formatting and all text edits that do not depend on new results are **already done**. They are in
the `paper/` folder that ships with this brief. Your job is the part that needs the code: new baselines,
figure checks and regeneration, one investigation, and filling in the placeholders the results feed.

---

## 0. Ground rules (read first)

1. **Never invent a number.** Every value in the paper and response letter must come from a run you did
   in this session or from an existing results file you can point to. If something can't be run, leave the
   `\TBD` in place and say why in `R2_RESULTS.md`.
2. **Don't touch AFM or FM model code, and don't re-run the AFM rows of Table 4.** Those numbers stay as
   published. The exception is Task 4 (Fig. B.4), which may need a diagnostic FM run. Do it in a separate
   script and output folder.
3. **Don't overwrite existing results or figures.** Write new outputs to new files
   (e.g. `results/r2/…`, `figures/fig2_r2.pdf`).
4. **Ask before long runs.** Before anything expected to take more than about 30 minutes, or anything that
   needs the GPU, time a small subset first, give Ahmed the estimate, and wait for a go-ahead.
5. **Keep a log.** Maintain `R2_RESULTS.md` at the repo root. For each task, record the commands, library
   versions, runtimes, the numbers, and any deviation from this brief.
6. **Start with a plan.** First do Task 0 (recon), write down what you found and your plan, and check it
   with Ahmed before running experiments.

---

## 1. What the reviewers asked (verbatim, for context)

**Associate Editor**
> The authors have now added the seasonal naive model as a baseline for all real-world datasets, including
> Exchange. However, this is not what I meant.
>
> 1. **Exchange dataset.** Exchange rates are not seasonal; therefore, the appropriate benchmark is the naive
>    (see Hewamalage et al., 2023) not the seasonal naive introduced by the authors. I ask the authors to
>    include the naive forecast as a baseline for this dataset and remove the seasonal naive.
> 2. **Datasets with multiple seasonalities.** For the multi-seasonal datasets (Solar and Traffic), the authors
>    use AutoARIMA and AutoETS as classical baselines. However, these models only deal with a single
>    seasonality, not multiple ones. The same for the seasonal naive model, which accounts for only one
>    seasonal period. If they used an implementation suitable for multiple seasonalities, this should be
>    detailed. I therefore ask, for the datasets with multiple seasonalities, to include baselines designed
>    for multiple seasonal patterns; see https://otexts.com/fpppy/12-advanced.html
> 3. **Manuscript formatting.** [author–year references, single column] ← *already done in `paper/`*

**Reviewer 2**
> 1) [Figure 2] Is the plotted data in the blue and yellow shaded area unseen during training or just the
>    yellow window was unseen? In case the latter is true, I'd redo the pics with completely unseen data after
>    the gray shaded area. […]
> 2) Table 4: AutoETS-Exchange should be the one in bold. ← *already done*
> 3) Can you explain the sudden drop in CRPS for FM method in figure B.4?

The AE has now had to repeat points 1–3 because last round's response didn't do what was asked. This round,
the goal is to do **exactly** what is asked, verifiably.

---

## 2. What's in `paper/` (already done, don't redo)

- `main.tex`: converted to `elsarticle` `[preprint,12pt,authoryear]`, single column, `elsarticle-harv`
  author–year references. Bibliography errors fixed. Abstract ≤150 words, 5 keywords.
- Text changes for this round are wrapped in `\rev{…}` (shown in blue). Last round's changes are wrapped in
  `\oldrev{…}` (now black).
- **Every place that needs a result from you is marked `\TBD` (red in the PDF) and/or a
  `%% TODO-R2:` comment.** List them with `grep -n 'TBD\|TODO-R2' paper/main.tex paper/response_R2.tex`.
- Table 4 already has the new rows (Naive on Exchange; MSTL-ETS and DHR-ARIMA on Electricity/Solar/Traffic),
  SN removed for Exchange, and AutoETS bolded on Exchange.
- The Baselines paragraph (Section 4.2) is drafted. It describes: Naive with Gaussian intervals; MSTL with
  periods (24, 168) + AutoETS on the seasonally adjusted series; DHR = ARIMA + Fourier terms with K₁, K₂
  pairs; single-season SN/AutoARIMA/AutoETS with period m; statsforecast version; quantiles from Gaussian
  intervals at levels 10…90; clipping to the domain Ω. **Make the code match this description, or edit the
  description to match what the code does.** They must agree exactly.
- The Section 4.1 text and Table 2 headers are relabelled "within training horizon (steps 76–150)" and
  "beyond training horizon (steps 151–200)". The Fig. 2 caption now says prediction interval and states
  that the plotted trajectories are held-out test trajectories (**Task 3 must verify this**).
- `response_R2.tex`: a draft response letter with `\TBD`s.

Suggested location: copy `paper/` into the repo (e.g. `paper_r2/`) so you can compile it and edit it directly.

---

## 3. Tasks

### Task 0: Recon (no experiments yet)

Find and write down in `R2_RESULTS.md`:

- How the five GluonTS datasets are loaded. They are likely the Salinas et al. (2019) versions
  (`electricity_nips`, `exchange_rate_nips`, `solar_nips`, `traffic_nips`, `wiki2000_nips` or similar).
  Record how the **test windows** are built: number of rolling evaluation windows, prediction length
  (24 hourly / 30 daily), and context length.
- The evaluation harness: `MultivariateEvaluator` config, the quantile grid `0.05…0.95`, and whether the
  forecasts passed in are `SampleForecast` or `QuantileForecast`. The Table 4 metric is
  `mean_wQuantileLoss`.
- **Where the existing SN / AutoARIMA / AutoETS numbers in Table 4 came from.** Are they computed in this
  repo, or quoted from Kollovieh et al. (TSFlow, ICLR 2025)? If computed here, find the script and record
  the seasonal period `m`, the library and version, and the history length used. This decides the
  `$^\dagger$` markers and the `m=\TBD` in the paper.
- The script that produced Figure 2 (`figures/fig2_updated.pdf`) and the data split it uses.
- The C-MAPSS / FM code and the script that produced Fig. B.4 (`figures/appendix/cmapss_FD001_crps_vs_horizon.pdf`
  and `..._calibration_vs_horizon.pdf`).

Then propose a plan with runtime estimates and **stop for confirmation**.

### Task 1: Naive baseline on Exchange (AE point 1)

- Model: `statsforecast.models.Naive()`, the random walk (last value repeated). Its Gaussian prediction
  intervals widen as σ√h. Fit per series on each test window's history.
- Quantiles: request `level=[10,20,30,40,50,60,70,80,90]`. Map each level L to quantiles
  `(1−L/100)/2` and `(1+L/100)/2`, so 90 → 0.05/0.95, …, 10 → 0.45/0.55. The point forecast is the 0.5
  quantile. That gives exactly the grid `{0.05,…,0.95}`.
- Wrap the results as GluonTS `QuantileForecast` objects (keys `"0.05"…"0.95"` plus `"mean"`) and score them
  with **the same `MultivariateEvaluator` and windows as AFM**. If the harness only accepts
  `SampleForecast`, sample from the implied Gaussian with many samples (≥1000) and note this in the log.
- **Validate the harness first.** Recompute AutoETS on Exchange with the same wrapper and confirm it
  reproduces Table 4's **0.008**. If Table 4's SN numbers were computed in-house, also recompute SN on
  Electricity (**0.060**). If they don't match to the third decimal, stop and report. Don't paper over it.
- Output: Naive CRPS on Exchange → Table 4 cell, the Results sentence on Exchange, and the response letter.
  If Naive turns out to be the **best** method on Exchange, bold it, re-assign the underline, and say so
  plainly in the text. Don't soften it.

### Task 2: Multi-seasonal baselines (AE point 2)

Datasets: **Electricity, Solar, Traffic** (all hourly). Periods **24 and 168** for all three.
(Solar's weekly cycle is weak, but use the same periods for consistency with the AE's framing. It's fine
to note the weak weekly cycle in the log.)

- **MSTL-ETS:** `MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"))` from
  statsforecast. The seasonal components are forecast by seasonal naive internally. Prediction intervals
  come from the trend forecaster.
- **DHR-ARIMA:** `AutoARIMA` (non-seasonal, e.g. `season_length=1`) with Fourier exogenous regressors for
  both periods. Use `utilsforecast.feature_engineering.fourier` or equivalent; check the installed API, it
  has changed across versions. Select `K₁` (daily) and `K₂` (weekly) per dataset by AICc over a small grid,
  e.g. `K₁ ∈ {2,4,6,8,10}`, `K₂ ∈ {2,4,6,8,10}`, on a random subset of about 20 series. Then fix them for
  the whole dataset and record them (they fill `K_1=\TBD, K_2=\TBD` in the paper).
- *Optional:* `AutoTBATS(season_length=[24,168])` (De Livera et al., 2011, already in the bib as
  `delivera2011forecasting`), but only if it's cheap enough. Traffic has 963 series. If added, add a row to
  Table 4 and a sentence to the Baselines paragraph.
- **History length:** use the same training history per window as the existing classical baselines, if
  they were computed in-house. Otherwise use a fixed recent history, e.g. the last 8 weeks = 1344 h, which
  is enough for MSTL's two weekly cycles, for all multi-seasonal models. Document it in the log and in the
  Baselines paragraph.
- **Domain clipping:** clip all quantiles to Ω. Electricity and Solar are ≥ 0; Traffic is in [0, 1].
  Clipping preserves quantile order.
- Same quantile construction, wrapper and evaluator as Task 1.
- **Timing:** first time 10 series × 1 window per dataset and extrapolate. Use `n_jobs=-1`. Report the
  estimate before the full run.
- Output: 6 cells of Table 4 (2 models × 3 datasets), plus the sentence in the Results paragraph and the
  small table in the response letter. The expectation is that these improve on the single-season
  classical rows but stay well behind the deep models. **Report what you actually find**, whatever it is.

Save the script as something like `baselines/r2_statistical_baselines.py`, runnable end to end, with pinned
versions in the log. It'll be part of the mandatory code release.

### Task 3: Figure 2 check and regeneration (Reviewer 2 point 1)

1. **Verify** that every trajectory plotted in Fig. 2 comes from the 400 held-out **test** trajectories,
   never from the 2000 training trajectories. Show the evidence (indices/split) in the log. Also verify
   the Table 2 metrics are computed on test trajectories only.
   - If **yes**, the new caption and Section 4.1 text are correct.
   - If **no**, regenerate from test trajectories (still a good idea anyway) and tell Ahmed. The response
     letter's first paragraph to R2 then needs changing.
2. Regenerate the figure as `figures/fig2_r2.pdf` with:
   - legend labels **"observed"**, **"within training horizon"**, **"beyond training horizon"** (replacing
     "prediction" / "extrapolation");
   - shaded bands = **95% prediction interval** (2.5%–97.5% sample quantiles). Confirm the dashed line is
     the predictive **mean**, as the caption says; if it's the median, change the caption instead;
   - *optional but recommended:* add a **Van der Pol** column (the one system where FM wins on CRPS), so
     the figure isn't seen as cherry-picked. If added, change "four" to "five" in the Fig. 2 reference in
     Section 4.1.
3. Point `\includegraphics` in `main.tex` to `figures/fig2_r2.pdf`.

### Task 4: FM CRPS drop in Fig. B.4 (Reviewer 2 point 3)

In the C-MAPSS horizon-wise CRPS plot, FM's CRPS falls sharply over the **last ~4 steps (≈104–108)**,
from about 0.165 to about 0.13. At the same time FM's 90% coverage jumps toward 1.0, and FM's intervals
flare at the final steps in Figs. B.3 and B.6. AFM shows nothing similar. Find the cause. Check, in order:

1. **Sample size / padding per horizon step.** How many engines or windows contribute to each forecast
   step? If test engines with less remaining life are padded (zeros, last value, NaN-filled), or if fewer
   engines reach steps 104–108, the per-step CRPS at the end averages over a different or smaller set.
   Plot the count per step. Check whether padded targets enter the loss or the metric.
2. **Boundary effect of the whole-window FM.** The Brownian-motion prior's variance is largest at the last
   time index, and the bidirectional LSTM velocity field behaves differently at the sequence edge. Check
   FM's per-step interval width at the last steps.
3. **Off-by-one / alignment** of the last target step vs. the prediction.
4. **Decisive experiment (ask before running):** train FM with a slightly longer horizon (e.g. 112) and
   evaluate only the first 108 steps. If the drop moves to the new last steps, it's a boundary artefact
   of the whole-window model, not skill.

Also compute: aggregate FM CRPS on C-MAPSS **with and without** the last 4 steps, to quantify the effect on
Table 6.

Write the conclusion (2–4 sentences with numbers) into `R2_RESULTS.md`. Then:

- if it's a bug (padding/alignment): fix it, regenerate B.4 and the Table 6 FM row, and say so;
- if it's a genuine boundary artefact: keep the figure, and fill the `\TBD` sentence in the Fig. B.4
  caption and the R2.3 response with the explanation and the numbers.

### Task 5: Fill placeholders, compile, and check

1. Replace every `\TBD` and resolve every `%% TODO-R2` in `paper/main.tex` and `paper/response_R2.tex`.
   Delete each TODO-R2 comment once it's resolved.
2. **Re-check bold/underline** in every column of Table 4 after the new rows are in.
3. Search for claims that the new numbers could invalidate ("best", "outperform", "tying",
   "competitive", "2 of 5") and fix any that are now wrong.
4. Compile with `latexmk -pdf main.tex` and `latexmk -pdf response_R2.tex`. Make sure there are no
   undefined references or citations, and that `grep -c TBD` prints 0 for both files.
5. Produce two PDFs: `main_marked.pdf` (as is, changes in blue) and `main_clean.pdf` (build with
   `\newcommand{\rev}[1]{#1}`, and drop the blue `\oldrev` handling, which is already black).

### Task 6: Code release and anonymity

- The manuscript links to the code via an **anonymised** URL placeholder
  (`https://anonymous.4open.science/r/afm-XXXX`). Prepare the repo so Ahmed can create that mirror
  himself: remove or replace names, emails, institution names and absolute paths such as `/home/<user>/`
  in README, LICENSE, config files and notebooks. List what you changed. **Don't touch git history or
  push anything**; Ahmed will create the anonymous mirror.
- Make sure the new baseline scripts, the Fig. 2 script and the B.4 diagnostic are committed, with a short
  README section on how to reproduce Table 4's new rows.
- At acceptance (not now): restore `https://github.com/elgazzarr/afm` in `main.tex` and archive a tagged
  release on Zenodo for a DOI.

---

## 4. Definition of done

- [ ] `R2_RESULTS.md` has recon notes, commands, versions, runtimes and all new numbers.
- [ ] Harness validated: AutoETS-Exchange reproduces 0.008 (and SN-Electricity 0.060 if in-house).
- [ ] Naive on Exchange computed; Table 4, Results text and response letter updated; bolding re-checked.
- [ ] MSTL-ETS and DHR-ARIMA on Electricity, Solar and Traffic computed; K₁/K₂, m, history length and
      version filled in; the Baselines paragraph matches the code exactly.
- [ ] Fig. 2 verified as test data, regenerated with the new labels (+ Van der Pol if feasible).
- [ ] Fig. B.4 drop explained with evidence; caption and response filled in; figure/table fixed if it was
      a bug.
- [ ] `grep TBD` returns nothing in either `.tex` file; both compile cleanly; marked and clean PDFs built.
- [ ] Repo ready for the anonymous mirror; scripts committed.

Summarise in one short message to Ahmed: the new numbers, whether any conclusion changed, and anything
he needs to decide.

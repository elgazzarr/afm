# Design: Real-World Degradation-Trajectory Benchmarks for AFM

**Date:** 2026-06-18
**Author:** Ahmed El-Gazzar (with Claude)
**Context:** Rebuttal for IJF-D-26-00120 ("Probabilistic Forecasting via Autoregressive Flow Matching"). The Editor's primary rejection risk is *"benchmark further than with simple toy datasets."* This adds real-world datasets that run on the **existing lightweight CPU pipeline** (no GluonTS, no GPU).

---

## 1. Goal & Constraints

Add two real-world **degradation-trajectory** datasets to the AFM repo so AFM can be evaluated against published baselines:

1. **NASA Li-ion Battery Aging** (B0005/6/7/18) — capacity-fade forecasting.
2. **NASA C-MAPSS Turbofan** (FD001 first; FD002–FD004 later) — multivariate sensor forecasting **and** derived RUL prediction.

**Hard constraints (from the user):**
- Stay on the lightweight in-repo pipeline (small JAX AFM/FM models + `compute_crps`/`compute_nrmse`). **No GluonTS, no GPU.** The GluonTS path was deleted in the `restructre repo` commit (`f4131fe`) and is intentionally not revived.
- Minimize effort and risk — the user is time-pressed.

**Approved scope decisions:**
- Battery + C-MAPSS (both). M5 explicitly excluded (hierarchical counts, bespoke WRMSSE, needs the heavy path).
- C-MAPSS evaluated with **both** framings: forecasting (NRMSE/CRPS) **and** derived RUL (RMSE + NASA Score).
- Ship **self-contained scripts** (modeled on the README quickstart), **not** wired through the `BaseExperiment`/wandb runner (its post-restructure 4-tuple test-loader path is inconsistent).
- C-MAPSS starts at **FD001 only**; expand to FD002–FD004 if time permits.
- Research published baseline numbers as part of this work (done — see §7).

---

## 2. Integration Strategy (minimal surgery)

The entire downstream pipeline (`create_infinite_dataloaders` → AFM/FM models → `AFMTrainer` → `compute_crps`/`compute_nrmse` → plots) depends only on `prepare_trajectories` returning `(train_traj, test_traj, covariates)`. New datasets are therefore **purely new loaders** that reproduce that contract. No model, trainer, or metric changes.

```python
@dataclass
class RealDataset:
    train_traj: jnp.ndarray   # (n_train, T, dim)
    test_traj:  jnp.ndarray   # (n_test,  T, dim)
    covariates: jnp.ndarray   # (T, c)  — time/cycle index, expanded like the synthetic path
    data_dim:   int
    meta:       dict          # scaler stats (mean/std per feature) + dataset-specific extras
```

**Normalization (new for real data):** real series are z-scored per feature; the scaler stats live in `meta` so errors can be reported back in **original units** (capacity in Ah, RUL in cycles) for comparison with published numbers. The synthetic path is untouched.

**Why scripts, not the runner:** `BaseExperiment.__init__` hardcodes `create_system(...)`/`self.system.data_dim`, and `comparison.py`'s `evaluate_models` passes a tuple where the evaluator expects a 4-tuple test loader — inconsistent post-restructure. Scripts call the loader → build AFM(+FM) directly → `AFMTrainer.train` → a small eval helper that calls `model.sample` + `compute_crps`/`compute_nrmse` directly. Full control, low risk, easy to rerun.

---

## 3. New Files

```
src/afm/datasets/
  __init__.py        # RealDataset dataclass + load_dataset(name, config) registry/dispatcher
  battery.py         # NASA Li-ion loader (.mat → capacity-fade trajectories)
  cmapss.py          # C-MAPSS loader (txt → multivariate trajectories) + RUL utilities
  download.py        # download helpers, URLs, manual-fallback instructions; caches to data/
  eval.py            # evaluate_on_trajectories(model, test_traj, covariates, config, meta)
examples/
  realworld_battery.py
  realworld_cmapss.py
tests/
  test_real_datasets.py   # shape/contract checks; skip if raw data absent
data/                       # gitignored; downloaded raw files
```

**Dependency additions (`pyproject.toml`):** `scipy` (read `.mat`), `pandas` (parse C-MAPSS txt), `requests` (downloads).

---

## 4. NASA Battery — Capacity-Fade Forecasting

- **Cells:** B0005, B0006, B0007, B0018 (room-temperature group; ~168 discharge cycles, B0018 ~132). EoL convention = 70% of rated capacity.
- **Signal:** **multivariate** — per-discharge-cycle features modeled jointly: capacity, mean discharge voltage, discharge time, and measured temperature. (A univariate capacity-only mode is also exposed via a flag for ablation/sanity checks.) The capacity channel is the headline forecast target for comparison with published numbers; the extra channels exploit AFM's joint-distribution modeling.
- **Task:** given the first *k* cycles (context), forecast the remaining capacity trajectory (prediction + extrapolation windows, mirroring the synthetic split). Training samples come from within-cell windowing (the infinite loader samples random start positions); evaluation uses **leave-one-battery-out** (train on 3 cells, hold out 1) — standard in the literature.
- **Report:** NRMSE/CRPS (normalized) **+ capacity RMSE/MAE in Ah and as normalized-SOH fraction** (de-normalized via `meta`).
- **Loader detail:** each `.mat` has cycles tagged `charge`/`discharge`/`impedance`; extract the `Capacity` field from `discharge` cycles to form the per-cell capacity sequence.

---

## 5. C-MAPSS — Forecasting + RUL

- **Files:** `train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`. Columns: `unit, cycle, 3 op-settings, 21 sensors`.
- **Sensor subset:** standard informative set (drop constant sensors) — sensors 2,3,4,7,8,9,11,12,13,14,15,17,20,21 (~14). Per-sensor z-score (stats in `meta`).
- **Forecasting framing:** context = early engine cycles → forecast future multivariate sensor trajectory; NRMSE/CRPS on held-out future. AFM vs FM + last-value/naive baseline. Showcases multivariate joint modeling (a reviewer ask). *Note: no external published baseline exists for this metric — comparison is internal (AFM vs FM vs naive).*
- **RUL framing (the one genuinely new piece of logic):**
  1. Build a 1-D **health index (HI)** = first principal component of the z-scored sensors (sign-oriented so HI increases toward failure), fit on train run-to-failure engines.
  2. Calibrate a **failure threshold** = HI value at end-of-life across train engines.
  3. For each test engine, forecast the HI forward from its truncation point with AFM; **RUL = cycles until the forecast HI crosses the threshold**.
  4. Apply the standard **piecewise-linear RUL cap** (`Rearly = 125`, the modern convention — see §7 caveat) when comparing to published numbers.
  5. Report **RUL-RMSE + NASA asymmetric Score** against `RUL_FD00X.txt`.

---

## 6. Evaluation Helper & Outputs

`datasets/eval.py::evaluate_on_trajectories(...)`:
- Builds context/history from `test_traj`, calls `model.sample(...)` → `(S, B, T, dim)`.
- Computes CRPS + NRMSE (normalized) via existing `metrics.py`, split into prediction vs extrapolation windows.
- De-normalizes via `meta` to report errors in original units.
- For C-MAPSS, additionally runs the HI→RUL derivation and computes RUL-RMSE + Score.

**Deliverable for the rebuttal:** a generated markdown + LaTeX results table — AFM and FM rows, one column per metric, plus placeholder rows pre-filled with the published baselines from §7 (each tagged with its protocol).

---

## 7. Published Baseline Numbers (researched 2026-06-18)

> **Cite-safely rule:** always state the evaluation **protocol** in the same breath as the number. Protocol mismatch is exactly what the reviewers flagged in the original submission.

### 7.1 NASA Battery — protocols are NOT comparable; never pool them

**Protocol A — capacity-fade forecasting (matches our task):**

| Method | Cells | Metric | Value | Protocol | Source | Confidence |
|---|---|---|---|---|---|---|
| GPR (compound kernel) | B0005/6/7/18 | RMSE_Q (norm. 0–1) | ≈0.02 @ 5-cycle horizon → ≈0.06–0.08 @ ~40-cycle horizon; **EoL RMSE ≈ 56 cycles** | Train on capacity up to cycle *c*, forecast remainder; averaged over **all** start cycles from 20% of life | **Richardson, Osborne & Howey, *J. Power Sources* 357 (2017) 209–219**, DOI 10.1016/j.jpowsour.2017.05.004, arXiv:1703.05687 | **Canonical forecasting ref.** Lead with this. |
| CEEMDAN+SVR+LSTM | B0005/6/7/18 | RMSE (Ah) | B0005 0.0044, B0006 0.0055, B0007 0.0045, B0018 0.0086 | First 100 cycles train, remainder forecast; rolling multi-step | Fan et al., *Sci. Rep.* 15:8161 (2025) | Medium (one-off; decomposition pre-smooths capacity → optimistic) |
| CEEMDAN+Transformer+DNN | B0005/6/7/18 | RMSE (norm.) | B0005 0.0208*, B0006 0.0097, B0007 0.0180, B0018 0.0070 | Early-prediction (25–30% history) + ~50% start; 10-step iterative | Cai et al., *Heliyon* 9(7) (2023), PMC10344747 | Low-Med (*digits disagreed across extractions; verify PDF) |

**Protocol B — per-cycle SOH estimation (NOT forecasting; context only, never in the same column as A):**

| Method | Cells | Metric | Value | Protocol | Source |
|---|---|---|---|---|---|
| MLP/TCN/LSTM/GRU | B0005/6/7 | RMSE (norm. SOH) | B0005: MLP 0.0069, TCN 0.0071, LSTM 0.0076 | Chronological 80:20 within-cell estimation | Bairwa et al., *Sci. Rep.* 15 (2025), PMC12550000 |
| TCN | B0005/7/18 | RMSE (%SOH) | ≤1.80% worst-case | 80 cycles train, predict from cycle 90/100/110 (forecasting-ish) | Zhang et al., *Front. Energy Res.* 10:929235 (2022) |

**By-hand add (bib confirmed, per-cell digits paywalled):** Zhang, Xiong, He, Pecht, "LSTM RNN for RUL Prediction of Li-ion Batteries," *IEEE TVT* 67(7):5695–5705 (2018) — heavily-cited LSTM forecasting baseline; pull exact RMSE from IEEE Xplore (doc 8289406).

**Unit caution:** papers report Ah vs normalized fraction vs %. Rated ≈ 2 Ah → Ah-RMSE × 0.5 ≈ fraction; × 100 ≈ %. Convert before tabulating.

### 7.2 C-MAPSS RUL — anchor on FD001

**RMSE (lower better):**

| Method (paper) | Year | FD001 | FD002 | FD003 | FD004 | Cap | Window |
|---|---|---|---|---|---|---|---|
| CNN — Babu et al. | 2016 DASFAA | 18.45 | 30.29 | 19.82 | 29.16 | dataset-specific (~130) | 15 |
| LSTM — Zheng et al. | 2017 ICPHM | 16.14 | 24.49 | 16.18 | 28.17 | 130 | 30 |
| DCNN — Li, Ding, Sun | 2018 RESS | **12.61** | 22.36 | 12.64 | 23.31 | 125 | 30/20/30/15 |
| BiLSTM — Wang et al. | 2018 PHM-CQ | 13.65 | 23.18 | 13.74 | 24.86 | ~130 | sliding |
| AGCNN — Liu et al. | 2021 TII | 12.42 | 19.43 | 13.39 | 21.50 | 125 | 30 |
| DAST — Zhang, Song, Li | 2022 TIM | **11.43** | 15.25 | 11.32 | 18.36 | 125 | 40/60 |
| RVE — Costa & Sánchez | 2022 RESS | 13.42 | **14.92** | 12.51 | **16.37** | 125 | tuned |
| BiGRU-TSAM — Zhang et al. | 2022 RESS | 12.56 | 19.04 | 12.45 | 20.47 | 125 | sliding |

**NASA Score (lower better):**

| Method | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| CNN (Babu) | 1287 | 13570 | 1596 | 7886 |
| LSTM (Zheng) | 338 | 4450 | 852 | 5550 |
| DCNN (Li) | 273.7 | 10412 | 284.1 | 12466 |
| BiLSTM (Wang) | ~295 | ~4130 | 317 | ~5430 |
| AGCNN (Liu) | 225.5 | 1492 | 227 | 3392 |
| DAST | **203.15** | 924.96 | **154.92** | 1490.72 |
| RVE | 323.82 | 1379.17 | 256.36 | **1845.99** |
| BiGRU-TSAM | 213.35 | 2264.13 | 232.86 | 3610.34 |

**Cite-safely caveats:**
- **125 vs 130 RUL cap:** Zheng/Wang lineage = 130; Li-DCNN and most 2021+ work = 125. Lower cap → lower RMSE/Score; never compare across caps without a "cap" column. Babu's cap is genuinely dataset-specific — state as such, not "130".
- **Safe FD001 anchors** (appear near-verbatim in most comparison tables): CNN 18.45/1287, LSTM 16.14/338, DCNN 12.61/273.7, BiLSTM 13.65/~295.
- **Traps:** Babu↔Zheng often swapped (CNN ≈18–30, LSTM ≈16–28). DCNN "13.32" is the uncapped ablation — use 12.61. A weak generic "Bi-LSTM" (≈17.60) is NOT Wang 2018 (13.65).

**Citations:** Babu DASFAA 2016 (LNCS 9642, DOI 10.1007/978-3-319-32025-0_14); Zheng ICPHM 2017 (DOI 10.1109/ICPHM.2017.7998311); Li RESS 172 (2018) (DOI 10.1016/j.ress.2017.11.021); Wang PHM-Chongqing 2018 (DOI 10.1109/PHM-Chongqing.2018.00184); Liu TII 17(2) 2021 (DOI 10.1109/TII.2020.2983760); Zhang DAST TIM 71 2022 (DOI 10.1109/TIM.2022.3160561, arXiv:2106.15842); Costa RVE RESS 222 2022 (DOI 10.1016/j.ress.2022.108353); BiGRU-TSAM RESS 221 2022 (DOI 10.1016/j.ress.2021.108297).

---

## 8. Data Acquisition

`download.py` attempts auto-download, caches under `data/`, and on network failure prints manual instructions + expected local paths.
- **Battery:** NASA PCoE / data.nasa.gov "Li-ion Battery Aging Datasets" (`.mat` zip); Kaggle/GitHub mirror fallback.
- **C-MAPSS:** NASA PCoE "Turbofan Engine Degradation Simulation" (`CMAPSSData.zip`); Kaggle/GitHub mirror fallback.
- `data/` added to `.gitignore`.

---

## 9. Effort & Risk

- **Battery loader + eval:** low.
- **C-MAPSS forecasting:** low–medium.
- **C-MAPSS RUL derivation (HI → threshold crossing):** medium — the main new logic; HI definition and threshold calibration need a sanity check against known FD001 RUL ranges.
- **Data download reliability:** medium (NASA site quirks) → mitigated by mirror + manual fallback.

---

## 10. Open Items / Decisions Deferred to Implementation

- Exact battery-baseline digits flagged "verify PDF" in §7.1 to be confirmed against source tables before they enter the manuscript.
- HI construction for C-MAPSS (PCA-1 vs a simpler monotonic sensor) — pick the one whose RUL-RMSE on FD001 lands in a defensible range; document the choice.

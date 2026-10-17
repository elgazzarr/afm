# Reproducibility Package

**Manuscript:** *Probabilistic Forecasting via Autoregressive Flow Matching*
(Ahmed El-Gazzar & Marcel van Gerven; International Journal of Forecasting,
manuscript IJF-D-26-00120).

This README documents the data and code needed to reproduce the results of the
paper. It is the map to this repository: read it top to bottom, and you know
exactly what to run, what you will get, and how close to the published numbers
it will be. An honest summary of what is and is not exactly reproducible is
given in §2 and §11 — **please read §11 before comparing any regenerated
number against a published one.**

---

## 1. Assembly date, authorship, contact

- **Package assembled:** 16 October 2026.
- **Author / maintainer of this package:** Ahmed El-Gazzar
  (ahmed.elgazzar@donders.ru.nl) — contact for anything that does not run.
- **Repository:** https://github.com/elgazzarr/afm (the code release is
  identified by the `ijf-reproducibility` git tag).
- **Paper (preprint):** https://arxiv.org/abs/2503.10375

## 2. What can be reproduced, and how exactly

| Reproducibility level | Outputs |
|---|---|
| **Exact** (deterministic, re-run gives the published numbers) | Table 4 statistical-baseline rows marked † (Naive, MSTL-ETS, DHR-ARIMA); harness validation against the quoted SeasonalNaive/AutoETS/AutoARIMA values |
| **From shipped artefacts** (saved model checkpoints + eval-only passes; deterministic given the checkpoints — verified: the evaluation-only commands regenerate the shipped result files **byte-identically**) | Table 6 (real-world degradation, 3 seeds); Tables B.7–B.8; Figs. B.3–B.7 (appendix diagnostics); Fig. B.4 supporting analysis |
| **Protocol-level** (the exact pipeline runs end-to-end, but the published numbers themselves cannot be regenerated bit-exactly — see §11) | Table 2 + Fig. 2 (synthetic systems); Table 4 AFM row + Table 5 (GluonTS benchmarks) |
| **Quoted from third parties** (cited, not re-run) | All other Table 4 rows (TSFlow, DeepAR, TimeGrad, CSDI, SSSD, WaveNet) — taken from Kollovieh et al. (2024) arXiv v1 and Rasul et al. (2021) as stated in the Table 4 caption |

Fig. 1 (framework diagram) is an illustration, not generated from data.

## 3. Repository structure

```
afm/
├── REPRODUCIBILITY.md        ← this file
├── README.md                 ← library overview + quick start
├── LICENSE                   ← MIT (code)
├── pyproject.toml            ← main environment pins; optional [gluonts] extra
├── verify_rng_streams.py     ← RNG-independence audit of the AFM/FM samplers
├── src/afm/                  ← the library
│   ├── models/               ← AFM + FM architectures (synthetic/real-world studies)
│   ├── systems/              ← synthetic dynamical systems (Table 1 / Table 2 data)
│   ├── training/             ← dataloaders (incl. evaluation loader) + trainers
│   ├── evaluation/           ← CRPS/NRMSE metrics, model evaluator
│   ├── experiments/          ← CLI experiment framework (afm run …)
│   ├── datasets/             ← NASA battery / C-MAPSS loaders, diagnostics, plots
│   ├── configs/experiments/  ← bundled experiment YAMLs (e.g. afm_vs_fm_single)
│   └── gluonts_pipeline/     ← GluonTS benchmark pipeline (Table 4 AFM row);
│                                faithful port of the exact pre-restructure code
├── examples/                 ← runnable scripts, one per published result (see §7)
├── baselines/                ← Table 4 statistical baselines + pinned env (§4.2)
├── data/                     ← raw real-world data, included in-repo (§5)
├── results/realworld/        ← shipped checkpoints + result artefacts (§6)
├── results/r2/               ← statistical-baseline result JSONs (§6)
└── tests/                    ← pytest suite (15 tests)
```

## 4. Computing environments

Two separate environments are used. All commands below assume the repository
root as working directory.

### 4.1 Main environment (library, synthetic experiments, real-world
degradation studies, GluonTS pipeline)

Python **3.12.7** with the pins in `pyproject.toml`. Exact versions used for
everything documented in this README:

```
jax==0.8.1            flax==0.10.4        diffrax==0.7.2     optax==0.2.5
lineax==0.0.8         jax-dataloader==0.1.4                 numpy==2.2.5
pandas==2.3.3         scipy==1.17.1      matplotlib==3.10.9 seaborn==0.13.2
wandb==0.27.2         gluonts==0.17.0    PyYAML==6.0.3      tqdm==4.67.1
```

Install and set up:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[gluonts]"     # [gluonts] adds gluonts==0.17.0 (Table 4 pipeline)
pytest tests/                   # 15 tests, ~30 s, CPU-only
```

CPU-only is sufficient for everything except full GluonTS benchmark training
(see §9). Weights & Biases is **optional**: the experiment CLI and the
GluonTS pipeline default to `wandb` disabled mode; set `WANDB_MODE=online`
(or pass `--use-wandb` to the GluonTS pipeline) to enable logging.

### 4.2 Statistical-baselines environment (Table 4 rows marked †)

A separate, pinned environment (kept separate because `statsforecast`
requires newer numpy than the JAX pins allow). Recreate exactly:

```bash
uv venv baselines/.venv --python 3.12
uv pip install --python baselines/.venv/bin/python -r baselines/requirements.txt
```

Frozen versions (Python 3.12.7): `gluonts==0.17.0`, `statsforecast==2.1.1`,
`utilsforecast==0.2.16`, `numpy==2.5.3`, `pandas==2.3.3` — full freeze in
`baselines/requirements.txt`.

## 5. Data

### 5.1 Synthetic dynamical systems (Tables 1–2, Fig. 2)

Generated by code (`src/afm/systems/systems.py`; simulate via
`afm create_system(name, stochastic=…)`). No external data. Available
systems: `lorenz`, `lotka_volterra`, `vanderpol`, `harmonic_oscillator`,
`fitzhugh_nagumo`, `brusselator`, `duffing`. Note (factual, for exact
re-generation): every simulator seeds its `VirtualBrownianTree` with a fixed
key (`PRNGKey(0)`), i.e. stochastic-mode trajectories share one Brownian
path per process.

### 5.2 NASA Li-ion Battery Aging dataset (Table 6, battery; Figs. B.6–B.7, Table B.7)

- **What:** capacity-fade measurements for cells B0005/B0006/B0007/B0018
  (`.mat` files, 4 channels: measured voltage/current/temperature/load).
- **Included in this repository at** `data/battery/` (~54 MB, unmodified).
- **Original source:** NASA Prognostics Center of Excellence,
  https://ti.arc.nasa.gov/c/5/ (interstitial-gated host; manual download).
  Kaggle mirror: `patrickfleith/nasa-battery-dataset`.
- **Licence:** NASA PCoE data are freely distributed for research; no
  explicit open licence is attached by NASA. Redistribution here is for
  the reproducibility check; please cite NASA PCoE.
- **Loader / protocol:** `src/afm/datasets/battery.py`
  (leave-one-battery-out, context 66 cycles ≈ forecast 66 cycles).

### 5.3 NASA C-MAPSS turbofan degradation dataset (Table 6, C-MAPSS; Figs. B.3–B.5, Table B.8)

- **What:** `train_FD001.txt` / `test_FD001.txt` / `RUL_FD001.txt`
  (FD001 subset; 14 selected sensors, first 20 engines, 128-cycle windows).
- **Included in this repository at** `data/cmapss/` (~5.5 MB, unmodified).
- **Original source:** https://ti.arc.nasa.gov/c/6/ (manual download).
  Kaggle mirror: `behrad3d/nasa-cmaps`. Same licence note as §5.2.
- **Loader / protocol:** `src/afm/datasets/cmapss.py` (context 20 cycles,
  forecast horizon 108).

### 5.4 GluonTS benchmark datasets (Table 4; `*_nips` variants)

`electricity_nips`, `solar_nips`, `traffic_nips`, `exchange_rate_nips`,
`wiki-rolling_nips` — **auto-downloaded on first use** by GluonTS
(`get_dataset(name, regenerate=False)`, ~250 MB total, cached in
`~/.gluonts/`). Distributed with GluonTS (Apache-2.0 code); the underlying
series come from the public NIPS 2017 forecasting competition corpus.
No registration required.

## 6. Intermediary datasets / saved artefacts (shipped in-repo)

These allow a reviewer to skip retraining and verify the published
real-world numbers by evaluation only:

| Artefact | Generated by | Used by |
|---|---|---|
| `results/realworld/models/*_{AFM,FM}_seed{0,1,2}.pkl` (12 checkpoints) | `examples/realworld_multiseed.py` (training pass, June 2026) | `examples/realworld_multiseed.py` (resume/eval-only), Table 6 |
| `results/realworld/cmapss_FD001_multiseed.json`, `battery_B0007_multiseed.json` (per-seed scalars + horizon curves) | `examples/realworld_multiseed.py` | Table 6, Figs. B.3–B.7, `examples/make_appendix.py`, `examples/r2_b4_diagnostic.py` |
| `results/realworld/*_figdata.npz` (seed-0 raw predictions/ground truth) | `examples/realworld_multiseed.py` | `examples/make_appendix.py`, `examples/r2_b4_diagnostic.py` |
| `results/realworld/appendix/*` (appendix PDFs + LaTeX tables) | `examples/make_appendix.py` | Appendix B |
| `results/r2/*.json` (Naive/MSTL/DHR results + timings + harness validation) | `baselines/r2_statistical_baselines.py`, `baselines/run_r2_dhr.sh` | Table 4 rows marked † |

Deleting `results/realworld/models/` forces full retraining instead
(≈3 min/seed on CPU for these datasets; see §8).

## 7. Which code produces which output

| Paper output | Script(s) | Command | Reproducibility |
|---|---|---|---|
| Table 1 (system specs) | `src/afm/systems/systems.py` | (specification table; systems listed in `README.md`) | documentation |
| Table 2 (AFM vs FM, 5 synthetic systems) | `src/afm/experiments/` via bundled configs | `afm run afm_vs_fm_single --system <name>` (one run per system; see `src/afm/configs/experiments/`) | protocol-level (§11) |
| Fig. 2 (example forecasts) | pre-restructure run + `examples/r2_fig2_relabel.py` | legend relabelling only: `python examples/r2_fig2_relabel.py` | protocol-level (§11) |
| Table 3 (dataset properties) | GluonTS metadata | (properties listed from `get_dataset(...).metadata`) | documentation |
| Table 4 — AFM row | `src/afm/gluonts_pipeline/` | `python -m afm.gluonts_pipeline.run --dataset_name <ds> [...]` | protocol-level (§11) |
| Table 4 — Naive†, MSTL-ETS†, DHR-ARIMA† rows | `baselines/r2_statistical_baselines.py`, `baselines/run_r2_dhr.sh` | see §7.1 | **exact** |
| Table 4 — other baseline rows | — | quoted from Kollovieh et al. (2024) v1 / Rasul et al. (2021) | cited |
| Table 5 (efficiency) | from Table 4 AFM runs + quoted TSFlow values | — | protocol-level |
| Table 6 (degradation, 3 seeds) | `examples/realworld_multiseed.py` | see §7.2 (eval-only) | **exact from shipped checkpoints** |
| Table B.7 (battery vs published) | `examples/make_appendix.py` | `python examples/make_appendix.py --in-dir results/realworld` | exact from artefacts |
| Table B.8 (C-MAPSS diagnostics) | `examples/make_appendix.py` | as above | exact from artefacts |
| Fig. B.3 (C-MAPSS forecast fan) | `examples/make_appendix.py` | as above | exact from artefacts |
| Fig. B.4 (C-MAPSS calibration/CRPS vs horizon) | `examples/make_appendix.py` + `examples/r2_b4_diagnostic.py` | as above, then `python examples/r2_b4_diagnostic.py` | exact from artefacts |
| Fig. B.5 (C-MAPSS correlation matrices) | `examples/make_appendix.py` | as above | exact from artefacts |
| Fig. B.6 (battery forecast fan) | `examples/make_appendix.py` | as above | exact from artefacts |
| Fig. B.7 (battery diagnostics) | `examples/make_appendix.py` | as above | exact from artefacts |
| RNG-independence analysis (rebuttal) | `verify_rng_streams.py` | `PYTHONPATH=src python verify_rng_streams.py [n_seeds]` | exact (CPU) |

### 7.1 Table 4 statistical baselines (exact)

```bash
# harness validation (reproduces the quoted SN/AutoETS/AutoARIMA values)
baselines/.venv/bin/python baselines/r2_statistical_baselines.py validate

# Naive on Exchange (Table 4)
baselines/.venv/bin/python baselines/r2_statistical_baselines.py naive

# MSTL-ETS on the hourly datasets
baselines/.venv/bin/python baselines/r2_statistical_baselines.py run --model mstl --dataset electricity_nips   # also solar_nips, traffic_nips

# DHR-ARIMA: choose (K1, K2) by AICc, then run
bash baselines/run_r2_dhr.sh
```

Seasonal periods: m = 24 (hourly), m = 30 (daily/Exchange, SN), m = 5
(business-daily, AutoETS/AutoARIMA); history = 1344 h for the multi-seasonal
models, all history before the origin for Naive/Exchange. Outputs are
written to `results/r2/*.json`. The shipped JSONs are the ones used in the
published table.

### 7.2 Table 6 and the appendix (evaluation-only, from shipped checkpoints)

```bash
# C-MAPSS FD001 (ctx=20, H=108, 3 seeds) — skips training, checkpoints exist
python examples/realworld_multiseed.py --dataset cmapss --subset FD001 --ctx 20 --seeds 3 --epochs 50

# Battery B0007 (ctx=66, H=66, leave-one-battery-out, 3 seeds)
python examples/realworld_multiseed.py --dataset battery --holdout B0007 --ctx 66 --seeds 3 --epochs 50

# regenerate all appendix figures + LaTeX tables from the artefacts
python examples/make_appendix.py --in-dir results/realworld

# Fig. B.4 supporting diagnostics (read-only analysis of the artefacts)
python examples/r2_b4_diagnostic.py
```

The multiseed script is resumable: a saved `results/realworld/models/*.pkl`
is loaded instead of retraining, so with the shipped checkpoints these
commands are evaluation-only and deterministic.

## 8. Hardware and expected runtimes

Hardware used for the shipped artefacts and all CPU measurements below:
**Apple M1 Pro, 10 cores, 32 GB RAM** (macOS). The paper's Table 2/Table 4
training runs used a single **NVIDIA A100-SXM4-40GB** (GPU cluster).

| Step | Hardware | Expected runtime |
|---|---|---|
| `pytest tests/` | CPU (M1 Pro) | ~30 s |
| `afm run afm_vs_fm_single --num-epochs 1` (smoke) | CPU (M1 Pro) | ~2 min (AFM ≈12 s/epoch, FM ≈20 s/epoch, eval ≈40 s) |
| `afm run afm_vs_fm_single` (default 50 epochs, one system) | CPU (M1 Pro) | ≈30 min; ≈7 min on A100 (as reported in the paper, training to convergence) |
| Table 2 protocol, all 5 systems | A100 (paper) | ≈35 min total |
| GluonTS pipeline smoke (Exchange, 3 training steps, 3 samples) | CPU (M1 Pro) | ~50 s incl. download + evaluation |
| GluonTS pipeline, full training per dataset | A100 (paper) | 40–100 min per dataset; **CPU not recommended** beyond Exchange |
| Table 6 evaluation-only (both datasets, 3 seeds) | CPU (M1 Pro) | ~4 min total (C-MAPSS ≈2.5 min, Battery ≈45 s); regenerates the shipped `*_multiseed.json` / `*_figdata.npz` **byte-identically** |
| Table 6 full retraining (delete `results/realworld/models/`) | CPU (M1 Pro) | tens of minutes per seed (AFM 50 epochs + FM 100 epochs; not timed exactly — use the shipped checkpoints to skip this) |
| Baselines `validate` | CPU (M1 Pro) | ≈5 min with cached GluonTS datasets (first use adds the ~250 MB download) |
| Baselines Naive (Exchange) | CPU (M1 Pro) | ~2 min |
| Baselines MSTL-ETS (full runs) | CPU (M1 Pro) | Solar 42 s; Electricity 113 s; Traffic 271 s |
| Baselines DHR-ARIMA (K selection + full runs) | CPU (M1 Pro, 8 workers) | K selection 1–1.5 min/dataset; full runs: Electricity ≈17 min, Solar ≈6 min, Traffic ≈93 min |
| `make_appendix.py` / `r2_b4_diagnostic.py` | CPU | seconds |
| `verify_rng_streams.py` | CPU | ~1 min/seed |

## 9. Special setup requirements

- **GPU:** required only to retrain the GluonTS benchmarks (Table 4 AFM row)
  at full scale, and recommended (not required) for full multi-system
  synthetic training. Everything else — the entire real-world degradation
  study, all statistical baselines, all artefact-based reproduction, and the
  GluonTS pipeline on Exchange with reduced training — runs on a laptop CPU.
- **GluonTS datasets:** auto-downloaded (~250 MB) on first use of the
  Table 4 pipeline or the baselines harness; cached in `~/.gluonts/`.
- **Separate baselines environment:** required (§4.2) — numpy pin conflict.
- **Internet:** needed once (GluonTS dataset download). All raw real-world
  data is already in the repository. NASA's own hosts are request-gated,
  which is why the battery/C-MAPSS files are shipped directly.
- **No cluster/HPC, no accounts, no manual steps** are required for anything
  in §7 except the full GluonTS retrains (wandb is optional and disabled by
  default).

## 10. Licences

- **Code:** MIT (`LICENSE`). Copyright 2025–2026 Ahmed El-Gazzar.
- **Battery + C-MAPSS data:** NASA PCoE, freely distributed for research,
  no explicit open licence; cite NASA when reusing.
- **GluonTS datasets:** distributed with GluonTS (Apache-2.0); underlying
  electricity/solar/traffic/exchange series from the public NIPS 2017
  forecasting corpus.
- **Quoted baseline numbers** remain the property of their original
  publications (cited in Table 4 of the paper).

## 11. Known limitations — read before comparing numbers

1. **Table 2 + Fig. 2 (synthetic systems): the producing run is not
   recoverable.** The published values were produced before the public
   repository restructure; no config or checkpoints for that specific run
   were archived, and the manuscript's Section 4.1 settings (2400 initial
   conditions, 2000/400 split) do not correspond to any config in the
   repository's history. The pipeline shipped here (`afm run
   afm_vs_fm_single`) implements the same protocol and reproduces the
   qualitative result (AFM beats FM on the chaotic/stochastic systems;
   e.g. a 1-epoch smoke run on Lorenz already gives AFM CRPS 0.155 vs FM
   0.176), but **regenerated Table 2 numbers will differ from the published
   ones**.
2. **Table 4 AFM row: run configurations were not archived.** The pipeline
   in `src/afm/gluonts_pipeline/` is a faithful port of the exact code used
   (git revision `f4131fe^`, verified byte-identical except for import
   paths, wandb-optionality, and one crash fix — see the module docstrings),
   and it runs end-to-end. However, the per-dataset hyperparameters of the
   published runs (notably `context_ratio`, which sets the context length,
   plus seeds/step counts; runs were logged to a private wandb project) were
   not recorded, so **the published AFM values cannot be regenerated
   bit-exactly**. Treat any regenerated Table 4/5 numbers as new runs, not
   verifications.
3. **CRPS estimator.** All CRPS values on the synthetic systems and the
   degradation benchmarks (Tables 2 and 6, Figs. B.3–B.7) use the
   sample-rank CRPS estimator described in Section 4.1 of the manuscript;
   it is not the integral-form CRPS (also discussed there). Table 4 is
   unaffected (GluonTS `mean_wQuantileLoss`).
4. **Fig. 2 panels** come from the pre-restructure run (composite figure);
   only the legend relabelling (`examples/r2_fig2_relabel.py`) is
   scriptable in this repository.
5. **Stochastic-mode simulators** share one Brownian path
   (`VirtualBrownianTree` seeded with `PRNGKey(0)`) across trajectories of
   a process — see §5.1.

If anything does not run as documented, or a regenerated number is so far
from a published one that it looks like a bug rather than the documented
non-exactness above, please contact the package author (§1).

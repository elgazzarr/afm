# AFM: Autoregressive Flow Matching for Probabilistic Time Series Forecasting

AFM is a JAX library for probabilistic multivariate time series forecasting. Given historical observations and optional future covariates, AFM models the joint distribution of future trajectories by decomposing it into a sequence of conditional densities, each learned via a shared flow matching transformation. This combines the strengths of autoregressive models (strong extrapolation, compact size, calibrated uncertainty) with the ability of transport-based generative models to capture complex, multi-modal distributions.

[[Paper]](https://arxiv.org/abs/2503.10375)

<p align="center">
  <img src="assets/framework.png" width="100%" alt="AFM Framework"/>
</p>

<p align="center">
  <img src="assets/examples.png" width="100%" alt="AFM vs Standard Flow Matching Predictions"/>
</p>

## Installation

```bash
pip install -e .
# with the GluonTS benchmark pipeline (Table 4 of the paper):
pip install -e ".[gluonts]"
```

Weights & Biases logging is optional and disabled by default; set
`WANDB_MODE=online` to enable it.

## Quick Start

```python
import jax.random as jrandom
import flax.nnx as nnx

from afm import AutoregressiveFlowMatching, Config, create_system
from afm.training.data import prepare_trajectories, create_infinite_dataloaders
from afm.training.trainers import AFMTrainer

# 1. Generate training data from a dynamical system
system = create_system("lorenz", stochastic=True)
config = Config(system_name="lorenz", system_stochastic=True)
train_traj, test_traj, ts = prepare_trajectories(system, config, jrandom.PRNGKey(0))
(train_loader, test_loader), _ = create_infinite_dataloaders(train_traj, test_traj, ts, config)

# 2. Build and train
model = AutoregressiveFlowMatching(
    dim_x=system.data_dim, dim_c=config.model_covariates_dim,
    dim_h=config.model_hidden_dim, dim_t=config.model_time_dim,
    dim_hidden=config.model_hidden_dim, hidden_blocks=config.model_hidden_blocks,
    n_rnn_layers=config.model_n_rnn_layers, rnn_hidden=config.model_rnn_hidden,
    rngs=nnx.Rngs(0),
)
trainer = AFMTrainer(model, learning_rate=config.training_learning_rate)
trainer.train(train_loader, test_loader, num_epochs=50, use_wandb=False)

# 3. Probabilistic forecast
samples = model.sample(
    jrandom.PRNGKey(42),
    test_traj[0, :config.context_length][None],   # history  (1, ctx, dim)
    ts[:config.context_length, None][None],        # covariates (1, ctx, 1)
    nfe=20, n_steps_pred=30, n_samples=50,
)  # → (50, 1, 30, dim)
```

## Available Systems

| System | Dimensions | Description |
|--------|-----------|-------------|
| `lorenz` | 3 | Chaotic attractor |
| `lotka_volterra` | 2 | Predator-prey dynamics |
| `vanderpol` | 2 | Nonlinear oscillator |
| `harmonic_oscillator` | 2 | Linear oscillator |
| `fitzhugh_nagumo` | 2 | Neuronal dynamics |
| `brusselator` | 2 | Chemical oscillator |
| `duffing` | 2 | Forced nonlinear oscillator |

All systems support both deterministic and stochastic (SDE) modes via `create_system(name, stochastic=True/False)`.

## Real-world datasets

AFM includes scripts to evaluate on two real-world datasets for degradation and failure prediction:

- `examples/realworld_battery.py`: Battery capacity fade forecasting on NASA battery cycles.
- `examples/realworld_cmapss.py`: Engine remaining useful life (RUL) prediction on NASA C-MAPSS data.
- `examples/realworld_multiseed.py`: the paper's Table 6 protocol (3 seeds, AFM vs FM, resumable from saved checkpoints in `results/realworld/models/`).

Raw data should be placed in the `data/` directory. Data URLs and acquisition instructions are documented in `src/afm/datasets/download.py`. Evaluation is CPU-only (no GluonTS or GPU required).

## GluonTS benchmarks (Table 4 of the paper)

The AFM runs on the GluonTS benchmarks (`electricity_nips`, `solar_nips`,
`traffic_nips`, `exchange_rate_nips`, `wiki-rolling_nips`) use the pipeline
in `src/afm/gluonts_pipeline/` — a faithful port of the exact code used for
the paper (datasets auto-download on first run; full-scale training needs a
GPU, ~40–100 min per dataset on an A100):

```bash
python -m afm.gluonts_pipeline.run --dataset_name exchange_rate_nips \
    --training_steps 1000 --context_ratio 2.0
```

Note: the per-dataset run configurations of the published Table 4 values
were not archived, so regenerated values will differ from the published
ones — see `REPRODUCIBILITY.md` §11.

## Statistical baselines for the GluonTS benchmarks (Table 4)

`baselines/r2_statistical_baselines.py` scores the `statsforecast` baselines (Naive on Exchange; MSTL-ETS and DHR-ARIMA with periods 24 and 168 on Electricity, Solar and Traffic) on the same GluonTS test windows and `MultivariateEvaluator` (quantiles 0.05, ..., 0.95) as AFM. It uses a separate environment:

```bash
uv venv baselines/.venv --python 3.12
uv pip install --python baselines/.venv/bin/python gluonts==0.17.0 statsforecast==2.1.1 utilsforecast==0.2.16
baselines/.venv/bin/python baselines/r2_statistical_baselines.py validate   # reproduces the quoted SN/AutoETS/AutoARIMA values
baselines/.venv/bin/python baselines/r2_statistical_baselines.py naive      # Naive on Exchange
baselines/.venv/bin/python baselines/r2_statistical_baselines.py run --model mstl --dataset solar_nips
bash baselines/run_r2_dhr.sh                                                # DHR-ARIMA: choose (K1, K2) by AICc, then run
```

Results are written to `results/r2/`. `examples/r2_b4_diagnostic.py` reproduces the analysis of the FM horizon-wise score on C-MAPSS (Fig. B.4) from the saved run artefacts.

## CLI for Paper Experiments

The package includes a CLI for reproducing the experiments from the paper:

```bash
afm list                           # list available experiments
afm run afm_vs_fm_single           # run bundled experiment config
afm run path/to/config.yaml        # run custom config
afm validate afm_vs_fm_single      # check config validity
```

See `docs/experiments.md` for details on the experiment system.

## Reproducibility

A complete map from every table and figure in the paper to the script that
produces it, the computing environments, data provenance, expected
runtimes, and an explicit account of what is (and is not) exactly
reproducible is given in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Citation

```bibtex
@article{el2025probabilistic,
  title={Probabilistic Forecasting via Autoregressive Flow Matching},
  author={El-Gazzar, Ahmed and van Gerven, Marcel},
  journal={arXiv preprint arXiv:2503.10375},
  year={2025}
}
```

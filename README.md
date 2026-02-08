# AFM: Autoregressive Flow Matching for Probabilistic Time Series Forecasting

AFM is a JAX library for probabilistic multivariate time series forecasting. Given historical observations and optional future covariates, AFM models the joint distribution of future trajectories by decomposing it into a sequence of conditional densities, each learned via a shared flow matching transformation. This combines the strengths of autoregressive models (strong extrapolation, compact size, calibrated uncertainty) with the ability of transport-based generative models to capture complex, multi-modal distributions.

[[Paper]](https://arxiv.org/abs/2503.10375)

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import jax
import jax.random as jrandom
import flax.nnx as nnx
import numpy as np
import matplotlib.pyplot as plt

from afm import AutoregressiveFlowMatching, Config, create_system
from afm.training.data import prepare_trajectories, create_infinite_dataloaders
from afm.training.trainers import AFMTrainer

# 1. Create a dynamical system and generate data
system = create_system("lorenz", stochastic=True)

config = Config(
    system_name="lorenz",
    system_stochastic=True,
    system_num_timepoints=100,
    system_t0=0.0,
    system_t1=2.0,
    context_length=50,
    data_n_train_test_samples=512,
    data_batch_size=64,
    steps_per_epoch=100,
    training_num_epochs=50,
)

key = jrandom.PRNGKey(0)
train_traj, test_traj, ts = prepare_trajectories(system, config, key)

# 2. Create dataloaders
(afm_train, afm_test), _ = create_infinite_dataloaders(
    train_traj, test_traj, ts, config
)

# 3. Build and train the model
model = AutoregressiveFlowMatching(
    dim_x=system.data_dim,
    dim_c=config.model_covariates_dim,
    dim_h=config.model_hidden_dim,
    dim_t=config.model_time_dim,
    dim_hidden=config.model_hidden_dim,
    hidden_blocks=config.model_hidden_blocks,
    n_rnn_layers=config.model_n_rnn_layers,
    rnn_hidden=config.model_rnn_hidden,
    rngs=nnx.Rngs(0),
)

trainer = AFMTrainer(model, learning_rate=config.training_learning_rate)
train_losses, test_losses = trainer.train(
    afm_train, afm_test, num_epochs=50, use_wandb=False
)

# 4. Generate probabilistic predictions
history = test_traj[0, :config.context_length]          # (context_len, dim)
covariates = ts[:config.context_length, None]            # (context_len, 1)

samples = model.sample(
    jrandom.PRNGKey(42),
    history[None],          # add batch dim
    covariates[None],       # add batch dim
    nfe=20,                 # ODE solver steps
    n_steps_pred=30,        # predict 30 steps ahead
    n_samples=50,           # 50 stochastic samples
)
# samples shape: (n_samples, 1, n_steps_pred, dim)

# 5. Plot predictions with uncertainty
samples_np = np.array(samples[:, 0, :, :])  # (n_samples, pred_len, dim)
mean = samples_np.mean(axis=0)
lower = np.percentile(samples_np, 2.5, axis=0)
upper = np.percentile(samples_np, 97.5, axis=0)

ground_truth = np.array(test_traj[0])
history_np = np.array(history)
ts_np = np.array(ts)

fig, axes = plt.subplots(system.data_dim, 1, figsize=(10, 3 * system.data_dim), sharex=True)
for d in range(system.data_dim):
    ax = axes[d]
    ax.plot(ts_np[:config.context_length], history_np[:, d], "k-", label="History")
    pred_ts = ts_np[config.context_length : config.context_length + 30]
    ax.plot(pred_ts, ground_truth[config.context_length : config.context_length + 30, d],
            "k--", alpha=0.5, label="Ground truth")
    ax.plot(pred_ts, mean[:, d], "C0-", label="AFM mean")
    ax.fill_between(pred_ts, lower[:, d], upper[:, d], color="C0", alpha=0.2, label="95% CI")
    ax.set_ylabel(f"$x_{d+1}$")
    if d == 0:
        ax.legend(fontsize=8)
ax.set_xlabel("Time")
plt.tight_layout()
plt.savefig("afm_predictions.png", dpi=150)
plt.show()
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

## CLI for Paper Experiments

The package includes a CLI for reproducing the experiments from the paper:

```bash
afm list                           # list available experiments
afm run afm_vs_fm_single           # run bundled experiment config
afm run path/to/config.yaml        # run custom config
afm validate afm_vs_fm_single      # check config validity
```

See `docs/experiments.md` for details on the experiment system.

## Citation

```bibtex
@article{el2025probabilistic,
  title={Probabilistic Forecasting via Autoregressive Flow Matching},
  author={El-Gazzar, Ahmed and van Gerven, Marcel},
  journal={arXiv preprint arXiv:2503.10375},
  year={2025}
}
```

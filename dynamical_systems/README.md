# Dynamical Systems Flow Matching

A comprehensive framework for comparing **Autoregressive Flow Matching (AFM)** vs **Flow Matching (FM)** on various dynamical systems for time series forecasting.

## Quick Start

```bash
# Install dependencies (if needed)
pip install jax flax diffrax wandb matplotlib seaborn pandas pyyaml

# Run a simple comparison experiment
python -m dynamical_systems.main --config configs/experiments/afm_vs_fm_single_system.yaml

# List all available experiments
python -m dynamical_systems.main --list-experiments
```

## What This Does

This framework compares two neural approaches for forecasting dynamical systems:

- **AFM (Autoregressive Flow Matching)**: Uses autoregressive prediction with flow matching
- **FM (Flow Matching)**: Uses direct flow matching for sequence generation

**Available Dynamical Systems:**
- Lorenz attractor 
- Van der Pol oscillator
- Lotka-Volterra 
- FitzHugh-Nagumo  
- Brusselator 
- Duffing oscillator
- Harmonic oscillator

## Available Experiments

| Experiment | Purpose | Runtime | Command |
|------------|---------|---------|---------|
| **Single System** | Compare AFM vs FM on one system | ~30 min | `python -m dynamical_systems.main --config configs/experiments/afm_vs_fm_single_system.yaml` |
| **All Systems** | Compare across all 7 systems | ~3-4 hrs | `python -m dynamical_systems.main --config configs/experiments/afm_vs_fm_all_systems.yaml` |
| **History Length** | Find optimal history length | ~2 hrs | `python -m dynamical_systems.main --config configs/experiments/history_length_sensitivity.yaml` |
| **NFE Efficiency** | Speed vs accuracy tradeoff | ~1.5 hrs | `python -m dynamical_systems.main --config configs/experiments/nfe_efficiency_study.yaml` |
| **Stochastic vs Det** | Robustness to noise | ~2.5 hrs | `python -m dynamical_systems.main --config configs/experiments/stochastic_vs_deterministic.yaml` |

## Example Usage

### 1. Run Basic Comparison
```bash
# Compare AFM vs FM on Lorenz system
python -m dynamical_systems.main --config configs/experiments/afm_vs_fm_single_system.yaml
```

**Output**: Performance metrics, trajectory plots, uncertainty quantification plots

### 2. Try Different System
```bash
# Same experiment but on Van der Pol oscillator
python -m dynamical_systems.main \
  --config configs/experiments/afm_vs_fm_single_system.yaml \
  --system vanderpol
```




## Understanding Results

After running an experiment, you'll find results in:

```
results/
├── afm_vs_fm_single/          # Experiment-specific folder
│   ├── figures/               # Generated plots
│   │   ├── trajectory_predictions.pdf
│   │   ├── metrics_comparison.pdf
│   │   └── uncertainty_quantification.pdf
│   ├── models/                # Trained model weights
│   └── afm_vs_fm_single_results.json  # Numerical results
```

**Key Metrics:**
- **CRPS (Continuous Ranked Probability Score)**: Lower = better uncertainty quantification
- **NRMSE (Normalized Root Mean Square Error)**: Lower = better point predictions
- Both measured on prediction horizon and extrapolation horizon

## Customizing Experiments

### Option 1: Modify Existing Config
```bash
# Copy and edit a config file
cp configs/experiments/afm_vs_fm_single_system.yaml my_experiment.yaml
# Edit my_experiment.yaml with your settings
python -m dynamical_systems.main --config my_experiment.yaml
```

### Option 2: Command Line Overrides
```bash
python -m dynamical_systems.main \
  --config configs/experiments/afm_vs_fm_single_system.yaml \
  --system brusselator \
  --num-epochs 100 \
  --history-size 60 \
  --prediction-size 40
```



### Getting Help

```bash
# See all options
python -m dynamical_systems.main --help

# List experiment types
python -m dynamical_systems.main --list-experiments

# Validate config before running
python -m dynamical_systems.main --config my_config.yaml --validate
```

## Project Structure

```
dynamical_systems/
├── README.md                    # This file
├── main.py                      # Main entry point
├── config_loader.py            # Configuration system
├── configs/                     # Experiment configurations
│   ├── experiments/            # Ready-to-run experiments
│   └── README.md
├── experiments/                 # Experiment implementations
│   ├── experiment_runner.py    # Central experiment dispatcher
│   ├── comparison_experiment.py
│   ├── parameter_sweep_experiment.py
│   └── multi_system_experiment.py
├── models/                      # AFM and FM model implementations
├── systems/                     # Dynamical system definitions
├── training/                    # Training utilities
├── evaluation/                  # Evaluation metrics
└── utils/                       # Plotting and utilities
```

## Citation


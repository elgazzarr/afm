# Dynamical Systems Flow Matching

A comprehensive framework for comparing **Autoregressive Flow Matching (AFM)** vs **Flow Matching (FM)** on various dynamical systems for time series forecasting.

## Quick Start

```bash
# Install dependencies (if needed)
pip install jax flax diffrax wandb matplotlib seaborn pandas pyyaml

# Login to W&B for experiment tracking
wandb login

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

### Local Results

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

### W&B Results

All results are automatically uploaded to W&B with:

- **Interactive plots**: Hover, zoom, and compare trajectories
- **Metrics dashboard**: Real-time training and evaluation metrics
- **Model artifacts**: Download trained models from any experiment
- **Configuration tracking**: Full reproducibility with saved configs

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

## W&B Integration & Experiment Tracking

This framework provides comprehensive **Weights & Biases** integration for experiment tracking, visualization, and reproducibility.

### Automatic Experiment Tracking

Every experiment automatically logs to W&B:

- **Real-time training metrics**: Loss curves, gradient norms, learning rate schedules
- **Model comparisons**: Side-by-side AFM vs FM performance
- **Experiment organization**: Hierarchical grouping by experiment type and system
- **Artifact management**: Model weights, results, and configurations

### W&B Configuration

Each experiment config includes W&B settings:

```yaml
wandb:
  project: "flowtime_single_system"        # W&B project name
  group: "afm_vs_fm_single_system/lorenz"  # Experiment group
  tags: ["single_system", "lorenz", "baseline"]  # Tags for filtering
  save_artifacts: true                     # Save models and results
  log_gradients: false                     # Enable gradient tracking (optional)
  log_frequency: 10                        # Log every N steps
  notes: "Baseline AFM vs FM comparison"   # Experiment description
```

### Viewing Results

After running experiments, view results at: `https://wandb.ai/YOUR_USERNAME/PROJECT_NAME`

**W&B Dashboard Features:**
- **Training curves**: Real-time loss and metrics during training
- **Model comparisons**: AFM vs FM performance side-by-side
- **Experiment groups**: Organized by system and experiment type
- **Artifact browser**: Download trained models and results
- **Interactive plots**: Trajectory predictions and uncertainty quantification

### Hyperparameter Optimization (Optional)

Enable advanced hyperparameter sweeps with W&B:

#### 1. Create Sweep Configuration

```bash
# Use existing sweep configs
ls configs/sweeps/
# simple_optimization.yaml     - Basic hyperparameter optimization
# history_length_sweep.yaml    - Optimize history length parameter
```

#### 2. Launch Sweep

```bash
# Initialize sweep
wandb sweep configs/sweeps/simple_optimization.yaml

# Run sweep agents (copy the sweep ID from above)
wandb agent YOUR_USERNAME/PROJECT/SWEEP_ID
```

#### 3. Single Run with Sweep Mode

```bash
# Enable sweep parameter overrides for single runs
python -m dynamical_systems.main \
  --config configs/experiments/afm_vs_fm_single_system.yaml \
  --sweep
```

### Advanced W&B Features

#### Artifact Management
```yaml
wandb:
  save_artifacts: true  # Saves:
    # - trained_models/AFM_model.pkl
    # - trained_models/FM_model.pkl  
    # - experiment_results/metrics.json
    # - experiment_config/config.yaml
```

#### Enhanced Logging
```yaml
wandb:
  log_gradients: true   # Track gradient norms during training
  log_frequency: 5      # Log metrics every 5 steps
  notes: "Comparing AFM vs FM with enhanced logging"
```

#### Custom Tags and Groups
```yaml
wandb:
  group: "sensitivity_analysis/history_length"  # Hierarchical grouping
  tags: ["ablation", "history_sweep", "lorenz", "v2.0"]  # Multiple tags
```

#### Automatic Plot Upload
All experiment plots are automatically uploaded to W&B as interactive images:

- **Trajectory predictions**: Compare AFM vs FM predictions with confidence intervals
- **Training curves**: Real-time loss and metric progression  
- **Parameter sensitivity**: Visualize performance vs hyperparameter sweeps
- **Multi-system comparisons**: Side-by-side performance across dynamical systems

Plots appear in your W&B dashboard under the "Media" tab and can be:
- Downloaded as high-resolution PDFs
- Shared via direct links
- Embedded in reports
- Compared across different experiment runs



## Troubleshooting

### Common Issues

**W&B Authentication:**
```bash
# If you see "wandb: ERROR authentication failed"
wandb login
# Then paste your API key from https://wandb.ai/authorize
```

**CUDA/JAX Issues:**
```bash
# For CPU-only usage, set:
export JAX_PLATFORM_NAME=cpu
```

**Memory Issues:**
- Reduce `data_batch_size` in config
- Reduce `model_hidden_dim` for smaller models
- Use fewer `flow_n_samples` for evaluation

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
│   ├── sweeps/                 # W&B hyperparameter sweep configs
│   └── README.md
├── experiments/                 # Experiment implementations
│   ├── experiment_runner.py    # Central experiment dispatcher
│   ├── base_experiment.py      # Base class with W&B integration
│   ├── comparison_experiment.py
│   ├── parameter_sweep_experiment.py
│   └── multi_system_experiment.py
├── models/                      # AFM and FM model implementations
├── systems/                     # Dynamical system definitions
├── training/                    # Training utilities with W&B logging
├── evaluation/                  # Evaluation metrics
└── utils/                       # Plotting and utilities with W&B upload
```

## Citation


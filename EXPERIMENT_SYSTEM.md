# Experiment-Focused Configuration System

This document describes the new experiment-focused configuration system for dynamical systems research.

## Overview

The configuration system has been completely restructured to focus on **end-to-end experiments** rather than scattered parameter definitions. Each experiment type has its own YAML configuration file that defines everything needed to run a complete study.

## Key Benefits

✅ **Single Command Execution**: Run complete experiments with one command  
✅ **Clear Organization**: Each config file defines one complete experiment  
✅ **Easy Reproducibility**: Save and share experiment configurations  
✅ **Parameter Sweeps**: Built-in support for systematic parameter studies  
✅ **Multi-System Studies**: Compare performance across different dynamical systems  
✅ **Publication-Ready**: Automated plot generation and result analysis  

## Available Experiment Types

| Experiment | Purpose | Runtime | Output |
|------------|---------|---------|---------|
| `afm_vs_fm_single_system` | Compare AFM vs FM on single system | ~30 min | Performance metrics, trajectory plots |
| `afm_vs_fm_all_systems` | Compare across all dynamical systems | ~3-4 hours | System ranking, heatmaps |
| `history_length_sensitivity` | Study effect of history length | ~2 hours | Sensitivity plots, optimal length |
| `nfe_efficiency_study` | Accuracy vs computational cost | ~1.5 hours | Efficiency frontier, timing analysis |
| `stochastic_vs_deterministic` | Robustness to noise analysis | ~2.5 hours | Robustness metrics, degradation plots |
| `architecture_ablation` | Model architecture effects | ~4-5 hours | Architecture performance heatmaps |

## Quick Start

```bash
# List available experiments
python -m dynamical_systems.main --list-experiments

# Run a single system comparison
python -m dynamical_systems.main --config configs/experiments/afm_vs_fm_single_system.yaml

# Run with parameter overrides  
python -m dynamical_systems.main \
  --config configs/experiments/afm_vs_fm_single_system.yaml \
  --system vanderpol --num-epochs 100

# Validate configuration without running
python -m dynamical_systems.main --config my_config.yaml --validate
```

## Configuration Structure

Each experiment configuration follows this structure:

```yaml
experiment:
  type: "experiment_identifier"        # Experiment type
  name: "Human Readable Name"          # Display name
  description: "What this does"        # Purpose
  
  # Experiment-specific parameters
  sweep_params: {...}                  # For parameter sweeps
  systems: [...]                       # For multi-system studies
  
  plots:                               # Plots to generate
    - "performance_comparison"
    - "trajectory_examples"
  
  results:                             # Output configuration
    save_models: true
    metrics_file: "results.json"
    figures_dir: "experiment_figures"

# Standard configuration sections
system: {...}                          # Dynamical system settings
data: {...}                           # Dataset configuration
model: {...}                          # Architecture parameters  
training: {...}                       # Training settings
flow: {...}                          # Flow matching parameters
evaluation: {...}                     # Evaluation settings
experiment_settings: {...}            # Metadata
```

## Example Experiments

### 1. Single System Analysis
```bash
python -m dynamical_systems.main --config configs/experiments/afm_vs_fm_single_system.yaml
```
- Compares AFM vs FM on one dynamical system
- Generates trajectory plots, uncertainty quantification
- Perfect for detailed model analysis

### 2. Parameter Sensitivity Study  
```bash
python -m dynamical_systems.main --config configs/experiments/history_length_sensitivity.yaml
```
- Sweeps history length from 10-80 timesteps
- Shows performance vs memory requirements
- Identifies optimal history length

### 3. Multi-System Comparison
```bash
python -m dynamical_systems.main --config configs/experiments/afm_vs_fm_all_systems.yaml
```
- Tests 7 different dynamical systems
- Creates performance heatmaps and system rankings
- Identifies which method works best for which systems

## Creating Custom Experiments

1. **Copy existing config**: Start with the closest experiment type
2. **Modify parameters**: Update system, training, evaluation settings  
3. **Update experiment info**: Change name, description, output directories
4. **Run and iterate**: Test with `--validate` first

Example custom config:
```yaml
experiment:
  type: "afm_vs_fm_single_system"
  name: "Custom Brusselator Study"
  description: "Detailed analysis of AFM vs FM on Brusselator system"

system:
  name: brusselator
  stochastic: true
  num_timepoints: 150
  
training:
  num_epochs_afm: 300
  num_epochs_fm: 600
  learning_rate: 0.0005
```

## Architecture

```
configs/
├── experiments/                     # Experiment configurations
│   ├── afm_vs_fm_single_system.yaml
│   ├── history_length_sensitivity.yaml  
│   └── ...
└── README.md                       # Configuration documentation

experiments/
├── base_experiment.py              # Base experiment class
├── comparison_experiment.py        # Single system comparisons  
├── parameter_sweep_experiment.py   # Parameter sweep studies
├── multi_system_experiment.py      # Multi-system studies
├── experiment_runner.py           # Central experiment dispatcher
└── __init__.py

utils/
└── experiment_plots.py            # Experiment-specific plotting

main.py                            # Updated main entry point
```

## Migration from Old System

**Before** (scattered configuration):
```python
# Parameters spread across multiple files
config.py: ModelConfig(history_size=50, ...)
main.py: argparse(--system, --epochs, ...)  
experiment.py: ExperimentConfig(wandb_project=...)
```

**After** (unified experiment configuration):
```yaml
# Single YAML file with everything
experiment: {...}
system: {...}
data: {...}
model: {...}
training: {...}
```

## Command Line Interface

The new CLI is experiment-focused:

```bash
# Old way - scattered parameters
python -m dynamical_systems.main --system lorenz --history-size 50 --num-epochs 200 --nfe 20

# New way - experiment-focused  
python -m dynamical_systems.main --config configs/experiments/afm_vs_fm_single_system.yaml

# Still supports overrides for flexibility
python -m dynamical_systems.main --config my_experiment.yaml --system vanderpol --num-epochs 100
```

## Result Organization

Results are automatically organized by experiment:

```
results/
├── afm_vs_fm_single/
│   ├── figures/
│   │   ├── trajectory_predictions.pdf
│   │   ├── metrics_comparison.pdf  
│   │   └── uncertainty_quantification.pdf
│   ├── models/
│   │   ├── afm_model.pkl
│   │   └── fm_model.pkl
│   └── afm_vs_fm_single_results.json
├── history_sensitivity/
│   ├── figures/
│   │   ├── history_vs_accuracy.pdf
│   │   └── optimal_history_comparison.pdf
│   └── history_sensitivity_results.json
└── ...
```

## Benefits Over Old System

| Aspect | Old System | New System |
|--------|------------|------------|
| **Configuration** | Scattered across 3+ files | Single YAML per experiment |
| **Reproducibility** | Hard to recreate exact settings | Complete experiment in one file |
| **Parameter Sweeps** | Manual loops in code | Built-in sweep support |
| **Multi-System** | Not supported | Native multi-system experiments |
| **Results** | Inconsistent organization | Structured, experiment-specific |
| **Plotting** | Manual plot creation | Automated, publication-ready |
| **Documentation** | Minimal | Comprehensive with examples |

## Future Extensions

The system is designed for easy extension:

- **New Experiment Types**: Add to `EXPERIMENT_REGISTRY` in `experiment_runner.py`  
- **Custom Plots**: Extend `experiment_plots.py` with new visualization functions
- **New Systems**: Add to `systems.py`, automatically available in all experiments
- **New Metrics**: Add to evaluation, automatically included in all experiments

This experiment-focused approach makes it much easier to conduct systematic research, reproduce results, and share experimental setups with collaborators.
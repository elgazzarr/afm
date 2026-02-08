# Experiment-Focused Configuration System

This document describes the experiment-focused configuration system for dynamical systems research.

## Overview

The configuration system is structured around **end-to-end experiments** rather than scattered parameter definitions. Each experiment type has its own YAML configuration file that defines everything needed to run a complete study.

## Available Experiment Types

| Experiment | Purpose | Output |
|------------|---------|--------|
| `afm_vs_fm_single_system` | Compare AFM vs FM on single system | Performance metrics, trajectory plots |
| `afm_vs_fm_all_systems` | Compare across all dynamical systems | System ranking, heatmaps |
| `history_length_sensitivity` | Study effect of history length | Sensitivity plots, optimal length |
| `nfe_efficiency_study` | Accuracy vs computational cost | Efficiency frontier, timing analysis |
| `stochastic_vs_deterministic` | Robustness to noise analysis | Robustness metrics, degradation plots |
| `architecture_ablation` | Model architecture effects | Architecture performance heatmaps |

## Quick Start

```bash
# List available experiments
afm list

# Run a single system comparison
afm run afm_vs_fm_single

# Run with parameter overrides
afm run afm_vs_fm_single --system vanderpol --num-epochs 100

# Validate configuration without running
afm validate afm_vs_fm_single
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

## Creating Custom Experiments

1. **Copy existing config**: Start with the closest experiment type
2. **Modify parameters**: Update system, training, evaluation settings
3. **Update experiment info**: Change name, description, output directories
4. **Run and iterate**: Test with `afm validate` first

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

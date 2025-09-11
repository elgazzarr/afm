# Experiment Configurations

This directory contains experiment-focused configuration files for comprehensive studies of Autoregressive Flow Matching (AFM) vs Flow Matching (FM) on dynamical systems.

## Quick Start

Run any experiment with:
```bash
python -m dynamical_systems.main --config configs/experiments/afm_vs_fm_single_system.yaml
```

## Available Experiments

### 🔬 Single System Analysis

#### `afm_vs_fm_single_system.yaml`
- **Purpose**: Compare AFM vs FM on a single dynamical system
- **Outputs**: Performance metrics, trajectory plots, uncertainty quantification
- **Use Case**: Detailed analysis of one system, model validation
- **Runtime**: ~30 minutes

### 🌐 Multi-System Analysis  

#### `afm_vs_fm_all_systems.yaml`
- **Purpose**: Compare AFM vs FM across all available dynamical systems
- **Systems**: Lorenz, Van der Pol, Lotka-Volterra, FitzHugh-Nagumo, Brusselator, Duffing, Harmonic Oscillator
- **Outputs**: Performance heatmap, system rankings, statistical significance tests
- **Use Case**: Comprehensive evaluation, identifying which method works best for which systems
- **Runtime**: ~3-4 hours

### 📏 Parameter Sensitivity Studies

#### `history_length_sensitivity.yaml`
- **Purpose**: Study how history length affects prediction performance
- **Sweep Range**: 10-80 timesteps
- **Outputs**: Performance vs history length plots, optimal history identification
- **Use Case**: Optimize history length, understand memory requirements
- **Runtime**: ~2 hours

#### `nfe_efficiency_study.yaml`
- **Purpose**: Analyze accuracy vs computational cost tradeoff
- **Sweep Range**: 5-100 function evaluations
- **Outputs**: Efficiency frontier, timing analysis, optimal NFE recommendation
- **Use Case**: Balance accuracy and speed, deployment optimization
- **Runtime**: ~1.5 hours

### 🎯 System Properties Analysis

#### `stochastic_vs_deterministic.yaml`
- **Purpose**: Compare performance on stochastic vs deterministic systems
- **Analysis**: Noise robustness, uncertainty quantification quality
- **Outputs**: Robustness analysis, prediction interval coverage
- **Use Case**: Understand model behavior under uncertainty
- **Runtime**: ~2.5 hours

### 🏗️ Architecture Studies

#### `architecture_ablation.yaml`
- **Purpose**: Study effect of model architecture choices
- **Parameters**: Hidden dimensions, flow blocks, RNN layers
- **Outputs**: Architecture performance heatmap, parameter sensitivity analysis
- **Use Case**: Optimize model architecture, understand architectural choices
- **Runtime**: ~4-5 hours (comprehensive sweep)

## Configuration Structure

Each experiment configuration contains:

```yaml
experiment:
  type: "experiment_type"          # Type identifier
  name: "Human-readable name"      # Display name
  description: "What this does"    # Purpose description
  
  # Experiment-specific parameters
  sweep_params: {...}              # Parameters to sweep over
  systems: [...]                   # Systems to test (if multi-system)
  
  # Output configuration
  plots: [...]                     # Types of plots to generate
  results: {...}                   # Result saving options

# Standard configuration sections
system: {...}                      # Dynamical system settings
data: {...}                        # Dataset configuration  
model: {...}                       # Model architecture
training: {...}                    # Training parameters
flow: {...}                        # Flow matching parameters
evaluation: {...}                  # Evaluation settings
experiment_settings: {...}         # Experiment metadata
```

## Customizing Experiments

### Modify Existing Experiments
1. Copy any existing config file
2. Modify parameters as needed  
3. Update `experiment.name` and output directories
4. Run with your modified config

### Create New Experiments
1. Start with the closest existing experiment type
2. Modify the `experiment.type` to a new identifier  
3. Add any new sweep parameters or analysis options
4. Implement corresponding experiment logic (if needed)

### Override Parameters
You can override any parameter from the command line:
```bash
python -m dynamical_systems.main \
  --config configs/experiments/afm_vs_fm_single_system.yaml \
  --system vanderpol \
  --num_epochs 100
```

## Output Organization

Results are saved in structured directories:
```
results/
├── afm_vs_fm_single/
│   ├── figures/
│   ├── models/ 
│   └── afm_vs_fm_single_results.json
├── history_sensitivity/
│   ├── figures/
│   └── history_sensitivity_results.json
└── ...
```

## Computational Requirements

| Experiment | Estimated Runtime | Memory | Storage |
|------------|------------------|---------|---------|
| Single System | 30 min | 4GB | 100MB |
| All Systems | 3-4 hours | 8GB | 500MB |
| History Sensitivity | 2 hours | 4GB | 200MB |
| NFE Efficiency | 1.5 hours | 4GB | 150MB |
| Stochastic vs Det | 2.5 hours | 6GB | 300MB |
| Architecture Ablation | 4-5 hours | 8GB | 400MB |

*Estimates based on default configurations with moderate hardware*

## Best Practices

1. **Start Small**: Use single system experiments for initial validation
2. **Resource Planning**: Check computational requirements before long experiments
3. **Version Control**: Save experiment configs in version control
4. **Reproducibility**: Always set random seeds consistently
5. **Documentation**: Update experiment names/descriptions for clarity

## Troubleshooting

- **Out of Memory**: Reduce `batch_size`, `n_train_test_samples`, or `n_eval_samples`
- **Slow Training**: Reduce `num_epochs` or increase `learning_rate`
- **Poor Results**: Check system parameters, try different architectures
- **File Not Found**: Ensure config path is correct relative to project root
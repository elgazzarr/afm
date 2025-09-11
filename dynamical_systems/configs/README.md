# Configuration Files

This directory contains example configuration files for different experimental scenarios.

## Usage

Use any configuration file with the main script:

```bash
python -m dynamical_systems.main --config configs/quick_test.yaml
```

You can also override specific parameters:

```bash
python -m dynamical_systems.main --config configs/high_quality.yaml --system vanderpol --num_epochs 100
```

## Available Configurations

### `quick_test.yaml`
- **Purpose**: Fast testing and debugging
- **System**: Lorenz (stochastic)
- **Dataset**: Small (64 samples)
- **Model**: Compact architecture
- **Training**: 10 epochs only
- **Use case**: Quick validation, development testing

### `high_quality.yaml`
- **Purpose**: Best performance results
- **System**: Lorenz (stochastic) 
- **Dataset**: Large (2048 samples)
- **Model**: Large architecture with deep networks
- **Training**: 500 epochs
- **Use case**: Final experiments, paper results

### `vanderpol_study.yaml` 
- **Purpose**: Van der Pol oscillator analysis
- **System**: Van der Pol (deterministic)
- **Dataset**: Medium (1024 samples)
- **Model**: Balanced architecture
- **Training**: 300 epochs
- **Use case**: System-specific studies

## Configuration Structure

Each YAML file contains the following sections:

- **system**: Dynamical system parameters (name, stochasticity, time range)
- **data**: Dataset parameters (sizes, batch size, number of samples)
- **model**: Neural network architecture (dimensions, layers, blocks)
- **training**: Optimization parameters (learning rate, epochs)
- **flow**: Flow matching parameters (sampling steps, NFE)
- **experiment**: Experiment metadata (name, logging, output directories)

## Creating Custom Configurations

Copy any existing config file and modify the parameters as needed. The system will validate all parameters automatically.
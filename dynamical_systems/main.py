"""Simplified main entry point for AFM vs FM experiments."""

import argparse
import warnings
import jax.random as jrandom
from pathlib import Path

from .experiments import AFMvsFMComparison
from .config_loader import Config

# Suppress JAX warnings
warnings.filterwarnings("ignore", message="Unable to parse backend config for custom call")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare Autoregressive Flow Matching (AFM) vs Flow Matching (FM) for forecasting dynamical systems'
    )

    # Primary argument: config file
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )
    
    # Backward compatibility arguments (override config file if provided)
    parser.add_argument(
        '--system', type=str,
        choices=['lorenz', 'lotka_volterra', 'vanderpol', 'harmonic_oscillator', 
                'brusselator', 'fitzhugh_nagumo', 'duffing'],
        help='Name of the dynamical system (overrides config)'
    )
    parser.add_argument(
        '--stochastic', action='store_true',
        help='Whether the system is stochastic (overrides config)'
    )
    parser.add_argument('--t1', type=float, help='Final time (overrides config)')
    parser.add_argument(
        '--num_timepoints', type=int,
        help='Number of timepoints (overrides config)'
    )
    parser.add_argument(
        '--history_size', type=int,
        help='History size (overrides config)'
    )
    parser.add_argument(
        '--prediction_size', type=int,
        help='Prediction size (overrides config)'
    )
    parser.add_argument(
        '--num_epochs', type=int,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--nfe', type=int,
        help='Number of function evaluations for sampling (overrides config)'
    )
    parser.add_argument(
        '--wandb_project', type=str,
        help='W&B project name (overrides config)'
    )
    parser.add_argument(
        '--seed', type=int,
        help='Random seed for reproducibility (overrides config)'
    )
    
    return parser.parse_args()


def main():
    """Run AFM vs FM comparison experiment."""
    args = parse_args()
    
    # Load configuration from YAML file
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Make path relative to this file's directory
        config_path = Path(__file__).parent / config_path
    
    try:
        config = Config.from_yaml(str(config_path))
        print(f"Loaded configuration from: {config_path}")
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        config = Config()
    
    # Override config with command-line arguments if provided
    if args.system:
        config.system_name = args.system
    if args.stochastic:
        config.system_stochastic = args.stochastic
    if args.t1:
        config.system_t1 = args.t1
    if args.num_timepoints:
        config.system_num_timepoints = args.num_timepoints
    if args.history_size:
        config.data_history_size = args.history_size
    if args.prediction_size:
        config.data_prediction_size = args.prediction_size
    if args.num_epochs:
        config.training_num_epochs = args.num_epochs
    if args.nfe:
        config.flow_nfe = args.nfe
        config.flow_n_steps_flow = args.nfe
    if args.wandb_project:
        config.experiment_wandb_project = args.wandb_project
    if args.seed:
        config.experiment_seed = args.seed
    
    # Recompute derived fields after potential updates
    config._compute_derived_fields()
    
    print(f"Starting AFM vs FM comparison on {config.system_name} system")
    print(f"Configuration: {config.training_num_epochs} epochs, {config.flow_nfe} NFE")
    print("=" * 60)
    print(config.summary())
    
    # Create and run experiment
    experiment = AFMvsFMComparison(config, seed=config.experiment_seed)
    results = experiment.run()
    
    # Print final comparison
    if len(results) >= 2:
        print("\nFINAL RESULTS SUMMARY:")
        print("=" * 60)
        
        afm_results = results.get('AFM', {})
        fm_results = results.get('FM', {})
        
        metrics = ['crps_prediction', 'nrmse_prediction', 'crps_extrapolation', 'nrmse_extrapolation']
        
        for metric in metrics:
            if metric in afm_results and metric in fm_results:
                afm_val = afm_results[metric]
                fm_val = fm_results[metric]
                better = "AFM" if afm_val < fm_val else "FM"
                print(f"{metric:<20}: AFM={afm_val:.4f}, FM={fm_val:.4f} (Better: {better})")
    
    print("\nExperiment completed successfully!")


if __name__ == '__main__':
    main()
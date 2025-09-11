"""Simplified main entry point for AFM vs FM experiments."""

import argparse
import warnings
import jax.random as jrandom

from .experiments import AFMvsFMComparison, ExperimentConfig

# Suppress JAX warnings
warnings.filterwarnings("ignore", message="Unable to parse backend config for custom call")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare Autoregressive Flow Matching (AFM) vs Flow Matching (FM) for forecasting dynamical systems'
    )

    parser.add_argument(
        '--system', type=str, default='lorenz',
        choices=['lorenz', 'lotka_volterra', 'vanderpol', 'harmonic_oscillator', 
                'brusselator', 'fitzhugh_nagumo', 'duffing'],
        help='Name of the dynamical system'
    )
    parser.add_argument(
        '--stochastic', action='store_true', default=True,
        help='Whether the system is stochastic'
    )
    parser.add_argument('--t1', type=float, default=20.0, help='Final time')
    parser.add_argument(
        '--num_timepoints', type=int, default=100,
        help='Number of timepoints'
    )
    parser.add_argument(
        '--history_size', type=int, default=50,
        help='History size'
    )
    parser.add_argument(
        '--prediction_size', type=int, default=25,
        help='Prediction size'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=200,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--nfe', type=int, default=20,
        help='Number of function evaluations for sampling'
    )
    parser.add_argument(
        '--wandb_project', type=str, default='flowtime_ds',
        help='W&B project name'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def main():
    """Run AFM vs FM comparison experiment."""
    args = parse_args()
    
    print(f"Starting AFM vs FM comparison on {args.system} system")
    print(f"Configuration: {args.num_epochs} epochs, {args.nfe} NFE")
    print("=" * 60)
    
    # Create experiment configuration
    config = ExperimentConfig.from_args(args)
    
    # Create and run experiment
    experiment = AFMvsFMComparison(config, seed=args.seed)
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
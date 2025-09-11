"""Main entry point for dynamical systems experiments with experiment-focused configuration."""

import argparse
import warnings
from pathlib import Path

from .experiments.experiment_runner import ExperimentRunner, run_experiment_from_config

# Suppress JAX warnings
warnings.filterwarnings("ignore", message="Unable to parse backend config for custom call")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run dynamical systems experiments with AFM vs FM comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single system comparison
  python -m dynamical_systems.main --config configs/experiments/afm_vs_fm_single_system.yaml
  
  # Run history length sensitivity study
  python -m dynamical_systems.main --config configs/experiments/history_length_sensitivity.yaml
  
  # Run all systems comparison
  python -m dynamical_systems.main --config configs/experiments/afm_vs_fm_all_systems.yaml
  
  # List available experiment types
  python -m dynamical_systems.main --list-experiments
  
  # Validate configuration without running
  python -m dynamical_systems.main --config my_config.yaml --validate

Available experiment types:
  afm_vs_fm_single_system    - Compare AFM vs FM on single system
  afm_vs_fm_all_systems      - Compare AFM vs FM across all systems  
  history_length_sensitivity - Study effect of history length
  nfe_efficiency_study       - Study NFE vs accuracy tradeoff
  stochastic_vs_deterministic - Compare stochastic vs deterministic
  architecture_ablation      - Study model architecture effects
        """
    )

    # Primary argument: experiment config file
    parser.add_argument(
        '--config', type=str, default='configs/experiments/afm_vs_fm_single_system.yaml',
        help='Path to experiment configuration file (default: configs/experiments/afm_vs_fm_single_system.yaml)'
    )
    
    # Experiment management
    parser.add_argument(
        '--list-experiments', action='store_true',
        help='List all available experiment types and exit'
    )
    
    parser.add_argument(
        '--validate', action='store_true',
        help='Validate configuration file and exit (do not run experiment)'
    )
    
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed for reproducibility (default: 0)'
    )
    
    # Backward compatibility: individual parameter overrides
    # (Note: These will override values in the config file if provided)
    parser.add_argument(
        '--system', type=str,
        choices=['lorenz', 'lotka_volterra', 'vanderpol', 'harmonic_oscillator', 
                'brusselator', 'fitzhugh_nagumo', 'duffing'],
        help='Override system name from config (for single-system experiments only)'
    )
    parser.add_argument(
        '--stochastic', action='store_true',
        help='Override stochastic setting from config'
    )
    parser.add_argument(
        '--num-epochs', type=int,
        help='Override number of training epochs from config'
    )
    parser.add_argument(
        '--nfe', type=int,
        help='Override NFE (number of function evaluations) from config'
    )
    parser.add_argument(
        '--history-size', type=int,
        help='Override history size from config'
    )
    parser.add_argument(
        '--prediction-size', type=int,
        help='Override prediction size from config'
    )
    
    return parser.parse_args()


def apply_arg_overrides(runner: ExperimentRunner, args) -> None:
    """Apply command line argument overrides to the experiment configuration.
    
    Args:
        runner: ExperimentRunner instance
        args: Parsed command line arguments
    """
    config = runner.config
    
    # System overrides (only for single-system experiments)
    if args.system and runner.experiment_type == 'afm_vs_fm_single_system':
        config.system_name = args.system
        print(f"Override: system = {args.system}")
    
    if args.stochastic and runner.experiment_type == 'afm_vs_fm_single_system':
        config.system_stochastic = args.stochastic
        print(f"Override: stochastic = {args.stochastic}")
    
    # Training overrides
    if args.num_epochs:
        config.training_num_epochs = args.num_epochs
        print(f"Override: num_epochs = {args.num_epochs}")
    
    # Flow overrides
    if args.nfe:
        config.flow_nfe = args.nfe
        config.flow_n_steps_flow = args.nfe
        print(f"Override: nfe = {args.nfe}")
    
    # Data overrides
    if args.history_size:
        config.data_history_size = args.history_size
        print(f"Override: history_size = {args.history_size}")
    
    if args.prediction_size:
        config.data_prediction_size = args.prediction_size  
        print(f"Override: prediction_size = {args.prediction_size}")
    
    # Recompute derived fields after overrides
    if any([args.system, args.stochastic, args.history_size, args.prediction_size]):
        config._compute_derived_fields()


def main():
    """Main function."""
    args = parse_args()
    
    # Handle special modes
    if args.list_experiments:
        print("Available Experiment Types:")
        print("=" * 50)
        experiments = ExperimentRunner.list_available_experiments()
        for exp_type, description in experiments.items():
            print(f"  {exp_type:<25} - {description}")
        print("\nSee configs/experiments/ for example configurations.")
        return
    
    # Validate config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Make path relative to this file's directory
        config_path = Path(__file__).parent / config_path
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Available example configs:")
        examples_dir = Path(__file__).parent / "configs" / "experiments"
        if examples_dir.exists():
            for config_file in examples_dir.glob("*.yaml"):
                print(f"  {config_file}")
        return
    
    try:
        # Create experiment runner
        runner = ExperimentRunner(str(config_path), args.seed)
        
        # Apply command line overrides
        apply_arg_overrides(runner, args)
        
        # Validation mode
        if args.validate:
            print(f"Validating configuration: {config_path}")
            is_valid = runner.validate_config()
            if is_valid:
                print("✓ Configuration is valid")
                print(f"  Experiment Type: {runner.experiment_type}")
                print(f"  Experiment Name: {runner.config.experiment_name}")
                print(f"  Description: {getattr(runner.config, 'experiment_description', 'N/A')}")
            else:
                print("✗ Configuration is invalid")
            return
        
        # Run experiment
        print(f"Loading experiment from: {config_path}")
        results = runner.run()
        
        # Print summary
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Experiment: {runner.config.experiment_name}")
        print(f"Type: {runner.experiment_type}")
        print(f"Results saved to: {getattr(runner.config, 'experiment_results_dir', 'results/')}")
        
        if hasattr(runner.config, 'results_metrics_file'):
            print(f"Metrics saved to: {runner.config.results_metrics_file}")
        
        print("\nResult summary:")
        if isinstance(results, dict):
            if runner.experiment_type in ['history_length_sensitivity', 'nfe_efficiency_study']:
                # For sweep experiments, show sweep summary
                sweep_points = len([k for k in results.keys() if not k.startswith('config')])
                print(f"  Completed {sweep_points} parameter sweep points")
                
            elif runner.experiment_type in ['afm_vs_fm_all_systems', 'stochastic_vs_deterministic']:
                # For multi-system experiments, show system summary
                system_count = len([k for k in results.keys() if not k.startswith('config')])
                print(f"  Tested {system_count} system configurations")
                
            else:
                # For single experiments, show model comparison
                if 'AFM' in results and 'FM' in results:
                    afm_crps = results['AFM'].get('crps_prediction', 'N/A')
                    fm_crps = results['FM'].get('crps_prediction', 'N/A')
                    print(f"  AFM CRPS: {afm_crps}")
                    print(f"  FM CRPS:  {fm_crps}")
                    if afm_crps != 'N/A' and fm_crps != 'N/A':
                        winner = "AFM" if afm_crps < fm_crps else "FM"
                        print(f"  Winner: {winner}")
        
        print("="*80)
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
"""Experiment runner that dispatches to appropriate experiment types based on configuration."""

from typing import Dict, Any, Type
import yaml
from pathlib import Path

from ..config import Config
from .base import BaseExperiment
from .comparison import AFMvsFMComparison
from .parameter_sweep import HistoryLengthSweepExperiment, NFEEfficiencyExperiment
from .multi_system import AFMvsFMAllSystemsExperiment, StochasticVsDeterministicExperiment


class ExperimentRunner:
    """Central experiment runner that dispatches to different experiment types."""

    # Registry of available experiments
    EXPERIMENT_REGISTRY: Dict[str, Type[BaseExperiment]] = {
        'afm_vs_fm_single_system': AFMvsFMComparison,
        'afm_vs_fm_all_systems': AFMvsFMAllSystemsExperiment,
        'history_length_sensitivity': HistoryLengthSweepExperiment,
        'nfe_efficiency_study': NFEEfficiencyExperiment,
        'stochastic_vs_deterministic': StochasticVsDeterministicExperiment,
    }

    def __init__(self, config_path: str, seed: int = 0):
        """Initialize experiment runner.

        Args:
            config_path: Path to experiment configuration file
            seed: Random seed for reproducibility
        """
        self.config_path = Path(config_path)
        self.seed = seed

        # Load configuration
        self.config = self.load_experiment_config()

        # Determine experiment type
        self.experiment_type = self.get_experiment_type()

        print(f"Loaded experiment: {self.experiment_type}")
        print(f"Experiment name: {getattr(self.config, 'experiment_name', 'Unknown')}")

    def load_experiment_config(self) -> Config:
        """Load and parse experiment configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        # Extract experiment info
        experiment_info = yaml_data.get('experiment', {})

        # Create base config from YAML
        config = Config.from_yaml(str(self.config_path))

        # Add experiment-specific attributes
        config.experiment_type = experiment_info.get('type', 'unknown')
        config.experiment_name = experiment_info.get('name', 'Unknown Experiment')
        config.experiment_description = experiment_info.get('description', '')

        # Add sweep parameters if present
        if 'sweep_params' in experiment_info:
            config.sweep_params = experiment_info['sweep_params']

        # Add systems list if present
        if 'systems' in experiment_info:
            config.systems = experiment_info['systems']

        # Add plots configuration
        if 'plots' in experiment_info:
            config.plots = experiment_info['plots']

        # Add results configuration
        if 'results' in experiment_info:
            results_config = experiment_info['results']
            for key, value in results_config.items():
                setattr(config, f'results_{key}', value)

        # Override results directory with experiment-specific directory
        if hasattr(config, 'results_figures_dir'):
            config.experiment_results_dir = config.results_figures_dir

        return config

    def get_experiment_type(self) -> str:
        """Get experiment type from configuration."""
        return getattr(self.config, 'experiment_type', 'afm_vs_fm_single_system')

    def create_experiment(self) -> BaseExperiment:
        """Create appropriate experiment instance based on type."""
        if self.experiment_type not in self.EXPERIMENT_REGISTRY:
            available_types = list(self.EXPERIMENT_REGISTRY.keys())
            raise ValueError(
                f"Unknown experiment type: {self.experiment_type}\n"
                f"Available types: {available_types}"
            )

        experiment_class = self.EXPERIMENT_REGISTRY[self.experiment_type]
        return experiment_class(self.config, self.seed)

    def run(self) -> Dict[str, Any]:
        """Run the experiment."""
        print(f"Starting experiment: {self.config.experiment_name}")
        print(f"Type: {self.experiment_type}")
        print(f"Description: {getattr(self.config, 'experiment_description', 'No description')}")
        print("=" * 60)

        # Create and run experiment
        experiment = self.create_experiment()
        results = experiment.run()

        print("\n" + "=" * 60)
        print(f"Experiment '{self.config.experiment_name}' completed successfully!")

        return results

    @classmethod
    def register_experiment(cls, experiment_type: str, experiment_class: Type[BaseExperiment]) -> None:
        """Register a new experiment type.

        Args:
            experiment_type: String identifier for experiment type
            experiment_class: Experiment class to register
        """
        cls.EXPERIMENT_REGISTRY[experiment_type] = experiment_class
        print(f"Registered experiment type: {experiment_type}")

    @classmethod
    def list_available_experiments(cls) -> Dict[str, str]:
        """List all available experiment types and their descriptions."""
        experiments = {}
        for exp_type, exp_class in cls.EXPERIMENT_REGISTRY.items():
            doc = exp_class.__doc__ or "No description available"
            experiments[exp_type] = doc.split('\n')[0]  # First line only
        return experiments

    def validate_config(self) -> bool:
        """Validate experiment configuration."""
        try:
            # Check required attributes
            required_attrs = ['experiment_type', 'experiment_name']
            for attr in required_attrs:
                if not hasattr(self.config, attr):
                    print(f"Missing required attribute: {attr}")
                    return False

            # Type-specific validation
            if self.experiment_type == 'history_length_sensitivity':
                if not hasattr(self.config, 'sweep_params'):
                    print("history_length_sensitivity requires sweep_params")
                    return False
                sweep_params = getattr(self.config, 'sweep_params', {})
                if 'history_size' not in sweep_params:
                    print("history_length_sensitivity requires history_size in sweep_params")
                    return False

            elif self.experiment_type == 'afm_vs_fm_all_systems':
                if not hasattr(self.config, 'systems'):
                    print("afm_vs_fm_all_systems requires systems list")
                    return False

            elif self.experiment_type == 'nfe_efficiency_study':
                if not hasattr(self.config, 'sweep_params'):
                    print("nfe_efficiency_study requires sweep_params")
                    return False
                sweep_params = getattr(self.config, 'sweep_params', {})
                if 'nfe_values' not in sweep_params:
                    print("nfe_efficiency_study requires nfe_values in sweep_params")
                    return False

            return True

        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False


def run_experiment_from_config(config_path: str, seed: int = 0) -> Dict[str, Any]:
    """Convenience function to run an experiment from a configuration file.

    Args:
        config_path: Path to experiment configuration file
        seed: Random seed for reproducibility

    Returns:
        Experiment results dictionary
    """
    runner = ExperimentRunner(config_path, seed)

    # Validate configuration
    if not runner.validate_config():
        raise ValueError("Configuration validation failed")

    return runner.run()


def main():
    """Main function for running experiments from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Run dynamical systems experiments')
    parser.add_argument('config', help='Path to experiment configuration file')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--list', action='store_true', help='List available experiment types')
    parser.add_argument('--validate', action='store_true', help='Only validate configuration')

    args = parser.parse_args()

    if args.list:
        print("Available experiment types:")
        experiments = ExperimentRunner.list_available_experiments()
        for exp_type, description in experiments.items():
            print(f"  {exp_type}: {description}")
        return

    runner = ExperimentRunner(args.config, args.seed)

    if args.validate:
        is_valid = runner.validate_config()
        print(f"Configuration is {'valid' if is_valid else 'invalid'}")
        return

    # Run experiment
    results = runner.run()
    print(f"Experiment completed. Results keys: {list(results.keys())}")


if __name__ == '__main__':
    main()

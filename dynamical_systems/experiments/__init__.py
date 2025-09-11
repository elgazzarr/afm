"""Experiment management module for AFM vs FM comparisons."""

from .base_experiment import BaseExperiment
from .comparison_experiment import AFMvsFMComparison
from .parameter_sweep_experiment import HistoryLengthSweepExperiment, NFEEfficiencyExperiment
from .multi_system_experiment import AFMvsFMAllSystemsExperiment, StochasticVsDeterministicExperiment
from .experiment_runner import ExperimentRunner, run_experiment_from_config
from ..config_loader import Config

# Backward compatibility
ExperimentConfig = Config

__all__ = [
    'BaseExperiment',
    'ExperimentConfig', 
    'Config',
    'AFMvsFMComparison',
    'HistoryLengthSweepExperiment',
    'NFEEfficiencyExperiment', 
    'AFMvsFMAllSystemsExperiment',
    'StochasticVsDeterministicExperiment',
    'ExperimentRunner',
    'run_experiment_from_config'
]
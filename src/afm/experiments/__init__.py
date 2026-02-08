"""Experiment management module for AFM vs FM comparisons."""

from .base import BaseExperiment
from .comparison import AFMvsFMComparison
from .parameter_sweep import HistoryLengthSweepExperiment, NFEEfficiencyExperiment
from .multi_system import AFMvsFMAllSystemsExperiment, StochasticVsDeterministicExperiment
from .runner import ExperimentRunner, run_experiment_from_config
from ..config import Config

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

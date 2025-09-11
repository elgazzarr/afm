"""Experiment management module for AFM vs FM comparisons."""

from .base_experiment import BaseExperiment
from .comparison_experiment import AFMvsFMComparison
from ..config_loader import Config

# Backward compatibility
ExperimentConfig = Config

__all__ = ['BaseExperiment', 'ExperimentConfig', 'AFMvsFMComparison', 'Config']
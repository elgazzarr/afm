"""Unified configuration loader for dynamical systems experiments."""

import yaml
from dataclasses import dataclass, field
from typing import Optional, Literal, List
from pathlib import Path
import jax.numpy as jnp


@dataclass
class Config:
    """Unified configuration class loaded from YAML."""
    
    # System parameters
    system_name: str = 'lorenz'
    system_stochastic: bool = True
    system_num_timepoints: int = 100
    system_t0: float = 0.0
    system_t1: float = 20.0
    
    # Data parameters
    data_history_size: int = 50
    data_prediction_size: int = 30
    data_n_train_test_samples: int = 512
    data_batch_size: int = 128
    
    # Model parameters
    model_covariates_dim: int = 1
    model_hidden_dim: int = 64
    model_time_dim: int = 16
    model_hidden_blocks: int = 3
    model_rnn_hidden: int = 64
    model_n_rnn_layers: int = 3
    
    # Training parameters
    training_learning_rate: float = 0.001
    training_num_epochs: int = 200
    
    # Flow parameters
    flow_n_samples: int = 50
    flow_n_steps_flow: int = 50
    flow_nfe: int = 20
    
    # Experiment parameters
    experiment_name: str = "afm_vs_fm_comparison"
    experiment_wandb_project: str = "flowtime_ds"
    experiment_save_figures: bool = True
    experiment_results_dir: str = "results"
    experiment_seed: int = 0
    experiment_num_eval_samples: int = 50
    
    # Derived fields (computed after initialization)
    extrapolation_size: int = field(init=False)
    ts: jnp.ndarray = field(init=False)
    dt: float = field(init=False)
    
    def __post_init__(self):
        """Validate configuration and compute derived fields."""
        self._validate_config()
        self._compute_derived_fields()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Time validation
        if self.system_t1 <= self.system_t0:
            raise ValueError(f"t1 ({self.system_t1}) must be greater than t0 ({self.system_t0})")
        
        # Data size validation
        if self.system_num_timepoints < self.data_history_size + self.data_prediction_size:
            raise ValueError(
                f"num_timepoints ({self.system_num_timepoints}) must be >= "
                f"history_size + prediction_size ({self.data_history_size + self.data_prediction_size})"
            )
        
        if self.data_history_size <= 0:
            raise ValueError(f"history_size ({self.data_history_size}) must be positive")
            
        if self.data_prediction_size <= 0:
            raise ValueError(f"prediction_size ({self.data_prediction_size}) must be positive")
        
        # Architecture validation
        if self.model_hidden_dim <= 0:
            raise ValueError(f"hidden_dim ({self.model_hidden_dim}) must be positive")
            
        if self.model_time_dim <= 0:
            raise ValueError(f"time_dim ({self.model_time_dim}) must be positive")
            
        if self.model_hidden_blocks <= 0:
            raise ValueError(f"hidden_blocks ({self.model_hidden_blocks}) must be positive")
            
        if self.model_n_rnn_layers <= 0:
            raise ValueError(f"n_rnn_layers ({self.model_n_rnn_layers}) must be positive")
        
        # Training validation
        if self.training_learning_rate <= 0:
            raise ValueError(f"learning_rate ({self.training_learning_rate}) must be positive")
            
        if self.data_batch_size <= 0:
            raise ValueError(f"batch_size ({self.data_batch_size}) must be positive")
            
        if self.data_n_train_test_samples <= 0:
            raise ValueError(f"n_train_test_samples ({self.data_n_train_test_samples}) must be positive")
        
        # System validation
        valid_systems = ['lorenz', 'lotka_volterra', 'vanderpol', 'harmonic_oscillator',
                        'brusselator', 'fitzhugh_nagumo', 'duffing']
        if self.system_name not in valid_systems:
            raise ValueError(f"Invalid system '{self.system_name}'. Valid options: {valid_systems}")
    
    def _compute_derived_fields(self) -> None:
        """Compute derived configuration fields."""
        self.extrapolation_size = self.system_num_timepoints - (self.data_history_size + self.data_prediction_size)
        self.ts = jnp.linspace(self.system_t0, self.system_t1, self.system_num_timepoints)
        self.dt = (self.ts[1] - self.ts[0]) / 5
    
    def get_system_choices(self) -> List[str]:
        """Get list of available dynamical systems."""
        return ['lorenz', 'lotka_volterra', 'vanderpol', 'harmonic_oscillator',
                'brusselator', 'fitzhugh_nagumo', 'duffing']
    
    def summary(self) -> str:
        """Get a summary of the configuration."""
        return f"""
Config Summary:
===============
System: {self.system_name} (stochastic: {self.system_stochastic})
Time: {self.system_t0} to {self.system_t1} ({self.system_num_timepoints} points)
Data split: {self.data_history_size} history + {self.data_prediction_size} pred + {self.extrapolation_size} extrap
Architecture: {self.model_hidden_dim}D hidden, {self.model_n_rnn_layers} RNN layers, {self.model_hidden_blocks} flow blocks
Training: lr={self.training_learning_rate}, batch_size={self.data_batch_size}, {self.data_n_train_test_samples} samples
Experiment: {self.experiment_name} (seed: {self.experiment_seed})
"""

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config instance with loaded parameters
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Flatten nested YAML structure to match dataclass fields
        flattened = {}
        
        # System section
        if 'system' in yaml_data:
            system = yaml_data['system']
            flattened['system_name'] = system.get('name', 'lorenz')
            flattened['system_stochastic'] = system.get('stochastic', True)
            flattened['system_num_timepoints'] = system.get('num_timepoints', 100)
            flattened['system_t0'] = system.get('t0', 0.0)
            flattened['system_t1'] = system.get('t1', 20.0)
        
        # Data section
        if 'data' in yaml_data:
            data = yaml_data['data']
            flattened['data_history_size'] = data.get('history_size', 50)
            flattened['data_prediction_size'] = data.get('prediction_size', 30)
            flattened['data_n_train_test_samples'] = data.get('n_train_test_samples', 512)
            flattened['data_batch_size'] = data.get('batch_size', 128)
        
        # Model section
        if 'model' in yaml_data:
            model = yaml_data['model']
            flattened['model_covariates_dim'] = model.get('covariates_dim', 1)
            flattened['model_hidden_dim'] = model.get('hidden_dim', 64)
            flattened['model_time_dim'] = model.get('time_dim', 16)
            flattened['model_hidden_blocks'] = model.get('hidden_blocks', 3)
            flattened['model_rnn_hidden'] = model.get('rnn_hidden', 64)
            flattened['model_n_rnn_layers'] = model.get('n_rnn_layers', 3)
        
        # Training section
        if 'training' in yaml_data:
            training = yaml_data['training']
            flattened['training_learning_rate'] = training.get('learning_rate', 0.001)
            flattened['training_num_epochs'] = training.get('num_epochs', 200)
        
        # Flow section
        if 'flow' in yaml_data:
            flow = yaml_data['flow']
            flattened['flow_n_samples'] = flow.get('n_samples', 50)
            flattened['flow_n_steps_flow'] = flow.get('n_steps_flow', 50)
            flattened['flow_nfe'] = flow.get('nfe', 20)
        
        # Experiment section
        if 'experiment' in yaml_data:
            experiment = yaml_data['experiment']
            flattened['experiment_name'] = experiment.get('name', 'afm_vs_fm_comparison')
            flattened['experiment_wandb_project'] = experiment.get('wandb_project', 'flowtime_ds')
            flattened['experiment_save_figures'] = experiment.get('save_figures', True)
            flattened['experiment_results_dir'] = experiment.get('results_dir', 'results')
            flattened['experiment_seed'] = experiment.get('seed', 0)
            flattened['experiment_num_eval_samples'] = experiment.get('num_eval_samples', 50)
        
        return cls(**flattened)

    @classmethod
    def from_args(cls, args) -> 'Config':
        """Create Config from command line arguments (for backward compatibility).
        
        Args:
            args: Argument namespace from argparse
            
        Returns:
            Config instance
        """
        return cls(
            system_name=getattr(args, 'system', 'lorenz'),
            system_stochastic=getattr(args, 'stochastic', True),
            system_num_timepoints=getattr(args, 'num_timepoints', 100),
            system_t0=0.0,
            system_t1=getattr(args, 't1', 20.0),
            data_history_size=getattr(args, 'history_size', 50),
            data_prediction_size=getattr(args, 'prediction_size', 30),
            data_n_train_test_samples=1024,
            data_batch_size=256,
            flow_n_steps_flow=getattr(args, 'nfe', 20),
            training_num_epochs=getattr(args, 'num_epochs', 200),
            flow_nfe=getattr(args, 'nfe', 20),
            experiment_name=f"{getattr(args, 'system', 'lorenz')}_{getattr(args, 'nfe', 20)}",
            experiment_wandb_project=getattr(args, 'wandb_project', 'flowtime_ds'),
            experiment_seed=getattr(args, 'seed', 0)
        )
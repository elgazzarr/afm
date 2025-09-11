"""Configuration classes with validation for AFM vs FM experiments."""

from dataclasses import dataclass, field
from typing import Optional, Literal, List
import jax.numpy as jnp


@dataclass
class ModelConfig:
    """Configuration for flow matching models with validation."""
    
    # Data parameters
    num_timepoints: int = 100
    t0: float = 0.0
    t1: float = 20.0
    history_size: int = 50
    prediction_size: int = 30 
    n_train_test_samples: int = 512
    batch_size: int = 128
    
    # Model architecture
    covariates_dim: int = 1
    hidden_dim: int = 64
    time_dim: int = 16
    hidden_blocks: int = 3
    rnn_hidden: int = 64
    n_rnn_layers: int = 3
    
    # Training parameters
    learning_rate: float = 1e-3
    
    # Flow prediction
    n_samples: int = 50
    n_steps_flow: int = 50
    
    # System specific
    system: Literal[
        'lorenz', 'lotka_volterra', 'vanderpol', 'harmonic_oscillator',
        'brusselator', 'fitzhugh_nagumo', 'duffing'
    ] = 'lorenz'
    stochastic: bool = False
    
    # Derived fields
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
        if self.t1 <= self.t0:
            raise ValueError(f"t1 ({self.t1}) must be greater than t0 ({self.t0})")
        
        # Data size validation
        if self.num_timepoints < self.history_size + self.prediction_size:
            raise ValueError(
                f"num_timepoints ({self.num_timepoints}) must be >= "
                f"history_size + prediction_size ({self.history_size + self.prediction_size})"
            )
        
        if self.history_size <= 0:
            raise ValueError(f"history_size ({self.history_size}) must be positive")
            
        if self.prediction_size <= 0:
            raise ValueError(f"prediction_size ({self.prediction_size}) must be positive")
        
        # Architecture validation
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be positive")
            
        if self.time_dim <= 0:
            raise ValueError(f"time_dim ({self.time_dim}) must be positive")
            
        if self.hidden_blocks <= 0:
            raise ValueError(f"hidden_blocks ({self.hidden_blocks}) must be positive")
            
        if self.n_rnn_layers <= 0:
            raise ValueError(f"n_rnn_layers ({self.n_rnn_layers}) must be positive")
        
        # Training validation
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate ({self.learning_rate}) must be positive")
            
        if self.batch_size <= 0:
            raise ValueError(f"batch_size ({self.batch_size}) must be positive")
            
        if self.n_train_test_samples <= 0:
            raise ValueError(f"n_train_test_samples ({self.n_train_test_samples}) must be positive")
    
    def _compute_derived_fields(self) -> None:
        """Compute derived configuration fields."""
        self.extrapolation_size = self.num_timepoints - (self.history_size + self.prediction_size)
        self.ts = jnp.linspace(self.t0, self.t1, self.num_timepoints)
        self.dt = (self.ts[1] - self.ts[0]) / 5
    
    def get_system_choices(self) -> List[str]:
        """Get list of available dynamical systems."""
        return ['lorenz', 'lotka_volterra', 'vanderpol', 'harmonic_oscillator',
                'brusselator', 'fitzhugh_nagumo', 'duffing']
    
    def summary(self) -> str:
        """Get a summary of the configuration."""
        return f"""
ModelConfig Summary:
==================
System: {self.system} (stochastic: {self.stochastic})
Time: {self.t0} to {self.t1} ({self.num_timepoints} points)
Data split: {self.history_size} history + {self.prediction_size} pred + {self.extrapolation_size} extrap
Architecture: {self.hidden_dim}D hidden, {self.n_rnn_layers} RNN layers, {self.hidden_blocks} flow blocks
Training: lr={self.learning_rate}, batch_size={self.batch_size}, {self.n_train_test_samples} samples
"""
        

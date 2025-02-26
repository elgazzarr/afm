from dataclasses import dataclass
from typing import Optional, Literal
import jax.numpy as jnp

@dataclass
class ModelConfig:
    # Data parameters
    num_timepoints: int = 100
    t0: float = 0
    t1: float = 20
    history_size: int = 50
    prediction_size: int = 30 
    n_train_test_samples: int = 2000
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
    system: Literal['lorenz', 'lotka_volterra', 'vanderpol', 'harmonic_oscillator'] = 'lorenz'
    stochastic: bool = False
    
    def __post_init__(self):
        self.extrapolation_size = self.num_timepoints - (self.history_size + self.prediction_size)
        self.ts = jnp.linspace(self.t0, self.t1, self.num_timepoints)
        self.dt = (self.ts[1] - self.ts[0])  / 5
        

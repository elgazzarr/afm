import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax as dfx
from typing import Tuple, Callable, Dict, Any
from dataclasses import dataclass

@dataclass
class TimeSeriesDataset:
    """Base class for time series datasets."""
    
    def __init__(self, config: Dict[str, Any], key: jrandom.PRNGKey):
        self.config = config
        self.key = key
        
    def generate_data(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate data splits."""
        raise NotImplementedError


class LotkaVolterraDataset(TimeSeriesDataset):
    """Lotka-Volterra time series dataset."""
    
    def __init__(self, config: Dict[str, Any], key: jrandom.PRNGKey):
        super().__init__(config, key)
        self.params = jnp.array([1.3, 0.9, 0.8, 1.8])  # alpha, beta, gamma, delta
        self.dt = config['data']['dt']
        self.history_length = config['data']['history_length']
        self.seq_length = config['data']['sequence_length']
        self.ts = jnp.linspace(0, self.seq_length * self.dt, self.seq_length)
        
    def lotka_volterra(self, t: float, x: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:
        """Lotka-Volterra dynamics."""
        alpha, beta, gamma, delta = args
        dx1 = alpha*x[0] - beta*x[0]*x[1]
        dx2 = -delta*x[1] + gamma*x[1]*x[0]
        return jnp.array([dx1, dx2])
        
    def generate_data(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate Lotka-Volterra trajectories."""
        n_samples = self.config['data']['n_samples']
        # Initial conditions
        mean = jnp.array([5.0, 5.0])
        std = jnp.array([1.0, 1.0])
        x0 = jrandom.normal(self.key, (n_samples, 2)) * std + mean
        # Solve ODE
        vf = dfx.ODETerm(self.lotka_volterra)
        solver = dfx.Tsit5()
        
        solve_single = lambda x0: dfx.diffeqsolve(
            vf, solver, self.ts[0], self.ts[-1], self.dt, x0,
            args=self.params, saveat=dfx.SaveAt(ts=self.ts)
        ).ys
        
        trajectories = jax.vmap(solve_single)(x0)
        
        # Split into history and prediction
        history = trajectories[:, :self.history_length, :]
        prediction = trajectories[:, self.history_length:, :]
        covariates = jnp.repeat(self.ts[None, :, None], n_samples, axis=0)
        
        return history, covariates, prediction
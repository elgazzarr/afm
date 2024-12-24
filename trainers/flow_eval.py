import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax as dfx
from diffrax import ODETerm, Tsit5
from typing import Callable, Dict, Any, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

class FlowEvaluator:
    """Evaluation utilities for flow-based forecasting."""
    
    def __init__(self, model, nfe:int):
        """
        Initialize evaluator.
        
        Args:
            model: Trained flow model
            nfe: Number of function evaluations for ODE solver
        """
        self.model = model
        self.nfe = nfe
        
    def predict_trajectory(self, 
                         key: jrandom.PRNGKey,
                         history: jnp.ndarray,
                         covariates: jnp.ndarray,
                         n_prediction_steps: int
                         ) -> jnp.ndarray:
        """Generate trajectory prediction."""
        
        def vf(t, x, args):
            history, covariates = args
            return self.model(x, history, covariates, t)
        
        term = ODETerm(vf)
        solver = Tsit5()
        dt = 1/self.nfe

        predictions = []
        for _ in range(n_prediction_steps):
            x0 = jrandom.normal(key, shape=(history.shape[-1],))
            x_next = dfx.diffeqsolve(term, solver, 0., 1., dt, x0, args=(history, covariates), saveat=dfx.SaveAt(t1=True)).ys
            predictions.append(x_next) 
            history = jnp.concatenate([history, x_next], axis=0)
        return jnp.vstack(predictions)
    

    def compute_metrics(self, 
                       true_trajectory: jnp.ndarray,
                       pred_trajectory: jnp.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        mse = jnp.mean(jnp.square(true_trajectory - pred_trajectory))
        mae = jnp.mean(jnp.abs(true_trajectory - pred_trajectory))
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(jnp.sqrt(mse))
        }
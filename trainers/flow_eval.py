import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax as dfx
from diffrax import ODETerm, Tsit5, Euler
from typing import Callable, Dict, Any, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.scalers import ContextScaler


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
        """Generate trajectory prediction via unrolling."""
        
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
    

    def predict_trajectory_fixed_context(self,
    key: jrandom.PRNGKey,
    history: jnp.ndarray,  # (time, features)
    covariates: jnp.ndarray,
    n_prediction_steps: int,
    context_length: int
    ) -> jnp.ndarray:
        
        
        def vf(t, x, args):
            history, covariates = args
            return self.model(x, history, covariates, t)
        
        term = ODETerm(vf)
        solver = Tsit5()
        dt = 1/self.nfe
        predictions = []
        current_history = history
        
        context_scaler = ContextScaler()

        for _ in range(n_prediction_steps):
            context = current_history[-context_length:]
            context, scale = context_scaler.fit_transform(context)
            x0 = jrandom.normal(key, shape=(history.shape[-1],)) + 1.0
            x_next = dfx.diffeqsolve(term, solver, 0., 1., dt, x0, args=(context, covariates), saveat=dfx.SaveAt(t1=True)).ys
            x_next = context_scaler.inverse_transform(x_next, scale)
            predictions.append(x_next)
            current_history = jnp.concatenate([current_history, x_next], axis=0)
        
        predictions = jnp.vstack(predictions)
        return predictions
    

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
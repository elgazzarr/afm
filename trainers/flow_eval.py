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
        
    def sample_flow(model, history, covariates, n_steps_pred, n_steps_flow, n_samples, key):
        
        def predict(key, history, covariates, n_steps_flow, n_steps_pred):
            def vf(t, x, args):
                history, covariates = args
                return model(x, history, covariates, t)
                
            term = ODETerm(vf)
            solver = dfx.Euler()
            dt = 1/n_steps_flow
            predictions = []
            current_history = history
            context_length = history.shape[0]
            context_scaler = ContextScaler()
            for _ in range(n_steps_pred):
                context = current_history[-context_length:]
                context, scale = context_scaler.fit_transform(context)
                x0 = model.p0(key, shape=(history.shape[-1],))
                x_next = dfx.diffeqsolve(
                    term, solver, 0., 1., dt, x0, 
                    args=(context, covariates), 
                    saveat=dfx.SaveAt(t1=True)
                ).ys
                x_next = context_scaler.inverse_transform(x_next, scale)
                predictions.append(x_next)
                current_history = jnp.concatenate([current_history, x_next], axis=0)
                
            predictions = jnp.vstack(predictions)
            return predictions

        key_samples = jrandom.split(key, n_samples)
        
        # First vmap over batch, then over samples
        vmapped_predict = jax.vmap(predict, in_axes=(None, 0, 0, None, None))  # vmap over batch
        vmapped_predict = jax.vmap(vmapped_predict, in_axes=(0, None, None, None, None))  # vmap over samples
        
        pred_samples = vmapped_predict(key_samples, history, covariates, n_steps_flow, n_steps_pred)
        return pred_samples
    

    def predict_trajectory_fixed_context(
        self,
        key: jrandom.PRNGKey,
        history: jnp.ndarray, 
        covariates: jnp.ndarray,
        n_prediction_steps: int,
        context_length: int
    ) -> jnp.ndarray:
        
        def vf(t, x, args):
            context, covariates = args
            return self.model(x, context, covariates, t)
        
        term = ODETerm(vf)
        solver = Euler()
        dt = 1/self.nfe
        #context_scaler = ContextScaler()

        def scan_step(carry, _):
            current_history, rng = carry
            
            # Split RNGs
            rng, sample_key = jax.random.split(rng)
            
            # Get context from last context_length steps
            context = current_history[-context_length:]
            #context_scaled, scale = context_scaler.fit_transform(context)
            
            # Generate latent trajectory
            z0 = self.model.p0(sample_key, shape=(context.shape[-1],))
            z_next = dfx.diffeqsolve(
                term, solver, 0., 1., dt, z0,
                args=(context, covariates),
                saveat=dfx.SaveAt(t1=True)
            ).ys
            
            #z_next = context_scaler.inverse_transform(z_next, scale)
            
            # Update history
            new_history = jnp.concatenate([current_history, z_next], axis=0)
            current_history = new_history[-context_length:]
            return (current_history, rng), z_next

        # Initial carry with history and RNG
        init_carry = (history, key)
        
        # Run scan for n_prediction_steps
        (final_history, _), predictions = jax.lax.scan(
            scan_step,
            init_carry,
            xs=None,
            length=n_prediction_steps
        )


        return jnp.squeeze(predictions, axis=1)
    

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
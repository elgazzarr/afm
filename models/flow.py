from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import List, Tuple, Union
from .embeddings import SinusodialPosEmb
from .layers import TimeConditionedDense, FlowWave 
from gluonts.transform import SimpleTransformation
from .networks import ContextNet
import diffrax as dfx
from diffrax import ODETerm, Tsit5, Euler
import jax.random as jrandom
from data.scalers import ContextScaler, NoContextScaler

class Flow(nnx.Module):
    """
    Flow-based autoregressive forecasting model.
    Args:
        dim_x (int): Dimension of input/output data
        dim_c (int): Dimension of covariates
        dim_h (int): Hidden dimension
        dim_t (int): Time embedding dimension
        dim_hidden (int): Hidden layer dimension
        hidden_blocks (int): Number of hidden blocks
        n_rnn_layers (int): Number of RNN layers
        rngs (nnx.Rngs): Random number generators
    """
    def __init__(self,
                 dim_x: int,
                 dim_c: int,
                 dim_h: int,
                 dim_t: int,
                 n_rnn_layers: int,
                 rnn_hidden: int,
                 residual_layers: int,
                 residual_channels: int,
                 residual_hidden: int,
                 rngs: nnx.Rngs,
                 scaler: SimpleTransformation,
                 context_scaling: bool,
                 p0: Union [nnx.Module, None] = None):
        super().__init__()

        self.scaler = scaler
        self.dim_x = dim_x
        self.dim_h = dim_h
        self.context_encoder = ContextNet(dim_x, dim_h, dim_c, n_rnn_layers, rnn_hidden, rngs)
        self.vector_field = FlowWave(dim_x, dim_h, dim_c, dim_t, residual_layers, residual_channels, residual_hidden, rngs=rngs)
        self.p0 = p0 if p0 is not None else jrandom.normal
        self.context_scaler = ContextScaler() if context_scaling else NoContextScaler()
    
    def __call__(self,
                 z: jnp.ndarray,
                 z_history: jnp.ndarray,
                 c: jnp.ndarray,
                 t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the flow model."""
        h = self.context_encoder(z_history, c)
        return self.vector_field(z, h, c[z_history.shape[0]], t)
    
    
    def sample_flow(self, history, covariates, n_steps_pred, n_steps_flow, n_samples, key):
        
        def predict(
            key: jax.random.PRNGKey,
            history: jnp.ndarray,   
            covariates: jnp.ndarray,
            n_steps_flow: int,
            n_steps_pred: int
        ) -> jnp.ndarray:
            
            def vf(t, x, args):
                history, covariates = args
                return self(x, history, covariates, t)
                
            term = ODETerm(vf)
            solver = dfx.Euler()
            dt = 1/n_steps_flow

            context_scaler = self.context_scaler
            context_length = history.shape[0]
            current_context = history
            predictions = []
            
            for _ in range(n_steps_pred):
                context_scaled, scale = context_scaler.fit_transform(current_context[-context_length:])
                x0 = self.p0(key, (self.dim_x,))
                x_next_scaled = dfx.diffeqsolve(
                    term,
                    solver,
                    0., 1.,
                    dt,
                    x0,
                    args=(context_scaled, covariates),
                    saveat=dfx.SaveAt(t1=True)
                ).ys
                x_next = context_scaler.inverse_transform(x_next_scaled, scale)
                predictions.append(x_next)
                current_context = jnp.concatenate([current_context, x_next], axis=0)
            

            return jnp.vstack(predictions)
        
        key_samples = jrandom.split(key, n_samples)
        
        # First vmap over batch, then over samples
        vmapped_predict = jax.vmap(predict, in_axes=(None, 0, 0, None, None))  # vmap over batch
        vmapped_predict = jax.vmap(vmapped_predict, in_axes=(0, None, None, None, None))  # vmap over samples
        
        pred_samples = vmapped_predict(key_samples, history, covariates, n_steps_flow, n_steps_pred)
        return pred_samples
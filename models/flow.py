from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import List, Tuple

from .embeddings import SinusodialPosEmb
from .layers import TimeConditionedDense

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
        rngs (nnx.Rngs): Random number generators
    """
    def __init__(self, 
                 dim_x: int, 
                 dim_c: int, 
                 dim_h: int, 
                 dim_t: int, 
                 dim_hidden: int, 
                 hidden_blocks: int, 
                 rngs: nnx.Rngs):
        super().__init__()
        self.cell = nnx.nn.recurrent.OptimizedLSTMCell(
            in_features=dim_x + dim_c,
            hidden_features=dim_h,
            rngs=rngs
        )
        self.dim_h = dim_h
        self.linear_in = nnx.Linear(dim_x + dim_h, dim_hidden, rngs=rngs)
        self.blocks = [TimeConditionedDense(dim_hidden, dim_h, dim_t, rngs) 
                      for _ in range(hidden_blocks)]
        self.linear_out = nnx.Linear(dim_hidden + dim_h, dim_x, rngs=rngs)
        
    def process_history(self, 
                       x_history: jnp.ndarray, 
                       c: jnp.ndarray) -> jnp.ndarray:
        """Process historical data through LSTM."""
        inputs = jnp.concatenate([x_history, c[:x_history.shape[0]]], axis=-1)
        c0 = h0 = jnp.zeros((self.dim_h))
        carry = (c0, h0)
        
        def scan_fn(carry: Tuple, x: jnp.ndarray) -> Tuple:
            return self.cell(carry, x)
        
        final_carry, outputs = jax.lax.scan(scan_fn, carry, inputs)
        return outputs[-1]
    
    def __call__(self, 
                 z: jnp.ndarray, 
                 x_history: jnp.ndarray, 
                 c: jnp.ndarray, 
                 t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the flow model."""
        h_last = self.process_history(x_history, c)
        z = nnx.gelu(self.linear_in(jnp.concatenate([z, h_last], axis=-1)))
        
        for block in self.blocks:
            z = nnx.gelu(block(z, h_last, t)) + z
            
        x_next = self.linear_out(jnp.concatenate([z, h_last], axis=-1))
        return x_next
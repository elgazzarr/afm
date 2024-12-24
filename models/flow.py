from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import List, Tuple
from .embeddings import SinusodialPosEmb
from .layers import TimeConditionedDense, FlowWave 
from gluonts.transform import SimpleTransformation


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
                 scaler: SimpleTransformation):
        super().__init__()

        self.scaler = scaler
        self.cells = [
            nnx.nn.recurrent.OptimizedLSTMCell(
                in_features=dim_x + dim_c if i == 0 else rnn_hidden,
                hidden_features=rnn_hidden,
                rngs=rngs
            ) for i in range(n_rnn_layers)
        ]
        self.rnn_hidden = rnn_hidden
        self.n_rnn_layers = n_rnn_layers
        self.dim_h = dim_h
        self.hidden_proj = nnx.Linear(rnn_hidden, dim_h, rngs=rngs)
        self.vector_field = FlowWave(dim_x, dim_h, dim_c, dim_t, residual_layers, residual_channels, residual_hidden, rngs=rngs)

    def process_history(self,
                       x_history: jnp.ndarray,
                       c: jnp.ndarray) -> jnp.ndarray:
        """Process historical data through multiple LSTM layers."""
        inputs = jnp.concatenate([x_history, c[:x_history.shape[0]]], axis=-1)
        
        # Initialize carries for all layers
        carries = [(jnp.zeros((self.rnn_hidden)), jnp.zeros((self.rnn_hidden))) 
                  for _ in range(self.n_rnn_layers)]
        
        def scan_fn(carries: List[Tuple], x: jnp.ndarray) -> Tuple:
            current_input = x
            new_carries = []
            for i, (carry, cell) in enumerate(zip(carries, self.cells)):
                carry, output = cell(carry, current_input)
                current_input = output
                new_carries.append(carry)
            return new_carries, current_input

        final_carries, outputs = jax.lax.scan(scan_fn, carries, inputs)
        return outputs[-1]  # Return the last hidden state from the last timestep

    def __call__(self,
                 z: jnp.ndarray,
                 x_history: jnp.ndarray,
                 c: jnp.ndarray,
                 t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the flow model."""
        h_last = self.process_history(x_history, c)
        h_last = self.hidden_proj(h_last)
        x_velocity = self.vector_field(z, h_last, c[x_history.shape[0]], t)
        return x_velocity
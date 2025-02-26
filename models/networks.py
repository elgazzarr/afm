
from .embeddings import SinusodialPosEmb
import flax.nnx as nnx
import jax.numpy as jnp
import jax
import math
from typing import Tuple, List



class ContextNet(nnx.Module):
    def __init__(self, dim_x, dim_h, dim_c, n_rnn_layers, rnn_hidden, rngs):

        self.rnn_hidden = rnn_hidden
        self.n_rnn_layers = n_rnn_layers
        self.cells = [nnx.nn.recurrent.OptimizedLSTMCell(
                in_features = dim_x + dim_c if i == 0 else rnn_hidden,
                hidden_features = rnn_hidden,
                rngs=rngs
            ) for i in range(n_rnn_layers)]
        self.context = nnx.Linear(rnn_hidden, dim_h, rngs=rngs)
    
    def __call__(self, x:jnp.ndarray, c:jnp.ndarray) -> jnp.ndarray:
        """Process historical data through multiple LSTM layers."""
        inputs = jnp.concatenate([x, c[:x.shape[0]]], axis=-1)
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
        return self.context(outputs)[-1]
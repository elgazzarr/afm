
from .embeddings import SinusodialPosEmb
import flax.nnx as nnx
import jax.numpy as jnp
import jax
import math
from typing import Tuple, List



class AbstractEncoder(nnx.Module):
    def __init__(self, dim_x, dim_z, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, x):
        raise NotImplementedError
    

class MLPEncoder(AbstractEncoder):
    def __init__(self, dim_x, dim_z, dim_c, *args, **kwargs):
        super().__init__(dim_x, dim_z, *args, **kwargs)
        self.net = nnx.Sequential([
            nnx.Linear(dim_x+dim_c, 128, rngs=kwargs['rngs']),
            nnx.gelu,
            nnx.Linear(128, 128, rngs=kwargs['rngs']),
            nnx.gelu,
            nnx.Linear(128, dim_z, rngs=kwargs['rngs'])
        ])

    def __call__(self, x:jnp.ndarray, c:jnp.ndarray) -> jnp.ndarray:
        """Process historical data through multiple LSTM layers."""
        inputs = jnp.concatenate([x, c[:x.shape[0]]], axis=-1)
        return self.net(inputs)
    


class RecurrentEncoder(AbstractEncoder):
    def __init__(self, dim_x, dim_z, dim_c, n_rnn_layers, rnn_hidden, rngs):

        self.rnn_hidden = rnn_hidden
        self.n_rnn_layers = n_rnn_layers
        self.cells = [
            nnx.nn.recurrent.OptimizedLSTMCell(
                in_features = dim_x + dim_c if i == 0 else rnn_hidden,
                hidden_features = rnn_hidden,
                rngs=rngs
            ) for i in range(n_rnn_layers)]
        self.latent = nnx.Linear(rnn_hidden, dim_z, rngs=rngs)
    
    def __call__(self, x:jnp.ndarray, c:jnp.ndarray) -> jnp.ndarray:
        """Process historical data through multiple LSTM layers."""
        inputs = jnp.concatenate([x, c[:x.shape[0]]], axis=-1)
        
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
        latents = self.latent(outputs)
        return latents




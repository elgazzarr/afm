"""Base flow model and common components."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import jax.numpy as jnp
import flax.nnx as nnx
import jax


@dataclass
class SinusoidalPosEmb(nnx.Module):
    """Sinusoidal positional embedding for time encoding."""
    dim: int
    
    def __call__(self, t):
        freqs = jnp.arange(1, self.dim//2 + 1) * jnp.pi
        emb = t * freqs
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


class TimeConditionedDense(nnx.Module):
    """Time-conditioned dense layer for flow matching."""
    
    def __init__(self, dim_x: int, dim_h: int, dim_t: int, rngs: nnx.Rngs):
        self.time_emb = SinusoidalPosEmb(dim_t)
        self.linear = nnx.Linear(dim_x + dim_h + dim_t, dim_x, rngs=rngs)
    
    def __call__(self, x, h, t):
        t = self.time_emb(t)
        x = jnp.concatenate([x, h, t], axis=-1)
        return self.linear(x)


class TimeRNNBlock(nnx.Module):
    """Time-conditioned RNN block with bidirectional processing."""
    
    def __init__(self, dim_in: int, dim_out: int, dim_h: int, time_dim: int, rngs: nnx.Rngs):
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.lstm_fwd = nnx.nn.recurrent.OptimizedLSTMCell(
            dim_in + dim_h + time_dim, dim_out, rngs=rngs)
        self.lstm_bwd = nnx.nn.recurrent.OptimizedLSTMCell(
            dim_in + dim_h + time_dim, dim_out, rngs=rngs)

    def __call__(self, x, h, t):
        time_emb = self.time_emb(t)
        time_emb = jnp.repeat(time_emb[None, :], x.shape[0], axis=0)
        x_combined = jnp.concatenate([x, h, time_emb], axis=-1)
        
        def scan_fn(carry, x_t):
            carry_fw, carry_bw = carry
            carry_fw, out_fw = self.lstm_fwd(carry_fw, x_t)
            carry_bw, out_bw = self.lstm_bwd(carry_bw, x_t[::-1])
            return (carry_fw, carry_bw), (out_fw, out_bw)
            
        hidden_dim = self.lstm_fwd.hidden_features
        initial_carry = (
            (jnp.zeros((hidden_dim,)), jnp.zeros((hidden_dim,))),
            (jnp.zeros((hidden_dim,)), jnp.zeros((hidden_dim,)))
        )
        _, (outputs_fw, outputs_bw) = jax.lax.scan(scan_fn, initial_carry, x_combined)
        return jnp.concatenate([outputs_fw, outputs_bw[::-1]], axis=-1)


class BaseFlow(nnx.Module):
    """Base class for flow matching models."""
    
    def process_history(self, x_history: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        """Process historical data through RNN layers.
        
        Args:
            x_history: Historical sequence data
            c: Covariate/time information
            
        Returns:
            Processed history representation
        """
        inputs = jnp.concatenate([x_history, c[:x_history.shape[0]]], axis=-1)
        
        carries = [(jnp.zeros((self.rnn_hidden,)), jnp.zeros((self.rnn_hidden,)))
                  for _ in range(self.n_rnn_layers)]
        
        def scan_fn(carries, x):
            current_input = x
            new_carries = []
            for i, (carry, cell) in enumerate(zip(carries, self.cells)):
                carry, output = cell(carry, current_input)
                current_input = output
                new_carries.append(carry)
            return new_carries, current_input

        _, outputs = jax.lax.scan(scan_fn, carries, inputs)
        return outputs[-1]
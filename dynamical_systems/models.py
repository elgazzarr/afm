from dataclasses import dataclass
from typing import List, Tuple, Optional
import jax.numpy as jnp
import flax.nnx as nnx
from functools import partial
import jax 

@dataclass
class SinusodialPosEmb(nnx.Module):
    dim: int
    def __call__(self, t):
        freqs = jnp.arange(1, self.dim//2 + 1) * jnp.pi
        emb = t * freqs
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

class TimeConditionedDense(nnx.Module):
    def __init__(self, dim_x: int, dim_h: int, dim_t: int, rngs: nnx.Rngs):
        self.time_emb = SinusodialPosEmb(dim_t)
        self.linear = nnx.Linear(dim_x + dim_h + dim_t, dim_x, rngs=rngs)
    
    def __call__(self, x, h, t):
        t = self.time_emb(t)
        x = jnp.concatenate([x, h, t], axis=-1)
        return self.linear(x)

class TimeRNNBlock(nnx.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_h: int, time_dim: int, rngs: nnx.Rngs):
        self.time_emb = SinusodialPosEmb(time_dim)
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
    def process_history(self, x_history: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
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

        _ , outputs = jax.lax.scan(scan_fn, carries, inputs)
        return outputs[-1]

class ARFlow(BaseFlow):
    def __init__(self, dim_x: int, dim_c: int, dim_h: int, dim_t: int, 
                 dim_hidden: int, hidden_blocks: int, n_rnn_layers: int, 
                 rnn_hidden: int, rngs: nnx.Rngs):
        super().__init__()
        self.n_rnn_layers = n_rnn_layers
        self.rnn_hidden = rnn_hidden
        self.cells = [
            nnx.nn.recurrent.OptimizedLSTMCell(
                in_features=dim_x + dim_c if i == 0 else rnn_hidden,
                hidden_features=rnn_hidden,
                rngs=rngs
            ) for i in range(n_rnn_layers)
        ]
        self.dim_h = dim_h
        self.linear_in = nnx.Linear(dim_x + dim_h, dim_hidden, rngs=rngs)
        self.blocks = [TimeConditionedDense(dim_hidden, dim_h, dim_t, rngs) 
                      for _ in range(hidden_blocks)]
        self.linear_out = nnx.Linear(dim_hidden + dim_h, dim_x, rngs=rngs)
    
    def __call__(self, z, x_history, c, t):
        h_last = self.process_history(x_history, c)
        z = nnx.gelu(self.linear_in(jnp.concatenate([z, h_last], axis=-1)))
        for block in self.blocks:
            z = nnx.gelu(block(z, h_last, t)) + z
        x_next = self.linear_out(jnp.concatenate([z, h_last], axis=-1))
        return x_next

class NoARFlow(BaseFlow):
    def __init__(self, dim_x: int, dim_c: int, dim_h: int, dim_t: int, dim_hidden: int, hidden_blocks: int, 
                 n_rnn_layers: int, rngs: nnx.Rngs):
        super().__init__()
        self.n_rnn_layers = n_rnn_layers
        self.rnn_hidden = dim_h
        self.cells = [
            nnx.nn.recurrent.OptimizedLSTMCell(
                in_features=dim_x + dim_c if i == 0 else dim_h,
                hidden_features=dim_h,
                rngs=rngs
            ) for i in range(n_rnn_layers)
        ]
        self.dim_h = dim_h
        self.linear_in = nnx.Conv(dim_x + dim_h, dim_hidden, kernel_size=1, rngs=rngs)
        self.hidden_blocks = [
            TimeRNNBlock(dim_hidden, dim_hidden//2, dim_h, dim_t, rngs=rngs)
            for _ in range(hidden_blocks)
        ]
        self.linear_out = nnx.Conv(dim_hidden, dim_x, kernel_size=1, rngs=rngs)
    
    def __call__(self, z, x_history, c, t):
        h = self.process_history(x_history, c)
        h = jnp.repeat(h[None, :], z.shape[0], axis=0)
        z = nnx.gelu(self.linear_in(jnp.concatenate([z, h], axis=-1)))
        for block in self.hidden_blocks:
            z = block(z, h, t)
        x_next = self.linear_out(z)
        return x_next

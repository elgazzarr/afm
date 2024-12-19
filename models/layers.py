
from .embeddings import SinusodialPosEmb
import flax.nnx as nnx
import jax.numpy as jnp

class TimeConditionedDense(nnx.Module):
    def __init__(self, dim_x: int, dim_h: int, dim_t:int, rngs: nnx.Rngs):
        self.time_emb = SinusodialPosEmb(dim_t) 
        self.linear = nnx.Linear(dim_x + dim_h + dim_t , dim_x, rngs=rngs)
    
    def __call__(self, x, h, t):
        t = self.time_emb(t)
        x = jnp.concatenate([x, h, t], axis=-1) 
        return self.linear(x)
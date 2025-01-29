
from .embeddings import SinusodialPosEmb
import flax.nnx as nnx
import jax.numpy as jnp
import jax
import math
from typing import Tuple, List



class AbstractDecoder(nnx.Module):
    def __init__(self, dim_x, dim_z, **kwargs):
        raise NotImplementedError
    
    def __call__(self, x):
        raise NotImplementedError
    

class LinearDecoder(AbstractDecoder):
    def __init__(self, dim_z, dim_x, rngs):
        self.linear = nnx.Linear(dim_z, dim_x, rngs=rngs)
    
    def __call__(self, z):
        return nnx.gelu(self.linear(z))

class MLPDecoder(AbstractDecoder):
    def __init__(self, dim_z, dim_x, rngs, hidden_dims: List[int] = [128]):
        self.mlp = nnx.Sequential(nnx.Linear(dim_z, hidden_dims[0], rngs=rngs), nnx.gelu)
        for i in range(1, len(hidden_dims)):
            self.mlp.append(nnx.Linear(hidden_dims[i-1], hidden_dims[i], rngs=rngs))
            self.mlp.append(nnx.gelu)
        self.mlp.append(nnx.Linear(hidden_dims[-1], dim_x, rngs=rngs))
    
    def __call__(self, z):
        return self.mlp(z)

class GaussianDecoder(AbstractDecoder):
    def __init__(self, dim_z, dim_x, rngs):
        self.mu = nnx.Sequential(nnx.Linear(dim_z, 128, rngs=rngs), nnx.gelu, nnx.Linear(128, dim_x, rngs=rngs))
        self.sd = nnx.Sequential(nnx.Linear(dim_z, 128, rngs=rngs), nnx.gelu, nnx.Linear(128, dim_x, rngs=rngs))
    
    def __call__(self, z):
        mu = self.mu(z)
        sd = jnp.exp(self.sd(z))
        return mu, sd




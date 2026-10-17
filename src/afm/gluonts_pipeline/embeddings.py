"""GluonTS benchmark pipeline (Table 4 of the paper). Faithful port of git revision f4131fe^ (models/embeddings.py); only imports/wandb-optionality were adapted. See REPRODUCIBILITY.md."""
import jax.numpy as jnp
from dataclasses import dataclass
import flax.nnx as nnx

@dataclass
class SinusodialPosEmb(nnx.Module):
    dim: int 
    def __call__(self, t):
        freqs = jnp.arange(1, self.dim//2 + 1) * jnp.pi
        emb = t * freqs
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    

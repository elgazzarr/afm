from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import List, Tuple
import diffrax as dfx
from diffrax import ODETerm, Tsit5, Euler
import jax.random as jrandom
from .embeddings import SinusodialPosEmb
from .layers import TimeConditionedDense, FlowWave 
from gluonts.transform import SimpleTransformation
from .encoders import RecurrentEncoder, MLPEncoder
from .decoders import LinearDecoder, GaussianDecoder, MLPDecoder

class LatentFlow(nnx.Module):
    """
    Latent Flow-based autoregressive forecasting model.
    Args:
        dim_x (int): Dimension of input/output data
        dim_z (int): Dimension of latent space
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
                 dim_z: int,
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
        self.latent_dim = dim_z
        self.encoder = RecurrentEncoder(dim_x, dim_z, dim_c, n_rnn_layers, rnn_hidden, rngs=rngs)
        self.context_map = nnx.Linear(dim_z, dim_h, rngs=rngs)
        self.vector_field = FlowWave(dim_z, dim_h, dim_c, dim_t, residual_layers, residual_channels, residual_hidden, rngs=rngs)
        self.decoder = MLPDecoder(dim_z, dim_x, rngs=rngs)

    @staticmethod
    def p0(key, shape):
        return jax.random.normal(key, shape)
    
    def p1(self, key: jrandom.PRNGKey, context: jnp.ndarray, covariates: jnp.ndarray) -> jnp.ndarray:

        def vf(t, x, args):
            history, covariates = args
            return self(x, history, covariates, t)
        
        term = ODETerm(vf)
        solver = Euler()
        dt = 1/50
        x0 = jrandom.normal(key, shape=(context.shape[-1],))
        x_next = dfx.diffeqsolve(term, solver, 0., 1., dt, x0, args=(context, covariates), saveat=dfx.SaveAt(t1=True)).ys
        return x_next    


    def __call__(self,
                 z: jnp.ndarray,
                 x_history: jnp.ndarray,
                 c: jnp.ndarray,
                 t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the latent flow model."""
        z_past = self.encoder(x_history, c[:x_history.shape[0]])
        h = nnx.gelu(self.context_map(z_past[-1]))
        z_velocity = self.vector_field(z, h, c[x_history.shape[0]], t)
        return z_velocity
    

    def encode(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        """Encodes the input data."""
        return self.encoder(x, c)
    
    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """Decodes the latent state."""
        return self.decoder(z)
    
    def get_context(self, z: jnp.ndarray) -> jnp.ndarray:
        """Returns the context vector."""
        return nnx.gelu(self.context_map(z[-1]))

    def predict(self, z: jnp.ndarray, h: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        z_velocity = self.vector_field(z, h, c, t)
        return z_velocity
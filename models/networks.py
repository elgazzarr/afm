
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



class p0_AR(nnx.Module):
    def __init__(self, latent_dim, rngs):
        # Initialize as [1, latent_dim] for broadcasting to batches
        self.initial_state = nnx.Param(jnp.zeros((1, latent_dim)), rngs=rngs)
        self.transition = nnx.Linear(latent_dim, 2 * latent_dim, rngs=rngs)
        
    def __call__(self, z_prev):
        # z_prev shape: [batch_size, latent_dim]
        out = self.transition(z_prev)
        prior_mu, prior_logvar = jnp.split(out, 2, axis=-1)
        prior_logvar = jnp.clip(prior_logvar, -10, 0)
        return prior_mu, prior_logvar
    
    def sample(self, key, z_prev):
        prior_mu, prior_logvar = self(z_prev)
        return jax.random.normal(key, shape=prior_mu.shape) * jnp.exp(0.5 * prior_logvar) + prior_mu
    

class p0_AR_fixed(nnx.Module):
    def __init__(self, phi=0.9, sigma=1.0):
        self.phi = phi  # AR coefficient (fixed)
        self.sigma = sigma  # Noise scale (fixed)
    
    def __call__(self, z_prev):
        prior_mu = self.phi * z_prev  # AR(1): μ_t = φ * z_{t-1}
        prior_logvar = 2 * jnp.log(self.sigma)  # log(σ²)
        return prior_mu, prior_logvar
    
    def initial_prior(self, batch_size, latent_dim):
        # Initial state: z_0 ~ N(0, σ²)
        prior_mu_0 = jnp.zeros((batch_size, latent_dim))
        prior_logvar_0 = 2 * jnp.log(self.sigma)
        return prior_mu_0, prior_logvar_0
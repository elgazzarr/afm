import flax.nnx as nnx
import jax.numpy as jnp
import jax


class p0_gaussian(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        self.initial_state = nnx.Variable(jnp.zeros((1, latent_dim)))
    
    def __call__(self, z_prev):
        prior_mu = jnp.zeros_like(z_prev)
        prior_logvar = jnp.zeros_like(z_prev)
        return prior_mu, prior_logvar
    
    def sample(self, key, z_prev):
        return jax.random.normal(key, shape=z_prev.shape)



class p0_AR_learnable(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.initial_state = nnx.Param(jnp.zeros((1, latent_dim)), rngs=rngs)
        self.transition = nnx.Linear(latent_dim, 2 * latent_dim, rngs=rngs)
        
    def __call__(self, z_prev):
        out = self.transition(z_prev)
        prior_mu, prior_logvar = jnp.split(out, 2, axis=-1)
        prior_logvar = jnp.clip(prior_logvar, -1, 1)
        return prior_mu, prior_logvar
    
    def sample(self, key, z_prev):
        prior_mu, prior_logvar = self(z_prev)
        return jax.random.normal(key, shape=prior_mu.shape) * jnp.exp(0.5 * prior_logvar) + prior_mu
    

class p0_AR_fixed(nnx.Module):
    def __init__(self, latent_dim, rngs, phi=0.9, sigma=1.0):
        self.latent_dim = latent_dim
        self.phi = phi      
        self.sigma = sigma  
        self.initial_state = nnx.Variable(jnp.zeros((1, latent_dim)))

    def __call__(self, z_prev):
        prior_mu = self.phi * z_prev          # AR(1): μ_t = φ * z_{t-1}
        prior_logvar = 2 * jnp.log(self.sigma) * jnp.ones_like(z_prev)  # log(σ²)
        prior_logvar = jnp.clip(prior_logvar, -1, 1)  
        return prior_mu, prior_logvar

    def sample(self, key, z_prev):
        prior_mu, prior_logvar = self(z_prev)
        return jax.random.normal(key, shape=prior_mu.shape) * jnp.exp(0.5 * prior_logvar) + prior_mu
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
        prior_logvar = jnp.clip(prior_logvar, -4, -2)  
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
        prior_logvar = jnp.clip(prior_logvar, -4, -2)  
        return prior_mu, prior_logvar

    def sample(self, key, z_prev):
        prior_mu, prior_logvar = self(z_prev)
        return jax.random.normal(key, shape=prior_mu.shape) * jnp.exp(0.5 * prior_logvar) + prior_mu
    

class p0_Seasonal_fixed(nnx.Module):
    def __init__(self, latent_dim, rngs, period=12, amplitude=1.0, sigma=1.0):
        self.latent_dim = latent_dim
        self.period = period
        self.amplitude = amplitude
        self.sigma = sigma
        self.initial_state =  nnx.Variable(jax.random(rngs, (1, latent_dim)))
        
    def __call__(self, z_prev):
        t = jnp.arange(z_prev.shape[0])[:, None]
        prior_mu = self.amplitude * jnp.sin(2 * jnp.pi * t / self.period)
        prior_logvar = 2 * jnp.log(self.sigma) * jnp.ones_like(z_prev)
        prior_logvar = jnp.clip(prior_logvar, -4, -2)  
        return prior_mu, prior_logvar
    
    def sample(self, key, z_prev):
        prior_mu, prior_logvar = self(z_prev)
        return jax.random.normal(key, shape=prior_mu.shape) * jnp.exp(0.5 * prior_logvar) + prior_mu

    

class p0_OU_fixed(nnx.Module):
    def __init__(self, latent_dim, rngs, theta=0.1, sigma=0.1):
        self.initial_state = nnx.Variable(jnp.zeros((1, latent_dim)))
        self.theta = theta
        self.sigma = sigma

    def __call__(self, z_prev):
        prior_mu = (1 - self.theta) * z_prev  # OU mean reversion
        prior_logvar = 2 * jnp.log(self.sigma) * jnp.ones_like(z_prev)
        prior_logvar = jnp.clip(prior_logvar, -1, 1)
        return prior_mu, prior_logvar

    def sample(self, key, z_prev):
        prior_mu, prior_logvar = self(z_prev)
        return jax.random.normal(key, shape=prior_mu.shape) * jnp.exp(0.5 * prior_logvar) + prior_mu
    


class p0_langevain_fixed(nnx.Module):
    def __init__(self, latent_dim, rngs, alpha=0.1, sigma=0.1):
        self.initial_state = nnx.Variable(jnp.zeros((1, latent_dim)))
        self.velocity = nnx.Variable(jnp.zeros((1, latent_dim)))
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, z_prev):
        v_mean = (1 - self.alpha) * self.velocity
        prior_mu = z_prev + v_mean
        prior_logvar = 2 * jnp.log(self.sigma) * jnp.ones_like(z_prev)
        prior_logvar = jnp.clip(prior_logvar, -1, 1)
        return prior_mu, prior_logvar

    def sample(self, key, z_prev):
        key1, key2 = jax.random.split(key)
        v_noise = jax.random.normal(key1, shape=self.velocity.shape) * self.sigma
        self.velocity = (1 - self.alpha) * self.velocity + v_noise
        prior_mu, prior_logvar = self(z_prev)
        return jax.random.normal(key2, shape=prior_mu.shape) * jnp.exp(0.5 * prior_logvar) + prior_mu
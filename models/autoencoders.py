import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial
from models.priors import p0_AR_learnable, p0_AR_fixed, p0_gaussian

class CausalVAE(nnx.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims,  n_layers, kernel_size, rngs, prior="AR_learn"):
        # Initialize with proper weight scaling for stability
        self.encoder = CausalEncoder(input_dim, latent_dim, hidden_dims, n_layers, kernel_size, rngs)
        self.decoder = CausalDecoder(latent_dim, input_dim, hidden_dims, n_layers, kernel_size, rngs)
        if prior == "AR_learn":
            self.prior = p0_AR_learnable(latent_dim, rngs)
        elif prior == "AR_fixed":
            self.prior = p0_AR_fixed(latent_dim, rngs)
        elif prior == "Gaussian":
            self.prior = p0_gaussian(latent_dim, rngs)
        else:
            raise ValueError(f"Prior {prior} not recognized")

        
    def __call__(self, x, key): 
        mean, logvar = self.encoder(x) 
        z = self.reparameterize(key, mean, logvar)
        recon = self.decoder(z) 
        return recon, mean, logvar
    
    def reparameterize(self, key, mean, logvar):
        std = jnp.exp(0.5 * jnp.clip(logvar, -10, 10))
        eps = jax.random.normal(key, mean.shape)
        return mean + eps * std
    
    def sample_posterior(self, x, key):
        mean, logvar = self.encoder(x)
        return self.reparameterize(key, mean, logvar)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu
    
    def decode(self, z):
        return self.decoder(z)

class CausalEncoder(nnx.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, n_layers, kernel_size, rngs):
        # Initialize with proper scaling
        kernel_init = nnx.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        
        self.conv_init = nnx.Conv(
            input_dim, hidden_dims, 
            kernel_size=1, 
            padding='VALID',
            kernel_init=kernel_init,
            rngs=rngs
        )
        
        self.conv_layers = [
            nnx.Conv(
                hidden_dims, hidden_dims, 
                kernel_size=kernel_size,
                padding='CAUSAL',
                kernel_init=kernel_init,
                rngs=rngs
            ) for _ in range(n_layers)
        ]
        
        self.mean = nnx.Linear(hidden_dims, latent_dim, rngs=rngs)
        self.logvar = nnx.Linear(hidden_dims, latent_dim, rngs=rngs)
        
    def __call__(self, x):
        x = self.conv_init(x)
        
        # Add residual connections and layer normalization
        for conv in self.conv_layers:
            residual = x
            x = nnx.gelu(conv(x))
            x = x + residual
            
        mean = self.mean(x)
        logvar = self.logvar(x)
        
        return mean, logvar

class CausalDecoder(nnx.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims, n_layers, kernel_size, rngs):
        # Initialize with proper scaling
        kernel_init = nnx.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        
        self.conv_init = nnx.Conv(
            latent_dim, hidden_dims, 
            kernel_size=1, 
            padding='VALID',
            kernel_init=kernel_init,
            rngs=rngs
        )
        
        self.conv_layers = [
            nnx.Conv(
                hidden_dims, hidden_dims,
                kernel_size=1, # not sure about this 
                padding='CAUSAL',
                kernel_init=kernel_init,
                rngs=rngs
            ) for _ in range(n_layers)
        ]
        
        self.output = nnx.Conv(
            hidden_dims, output_dim, 
            kernel_size=1, 
            padding='VALID',
            kernel_init=kernel_init,
            rngs=rngs
        )
        
    def __call__(self, z):
        x = self.conv_init(z)
        
        # Add residual connections and layer normalization
        for conv in self.conv_layers:
            residual = x
            x = nnx.gelu(conv(x))
            x = x + residual
            
        x = self.output(x)
        return x


class IdentityVAE(nnx.Module):

    def __init__(self):
        print("IdentityVAE")
        
    def __call__(self, x, key):  # x: [seq_len, input_dim]
        return 
    
    def reparameterize(self, key, mean, logvar):
        return

    def encode(self, x):
        return x
    
    def decode(self, z):
        return z

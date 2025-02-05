import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from typing import Dict, Any
import wandb
from tqdm import tqdm
from trainers.flow_eval import FlowEvaluator
from utils.visualization import Visualizer
import jax.random as jrandom
from matplotlib import pyplot as plt
from data.scalers import ContextScaler
from copy import deepcopy
import diffrax as dfx
from diffrax import ODETerm, Tsit5, Euler


class AutoEncoderTrainer:
    def __init__(self, 
                 config: Dict[str, Any],
                 model: nnx.Module,
                 train_loader: Any,
                 val_loader: Any,
                 rngs: nnx.Rngs):
        
        self.config = config
        self.model = model
        self.best_model = None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.rngs = rngs 
        self.optimizer = nnx.Optimizer(self.model , optax.adam(config.learning_rate/5.0))
        self.visualizer = Visualizer()
        self.beta_scheduler =  optax.schedules.linear_schedule(0.0, 1.0, config.training_steps//3)

    def loss_fn(
        self,
        model: nnx.Module,
        x: jnp.ndarray,  # Shape: [batch_size, seq_len, input_dim]
        beta: float,
        rng: nnx.Rngs
    ) -> float:
        batch_size = x.shape[0]
        x_hat, mu, logvar = model(x, rng)  # mu/logvar: [batch_size, seq_len, latent_dim]
        z = model.reparameterize(rng, mu, logvar)  # z: [batch_size, seq_len, latent_dim]

        # Initialize carry with broadcasted prior.initial_state
        initial_carry = jnp.broadcast_to(
            model.prior.initial_state.value, 
            (batch_size, model.prior.initial_state.value.shape[-1])
        )

        def scan_fn(carry, t):
            prior_mu_t, prior_logvar_t = model.prior(carry)
            next_carry = z[:, t, :]  # Use posterior sample for next carry (for KL calculation)
            return next_carry, (prior_mu_t, prior_logvar_t)

        # Scan over sequence length
        _, (prior_mu, prior_logvar) = jax.lax.scan(
            scan_fn, initial_carry, jnp.arange(z.shape[1])
        )
        
        # Transpose to [batch_size, seq_len, latent_dim]
        prior_mu = jnp.moveaxis(prior_mu, 0, 1)
        prior_logvar = jnp.moveaxis(prior_logvar, 0, 1)

        # Compute KL divergence
        kl_per_t = 0.5 * (
            prior_logvar - logvar + 
            (jnp.exp(logvar) + (mu - prior_mu)**2) / jnp.exp(prior_logvar) - 1
        )
        kl_loss = jnp.mean(kl_per_t)
        
        recon_loss = jnp.mean((x - x_hat) ** 2)
        loss = recon_loss + beta * kl_loss
        return loss



    @nnx.jit(static_argnums=(0,))
    def train_step(self,
                  model: nnx.Module,
                  optimizer: nnx.Optimizer,
                  x: jnp.ndarray,
                  beta: float,
                  rng: nnx.Rngs) -> float:
        
        loss, grads = nnx.value_and_grad(self.loss_fn)(model, x, beta, rng)
        optimizer.update(grads)
        return loss
    
    def validate(self, model, plot=False, n_plots=3):
        val_losses = []
        for i, (x_history, _, x_future) in enumerate(self.val_loader):
            x = jnp.concatenate([x_history, x_future], axis=1)
            val_loss  = self.loss_fn(model, x, 0.0, self.rngs())
            val_losses.append(val_loss)
        avg_val_loss = jnp.mean(jnp.array(val_losses))
        if plot:
            x_sample = x[0]
            x_hat = model(x_sample, self.rngs())[0]
            ts = jnp.arange(x_sample.shape[0])
            for i in range(n_plots):
                fig = self.visualizer.plot_reconstruction(
                    ts,
                    x_sample[:, i:i+1],
                    x_hat[:, i:i+1],
                    channel_names=['data']
                )
                wandb.log({f"Reconstruction": wandb.Image(fig)})
                plt.close(fig)
        
        return avg_val_loss
     

    
    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for step in tqdm(range(self.config.training_steps)):
            x_history, _, x_future = next(iter(self.train_loader))
            x = jnp.concatenate([x_history, x_future], axis=1)
            beta = self.beta_scheduler(step)
            loss = self.train_step(self.model, self.optimizer, x, beta, self.rngs())
            
            if step % self.config.log_interval == 0:
                val_loss = self.validate(self.model, plot=True)

                metrics = {
                    'ae/train_loss': float(loss),
                    'ae/val_loss': float(val_loss),
                    'ae/beta': float(beta)
                }
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model = deepcopy(self.model)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at step {step}")
                    return self.best_model
                
                wandb.log(metrics, step=step)

        return self.best_model
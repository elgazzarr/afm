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


def normal_ll(predictions: jnp.ndarray, mu: jnp.ndarray, sd: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Gaussian log-likelihood loss for time series data.
    
    Args:
        predictions: Ground truth data of shape (N, T, C)
        mu: Predicted means of shape (N, T, C)
        sd: Standard deviations of shape (N, T, C)
    
    Returns:
        Negative log-likelihood loss (scalar)
    """
    point_nlls = (
        0.5 * jnp.log(2 * jnp.pi) +  
        jnp.log(sd) +                
        0.5 * jnp.square((predictions - mu) / sd)
    )
    
    return jnp.mean(point_nlls)

class FlowTrainer:
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
        self.optimizer = nnx.Optimizer(model, optax.adam(config.learning_rate))
        self.visualizer = Visualizer()
        self.lambda_scheduler = optax.linear_schedule(0.0, 1.0, config.training_steps // 2)
    
    
    def loss_fn(
        self,
        model: nnx.Module,
        history: jnp.ndarray,
        covariates: jnp.ndarray,
        future: jnp.ndarray,
        step: int,
        rngs: nnx.Rngs
    ) -> float:
        # Initial scaling and encoding (once per batch)
        context_scaler = ContextScaler()
        x_past_scaled, scale = jax.vmap(context_scaler.fit_transform)(history)
        x_future_scaled = jax.vmap(context_scaler.transform)(future, scale)
        
        # Encode once outside loop
        z_past_init = jax.vmap(model.encode)(x_past_scaled, covariates[:, :history.shape[1]])
        z_future = jax.vmap(model.encode)(x_future_scaled, covariates[:, history.shape[1]:])
        
        batch_size, window_size = history.shape[:2]
        future_steps = future.shape[1]

        # Initial carry for scan
        init_carry = {
            'z_past': z_past_init,
            'cum_flow_loss': 0.0,
            'rng': rngs(),
        }

        def scan_step(carry, i):
            z_past = carry['z_past']
            rng = carry['rng']
            
            # Split RNGs for this step
            rng, t_rng, flow_rng = jax.random.split(rng, 3)
            
            # Get current context and target
            h = jax.vmap(model.get_context)(z_past)  # No covariates needed
            z_next = z_future[:, i]
            z_next_target = jax.lax.stop_gradient(z_future[:, i])
            c_t = covariates[:, history.shape[1] + i]

            # Flow matching computation
            t = jax.random.uniform(t_rng, (batch_size,))
            z0 = self.model.p0(flow_rng, (batch_size, model.latent_dim))
            z_mid = (1 - t[:, None])*z0 + t[:, None]*z_next_target
            
            pred_flow = jax.vmap(model.predict)(z_mid, h, c_t, t)
            flow_loss = jnp.mean(jnp.square(pred_flow - (z_next_target - z0)))

            # Update context window
            new_z_past = jnp.concatenate([z_past, z_next[:, None]], axis=1)
            new_z_past = new_z_past[:, -window_size:]

            # Weight and accumulate loss
            lambda_ = self.lambda_scheduler(step)
            return {
                'z_past': new_z_past,
                'cum_flow_loss': carry['cum_flow_loss'] + lambda_*flow_loss,
                'rng': rng
            }, None

        # Run scan over all future steps
        final_carry, _ = jax.lax.scan(
            scan_step,
            init_carry,
            jnp.arange(future_steps)
        )

        # Compute reconstruction loss ONCE at the end
        x_recon = jax.vmap(model.decode)(jnp.concatenate([z_past_init, z_future], axis=1))

        pred_loss = optax.losses.squared_error(x_recon, jnp.concatenate([x_past_scaled, x_future_scaled], axis=1)).mean()
        
        # Combine losses
        total_loss = final_carry['cum_flow_loss'] + (1 - self.lambda_scheduler(step)) * pred_loss
        
        return (total_loss, final_carry['cum_flow_loss'], pred_loss)

    @nnx.jit(static_argnums=(0,))
    def train_step(self,
                  model: nnx.Module,
                  optimizer: nnx.Optimizer,
                  history: jnp.ndarray,
                  covariates: jnp.ndarray,
                  prediction: jnp.ndarray,
                  step: int,
                  rngs: nnx.Rngs) -> float:
        loss, grads = nnx.value_and_grad(self.loss_fn)(
            model, history, covariates, prediction,step, rngs
        )
        optimizer.update(grads)
        return loss
    
    def evaluate(self, model, step, plot, n_plots=5):
        keys = jrandom.split(self.rngs(), self.config.mcmc_samples)
        evaluator = FlowEvaluator(model, nfe=self.config.nfe)
        val_loss = 0.0
        for i, (history, covariates, true) in enumerate(self.val_loader):
            val_loss = self.loss_fn(model, history, covariates, true, step, self.rngs)
            predictions = jax.vmap(jax.vmap(evaluator.predict_trajectory_fixed_context, in_axes=(0,None,None,None, None)),in_axes=(None,0,0,None, None))(keys, history, covariates, true.shape[1], history.shape[1])
        val_loss = val_loss / (i + 1)
        wandb.log({'val_loss': val_loss})
        if plot:
            ts = jnp.arange(covariates.shape[1])
            history_scaled = model.scaler.inverse_transform(history[0])
            true_scaled = model.scaler.inverse_transform(true[0])
            predictions_scaled = jax.vmap(model.scaler.inverse_transform)(predictions[0])
            for i in range(n_plots):
                fig = self.visualizer.plot_trajectories(
                    ts,
                    history_scaled[:, i:i+1],
                    predictions_scaled[:, :, i:i+1],
                    true_scaled[:, i:i+1],
                    channel_names=['data'],
                    confidence_interval=0.90
                )
                wandb.log({f"Sample": wandb.Image(plt)})
                plt.close(fig)
            
            context_scaler = ContextScaler()
            history_, scale = context_scaler.fit_transform(history_scaled)
            true_ = context_scaler.transform(true_scaled, scale)
            x_true = jnp.concatenate([history_, true_], axis=1)
            z = model.encode(x_true, covariates[0])
            x_recon = jax.vmap(model.decode)(z)
            x_recon = context_scaler.inverse_transform(model.scaler.inverse_transform(x_recon), scale)
            for i in range(n_plots):
                fig = self.visualizer.plot_reconstruction(
                    ts,
                    x_true[:, i:i+1],
                    x_recon[:, i:i+1],
                    channel_names=['data']
                )
                wandb.log({f"Reconstruction": wandb.Image(plt)})
        
        return val_loss
     

    
    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0  # Track validation checks without improvement
        patience = self.config.patience
        
        for step in tqdm(range(self.config.training_steps)):
            history, covariates, prediction = next(iter(self.train_loader))
            loss, flow_loss, recon_loss = self.train_step(
                self.model,
                self.optimizer,
                history,
                covariates,
                prediction,
                step,
                self.rngs
            )
            
            if step % self.config.log_interval == 0:
                metrics = {'loss': round(loss, 4)}
                wandb.log(metrics, step=step)
                metrics = {'flow_loss': round(flow_loss, 4)}
                wandb.log(metrics, step=step)
                metrics = {'recon_loss': round(recon_loss, 4)}
                wandb.log(metrics, step=step)
                
            if step % self.config.val_interval == 0:
                val_loss,_,_ = self.evaluate(self.model, step, plot=True)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model = deepcopy(self.model)
                    patience_counter = 0  # Reset patience counter
                    wandb.log({'best_val_loss': best_val_loss})
                else:
                    patience_counter += 1
                    
                # Log patience counter for monitoring
                wandb.log({'patience_counter': patience_counter})
                
                # Check if we should stop training
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after {step} steps. Best val_loss: {best_val_loss}')
                    break
                    
        return self.best_model
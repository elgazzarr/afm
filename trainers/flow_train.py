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
from copy import deepcopy
import diffrax as dfx
from diffrax import ODETerm, Tsit5, Euler


class FlowTrainer:
    def __init__(self, 
                 config: Dict[str, Any],
                 flow_model: nnx.Module,
                 autoencoder: nnx.Module,
                 train_loader: Any,
                 val_loader: Any,
                 rngs: nnx.Rngs):
        
        self.config = config
        self.flow_model = flow_model
        self.autoencoder = autoencoder
        self.best_model = None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.rngs = rngs 
        self.optimizer = nnx.Optimizer(self.flow_model, optax.adam(config.learning_rate))
        self.visualizer = Visualizer()
        self.lambda_scheduler = optax.linear_schedule(0.0, 1.0, config.training_steps // 2)
    
    def loss_fn(
    self,
    flow_model: nnx.Module,
    history: jnp.ndarray, 
    covariates: jnp.ndarray,
    future: jnp.ndarray,  
    step: int,
    rngs: nnx.Rngs
    ) -> float:
        # 1. Compute scale ONCE using context window
        context_scaler = flow_model.context_scaler
        scaled_history, scale = jax.vmap(context_scaler.fit_transform)(history)
        scaled_future = jax.vmap(context_scaler.transform)(future, scale)  # (batch, pred_len, dim)

        # 2. Initialize with scaled history
        window_size = scaled_history.shape[1]
        init_carry = {
            'x_past': scaled_history,
            'loss': 0.0,
            'rng': rngs(),
        }

        def scan_step(carry, i):
            x_past = carry['x_past']
            rng = carry['rng']
            rng, t_rng, flow_rng = jax.random.split(rng, 3)
            x_next = scaled_future[:, i]  
            t = jax.random.uniform(t_rng, (x_next.shape[0],), minval=0.0, maxval=1.0)
            x0 = jax.lax.stop_gradient(flow_model.p0.sample(flow_rng, x_past[:, -1]))
            xt = (1 - t[:, None])*x0 + t[:, None]*x_next
            pred_flow = jax.vmap(flow_model)(xt, x_past, covariates, t)
            optimal_flow = x_next - x0
            flow_loss = jnp.mean(jnp.square(pred_flow - optimal_flow))            
            new_x_past = jnp.concatenate([x_past, x_next[:, None]], axis=1)[:, -window_size:]
            
            return {
                'x_past': new_x_past,
                'loss': carry['loss'] + flow_loss,
                'rng': rng
            }, None

        final_carry, _ = jax.lax.scan(scan_step, init_carry, jnp.arange(future.shape[1]))
        return final_carry['loss']
    

    @nnx.jit(static_argnums=(0,))
    def train_step(self,
                  model: nnx.Module,
                  optimizer: nnx.Optimizer,
                  history: jnp.ndarray,
                  covariates: jnp.ndarray,
                  prediction: jnp.ndarray,
                  step: int,
                  rngs: nnx.Rngs) -> float:
        
        loss, grads = nnx.value_and_grad(self.loss_fn)(model, history, covariates, prediction, step, rngs)
        optimizer.update(grads)
        return loss
    
    def evaluate(self, model, step, plot, n_plots=5):
        val_losses = []
        
        for i, (x_history, covariates, x_future) in enumerate(self.val_loader):
            z_history = self.autoencoder.encode(x_history)
            z_future = self.autoencoder.encode(x_future)
            val_loss = self.loss_fn(model, z_history, covariates, z_future, step, self.rngs)
            val_losses.append(val_loss)

        if plot:
            ts = jnp.arange(covariates.shape[1])
            z_predictions = model.sample_flow(z_history, covariates, z_future.shape[1], self.config.nfe, self.config.mcmc_samples, self.rngs())
            x_predictions = jax.vmap(self.autoencoder.decode)(z_predictions)
            for i in range(n_plots):
                fig = self.visualizer.plot_trajectories(
                    ts,
                    z_history[0, :, i:i+1],
                    z_predictions[:, 0, :, i:i+1],
                    z_future[0, :, i:i+1],
                    channel_names=['data'],
                    confidence_interval=0.90
                )
                wandb.log({f"Latent Forecast": wandb.Image(plt)})
                plt.close(fig)

                fig = self.visualizer.plot_trajectories(
                    ts,
                    x_history[0, :, i:i+1],
                    x_predictions[:, 0, :, i:i+1],
                    x_future[0, :, i:i+1],
                    channel_names=['data'],
                    confidence_interval=0.90
                )
                wandb.log({f"Data Forecast": wandb.Image(plt)})
                plt.close(fig)
        
        return float(jnp.mean(jnp.array(val_losses)))
     

    
    def train(self):

        best_val_loss = float('inf')
        patience_counter = 0 

        for step in tqdm(range(self.config.training_steps)):
            x_history, covariates, x_prediction = next(iter(self.train_loader))
            z_history = self.autoencoder.encode(x_history)
            z_prediction = self.autoencoder.encode(x_prediction)

            loss = self.train_step(
                self.flow_model,
                self.optimizer,
                z_history,
                covariates,
                z_prediction,
                step,
                self.rngs
            )
            
            if step % self.config.log_interval == 0:
                val_loss = self.evaluate(self.flow_model, step, plot=True)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model = deepcopy(self.flow_model)
                    patience_counter = 0 
                else:
                    patience_counter += 1
                    
                metrics = {'fm/train_loss': round(loss, 4),
                            'fm/val_loss': round(val_loss, 4),
                            'fm/best_val_loss': round(best_val_loss, 4),
                            'fm/patience_counter': patience_counter}
                
                if patience_counter >= self.config.patience:
                    print(f'\nEarly stopping triggered after {step} steps. Best val_loss: {best_val_loss}')
                    return self.best_model
            
                wandb.log(metrics)
                    
        return self.best_model
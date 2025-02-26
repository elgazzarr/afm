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
        
    
    
    def loss_fn(self,
                flow_model: nnx.Module,
                history: jnp.ndarray, 
                covariates: jnp.ndarray,
                future: jnp.ndarray,  
                rngs: nnx.Rngs) -> float:
        
        context_scaler = flow_model.context_scaler
        scaled_history, scale = jax.vmap(context_scaler.fit_transform)(history)
        scaled_future = jax.vmap(context_scaler.transform)(future, scale) 

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
            x0 = flow_model.p0(flow_rng, x_next.shape)
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
                  rngs: nnx.Rngs) -> float:
        loss, grads = nnx.value_and_grad(self.loss_fn)(model, history, covariates, prediction, rngs)
        optimizer.update(grads)
        return loss
    
    def evaluate(self, model, plot, n_plots=5):
        val_losses = []
        for i, (history, covariates, true) in enumerate(self.val_loader):
            val_loss = self.loss_fn(model, history, covariates, true, self.rngs)
            val_losses.append(val_loss)

        val_loss = val_loss / (i + 1)
        wandb.log({'val_loss': val_loss})

        if plot:
            ts = jnp.arange(covariates.shape[1])
            predictions = model.sample_flow(history, covariates, true.shape[1], self.config.nfe, self.config.mcmc_samples, self.rngs())
            for i in range(n_plots):
                fig = self.visualizer.plot_trajectories(
                    ts,
                    history[0, :, i:i+1],
                    predictions[:, 0, :, i:i+1],
                    true[0, :, i:i+1],
                    channel_names=['data'],
                    confidence_interval=0.95
                )
                wandb.log({f"Sample": wandb.Image(plt)})
                plt.close(fig)
        
        return val_loss
     

    
    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0  # Track validation checks without improvement
        patience = self.config.patience
        
        for step in tqdm(range(self.config.training_steps)):
            history, covariates, prediction = next(iter(self.train_loader))
            loss = self.train_step(self.model,self.optimizer, history, covariates,prediction,self.rngs)
            
            if step % self.config.log_interval == 0:
                metrics = {'loss': round(loss, 4)}
                wandb.log(metrics, step=step)
                val_loss = self.evaluate(self.model, plot=True)
                
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
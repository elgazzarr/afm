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

class FlowTrainer:
    def __init__(self, 
                 config: Dict[str, Any],
                 model: nnx.Module,
                 train_loader: Any,
                 val_loader: Any,
                 rngs: nnx.Rngs):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.rngs = rngs 
        self.optimizer = nnx.Optimizer(
            model, 
            optax.adam(config['training']['learning_rate'])
        )
        self.visualizer = Visualizer()
        
    
    def p0(self, key, shape):
        return jax.random.normal(key, shape)
    
    def loss_fn(self, 
                model: nnx.Module,
                history: jnp.ndarray, 
                covariates: jnp.ndarray, 
                prediction: jnp.ndarray,
                rngs: nnx.Rngs) -> float:
        x_past = history
        loss = 0.0
        
        for i in range(prediction.shape[1]):
            x_next = prediction[:, i]
            t = jax.random.uniform(rngs(), (x_next.shape[0],), minval=0.0, maxval=1.0)
            x0 = self.p0(rngs(), shape=x_next.shape)
            z = (1 - t[:, None]) * x0 + t[:, None] * x_next
            pred_flow = jax.vmap(model)(z, x_past, covariates, t)
            optimal_flow = x_next - x0
            x_past = jnp.concatenate([x_past, x_next[:, None]], axis=1)
            loss += jnp.mean(jnp.square(pred_flow - optimal_flow))
        return loss

    @nnx.jit(static_argnums=(0,))
    def train_step(self,
                  model: nnx.Module,
                  optimizer: nnx.Optimizer,
                  history: jnp.ndarray,
                  covariates: jnp.ndarray,
                  prediction: jnp.ndarray,
                  rngs: nnx.Rngs) -> float:
        loss, grads = nnx.value_and_grad(self.loss_fn)(
            model, history, covariates, prediction, rngs
        )
        optimizer.update(grads)
        return loss
    
    def evaluate(self, model, step, n_plots=5):
        keys = jrandom.split(self.rngs(), self.config.inference.mcmc_samples)
        evaluator = FlowEvaluator(model, nfe=self.config.inference.nfe)
        val_loss = 0.0
        mse = 0.0
        for i, (history, covariates, true) in enumerate(self.val_loader):
            val_loss = self.loss_fn(model, history, covariates, true, self.rngs)
            predictions = jax.vmap(jax.vmap(evaluator.predict_trajectory, in_axes=(0,None,None,None)),in_axes=(None,0,0,None))(keys, history, covariates, true.shape[1])
            mse += jnp.mean(jnp.square(jnp.mean(predictions, axis=1) - true))
        val_loss = val_loss / (i + 1)
        wandb.log({'val_loss': val_loss})
        wandb.log({'val_mse': mse / (i + 1)})
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
            wandb.log({f"Sample {i}": wandb.Image(plt)}, step=step)
            plt.close(fig)
     

    
    def train(self):
        for step in tqdm(range(self.config['training']['num_steps'])):
            history, covariates, prediction = next(iter(self.train_loader))
            loss = self.train_step(
                self.model,
                self.optimizer,
                history,
                covariates,
                prediction,
                self.rngs
            )

            if step % self.config['logging']['wandb']['log_interval'] == 0:
                metrics = {'loss': round(loss, 4)}
                wandb.log(metrics, step=step)

            if  step % self.config['training']['val_interval'] == 0:
                self.evaluate(self.model, step)
                
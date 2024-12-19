import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from typing import Dict, Any
import wandb
from tqdm import tqdm

class FlowTrainer:
    def __init__(self, 
                 config: Dict[str, Any],
                 model: nnx.Module,
                 dataloader: Any,
                 rngs: nnx.Rngs):
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.rngs = rngs 
        self.optimizer = nnx.Optimizer(
            model, 
            optax.adam(config['training']['learning_rate'])
        )
        
    
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
    
    def train(self):
        losses = []

        for epoch in tqdm(range(self.config['training']['num_epochs'])):
            epoch_loss = 0.0
            
            for batch_idx, (history, covariates, prediction) in enumerate(self.dataloader):
                loss = self.train_step(
                    self.model,
                    self.optimizer,
                    history,
                    covariates,
                    prediction,
                    self.rngs
                )
                epoch_loss += loss
            
            avg_epoch_loss = epoch_loss / (batch_idx + 1)
            losses.append(float(avg_epoch_loss))
            
            
            if epoch % self.config['logging']['wandb']['log_interval'] == 0:
                metrics = {
                    'loss': float(avg_epoch_loss),
                }
                wandb.log(metrics, step=epoch)
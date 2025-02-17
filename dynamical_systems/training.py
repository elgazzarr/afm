import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax.nnx as nnx
import optax
import diffrax as dfx
from typing import Tuple, Callable
import jax_dataloader as jdl
from dataclasses import dataclass
from tqdm import tqdm
from functools import partial
from seq_priors import generate_brownian_timeseries


class BaseTrainer:
    def __init__(self, model: nnx.Module, learning_rate: float):
        self.model = model
        self.optimizer = nnx.Optimizer(model, optax.adam(learning_rate))

    @nnx.jit(static_argnums=(0,))
    def train_step(self, model, optimizer, history, covariates, prediction, rng):
        loss, grads = nnx.value_and_grad(self.loss_fn)(model,
            rng,
            history,
            covariates,
            prediction
        )
        optimizer.update(grads)
        return loss

    @nnx.jit(static_argnums=(0,))
    def eval_step(self, model, history, covariates, prediction, rng):
        return self.loss_fn(
            model,
            rng,
            history,
            covariates,
            prediction
        )

    def train(
        self,
        train_dataloader: jdl.DataLoader,
        test_dataloader: jdl.DataLoader,
        num_epochs: int = 100,
        seed: int = 0
    ) -> Tuple[list, list]:

        train_losses = []
        test_losses = []

        rng = jrandom.PRNGKey(seed)

        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0.0
            for batch_idx, (history, covariates, prediction, _) in enumerate(train_dataloader):
                rng, step_rng = jrandom.split(rng)
                loss = self.train_step(self.model, self.optimizer, history, covariates, prediction, step_rng)
                epoch_loss += loss

            avg_train_loss = epoch_loss / (batch_idx + 1)
            train_losses.append(float(avg_train_loss))

            # Evaluation
            epoch_loss = 0.0
            for batch_idx, (history, covariates, prediction, _) in enumerate(test_dataloader):
                rng, step_rng = jrandom.split(rng)
                loss = self.eval_step(self.model, history, covariates, prediction, step_rng)
                epoch_loss += loss

            avg_test_loss = epoch_loss / (batch_idx + 1)
            test_losses.append(float(avg_test_loss))

        return train_losses, test_losses


class ARTrainer(BaseTrainer):
    def loss_fn(self, model, rng, history, covariates, prediction):
        """
        Autoregressive loss function. 
        'model' here is optimizer.target (parameters) passed in from train_step or eval_step.
        """
        x_past = history
        window_size = history.shape[1]
        batch_size = history.shape[0]
        pred_steps = prediction.shape[1]
        
        def body_fun(i, val):
            rng_step, x_past, loss = val
            x_next = prediction[:, i]

            rng_step, rng_t, rng_x0 = jrandom.split(rng_step, 3)
            t = jrandom.uniform(rng_t, (batch_size,), minval=0.0, maxval=1.0)
            x0 = jrandom.normal(rng_x0, shape=x_next.shape)
            z = (1 - t[:, None]) * x0 + t[:, None] * x_next

            # Predict flow
            pred_flow = jax.vmap(model)(z, x_past, covariates, t)
            # Optimal flow = x_next - x0
            optimal_flow = x_next - x0

            # Update x_past (autoregressive)
            x_past = jnp.concatenate([x_past, x_next[:, None]], axis=1)
            x_past = x_past[:, -window_size:]

            loss += jnp.mean(jnp.square(pred_flow - optimal_flow))
            return (rng_step, x_past, loss)

        rng_keys = jrandom.split(rng, pred_steps + 1)[:-1]
        _, _, total_loss = jax.lax.fori_loop(
            0, pred_steps, body_fun, (rng_keys[0], x_past, 0.0)
        )
        return total_loss


class NoARTrainer(BaseTrainer):
    def loss_fn(self, model, rng, history, covariates, prediction):
        """
        Non-autoregressive loss function.
        """
        x_past = history
        x1 = prediction
        batch_size = history.shape[0]

        rng_t, rng_x0 = jrandom.split(rng)
        t = jrandom.uniform(rng_t, (batch_size,), minval=0.0, maxval=1.0)

        # Generate some sample x0 using a Brownian timeseries prior
        x0_keys = jrandom.split(rng_x0, x1.shape[0])
        x0 = jax.vmap(generate_brownian_timeseries, in_axes=(0, None, None))(
            x0_keys, x1.shape[2], x1.shape[1]
        )
        # x0 has shape (batch_size, length, dim), so we transpose to match x1
        x0 = jnp.transpose(x0, (0, 2, 1))

        # Interpolate between x0 and x1
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1

        # Model predicts the flow for each sample
        pred_flow = jax.vmap(model)(x_t, x_past, covariates, t)
        optimal_flow = x1 - x0

        return jnp.mean(jnp.square(pred_flow - optimal_flow))


def prepare_data(system, config, key):
    """Prepare train and test dataloaders."""
    solve_single = lambda x0: dfx.diffeqsolve(
        system.vector_field, system.solver, 
        config.ts[0], config.ts[-1], config.dt, 
        x0, args=system.params, 
        saveat=dfx.SaveAt(ts=config.ts)).ys
    
    dataloaders = []
    extrapolation_size = config.extrapolation_size
    for i in range(2):  # Train and test
        key, subkey = jrandom.split(key)
        x0 = system.initial_state(subkey, config.n_train_test_samples)
        trajectories = jax.vmap(solve_single)(x0)
        
        # Split trajectories
        history = trajectories[:, :config.history_size, :]
        prediction = trajectories[:, config.history_size:config.history_size + config.prediction_size, :]
        extrapolation = trajectories[:, config.history_size + config.prediction_size:, :]
        covariates = jnp.repeat(
            config.ts[None, :, None], 
            config.n_train_test_samples, 
            axis=0
        )
        
        dataloader = jdl.DataLoader(
            jdl.ArrayDataset(history, covariates, prediction, extrapolation),
            'jax',
            batch_size=config.batch_size,
            shuffle=True
        )
        dataloaders.append((dataloader, trajectories))
    
    return dataloaders[0], dataloaders[1]  # (train_data, train_traj), (test_data, test_traj)
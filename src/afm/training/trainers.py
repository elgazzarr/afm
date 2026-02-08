"""Training classes for AFM and FM models with enhanced wandb integration."""

from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Optional, Dict
import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax.nnx as nnx
import optax
import jax_dataloader as jdl
from tqdm import tqdm
import wandb

from ..utils.priors import generate_brownian_timeseries


class BaseTrainer(ABC):
    """Base trainer class for flow matching models."""

    def __init__(
        self,
        model: nnx.Module,
        learning_rate: float = 1e-3,
        model_name: str = "model",
        log_gradients: bool = False,
        log_frequency: int = 1
    ):
        """Initialize base trainer.

        Args:
            model: Model to train
            learning_rate: Learning rate for optimization
            model_name: Name for wandb logging (AFM/FM)
            log_gradients: Whether to log gradient norms
            log_frequency: How often to log metrics (every N epochs)
        """
        self.model = model
        self.optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
        self.model_name = model_name
        self.log_gradients = log_gradients
        self.log_frequency = log_frequency
        self.learning_rate = learning_rate

    @nnx.jit(static_argnums=(0,))
    def train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        history: jnp.ndarray,
        covariates: jnp.ndarray,
        prediction: jnp.ndarray,
        rng: jax.random.PRNGKey
    ) -> float:
        """Perform a single training step.

        Args:
            model: Model being trained
            optimizer: Optimizer instance
            history: Historical sequence data
            covariates: Covariate information
            prediction: Target prediction data
            rng: Random key

        Returns:
            Training loss for this step
        """
        loss, grads = nnx.value_and_grad(self.loss_fn)(
            model, rng, history, covariates, prediction
        )
        optimizer.update(grads)
        return loss

    @nnx.jit(static_argnums=(0,))
    def eval_step(
        self,
        model: nnx.Module,
        history: jnp.ndarray,
        covariates: jnp.ndarray,
        prediction: jnp.ndarray,
        rng: jax.random.PRNGKey
    ) -> float:
        """Perform evaluation step.

        Args:
            model: Model being evaluated
            history: Historical sequence data
            covariates: Covariate information
            prediction: Target prediction data
            rng: Random key

        Returns:
            Evaluation loss
        """
        return self.loss_fn(model, rng, history, covariates, prediction)

    def train(
        self,
        train_dataloader: jdl.DataLoader,
        test_dataloader: jdl.DataLoader,
        num_epochs: int = 100,
        seed: int = 0,
        use_wandb: bool = True
    ) -> Tuple[List[float], List[float]]:
        """Train the model with enhanced wandb logging.

        Args:
            train_dataloader: Training data loader
            test_dataloader: Test data loader
            num_epochs: Number of training epochs
            seed: Random seed for training
            use_wandb: Whether to log to wandb

        Returns:
            Tuple of (train_losses, test_losses)
        """
        train_losses = []
        test_losses = []

        rng = jrandom.PRNGKey(seed)

        for epoch in tqdm(range(num_epochs), desc=f"Training {self.model_name}"):
            # Training loop
            epoch_loss = 0.0
            batch_count = 0
            batch_losses = []

            for history, covariates, prediction in train_dataloader:
                rng, step_rng = jrandom.split(rng)
                loss = self.train_step(
                    self.model, self.optimizer, history, covariates, prediction, step_rng
                )
                epoch_loss += loss
                batch_count += 1
                batch_losses.append(float(loss))

            avg_train_loss = epoch_loss / batch_count
            train_losses.append(float(avg_train_loss))

            # Evaluation loop
            epoch_loss = 0.0
            batch_count = 0

            for history, covariates, prediction in test_dataloader:
                rng, step_rng = jrandom.split(rng)
                loss = self.eval_step(
                    self.model, history, covariates, prediction, step_rng
                )
                epoch_loss += loss
                batch_count += 1

            avg_test_loss = epoch_loss / batch_count
            test_losses.append(float(avg_test_loss))

            # Enhanced wandb logging
            if use_wandb and wandb.run is not None and (epoch + 1) % self.log_frequency == 0:
                wandb_metrics = {
                    f"{self.model_name}/train_loss": avg_train_loss,
                    f"{self.model_name}/val_loss": avg_test_loss,
                }

                # Log gradient norms if requested
                if self.log_gradients:
                    grad_norms = self._compute_gradient_norms()
                    if grad_norms:
                        for param_name, grad_norm in grad_norms.items():
                            wandb_metrics[f"{self.model_name}/grad_norm/{param_name}"] = grad_norm

                wandb.log(wandb_metrics)

        # Log final training curves
        if use_wandb and wandb.run is not None:
            # Create training curve plot
            epochs_range = list(range(1, len(train_losses) + 1))

            # Log as wandb plot
            data = [[epoch, train_loss, test_loss] for epoch, train_loss, test_loss
                   in zip(epochs_range, train_losses, test_losses)]

            table = wandb.Table(data=data, columns=["epoch", "train_loss", "val_loss"])
            wandb.log({
                f"{self.model_name}/training_curves": wandb.plot.line(
                    table, "epoch", ["train_loss", "val_loss"],
                    title=f"{self.model_name} Training Curves"
                )
            })

        return train_losses, test_losses

    def _compute_gradient_norms(self) -> Optional[Dict[str, float]]:
        """Compute gradient norms for monitoring.

        Returns:
            Dictionary of parameter names to gradient norms
        """
        try:
            grad_norms = {}
            # Get the current gradients from the optimizer
            if hasattr(self.optimizer, 'opt_state') and self.optimizer.opt_state is not None:
                # This is a simplified version - actual implementation would depend on
                # the specific optimizer state structure
                for param_name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = float(jnp.linalg.norm(param.grad.flatten()))
                        grad_norms[param_name.replace('/', '_')] = grad_norm
            return grad_norms
        except Exception:
            # If gradient norm computation fails, return None
            return None

    @abstractmethod
    def loss_fn(
        self,
        model: nnx.Module,
        rng: jax.random.PRNGKey,
        history: jnp.ndarray,
        covariates: jnp.ndarray,
        prediction: jnp.ndarray
    ) -> float:
        """Compute loss for the model.

        Args:
            model: Model instance
            rng: Random key
            history: Historical sequence data
            covariates: Covariate information
            prediction: Target prediction data

        Returns:
            Loss value
        """
        pass


class AFMTrainer(BaseTrainer):
    """Trainer for Autoregressive Flow Matching models with efficient parallel training."""

    def loss_fn(
        self,
        model: nnx.Module,
        rng: jax.random.PRNGKey,
        history: jnp.ndarray,
        covariates: jnp.ndarray,
        prediction: jnp.ndarray
    ) -> float:
        """Efficient single-step parallel AFM loss function.

        This approach enables fully parallel training for AFM by training on
        single-step (context -> next_state) pairs sampled from the data,
        rather than sequential autoregressive prediction during training.

        Args:
            model: AFM model instance
            rng: Random key
            history: Historical context (batch_size, context_length, state_dim)
            covariates: Context covariates (batch_size, context_length, control_dim)
            prediction: Single next state (batch_size, 1, state_dim) or (batch_size, state_dim)

        Returns:
            Single-step flow matching loss
        """
        batch_size = history.shape[0]

        # Handle both (batch, 1, dim) and (batch, dim) prediction shapes
        if prediction.ndim == 3 and prediction.shape[1] == 1:
            x_next = prediction[:, 0]  # (batch_size, state_dim)
        elif prediction.ndim == 2:
            x_next = prediction  # (batch_size, state_dim)
        else:
            raise ValueError(f"Invalid prediction shape: {prediction.shape}. Expected (batch, state_dim) or (batch, 1, state_dim)")

        # Generate flow matching training data
        rng_t, rng_x0 = jrandom.split(rng)
        t = jrandom.uniform(rng_t, (batch_size,), minval=0.0, maxval=1.0)
        x0 = jrandom.normal(rng_x0, shape=x_next.shape)

        # Interpolate between x0 and x_next
        z = (1 - t[:, None]) * x0 + t[:, None] * x_next

        # Predict flow using model
        pred_flow = jax.vmap(model)(z, history, covariates, t)

        # Optimal flow = x_next - x0
        optimal_flow = x_next - x0

        return jnp.mean(jnp.square(pred_flow - optimal_flow))


class FMTrainer(BaseTrainer):
    """Trainer for non-autoregressive Flow Matching models."""

    def loss_fn(
        self,
        model: nnx.Module,
        rng: jax.random.PRNGKey,
        history: jnp.ndarray,
        covariates: jnp.ndarray,
        prediction: jnp.ndarray
    ) -> float:
        """Non-autoregressive flow matching loss function.

        Trains the model to predict all future states simultaneously
        using parallel processing. Works with both old fixed-split data
        and new dynamic context window data.

        Args:
            model: FM model instance
            rng: Random key
            history: Historical context (batch_size, context_length, state_dim)
            covariates: Context covariates (batch_size, context_length, control_dim)
            prediction: Target future sequence (batch_size, prediction_length, state_dim)

        Returns:
            Flow matching loss
        """
        x_past = history
        x1 = prediction
        batch_size = history.shape[0]

        rng_t, rng_x0 = jrandom.split(rng)
        t = jrandom.uniform(rng_t, (batch_size,), minval=0.0, maxval=1.0)

        # Generate sample x0 using Brownian timeseries prior
        x0_keys = jrandom.split(rng_x0, x1.shape[0])
        x0 = jax.vmap(generate_brownian_timeseries, in_axes=(0, None, None))(
            x0_keys, x1.shape[2], x1.shape[1]
        )
        # x0 has shape (batch_size, length, dim), transpose to match x1
        x0 = jnp.transpose(x0, (0, 2, 1))

        # Interpolate between x0 and x1
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1

        # Model predicts the flow for each sample
        pred_flow = jax.vmap(model)(x_t, x_past, covariates, t)
        optimal_flow = x1 - x0

        return jnp.mean(jnp.square(pred_flow - optimal_flow))

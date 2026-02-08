"""Infinite dataloader with dynamic context window sampling for efficient AFM training."""

import jax
import jax.numpy as jnp
from typing import Tuple, Iterator, Union
import jax.random as jrandom


def create_infinite_dataloader(
    trajectories: jnp.ndarray,
    control: jnp.ndarray,
    context_length: int,
    prediction_length: int = 1,
    batch_size: int = 32,
    seed: int = 0
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Create an infinite dataloader that yields batches directly from trajectory data.

    This dataloader randomly samples context windows from trajectories, enabling
    much more efficient training by:
    1. Providing many more training samples from the same data
    2. Allowing parallel training for AFM (no sequential dependencies)
    3. Supporting dynamic context lengths

    Args:
        trajectories: Array of shape (n_samples, n_timesteps, state_dim)
        control: Array of shape (n_samples, n_timesteps, control_dim)
        context_length: Number of time steps to use as context window
        prediction_length: Number of future steps to predict (1 for AFM, >1 for FM)
        batch_size: Batch size for training
        seed: Random seed

    Returns:
        Iterator yielding (x_prev, c_prev, x_next) batches where:
        - x_prev: (batch_size, context_length, state_dim) - context history
        - c_prev: (batch_size, context_length, control_dim) - context covariates
        - x_next: (batch_size, prediction_length, state_dim) - prediction targets
    """
    n_samples, n_timesteps, state_dim = trajectories.shape
    control_dim = control.shape[-1]

    # Ensure we have enough timesteps for context + prediction
    max_start_idx = n_timesteps - context_length - prediction_length
    if max_start_idx <= 0:
        raise ValueError(
            f"Not enough timesteps ({n_timesteps}) for context_length ({context_length}) "
            f"+ prediction_length ({prediction_length})"
        )

    def get_batch(rng: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate a random batch of (x_prev, c_prev, x_next) examples."""
        rng_traj, rng_start = jax.random.split(rng)

        # Sample random trajectories
        traj_indices = jax.random.randint(
            rng_traj,
            shape=(batch_size,),
            minval=0,
            maxval=n_samples
        )

        # Sample random start positions within valid range
        start_indices = jax.random.randint(
            rng_start,
            shape=(batch_size,),
            minval=0,
            maxval=max_start_idx + 1
        )

        def get_single_sample(traj_idx: int, start_idx: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Extract context and prediction for a single sample."""
            # Extract context window
            x_context = jax.lax.dynamic_slice(
                trajectories,
                (traj_idx, start_idx, 0),
                (1, context_length, state_dim)
            )
            x_context = jnp.squeeze(x_context, axis=0)

            c_context = jax.lax.dynamic_slice(
                control,
                (traj_idx, start_idx, 0),
                (1, context_length, control_dim)
            )
            c_context = jnp.squeeze(c_context, axis=0)

            # Extract prediction target
            pred_start_idx = start_idx + context_length
            x_next = jax.lax.dynamic_slice(
                trajectories,
                (traj_idx, pred_start_idx, 0),
                (1, prediction_length, state_dim)
            )
            x_next = jnp.squeeze(x_next, axis=0)

            return x_context, c_context, x_next

        # Vectorize over the batch
        get_batch_fn = jax.vmap(get_single_sample)
        x_prev, c_prev, x_next = get_batch_fn(traj_indices, start_indices)

        return x_prev, c_prev, x_next

    def data_generator():
        """Infinite generator yielding batches."""
        rng = jax.random.PRNGKey(seed)
        while True:
            rng, sample_rng = jax.random.split(rng)
            yield get_batch(sample_rng)

    return data_generator()


def create_afm_dataloader(
    trajectories: jnp.ndarray,
    control: jnp.ndarray,
    context_length: int,
    batch_size: int = 32,
    seed: int = 0
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Create dataloader specifically for AFM training (single-step prediction).

    Args:
        trajectories: Full trajectory data
        control: Control/covariate data
        context_length: Length of context window
        batch_size: Batch size
        seed: Random seed

    Returns:
        Iterator yielding (x_prev, c_prev, x_next) where x_next is single timestep
    """
    return create_infinite_dataloader(
        trajectories=trajectories,
        control=control,
        context_length=context_length,
        prediction_length=1,  # AFM predicts one step at a time
        batch_size=batch_size,
        seed=seed
    )


def create_fm_dataloader(
    trajectories: jnp.ndarray,
    control: jnp.ndarray,
    context_length: int,
    prediction_length: int,
    batch_size: int = 32,
    seed: int = 0
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Create dataloader specifically for FM training (multi-step prediction).

    Args:
        trajectories: Full trajectory data
        control: Control/covariate data
        context_length: Length of context window
        prediction_length: Number of future steps to predict
        batch_size: Batch size
        seed: Random seed

    Returns:
        Iterator yielding (x_prev, c_prev, x_future) where x_future is multi-step
    """
    return create_infinite_dataloader(
        trajectories=trajectories,
        control=control,
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        seed=seed
    )


class InfiniteDataloaderWrapper:
    """Wrapper to make infinite dataloader compatible with existing training loops."""

    def __init__(self, generator: Iterator, steps_per_epoch: int):
        """
        Initialize wrapper.

        Args:
            generator: Infinite data generator
            steps_per_epoch: Number of steps to take per epoch
        """
        self.generator = generator
        self.steps_per_epoch = steps_per_epoch
        self._step_count = 0

    def __iter__(self):
        """Reset step count and return self."""
        self._step_count = 0
        return self

    def __next__(self):
        """Get next batch, stopping after steps_per_epoch."""
        if self._step_count >= self.steps_per_epoch:
            raise StopIteration

        self._step_count += 1
        return next(self.generator)


def create_wrapped_infinite_dataloader(
    trajectories: jnp.ndarray,
    control: jnp.ndarray,
    context_length: int,
    prediction_length: int = 1,
    batch_size: int = 32,
    steps_per_epoch: int = 100,
    seed: int = 0
) -> InfiniteDataloaderWrapper:
    """
    Create wrapped infinite dataloader compatible with existing training code.

    Args:
        trajectories: Full trajectory data
        control: Control/covariate data
        context_length: Context window length
        prediction_length: Prediction horizon
        batch_size: Batch size
        steps_per_epoch: Steps per training epoch
        seed: Random seed

    Returns:
        Wrapped dataloader that stops after steps_per_epoch
    """
    generator = create_infinite_dataloader(
        trajectories=trajectories,
        control=control,
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        seed=seed
    )

    return InfiniteDataloaderWrapper(generator, steps_per_epoch)

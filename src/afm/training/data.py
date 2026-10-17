"""Data preparation utilities for training flow matching models."""

from typing import Tuple, Any
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax_dataloader as jdl
import diffrax as dfx

from ..systems.systems import DynamicalSystem
from ..config import Config
from .dataloader import (
    create_afm_dataloader,
    create_fm_dataloader,
    create_wrapped_infinite_dataloader
)


def create_evaluation_dataloader(
    test_trajectories: jnp.ndarray,
    covariates: jnp.ndarray,
    config: Config,
    batch_size: int = None
) -> Any:
    """Create a deterministic evaluation loader for held-out test trajectories.

    Unlike the infinite training loaders (which yield randomly sampled
    (context, next-step) windows as 3-tuples), this loader yields fixed
    (history, covariates, prediction, extrapolation) 4-tuple windows expected
    by ``ModelEvaluator``: for each test trajectory, the first
    ``context_length`` steps form the history, the next
    ``fm_prediction_horizon`` steps the prediction target, and the remaining
    ``extrapolation_size`` steps the extrapolation target.

    Args:
        test_trajectories: Held-out trajectories (n_test, n_timepoints, state_dim)
        covariates: Time grid (n_timepoints,)
        config: Model configuration
        batch_size: Batch size (defaults to ``config.data_batch_size``)

    Returns:
        Iterator yielding (history, covariates, prediction, extrapolation)
        batches of shapes (B, context, dim), (B, context, 1),
        (B, fm_prediction_horizon, dim), (B, extrapolation_size, dim).
    """
    if batch_size is None:
        batch_size = config.data_batch_size

    context_length = config.context_length
    pred_length = config.fm_prediction_horizon
    extrap_length = config.extrapolation_size
    window_length = context_length + pred_length + extrap_length

    n_traj, n_timepoints = test_trajectories.shape[0], test_trajectories.shape[1]
    if n_timepoints < window_length:
        raise ValueError(
            f"Trajectories have {n_timepoints} timepoints, but evaluation needs "
            f"context ({context_length}) + prediction ({pred_length}) + "
            f"extrapolation ({extrap_length}) = {window_length}"
        )

    # Expand the shared time grid to (n_traj, n_timepoints, 1)
    traj_covariates = jnp.repeat(covariates[None, :, None], n_traj, axis=0)

    class _EvalLoader:
        """Re-iterable evaluation loader (a fresh pass over the test set on
        every ``iter()`` call), so several models can be evaluated on the
        same windows in sequence."""

        def __iter__(self):
            def batches():
                for start in range(0, n_traj, batch_size):
                    traj = test_trajectories[start:start + batch_size]
                    cov = traj_covariates[start:start + batch_size]
                    yield (
                        traj[:, :context_length],
                        cov[:, :context_length],
                        traj[:, context_length:context_length + pred_length],
                        traj[:, context_length + pred_length:window_length],
                    )
            return batches()

    return _EvalLoader()


def prepare_trajectories(
    system: DynamicalSystem,
    config: Config,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate trajectory data for train and test sets.

    Args:
        system: Dynamical system to generate data from
        config: Model configuration containing data parameters
        key: Random key for data generation

    Returns:
        Tuple containing (train_trajectories, test_trajectories, covariates)
    """
    def solve_single(x0: jnp.ndarray) -> jnp.ndarray:
        """Solve a single trajectory from initial condition."""
        return dfx.diffeqsolve(
            system.vector_field,
            system.solver,
            config.ts[0],
            config.ts[-1],
            config.dt,
            x0,
            args=system.params,
            saveat=dfx.SaveAt(ts=config.ts)
        ).ys

    trajectories = []

    for i in range(2):  # Train and test
        key, subkey = jrandom.split(key)

        # Determine number of samples
        n_samples = (
            config.data_n_train_test_samples if i == 0
            else int(0.2 * config.data_n_train_test_samples)
        )

        # Generate initial conditions and solve trajectories
        x0 = system.initial_state(subkey, n_samples)
        traj = jax.vmap(solve_single)(x0)
        trajectories.append(traj)

    # Create covariate information (time stamps) for all time points
    total_time_points = config.data_history_size + config.data_prediction_size + config.extrapolation_size
    relevant_ts = config.ts[:total_time_points]

    return trajectories[0], trajectories[1], relevant_ts


def create_infinite_dataloaders(
    train_trajectories: jnp.ndarray,
    test_trajectories: jnp.ndarray,
    covariates: jnp.ndarray,
    config: Config
) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
    """Create infinite dataloaders for AFM and FM training.

    Args:
        train_trajectories: Training trajectory data (n_train, n_timepoints, state_dim)
        test_trajectories: Test trajectory data (n_test, n_timepoints, state_dim)
        covariates: Time stamp covariates (n_timepoints,)
        config: Model configuration containing infinite dataloader parameters

    Returns:
        Tuple containing ((afm_train_loader, afm_test_loader), (fm_train_loader, fm_test_loader))
    """

    # Expand covariates to match trajectory format
    # covariates shape: (n_timepoints,) -> (n_samples, n_timepoints, 1)
    n_train = train_trajectories.shape[0]
    n_test = test_trajectories.shape[0]

    train_covariates = jnp.repeat(
        covariates[None, :, None],
        n_train,
        axis=0
    )
    test_covariates = jnp.repeat(
        covariates[None, :, None],
        n_test,
        axis=0
    )

    # Create AFM dataloaders (single-step prediction)
    afm_train_loader = create_wrapped_infinite_dataloader(
        trajectories=train_trajectories,
        control=train_covariates,
        context_length=config.context_length,
        prediction_length=config.afm_prediction_horizon,
        batch_size=config.data_batch_size,
        steps_per_epoch=config.steps_per_epoch,
        seed=0
    )

    afm_test_loader = create_wrapped_infinite_dataloader(
        trajectories=test_trajectories,
        control=test_covariates,
        context_length=config.context_length,
        prediction_length=config.afm_prediction_horizon,
        batch_size=config.data_batch_size,
        steps_per_epoch=config.steps_per_epoch // 4,  # Fewer steps for validation
        seed=1
    )

    # Create FM dataloaders (multi-step prediction)
    fm_train_loader = create_wrapped_infinite_dataloader(
        trajectories=train_trajectories,
        control=train_covariates,
        context_length=config.context_length,
        prediction_length=config.fm_prediction_horizon,
        batch_size=config.data_batch_size,
        steps_per_epoch=config.steps_per_epoch,
        seed=2
    )

    fm_test_loader = create_wrapped_infinite_dataloader(
        trajectories=test_trajectories,
        control=test_covariates,
        context_length=config.context_length,
        prediction_length=config.fm_prediction_horizon,
        batch_size=config.data_batch_size,
        steps_per_epoch=config.steps_per_epoch // 4,  # Fewer steps for validation
        seed=3
    )

    return (afm_train_loader, afm_test_loader), (fm_train_loader, fm_test_loader)


def prepare_data(
    system: DynamicalSystem,
    config: Config,
    key: jax.random.PRNGKey
) -> Tuple[Any, Any]:
    """Prepare data using infinite dataloaders with dynamic context sampling.

    Args:
        system: Dynamical system to generate data from
        config: Model configuration containing data parameters
        key: Random key for data generation

    Returns:
        Tuple containing ((afm_train, afm_test), (fm_train, fm_test))
    """
    # Generate trajectory data
    train_trajectories, test_trajectories, covariates = prepare_trajectories(system, config, key)

    # Use infinite dataloaders with dynamic context sampling
    return create_infinite_dataloaders(train_trajectories, test_trajectories, covariates, config)

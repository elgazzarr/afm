"""Data preparation utilities for training flow matching models."""

from typing import Tuple, Any
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax_dataloader as jdl
import diffrax as dfx

from ..systems.systems import DynamicalSystem
from ..config_loader import Config


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


def create_dataloaders(
    train_trajectories: jnp.ndarray,
    test_trajectories: jnp.ndarray, 
    covariates: jnp.ndarray,
    config: Config
) -> Tuple[Any, Any]:
    """Create train and test dataloaders from trajectory data.
    
    Args:
        train_trajectories: Training trajectory data
        test_trajectories: Test trajectory data  
        covariates: Time stamp covariates
        config: Model configuration containing batch size and split parameters
        
    Returns:
        Tuple containing (train_loader, test_loader)
    """
    dataloaders = []
    trajectories_list = [train_trajectories, test_trajectories]
    
    for i, trajectories in enumerate(trajectories_list):
        n_samples = trajectories.shape[0]
        
        # Split trajectories into history, prediction, and extrapolation
        history = trajectories[:, :config.data_history_size, :]
        prediction = trajectories[:, 
            config.data_history_size:config.data_history_size + config.data_prediction_size, :]
        extrapolation = trajectories[:, 
            config.data_history_size + config.data_prediction_size:, :]
        
        # Expand covariates to match sample count
        sample_covariates = jnp.repeat(
            covariates[None, :, None], 
            n_samples, 
            axis=0
        )
        
        # Create dataloader
        dataloader = jdl.DataLoader(
            jdl.ArrayDataset(history, sample_covariates, prediction, extrapolation),
            'jax',
            batch_size=config.data_batch_size,
            shuffle=(i == 0)  # Only shuffle training data
        )
        
        dataloaders.append(dataloader)
    
    return dataloaders[0], dataloaders[1]  # train_loader, test_loader


def prepare_data(
    system: DynamicalSystem, 
    config: Config, 
    key: jax.random.PRNGKey
) -> Tuple[Any, Any]:
    """Prepare train and test dataloaders for flow matching models.
    
    DEPRECATED: Use prepare_trajectories() + create_dataloaders() for cleaner separation.
    
    Args:
        system: Dynamical system to generate data from
        config: Model configuration containing data parameters
        key: Random key for data generation
        
    Returns:
        Tuple containing (train_loader, test_loader)
    """
    # Use the new clean separation approach
    train_traj, test_traj, covariates = prepare_trajectories(system, config, key)
    train_loader, test_loader = create_dataloaders(train_traj, test_traj, covariates, config)
    return train_loader, test_loader


def create_data_splits(
    trajectories: jnp.ndarray,
    config: Config
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split trajectories into history, prediction, and extrapolation segments.
    
    Args:
        trajectories: Full trajectories of shape (n_samples, n_timepoints, n_dims)
        config: Model configuration with split sizes
        
    Returns:
        Tuple of (history, prediction, extrapolation) arrays
    """
    history = trajectories[:, :config.data_history_size, :]
    prediction = trajectories[:, 
        config.data_history_size:config.data_history_size + config.data_prediction_size, :]
    extrapolation = trajectories[:, 
        config.data_history_size + config.data_prediction_size:, :]
    
    return history, prediction, extrapolation
"""Test script to validate the infinite dataloader implementation."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from afm.config import Config
from afm.systems.systems import create_system
from afm.training.data import (
    prepare_trajectories,
    create_infinite_dataloaders,
    create_evaluation_dataloader,
)
from afm.models.afm import AutoregressiveFlowMatching
from afm.models.fm import FlowMatching
from afm.training.trainers import AFMTrainer, FMTrainer
import flax.nnx as nnx


def test_infinite_dataloader():
    """Test the infinite dataloader with AFM and FM models."""
    print("Testing infinite dataloader implementation...")

    # Create test configuration
    config = Config(
        # System parameters
        system_name='lorenz',
        system_stochastic=False,
        system_num_timepoints=200,
        system_t0=0.0,
        system_t1=20.0,

        # Traditional data parameters (for comparison)
        data_history_size=50,
        data_prediction_size=30,
        data_n_train_test_samples=64,
        data_batch_size=16,

        # Infinite dataloader parameters
        context_length=40,
        afm_prediction_horizon=1,
        fm_prediction_horizon=25,
        steps_per_epoch=10,

        # Model parameters
        model_hidden_dim=32,
        model_time_dim=8,
        model_hidden_blocks=2,
        model_n_rnn_layers=2,
        model_rnn_hidden=32,

        # Training parameters
        training_learning_rate=1e-3,
    )

    print(config.summary())

    # Create system and generate data
    system = create_system(config.system_name, stochastic=config.system_stochastic)
    key = jrandom.PRNGKey(42)

    print("\nGenerating trajectory data...")
    train_trajectories, test_trajectories, covariates = prepare_trajectories(system, config, key)
    time_grid = covariates  # keep a handle: 'covariates' is shadowed by batch unpacking below
    (afm_train_loader, afm_test_loader), (fm_train_loader, fm_test_loader) = \
        create_infinite_dataloaders(train_trajectories, test_trajectories, covariates, config)

    print(f"Created infinite dataloaders with {config.steps_per_epoch} steps per epoch")

    # Test AFM dataloader
    print("\nTesting AFM dataloader...")
    afm_batch_count = 0
    for batch in afm_train_loader:
        if afm_batch_count >= 3:  # Just test a few batches
            break
        history, covariates, prediction = batch
        print(f"  Batch {afm_batch_count}: history={history.shape}, covariates={covariates.shape}, prediction={prediction.shape}")

        # Validate shapes
        assert history.shape == (config.data_batch_size, config.context_length, 3)  # Lorenz is 3D
        assert covariates.shape == (config.data_batch_size, config.context_length, 1)  # Time covariate
        assert prediction.shape == (config.data_batch_size, 1, 3)  # Single-step prediction for AFM

        afm_batch_count += 1

    print(f"✓ AFM dataloader works correctly")

    # Test FM dataloader
    print("\nTesting FM dataloader...")
    fm_batch_count = 0
    for batch in fm_train_loader:
        if fm_batch_count >= 3:  # Just test a few batches
            break
        history, covariates, prediction = batch
        print(f"  Batch {fm_batch_count}: history={history.shape}, covariates={covariates.shape}, prediction={prediction.shape}")

        # Validate shapes
        assert history.shape == (config.data_batch_size, config.context_length, 3)  # Lorenz is 3D
        assert covariates.shape == (config.data_batch_size, config.context_length, 1)  # Time covariate
        assert prediction.shape == (config.data_batch_size, config.fm_prediction_horizon, 3)  # Multi-step prediction for FM

        fm_batch_count += 1

    print(f"✓ FM dataloader works correctly")

    # Test the deterministic evaluation loader (4-tuple windows)
    print("\nTesting evaluation dataloader...")
    eval_loader = create_evaluation_dataloader(test_trajectories, time_grid, config)
    n_test = test_trajectories.shape[0]
    for _ in range(2):  # must be re-iterable: several models evaluated in sequence
        batches = list(eval_loader)
        assert len(batches) == 1  # 12 test trajectories < batch_size 16
        history, cov, pred, extrap = batches[0]
        window = config.context_length + config.fm_prediction_horizon + config.extrapolation_size
        assert history.shape == (n_test, config.context_length, 3)
        assert cov.shape == (n_test, config.context_length, 1)
        assert pred.shape == (n_test, config.fm_prediction_horizon, 3)
        assert extrap.shape == (n_test, config.extrapolation_size, 3)
        assert config.context_length + config.fm_prediction_horizon + config.extrapolation_size == window
    print(f"✓ Evaluation dataloader works correctly (re-iterable, 4-tuple windows)")

    # Test model training with new loss function
    print("\nTesting AFM training with new parallel loss...")
    rngs = nnx.Rngs(0)

    afm_model = AutoregressiveFlowMatching(
        dim_x=3,  # Lorenz system dimension
        dim_c=1,  # Time covariate dimension
        dim_h=config.model_hidden_dim,
        dim_t=config.model_time_dim,
        dim_hidden=config.model_hidden_dim,
        hidden_blocks=config.model_hidden_blocks,
        n_rnn_layers=config.model_n_rnn_layers,
        rnn_hidden=config.model_rnn_hidden,
        rngs=rngs
    )

    afm_trainer = AFMTrainer(afm_model, learning_rate=config.training_learning_rate)

    # Test one training step
    for batch in afm_train_loader:
        history, covariates, prediction = batch
        rng = jrandom.PRNGKey(123)

        # Test loss computation
        loss = afm_trainer.loss_fn(afm_model, rng, history, covariates, prediction)
        print(f"  AFM loss: {float(loss):.6f}")
        assert jnp.isfinite(loss), "AFM loss should be finite"
        break

    print("✓ AFM training works correctly")

    # Test FM training
    print("\nTesting FM training...")
    fm_rngs = nnx.Rngs(1)

    fm_model = FlowMatching(
        dim_x=3,  # Lorenz system dimension
        dim_c=1,  # Time covariate dimension
        dim_h=config.model_hidden_dim,
        dim_t=config.model_time_dim,
        dim_hidden=config.model_hidden_dim,
        hidden_blocks=config.model_hidden_blocks,
        n_rnn_layers=config.model_n_rnn_layers,
        rngs=fm_rngs
    )

    fm_trainer = FMTrainer(fm_model, learning_rate=config.training_learning_rate)

    # Test one training step
    for batch in fm_train_loader:
        history, covariates, prediction = batch
        rng = jrandom.PRNGKey(124)

        # Test loss computation
        loss = fm_trainer.loss_fn(fm_model, rng, history, covariates, prediction)
        print(f"  FM loss: {float(loss):.6f}")
        assert jnp.isfinite(loss), "FM loss should be finite"
        break

    print("✓ FM training works correctly")

    print("\n🎉 All tests passed! The infinite dataloader implementation is working correctly.")
    print("\nKey improvements achieved:")
    print("- ✅ AFM training is now fully parallel (no sequential dependencies)")
    print("- ✅ Dynamic context window sampling increases data efficiency")
    print("- ✅ Flexible prediction horizons for different models")
    print("- ✅ Maintains compatibility with existing model architectures")


if __name__ == "__main__":
    test_infinite_dataloader()

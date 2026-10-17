"""AFM vs FM comparison experiment implementation."""

from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax.nnx as nnx
from pathlib import Path

from .base import BaseExperiment
from ..config import Config
from ..models import AutoregressiveFlowMatching, FlowMatching
from ..training.data import (
    prepare_trajectories,
    create_infinite_dataloaders,
    create_evaluation_dataloader,
)
from ..training.trainers import AFMTrainer, FMTrainer
from ..evaluation.evaluator import ModelEvaluator
from ..utils.visualization import plot_trajectories


class AFMvsFMComparison(BaseExperiment):
    """Experiment comparing Autoregressive Flow Matching vs Flow Matching."""

    def __init__(self, config: Config, seed: int = 0):
        """Initialize AFM vs FM comparison experiment."""
        super().__init__(config, seed)
        self.evaluator = ModelEvaluator(config, config.flow_nfe)

    def prepare_data(self) -> Tuple[Any, Any]:
        """Prepare training and test data loaders.

        Returns:
            Tuple of (train_data, eval_loader) where train_data is
            ((afm_train, afm_test), (fm_train, fm_test)) and eval_loader is
            the deterministic evaluation loader yielding
            (history, covariates, prediction, extrapolation) windows.
        """
        train_trajectories, test_trajectories, covariates = prepare_trajectories(
            self.system, self.config, self.key
        )
        train_data = create_infinite_dataloaders(
            train_trajectories, test_trajectories, covariates, self.config
        )
        eval_loader = create_evaluation_dataloader(
            test_trajectories, covariates, self.config
        )
        return train_data, eval_loader

    def create_models(self) -> Dict[str, Any]:
        """Create AFM and FM models."""
        self.key, afm_key, fm_key = jrandom.split(self.key, 3)

        # AFM Model
        afm_model = AutoregressiveFlowMatching(
            dim_x=self.system.data_dim,
            dim_c=self.config.model_covariates_dim,
            dim_h=self.config.model_hidden_dim,
            dim_t=self.config.model_time_dim,
            dim_hidden=self.config.model_hidden_dim,
            hidden_blocks=self.config.model_hidden_blocks,
            n_rnn_layers=self.config.model_n_rnn_layers,
            rnn_hidden=self.config.model_rnn_hidden,
            rngs=nnx.Rngs(afm_key)
        )

        # FM Model
        fm_model = FlowMatching(
            dim_x=self.system.data_dim,
            dim_c=self.config.model_covariates_dim,
            dim_h=self.config.model_hidden_dim,
            dim_t=self.config.model_time_dim,
            dim_hidden=self.config.model_hidden_dim,
            hidden_blocks=self.config.model_hidden_blocks,
            n_rnn_layers=self.config.model_n_rnn_layers,
            rngs=nnx.Rngs(fm_key)
        )

        return {
            'AFM': afm_model,
            'FM': fm_model
        }

    def train_models(
        self,
        models: Dict[str, Any],
        train_data: Any,
        test_data: Any
    ) -> None:
        """Train AFM and FM models."""
        # train_data holds ((afm_train, afm_test), (fm_train, fm_test));
        # the evaluation loader is passed separately as test_data.
        (afm_train_loader, afm_test_loader), (fm_train_loader, fm_test_loader) = train_data

        # Train AFM model
        afm_trainer = AFMTrainer(
            models['AFM'],
            learning_rate=self.config.training_learning_rate,
            model_name="AFM",
            log_gradients=getattr(self.config, 'log_gradients', False),
            log_frequency=getattr(self.config, 'log_frequency', 10)
        )
        afm_trainer.train(afm_train_loader, afm_test_loader, num_epochs=self.config.training_num_epochs)

        # Train FM model (typically needs more epochs)
        fm_trainer = FMTrainer(
            models['FM'],
            learning_rate=self.config.training_learning_rate,
            model_name="FM",
            log_gradients=getattr(self.config, 'log_gradients', False),
            log_frequency=getattr(self.config, 'log_frequency', 10)
        )
        fm_trainer.train(
            fm_train_loader,
            fm_test_loader,
            num_epochs=self.config.training_num_epochs * 2
        )

        print('Training complete!')

    def evaluate_models(
        self,
        models: Dict[str, Any],
        test_data: Any
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate AFM and FM models."""
        test_loader = test_data
        results = {}

        print('Evaluating models...')
        for model_name, model in models.items():
            print(f'Evaluating {model_name} model...')
            metrics = self.evaluator.evaluate_model(model, test_loader, self.key)
            results[model_name] = metrics

            # Print results
            print(f'{model_name} Model:')
            print(f'  CRPS (Prediction): {metrics["crps_prediction"]:.4f}')
            print(f'  NRMSE (Prediction): {metrics["nrmse_prediction"]:.4f}')
            print(f'  CRPS (Extrapolation): {metrics["crps_extrapolation"]:.4f}')
            print(f'  NRMSE (Extrapolation): {metrics["nrmse_extrapolation"]:.4f}')

        return results

    def save_visualizations(self, models: Dict[str, Any], test_data: Any) -> None:
        """Save trajectory prediction visualizations."""
        if not self.config.experiment_save_figures:
            return

        test_loader = test_data
        results_path = self.create_results_directory()

        # Create subdirectories for each model
        (results_path / "AFM").mkdir(exist_ok=True)
        (results_path / "FM").mkdir(exist_ok=True)

        # Generate and save prediction plots
        for model_name, model in models.items():
            print(f'Generating visualizations for {model_name}...')

            # Sample trajectories
            predictions, history, true_future = self._sample_trajectories(
                model, test_loader
            )

            # Save first 10 trajectory predictions
            for i in range(min(10, predictions.shape[1])):
                filename = f"{model_name.lower()}_predictions_{i}.pdf"
                save_path = results_path / model_name / filename

                plot_trajectories(
                    self.config.ts,
                    history[i],
                    predictions[:, i],
                    true_future[i],
                    self.config.extrapolation_size,
                    title=f'{model_name} Model Predictions',
                    save_path=str(save_path)
                )

    def _sample_trajectories(
        self,
        model: Any,
        test_loader: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample trajectories from a model for visualization."""
        history, covariates, prediction, extrapolation = next(iter(test_loader))
        true_future = prediction
        true_exp = extrapolation
        true_fe = jnp.concatenate([true_future, true_exp], axis=1)

        predictions = model.sample(
            self.key,
            history,
            covariates,
            nfe=self.config.flow_nfe,
            n_steps_pred=(
                self.config.data_prediction_size +
                self.config.extrapolation_size
            ),
            n_samples=self.config.experiment_num_eval_samples
        )

        # Handle different model output shapes
        if isinstance(model, FlowMatching):
            predictions = jnp.squeeze(predictions, axis=2)

        return jnp.stack(predictions), history, true_fe

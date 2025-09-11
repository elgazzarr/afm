"""AFM vs FM comparison experiment implementation."""

from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax.nnx as nnx
from pathlib import Path

from .base_experiment import BaseExperiment, ExperimentConfig
from ..models import AutoregressiveFlowMatching, FlowMatching
from ..training.data_preparation import prepare_data
from ..training.trainers import AFMTrainer, FMTrainer
from ..evaluation.evaluator import ModelEvaluator
from ..utils.visualization import plot_trajectories


class AFMvsFMComparison(BaseExperiment):
    """Experiment comparing Autoregressive Flow Matching vs Flow Matching."""
    
    def __init__(self, config: ExperimentConfig, seed: int = 0):
        """Initialize AFM vs FM comparison experiment."""
        super().__init__(config, seed)
        self.evaluator = ModelEvaluator(config.model_config, config.nfe)
    
    def prepare_data(self) -> Tuple[Any, Any]:
        """Prepare training and test data loaders."""
        return prepare_data(self.system, self.config.model_config, self.key)
    
    def create_models(self) -> Dict[str, Any]:
        """Create AFM and FM models."""
        self.key, afm_key, fm_key = jrandom.split(self.key, 3)
        
        # AFM Model
        afm_model = AutoregressiveFlowMatching(
            dim_x=self.system.data_dim,
            dim_c=self.config.model_config.covariates_dim,
            dim_h=self.config.model_config.hidden_dim,
            dim_t=self.config.model_config.time_dim,
            dim_hidden=self.config.model_config.hidden_dim,
            hidden_blocks=self.config.model_config.hidden_blocks,
            n_rnn_layers=self.config.model_config.n_rnn_layers,
            rnn_hidden=self.config.model_config.rnn_hidden,
            rngs=nnx.Rngs(afm_key)
        )
        
        # FM Model  
        fm_model = FlowMatching(
            dim_x=self.system.data_dim,
            dim_c=self.config.model_config.covariates_dim,
            dim_h=self.config.model_config.hidden_dim,
            dim_t=self.config.model_config.time_dim,
            dim_hidden=self.config.model_config.hidden_dim,
            hidden_blocks=self.config.model_config.hidden_blocks,
            n_rnn_layers=self.config.model_config.n_rnn_layers,
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
        train_loader = train_data
        test_loader = test_data
        
        # Train AFM model
        print('Training AFM Model...')
        afm_trainer = AFMTrainer(
            models['AFM'], 
            learning_rate=self.config.model_config.learning_rate
        )
        afm_trainer.train(train_loader, test_loader, num_epochs=self.config.num_epochs)
        
        # Train FM model (typically needs more epochs)
        print('Training FM Model...')
        fm_trainer = FMTrainer(
            models['FM'],
            learning_rate=self.config.model_config.learning_rate
        )
        fm_trainer.train(
            train_loader, 
            test_loader, 
            num_epochs=self.config.num_epochs * 2
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
        if not self.config.save_figures:
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
                    self.config.model_config.ts,
                    history[i],
                    predictions[:, i], 
                    true_future[i],
                    self.config.model_config.extrapolation_size,
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
            n_steps_flow=self.config.nfe,
            n_steps_pred=(
                self.config.model_config.prediction_size + 
                self.config.model_config.extrapolation_size
            ),
            n_samples=self.config.num_eval_samples
        )
        
        # Handle different model output shapes
        if isinstance(model, FlowMatching):
            predictions = jnp.squeeze(predictions, axis=2)
            
        return jnp.stack(predictions), history, true_fe
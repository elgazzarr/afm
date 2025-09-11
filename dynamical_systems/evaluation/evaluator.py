"""Model evaluation utilities for systematic comparison."""

from typing import Dict, Any
import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax.nnx as nnx

from ..models import FlowMatching
from ..config_loader import Config
from .metrics import compute_crps, compute_nrmse


class ModelEvaluator:
    """Systematic model evaluation for AFM vs FM comparison."""
    
    def __init__(self, config: Config, nfe: int = 50):
        """Initialize evaluator.
        
        Args:
            config: Model configuration
            nfe: Number of function evaluations for sampling
        """
        self.config = config
        self.nfe = nfe
        
        # JIT compile evaluation function for speed
        self.evaluate_batch = nnx.jit(self._evaluate_batch)
    
    @staticmethod
    def _evaluate_batch(
        predictions: jnp.ndarray, 
        true_future: jnp.ndarray
    ) -> tuple:
        """Evaluate metrics for a single batch.
        
        Args:
            predictions: Model predictions of shape (n_samples, batch_size, seq_len, dims)
            true_future: Ground truth of shape (batch_size, seq_len, dims)
            
        Returns:
            Tuple of (crps, nrmse)
        """
        crps = compute_crps(predictions, true_future)
        nrmse = compute_nrmse(predictions, true_future)
        return crps, nrmse
    
    def evaluate_model(
        self,
        model: Any,
        test_loader: Any,
        key: jax.random.PRNGKey,
        n_samples: int = 50
    ) -> Dict[str, float]:
        """Evaluate a single model on test data.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            key: Random key for sampling
            n_samples: Number of samples for uncertainty quantification
            
        Returns:
            Dictionary containing evaluation metrics
        """
        crps_pred_list = []
        nrmse_pred_list = []
        crps_exp_list = []
        nrmse_exp_list = []
        
        # Unpack 4 components from dataloader: history, covariates, prediction, extrapolation
        for history, covariates, prediction, extrapolation in test_loader:
            # Use prediction as true_future and extrapolation as true_exp for evaluation
            true_future = prediction
            true_exp = extrapolation
            
            # Generate predictions
            predictions = model.sample(
                key, 
                history, 
                covariates,
                n_steps_flow=self.nfe,
                n_steps_pred=(
                    self.config.data_prediction_size + 
                    self.config.extrapolation_size
                ),
                n_samples=n_samples
            )
            
            # Handle different model output shapes
            if isinstance(model, FlowMatching):
                predictions = jnp.squeeze(predictions, axis=2)
            
            # Split predictions into forecast and extrapolation
            pred_forecast = predictions[:, :, :self.config.data_prediction_size]
            pred_extrapolation = predictions[:, :, self.config.data_prediction_size:]
            
            # Evaluate forecast performance
            crps_p, nrmse_p = self.evaluate_batch(pred_forecast, true_future)
            crps_pred_list.append(crps_p)
            nrmse_pred_list.append(nrmse_p)
            
            # Evaluate extrapolation performance
            crps_e, nrmse_e = self.evaluate_batch(pred_extrapolation, true_exp)
            crps_exp_list.append(crps_e)
            nrmse_exp_list.append(nrmse_e)
        
        # Aggregate results
        results = {
            'crps_prediction': float(jnp.mean(jnp.stack(crps_pred_list))),
            'nrmse_prediction': float(jnp.mean(jnp.stack(nrmse_pred_list))),
            'crps_extrapolation': float(jnp.mean(jnp.stack(crps_exp_list))),
            'nrmse_extrapolation': float(jnp.mean(jnp.stack(nrmse_exp_list)))
        }
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, Any],
        test_loader: Any,
        key: jax.random.PRNGKey,
        n_samples: int = 50
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models on the same test data.
        
        Args:
            models: Dictionary mapping model names to model instances
            test_loader: Test data loader
            key: Random key for sampling
            n_samples: Number of samples for uncertainty quantification
            
        Returns:
            Dictionary mapping model names to their evaluation results
        """
        results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name} model...")
            key, eval_key = jrandom.split(key)
            
            model_results = self.evaluate_model(
                model, test_loader, eval_key, n_samples
            )
            results[model_name] = model_results
            
            # Print results
            print(f"{model_name} Results:")
            for metric, value in model_results.items():
                print(f"  {metric}: {value:.4f}")
            print()
        
        return results
    
    def print_comparison_summary(
        self, 
        results: Dict[str, Dict[str, float]]
    ) -> None:
        """Print a summary comparison of model results.
        
        Args:
            results: Dictionary mapping model names to their metrics
        """
        print("="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        if len(results) < 2:
            print("Need at least 2 models for comparison.")
            return
        
        model_names = list(results.keys())
        metrics = list(results[model_names[0]].keys())
        
        for metric in metrics:
            print(f"\n{metric.upper()}:")
            print("-" * 40)
            
            # Find best performing model for this metric
            best_model = min(model_names, key=lambda m: results[m][metric])
            
            for model_name in model_names:
                value = results[model_name][metric]
                marker = " ★" if model_name == best_model else "  "
                print(f"{marker} {model_name:<15}: {value:.4f}")
        
        print("\n★ = Best performance")
        print("="*60)
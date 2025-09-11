"""Base experiment class for organizing and running experiments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import jax.numpy as jnp
import jax.random as jrandom
import wandb
import os
from pathlib import Path

from ..config_loader import Config
from ..systems.systems import DynamicalSystem, create_system


# Alias for backward compatibility
ExperimentConfig = Config


class BaseExperiment(ABC):
    """Base class for flow matching experiments."""
    
    def __init__(self, config: Config, seed: int = 0):
        """Initialize experiment.
        
        Args:
            config: Experiment configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        self.key = jrandom.PRNGKey(seed)
        
        # Initialize system
        self.system = create_system(
            config.system_name, 
            config.system_stochastic
        )
        
        # Results storage
        self.results: Dict[str, Any] = {}
        
    def setup_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        # Create a serializable config dict, excluding problematic fields
        config_dict = {
            # Experiment config
            'experiment_name': self.config.experiment_name,
            'wandb_project': self.config.experiment_wandb_project,
            'num_epochs': self.config.training_num_epochs,
            'nfe': self.config.flow_nfe,
            'num_eval_samples': self.config.experiment_num_eval_samples,
            'save_figures': self.config.experiment_save_figures,
            'results_dir': self.config.experiment_results_dir,
            # System and data config
            'num_timepoints': self.config.system_num_timepoints,
            't0': self.config.system_t0,
            't1': self.config.system_t1,
            'history_size': self.config.data_history_size,
            'prediction_size': self.config.data_prediction_size,
            'n_train_test_samples': self.config.data_n_train_test_samples,
            'batch_size': self.config.data_batch_size,
            # Model config
            'covariates_dim': self.config.model_covariates_dim,
            'hidden_dim': self.config.model_hidden_dim,
            'time_dim': self.config.model_time_dim,
            'hidden_blocks': self.config.model_hidden_blocks,
            'rnn_hidden': self.config.model_rnn_hidden,
            'n_rnn_layers': self.config.model_n_rnn_layers,
            'learning_rate': self.config.training_learning_rate,
            'n_samples': self.config.flow_n_samples,
            'n_steps_flow': self.config.flow_n_steps_flow,
            'system': self.config.system_name,
            'stochastic': self.config.system_stochastic,
            'extrapolation_size': self.config.extrapolation_size,
            'dt': float(self.config.dt),
        }
        
        wandb.init(
            project=self.config.experiment_wandb_project,
            config=config_dict,
            name=self.config.experiment_name
        )
    
    def create_results_directory(self) -> Path:
        """Create results directory for experiment outputs."""
        results_path = Path(self.config.experiment_results_dir) / self.config.system_name
        results_path.mkdir(parents=True, exist_ok=True)
        return results_path
    
    @abstractmethod
    def prepare_data(self) -> Tuple[Any, Any]:
        """Prepare training and test data.
        
        Returns:
            Tuple of (train_data, test_data)
        """
        pass
    
    @abstractmethod
    def create_models(self) -> Dict[str, Any]:
        """Create models for the experiment.
        
        Returns:
            Dictionary mapping model names to model instances
        """
        pass
    
    @abstractmethod 
    def train_models(self, models: Dict[str, Any], train_data: Any, test_data: Any) -> None:
        """Train all models.
        
        Args:
            models: Dictionary of models to train
            train_data: Training data
            test_data: Test data
        """
        pass
    
    @abstractmethod
    def evaluate_models(self, models: Dict[str, Any], test_data: Any) -> Dict[str, Dict[str, float]]:
        """Evaluate all models.
        
        Args:
            models: Dictionary of trained models
            test_data: Test data
            
        Returns:
            Dictionary mapping model names to metrics
        """
        pass
    
    def log_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Log results to wandb and store internally.
        
        Args:
            results: Dictionary mapping model names to metrics
        """
        self.results.update(results)
        
        # Log to wandb
        for model_name, metrics in results.items():
            wandb_metrics = {f"{model_name}/{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics)
    
    def save_visualizations(self, models: Dict[str, Any], test_data: Any) -> None:
        """Save visualization plots.
        
        Args:
            models: Dictionary of trained models  
            test_data: Test data for plotting
        """
        if not self.config.experiment_save_figures:
            return
            
        results_path = self.create_results_directory()
        # Implementation will be added by subclasses
        pass
    
    def run(self) -> Dict[str, Any]:
        """Run the complete experiment.
        
        Returns:
            Dictionary containing experiment results
        """
        print(f"Starting experiment: {self.config.experiment_name}")
        
        # Setup
        self.setup_wandb()
        
        # Prepare data
        print("Preparing data...")
        train_data, test_data = self.prepare_data()
        
        # Create models
        print("Creating models...")
        models = self.create_models()
        
        # Train models
        print("Training models...")
        self.train_models(models, train_data, test_data)
        
        # Evaluate models
        print("Evaluating models...")
        results = self.evaluate_models(models, test_data)
        
        # Log and save results
        self.log_results(results)
        self.save_visualizations(models, test_data)
        
        print("Experiment completed!")
        print("#" * 50)
        
        return self.results
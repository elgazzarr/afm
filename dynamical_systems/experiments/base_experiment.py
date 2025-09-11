"""Base experiment class for organizing and running experiments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import jax.numpy as jnp
import jax.random as jrandom
import wandb
import os
from pathlib import Path

from ..config import ModelConfig
from ..systems.systems import DynamicalSystem, create_system


@dataclass 
class ExperimentConfig:
    """Configuration for experiments.
    
    Extends ModelConfig with experiment-specific parameters.
    """
    # Experiment settings
    experiment_name: str = "afm_vs_fm_comparison"
    wandb_project: str = "flowtime_ds"
    num_epochs: int = 200
    nfe: int = 20  # Number of function evaluations
    num_eval_samples: int = 50
    save_figures: bool = True
    results_dir: str = "results"
    
    # Model configuration
    model_config: Optional[ModelConfig] = None
    
    def __post_init__(self):
        """Initialize model config if not provided."""
        if self.model_config is None:
            self.model_config = ModelConfig()
    
    @classmethod
    def from_args(cls, args) -> 'ExperimentConfig':
        """Create ExperimentConfig from command line arguments."""
        model_config = ModelConfig(
            system=args.system,
            num_timepoints=args.num_timepoints,
            history_size=args.history_size,
            prediction_size=args.prediction_size,
            t0=0.0,
            t1=args.t1,
            n_train_test_samples=1024,
            batch_size=256,
            n_steps_flow=args.nfe,
        )
        
        return cls(
            experiment_name=f"{args.system}_{args.nfe}",
            wandb_project=args.wandb_project,
            num_epochs=args.num_epochs,
            nfe=args.nfe,
            model_config=model_config
        )


class BaseExperiment(ABC):
    """Base class for flow matching experiments."""
    
    def __init__(self, config: ExperimentConfig, seed: int = 0):
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
            config.model_config.system, 
            config.model_config.stochastic
        )
        
        # Results storage
        self.results: Dict[str, Any] = {}
        
    def setup_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        # Create a serializable config dict, excluding problematic fields
        config_dict = {
            # Experiment config
            'experiment_name': self.config.experiment_name,
            'wandb_project': self.config.wandb_project,
            'num_epochs': self.config.num_epochs,
            'nfe': self.config.nfe,
            'num_eval_samples': self.config.num_eval_samples,
            'save_figures': self.config.save_figures,
            'results_dir': self.config.results_dir,
            # Model config (excluding JAX arrays)
            'num_timepoints': self.config.model_config.num_timepoints,
            't0': self.config.model_config.t0,
            't1': self.config.model_config.t1,
            'history_size': self.config.model_config.history_size,
            'prediction_size': self.config.model_config.prediction_size,
            'n_train_test_samples': self.config.model_config.n_train_test_samples,
            'batch_size': self.config.model_config.batch_size,
            'covariates_dim': self.config.model_config.covariates_dim,
            'hidden_dim': self.config.model_config.hidden_dim,
            'time_dim': self.config.model_config.time_dim,
            'hidden_blocks': self.config.model_config.hidden_blocks,
            'rnn_hidden': self.config.model_config.rnn_hidden,
            'n_rnn_layers': self.config.model_config.n_rnn_layers,
            'learning_rate': self.config.model_config.learning_rate,
            'n_samples': self.config.model_config.n_samples,
            'n_steps_flow': self.config.model_config.n_steps_flow,
            'system': self.config.model_config.system,
            'stochastic': self.config.model_config.stochastic,
            'extrapolation_size': self.config.model_config.extrapolation_size,
            'dt': float(self.config.model_config.dt),
        }
        
        wandb.init(
            project=self.config.wandb_project,
            config=config_dict,
            name=self.config.experiment_name
        )
    
    def create_results_directory(self) -> Path:
        """Create results directory for experiment outputs."""
        results_path = Path(self.config.results_dir) / self.config.model_config.system
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
        if not self.config.save_figures:
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
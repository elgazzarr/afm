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
        """Initialize Weights & Biases logging with enhanced organization."""
        # Create experiment group and tags
        experiment_type = getattr(self.config, 'experiment_type', 'unknown')
        system_name = self.config.system_name
        stochastic_str = 'stochastic' if self.config.system_stochastic else 'deterministic'
        
        # Hierarchical group naming
        group_name = f"{experiment_type}/{system_name}/{stochastic_str}"
        
        # Generate meaningful tags
        tags = [
            experiment_type,
            system_name,
            stochastic_str,
            'afm_vs_fm',
            f"nfe_{self.config.flow_nfe}",
            f"epochs_{self.config.training_num_epochs}"
        ]
        
        # Add sweep tag if this is a parameter sweep
        if hasattr(self.config, 'sweep_params'):
            tags.append('parameter_sweep')
        
        # Create a serializable config dict, excluding problematic fields
        config_dict = {
            # Experiment metadata
            'experiment_type': experiment_type,
            'experiment_name': self.config.experiment_name,
            'experiment_description': getattr(self.config, 'experiment_description', ''),
            
            # System configuration
            'system_name': self.config.system_name,
            'system_stochastic': self.config.system_stochastic,
            'system_num_timepoints': self.config.system_num_timepoints,
            'system_t0': self.config.system_t0,
            'system_t1': self.config.system_t1,
            'system_dt': float(self.config.dt),
            
            # Data configuration
            'data_history_size': self.config.data_history_size,
            'data_prediction_size': self.config.data_prediction_size,
            'data_extrapolation_size': self.config.extrapolation_size,
            'data_n_train_test_samples': self.config.data_n_train_test_samples,
            'data_batch_size': self.config.data_batch_size,
            
            # Model architecture
            'model_covariates_dim': self.config.model_covariates_dim,
            'model_hidden_dim': self.config.model_hidden_dim,
            'model_time_dim': self.config.model_time_dim,
            'model_hidden_blocks': self.config.model_hidden_blocks,
            'model_rnn_hidden': self.config.model_rnn_hidden,
            'model_n_rnn_layers': self.config.model_n_rnn_layers,
            
            # Training configuration
            'training_learning_rate': self.config.training_learning_rate,
            'training_num_epochs': self.config.training_num_epochs,
            
            # Flow matching configuration
            'flow_n_samples': self.config.flow_n_samples,
            'flow_n_steps_flow': self.config.flow_n_steps_flow,
            'flow_nfe': self.config.flow_nfe,
            
            # Evaluation configuration
            'eval_num_samples': self.config.experiment_num_eval_samples,
        }
        
        # Add sweep parameters if they exist
        if hasattr(self.config, 'sweep_params'):
            config_dict['sweep_params'] = getattr(self.config, 'sweep_params')
        
        # Add custom wandb settings from config
        wandb_config = getattr(self.config, 'wandb', {})
        custom_group = wandb_config.get('group', group_name)
        custom_tags = wandb_config.get('tags', [])
        notes = wandb_config.get('notes', getattr(self.config, 'experiment_description', ''))
        
        # Initialize wandb with enhanced organization
        wandb.init(
            project=self.config.experiment_wandb_project,
            group=custom_group,
            name=self.config.experiment_name,
            tags=tags + custom_tags,
            notes=notes,
            config=config_dict,
            # Allow multiple runs in same process for parameter sweeps
            reinit=True
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
        """Log results to wandb with artifact management.
        
        Args:
            results: Dictionary mapping model names to metrics
        """
        self.results.update(results)
        
        # Log metrics to wandb
        for model_name, metrics in results.items():
            wandb_metrics = {f"{model_name}/{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics)
        
        # Save results as wandb artifact if enabled
        wandb_config = getattr(self.config, 'wandb', {})
        save_artifacts = wandb_config.get('save_artifacts', True)
        
        if save_artifacts and wandb.run is not None:
            self._save_results_artifact(results)
    
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
        
        # Save configuration as artifact
        if wandb.run is not None:
            self._save_config_artifact()
        
        # Prepare data
        print("Preparing data...")
        train_data, test_data = self.prepare_data()
        
        # Create models
        print("Creating models...")
        models = self.create_models()
        
        # Train models
        print("Training models...")
        self.train_models(models, train_data, test_data)
        
        # Save model artifacts if requested
        self.save_model_artifacts(models)
        
        # Evaluate models
        print("Evaluating models...")
        results = self.evaluate_models(models, test_data)
        
        # Log and save results
        self.log_results(results)
        self.save_visualizations(models, test_data)
        
        print("Experiment completed!")
        print("#" * 50)
        
        return self.results
    
    def _save_results_artifact(self, results: Dict[str, Dict[str, float]]) -> None:
        """Save experiment results as wandb artifact.
        
        Args:
            results: Dictionary of experiment results
        """
        try:
            import json
            import tempfile
            
            # Create temporary file for results
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(results, f, indent=2)
                temp_path = f.name
            
            # Create and log artifact
            artifact = wandb.Artifact(
                name=f"{self.config.experiment_name}_results",
                type="experiment_results",
                description=f"Results from {self.config.experiment_name}",
                metadata={
                    'experiment_type': getattr(self.config, 'experiment_type', 'unknown'),
                    'system_name': self.config.system_name,
                    'stochastic': self.config.system_stochastic
                }
            )
            
            artifact.add_file(temp_path, name='results.json')
            wandb.log_artifact(artifact)
            
            # Clean up temp file
            import os
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"Warning: Failed to save results artifact: {e}")
    
    def _save_config_artifact(self) -> None:
        """Save experiment configuration as wandb artifact."""
        try:
            import json
            import tempfile
            
            # Extract config as dictionary
            config_dict = {
                'experiment_name': self.config.experiment_name,
                'experiment_type': getattr(self.config, 'experiment_type', 'unknown'),
                'system_name': self.config.system_name,
                'system_stochastic': self.config.system_stochastic,
                'data_history_size': self.config.data_history_size,
                'data_prediction_size': self.config.data_prediction_size,
                'training_learning_rate': self.config.training_learning_rate,
                'training_num_epochs': self.config.training_num_epochs,
                'flow_nfe': self.config.flow_nfe,
                # Add other relevant config parameters
            }
            
            # Create temporary file for config
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_dict, f, indent=2)
                temp_path = f.name
            
            # Create and log artifact
            artifact = wandb.Artifact(
                name=f"{self.config.experiment_name}_config",
                type="experiment_config",
                description=f"Configuration for {self.config.experiment_name}",
                metadata={
                    'experiment_type': getattr(self.config, 'experiment_type', 'unknown'),
                    'system_name': self.config.system_name
                }
            )
            
            artifact.add_file(temp_path, name='config.json')
            wandb.log_artifact(artifact)
            
            # Clean up temp file
            import os
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"Warning: Failed to save config artifact: {e}")
    
    def save_model_artifacts(self, models: Dict[str, Any]) -> None:
        """Save trained models as wandb artifacts.
        
        Args:
            models: Dictionary of trained models
        """
        wandb_config = getattr(self.config, 'wandb', {})
        save_models = wandb_config.get('save_models', False)
        
        if not save_models or wandb.run is None:
            return
        
        try:
            import pickle
            import tempfile
            import os
            
            for model_name, model in models.items():
                # Create temporary file for model
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                    # Save model state (this is a simplified version)
                    pickle.dump(model, f)
                    temp_path = f.name
                
                # Create and log artifact
                artifact = wandb.Artifact(
                    name=f"{self.config.experiment_name}_{model_name}_model",
                    type="model",
                    description=f"Trained {model_name} model from {self.config.experiment_name}",
                    metadata={
                        'model_type': model_name,
                        'experiment_type': getattr(self.config, 'experiment_type', 'unknown'),
                        'system_name': self.config.system_name,
                        'training_epochs': self.config.training_num_epochs,
                        'learning_rate': self.config.training_learning_rate
                    }
                )
                
                artifact.add_file(temp_path, name=f'{model_name.lower()}_model.pkl')
                wandb.log_artifact(artifact)
                
                # Clean up temp file
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"Warning: Failed to save model artifacts: {e}")
"""Parameter sweep experiment for studying sensitivity to hyperparameters."""

from typing import Dict, Any, List, Tuple
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from pathlib import Path
import json

from .base_experiment import BaseExperiment
from ..config_loader import Config
from ..models import AutoregressiveFlowMatching, FlowMatching
from ..training.data_preparation import prepare_data
from ..training.trainers import AFMTrainer, FMTrainer
from ..evaluation.evaluator import ModelEvaluator


class ParameterSweepExperiment(BaseExperiment):
    """Base class for experiments that sweep over parameters."""
    
    def __init__(self, config: Config, seed: int = 0):
        """Initialize parameter sweep experiment."""
        super().__init__(config, seed)
        self.sweep_results = {}
        
    def create_sweep_configs(self, base_config: Config, sweep_params: Dict[str, List]) -> List[Config]:
        """Create configurations for parameter sweep.
        
        Args:
            base_config: Base configuration to modify
            sweep_params: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of configurations for the sweep
        """
        configs = []
        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())
        
        # Generate all combinations
        import itertools
        for combo in itertools.product(*param_values):
            # Create copy of base config
            sweep_config = Config()
            
            # Copy all attributes from base config
            for attr in dir(base_config):
                if not attr.startswith('_') and hasattr(sweep_config, attr):
                    setattr(sweep_config, attr, getattr(base_config, attr))
            
            # Apply sweep parameters
            for param_name, param_value in zip(param_names, combo):
                if hasattr(sweep_config, param_name):
                    setattr(sweep_config, param_name, param_value)
                else:
                    print(f"Warning: Parameter {param_name} not found in config")
            
            # Recompute derived fields after modification
            sweep_config._compute_derived_fields()
            configs.append(sweep_config)
            
        return configs
    
    def run_single_sweep_point(self, config: Config) -> Dict[str, Any]:
        """Run experiment for a single parameter combination.
        
        Args:
            config: Configuration for this sweep point
            
        Returns:
            Dictionary containing results for this sweep point
        """
        print(f"Running sweep point with config: {self.get_config_identifier(config)}")
        
        # Prepare data
        train_data, test_data = prepare_data(self.system, config, self.key)
        
        # Create models
        self.key, afm_key, fm_key = jrandom.split(self.key, 3)
        
        afm_model = AutoregressiveFlowMatching(
            dim_x=self.system.data_dim,
            dim_c=config.model_covariates_dim,
            dim_h=config.model_hidden_dim,
            dim_t=config.model_time_dim,
            dim_hidden=config.model_hidden_dim,
            hidden_blocks=config.model_hidden_blocks,
            n_rnn_layers=config.model_n_rnn_layers,
            rnn_hidden=config.model_rnn_hidden,
            rngs=jax.nn.initializers.Rngs(afm_key)
        )
        
        fm_model = FlowMatching(
            dim_x=self.system.data_dim,
            dim_c=config.model_covariates_dim,
            dim_h=config.model_hidden_dim,
            dim_t=config.model_time_dim,
            dim_hidden=config.model_hidden_dim,
            hidden_blocks=config.model_hidden_blocks,
            n_rnn_layers=config.model_n_rnn_layers,
            rngs=jax.nn.initializers.Rngs(fm_key)
        )
        
        models = {'AFM': afm_model, 'FM': fm_model}
        
        # Train models
        afm_trainer = AFMTrainer(afm_model, learning_rate=config.training_learning_rate)
        fm_trainer = FMTrainer(fm_model, learning_rate=config.training_learning_rate)
        
        afm_trainer.train(train_data, test_data, num_epochs=config.training_num_epochs)
        fm_trainer.train(train_data, test_data, num_epochs=getattr(config, 'training_num_epochs_fm', config.training_num_epochs * 2))
        
        # Evaluate models
        evaluator = ModelEvaluator(config, config.flow_nfe)
        results = {}
        
        for model_name, model in models.items():
            metrics = evaluator.evaluate_model(model, test_data, self.key)
            results[model_name] = metrics
            
        # Add configuration info to results
        results['config'] = self.get_config_dict(config)
        
        return results
    
    def get_config_identifier(self, config: Config) -> str:
        """Get a string identifier for a configuration."""
        return f"system={config.system_name}_hist={config.data_history_size}_pred={config.data_prediction_size}"
    
    def get_config_dict(self, config: Config) -> Dict[str, Any]:
        """Extract relevant config parameters as dictionary."""
        return {
            'system_name': config.system_name,
            'data_history_size': config.data_history_size,
            'data_prediction_size': config.data_prediction_size,
            'model_hidden_dim': config.model_hidden_dim,
            'model_hidden_blocks': config.model_hidden_blocks,
            'model_n_rnn_layers': config.model_n_rnn_layers,
            'training_learning_rate': config.training_learning_rate,
            'training_num_epochs': config.training_num_epochs,
            'flow_nfe': config.flow_nfe
        }
    
    def save_sweep_results(self) -> None:
        """Save sweep results to file."""
        results_dir = Path(self.config.experiment_results_dir) / self.config.system_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"{self.config.experiment_name}_sweep_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in self.sweep_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, (jnp.ndarray, np.ndarray)):
                        serializable_results[key][k] = v.tolist()
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Saved sweep results to {results_file}")


class HistoryLengthSweepExperiment(ParameterSweepExperiment):
    """Experiment that sweeps over history lengths."""
    
    def __init__(self, config: Config, seed: int = 0):
        super().__init__(config, seed)
        
    def run(self) -> Dict[str, Any]:
        """Run history length sweep experiment."""
        print(f"Starting history length sweep experiment: {self.config.experiment_name}")
        
        # Setup wandb
        self.setup_wandb()
        
        # Get sweep parameters from config
        sweep_params = getattr(self.config, 'sweep_params', {'data_history_size': [20, 30, 40, 50, 60]})
        history_sizes = sweep_params.get('data_history_size', [20, 30, 40, 50, 60])
        
        # Run sweep
        for history_size in history_sizes:
            print(f"\n--- Testing history_size = {history_size} ---")
            
            # Create modified config
            sweep_config = Config()
            for attr in dir(self.config):
                if not attr.startswith('_') and hasattr(sweep_config, attr):
                    setattr(sweep_config, attr, getattr(self.config, attr))
            
            sweep_config.data_history_size = history_size
            
            # Adjust num_timepoints if necessary
            min_timepoints = history_size + sweep_config.data_prediction_size + 10
            if sweep_config.system_num_timepoints < min_timepoints:
                sweep_config.system_num_timepoints = min_timepoints
            
            sweep_config._compute_derived_fields()
            
            # Run single sweep point
            results = self.run_single_sweep_point(sweep_config)
            self.sweep_results[f"history_{history_size}"] = results
            
            # Log to wandb
            for model_name in ['AFM', 'FM']:
                if model_name in results:
                    metrics = results[model_name]
                    wandb_metrics = {
                        f"{model_name}/history_size": history_size,
                        f"{model_name}/crps_prediction": metrics.get('crps_prediction', 0),
                        f"{model_name}/nrmse_prediction": metrics.get('nrmse_prediction', 0),
                        f"{model_name}/crps_extrapolation": metrics.get('crps_extrapolation', 0),
                        f"{model_name}/nrmse_extrapolation": metrics.get('nrmse_extrapolation', 0)
                    }
                    import wandb
                    wandb.log(wandb_metrics)
        
        # Save results
        self.save_sweep_results()
        
        # Analyze results
        self.analyze_sweep_results()
        
        return self.sweep_results
    
    def analyze_sweep_results(self) -> None:
        """Analyze sweep results and find optimal history length."""
        print("\n" + "="*60)
        print("HISTORY LENGTH SWEEP ANALYSIS")
        print("="*60)
        
        history_sizes = []
        afm_crps = []
        fm_crps = []
        afm_nrmse = []
        fm_nrmse = []
        
        for key, results in self.sweep_results.items():
            if key.startswith("history_"):
                history_size = int(key.split("_")[1])
                history_sizes.append(history_size)
                
                if 'AFM' in results:
                    afm_crps.append(results['AFM'].get('crps_prediction', float('inf')))
                    afm_nrmse.append(results['AFM'].get('nrmse_prediction', float('inf')))
                else:
                    afm_crps.append(float('inf'))
                    afm_nrmse.append(float('inf'))
                    
                if 'FM' in results:
                    fm_crps.append(results['FM'].get('crps_prediction', float('inf')))
                    fm_nrmse.append(results['FM'].get('nrmse_prediction', float('inf')))
                else:
                    fm_crps.append(float('inf'))
                    fm_nrmse.append(float('inf'))
        
        # Sort by history size
        sorted_indices = sorted(range(len(history_sizes)), key=lambda i: history_sizes[i])
        
        print("\nHistory Size vs Performance:")
        print("-" * 80)
        print(f"{'History':<8} {'AFM CRPS':<12} {'FM CRPS':<12} {'AFM NRMSE':<12} {'FM NRMSE':<12} {'Winner':<8}")
        print("-" * 80)
        
        for i in sorted_indices:
            hist = history_sizes[i]
            a_crps = afm_crps[i]
            f_crps = fm_crps[i] 
            a_nrmse = afm_nrmse[i]
            f_nrmse = fm_nrmse[i]
            winner = "AFM" if a_crps < f_crps else "FM"
            
            print(f"{hist:<8} {a_crps:<12.4f} {f_crps:<12.4f} {a_nrmse:<12.4f} {f_nrmse:<12.4f} {winner:<8}")
        
        # Find optimal history lengths
        best_afm_idx = min(range(len(afm_crps)), key=lambda i: afm_crps[i])
        best_fm_idx = min(range(len(fm_crps)), key=lambda i: fm_crps[i])
        
        print(f"\nOptimal History Lengths:")
        print(f"AFM best: {history_sizes[best_afm_idx]} (CRPS: {afm_crps[best_afm_idx]:.4f})")
        print(f"FM best:  {history_sizes[best_fm_idx]} (CRPS: {fm_crps[best_fm_idx]:.4f})")
        print("="*60)


class NFEEfficiencyExperiment(ParameterSweepExperiment):
    """Experiment that sweeps over NFE values to study efficiency."""
    
    def run(self) -> Dict[str, Any]:
        """Run NFE efficiency experiment."""
        print(f"Starting NFE efficiency experiment: {self.config.experiment_name}")
        
        # Setup wandb
        self.setup_wandb()
        
        # Get sweep parameters
        sweep_params = getattr(self.config, 'sweep_params', {})
        nfe_values = sweep_params.get('nfe_values', [5, 10, 15, 20, 30, 40, 50])
        
        # Run sweep
        for nfe in nfe_values:
            print(f"\n--- Testing NFE = {nfe} ---")
            
            # Create modified config
            sweep_config = Config()
            for attr in dir(self.config):
                if not attr.startswith('_') and hasattr(sweep_config, attr):
                    setattr(sweep_config, attr, getattr(self.config, attr))
            
            sweep_config.flow_nfe = nfe
            sweep_config.flow_n_steps_flow = nfe
            sweep_config._compute_derived_fields()
            
            # Measure timing
            import time
            start_time = time.time()
            
            results = self.run_single_sweep_point(sweep_config)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Add timing info
            results['timing'] = {
                'total_time': elapsed_time,
                'nfe': nfe
            }
            
            self.sweep_results[f"nfe_{nfe}"] = results
            
            # Log to wandb with timing
            for model_name in ['AFM', 'FM']:
                if model_name in results:
                    metrics = results[model_name]
                    wandb_metrics = {
                        f"{model_name}/nfe": nfe,
                        f"{model_name}/crps_prediction": metrics.get('crps_prediction', 0),
                        f"{model_name}/nrmse_prediction": metrics.get('nrmse_prediction', 0),
                        f"{model_name}/total_time": elapsed_time
                    }
                    import wandb
                    wandb.log(wandb_metrics)
        
        # Save and analyze results
        self.save_sweep_results()
        self.analyze_nfe_results()
        
        return self.sweep_results
        
    def analyze_nfe_results(self) -> None:
        """Analyze NFE efficiency results."""
        print("\n" + "="*60)
        print("NFE EFFICIENCY ANALYSIS")
        print("="*60)
        
        nfe_values = []
        afm_crps = []
        fm_crps = []
        timings = []
        
        for key, results in self.sweep_results.items():
            if key.startswith("nfe_"):
                nfe = int(key.split("_")[1])
                nfe_values.append(nfe)
                timings.append(results.get('timing', {}).get('total_time', 0))
                
                afm_crps.append(results.get('AFM', {}).get('crps_prediction', float('inf')))
                fm_crps.append(results.get('FM', {}).get('crps_prediction', float('inf')))
        
        # Sort by NFE
        sorted_indices = sorted(range(len(nfe_values)), key=lambda i: nfe_values[i])
        
        print("\nNFE vs Performance vs Time:")
        print("-" * 70)
        print(f"{'NFE':<5} {'AFM CRPS':<12} {'FM CRPS':<12} {'Time (s)':<10} {'Winner':<8}")
        print("-" * 70)
        
        for i in sorted_indices:
            nfe = nfe_values[i]
            a_crps = afm_crps[i]
            f_crps = fm_crps[i]
            time_s = timings[i]
            winner = "AFM" if a_crps < f_crps else "FM"
            
            print(f"{nfe:<5} {a_crps:<12.4f} {f_crps:<12.4f} {time_s:<10.2f} {winner:<8}")
        
        print("="*60)
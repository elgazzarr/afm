"""Parameter sweep experiment for studying sensitivity to hyperparameters."""

from typing import Dict, Any, List, Tuple
import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax.nnx as nnx
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

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
        
        # Prepare data: infinite training loaders + deterministic evaluation loader
        train_trajectories, test_trajectories, covariates = prepare_trajectories(
            self.system, config, self.key
        )
        (afm_train_data, afm_test_data), (fm_train_data, fm_test_data) = \
            create_infinite_dataloaders(
                train_trajectories, test_trajectories, covariates, config
            )
        eval_loader = create_evaluation_dataloader(
            test_trajectories, covariates, config
        )
        
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
            rngs=nnx.Rngs(afm_key)
        )
        
        fm_model = FlowMatching(
            dim_x=self.system.data_dim,
            dim_c=config.model_covariates_dim,
            dim_h=config.model_hidden_dim,
            dim_t=config.model_time_dim,
            dim_hidden=config.model_hidden_dim,
            hidden_blocks=config.model_hidden_blocks,
            n_rnn_layers=config.model_n_rnn_layers,
            rngs=nnx.Rngs(fm_key)
        )
        
        models = {'AFM': afm_model, 'FM': fm_model}
        
        # Train models
        afm_trainer = AFMTrainer(afm_model, learning_rate=config.training_learning_rate)
        fm_trainer = FMTrainer(fm_model, learning_rate=config.training_learning_rate)
        
        afm_trainer.train(afm_train_data, afm_test_data, num_epochs=config.training_num_epochs)
        fm_trainer.train(fm_train_data, fm_test_data, num_epochs=getattr(config, 'training_num_epochs_fm', config.training_num_epochs * 2))
        
        # Evaluate models
        evaluator = ModelEvaluator(config, config.flow_nfe)
        results = {}
        
        for model_name, model in models.items():
            metrics = evaluator.evaluate_model(model, eval_loader, self.key)
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
    """Experiment that sweeps over history lengths, comparing AFM vs FM performance.
    
    Unlike NFE experiments, this requires retraining models for each history length
    since the history size affects model architecture and data preparation.
    """
    
    def __init__(self, config: Config, seed: int = 0):
        super().__init__(config, seed)
        self.optimal_histories = {'AFM': None, 'FM': None}  # Track optimal history lengths
    
    def prepare_data(self) -> Tuple[Any, Any]:
        """Prepare training and evaluation data.

        Returns (train_data, eval_loader): train_data is
        ((afm_train, afm_test), (fm_train, fm_test)); eval_loader is the
        deterministic evaluation loader yielding (history, covariates,
        prediction, extrapolation) windows.
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
        """Create models for the experiment."""
        self.key, afm_key, fm_key = jrandom.split(self.key, 3)
        
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
        
        return {'AFM': afm_model, 'FM': fm_model}
    
    def train_models(self, models: Dict[str, Any], train_data: Any, test_data: Any) -> None:
        """Train all models."""
        # train_data holds ((afm_train, afm_test), (fm_train, fm_test));
        # the evaluation loader is passed separately as test_data.
        (afm_train_loader, afm_test_loader), (fm_train_loader, fm_test_loader) = train_data

        afm_trainer = AFMTrainer(models['AFM'], learning_rate=self.config.training_learning_rate)
        fm_trainer = FMTrainer(models['FM'], learning_rate=self.config.training_learning_rate)

        afm_trainer.train(afm_train_loader, afm_test_loader, num_epochs=self.config.training_num_epochs)
        fm_trainer.train(fm_train_loader, fm_test_loader, num_epochs=getattr(self.config, 'training_num_epochs_fm', self.config.training_num_epochs * 2))
    
    def evaluate_models(self, models: Dict[str, Any], test_data: Any) -> Dict[str, Dict[str, float]]:
        """Evaluate all models."""
        evaluator = ModelEvaluator(self.config, self.config.flow_nfe)
        results = {}
        
        for model_name, model in models.items():
            metrics = evaluator.evaluate_model(model, test_data, self.key)
            results[model_name] = metrics
            
        return results
        
    def run(self) -> Dict[str, Any]:
        """Run history length sweep experiment with AFM vs FM comparison."""
        print(f"Starting AFM vs FM history length sweep: {self.config.experiment_name}")
        print("="*80)
        print("NOTE: This experiment retrains models for each history length")
        print("="*80)
        
        # Setup wandb
        self.setup_wandb()
        
        # Get sweep parameters from config - support both history_size and data_history_size
        sweep_params = getattr(self.config, 'sweep_params', {})
        history_sizes = sweep_params.get('history_size', sweep_params.get('data_history_size', [10, 20, 30, 40, 50]))
        
        print(f"Testing history sizes: {history_sizes}")
        print(f"System: {self.config.system_name} ({'stochastic' if self.config.system_stochastic else 'deterministic'})")
        
        # Run sweep
        for i, history_size in enumerate(history_sizes):
            print(f"\n{'='*20} History Size {history_size} ({i+1}/{len(history_sizes)}) {'='*20}")
            
            # Create modified config for this history size
            sweep_config = Config()
            for attr in dir(self.config):
                if not attr.startswith('_') and hasattr(sweep_config, attr):
                    setattr(sweep_config, attr, getattr(self.config, attr))
            
            # Set the history size for this sweep point
            sweep_config.data_history_size = history_size
            
            # Adjust num_timepoints if necessary to accommodate history + prediction
            min_timepoints = history_size + sweep_config.data_prediction_size + 20  # Extra buffer
            if sweep_config.system_num_timepoints < min_timepoints:
                old_timepoints = sweep_config.system_num_timepoints
                sweep_config.system_num_timepoints = min_timepoints
                print(f"Adjusted num_timepoints: {old_timepoints} -> {min_timepoints}")
            
            # Recompute derived fields
            sweep_config._compute_derived_fields()
            
            # Run single sweep point (this will retrain models)
            print(f"Running experiment with history_size={history_size}...")
            results = self.run_single_sweep_point(sweep_config)
            self.sweep_results[f"history_{history_size}"] = results
            
            # Enhanced logging to wandb
            self._log_sweep_point_results(history_size, results)
            
            # Print summary for this history size
            self._print_history_summary(history_size, results)
        
        # Save results and perform analysis
        self.save_sweep_results()
        self.analyze_sweep_results()
        self.create_comparison_plots()
        
        # Print final summary
        self._print_final_summary()
        
        return self.sweep_results
    
    def _log_sweep_point_results(self, history_size: int, results: Dict[str, Any]) -> None:
        """Log results for a single sweep point to wandb."""
        import wandb
        
        wandb_metrics = {'history_size': history_size}
        
        # Log metrics for each model
        for model_name in ['AFM', 'FM']:
            if model_name in results:
                metrics = results[model_name]
                wandb_metrics.update({
                    f"{model_name}/crps_prediction": metrics.get('crps_prediction', 0),
                    f"{model_name}/nrmse_prediction": metrics.get('nrmse_prediction', 0),
                    f"{model_name}/crps_extrapolation": metrics.get('crps_extrapolation', 0),
                    f"{model_name}/nrmse_extrapolation": metrics.get('nrmse_extrapolation', 0)
                })
        
        # Compute and log comparison metrics if both models present
        if 'AFM' in results and 'FM' in results:
            afm_crps = results['AFM'].get('crps_prediction', float('inf'))
            fm_crps = results['FM'].get('crps_prediction', float('inf'))
            afm_nrmse = results['AFM'].get('nrmse_prediction', float('inf'))
            fm_nrmse = results['FM'].get('nrmse_prediction', float('inf'))
            
            wandb_metrics.update({
                'crps_difference': afm_crps - fm_crps,  # Negative means AFM is better
                'nrmse_difference': afm_nrmse - fm_nrmse,
                'afm_wins_crps': afm_crps < fm_crps,
                'afm_wins_nrmse': afm_nrmse < fm_nrmse,
            })
        
        wandb.log(wandb_metrics)
    
    def _print_history_summary(self, history_size: int, results: Dict[str, Any]) -> None:
        """Print summary results for a single history size."""
        print(f"\nResults for history_size = {history_size}:")
        print("-" * 50)
        
        if 'AFM' in results and 'FM' in results:
            afm_metrics = results['AFM']
            fm_metrics = results['FM']
            
            afm_crps = afm_metrics.get('crps_prediction', float('inf'))
            fm_crps = fm_metrics.get('crps_prediction', float('inf'))
            afm_nrmse = afm_metrics.get('nrmse_prediction', float('inf'))
            fm_nrmse = fm_metrics.get('nrmse_prediction', float('inf'))
            
            crps_winner = "AFM" if afm_crps < fm_crps else "FM"
            nrmse_winner = "AFM" if afm_nrmse < fm_nrmse else "FM"
            
            print(f"{'Model':<6} {'CRPS':<12} {'NRMSE':<12} {'Winner':<8}")
            print("-" * 50)
            print(f"{'AFM':<6} {afm_crps:<12.4f} {afm_nrmse:<12.4f} {crps_winner if crps_winner=='AFM' else '':<8}")
            print(f"{'FM':<6} {fm_crps:<12.4f} {fm_nrmse:<12.4f} {crps_winner if crps_winner=='FM' else '':<8}")
            print(f"Winner: CRPS={crps_winner}, NRMSE={nrmse_winner}")
        else:
            print("Missing results for one or both models")
    
    def _print_final_summary(self) -> None:
        """Print final summary of the experiment."""
        print("\n" + "="*80)
        print("FINAL EXPERIMENT SUMMARY")
        print("="*80)
        
        if self.optimal_histories['AFM'] is not None:
            print(f"Optimal history lengths:")
            print(f"  AFM: {self.optimal_histories['AFM']['size']} (CRPS: {self.optimal_histories['AFM']['crps']:.4f})")
            print(f"  FM:  {self.optimal_histories['FM']['size']} (CRPS: {self.optimal_histories['FM']['crps']:.4f})")
        
        # Count wins
        afm_wins = 0
        fm_wins = 0
        for key, results in self.sweep_results.items():
            if key.startswith("history_") and 'AFM' in results and 'FM' in results:
                afm_crps = results['AFM'].get('crps_prediction', float('inf'))
                fm_crps = results['FM'].get('crps_prediction', float('inf'))
                if afm_crps < fm_crps:
                    afm_wins += 1
                else:
                    fm_wins += 1
        
        total_comparisons = afm_wins + fm_wins
        if total_comparisons > 0:
            print(f"\nOverall comparison:")
            print(f"  AFM wins: {afm_wins}/{total_comparisons} ({100*afm_wins/total_comparisons:.1f}%)")
            print(f"  FM wins:  {fm_wins}/{total_comparisons} ({100*fm_wins/total_comparisons:.1f}%)")
        
        print("="*80)
    
    def analyze_sweep_results(self) -> None:
        """Analyze sweep results and find optimal history length for AFM vs FM."""
        print("\n" + "="*60)
        print("HISTORY LENGTH AFM vs FM ANALYSIS")
        print("="*60)
        
        history_sizes = []
        afm_crps = []
        fm_crps = []
        afm_nrmse = []
        fm_nrmse = []
        
        # Extract data from sweep results
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
        
        print("\nHistory Size vs Performance Comparison:")
        print("-" * 85)
        print(f"{'History':<8} {'AFM CRPS':<12} {'FM CRPS':<12} {'Δ CRPS':<12} {'AFM NRMSE':<12} {'FM NRMSE':<12} {'Winner':<8}")
        print("-" * 85)
        
        afm_wins = 0
        fm_wins = 0
        
        for i in sorted_indices:
            hist = history_sizes[i]
            a_crps = afm_crps[i]
            f_crps = fm_crps[i] 
            a_nrmse = afm_nrmse[i]
            f_nrmse = fm_nrmse[i]
            
            delta_crps = a_crps - f_crps  # Negative means AFM is better
            winner = "AFM" if a_crps < f_crps else "FM"
            if a_crps < f_crps:
                afm_wins += 1
            else:
                fm_wins += 1
            
            print(f"{hist:<8} {a_crps:<12.4f} {f_crps:<12.4f} {delta_crps:<+12.4f} {a_nrmse:<12.4f} {f_nrmse:<12.4f} {winner:<8}")
        
        # Find optimal history lengths for each model
        best_afm_idx = min(range(len(afm_crps)), key=lambda i: afm_crps[i])
        best_fm_idx = min(range(len(fm_crps)), key=lambda i: fm_crps[i])
        
        # Store optimal histories for final summary
        self.optimal_histories['AFM'] = {
            'size': history_sizes[best_afm_idx],
            'crps': afm_crps[best_afm_idx],
            'nrmse': afm_nrmse[best_afm_idx]
        }
        self.optimal_histories['FM'] = {
            'size': history_sizes[best_fm_idx], 
            'crps': fm_crps[best_fm_idx],
            'nrmse': fm_nrmse[best_fm_idx]
        }
        
        print(f"\nOptimal History Lengths (by CRPS):")
        print(f"AFM best: {history_sizes[best_afm_idx]} (CRPS: {afm_crps[best_afm_idx]:.4f}, NRMSE: {afm_nrmse[best_afm_idx]:.4f})")
        print(f"FM best:  {history_sizes[best_fm_idx]} (CRPS: {fm_crps[best_fm_idx]:.4f}, NRMSE: {fm_nrmse[best_fm_idx]:.4f})")
        
        total_comparisons = afm_wins + fm_wins
        print(f"\nOverall Winner Summary:")
        print(f"AFM wins: {afm_wins}/{total_comparisons} ({100*afm_wins/total_comparisons:.1f}%)")
        print(f"FM wins:  {fm_wins}/{total_comparisons} ({100*fm_wins/total_comparisons:.1f}%)")
        
        # Find best overall performance
        best_overall_crps = min(min(afm_crps), min(fm_crps))
        if min(afm_crps) == best_overall_crps:
            best_model = "AFM"
            best_hist = history_sizes[afm_crps.index(best_overall_crps)]
        else:
            best_model = "FM" 
            best_hist = history_sizes[fm_crps.index(best_overall_crps)]
        
        print(f"\nBest Overall Performance:")
        print(f"Model: {best_model}, History: {best_hist}, CRPS: {best_overall_crps:.4f}")
        print("="*60)
    
    def create_comparison_plots(self) -> None:
        """Create comprehensive comparison plots for AFM vs FM across history lengths."""
        print("\nCreating AFM vs FM comparison plots...")
        
        # Extract data for plotting
        history_sizes = []
        afm_crps = []
        fm_crps = []
        afm_nrmse = []
        fm_nrmse = []
        
        for key, results in self.sweep_results.items():
            if key.startswith("history_"):
                history_size = int(key.split("_")[1])
                history_sizes.append(history_size)
                
                afm_crps.append(results.get('AFM', {}).get('crps_prediction', float('inf')))
                fm_crps.append(results.get('FM', {}).get('crps_prediction', float('inf')))
                afm_nrmse.append(results.get('AFM', {}).get('nrmse_prediction', float('inf')))
                fm_nrmse.append(results.get('FM', {}).get('nrmse_prediction', float('inf')))
        
        # Sort data by history size
        sorted_data = sorted(zip(history_sizes, afm_crps, fm_crps, afm_nrmse, fm_nrmse))
        history_sorted = [x[0] for x in sorted_data]
        afm_crps_sorted = [x[1] for x in sorted_data]
        fm_crps_sorted = [x[2] for x in sorted_data]
        afm_nrmse_sorted = [x[3] for x in sorted_data]
        fm_nrmse_sorted = [x[4] for x in sorted_data]
        
        # Create results directory
        results_dir = Path(self.config.experiment_results_dir) / self.config.system_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive comparison plots
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: CRPS vs History Length
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(history_sorted, afm_crps_sorted, 'o-', label='AFM', linewidth=2, markersize=8, color='#1f77b4')
        ax1.plot(history_sorted, fm_crps_sorted, 's-', label='FM', linewidth=2, markersize=8, color='#ff7f0e')
        ax1.set_xlabel('History Length', fontsize=12)
        ax1.set_ylabel('CRPS (Prediction)', fontsize=12)
        ax1.set_title('CRPS vs History Length', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: NRMSE vs History Length
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(history_sorted, afm_nrmse_sorted, 'o-', label='AFM', linewidth=2, markersize=8, color='#1f77b4')
        ax2.plot(history_sorted, fm_nrmse_sorted, 's-', label='FM', linewidth=2, markersize=8, color='#ff7f0e')
        ax2.set_xlabel('History Length', fontsize=12)
        ax2.set_ylabel('NRMSE (Prediction)', fontsize=12)
        ax2.set_title('NRMSE vs History Length', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance Difference (AFM - FM)
        ax3 = plt.subplot(2, 3, 3)
        crps_diff = [a - f for a, f in zip(afm_crps_sorted, fm_crps_sorted)]
        nrmse_diff = [a - f for a, f in zip(afm_nrmse_sorted, fm_nrmse_sorted)]
        
        ax3.plot(history_sorted, crps_diff, 'o-', label='CRPS Difference', linewidth=2, markersize=8, color='#2ca02c')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('History Length', fontsize=12)
        ax3.set_ylabel('Performance Difference (AFM - FM)', fontsize=12)
        ax3.set_title('AFM vs FM Performance Gap', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Add shading for AFM better/worse regions
        ax3.fill_between(history_sorted, 0, crps_diff, where=[d < 0 for d in crps_diff], 
                        alpha=0.3, color='#1f77b4', label='AFM Better')
        ax3.fill_between(history_sorted, 0, crps_diff, where=[d > 0 for d in crps_diff], 
                        alpha=0.3, color='#ff7f0e', label='FM Better')
        
        # Plot 4: Bar Chart - Win Counts
        ax4 = plt.subplot(2, 3, 4)
        afm_wins = sum(1 for a, f in zip(afm_crps_sorted, fm_crps_sorted) if a < f)
        fm_wins = len(afm_crps_sorted) - afm_wins
        
        models = ['AFM', 'FM']
        wins = [afm_wins, fm_wins]
        colors = ['#1f77b4', '#ff7f0e']
        
        bars = ax4.bar(models, wins, color=colors, alpha=0.7)
        ax4.set_ylabel('Number of Wins', fontsize=12)
        ax4.set_title('Overall Winner Count', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, win in zip(bars, wins):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{win}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Plot 5: Optimal History Comparison
        ax5 = plt.subplot(2, 3, 5)
        if self.optimal_histories['AFM'] and self.optimal_histories['FM']:
            opt_histories = [self.optimal_histories['AFM']['size'], self.optimal_histories['FM']['size']]
            opt_crps = [self.optimal_histories['AFM']['crps'], self.optimal_histories['FM']['crps']]
            
            bars = ax5.bar(models, opt_histories, color=colors, alpha=0.7)
            ax5.set_ylabel('Optimal History Length', fontsize=12)
            ax5.set_title('Optimal History Length by Model', fontsize=14, fontweight='bold')
            
            # Add CRPS value labels
            for i, (bar, hist, crps) in enumerate(zip(bars, opt_histories, opt_crps)):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{hist}\n(CRPS: {crps:.3f})', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')
        
        # Plot 6: Relative Performance (FM/AFM ratio)
        ax6 = plt.subplot(2, 3, 6)
        crps_ratio = [f/a if a > 0 else 1 for a, f in zip(afm_crps_sorted, fm_crps_sorted)]
        
        ax6.plot(history_sorted, crps_ratio, 'o-', linewidth=2, markersize=8, color='#d62728')
        ax6.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
        ax6.set_xlabel('History Length', fontsize=12)
        ax6.set_ylabel('FM/AFM CRPS Ratio', fontsize=12)
        ax6.set_title('Relative Performance Ratio', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=11)
        ax6.grid(True, alpha=0.3)
        
        # Add shading
        ax6.fill_between(history_sorted, 1, crps_ratio, where=[r > 1 for r in crps_ratio], 
                        alpha=0.3, color='#1f77b4', label='AFM Better')
        ax6.fill_between(history_sorted, 1, crps_ratio, where=[r < 1 for r in crps_ratio], 
                        alpha=0.3, color='#ff7f0e', label='FM Better')
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        plot_path = results_dir / f"{self.config.experiment_name.replace(' ', '_')}_afm_vs_fm_comparison.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"AFM vs FM comparison plots saved to: {plot_path}")
        
        # Upload to wandb if configured
        wandb_config = getattr(self.config, 'wandb', {}) or {}
        if wandb_config.get('upload_plots', False):
            try:
                import wandb
                wandb.log({"AFM_vs_FM_History_Comparison": wandb.Image(str(plot_path))})
                print("Plots uploaded to WandB")
            except Exception as e:
                print(f"Warning: Failed to upload plots to WandB: {e}")
        
        plt.show()
        plt.close()


class NFEEfficiencyExperiment(ParameterSweepExperiment):
    """Experiment that sweeps over NFE values to study efficiency."""
    
    def prepare_data(self) -> Tuple[Any, Any]:
        """Prepare training and evaluation data.

        Returns (train_data, eval_loader): train_data is
        ((afm_train, afm_test), (fm_train, fm_test)); eval_loader is the
        deterministic evaluation loader yielding (history, covariates,
        prediction, extrapolation) windows.
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
        """Create models for the experiment."""
        self.key, afm_key, fm_key = jrandom.split(self.key, 3)
        
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
        
        return {'AFM': afm_model, 'FM': fm_model}
    
    def train_models(self, models: Dict[str, Any], train_data: Any, test_data: Any) -> None:
        """Train all models."""
        # train_data holds ((afm_train, afm_test), (fm_train, fm_test));
        # the evaluation loader is passed separately as test_data.
        (afm_train_loader, afm_test_loader), (fm_train_loader, fm_test_loader) = train_data

        afm_trainer = AFMTrainer(models['AFM'], learning_rate=self.config.training_learning_rate)
        fm_trainer = FMTrainer(models['FM'], learning_rate=self.config.training_learning_rate)

        afm_trainer.train(afm_train_loader, afm_test_loader, num_epochs=self.config.training_num_epochs)
        fm_trainer.train(fm_train_loader, fm_test_loader, num_epochs=getattr(self.config, 'training_num_epochs_fm', self.config.training_num_epochs * 2))
    
    def evaluate_models(self, models: Dict[str, Any], test_data: Any) -> Dict[str, Dict[str, float]]:
        """Evaluate all models."""
        evaluator = ModelEvaluator(self.config, self.config.flow_nfe)
        results = {}
        
        for model_name, model in models.items():
            metrics = evaluator.evaluate_model(model, test_data, self.key)
            results[model_name] = metrics
            
        return results
    
    def run(self) -> Dict[str, Any]:
        """Run NFE efficiency experiment."""
        print(f"Starting NFE efficiency experiment: {self.config.experiment_name}")
        
        # Setup wandb
        self.setup_wandb()
        
        # Train models once with original config
        print("Training models once...")
        train_data, test_data = self.prepare_data()
        models = self.create_models()
        self.train_models(models, train_data, test_data)
        print("Training completed!")
        
        # Get sweep parameters
        sweep_params = getattr(self.config, 'sweep_params', {})
        nfe_values = sweep_params.get('nfe_values', [5, 10, 15, 20, 30, 40, 50])
        
        # Evaluate with different NFE values
        for nfe in nfe_values:
            print(f"\n--- Testing NFE = {nfe} ---")
            
            # Measure timing for inference only
            import time
            start_time = time.time()
            
            # Create evaluator with specific NFE
            evaluator = ModelEvaluator(self.config, nfe)
            results = {}
            
            # Generate new key for each NFE to ensure different randomness
            self.key, eval_key = jrandom.split(self.key)
            
            for model_name, model in models.items():
                metrics = evaluator.evaluate_model(model, test_data, eval_key)
                results[model_name] = metrics
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Add timing info
            results['timing'] = {
                'inference_time': inference_time,
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
                        f"{model_name}/inference_time": inference_time
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
                timings.append(results.get('timing', {}).get('inference_time', 0))
                
                afm_crps.append(results.get('AFM', {}).get('crps_prediction', float('inf')))
                fm_crps.append(results.get('FM', {}).get('crps_prediction', float('inf')))
        
        # Sort by NFE
        sorted_indices = sorted(range(len(nfe_values)), key=lambda i: nfe_values[i])
        
        print("\nNFE vs Performance vs Inference Time:")
        print("-" * 70)
        print(f"{'NFE':<5} {'AFM CRPS':<12} {'FM CRPS':<12} {'Inf Time (s)':<12} {'Winner':<8}")
        print("-" * 70)
        
        for i in sorted_indices:
            nfe = nfe_values[i]
            a_crps = afm_crps[i]
            f_crps = fm_crps[i]
            time_s = timings[i]
            winner = "AFM" if a_crps < f_crps else "FM"
            
            print(f"{nfe:<5} {a_crps:<12.4f} {f_crps:<12.4f} {time_s:<12.2f} {winner:<8}")
        
        print("="*60)
        
        # Create plots
        self.plot_nfe_results(nfe_values, afm_crps, fm_crps, timings)
    
    def plot_nfe_results(self, nfe_values: List[int], afm_crps: List[float], fm_crps: List[float], timings: List[float]) -> None:
        """Create plots for NFE efficiency analysis."""
        # Sort data by NFE values
        sorted_data = sorted(zip(nfe_values, afm_crps, fm_crps, timings))
        nfe_sorted = [x[0] for x in sorted_data]
        afm_crps_sorted = [x[1] for x in sorted_data]
        fm_crps_sorted = [x[2] for x in sorted_data]
        timings_sorted = [x[3] for x in sorted_data]
        
        # Also get NRMSE data
        afm_nrmse = []
        fm_nrmse = []
        for key, results in self.sweep_results.items():
            if key.startswith("nfe_"):
                nfe = int(key.split("_")[1])
                afm_nrmse.append(results.get('AFM', {}).get('nrmse_prediction', float('inf')))
                fm_nrmse.append(results.get('FM', {}).get('nrmse_prediction', float('inf')))
        
        # Sort NRMSE data by NFE
        nrmse_sorted_data = sorted(zip(nfe_values, afm_nrmse, fm_nrmse))
        afm_nrmse_sorted = [x[1] for x in nrmse_sorted_data]
        fm_nrmse_sorted = [x[2] for x in nrmse_sorted_data]
        
        # Create results directory
        results_dir = Path(self.config.experiment_results_dir) / self.config.system_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: CRPS vs NFE
        ax1.plot(nfe_sorted, afm_crps_sorted, 'o-', label='AFM', linewidth=2, markersize=6)
        ax1.plot(nfe_sorted, fm_crps_sorted, 's-', label='FM', linewidth=2, markersize=6)
        ax1.set_xlabel('Number of Function Evaluations (NFE)', fontsize=12)
        ax1.set_ylabel('CRPS (Prediction)', fontsize=12)
        ax1.set_title('CRPS vs NFE', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log') if max(nfe_sorted) > 10*min(nfe_sorted) else None
        
        # Plot 2: NRMSE vs NFE
        ax2.plot(nfe_sorted, afm_nrmse_sorted, 'o-', label='AFM', linewidth=2, markersize=6)
        ax2.plot(nfe_sorted, fm_nrmse_sorted, 's-', label='FM', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Function Evaluations (NFE)', fontsize=12)
        ax2.set_ylabel('NRMSE (Prediction)', fontsize=12)
        ax2.set_title('NRMSE vs NFE', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log') if max(nfe_sorted) > 10*min(nfe_sorted) else None
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = results_dir / f"{self.config.experiment_name.replace(' ', '_')}_nfe_performance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPerformance plots saved to: {plot_path}")
        
        # Also create a combined efficiency plot (performance vs time)
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 3: CRPS vs Inference Time
        ax3.scatter(timings_sorted, afm_crps_sorted, label='AFM', s=60, alpha=0.7)
        ax3.scatter(timings_sorted, fm_crps_sorted, label='FM', s=60, alpha=0.7, marker='s')
        
        # Add NFE annotations
        for i, nfe in enumerate(nfe_sorted):
            ax3.annotate(f'{nfe}', (timings_sorted[i], afm_crps_sorted[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax3.annotate(f'{nfe}', (timings_sorted[i], fm_crps_sorted[i]), 
                        xytext=(5, -10), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Inference Time (seconds)', fontsize=12)
        ax3.set_ylabel('CRPS (Prediction)', fontsize=12)
        ax3.set_title('Accuracy vs Inference Time (CRPS)', fontsize=14)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: NRMSE vs Inference Time
        ax4.scatter(timings_sorted, afm_nrmse_sorted, label='AFM', s=60, alpha=0.7)
        ax4.scatter(timings_sorted, fm_nrmse_sorted, label='FM', s=60, alpha=0.7, marker='s')
        
        # Add NFE annotations
        for i, nfe in enumerate(nfe_sorted):
            ax4.annotate(f'{nfe}', (timings_sorted[i], afm_nrmse_sorted[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax4.annotate(f'{nfe}', (timings_sorted[i], fm_nrmse_sorted[i]), 
                        xytext=(5, -10), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Inference Time (seconds)', fontsize=12)
        ax4.set_ylabel('NRMSE (Prediction)', fontsize=12)
        ax4.set_title('Accuracy vs Inference Time (NRMSE)', fontsize=14)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the efficiency plot
        efficiency_path = results_dir / f"{self.config.experiment_name.replace(' ', '_')}_nfe_efficiency.png"
        plt.savefig(efficiency_path, dpi=300, bbox_inches='tight')
        print(f"Efficiency plots saved to: {efficiency_path}")
        
        plt.show()
"""Multi-system experiments for comparing performance across different dynamical systems."""

from typing import Dict, Any, List, Tuple
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from pathlib import Path
import json

from .base_experiment import BaseExperiment
from ..config_loader import Config
from ..systems.systems import create_system
from ..models import AutoregressiveFlowMatching, FlowMatching
from ..training.data_preparation import prepare_data
from ..training.trainers import AFMTrainer, FMTrainer
from ..evaluation.evaluator import ModelEvaluator


class MultiSystemExperiment(BaseExperiment):
    """Base class for experiments that test multiple dynamical systems."""
    
    def __init__(self, config: Config, seed: int = 0):
        """Initialize multi-system experiment."""
        super().__init__(config, seed)
        self.system_results = {}
        
    def run_single_system(self, system_name: str, stochastic: bool) -> Dict[str, Any]:
        """Run experiment on a single system.
        
        Args:
            system_name: Name of the dynamical system
            stochastic: Whether to use stochastic version
            
        Returns:
            Dictionary containing results for this system
        """
        print(f"\n--- Testing {system_name} ({'stochastic' if stochastic else 'deterministic'}) ---")
        
        # Create system
        system = create_system(system_name, stochastic, self.config.system_t0, self.config.system_t1)
        
        # Create modified config for this system
        system_config = Config()
        for attr in dir(self.config):
            if not attr.startswith('_') and hasattr(system_config, attr):
                setattr(system_config, attr, getattr(self.config, attr))
        
        system_config.system_name = system_name
        system_config.system_stochastic = stochastic
        system_config._compute_derived_fields()
        
        # Prepare data
        train_data, test_data = prepare_data(system, system_config, self.key)
        
        # Create models
        self.key, afm_key, fm_key = jrandom.split(self.key, 3)
        
        afm_model = AutoregressiveFlowMatching(
            dim_x=system.data_dim,
            dim_c=system_config.model_covariates_dim,
            dim_h=system_config.model_hidden_dim,
            dim_t=system_config.model_time_dim,
            dim_hidden=system_config.model_hidden_dim,
            hidden_blocks=system_config.model_hidden_blocks,
            n_rnn_layers=system_config.model_n_rnn_layers,
            rnn_hidden=system_config.model_rnn_hidden,
            rngs=jax.nn.initializers.Rngs(afm_key)
        )
        
        fm_model = FlowMatching(
            dim_x=system.data_dim,
            dim_c=system_config.model_covariates_dim,
            dim_h=system_config.model_hidden_dim,
            dim_t=system_config.model_time_dim,
            dim_hidden=system_config.model_hidden_dim,
            hidden_blocks=system_config.model_hidden_blocks,
            n_rnn_layers=system_config.model_n_rnn_layers,
            rngs=jax.nn.initializers.Rngs(fm_key)
        )
        
        # Train models
        print("Training AFM...")
        afm_trainer = AFMTrainer(afm_model, learning_rate=system_config.training_learning_rate)
        afm_trainer.train(
            train_data, test_data, 
            num_epochs=getattr(system_config, 'training_num_epochs_afm', system_config.training_num_epochs)
        )
        
        print("Training FM...")
        fm_trainer = FMTrainer(fm_model, learning_rate=system_config.training_learning_rate)
        fm_trainer.train(
            train_data, test_data,
            num_epochs=getattr(system_config, 'training_num_epochs_fm', system_config.training_num_epochs * 2)
        )
        
        # Evaluate models
        evaluator = ModelEvaluator(system_config, system_config.flow_nfe)
        results = {}
        
        for model_name, model in [('AFM', afm_model), ('FM', fm_model)]:
            print(f"Evaluating {model_name}...")
            metrics = evaluator.evaluate_model(model, test_data, self.key)
            results[model_name] = metrics
        
        # Add system info
        results['system_info'] = {
            'name': system_name,
            'stochastic': stochastic,
            'data_dim': system.data_dim
        }
        
        return results
    
    def save_system_results(self) -> None:
        """Save multi-system results to file."""
        results_dir = Path(self.config.experiment_results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"{self.config.experiment_name}_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in self.system_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        serializable_results[key][k] = {}
                        for k2, v2 in v.items():
                            if isinstance(v2, (jnp.ndarray, np.ndarray)):
                                serializable_results[key][k][k2] = v2.tolist()
                            else:
                                serializable_results[key][k][k2] = v2
                    elif isinstance(v, (jnp.ndarray, np.ndarray)):
                        serializable_results[key][k] = v.tolist()
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Saved multi-system results to {results_file}")


class AFMvsFMAllSystemsExperiment(MultiSystemExperiment):
    """Experiment comparing AFM vs FM across all dynamical systems."""
    
    def run(self) -> Dict[str, Any]:
        """Run AFM vs FM comparison across all systems."""
        print(f"Starting multi-system experiment: {self.config.experiment_name}")
        
        # Setup wandb
        self.setup_wandb()
        
        # Get systems to test from config
        systems_config = getattr(self.config, 'systems', [
            {'name': 'lorenz', 'stochastic': True},
            {'name': 'lorenz', 'stochastic': False},
            {'name': 'vanderpol', 'stochastic': True},
            {'name': 'lotka_volterra', 'stochastic': True},
            {'name': 'harmonic_oscillator', 'stochastic': False},
        ])
        
        # Run experiments on all systems
        for system_config in systems_config:
            system_name = system_config['name']
            stochastic = system_config['stochastic']
            
            system_key = f"{system_name}_{'stochastic' if stochastic else 'deterministic'}"
            
            try:
                results = self.run_single_system(system_name, stochastic)
                self.system_results[system_key] = results
                
                # Log to wandb
                for model_name in ['AFM', 'FM']:
                    if model_name in results:
                        metrics = results[model_name]
                        wandb_metrics = {
                            f"{model_name}/system": system_name,
                            f"{model_name}/stochastic": stochastic,
                            f"{model_name}/crps_prediction": metrics.get('crps_prediction', 0),
                            f"{model_name}/nrmse_prediction": metrics.get('nrmse_prediction', 0),
                            f"{model_name}/crps_extrapolation": metrics.get('crps_extrapolation', 0),
                            f"{model_name}/nrmse_extrapolation": metrics.get('nrmse_extrapolation', 0)
                        }
                        import wandb
                        wandb.log(wandb_metrics)
                        
            except Exception as e:
                print(f"Error running {system_key}: {e}")
                self.system_results[system_key] = {'error': str(e)}
        
        # Save and analyze results
        self.save_system_results()
        self.analyze_multi_system_results()
        
        return self.system_results
    
    def analyze_multi_system_results(self) -> None:
        """Analyze results across all systems."""
        print("\n" + "="*80)
        print("MULTI-SYSTEM ANALYSIS")
        print("="*80)
        
        # Collect results
        systems = []
        afm_scores = []
        fm_scores = []
        winners = []
        
        print("\nSystem-by-System Results:")
        print("-" * 100)
        print(f"{'System':<25} {'AFM CRPS':<12} {'FM CRPS':<12} {'AFM NRMSE':<12} {'FM NRMSE':<12} {'Winner':<8}")
        print("-" * 100)
        
        for system_key, results in self.system_results.items():
            if 'error' in results:
                print(f"{system_key:<25} ERROR: {results['error']}")
                continue
                
            systems.append(system_key)
            
            afm_crps = results.get('AFM', {}).get('crps_prediction', float('inf'))
            fm_crps = results.get('FM', {}).get('crps_prediction', float('inf'))
            afm_nrmse = results.get('AFM', {}).get('nrmse_prediction', float('inf'))
            fm_nrmse = results.get('FM', {}).get('nrmse_prediction', float('inf'))
            
            afm_scores.append(afm_crps)
            fm_scores.append(fm_crps)
            
            winner = "AFM" if afm_crps < fm_crps else "FM"
            winners.append(winner)
            
            print(f"{system_key:<25} {afm_crps:<12.4f} {fm_crps:<12.4f} {afm_nrmse:<12.4f} {fm_nrmse:<12.4f} {winner:<8}")
        
        print("-" * 100)
        
        # Summary statistics
        afm_wins = winners.count('AFM')
        fm_wins = winners.count('FM')
        
        print(f"\nSummary:")
        print(f"AFM wins: {afm_wins}/{len(winners)} systems")
        print(f"FM wins:  {fm_wins}/{len(winners)} systems")
        
        if len(afm_scores) > 0:
            afm_mean = np.mean(afm_scores)
            fm_mean = np.mean(fm_scores)
            print(f"Mean CRPS - AFM: {afm_mean:.4f}, FM: {fm_mean:.4f}")
            
            # Find best and worst systems for each method
            best_afm_idx = np.argmin(afm_scores)
            worst_afm_idx = np.argmax(afm_scores)
            best_fm_idx = np.argmin(fm_scores)
            worst_fm_idx = np.argmax(fm_scores)
            
            print(f"\nAFM performs best on: {systems[best_afm_idx]} (CRPS: {afm_scores[best_afm_idx]:.4f})")
            print(f"AFM performs worst on: {systems[worst_afm_idx]} (CRPS: {afm_scores[worst_afm_idx]:.4f})")
            print(f"FM performs best on: {systems[best_fm_idx]} (CRPS: {fm_scores[best_fm_idx]:.4f})")
            print(f"FM performs worst on: {systems[worst_fm_idx]} (CRPS: {fm_scores[worst_fm_idx]:.4f})")
        
        print("="*80)


class StochasticVsDeterministicExperiment(MultiSystemExperiment):
    """Experiment comparing performance on stochastic vs deterministic systems."""
    
    def run(self) -> Dict[str, Any]:
        """Run stochastic vs deterministic comparison."""
        print(f"Starting stochastic vs deterministic experiment: {self.config.experiment_name}")
        
        # Setup wandb
        self.setup_wandb()
        
        # Get systems to test
        system_names = getattr(self.config, 'systems', ['lorenz', 'vanderpol', 'lotka_volterra'])
        
        # Test each system in both stochastic and deterministic modes
        for system_name in system_names:
            for stochastic in [False, True]:
                system_key = f"{system_name}_{'stochastic' if stochastic else 'deterministic'}"
                
                try:
                    results = self.run_single_system(system_name, stochastic)
                    self.system_results[system_key] = results
                    
                    # Log to wandb
                    for model_name in ['AFM', 'FM']:
                        if model_name in results:
                            metrics = results[model_name]
                            wandb_metrics = {
                                f"{model_name}/system": system_name,
                                f"{model_name}/stochastic": stochastic,
                                f"{model_name}/crps_prediction": metrics.get('crps_prediction', 0),
                                f"{model_name}/nrmse_prediction": metrics.get('nrmse_prediction', 0)
                            }
                            import wandb
                            wandb.log(wandb_metrics)
                            
                except Exception as e:
                    print(f"Error running {system_key}: {e}")
                    self.system_results[system_key] = {'error': str(e)}
        
        # Save and analyze results
        self.save_system_results()
        self.analyze_stochastic_deterministic_results()
        
        return self.system_results
    
    def analyze_stochastic_deterministic_results(self) -> None:
        """Analyze stochastic vs deterministic results."""
        print("\n" + "="*80)
        print("STOCHASTIC vs DETERMINISTIC ANALYSIS")
        print("="*80)
        
        # Group results by system
        systems = {}
        for system_key, results in self.system_results.items():
            if 'error' in results:
                continue
                
            parts = system_key.split('_')
            system_name = parts[0]
            mode = parts[1]
            
            if system_name not in systems:
                systems[system_name] = {}
            systems[system_name][mode] = results
        
        print("\nStochastic vs Deterministic Comparison:")
        print("-" * 120)
        print(f"{'System':<15} {'Mode':<12} {'AFM CRPS':<12} {'FM CRPS':<12} {'AFM Winner':<12} {'Improvement':<15}")
        print("-" * 120)
        
        for system_name, modes in systems.items():
            for mode in ['deterministic', 'stochastic']:
                if mode in modes:
                    results = modes[mode]
                    afm_crps = results.get('AFM', {}).get('crps_prediction', float('inf'))
                    fm_crps = results.get('FM', {}).get('crps_prediction', float('inf'))
                    afm_winner = "Yes" if afm_crps < fm_crps else "No"
                    improvement = f"{((fm_crps - afm_crps) / fm_crps * 100):.1f}%" if fm_crps != float('inf') else "N/A"
                    
                    print(f"{system_name:<15} {mode:<12} {afm_crps:<12.4f} {fm_crps:<12.4f} {afm_winner:<12} {improvement:<15}")
            
            # Compare stochastic vs deterministic for same system
            if 'stochastic' in modes and 'deterministic' in modes:
                det_afm = modes['deterministic'].get('AFM', {}).get('crps_prediction', float('inf'))
                sto_afm = modes['stochastic'].get('AFM', {}).get('crps_prediction', float('inf'))
                det_fm = modes['deterministic'].get('FM', {}).get('crps_prediction', float('inf'))
                sto_fm = modes['stochastic'].get('FM', {}).get('crps_prediction', float('inf'))
                
                afm_degradation = ((sto_afm - det_afm) / det_afm * 100) if det_afm != float('inf') else 0
                fm_degradation = ((sto_fm - det_fm) / det_fm * 100) if det_fm != float('inf') else 0
                
                print(f"{system_name:<15} {'degradation':<12} {afm_degradation:<12.1f}% {fm_degradation:<12.1f}% {'--':<12} {'--':<15}")
            
            print("-" * 60)
        
        print("="*80)
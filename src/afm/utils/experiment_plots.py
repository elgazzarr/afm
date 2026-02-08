"""Experiment-specific plotting utilities with enhanced wandb integration."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import wandb


def setup_matplotlib_style():
    """Setup matplotlib for publication-ready plots."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3
    })


def upload_plot_to_wandb(fig, plot_name: str, upload_enabled: bool = True) -> None:
    """Upload matplotlib figure to wandb.
    
    Args:
        fig: Matplotlib figure object
        plot_name: Name for the plot in wandb
        upload_enabled: Whether to upload to wandb
    """
    if upload_enabled and wandb.run is not None:
        try:
            wandb.log({plot_name: wandb.Image(fig)})
        except Exception as e:
            print(f"Warning: Failed to upload plot {plot_name} to wandb: {e}")


def plot_history_sensitivity_results(
    sweep_results: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "History Length Sensitivity Analysis",
    upload_to_wandb: bool = True
) -> None:
    """Plot history length sensitivity results.
    
    Args:
        sweep_results: Results from history length sweep
        save_path: Path to save figure
        title: Plot title
    """
    setup_matplotlib_style()
    
    # Extract data
    history_sizes = []
    afm_crps = []
    fm_crps = []
    afm_nrmse = []
    fm_nrmse = []
    
    for key, results in sweep_results.items():
        if key.startswith("history_") and 'AFM' in results and 'FM' in results:
            history_size = int(key.split("_")[1])
            history_sizes.append(history_size)
            
            afm_crps.append(results['AFM'].get('crps_prediction', np.nan))
            fm_crps.append(results['FM'].get('crps_prediction', np.nan))
            afm_nrmse.append(results['AFM'].get('nrmse_prediction', np.nan))
            fm_nrmse.append(results['FM'].get('nrmse_prediction', np.nan))
    
    # Sort by history size
    sorted_indices = np.argsort(history_sizes)
    history_sizes = [history_sizes[i] for i in sorted_indices]
    afm_crps = [afm_crps[i] for i in sorted_indices]
    fm_crps = [fm_crps[i] for i in sorted_indices]
    afm_nrmse = [afm_nrmse[i] for i in sorted_indices]
    fm_nrmse = [fm_nrmse[i] for i in sorted_indices]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # CRPS plot
    ax1.plot(history_sizes, afm_crps, 'o-', label='AFM', color='#2E86AB', linewidth=2, markersize=8)
    ax1.plot(history_sizes, fm_crps, 's-', label='FM', color='#A23B72', linewidth=2, markersize=8)
    ax1.set_xlabel('History Length')
    ax1.set_ylabel('CRPS (Prediction)')
    ax1.set_title('CRPS vs History Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Find optimal points
    if afm_crps:
        best_afm_idx = np.argmin(afm_crps)
        best_fm_idx = np.argmin(fm_crps)
        ax1.axvline(history_sizes[best_afm_idx], color='#2E86AB', linestyle='--', alpha=0.7, 
                   label=f'AFM Optimal: {history_sizes[best_afm_idx]}')
        ax1.axvline(history_sizes[best_fm_idx], color='#A23B72', linestyle='--', alpha=0.7,
                   label=f'FM Optimal: {history_sizes[best_fm_idx]}')
    
    # NRMSE plot
    ax2.plot(history_sizes, afm_nrmse, 'o-', label='AFM', color='#2E86AB', linewidth=2, markersize=8)
    ax2.plot(history_sizes, fm_nrmse, 's-', label='FM', color='#A23B72', linewidth=2, markersize=8)
    ax2.set_xlabel('History Length')
    ax2.set_ylabel('NRMSE (Prediction)')
    ax2.set_title('NRMSE vs History Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved history sensitivity plot to {save_path}")
    
    # Upload to wandb
    upload_plot_to_wandb(fig, "plots/history_sensitivity_analysis", upload_to_wandb)
    
    plt.show()


def plot_nfe_efficiency_results(
    sweep_results: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "NFE Efficiency Analysis"
) -> None:
    """Plot NFE efficiency results.
    
    Args:
        sweep_results: Results from NFE efficiency sweep
        save_path: Path to save figure
        title: Plot title
    """
    setup_matplotlib_style()
    
    # Extract data
    nfe_values = []
    afm_crps = []
    fm_crps = []
    timings = []
    
    for key, results in sweep_results.items():
        if key.startswith("nfe_") and 'AFM' in results and 'FM' in results:
            nfe = int(key.split("_")[1])
            nfe_values.append(nfe)
            
            afm_crps.append(results['AFM'].get('crps_prediction', np.nan))
            fm_crps.append(results['FM'].get('crps_prediction', np.nan))
            timings.append(results.get('timing', {}).get('total_time', np.nan))
    
    # Sort by NFE
    sorted_indices = np.argsort(nfe_values)
    nfe_values = [nfe_values[i] for i in sorted_indices]
    afm_crps = [afm_crps[i] for i in sorted_indices]
    fm_crps = [fm_crps[i] for i in sorted_indices]
    timings = [timings[i] for i in sorted_indices]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy vs NFE
    ax1.plot(nfe_values, afm_crps, 'o-', label='AFM', color='#2E86AB', linewidth=2, markersize=8)
    ax1.plot(nfe_values, fm_crps, 's-', label='FM', color='#A23B72', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Function Evaluations (NFE)')
    ax1.set_ylabel('CRPS (Prediction)')
    ax1.set_title('Accuracy vs NFE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Efficiency frontier (Accuracy vs Time)
    ax2.scatter(timings, afm_crps, label='AFM', color='#2E86AB', s=100, alpha=0.7)
    ax2.scatter(timings, fm_crps, label='FM', color='#A23B72', s=100, alpha=0.7, marker='s')
    
    # Annotate points with NFE values
    for i, (t, a_crps, f_crps, nfe) in enumerate(zip(timings, afm_crps, fm_crps, nfe_values)):
        if not np.isnan(t) and not np.isnan(a_crps):
            ax2.annotate(f'{nfe}', (t, a_crps), xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, color='#2E86AB')
        if not np.isnan(t) and not np.isnan(f_crps):
            ax2.annotate(f'{nfe}', (t, f_crps), xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, color='#A23B72')
    
    ax2.set_xlabel('Total Time (seconds)')
    ax2.set_ylabel('CRPS (Prediction)')
    ax2.set_title('Efficiency Frontier')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved NFE efficiency plot to {save_path}")
    
    plt.show()


def plot_multi_system_heatmap(
    system_results: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Multi-System Performance Comparison"
) -> None:
    """Plot heatmap of performance across multiple systems.
    
    Args:
        system_results: Results from multi-system experiment
        save_path: Path to save figure
        title: Plot title
    """
    setup_matplotlib_style()
    
    # Extract data for heatmap
    systems = []
    afm_scores = []
    fm_scores = []
    
    for system_key, results in system_results.items():
        if 'error' in results or 'AFM' not in results or 'FM' not in results:
            continue
            
        systems.append(system_key.replace('_', '\n'))  # Better formatting
        afm_scores.append(results['AFM'].get('crps_prediction', np.nan))
        fm_scores.append(results['FM'].get('crps_prediction', np.nan))
    
    if not systems:
        print("No valid results found for heatmap")
        return
    
    # Create data matrix
    data = np.array([afm_scores, fm_scores]).T
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Heatmap of raw scores
    sns.heatmap(data, annot=True, fmt='.3f', cmap='viridis_r', 
                xticklabels=['AFM', 'FM'], yticklabels=systems,
                cbar_kws={'label': 'CRPS (lower is better)'}, ax=ax1)
    ax1.set_title('CRPS Scores by System')
    ax1.set_xlabel('Method')
    
    # Bar plot comparison
    x = np.arange(len(systems))
    width = 0.35
    
    ax2.bar(x - width/2, afm_scores, width, label='AFM', color='#2E86AB', alpha=0.8)
    ax2.bar(x + width/2, fm_scores, width, label='FM', color='#A23B72', alpha=0.8)
    
    ax2.set_xlabel('Dynamical System')
    ax2.set_ylabel('CRPS (Prediction)')
    ax2.set_title('CRPS Comparison by System')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace('\n', ' ') for s in systems], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved multi-system heatmap to {save_path}")
    
    plt.show()


def plot_stochastic_deterministic_comparison(
    system_results: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Stochastic vs Deterministic Systems"
) -> None:
    """Plot comparison of stochastic vs deterministic performance.
    
    Args:
        system_results: Results from stochastic vs deterministic experiment
        save_path: Path to save figure  
        title: Plot title
    """
    setup_matplotlib_style()
    
    # Group results by system
    systems = {}
    for system_key, results in system_results.items():
        if 'error' in results:
            continue
            
        parts = system_key.split('_')
        system_name = parts[0]
        mode = parts[1]
        
        if system_name not in systems:
            systems[system_name] = {}
        systems[system_name][mode] = results
    
    if not systems:
        print("No valid results found for comparison")
        return
    
    # Prepare data for plotting
    system_names = list(systems.keys())
    afm_det_scores = []
    afm_sto_scores = []
    fm_det_scores = []
    fm_sto_scores = []
    
    for system_name in system_names:
        det_results = systems[system_name].get('deterministic', {})
        sto_results = systems[system_name].get('stochastic', {})
        
        afm_det_scores.append(det_results.get('AFM', {}).get('crps_prediction', np.nan))
        afm_sto_scores.append(sto_results.get('AFM', {}).get('crps_prediction', np.nan))
        fm_det_scores.append(det_results.get('FM', {}).get('crps_prediction', np.nan))
        fm_sto_scores.append(sto_results.get('FM', {}).get('crps_prediction', np.nan))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Grouped bar plot
    x = np.arange(len(system_names))
    width = 0.2
    
    ax1.bar(x - 1.5*width, afm_det_scores, width, label='AFM Deterministic', 
           color='#2E86AB', alpha=0.8)
    ax1.bar(x - 0.5*width, afm_sto_scores, width, label='AFM Stochastic', 
           color='#2E86AB', alpha=0.5)
    ax1.bar(x + 0.5*width, fm_det_scores, width, label='FM Deterministic', 
           color='#A23B72', alpha=0.8)
    ax1.bar(x + 1.5*width, fm_sto_scores, width, label='FM Stochastic', 
           color='#A23B72', alpha=0.5)
    
    ax1.set_xlabel('Dynamical System')
    ax1.set_ylabel('CRPS (Prediction)')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(system_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance degradation plot
    afm_degradation = []
    fm_degradation = []
    
    for i in range(len(system_names)):
        if not np.isnan(afm_det_scores[i]) and not np.isnan(afm_sto_scores[i]) and afm_det_scores[i] > 0:
            afm_degradation.append((afm_sto_scores[i] - afm_det_scores[i]) / afm_det_scores[i] * 100)
        else:
            afm_degradation.append(0)
            
        if not np.isnan(fm_det_scores[i]) and not np.isnan(fm_sto_scores[i]) and fm_det_scores[i] > 0:
            fm_degradation.append((fm_sto_scores[i] - fm_det_scores[i]) / fm_det_scores[i] * 100)
        else:
            fm_degradation.append(0)
    
    x = np.arange(len(system_names))
    width = 0.35
    
    ax2.bar(x - width/2, afm_degradation, width, label='AFM', color='#2E86AB', alpha=0.8)
    ax2.bar(x + width/2, fm_degradation, width, label='FM', color='#A23B72', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax2.set_xlabel('Dynamical System')
    ax2.set_ylabel('Performance Degradation (%)')
    ax2.set_title('Stochastic vs Deterministic Degradation')
    ax2.set_xticks(x)
    ax2.set_xticklabels(system_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved stochastic vs deterministic plot to {save_path}")
    
    plt.show()


def plot_architecture_ablation(
    sweep_results: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Architecture Ablation Study"
) -> None:
    """Plot architecture ablation results.
    
    Args:
        sweep_results: Results from architecture ablation
        save_path: Path to save figure
        title: Plot title
    """
    setup_matplotlib_style()
    
    # This is a placeholder - actual implementation would depend on the
    # specific structure of architecture ablation results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.text(0.5, 0.5, 'Architecture Ablation Plot\n(Implementation needed based on actual results structure)', 
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved architecture ablation plot to {save_path}")
    
    plt.show()


def create_experiment_plots(
    experiment_type: str,
    results: Dict[str, Any],
    plots_config: List[str],
    save_dir: Optional[str] = None
) -> None:
    """Create plots based on experiment type and configuration.
    
    Args:
        experiment_type: Type of experiment
        results: Experiment results
        plots_config: List of plot types to create
        save_dir: Directory to save plots
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for plot_type in plots_config:
        save_path = None
        if save_dir:
            save_path = str(save_dir / f"{plot_type}.pdf")
        
        if experiment_type == "history_length_sensitivity":
            if plot_type in ["history_vs_accuracy", "history_vs_crps"]:
                plot_history_sensitivity_results(results, save_path)
                
        elif experiment_type == "nfe_efficiency_study":
            if plot_type in ["accuracy_vs_nfe", "efficiency_frontier"]:
                plot_nfe_efficiency_results(results, save_path)
                
        elif experiment_type == "afm_vs_fm_all_systems":
            if plot_type == "systems_comparison_heatmap":
                plot_multi_system_heatmap(results, save_path)
                
        elif experiment_type == "stochastic_vs_deterministic":
            if plot_type == "stochastic_vs_deterministic_comparison":
                plot_stochastic_deterministic_comparison(results, save_path)
                
        elif experiment_type == "architecture_ablation":
            if plot_type == "architecture_performance_heatmap":
                plot_architecture_ablation(results, save_path)
        
        else:
            print(f"Unknown plot type '{plot_type}' for experiment '{experiment_type}'")
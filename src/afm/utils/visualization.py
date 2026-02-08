import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jax.numpy as jnp
from typing import List, Optional, Tuple
import matplotlib.colors as mcolors
import os

def plot_loss(train_losses: List[float],
              test_losses: List[float],
              title: str = "",
              save_path: Optional[str] = None):
    """Plot training and test losses."""
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.semilogy(train_losses, label='Train Loss')
    plt.semilogy(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    return
    #plt.close()

def plot_trajectories(ts: np.ndarray,
                     history: np.ndarray,
                     predictions: np.ndarray,
                     true_future: Optional[np.ndarray],
                     extrapolation_size: int = 0,
                     channel_names: Optional[List[str]] = None,
                     title: str = "Trajectory Predictions",
                     confidence_interval: float = 0.95,
                     save_path: Optional[str] = None):
    """Plot trajectory predictions with confidence intervals."""
    plt.close()
    fig, ax = plt.subplots(figsize=(9, 6))

    n_channels = history.shape[1]
    colors = list(mcolors.TABLEAU_COLORS)[:n_channels]

    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(n_channels)]

    # Calculate prediction statistics
    pred_mean = np.mean(predictions, axis=0)
    lower_percentile = (1 - confidence_interval) / 2 * 100
    upper_percentile = (1 + confidence_interval) / 2 * 100
    pred_lower = np.percentile(predictions, lower_percentile, axis=0)
    pred_upper = np.percentile(predictions, upper_percentile, axis=0)

    history_end_idx = len(history) - 1
    pred_end_idx = (history_end_idx + len(pred_mean)) - extrapolation_size
    extra_end_idx = history_end_idx + len(pred_mean)

    # Add background shading
    ax.axvspan(ts[0], ts[history_end_idx], color="lightgray", alpha=0.7)  # History (soft gray)
    ax.axvspan(ts[history_end_idx], ts[pred_end_idx], color="lightblue", alpha=0.7)  # Prediction (soft blue)
    ax.axvspan(ts[pred_end_idx], ts[extra_end_idx], color="lightyellow", alpha=0.7)  # Extrapolation (soft yellow)


    for i in range(n_channels):
        # Plot confidence intervals
        pred_ts = ts[history_end_idx:extra_end_idx+1]
        pred_line = np.concatenate([history[-1:, i], pred_mean[:, i]])
        pred_lower_extended = np.concatenate([history[-1:, i], pred_lower[:, i]])
        pred_upper_extended = np.concatenate([history[-1:, i], pred_upper[:, i]])

        ax.fill_between(pred_ts,
                       pred_lower_extended,
                       pred_upper_extended,
                       color=colors[i],
                       alpha=0.2)

        # Plot history
        ax.plot(ts[:len(history)], history[:, i],
                color=colors[i],
                #label=channel_names[i],
                linewidth=1.5)

        # Plot prediction mean
        ax.plot(pred_ts, pred_line,
                color=colors[i],
                linestyle='--',
                linewidth=1.5)

        # Plot ground truth if available
        if true_future is not None:
            true_ts = ts[history_end_idx:history_end_idx+len(true_future)+1]
            true_line = np.concatenate([history[-1:, i], true_future[:, i]])
            ax.plot(true_ts,
                    true_line,
                    color=colors[i],
                    linewidth=1.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')

                    # Add region labels
    y_range = ax.get_ylim()
    y_pos = y_range[1] - 0.05 * (y_range[1] - y_range[0])
    #ax.legend()

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight')
    return fig

def plot_phase_portrait(system,
                       trajectories: np.ndarray,
                       title: str = "",
                       save_path: Optional[str] = None):
    """Plot phase portrait for 2D or 3D systems."""
    if system.data_dim == 2:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(trajectories[..., 0], trajectories[..., 1], 'b-', alpha=0.5)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
    elif system.data_dim == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trajectories[..., 0],
                trajectories[..., 1],
                trajectories[..., 2],
                'b-', alpha=0.5)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')

    if save_path:
        plt.savefig(save_path)
    plt.close()

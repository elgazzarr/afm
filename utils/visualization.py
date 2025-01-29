import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

class Visualizer:
    """Visualization utilities for time series forecasting."""
    
    @staticmethod
    def plot_trajectories(
        ts: np.ndarray,
        history: np.ndarray,  # Shape: (T_hist, C)
        predictions: np.ndarray,  # Shape: (N, T_pred, C)
        true_future: Optional[np.ndarray] = None,  # Shape: (T_pred, C)
        channel_names: Optional[List[str]] = None,
        title: str = "Trajectory Predictions",
        confidence_interval: float = 0.95
    ) -> Figure:
        """
        Plot trajectory predictions with confidence intervals using a minimalist design.
        
        Args:
            ts: Time points
            history: Historical data with shape (T_hist, C) where C is number of channels
            predictions: MCMC predictions with shape (N, T_pred, C) where N is number of samples
            true_future: Optional ground truth future trajectory with shape (T_pred, C)
            channel_names: Optional list of channel names
            title: Plot title
            confidence_interval: Confidence interval for the prediction bounds (default: 0.95)
            
        Returns:
            matplotlib.figure.Figure: The figure object containing the plot
        """
        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        # Get number of channels
        n_channels = history.shape[1]
        
        colors = list(mcolors.TABLEAU_COLORS)[:n_channels]
        colors_ = list(mcolors.XKCD_COLORS)[:n_channels]
        if channel_names is None:
            channel_names = [f'Channel {i+1}' for i in range(n_channels)]
        
        # Calculate prediction statistics
        pred_mean = np.mean(predictions, axis=0)  # Shape: (T_pred, C)
        lower_percentile = (1 - confidence_interval) / 2 * 100
        upper_percentile = (1 + confidence_interval) / 2 * 100
        pred_lower = np.percentile(predictions, lower_percentile, axis=0)
        pred_upper = np.percentile(predictions, upper_percentile, axis=0)
        
        # Get indices for the different regions
        history_end_idx = len(history) - 1
        pred_end_idx = history_end_idx + len(pred_mean)
        
        # Add background shading for history and prediction regions
        ax.axvspan(ts[0], ts[history_end_idx], color='#f0f0f0', alpha=0.0, zorder=1)
        ax.axvspan(ts[history_end_idx], ts[pred_end_idx], color='#e0e0e0', alpha=0.5, zorder=1)
        

        
        for i in range(n_channels):
            # Plot history line
            ax.plot(ts[:len(history)], history[:, i],
                   color=colors[i],
                   label=channel_names[i],
                   linewidth=2,
                   zorder=3)
            
            # Plot prediction mean (starting from the last history point)
            pred_ts = ts[history_end_idx:pred_end_idx+1]
            pred_line = np.concatenate([history[-1:, i], pred_mean[:, i]])
            ax.plot(pred_ts, pred_line,
                   color=colors_[i],
                   label="forecast",
                   linewidth=2,
                   linestyle='--',  # Changed to dashed for predictions
                   zorder=3)
            
            # Plot confidence intervals (align with prediction mean)
            pred_lower_extended = np.concatenate([history[-1:, i], pred_lower[:, i]])
            pred_upper_extended = np.concatenate([history[-1:, i], pred_upper[:, i]])
            ax.fill_between(pred_ts,
                          pred_lower_extended,
                          pred_upper_extended,
                          color=colors[i],
                          alpha=0.2,
                          zorder=2,
                          label=f'{int(confidence_interval*100)}% CI')
            
            # Plot ground truth if available (starting from last history point)
            if true_future is not None:
                true_ts = ts[history_end_idx:history_end_idx+len(true_future)+1]
                true_line = np.concatenate([history[-1:, i], true_future[:, i]])
                ax.plot(true_ts,
                       true_line,
                       color=colors[i],
                       linestyle='-',  # Changed to solid for true future
                       linewidth=2,
                       zorder=3)
        
                # Add region labels
        y_range = ax.get_ylim()
        y_pos = y_range[1] - 0.05 * (y_range[1] - y_range[0])
        ax.text(ts[history_end_idx//2], y_pos, 'History', 
                horizontalalignment='center', verticalalignment='top', fontsize=10, color='#666666')
        ax.text(ts[history_end_idx + len(pred_mean)//2], y_pos, 'Prediction', 
                horizontalalignment='center', verticalalignment='top', fontsize=10, color='#666666')
        
        # Customize appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        
        # Add labels
        ax.set_xlabel('Time', color='#666666')
        ax.set_ylabel('Value', color='#666666')
        ax.set_title(title, pad=20)
        
        # Customize legend
        legend = ax.legend(bbox_to_anchor=(1.02, 1),
                         loc='upper left',
                         frameon=False)
        
        # Adjust layout to prevent legend cutoff
        plt.subplots_adjust(right=0.85, top=0.85)  # Adjusted top margin for region labels
        
        return fig
    

    @staticmethod
    def plot_reconstruction(
        ts: np.ndarray,
        original: np.ndarray,
        reconstructed: np.ndarray,
        title: str = "Reconstruction",
        channel_names: Optional[List[str]] = None
    ) -> Figure:
        """
        Plot the original and reconstructed time series data with uncertainty bounds.

        Args:
            ts: Time points
            original: Original time series data with shape (T, C)
            reconstructed: Reconstructed mean with shape (T, C)
            title: Plot title
            channel_names: Optional list of channel names

        Returns:
            matplotlib.figure.Figure: The figure object containing the plot
        """
        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        # Get number of channels
        n_channels = original.shape[1]
        
        # Set up colors and channel names
        colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        if channel_names is None:
            channel_names = [f'Channel {i+1}' for i in range(n_channels)]
            

        
        for i in range(n_channels):
            # Plot original data
            ax.plot(ts, original[:, i],
                   color=colors[i],
                   label=f'{channel_names[i]} (Original)',
                   linewidth=2,
                   zorder=3)
            
            # Plot reconstructed mean
            ax.plot(ts, reconstructed[:, i],
                   color=colors[i],
                   label=f'{channel_names[i]} (Reconstructed)',
                   linewidth=2,
                   linestyle='--',
                   zorder=3)
            
        
        # Customize appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        
        # Add labels
        ax.set_xlabel('Time', color='#666666')
        ax.set_ylabel('Value', color='#666666')
        ax.set_title(title, pad=20)
        
        # Customize legend
        legend = ax.legend(bbox_to_anchor=(1.02, 1),
                         loc='upper left',
                         frameon=False)
        
        # Adjust layout to prevent legend cutoff
        plt.subplots_adjust(right=0.85)
        
        return fig
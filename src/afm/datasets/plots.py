"""Sample-forecast visualization for real-world datasets.

Thin wrapper around ``afm.utils.visualization.plot_trajectories`` that selects one
example trajectory and one channel, de-normalizes it back to original units via the
dataset's scaler, and renders the probabilistic forecast fan (history, ensemble mean,
confidence band, ground truth) to a file.
"""
from __future__ import annotations
import numpy as np
from ..utils.visualization import plot_trajectories


def save_forecast_plot(dataset, preds, gt, config, save_path,
                       channel=0, channel_name=None, example=0, title=None):
    """Save a forecast-sample fan plot for one example/channel, in original units.

    Args:
        dataset: RealDataset (provides test_traj + meta["scaler"]).
        preds: array (S, B, H, dim) of forecast samples (normalized space).
        gt: array (B, H, dim) of ground-truth future (normalized space).
        config: Config (uses context_length).
        save_path: output image path.
        channel: which feature/sensor index to plot.
        channel_name: legend/axis label for the channel.
        example: which test trajectory (batch index) to plot.
        title: plot title.
    """
    ctx = config.context_length
    scaler = dataset.meta["scaler"]
    m = float(scaler.mean[channel]); s = float(scaler.std[channel]) + scaler.eps

    history = np.asarray(dataset.test_traj)[example, :ctx, channel] * s + m   # (ctx,)
    pr = np.asarray(preds)[:, example, :, channel] * s + m                    # (S, H)
    tf = np.asarray(gt)[example, :, channel] * s + m                          # (H,)

    ts = np.arange(ctx + pr.shape[1])
    return plot_trajectories(
        ts,
        history[:, None],          # (ctx, 1)
        pr[:, :, None],            # (S, H, 1)
        tf[:, None],               # (H, 1)
        extrapolation_size=0,
        channel_names=[channel_name or f"channel {channel}"],
        title=title or "Forecast samples",
        save_path=str(save_path),
    )

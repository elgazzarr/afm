"""Helper to build a valid Config for real-world AFM training runs.

The key invariant is that fm_prediction_horizon must be strictly less than
(length - context_length) so the FM dataloader window is constructible.
"""
from __future__ import annotations

from afm.config import Config


def build_afm_config(
    system_name: str,
    length: int,
    context_length: int,
    forecast_horizon: int,
    data_batch_size: int = 64,
    steps_per_epoch: int = 100,
    flow_nfe: int = 20,
    flow_n_samples: int = 50,
    **overrides,
) -> Config:
    """Build a Config for a real-world AFM training run.

    Sets afm_prediction_horizon=1 (always single-step for the AFM loaders) and
    keeps fm_prediction_horizon strictly below (length - context_length) so the
    FM dataloader window is constructible (the FM loaders are built
    unconditionally by create_infinite_dataloaders even though only the AFM
    loaders are used for training).

    AFM autoregressive evaluation uses n_steps_pred=forecast_horizon passed to
    evaluate_on_trajectories — that value is independent of fm_prediction_horizon.

    Args:
        system_name: A valid Config system_name (e.g. "nasa_battery").
        length: Number of time-steps per trajectory (system_num_timepoints).
        context_length: Context window length.  Must be < length.
        forecast_horizon: Evaluation horizon (passed as n_steps_pred to
            evaluate_on_trajectories; also stored as data_prediction_size).
            Must be >= 1.
        data_batch_size: Mini-batch size for the dataloaders.
        steps_per_epoch: Gradient steps per epoch.
        flow_nfe: Number of function evaluations for the ODE solver.
        flow_n_samples: Ensemble size for Monte-Carlo evaluation.
        **overrides: Any extra Config keyword arguments (passed through last,
            so they override the computed defaults).

    Returns:
        A fully validated Config instance.

    Raises:
        ValueError: If context_length >= length or forecast_horizon < 1.
    """
    if context_length >= length:
        raise ValueError(
            f"context_length ({context_length}) must be < length ({length})"
        )
    if forecast_horizon < 1:
        raise ValueError(f"forecast_horizon must be >= 1, got {forecast_horizon}")

    slack = length - context_length  # maximum constructible fm window
    # fm_prediction_horizon must satisfy: length - context_length - fm_ph >= 1
    # i.e. fm_ph <= slack - 1.  Floor at 1.
    fm_prediction_horizon = max(1, slack - 1)
    # Avoid the value 30: Config._compute_derived_fields treats fm_prediction_horizon==30
    # as the "unset default" and silently remaps it to data_prediction_size, which would
    # reintroduce a zero-width FM window. Nudge off the sentinel.
    if fm_prediction_horizon == 30:
        fm_prediction_horizon = 29

    kwargs = dict(
        system_name=system_name,
        system_num_timepoints=length,
        context_length=context_length,
        afm_prediction_horizon=1,
        fm_prediction_horizon=fm_prediction_horizon,
        data_prediction_size=forecast_horizon,
        model_covariates_dim=1,
        data_batch_size=data_batch_size,
        steps_per_epoch=steps_per_epoch,
        flow_nfe=flow_nfe,
        flow_n_samples=flow_n_samples,
    )
    kwargs.update(overrides)
    return Config(**kwargs)

"""Probabilistic-forecast diagnostics that target AFM's structural strengths:
calibration (prediction-interval coverage), horizon-wise error, and multivariate
cross-channel dependence. These compete on the axes a whole-window FM baseline
cannot, and they are the diagnostics the reviewers explicitly requested.

All functions take:
    preds : (S, B, H, dim)  forecast samples
    gt    : (B, H, dim)     ground-truth future
in normalized space (scale-invariant metrics, so de-normalization is unnecessary).
"""
from __future__ import annotations
import numpy as np
from ..evaluation.metrics import compute_crps, compute_nrmse


def coverage(preds, gt, level=0.9):
    """Empirical coverage of the central `level` prediction interval.

    Returns (overall_coverage_float, per_horizon_array_of_len_H).
    Well-calibrated => coverage ≈ level.
    """
    preds = np.asarray(preds); gt = np.asarray(gt)
    a = (1.0 - level) / 2.0
    lo = np.quantile(preds, a, axis=0)        # (B,H,dim)
    hi = np.quantile(preds, 1.0 - a, axis=0)  # (B,H,dim)
    inside = (gt >= lo) & (gt <= hi)          # (B,H,dim)
    return float(inside.mean()), inside.mean(axis=(0, 2))  # scalar, (H,)


def horizonwise(preds, gt):
    """Per-forecast-step CRPS and NRMSE. Returns (crps_H, nrmse_H), each length H."""
    preds = np.asarray(preds); gt = np.asarray(gt)
    H = gt.shape[1]
    crps = np.empty(H); nrmse = np.empty(H)
    for h in range(H):
        crps[h] = float(compute_crps(preds[:, :, h:h + 1, :], gt[:, h:h + 1, :]))
        nrmse[h] = float(compute_nrmse(preds[:, :, h:h + 1, :], gt[:, h:h + 1, :]))
    return crps, nrmse


def correlation_distance(preds, gt):
    """Frobenius distance between the cross-channel correlation matrix of the
    forecast samples and of the true future (pooled over batch/horizon/samples).
    Lower => the forecast better reproduces inter-channel dependence. Returns None
    for univariate data.
    """
    preds = np.asarray(preds); gt = np.asarray(gt)
    dim = gt.shape[-1]
    if dim < 2:
        return None
    true_mat = np.corrcoef(gt.reshape(-1, dim).T)                # (dim,dim)
    pred_mat = np.corrcoef(preds.reshape(-1, dim).T)             # (dim,dim)
    return float(np.linalg.norm(true_mat - pred_mat))


def summarize(preds, gt, levels=(0.5, 0.9)):
    """Convenience bundle: overall coverage at each level, horizon curves, corr distance."""
    cov = {}
    per_h = {}
    for lv in levels:
        cov[lv], per_h[lv] = coverage(preds, gt, lv)
    crps_h, nrmse_h = horizonwise(preds, gt)
    return {
        "coverage": cov,              # {level: float}
        "coverage_per_h": per_h,      # {level: (H,)}
        "crps_h": crps_h,             # (H,)
        "nrmse_h": nrmse_h,           # (H,)
        "corr_distance": correlation_distance(preds, gt),  # float or None
    }

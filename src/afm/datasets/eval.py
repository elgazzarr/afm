from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from ..evaluation.metrics import compute_crps, compute_nrmse

def evaluate_on_trajectories(model, dataset, config, key, n_steps_pred,
                             n_samples=50, nfe=20, channel=0) -> dict:
    ctx = config.context_length
    test = jnp.asarray(dataset.test_traj)                  # (B, T, dim)
    B, T, dim = test.shape
    history = test[:, :ctx, :]                              # (B, ctx, dim)
    cov = jnp.asarray(dataset.covariates)[:ctx]             # (ctx,)
    cov = jnp.broadcast_to(cov[None, :, None], (B, ctx, 1)) # (B, ctx, 1)

    gt = test[:, ctx:ctx + n_steps_pred, :]                 # (B, H, dim)
    preds = model.sample(key, history, cov, nfe=nfe,
                         n_steps_pred=n_steps_pred, n_samples=n_samples)  # (S,B,H,dim)

    crps = float(compute_crps(preds, gt))
    nrmse = float(compute_nrmse(preds, gt))

    # De-normalized error on the headline channel (capacity / sensor of interest).
    scaler = dataset.meta["scaler"]
    mean_c = float(scaler.mean[channel]); std_c = float(scaler.std[channel]) + scaler.eps
    pred_mean = np.asarray(preds.mean(axis=0))[..., channel] * std_c + mean_c   # (B,H)
    gt_c = np.asarray(gt)[..., channel] * std_c + mean_c
    rmse_orig = float(np.sqrt(np.mean((pred_mean - gt_c) ** 2)))
    mae_orig = float(np.mean(np.abs(pred_mean - gt_c)))
    return {"crps": crps, "nrmse": nrmse, "rmse_orig": rmse_orig, "mae_orig": mae_orig}

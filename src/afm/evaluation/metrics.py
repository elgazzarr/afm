import jax.numpy as jnp
from jax import jit

@jit
def compute_crps(predictions, groundtruth):
    """
    Sample-rank CRPS: mean over sorted samples x_(i) of (i/S - 1{x_(i) >= y})^2.

    This evaluates the CRPS integrand at the S samples and averages with equal weights,
    i.e. it integrates with respect to the predictive distribution rather than over y.
    It depends only on the rank of y among the samples (about (p^3 + (1-p)^3)/3 for
    PIT value p), ranges from 1/12 (y at the median) to 1/3 (y outside the samples),
    and does not penalise over-dispersed ensembles. It is the score reported for the
    synthetic systems and degradation benchmarks in the paper (Section 4.1); it is not
    the scale-dependent CRPS computed by GluonTS for the Table 4 datasets.

    Args:
        predictions: Array of shape (S,B,T,C)
        groundtruth: Array of shape (B,T,C)
    """
    # Sort predictions along ensemble dimension
    sorted_preds = jnp.sort(predictions, axis=0)

    # Compute empirical CDF
    S = predictions.shape[0]
    cdf = jnp.arange(1, S + 1) / S

    # Expand dimensions for broadcasting
    gt_expanded = jnp.expand_dims(groundtruth, axis=0)
    cdf_expanded = jnp.expand_dims(cdf, axis=(1,2,3))

    # Compute Heaviside function
    heaviside = (sorted_preds >= gt_expanded).astype(float)

    # Compute squared difference
    squared_diff = (heaviside - cdf_expanded) ** 2

    # Average over all dimensions
    return jnp.mean(squared_diff)

@jit
def compute_nrmse(predictions, groundtruth):
    """
    JAX implementation of NRMSE computation.

    Args:
        predictions: Array of shape (S,B,T,C)
        groundtruth: Array of shape (B,T,C)
    """
    # Compute ensemble mean
    pred_mean = jnp.mean(predictions, axis=0)

    # Compute MSE
    mse = jnp.mean((pred_mean - groundtruth) ** 2)

    # Compute normalization factor
    norm_factor = jnp.var(groundtruth)

    # Add small epsilon to avoid division by zero
    return jnp.sqrt(mse / (norm_factor + 1e-8))

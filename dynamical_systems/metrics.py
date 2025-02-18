import jax.numpy as jnp
from jax import jit
from sample import sample_flow
from models import NoARFlow

@jit
def compute_crps(predictions, groundtruth):
    """
    JAX implementation of CRPS computation.
    
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


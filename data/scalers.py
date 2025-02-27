import jax.numpy as jnp
from typing import Dict, Tuple, Optional
from gluonts.transform import  SimpleTransformation
import numpy as np
from functools import partial
from jax import jit
import jax

from typing import Tuple
import jax.numpy as jnp

class ContextScaler:
    def fit_transform(self, context: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # context is (time, features)
        scale = jnp.mean(context, axis=0, keepdims=True)  # (1, features) 
        scale = jnp.where(scale == 0, 1.0, scale)
        return context / scale, scale
    
    def transform(self, x: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return x / scale
    
    def transform_features(self, x: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        # x is (features,)
        return x / scale[0, 0]
    
    def inverse_transform(self, x: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return x * scale
    
    
class NoContextScaler:
    def __init__(self, minimum_scale: float = 1e-10):
        """Initialize with same interface as ContextScaler but doesn't use parameters."""
        pass

    def fit_transform(self, context: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Pass through the context unchanged and return unit scale.
        Args:
            context: Input array of shape (time, features)
        Returns:
            Tuple of (context, ones_scale)
        """
        scale = jnp.ones((1, context.shape[1]))  # (1, features)
        return context, scale

    def transform(self, x: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        """Pass through data unchanged."""
        return x

    def transform_features(self, x: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        """Pass through features unchanged."""
        return x

    def inverse_transform(self, x: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        """Pass through data unchanged."""
        return x
    

class NothingScaler(SimpleTransformation):
    def __init__(self, *args, **kwargs):
        pass
        
    def fit(self, data_iterator):
        pass
            
    def transform(self, data):
        return data
        
    def transform_instance(self, data: jnp.ndarray) -> jnp.ndarray:
        return data
        
    def inverse_transform(self, data: jnp.ndarray) -> jnp.ndarray:
        return data
    
class StandardScaler(SimpleTransformation):
    def __init__(
        self,
        dim=0,  # dimension 0 for channels in [channels, time]
        keepdim=True,
        minimum_scale=1e-10,
        field_name="target",
    ):
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale
        self.field_name = field_name
        self.mean = None
        self.std = None

    def fit(self, data_iterator):
        # Collect all targets to compute statistics in one pass
        all_targets = []
        for data in data_iterator:
            target = data[self.field_name]  # [channels, time]
            all_targets.append(target)
        
        # Concatenate along the time dimension (axis=1)
        concatenated = np.concatenate(all_targets, axis=1)  # [channels, total_time]
        
        # Compute statistics along time dimension (axis=1)
        self.mean = np.mean(concatenated, axis=1, keepdims=self.keepdim)  # [channels, 1]
        self.std = np.std(concatenated, axis=1, keepdims=self.keepdim)    # [channels, 1]
        self.std = np.maximum(self.std, self.minimum_scale)

    @partial(jax.jit, static_argnums=(0,))
    def _transform_impl(self, data: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> Dict:
        """JIT-compiled implementation of the transform operation"""
        # Ensure broadcasting for [channels, time]
        scaled_data = (data - mean) / std  # mean and std are [channels, 1]
        return {
            self.field_name: scaled_data,
            'scale': std,
            'loc': mean
        }

    def transform(self, data: Dict) -> Dict:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform")
        
        target = data[self.field_name]  # [channels, time]
        transformed = self._transform_impl(target, self.mean, self.std)
        data.update(transformed)
        return data

    @partial(jax.jit, static_argnums=(0,))
    def transform_instance(self, data: jnp.ndarray) -> jnp.ndarray:
        """Transform a single instance of shape [time, channels]"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform")
        
        data = jnp.transpose(data, (1, 0))
        scaled_data = (data - self.mean) / self.std
        scaled_data = jnp.transpose(scaled_data, (1, 0))
        return scaled_data

    @partial(jax.jit, static_argnums=(0,))
    def inverse_transform(self, scaled_data: jnp.ndarray) -> jnp.ndarray:
        """Inverse transform data of shape [time, channels]"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before inverse transform")

        scaled_data = jnp.transpose(scaled_data, (1, 0))  # [time, channels]
        data = (scaled_data * self.std) + self.mean
        data = jnp.transpose(data, (1, 0))
        return data
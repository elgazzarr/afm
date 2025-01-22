import jax.numpy as jnp
from typing import Tuple
from gluonts.transform import  SimpleTransformation
import numpy as np


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
        # First pass: compute means per channel
        sum_x = None
        count = 0
        
        for data in data_iterator:
            target = data[self.field_name]  # Shape: [channels, time]
            # Sum along time dimension (1) to get per-channel statistics
            if sum_x is None:
                sum_x = np.sum(target, axis=1, keepdims=self.keepdim)  # [channels, 1]
            else:
                sum_x += np.sum(target, axis=1, keepdims=self.keepdim)
            count += target.shape[1]  # Add number of timesteps
        
        self.mean = sum_x / count  # [channels, 1]
        
        # Second pass: compute std per channel
        sum_squared_dev = None
        for data in data_iterator:
            target = data[self.field_name]
            squared_dev = (target - self.mean) ** 2
            if sum_squared_dev is None:
                sum_squared_dev = np.sum(squared_dev, axis=1, keepdims=self.keepdim)
            else:
                sum_squared_dev += np.sum(squared_dev, axis=1, keepdims=self.keepdim)
        
        self.std = np.sqrt(sum_squared_dev / count)
        self.std = np.maximum(self.std, self.minimum_scale)

    def transform(self, data):
        target = data[self.field_name]  # [channels, time]
        scaled_data = (target - self.mean) / self.std
        data[self.field_name] = scaled_data
        data['scale'] = self.std  # [channels, 1]
        data['loc'] = self.mean   # [channels, 1]
        return data
    
    def transform_instance(self, data: jnp.ndarray) -> jnp.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform")            
        data = jnp.transpose(data)
        scaled_data = (data - self.mean) / self.std
        return jnp.transpose(scaled_data)
    
    def inverse_transform(self, data: jnp.ndarray) -> jnp.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before inverse transform")            
        data = jnp.transpose(data)
        original_scale_data = (data * self.std) + self.mean
        return jnp.transpose(original_scale_data)

"""GluonTS benchmark pipeline (Table 4 of the paper). Faithful port of git revision f4131fe^ (models/layers.py); only imports/wandb-optionality were adapted. See REPRODUCIBILITY.md."""

from .embeddings import SinusodialPosEmb
import flax.nnx as nnx
import jax.numpy as jnp
import math
from typing import Tuple


class TimeConditionedDense(nnx.Module):
    def __init__(self, dim_x: int, dim_h: int, dim_t:int, rngs: nnx.Rngs):
        self.time_emb = SinusodialPosEmb(dim_t) 
        self.linear = nnx.Linear(dim_x + dim_h + dim_t , dim_x, rngs=rngs)
    
    def __call__(self, x, h, t):
        t = self.time_emb(t)
        x = jnp.concatenate([x, h, t], axis=-1) 
        return self.linear(x)
    

class CondUpsampler(nnx.Module):
    """Upsamples conditioning (hidden state and covariates) to match data dimension."""
    
    def __init__(self, x_dim: int, h_dim: int, c_dim: int, rngs: nnx.Rngs):
        self.hidden_proj1 = nnx.Linear(h_dim, x_dim // 2, rngs=rngs)
        self.hidden_proj2 = nnx.Linear(x_dim // 2, x_dim, rngs=rngs)

        self.covariate_proj1 = nnx.Linear(c_dim, x_dim // 2, rngs=rngs)
        self.covariate_proj2 = nnx.Linear(x_dim // 2, x_dim, rngs=rngs)
    
    def __call__(self, h: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        h = self.hidden_proj1(h)
        h = nnx.leaky_relu(h, 0.4)
        h = self.hidden_proj2(h)
        h = nnx.leaky_relu(h, 0.4)
        
        c = self.covariate_proj1(c)
        c = nnx.leaky_relu(c, 0.4)
        c = self.covariate_proj2(c)
        c = nnx.leaky_relu(c, 0.4)

        return h + c

class ResidualBlock(nnx.Module):
    """Single residual block with dilated convolution and conditional inputs.
    
    The block performs temporal and conditional modulation of the input features
    through a gated activation mechanism, followed by residual connection and skip connection
    for hierarchical feature extraction.
    
    Args:
        hidden_size (int): Dimension of the hidden/conditioning features
        residual_channels (int): Number of channels in the residual connections
        dilation (int): Dilation factor for the convolution
        rngs (nnx.Rngs): Random number generators for initialization
        
    Returns:
        tuple: A tuple containing:
            - jnp.ndarray: Residual output of shape [x_dim, residual_channels]
            - jnp.ndarray: Skip connection output of shape [x_dim, residual_channels]
    """
    
    def __init__(self, hidden_size:int, residual_channels: int, dilation: int, rngs: nnx.Rngs):

        self.time_proj = nnx.Linear(hidden_size, residual_channels, rngs=rngs)
        self.cond_proj = nnx.Conv(1, 2*residual_channels, 1, padding='CIRCULAR', rngs=rngs)

        
        # Dilated Conv1d
        self.dilated_conv = nnx.Conv(
            in_features=residual_channels,
            out_features=2 * residual_channels,
            kernel_size=3,
            kernel_dilation=dilation,
            padding='CIRCULAR',  # Maintain spatial size x_dim after convolution
            rngs=rngs
        )
        
        # Conv1x1 output
        self.out_proj = nnx.Conv(
            in_features=residual_channels,
            out_features=2 * residual_channels,  # residual and skip
            kernel_size=1,
            rngs=rngs
        )
        
    def __call__(self, x: jnp.ndarray, 
                 hc: jnp.ndarray,
                 t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass of the residual block. 

        Args:
            x (jnp.ndarray): Input features of shape (x_dim, residual_channels)
            hc (jnp.ndarray): Upsampled conditional features of shape (x_dim, 1)
            t (jnp.ndarray): Time embedding of shape (residual_hidden,)

        Returns:
            tuple: A tuple containing:
                - jnp.ndarray: Residual output of shape (x_dim, residual_channels)
                - jnp.ndarray: Skip connection output of shape (x_dim, residual_channels)
        """
        t_up = self.time_proj(t) # [residual_hidden] -> [residual_channels]
        y = x + t_up[None, :]   # [x_dim, residual_channels] + [1, residual_channels] -> [x_dim, residual_channels]
        y = self.dilated_conv(y) # [x_dim, residual_channels] -> [x_dim, 2*residual_channels]
        c_up = self.cond_proj(hc) # [x_dim, 1] -> [x_dim, 2*residual_channels]
        y = y + c_up # [x_dim, 2*residual_channels] + [x_dim, 2*residual_channels] -> [x_dim, 2*residual_channels]
        gate_, filter_ = jnp.split(y, 2, axis=-1)  # Split along channel dimension
        y = nnx.sigmoid(gate_) * nnx.tanh(filter_)  # Gated activation
        y = nnx.leaky_relu(self.out_proj(y), 0.4)  #  [x_dim, residual_channels] -> [x_dim, 2*residual_channels]
        residual, skip = jnp.split(y, 2, axis=-1) # Split along channel dimension 
        return (x + residual) / math.sqrt(2.0), skip

class FlowWave(nnx.Module):
    """Conditional vector filed with WaveNet-style architecture.
    
    This network predicts the flow velocity given a noisy input signal and various conditioning
    information. It uses dilated convolutions organized in residual blocks with
    skip connections, similar to WaveNet. The network can condition on:
        - Hidden states (e.g., context from an encoder)
        - Covariates (additional context)
        - Current flow timestep

    Args:
        x_dim (int): Dimension of the input signal
        h_dim (int): Dimension of the hidden state conditioning
        c_dim (int): Dimension of the covariate conditioning
        t_dim (int): Dimension of the time embedding
        residual_layers (int, optional): Number of residual layers. Defaults to 8
        residual_channels (int, optional): Number of channels in residual path. Defaults to 8
        residual_hidden (int, optional): Hidden dimension in residual blocks. Defaults to 64
        dilation_cycle_length (int, optional): Pattern length for dilation cycle. Defaults to 2
        rngs (nnx.Rngs, optional): Random number generators for initialization
        
    Input Shapes:
        x (jnp.ndarray): Input signal of shape [x_dim]
        h (jnp.ndarray): Hidden state of shape [h_dim]
        c (jnp.ndarray): Covariates of shape [c_dim]
        t (int): Current denoising timestep
        
    Returns:
        jnp.ndarray: Predicted noise of shape [x_dim]
    """
    def __init__(self,
                x_dim: int,
                h_dim: int,
                c_dim: int,
                t_dim: int,
                residual_layers: int = 8,
                residual_channels: int = 8,
                residual_hidden: int = 64,
                dilation_cycle_length: int = 2,
                rngs: nnx.Rngs = None):
        

        # Initial Conv1x1 for x
        self.conv_in = nnx.Conv(
                in_features=1, 
                out_features=residual_channels,
                kernel_size=1,
                padding='CIRCULAR',
                rngs=rngs
            )
            
        # Time embedding
        self.time_embedding = SinusodialPosEmb(t_dim)
        self.time_projection = nnx.Sequential(nnx.Linear(t_dim, residual_hidden, rngs=rngs),
                                              nnx.silu,
                                              nnx.Linear(residual_hidden, residual_hidden, rngs=rngs),
                                              nnx.silu)

        # Upsample conditions (hidden state and covariates) to match data dimension
        self.cond_upsampler = CondUpsampler(x_dim, h_dim, c_dim, rngs)
        

        # Residual blocks
        self.residual_layers = [
            ResidualBlock(
                hidden_size=residual_hidden,
                residual_channels=residual_channels,
                dilation=2 ** (i % dilation_cycle_length),
                rngs=rngs
            )
            for i in range(residual_layers)
        ]
        
        # Final Conv1d and Conv1x1
        self.conv_out1 = nnx.Conv(
            in_features=residual_channels,
            out_features=residual_channels,
            kernel_size=3,
            padding='CIRCULAR',
            rngs=rngs
        )
        
        self.conv_out2 = nnx.Conv(
            in_features=residual_channels,
            out_features=1,  # Back to 1 channel
            kernel_size=1,
            rngs=rngs
        )
        
    def __call__(self,
                 x: jnp.ndarray,     
                 h: jnp.ndarray,     
                 c: jnp.ndarray,     
                 t: int              
                 ) -> jnp.ndarray:   
        
        """Forward pass of the FlowWave model. 

        Args:
            x (jnp.ndarray): Input signal of shape (x_dim,)
            h (jnp.ndarray): Hidden state of shape (h_dim,)
            c (jnp.ndarray): Covariates of shape (c_dim,)
            t (int): Current flow timestep

        Returns:
            jnp.ndarray: Predicted velocity of shape (x_dim,)
        """

        
        # Add channel dimension
        x = x[:, None]  # [x_dim] -> [x_dim, 1]
        
        # Initial Conv1x1
        x = self.conv_in(x)  # [x_dim, 1] -> [x_dim, residual_channels]
        x = nnx.leaky_relu(x, 0.4)
        
        # Time embedding
        t_emb = self.time_embedding(t) # [1] -> [t_dim]
        t_upsampled = self.time_projection(t_emb)  # [t_dim] -> [residual_hidden]

        cond_upsampled = self.cond_upsampler(h, c)[:, None] # (h_dim, c_dim) -> [[x_dim, 1]]
        
        # Residual blocks
        skip_connections = []
        for layer in self.residual_layers:
            x, skip = layer(x, cond_upsampled, t_upsampled)
            skip_connections.append(skip)
        
        # Combine skips and final processing
        x = jnp.sum(jnp.stack(skip_connections), axis=0) / math.sqrt(len(self.residual_layers))
        x = nnx.leaky_relu(x, 0.4)
        x = self.conv_out1(x)  # Conv1d
        x = nnx.leaky_relu(x, 0.4)
        x = self.conv_out2(x)  # Conv1x1
        
        return x[:, 0]  # Remove channel dimension and return [x_dim]

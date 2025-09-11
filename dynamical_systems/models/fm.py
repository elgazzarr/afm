"""Flow Matching (FM) model implementation."""

from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax.nnx as nnx
import diffrax as dfx
from .base_flow import BaseFlow, TimeRNNBlock


class FlowMatching(BaseFlow):
    """Non-autoregressive Flow Matching model.
    
    This model predicts all future states simultaneously using parallel processing
    rather than sequentially like the autoregressive version.
    """
    
    def __init__(
        self, 
        dim_x: int, 
        dim_c: int, 
        dim_h: int, 
        dim_t: int, 
        dim_hidden: int, 
        hidden_blocks: int, 
        n_rnn_layers: int, 
        rngs: nnx.Rngs
    ):
        """Initialize FM model.
        
        Args:
            dim_x: Dimension of input/output data
            dim_c: Dimension of covariates (time)
            dim_h: Hidden state dimension
            dim_t: Time embedding dimension
            dim_hidden: Internal hidden dimension
            hidden_blocks: Number of hidden blocks
            n_rnn_layers: Number of RNN layers
            rngs: Random number generator state
        """
        super().__init__()
        self.n_rnn_layers = n_rnn_layers
        self.rnn_hidden = dim_h
        
        # RNN cells for processing history
        self.cells = [
            nnx.nn.recurrent.OptimizedLSTMCell(
                in_features=dim_x + dim_c if i == 0 else dim_h,
                hidden_features=dim_h,
                rngs=rngs
            ) for i in range(n_rnn_layers)
        ]
        
        self.dim_h = dim_h
        
        # Flow matching network using 1D convolutions for parallel processing
        self.linear_in = nnx.Conv(dim_x + dim_h, dim_hidden, kernel_size=1, rngs=rngs)
        self.hidden_blocks = [
            TimeRNNBlock(dim_hidden, dim_hidden//2, dim_h, dim_t, rngs=rngs)
            for _ in range(hidden_blocks)
        ]
        self.linear_out = nnx.Conv(dim_hidden, dim_x, kernel_size=1, rngs=rngs)
    
    def __call__(
        self, 
        z: jnp.ndarray, 
        x_history: jnp.ndarray, 
        c: jnp.ndarray, 
        t: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass of FM model.
        
        Args:
            z: Noise/interpolation input (batch_size, seq_len, dim)
            x_history: Historical sequence data
            c: Covariate information
            t: Time parameter for flow matching
            
        Returns:
            Flow field prediction
        """
        # Process history to get context
        h = self.process_history(x_history, c)
        
        # Expand history context to match sequence length
        h = jnp.repeat(h[None, :], z.shape[0], axis=0)
        
        # Apply flow matching network
        z = nnx.gelu(self.linear_in(jnp.concatenate([z, h], axis=-1)))
        
        for block in self.hidden_blocks:
            z = block(z, h, t)
            
        x_next = self.linear_out(z)
        return x_next
    
    def sample(
        self,
        key: jax.random.PRNGKey,
        history: jnp.ndarray,
        covariates: jnp.ndarray, 
        nfe: int,
        n_steps_pred: int,
        n_samples: int = 50
    ) -> jnp.ndarray:
        """Sample predictions from the FM model.
        
        Args:
            key: Random key for sampling
            history: Historical sequence data
            covariates: Covariate information
            nfe: Number of function evaluations (ODE integration steps)  
            n_steps_pred: Number of prediction steps
            n_samples: Number of samples to generate
            
        Returns:
            Predictions array of shape (n_samples, batch_size, 1, n_steps_pred, dim)
        """
        def predict_single(
            key: jax.random.PRNGKey,
            history: jnp.ndarray,
            covariates: jnp.ndarray,
            nfe: int,
            n_steps_pred: int
        ) -> jnp.ndarray:
            """Generate a single prediction sample."""
            def vf(t, x, args):
                history, covariates = args
                return self(x, history, covariates, t)
                
            term = dfx.ODETerm(vf)
            solver = dfx.Euler()
            dt = 1 / nfe
            
            # Generate Brownian motion as initial condition
            from ..utils.priors import generate_brownian_timeseries
            x0 = generate_brownian_timeseries(key, history.shape[-1], n_steps_pred).T
            
            predictions = dfx.diffeqsolve(
                term, solver, 0., 1., dt, x0,
                args=(history, covariates),
                saveat=dfx.SaveAt(t1=True)
            ).ys
            
            return predictions
        
        # Split keys for sampling
        key_samples = jrandom.split(key, n_samples)
        
        # Vectorize over samples and batch  
        vmapped_predict = jax.vmap(predict_single, in_axes=(None, 0, 0, None, None))
        vmapped_predict = jax.vmap(vmapped_predict, in_axes=(0, None, None, None, None))
        
        pred_samples = vmapped_predict(
            key_samples, history, covariates, nfe, n_steps_pred
        )
        
        return pred_samples



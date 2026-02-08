"""Autoregressive Flow Matching (AFM) model implementation."""

from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax.nnx as nnx
import diffrax as dfx
from .base import BaseFlow, TimeConditionedDense


class AutoregressiveFlowMatching(BaseFlow):
    """Autoregressive Flow Matching model.

    This model predicts future states one step at a time, using the previously
    predicted states as context for the next prediction.
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
        rnn_hidden: int,
        rngs: nnx.Rngs
    ):
        """Initialize AFM model.

        Args:
            dim_x: Dimension of input/output data
            dim_c: Dimension of covariates (time)
            dim_h: Hidden state dimension
            dim_t: Time embedding dimension
            dim_hidden: Internal hidden dimension
            hidden_blocks: Number of hidden blocks
            n_rnn_layers: Number of RNN layers
            rnn_hidden: RNN hidden size
            rngs: Random number generator state
        """
        super().__init__()
        self.n_rnn_layers = n_rnn_layers
        self.rnn_hidden = rnn_hidden

        # RNN cells for processing history
        self.cells = [
            nnx.nn.recurrent.OptimizedLSTMCell(
                in_features=dim_x + dim_c if i == 0 else rnn_hidden,
                hidden_features=rnn_hidden,
                rngs=rngs
            ) for i in range(n_rnn_layers)
        ]

        self.dim_h = dim_h

        # Flow matching network
        self.linear_in = nnx.Linear(dim_x + dim_h, dim_hidden, rngs=rngs)
        self.blocks = [
            TimeConditionedDense(dim_hidden, dim_h, dim_t, rngs)
            for _ in range(hidden_blocks)
        ]
        self.linear_out = nnx.Linear(dim_hidden + dim_h, dim_x, rngs=rngs)

    def __call__(
        self,
        z: jnp.ndarray,
        x_history: jnp.ndarray,
        c: jnp.ndarray,
        t: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass of AFM model.

        Args:
            z: Noise/interpolation input
            x_history: Historical sequence data
            c: Covariate information
            t: Time parameter for flow matching

        Returns:
            Flow field prediction
        """
        # Process history to get context
        h_last = self.process_history(x_history, c)

        # Apply flow matching network
        z = nnx.gelu(self.linear_in(jnp.concatenate([z, h_last], axis=-1)))

        for block in self.blocks:
            z = nnx.gelu(block(z, h_last, t)) + z

        x_next = self.linear_out(jnp.concatenate([z, h_last], axis=-1))
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
        """Sample predictions from the AFM model.

        Args:
            key: Random key for sampling
            history: Historical sequence data
            covariates: Covariate information
            nfe: Number of function evaluations (ODE integration steps)
            n_steps_pred: Number of prediction steps
            n_samples: Number of samples to generate

        Returns:
            Predictions array of shape (n_samples, batch_size, n_steps_pred, dim)
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
                return self(x, history, covariates, jnp.array([t]))

            term = dfx.ODETerm(vf)
            solver = dfx.Euler()
            dt = 1 / nfe

            predictions = []
            current_history = history
            context_length = history.shape[0]

            for _ in range(n_steps_pred):
                context = current_history[-context_length:]
                x0 = jrandom.normal(key, shape=context.shape[-1:])

                x_next = dfx.diffeqsolve(
                    term, solver, 0., 1., dt, x0,
                    args=(context, covariates),
                    saveat=dfx.SaveAt(t1=True)
                ).ys

                # Squeeze out the time dimension - we only want the final state
                x_next = jnp.squeeze(x_next, axis=-2)  # Remove time dimension

                predictions.append(x_next)
                current_history = jnp.concatenate([current_history, x_next[None]], axis=0)

                # Update key for next step
                key, _ = jrandom.split(key)

            return jnp.vstack(predictions)

        # Split keys for sampling
        key_samples = jrandom.split(key, n_samples)

        # Vectorize over samples and batch
        vmapped_predict = jax.vmap(predict_single, in_axes=(None, 0, 0, None, None))
        vmapped_predict = jax.vmap(vmapped_predict, in_axes=(0, None, None, None, None))

        pred_samples = vmapped_predict(
            key_samples, history, covariates, nfe, n_steps_pred
        )

        return pred_samples

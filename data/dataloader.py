import jax
import jax.numpy as jnp
import jax_dataloader as jdl
from typing import Tuple, Iterator
import jax.random as jrandom
from typing import Any
from dataclasses import dataclass, field

class TimeSeriesDataLoader:
    """Data loader for time series data."""
    
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        """
        Initialize the data loader.
        
        Args:
            dataset: Dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle the data
        """
        self.history, self.covariates, self.prediction = dataset.generate_data()
        self.dataloader = jdl.DataLoader(
            jdl.ArrayDataset(self.history, self.covariates, self.prediction),
            'jax',
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        return iter(self.dataloader)
    
    def sample(self, key) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get one random instance of the dataset with the given key.

        Args:
            key: Random key for sampling

        Returns:
            Tuple of history, covariates, and prediction
        """
        idx = jrandom.randint(key, (), 0, self.history.shape[0])
        return self.history[idx], self.covariates[idx], self.prediction[idx]
    


@dataclass
class GluonTSDataLoaderAdapter:
    """Adapter class to convert GluonTS TrainDataLoader into the expected format."""
    data_loader: Any
    history_field: str = "past_target"
    covariate_field: list[str] = field(default_factory=lambda: ["past_time_feat", "future_time_feat"])
    prediction_field: str = "future_target"

    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        Converts GluonTS dictionary batches into the expected (history, covariates, prediction) format.
        """
        for batch in self.data_loader:
            history = jnp.array(batch[self.history_field])
            covariates = jnp.concatenate([jnp.array(batch[field]) for field in self.covariate_field], axis=1)
            prediction = jnp.array(batch[self.prediction_field])
            yield history, covariates, prediction
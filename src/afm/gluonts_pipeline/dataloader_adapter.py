"""GluonTS benchmark pipeline (Table 4 of the paper). Faithful port of git revision f4131fe^ (only the GluonTSDataLoaderAdapter class from data/dataloader.py); only imports/wandb-optionality were adapted. See REPRODUCIBILITY.md."""
import jax.numpy as jnp
from typing import Tuple, Iterator, Any
from dataclasses import dataclass, field

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

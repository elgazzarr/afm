"""GluonTS benchmark pipeline used for Table 4 of the paper
("Probabilistic Forecasting via Autoregressive Flow Matching").

Faithful port of the pipeline as it existed at git revision f4131fe^
(the tree before the public repo restructure removed it). Only imports
and wandb-optionality were adapted; see the per-module docstrings for the
exact deviations and REPRODUCIBILITY.md (repository root) for scope and
limitations. Requires the optional ``gluonts`` dependency group:

    pip install -e ".[gluonts]"

Entry point::

    python -m afm.gluonts_pipeline.run --dataset_name exchange_rate_nips
"""

from .flow import Flow
from .trainer import FlowTrainer
from .predictor import JaxPredictor
from .loaders import get_dataset_loader

__all__ = [
    "Flow",
    "FlowTrainer",
    "JaxPredictor",
    "get_dataset_loader",
]

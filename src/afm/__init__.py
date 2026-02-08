"""AFM - Autoregressive Flow Matching for dynamical systems forecasting."""

__version__ = "0.1.0"

from .models import AutoregressiveFlowMatching, FlowMatching
from .config import Config
from .systems import DynamicalSystem, create_system

__all__ = [
    "AutoregressiveFlowMatching",
    "FlowMatching",
    "Config",
    "DynamicalSystem",
    "create_system",
]

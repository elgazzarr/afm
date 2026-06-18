from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

REAL_DATASET_NAMES = ("nasa_battery", "cmapss_fd001", "cmapss_fd002", "cmapss_fd003", "cmapss_fd004")

@dataclass
class RealDataset:
    train_traj: np.ndarray   # (n_train, T, dim)
    test_traj: np.ndarray    # (n_test,  T, dim)
    covariates: np.ndarray   # (T,)  normalized time/cycle index
    data_dim: int
    meta: Dict[str, Any]

def load_dataset(name: str, config, **kw) -> RealDataset:
    if name == "nasa_battery":
        from .battery import load_battery
        return load_battery(config, **kw)
    if name.startswith("cmapss_"):
        from .cmapss import load_cmapss
        return load_cmapss(config, subset=name.split("_", 1)[1].upper(), **kw)
    raise ValueError(f"Unknown real dataset '{name}'. Known: {REAL_DATASET_NAMES}")

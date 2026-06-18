from __future__ import annotations
import numpy as np

class Scaler:
    """Per-feature z-score scaler. Stats are computed over all axes except the last (feature) axis."""
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray) -> "Scaler":
        x = np.asarray(x, dtype=np.float64)
        axes = tuple(range(x.ndim - 1))
        self.mean = x.mean(axis=axes)
        self.std = x.std(axis=axes)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * (self.std + self.eps) + self.mean

# src/afm/datasets/battery.py
from __future__ import annotations
import numpy as np
from scipy.io import loadmat
from . import RealDataset
from .scaling import Scaler
from .download import ensure_battery_files

FEATURE_NAMES = ["capacity", "mean_voltage", "discharge_time", "mean_temp"]

def _as_float(v) -> float:
    return float(np.asarray(v).ravel()[0])

def discharge_features(cycles: list[dict]) -> np.ndarray:
    """Per-discharge-cycle features: [capacity, mean_voltage, discharge_time, mean_temp]."""
    rows = []
    for cyc in cycles:
        if str(cyc.get("type", "")).strip() != "discharge":
            continue
        d = cyc["data"]
        rows.append([
            _as_float(d["Capacity"]),
            float(np.mean(np.asarray(d["Voltage_measured"], dtype=float))),
            float(np.max(np.asarray(d["Time"], dtype=float))),
            float(np.mean(np.asarray(d["Temperature_measured"], dtype=float))),
        ])
    if not rows:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)

def _load_cell_features(mat_path: str, cell: str) -> np.ndarray:
    mat = loadmat(mat_path, simplify_cells=True)
    cycles = mat[cell]["cycle"]
    if isinstance(cycles, dict):       # single-cycle edge case
        cycles = [cycles]
    return discharge_features(list(cycles))

def load_battery(config, cells=("B0005", "B0006", "B0007", "B0018"),
                 holdout="B0018", univariate=False, data_dir="data/battery") -> RealDataset:
    cells = list(cells)
    if holdout not in cells:
        raise ValueError(f"holdout {holdout!r} not in {cells}")
    paths = ensure_battery_files(cells, data_dir=data_dir)

    feats = {c: _load_cell_features(paths[c], c) for c in cells}
    L = min(f.shape[0] for f in feats.values())          # common length (truncate)
    feats = {c: f[:L] for c, f in feats.items()}
    if univariate:
        feats = {c: f[:, :1] for c, f in feats.items()}   # capacity only

    train_cells = [c for c in cells if c != holdout]
    train_raw = np.stack([feats[c] for c in train_cells], axis=0)   # (n_train, L, dim)
    test_raw = feats[holdout][None]                                 # (1, L, dim)

    scaler = Scaler().fit(train_raw.reshape(-1, train_raw.shape[-1]))
    train_traj = scaler.transform(train_raw).astype(np.float32)
    test_traj = scaler.transform(test_raw).astype(np.float32)

    covariates = np.linspace(0.0, 1.0, L).astype(np.float32)
    names = FEATURE_NAMES[:1] if univariate else FEATURE_NAMES
    meta = {"scaler": scaler, "feature_names": names, "capacity_index": 0,
            "holdout": holdout, "cells": cells, "raw_length": L}
    return RealDataset(train_traj, test_traj, covariates, train_traj.shape[-1], meta)

from __future__ import annotations
import numpy as np
import pandas as pd
from . import RealDataset
from .scaling import Scaler
from .download import ensure_cmapss_files

COLUMNS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
# Standard informative-sensor subset (constant sensors 1,5,6,10,16,18,19 dropped):
SENSOR_COLS = [f"s{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]

def parse_cmapss(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")          # trailing-whitespace columns
    df.columns = COLUMNS[: df.shape[1]]
    return df

def build_trajectories(df: pd.DataFrame, cols, length: int) -> np.ndarray:
    out = []
    for _, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        if len(g) < length:
            continue
        out.append(g[cols].to_numpy(dtype=np.float64)[:length])
    return np.asarray(out, dtype=np.float64)

def load_cmapss(config, subset="FD001", data_dir="data/cmapss",
                length: int | None = None, test_frac: float = 0.2) -> RealDataset:
    paths = ensure_cmapss_files(subset, data_dir=data_dir)
    df = parse_cmapss(paths["train"])
    if length is None:
        length = int(df.groupby("unit").size().min())   # common length = shortest train engine
    arr = build_trajectories(df, SENSOR_COLS, length)    # (n_units, length, dim)

    n_test = max(1, int(round(arr.shape[0] * test_frac)))
    test_raw, train_raw = arr[:n_test], arr[n_test:]

    scaler = Scaler().fit(train_raw.reshape(-1, train_raw.shape[-1]))
    train_traj = scaler.transform(train_raw).astype(np.float32)
    test_traj = scaler.transform(test_raw).astype(np.float32)
    covariates = np.linspace(0.0, 1.0, length).astype(np.float32)
    meta = {"scaler": scaler, "sensor_cols": SENSOR_COLS, "subset": subset, "length": length}
    return RealDataset(train_traj, test_traj, covariates, train_traj.shape[-1], meta)

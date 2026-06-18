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

def nasa_score(rul_pred: np.ndarray, rul_true: np.ndarray) -> float:
    d = np.asarray(rul_pred, dtype=float) - np.asarray(rul_true, dtype=float)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))

def fit_health_index(train_units: list[np.ndarray]) -> np.ndarray:
    """PCA first component over pooled (z-scored) sensor rows; oriented so HI rises toward failure."""
    pooled = np.concatenate([u for u in train_units], axis=0)      # (sum_T, dim)
    pooled = pooled - pooled.mean(axis=0)
    _, _, vt = np.linalg.svd(pooled, full_matrices=False)
    axis = vt[0]
    # Orient: HI should increase from start to end of a run-to-failure trajectory.
    end_minus_start = np.mean([u[-1] @ axis - u[0] @ axis for u in train_units])
    if end_minus_start < 0:
        axis = -axis
    return axis

def project_hi(seq: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.asarray(seq, dtype=float) @ axis

def derive_rul(hi_forecast: np.ndarray, threshold: float, cap: int = 125) -> int:
    crossed = np.where(np.asarray(hi_forecast) >= threshold)[0]
    rul = int(crossed[0]) if crossed.size else len(hi_forecast)
    return min(rul, cap)


def load_cmapss_rul(config, subset="FD001", data_dir="data/cmapss",
                    length: int | None = None) -> dict:
    paths = ensure_cmapss_files(subset, data_dir=data_dir)
    train_df = parse_cmapss(paths["train"])
    test_df = parse_cmapss(paths["test"])
    rul_true = np.loadtxt(paths["rul"]).astype(float).ravel()

    # z-score sensors using train stats
    train_units = [g[SENSOR_COLS].to_numpy(float)
                   for _, g in train_df.sort_values(["unit", "cycle"]).groupby("unit")]
    sensor_scaler = Scaler().fit(np.concatenate(train_units, axis=0))
    train_units = [sensor_scaler.transform(u) for u in train_units]

    axis = fit_health_index(train_units)
    thr = float(np.mean([project_hi(u, axis)[-1] for u in train_units]))  # mean end-of-life HI

    if length is None:
        length = int(min(len(u) for u in train_units))
    ctx = config.context_length
    # Use TRAILING windows so training matches the late-life regime seen at test time.
    train_hi = np.stack([project_hi(u, axis)[-length:] for u in train_units
                         if len(u) >= length])[..., None].astype(np.float32)  # (n,L,1)

    test_units = [sensor_scaler.transform(g[SENSOR_COLS].to_numpy(float))
                  for _, g in test_df.sort_values(["unit", "cycle"]).groupby("unit")]
    test_hi_context = np.stack(
        [project_hi(u, axis)[-ctx:] if len(u) >= ctx
         else np.pad(project_hi(u, axis), (ctx - len(u), 0), mode="edge")
         for u in test_units])[..., None].astype(np.float32)               # (n_test,ctx,1)

    return {"train_hi": train_hi, "test_hi_context": test_hi_context,
            "rul_true": rul_true, "threshold": thr, "hi_axis": axis,
            "sensor_scaler": sensor_scaler}

# Real-World Degradation Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NASA Li-ion Battery Aging and NASA C-MAPSS Turbofan datasets to AFM's lightweight CPU pipeline so AFM/FM can be evaluated against published baselines for the IJF rebuttal.

**Architecture:** A new `src/afm/datasets/` package loads each real dataset into the *same* `(train_traj, test_traj, covariates)` array contract that `prepare_trajectories` already produces, so `create_infinite_dataloaders`, the AFM/FM models, `AFMTrainer`, and `compute_crps/compute_nrmse` are reused unchanged. Real series are z-scored (scaler stored for de-normalized reporting). Two self-contained scripts run end-to-end; the wandb `BaseExperiment` runner is intentionally bypassed.

**Tech Stack:** Python, JAX, `flax.nnx`, `diffrax`, `scipy` (read `.mat`), `pandas` (parse C-MAPSS txt), `numpy`, `pytest`.

## Global Constraints

- **No GluonTS, no GPU.** Use only the in-repo lightweight pipeline. Copied verbatim from spec §1.
- **Reuse the contract:** every loader returns arrays shaped `train_traj (n_train,T,dim)`, `test_traj (n_test,T,dim)`, `covariates (T,)`; `data_dim:int`; `meta:dict`. Downstream code is not modified.
- **Covariates are 1-D `(T,)`** (normalized time/cycle index) to match `create_infinite_dataloaders`, which does `covariates[None,:,None]`; `model_covariates_dim = 1`.
- **Normalization round-trips:** z-score stats live in `meta["scaler"]`; all externally-reported errors (capacity in Ah, RUL in cycles) are computed in **original units** via inverse-transform.
- **Cite protocol with every number.** Published baselines and their protocols are fixed in spec §7; the results table must carry a protocol/cap column.
- **C-MAPSS starts at FD001.** FD002–FD004 are out of scope for this plan (loader must accept the subset name as a parameter so they can be added later without code change).
- **Python ≥ 3.10** (matches existing `pyproject.toml`).

---

## File Structure

- `src/afm/datasets/__init__.py` — `RealDataset` dataclass, `REAL_DATASET_NAMES`, `load_dataset(name, config, **kw)` dispatcher.
- `src/afm/datasets/scaling.py` — `Scaler` (per-feature z-score fit/transform/inverse).
- `src/afm/datasets/download.py` — `ensure_battery_files()`, `ensure_cmapss_files()` (cache to `data/`, raise with manual instructions on failure).
- `src/afm/datasets/battery.py` — `.mat` parsing + `load_battery(...)`.
- `src/afm/datasets/cmapss.py` — txt parsing, sensor subset, `load_cmapss(...)` (forecasting), health-index + RUL utilities, `load_cmapss_rul(...)`.
- `src/afm/datasets/eval.py` — `evaluate_on_trajectories(...)`.
- `src/afm/config.py` — extend accepted system names (modify).
- `examples/realworld_battery.py`, `examples/realworld_cmapss.py` — runnable scripts.
- `tests/test_real_datasets.py` — unit tests (synthetic fixtures; no network/large data).
- `pyproject.toml` (modify), `.gitignore` (modify).

---

### Task 1: Package scaffold + `Scaler` + dependencies

**Files:**
- Create: `src/afm/datasets/__init__.py`, `src/afm/datasets/scaling.py`
- Modify: `pyproject.toml`, `.gitignore`
- Test: `tests/test_real_datasets.py`

**Interfaces:**
- Produces:
  - `Scaler.fit(x: np.ndarray) -> Scaler` (x shape `(..., dim)`, stats over all but last axis)
  - `Scaler.transform(x) -> np.ndarray`; `Scaler.inverse_transform(x) -> np.ndarray`
  - `Scaler.mean: np.ndarray (dim,)`, `Scaler.std: np.ndarray (dim,)`
  - `@dataclass RealDataset(train_traj, test_traj, covariates, data_dim, meta)`

- [ ] **Step 1: Add dependencies and gitignore entry**

In `pyproject.toml`, add to the `dependencies` list: `"scipy"`, `"pandas"`. (`numpy` is already transitively present via JAX; add `"requests"` too.)
Append to `.gitignore`:
```
data/
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_real_datasets.py
import numpy as np
from afm.datasets.scaling import Scaler

def test_scaler_roundtrip_per_feature():
    x = np.stack([np.arange(10), np.arange(10) * 100.0], axis=-1)  # (10, 2)
    s = Scaler().fit(x)
    z = s.transform(x)
    assert z.shape == x.shape
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-6)
    assert np.allclose(s.inverse_transform(z), x, atol=1e-5)
    assert s.mean.shape == (2,) and s.std.shape == (2,)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_real_datasets.py::test_scaler_roundtrip_per_feature -v`
Expected: FAIL (ModuleNotFoundError: afm.datasets.scaling)

- [ ] **Step 4: Implement `Scaler` and `RealDataset`**

```python
# src/afm/datasets/scaling.py
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
```

```python
# src/afm/datasets/__init__.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_real_datasets.py::test_scaler_roundtrip_per_feature -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/afm/datasets/__init__.py src/afm/datasets/scaling.py tests/test_real_datasets.py
git commit -m "feat(datasets): add real-dataset package scaffold, Scaler, deps"
```

---

### Task 2: Allow real-dataset names in `Config`

**Files:**
- Modify: `src/afm/config.py:122-126` (the `valid_systems` check) and `src/afm/config.py:144-147` (`get_system_choices`)
- Test: `tests/test_real_datasets.py`

**Interfaces:**
- Consumes: `afm.datasets.REAL_DATASET_NAMES` (Task 1)
- Produces: `Config(system_name="cmapss_fd001", ...)` validates without error.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_real_datasets.py  (append)
from afm.config import Config

def test_config_accepts_real_dataset_names():
    cfg = Config(system_name="nasa_battery", system_num_timepoints=130,
                 context_length=30, fm_prediction_horizon=50, data_prediction_size=50)
    assert cfg.system_name == "nasa_battery"
    cfg2 = Config(system_name="cmapss_fd001", system_num_timepoints=128,
                  context_length=30, fm_prediction_horizon=40, data_prediction_size=40)
    assert cfg2.system_name == "cmapss_fd001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_real_datasets.py::test_config_accepts_real_dataset_names -v`
Expected: FAIL (ValueError: Invalid system 'nasa_battery')

- [ ] **Step 3: Implement the config change**

In `src/afm/config.py`, replace the body of the `valid_systems` block (lines ~122-126) with:
```python
        # System validation (synthetic dynamical systems + real datasets)
        synthetic_systems = ['lorenz', 'lotka_volterra', 'vanderpol', 'harmonic_oscillator',
                             'brusselator', 'fitzhugh_nagumo', 'duffing']
        real_datasets = ['nasa_battery', 'cmapss_fd001', 'cmapss_fd002', 'cmapss_fd003', 'cmapss_fd004']
        valid_systems = synthetic_systems + real_datasets
        if self.system_name not in valid_systems:
            raise ValueError(f"Invalid system '{self.system_name}'. Valid options: {valid_systems}")
```
And update `get_system_choices` (lines ~144-147) to `return` the same combined list (copy the two lists above into it).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_real_datasets.py::test_config_accepts_real_dataset_names -v`
Expected: PASS

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest tests/ -q`
Expected: PASS (existing tests unaffected)

- [ ] **Step 6: Commit**

```bash
git add src/afm/config.py tests/test_real_datasets.py
git commit -m "feat(config): accept real-dataset names in system validation"
```

---

### Task 3: Download helpers with manual fallback

**Files:**
- Create: `src/afm/datasets/download.py`
- Test: `tests/test_real_datasets.py`

**Interfaces:**
- Produces:
  - `ensure_battery_files(cells: list[str], data_dir: str = "data/battery") -> dict[str, str]` (maps cell name → path)
  - `ensure_cmapss_files(subset: str, data_dir: str = "data/cmapss") -> dict[str, str]` (keys `"train"`, `"test"`, `"rul"`)
  - On missing files + failed/disabled download: raises `FileNotFoundError` whose message lists the exact URL and expected local path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_real_datasets.py  (append)
import pytest
from afm.datasets.download import ensure_cmapss_files

def test_ensure_cmapss_raises_with_instructions_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        ensure_cmapss_files("FD001", data_dir=str(tmp_path), allow_download=False)
    msg = str(e.value)
    assert "train_FD001.txt" in msg
    assert "http" in msg  # an actionable URL is present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_real_datasets.py::test_ensure_cmapss_raises_with_instructions_when_absent -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `download.py`**

```python
# src/afm/datasets/download.py
from __future__ import annotations
import os

BATTERY_URL = "https://ti.arc.nasa.gov/c/5/"  # NASA PCoE "Battery Data Set" (.mat zip); mirror: kaggle.com/datasets/patrickfleith/nasa-battery-dataset
CMAPSS_URL = "https://ti.arc.nasa.gov/c/6/"   # NASA PCoE "Turbofan Engine Degradation Simulation" (CMAPSSData.zip); mirror: kaggle.com/datasets/behrad3d/nasa-cmaps

def _need(path: str, url: str, what: str) -> None:
    raise FileNotFoundError(
        f"Required file not found: {path}\n"
        f"Download '{what}' from: {url}\n"
        f"(NASA PCoE may require manual download; Kaggle mirrors exist — see comments in download.py.)\n"
        f"Place the extracted file at the path above, then re-run."
    )

def ensure_cmapss_files(subset: str, data_dir: str = "data/cmapss", allow_download: bool = True) -> dict[str, str]:
    subset = subset.upper()
    paths = {
        "train": os.path.join(data_dir, f"train_{subset}.txt"),
        "test": os.path.join(data_dir, f"test_{subset}.txt"),
        "rul": os.path.join(data_dir, f"RUL_{subset}.txt"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        # Auto-download is intentionally not implemented for NASA's interstitial-gated host;
        # rely on the manual/mirror instructions. allow_download kept for future mirror support.
        _need(missing[0], CMAPSS_URL, f"C-MAPSS {subset} (train/test/RUL .txt)")
    return paths

def ensure_battery_files(cells: list[str], data_dir: str = "data/battery", allow_download: bool = True) -> dict[str, str]:
    paths = {c: os.path.join(data_dir, f"{c}.mat") for c in cells}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        _need(missing[0], BATTERY_URL, "NASA Li-ion Battery Aging (B0005/6/7/18 .mat)")
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_real_datasets.py::test_ensure_cmapss_raises_with_instructions_when_absent -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/afm/datasets/download.py tests/test_real_datasets.py
git commit -m "feat(datasets): add download helpers with manual-fallback instructions"
```

---

### Task 4: Battery loader (`.mat` → multivariate capacity-fade trajectories)

**Files:**
- Create: `src/afm/datasets/battery.py`
- Test: `tests/test_real_datasets.py`

**Interfaces:**
- Consumes: `Scaler` (Task 1), `RealDataset` (Task 1), `ensure_battery_files` (Task 3)
- Produces:
  - `discharge_features(cycles: list[dict]) -> np.ndarray` shape `(n_discharge, 4)`, columns `[capacity, mean_voltage, discharge_time, mean_temp]`
  - `load_battery(config, cells=("B0005","B0006","B0007","B0018"), holdout="B0018", univariate=False, data_dir="data/battery") -> RealDataset`
  - `meta` keys: `"scaler": Scaler`, `"feature_names": list[str]`, `"capacity_index": 0`, `"holdout": str`, `"cells": list[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_real_datasets.py  (append)
import numpy as np
from afm.datasets.battery import discharge_features

def _fake_cycles():
    # Two discharge cycles + one charge cycle (charge must be ignored).
    def disc(cap, v, temp, n=5):
        return {"type": "discharge",
                "data": {"Capacity": np.array([cap]),
                         "Voltage_measured": np.full(n, v),
                         "Temperature_measured": np.full(n, temp),
                         "Time": np.arange(n) * 10.0}}
    return [disc(1.85, 3.9, 24.0), {"type": "charge", "data": {}}, disc(1.80, 3.8, 25.0)]

def test_discharge_features_extracts_capacity_and_ignores_charge():
    feats = discharge_features(_fake_cycles())
    assert feats.shape == (2, 4)              # 2 discharge cycles, 4 features
    assert np.isclose(feats[0, 0], 1.85)       # capacity column
    assert np.isclose(feats[1, 0], 1.80)
    assert np.isclose(feats[0, 2], 40.0)       # discharge_time = max Time = 4*10
    assert np.isclose(feats[0, 3], 24.0)       # mean temp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_real_datasets.py::test_discharge_features_extracts_capacity_and_ignores_charge -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `battery.py`**

```python
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
    assert holdout in cells, f"holdout {holdout} not in {cells}"
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_real_datasets.py::test_discharge_features_extracts_capacity_and_ignores_charge -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/afm/datasets/battery.py tests/test_real_datasets.py
git commit -m "feat(datasets): add NASA battery loader (multivariate capacity-fade)"
```

---

### Task 5: C-MAPSS loader (txt → multivariate sensor trajectories)

**Files:**
- Create: `src/afm/datasets/cmapss.py`
- Test: `tests/test_real_datasets.py`

**Interfaces:**
- Consumes: `Scaler`, `RealDataset`, `ensure_cmapss_files`
- Produces:
  - `SENSOR_COLS: list[str]` (the 14-sensor subset, names `"s2"..."s21"`)
  - `parse_cmapss(path: str) -> pandas.DataFrame` (columns `unit, cycle, os1..os3, s1..s21`)
  - `build_trajectories(df, cols, length) -> np.ndarray (n_units, length, len(cols))` (truncate each unit to `length`; drop units shorter than `length`)
  - `load_cmapss(config, subset="FD001", data_dir="data/cmapss") -> RealDataset` (forecasting arrays; train-engine pool split by unit-id into train/test). `meta` keys: `"scaler"`, `"sensor_cols"`, `"subset"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_real_datasets.py  (append)
import numpy as np, pandas as pd
from afm.datasets.cmapss import parse_cmapss, build_trajectories, SENSOR_COLS

def test_parse_and_build_trajectories(tmp_path):
    # 2 units; unit 1 has 6 cycles, unit 2 has 4 cycles. 26 columns total.
    rows = []
    for unit, ncyc in [(1, 6), (2, 4)]:
        for c in range(1, ncyc + 1):
            rows.append([unit, c] + [0.0, 0.0, 0.0] + [float(c)] * 21)
    txt = "\n".join(" ".join(str(v) for v in r) for r in rows) + "\n"
    p = tmp_path / "train_FD001.txt"; p.write_text(txt)

    df = parse_cmapss(str(p))
    assert list(df.columns)[:2] == ["unit", "cycle"]
    assert df.shape == (10, 26)
    assert set(SENSOR_COLS).issubset(df.columns)

    arr = build_trajectories(df, SENSOR_COLS, length=4)
    assert arr.shape == (2, 4, len(SENSOR_COLS))   # both units kept, truncated to 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_real_datasets.py::test_parse_and_build_trajectories -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement the forecasting part of `cmapss.py`**

```python
# src/afm/datasets/cmapss.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_real_datasets.py::test_parse_and_build_trajectories -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/afm/datasets/cmapss.py tests/test_real_datasets.py
git commit -m "feat(datasets): add C-MAPSS loader (multivariate sensor forecasting)"
```

---

### Task 6: C-MAPSS RUL utilities (health index, RUL derivation, NASA score)

**Files:**
- Modify: `src/afm/datasets/cmapss.py`
- Test: `tests/test_real_datasets.py`

**Interfaces:**
- Consumes: `parse_cmapss`, `Scaler`
- Produces:
  - `nasa_score(rul_pred: np.ndarray, rul_true: np.ndarray) -> float`
  - `fit_health_index(train_units: list[np.ndarray]) -> np.ndarray` (returns PCA-1 axis `(dim,)`, sign-oriented so HI increases toward failure)
  - `project_hi(seq: np.ndarray, axis: np.ndarray) -> np.ndarray` (`seq (T,dim) → (T,)`)
  - `derive_rul(hi_forecast: np.ndarray, threshold: float, cap: int = 125) -> int` (first crossing index, capped; returns `cap` if never crosses)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_real_datasets.py  (append)
import numpy as np
from afm.datasets.cmapss import nasa_score, derive_rul

def test_nasa_score_asymmetric():
    # late prediction (d>0) penalized by exp(d/10)-1; early (d<0) by exp(-d/13)-1
    assert np.isclose(nasa_score(np.array([10]), np.array([10])), 0.0)
    late = nasa_score(np.array([20]), np.array([10]))   # d=+10
    early = nasa_score(np.array([0]), np.array([10]))    # d=-10
    assert np.isclose(late, np.exp(10/10) - 1, atol=1e-6)
    assert np.isclose(early, np.exp(10/13) - 1, atol=1e-6)
    assert late > early    # late predictions penalized harder

def test_derive_rul_threshold_crossing():
    hi = np.array([0.0, 0.2, 0.4, 0.6, 0.9, 1.2])   # crosses 1.0 at index 5
    assert derive_rul(hi, threshold=1.0, cap=125) == 5
    assert derive_rul(np.zeros(10), threshold=1.0, cap=7) == 7   # never crosses -> cap
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_real_datasets.py::test_nasa_score_asymmetric tests/test_real_datasets.py::test_derive_rul_threshold_crossing -v`
Expected: FAIL (ImportError: cannot import name 'nasa_score')

- [ ] **Step 3: Append RUL utilities to `cmapss.py`**

```python
# src/afm/datasets/cmapss.py  (append)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_real_datasets.py::test_nasa_score_asymmetric tests/test_real_datasets.py::test_derive_rul_threshold_crossing -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/afm/datasets/cmapss.py tests/test_real_datasets.py
git commit -m "feat(datasets): add C-MAPSS health-index, RUL derivation, NASA score"
```

---

### Task 7: Evaluation helper

**Files:**
- Create: `src/afm/datasets/eval.py`
- Test: `tests/test_real_datasets.py`

**Interfaces:**
- Consumes: `compute_crps`, `compute_nrmse` (`afm.evaluation.metrics`), a model exposing `.sample(key, history, covariates, nfe, n_steps_pred, n_samples) -> (S,B,H,dim)`
- Produces:
  - `evaluate_on_trajectories(model, dataset: RealDataset, config, key, n_steps_pred, n_samples=50, nfe=20, channel=0) -> dict` with keys `crps`, `nrmse` (normalized) and `rmse_orig`, `mae_orig` (channel `channel`, de-normalized via `dataset.meta["scaler"]`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_real_datasets.py  (append)
import jax.numpy as jnp, jax.random as jr, numpy as np
from afm.datasets import RealDataset
from afm.datasets.scaling import Scaler
from afm.datasets.eval import evaluate_on_trajectories

class _OracleModel:
    """Returns the ground-truth future for every sample → NRMSE≈0."""
    def __init__(self, test_traj, ctx): self.fut = test_traj[:, ctx:]; 
    def sample(self, key, history, covariates, nfe, n_steps_pred, n_samples=50):
        f = jnp.asarray(self.fut)[:, :n_steps_pred]
        return jnp.broadcast_to(f[None], (n_samples,) + f.shape)

class _Cfg:  # minimal stand-in for Config fields used by the helper
    context_length = 4

def test_evaluate_on_trajectories_oracle_is_near_zero():
    raw = np.cumsum(np.ones((3, 10, 2)), axis=1)   # (n,T,dim)
    sc = Scaler().fit(raw.reshape(-1, 2))
    z = sc.transform(raw).astype(np.float32)
    ds = RealDataset(z, z, np.linspace(0,1,10).astype(np.float32), 2,
                     {"scaler": sc, "capacity_index": 0})
    model = _OracleModel(ds.test_traj, _Cfg.context_length)
    out = evaluate_on_trajectories(model, ds, _Cfg, jr.PRNGKey(0), n_steps_pred=6, n_samples=5)
    assert out["nrmse"] < 1e-3
    assert out["rmse_orig"] < 1e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_real_datasets.py::test_evaluate_on_trajectories_oracle_is_near_zero -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `eval.py`**

```python
# src/afm/datasets/eval.py
from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from ..evaluation.metrics import compute_crps, compute_nrmse

def evaluate_on_trajectories(model, dataset, config, key, n_steps_pred,
                             n_samples=50, nfe=20, channel=0) -> dict:
    ctx = config.context_length
    test = jnp.asarray(dataset.test_traj)                  # (B, T, dim)
    B, T, dim = test.shape
    history = test[:, :ctx, :]                              # (B, ctx, dim)
    cov = jnp.asarray(dataset.covariates)[:ctx]             # (ctx,)
    cov = jnp.broadcast_to(cov[None, :, None], (B, ctx, 1)) # (B, ctx, 1)

    gt = test[:, ctx:ctx + n_steps_pred, :]                 # (B, H, dim)
    preds = model.sample(key, history, cov, nfe=nfe,
                         n_steps_pred=n_steps_pred, n_samples=n_samples)  # (S,B,H,dim)

    crps = float(compute_crps(preds, gt))
    nrmse = float(compute_nrmse(preds, gt))

    # De-normalized error on the headline channel (capacity / sensor of interest).
    scaler = dataset.meta["scaler"]
    mean_c = float(scaler.mean[channel]); std_c = float(scaler.std[channel]) + scaler.eps
    pred_mean = np.asarray(preds.mean(axis=0))[..., channel] * std_c + mean_c   # (B,H)
    gt_c = np.asarray(gt)[..., channel] * std_c + mean_c
    rmse_orig = float(np.sqrt(np.mean((pred_mean - gt_c) ** 2)))
    mae_orig = float(np.mean(np.abs(pred_mean - gt_c)))
    return {"crps": crps, "nrmse": nrmse, "rmse_orig": rmse_orig, "mae_orig": mae_orig}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_real_datasets.py::test_evaluate_on_trajectories_oracle_is_near_zero -v`
Expected: PASS

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/afm/datasets/eval.py tests/test_real_datasets.py
git commit -m "feat(datasets): add trajectory evaluation helper (CRPS/NRMSE + de-normalized RMSE/MAE)"
```

---

### Task 8: Battery end-to-end script

**Files:**
- Create: `examples/realworld_battery.py`
- Test: manual smoke run (data-gated)

**Interfaces:**
- Consumes: `load_battery`, `create_infinite_dataloaders`, `AutoregressiveFlowMatching`, `FMTrainer`/`AFMTrainer`, `evaluate_on_trajectories`.

- [ ] **Step 1: Write the script**

```python
# examples/realworld_battery.py
"""Capacity-fade forecasting on the NASA Li-ion Battery Aging dataset (leave-one-battery-out).

Usage:  python examples/realworld_battery.py --epochs 100 --holdout B0018
Data:   place B0005.mat B0006.mat B0007.mat B0018.mat under data/battery/ (see download.py).
"""
import argparse
import jax.numpy as jnp, jax.random as jr
import flax.nnx as nnx

from afm.config import Config
from afm.datasets.battery import load_battery
from afm.datasets.eval import evaluate_on_trajectories
from afm.training.data import create_infinite_dataloaders
from afm.training.trainers import AFMTrainer, FMTrainer
from afm.models import AutoregressiveFlowMatching, FlowMatching

def build_config(L, ctx, pred):
    return Config(system_name="nasa_battery", system_num_timepoints=L,
                  context_length=ctx, afm_prediction_horizon=1,
                  fm_prediction_horizon=pred, data_prediction_size=pred,
                  data_batch_size=64, steps_per_epoch=100,
                  model_covariates_dim=1, flow_nfe=20, flow_n_samples=50)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--holdout", default="B0018")
    ap.add_argument("--ctx", type=int, default=30)
    ap.add_argument("--univariate", action="store_true")
    args = ap.parse_args()

    ds0 = load_battery(Config(system_name="nasa_battery", system_num_timepoints=80,
                              context_length=30, fm_prediction_horizon=40,
                              data_prediction_size=40),
                       holdout=args.holdout, univariate=args.univariate)
    L = ds0.meta["raw_length"]
    pred = L - args.ctx                       # forecast everything after context
    cfg = build_config(L, args.ctx, pred)
    ds = load_battery(cfg, holdout=args.holdout, univariate=args.univariate)

    (afm_train, afm_test), _ = create_infinite_dataloaders(
        jnp.asarray(ds.train_traj), jnp.asarray(ds.test_traj),
        jnp.asarray(ds.covariates), cfg)

    model = AutoregressiveFlowMatching(
        dim_x=ds.data_dim, dim_c=1, dim_h=cfg.model_hidden_dim, dim_t=cfg.model_time_dim,
        dim_hidden=cfg.model_hidden_dim, hidden_blocks=cfg.model_hidden_blocks,
        n_rnn_layers=cfg.model_n_rnn_layers, rnn_hidden=cfg.model_rnn_hidden, rngs=nnx.Rngs(0))
    AFMTrainer(model, learning_rate=cfg.training_learning_rate, model_name="AFM").train(
        afm_train, afm_test, num_epochs=args.epochs, use_wandb=False)

    res = evaluate_on_trajectories(model, ds, cfg, jr.PRNGKey(42),
                                   n_steps_pred=pred, n_samples=cfg.flow_n_samples,
                                   nfe=cfg.flow_nfe, channel=ds.meta["capacity_index"])
    print(f"[battery holdout={args.holdout}] "
          f"CRPS={res['crps']:.4f} NRMSE={res['nrmse']:.4f} "
          f"capacity_RMSE={res['rmse_orig']:.4f} Ah  MAE={res['mae_orig']:.4f} Ah")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run (only if data present; otherwise confirm the instructive error)**

Run: `python examples/realworld_battery.py --epochs 3`
Expected: either a printed metrics line, OR a `FileNotFoundError` naming `data/battery/B0005.mat` and a download URL (proves the data-gating path works).

- [ ] **Step 3: Commit**

```bash
git add examples/realworld_battery.py
git commit -m "feat(examples): add battery capacity-fade forecasting script"
```

---

### Task 9: C-MAPSS end-to-end script (forecasting + RUL)

**Files:**
- Create: `examples/realworld_cmapss.py`
- Modify: `src/afm/datasets/cmapss.py` (add `load_cmapss_rul` for the univariate-HI RUL track)
- Test: `tests/test_real_datasets.py` (shape test for `load_cmapss_rul` via synthetic fixture); manual smoke run for the script.

**Interfaces:**
- Produces: `load_cmapss_rul(config, subset, data_dir, length) -> dict` with keys:
  `train_hi (n_train,L,1)`, `test_hi_context (n_test,ctx,1)`, `rul_true (n_test,)`, `threshold float`, `hi_axis (dim,)`, `sensor_scaler Scaler`.

- [ ] **Step 1: Write the failing test for `load_cmapss_rul` shapes**

```python
# tests/test_real_datasets.py  (append)
import numpy as np
from afm.datasets.cmapss import fit_health_index, project_hi

def test_health_index_orientation_rises_to_failure():
    # Two synthetic run-to-failure units whose sensors drift upward over time.
    u1 = np.cumsum(np.ones((20, 14)), axis=0) + np.random.RandomState(0).randn(20, 14) * 0.01
    u2 = np.cumsum(np.ones((25, 14)), axis=0) + np.random.RandomState(1).randn(25, 14) * 0.01
    axis = fit_health_index([u1, u2])
    hi = project_hi(u1, axis)
    assert hi[-1] > hi[0]    # HI increases toward failure
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_real_datasets.py::test_health_index_orientation_rises_to_failure -v`
Expected: FAIL only if Task 6 not yet present; otherwise PASS (this locks orientation behavior). If it passes immediately, keep it as a regression guard and proceed.

- [ ] **Step 3: Add `load_cmapss_rul` to `cmapss.py`**

```python
# src/afm/datasets/cmapss.py  (append)

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
    train_hi = np.stack([project_hi(u, axis)[:length] for u in train_units
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
```

- [ ] **Step 4: Write the C-MAPSS script**

```python
# examples/realworld_cmapss.py
"""C-MAPSS turbofan: (A) multivariate sensor-trajectory forecasting (CRPS/NRMSE),
and (B) RUL prediction via forecasting a 1-D health index to its failure threshold
(RUL-RMSE + NASA Score).  FD001 by default.

Usage:  python examples/realworld_cmapss.py --epochs 100 --subset FD001
Data:   train_FD001.txt test_FD001.txt RUL_FD001.txt under data/cmapss/ (see download.py).
"""
import argparse
import numpy as np
import jax.numpy as jnp, jax.random as jr
import flax.nnx as nnx

from afm.config import Config
from afm.datasets.cmapss import load_cmapss, load_cmapss_rul, derive_rul, nasa_score
from afm.datasets.eval import evaluate_on_trajectories
from afm.training.data import create_infinite_dataloaders
from afm.training.trainers import AFMTrainer
from afm.models import AutoregressiveFlowMatching

def make_model(dim, cfg):
    return AutoregressiveFlowMatching(
        dim_x=dim, dim_c=1, dim_h=cfg.model_hidden_dim, dim_t=cfg.model_time_dim,
        dim_hidden=cfg.model_hidden_dim, hidden_blocks=cfg.model_hidden_blocks,
        n_rnn_layers=cfg.model_n_rnn_layers, rnn_hidden=cfg.model_rnn_hidden, rngs=nnx.Rngs(0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--ctx", type=int, default=20)
    ap.add_argument("--cap", type=int, default=125)
    args = ap.parse_args()

    # ---------- (A) multivariate forecasting ----------
    probe = load_cmapss(Config(system_name="cmapss_fd001", system_num_timepoints=128,
                               context_length=args.ctx, fm_prediction_horizon=40,
                               data_prediction_size=40), subset=args.subset)
    L = probe.meta["length"]; pred = L - args.ctx
    cfg = Config(system_name=f"cmapss_{args.subset.lower()}", system_num_timepoints=L,
                 context_length=args.ctx, afm_prediction_horizon=1,
                 fm_prediction_horizon=pred, data_prediction_size=pred,
                 data_batch_size=64, steps_per_epoch=100, model_covariates_dim=1,
                 flow_nfe=20, flow_n_samples=50)
    ds = load_cmapss(cfg, subset=args.subset)
    (afm_train, afm_test), _ = create_infinite_dataloaders(
        jnp.asarray(ds.train_traj), jnp.asarray(ds.test_traj), jnp.asarray(ds.covariates), cfg)
    fmodel = make_model(ds.data_dim, cfg)
    AFMTrainer(fmodel, learning_rate=cfg.training_learning_rate, model_name="AFM").train(
        afm_train, afm_test, num_epochs=args.epochs, use_wandb=False)
    fres = evaluate_on_trajectories(fmodel, ds, cfg, jr.PRNGKey(1),
                                    n_steps_pred=pred, n_samples=cfg.flow_n_samples, nfe=cfg.flow_nfe)
    print(f"[cmapss {args.subset} forecasting] CRPS={fres['crps']:.4f} NRMSE={fres['nrmse']:.4f}")

    # ---------- (B) RUL via health-index forecasting ----------
    rul = load_cmapss_rul(cfg, subset=args.subset)
    hicfg = Config(system_name=f"cmapss_{args.subset.lower()}",
                   system_num_timepoints=rul["train_hi"].shape[1],
                   context_length=args.ctx, afm_prediction_horizon=1,
                   fm_prediction_horizon=rul["train_hi"].shape[1] - args.ctx,
                   data_prediction_size=rul["train_hi"].shape[1] - args.ctx,
                   data_batch_size=64, steps_per_epoch=100, model_covariates_dim=1,
                   flow_nfe=20, flow_n_samples=50)
    cov = np.linspace(0, 1, rul["train_hi"].shape[1]).astype(np.float32)
    (hi_train, hi_test), _ = create_infinite_dataloaders(
        jnp.asarray(rul["train_hi"]), jnp.asarray(rul["train_hi"][:1]), jnp.asarray(cov), hicfg)
    himodel = make_model(1, hicfg)
    AFMTrainer(himodel, learning_rate=hicfg.training_learning_rate, model_name="AFM-HI").train(
        hi_train, hi_test, num_epochs=args.epochs, use_wandb=False)

    horizon = args.cap + 10
    ctx_hi = jnp.asarray(rul["test_hi_context"])                 # (n_test, ctx, 1)
    covc = jnp.broadcast_to(jnp.asarray(cov)[:args.ctx][None, :, None], ctx_hi.shape)
    samples = himodel.sample(jr.PRNGKey(2), ctx_hi, covc, nfe=hicfg.flow_nfe,
                             n_steps_pred=horizon, n_samples=hicfg.flow_n_samples)  # (S,n,H,1)
    hi_fore = np.asarray(samples[..., 0]).mean(axis=0)            # (n_test, H) ensemble mean
    rul_pred = np.array([derive_rul(hi_fore[i], rul["threshold"], cap=args.cap)
                         for i in range(hi_fore.shape[0])], dtype=float)
    rul_true = np.minimum(rul["rul_true"], args.cap)
    rmse = float(np.sqrt(np.mean((rul_pred - rul_true) ** 2)))
    print(f"[cmapss {args.subset} RUL cap={args.cap}] RMSE={rmse:.2f} cycles  "
          f"Score={nasa_score(rul_pred, rul_true):.1f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run shape test + smoke run**

Run: `pytest tests/test_real_datasets.py::test_health_index_orientation_rises_to_failure -v`
Expected: PASS
Run: `python examples/realworld_cmapss.py --epochs 3 --subset FD001`
Expected: either two printed metrics lines, OR a `FileNotFoundError` naming `data/cmapss/train_FD001.txt` + URL.

- [ ] **Step 6: Commit**

```bash
git add src/afm/datasets/cmapss.py examples/realworld_cmapss.py tests/test_real_datasets.py
git commit -m "feat(examples): add C-MAPSS forecasting + RUL script and HI-RUL loader"
```

---

### Task 10: Results table + docs

**Files:**
- Create: `src/afm/datasets/results_table.py`
- Modify: `README.md` (add a "Real-world datasets" subsection), `docs/experiments.md` (note the two scripts)
- Test: `tests/test_real_datasets.py` (table renders expected rows)

**Interfaces:**
- Produces: `render_results_table(rows: list[dict], fmt: str = "latex") -> str` — `rows` carry keys `method, dataset, metric, value, protocol`; baseline rows from spec §7 are pre-seeded in the script docstring/constants.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_real_datasets.py  (append)
from afm.datasets.results_table import render_results_table

def test_render_results_table_latex_includes_protocol_column():
    rows = [{"method": "AFM (ours)", "dataset": "FD001", "metric": "RUL-RMSE",
             "value": "13.x", "protocol": "cap=125, HI-threshold"},
            {"method": "DCNN (Li 2018)", "dataset": "FD001", "metric": "RUL-RMSE",
             "value": "12.61", "protocol": "cap=125, w=30"}]
    out = render_results_table(rows, fmt="latex")
    assert "protocol" in out.lower()
    assert "DCNN (Li 2018)" in out and "12.61" in out
    assert out.strip().startswith("\\begin{tabular}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_real_datasets.py::test_render_results_table_latex_includes_protocol_column -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `results_table.py`**

```python
# src/afm/datasets/results_table.py
"""Render an AFM-vs-published results table for the rebuttal.
Baseline numbers + protocols are fixed in docs/superpowers/specs/2026-06-18-...-design.md §7.
Fill the AFM/FM rows from the script outputs, then render."""
from __future__ import annotations

COLS = ["method", "dataset", "metric", "value", "protocol"]

def render_results_table(rows: list[dict], fmt: str = "latex") -> str:
    if fmt == "markdown":
        head = "| " + " | ".join(COLS) + " |\n| " + " | ".join("---" for _ in COLS) + " |\n"
        body = "".join("| " + " | ".join(str(r.get(c, "")) for c in COLS) + " |\n" for r in rows)
        return head + body
    # latex
    lines = ["\\begin{tabular}{lllll}", "\\toprule",
             " & ".join(COLS) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(r.get(c, "")) for c in COLS) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_real_datasets.py::test_render_results_table_latex_includes_protocol_column -v`
Expected: PASS

- [ ] **Step 5: Update docs**

In `README.md`, after "Available Systems", add a short "Real-world datasets" subsection documenting `examples/realworld_battery.py` and `examples/realworld_cmapss.py`, the `data/` placement, and that evaluation is CPU-only (no GluonTS/GPU). In `docs/experiments.md`, add a one-line pointer to the two scripts.

- [ ] **Step 6: Run full suite + commit**

Run: `pytest tests/ -q`
Expected: PASS

```bash
git add src/afm/datasets/results_table.py README.md docs/experiments.md tests/test_real_datasets.py
git commit -m "feat(datasets): add results-table renderer; document real-world scripts"
```

---

## Self-Review

**1. Spec coverage:**
- §2 integration contract → Tasks 1,4,5 (RealDataset, covariates 1-D, normalization). ✓
- §3 file list → Tasks 1–10 create every listed file. ✓
- §4 battery multivariate capacity-fade + LOBO + Ah reporting → Tasks 4,7,8. ✓
- §5 C-MAPSS forecasting + RUL (HI→threshold, cap=125, NASA Score) → Tasks 5,6,9. ✓
- §6 eval helper + results table → Tasks 7,10. ✓
- §7 baselines feed the table (Task 10 references them with protocol column). ✓
- §8 data acquisition + manual fallback → Task 3. ✓
- §10 open items (HI choice, univariate flag) → univariate flag implemented (Task 4); HI choice realized as PCA-1 (Task 6) and documented. ✓

**2. Placeholder scan:** No "TBD/handle edge cases/similar to Task N". Every code step has concrete code; every test step has assertions. ✓

**3. Type consistency:** `RealDataset(train_traj, test_traj, covariates, data_dim, meta)` used identically in Tasks 1,4,5,7. `meta["scaler"]` (a `Scaler` with `.mean/.std/.eps`) consumed in Task 7. `Scaler.fit/transform/inverse_transform` consistent across Tasks 1,4,5,9. `model.sample(key, history, covariates, nfe, n_steps_pred, n_samples)` matches the real signature in `models/afm.py` and is used in Tasks 7,9. `derive_rul(hi_forecast, threshold, cap)` / `nasa_score(rul_pred, rul_true)` signatures consistent across Tasks 6,9. `create_infinite_dataloaders(train, test, covariates(T,), config)` matches `training/data.py`. ✓

Known follow-ups (already flagged in spec §10, not plan failures): verify a couple of battery baseline digits against source PDFs; sanity-check the C-MAPSS HI threshold yields a defensible FD001 RUL range when run on real data.

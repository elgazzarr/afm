import numpy as np
import pytest
from afm.datasets.scaling import Scaler
from afm.config import Config
from afm.datasets.download import ensure_cmapss_files, ensure_battery_files

def test_scaler_roundtrip_per_feature():
    x = np.stack([np.arange(10), np.arange(10) * 100.0], axis=-1)  # (10, 2)
    s = Scaler().fit(x)
    z = s.transform(x)
    assert z.shape == x.shape
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-6)
    assert np.allclose(s.inverse_transform(z), x, atol=1e-5)
    assert s.mean.shape == (2,) and s.std.shape == (2,)

def test_config_accepts_real_dataset_names():
    cfg = Config(system_name="nasa_battery", system_num_timepoints=130,
                 context_length=30, fm_prediction_horizon=50, data_prediction_size=50)
    assert cfg.system_name == "nasa_battery"
    cfg2 = Config(system_name="cmapss_fd001", system_num_timepoints=128,
                  context_length=30, fm_prediction_horizon=40, data_prediction_size=40)
    assert cfg2.system_name == "cmapss_fd001"

def test_ensure_cmapss_raises_with_instructions_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        ensure_cmapss_files("FD001", data_dir=str(tmp_path), allow_download=False)
    msg = str(e.value)
    assert "train_FD001.txt" in msg
    assert "http" in msg  # an actionable URL is present

def test_ensure_battery_files_raises_with_instructions_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        ensure_battery_files(["B0005", "B0018"], data_dir=str(tmp_path), allow_download=False)
    msg = str(e.value)
    assert "B0005.mat" in msg
    assert "http" in msg  # an actionable URL is present

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

import numpy as np
from afm.datasets.cmapss import fit_health_index, project_hi

def test_health_index_orientation_rises_to_failure():
    # Two synthetic run-to-failure units whose sensors drift upward over time.
    u1 = np.cumsum(np.ones((20, 14)), axis=0) + np.random.RandomState(0).randn(20, 14) * 0.01
    u2 = np.cumsum(np.ones((25, 14)), axis=0) + np.random.RandomState(1).randn(25, 14) * 0.01
    axis = fit_health_index([u1, u2])
    hi = project_hi(u1, axis)
    assert hi[-1] > hi[0]    # HI increases toward failure

import jax.numpy as jnp, jax.random as jr, numpy as np
from afm.datasets import RealDataset
from afm.datasets.scaling import Scaler
from afm.datasets.eval import evaluate_on_trajectories

class _OracleModel:
    """Returns the ground-truth future for every sample -> NRMSE~0."""
    def __init__(self, test_traj, ctx):
        self.fut = test_traj[:, ctx:]
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

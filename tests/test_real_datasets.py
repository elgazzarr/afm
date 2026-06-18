import math

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp
import jax.random as jr
import flax.nnx as nnx

from afm.config import Config
from afm.datasets import RealDataset
from afm.datasets.battery import discharge_features
from afm.datasets.cmapss import (
    parse_cmapss,
    build_trajectories,
    SENSOR_COLS,
    nasa_score,
    derive_rul,
    fit_health_index,
    project_hi,
)
from afm.datasets.download import ensure_cmapss_files, ensure_battery_files
from afm.datasets.eval import evaluate_on_trajectories
from afm.datasets.results_table import render_results_table
from afm.datasets.run_config import build_afm_config
from afm.datasets.scaling import Scaler
from afm.models import AutoregressiveFlowMatching
from afm.training.data import create_infinite_dataloaders
from afm.training.trainers import AFMTrainer


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


def test_health_index_orientation_rises_to_failure():
    # Two synthetic run-to-failure units whose sensors drift upward over time.
    u1 = np.cumsum(np.ones((20, 14)), axis=0) + np.random.RandomState(0).randn(20, 14) * 0.01
    u2 = np.cumsum(np.ones((25, 14)), axis=0) + np.random.RandomState(1).randn(25, 14) * 0.01
    axis = fit_health_index([u1, u2])
    hi = project_hi(u1, axis)
    assert hi[-1] > hi[0]    # HI increases toward failure


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
    ds = RealDataset(z, z, np.linspace(0, 1, 10).astype(np.float32), 2,
                     {"scaler": sc, "capacity_index": 0})
    model = _OracleModel(ds.test_traj, _Cfg.context_length)
    out = evaluate_on_trajectories(model, ds, _Cfg, jr.PRNGKey(0), n_steps_pred=6, n_samples=5)
    assert out["nrmse"] < 1e-3
    assert out["rmse_orig"] < 1e-2


def test_render_results_table_latex_includes_protocol_column():
    rows = [{"method": "AFM (ours)", "dataset": "FD001", "metric": "RUL-RMSE",
             "value": "13.x", "protocol": "cap=125, HI-threshold"},
            {"method": "DCNN (Li 2018)", "dataset": "FD001", "metric": "RUL-RMSE",
             "value": "12.61", "protocol": "cap=125, w=30"}]
    out = render_results_table(rows, fmt="latex")
    assert "protocol" in out.lower()
    assert "DCNN (Li 2018)" in out and "12.61" in out
    assert out.strip().startswith("\\begin{tabular}")


def test_realworld_pipeline_end_to_end():
    """End-to-end CPU pipeline: dataloaders -> train 1 epoch -> evaluate.

    This is the TDD test for C1 (FM-window crash). It MUST fail with ValueError
    before the build_afm_config fix and pass after.
    """
    length = 40
    context_length = 10
    forecast_horizon = 30  # = length - ctx, the full eval horizon

    train = np.random.RandomState(0).randn(4, length, 2).astype("float32")
    test = np.random.RandomState(1).randn(2, length, 2).astype("float32")
    cov = np.linspace(0, 1, length).astype("float32")

    cfg = build_afm_config(
        "nasa_battery",
        length=length,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        data_batch_size=8,
        steps_per_epoch=8,
        flow_n_samples=4,
        flow_nfe=3,
    )

    # C1: must NOT raise ValueError
    (afm_train, afm_test), _ = create_infinite_dataloaders(
        jnp.asarray(train), jnp.asarray(test), jnp.asarray(cov), cfg
    )

    model = AutoregressiveFlowMatching(
        dim_x=2, dim_c=1, dim_h=16, dim_t=8, dim_hidden=16, hidden_blocks=1,
        n_rnn_layers=1, rnn_hidden=16, rngs=nnx.Rngs(0),
    )
    AFMTrainer(model, learning_rate=cfg.training_learning_rate, model_name="AFM").train(
        afm_train, afm_test, num_epochs=1, use_wandb=False
    )

    sc = Scaler().fit(train.reshape(-1, 2))
    ds = RealDataset(
        sc.transform(train).astype("float32"),
        sc.transform(test).astype("float32"),
        cov,
        2,
        {"scaler": sc, "capacity_index": 0},
    )
    metrics = evaluate_on_trajectories(
        model, ds, cfg, jr.PRNGKey(0),
        n_steps_pred=forecast_horizon, n_samples=4, nfe=3,
    )
    assert all(math.isfinite(v) for v in metrics.values()), (
        f"Non-finite metric(s): {metrics}"
    )


def test_diagnostics_coverage_horizonwise_correlation():
    from afm.datasets.diagnostics import coverage, horizonwise, correlation_distance
    rng = np.random.RandomState(0)
    S, B, H, D = 200, 4, 3, 2
    preds = rng.randn(S, B, H, D)               # samples ~ N(0,1)

    # gt at the center of the predictive dist -> always inside the interval.
    ov_in, per_h = coverage(preds, np.zeros((B, H, D)), level=0.9)
    assert ov_in > 0.8 and per_h.shape == (H,)
    # gt far outside -> essentially never covered.
    ov_out, _ = coverage(preds, np.full((B, H, D), 10.0), level=0.9)
    assert ov_out < 0.1

    crps_h, nrmse_h = horizonwise(preds, np.zeros((B, H, D)))
    assert crps_h.shape == (H,) and nrmse_h.shape == (H,)

    # preds identical to (broadcast) gt -> matching correlation structure.
    gt2 = rng.randn(B, H, D)
    d = correlation_distance(np.broadcast_to(gt2[None], (S, B, H, D)), gt2)
    assert d is not None and d < 1e-6
    # univariate -> no cross-channel correlation defined.
    assert correlation_distance(preds[..., :1], np.zeros((B, H, 1))) is None

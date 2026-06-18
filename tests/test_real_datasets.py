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

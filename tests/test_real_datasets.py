import numpy as np
from afm.datasets.scaling import Scaler
from afm.config import Config

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

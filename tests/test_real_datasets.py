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

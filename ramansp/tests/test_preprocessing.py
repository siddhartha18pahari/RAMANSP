import numpy as np

from ramansp import preprocessing as pp
from ramansp.containers import SpectralImage


def _img(H=3, W=4, K=120, slope=0.5, seed=0):
    rng = np.random.default_rng(seed)
    wn = np.linspace(1000.0, 1800.0, K)
    band = 30 * np.exp(-((wn - 1350) / 25) ** 2) + 25 * np.exp(-((wn - 1585) / 30) ** 2)
    cube = np.empty((H, W, K))
    for i in range(H):
        for j in range(W):
            cube[i, j] = 200 + slope * (wn - wn[0]) + band + rng.normal(0, 0.8, K)
    return SpectralImage(cube, wn, {})


def test_pipeline_is_deterministic():
    img = _img()
    a = pp.protocol("carbon_dg").apply(img)
    b = pp.protocol("carbon_dg").apply(img)
    np.testing.assert_allclose(a.intensities, b.intensities)


def test_arpls_removes_the_slope():
    img = _img(slope=1.0)
    out = pp.protocol("arpls").apply(img)
    # after baseline + vector norm, the band-free tail near 1750 should sit near 0
    tail = (out.wavenumber > 1740)
    assert np.abs(out.flat()[:, tail].mean()) < 0.02


def test_protocols_registered():
    for name in pp.protocol_names():
        assert isinstance(pp.protocol(name), pp.Pipeline)

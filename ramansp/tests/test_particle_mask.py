"""The particle mask must be chosen by spectroscopy, not by brightness.

Regression test for a silent failure: the original detector assumed the carbon
grain was the darker side, which is true of one specimen in this corpus and
false of others. Where it is false, every "within the particle" statistic was
computed on the substrate, and nothing raised.
"""

import numpy as np

from ramansp.analysis import detect_particle
from ramansp.containers import SpectralImage


def _map(bright_particle: bool, H=16, W=16, K=120, seed=0):
    rng = np.random.default_rng(seed)
    wn = np.linspace(1000.0, 1800.0, K)
    bands = (np.exp(-((wn - 1350) / 55) ** 2) + 1.3 * np.exp(-((wn - 1585) / 32) ** 2))
    yy, xx = np.mgrid[0:H, 0:W]
    core = (((xx - 8) ** 2 + (yy - 8) ** 2) < 25)

    cube = np.empty((H, W, K))
    # the particle always carries the bands; only its brightness changes
    hi, lo = (900.0, 300.0) if bright_particle else (300.0, 900.0)
    for i in range(H):
        for j in range(W):
            if core[i, j]:
                cube[i, j] = hi + 120 * bands + rng.normal(0, 2, K)
            else:
                cube[i, j] = lo + 0.25 * (wn - wn[0]) + rng.normal(0, 2, K)
    return SpectralImage(cube, wn, {}), core


def test_finds_the_banded_region_when_particle_is_dark():
    img, core = _map(bright_particle=False)
    det = detect_particle(img)
    got = det["mask"].reshape(core.shape)
    assert (got == core).mean() > 0.95
    assert det["band_contrast_particle"] > det["band_contrast_substrate"]


def test_finds_the_banded_region_when_particle_is_bright():
    """The case the brightness assumption got backwards."""
    img, core = _map(bright_particle=True)
    det = detect_particle(img)
    got = det["mask"].reshape(core.shape)
    assert (got == core).mean() > 0.95, "mask selected the substrate"
    assert det["band_contrast_particle"] > det["band_contrast_substrate"]

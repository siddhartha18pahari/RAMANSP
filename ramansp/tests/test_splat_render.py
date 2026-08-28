"""The truncated renderer must agree with the exact dense one.

This is the regression test for a bug that did not announce itself: the 3-sigma
box was located by assuming the wavenumber axis was uniformly spaced. It is
not -- holding out every 5th channel leaves gaps of 1,1,1,2,... -- so the box
drifted steadily off its own Gaussian along z, and the field silently lost the
contributions nearest each centre while still producing a plausible-looking
map and a decreasing loss.
"""

import numpy as np

from ramansp.splatting.model import GaussianField


def _axes(H=9, W=11, K=40, hold_out=5):
    xa = np.linspace(0.0, 1.0, W)
    ya = np.linspace(0.0, 1.0, H)
    z_all = np.linspace(0.0, 1.0, K)
    ev = np.arange(K) % hold_out == hold_out // 2
    return xa, ya, z_all[~ev], z_all[ev]


def _field(axes, n=40, seed=3):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.95, size=(n, 3))
    log_s = np.log(rng.uniform(0.02, 0.06, size=(n, 3)))
    quat = np.array([[1.0, 0, 0, 0]] * n) + rng.normal(0, 0.2, size=(n, 4))
    log_a = np.log(rng.uniform(0.2, 1.0, size=n))
    return GaussianField(mu, log_s, quat, log_a, world_axes=axes, trunc=4.0)


def _dense(f, xa, ya, z):
    shp = (ya.size, xa.size, z.size)
    cx = np.broadcast_to(xa[None, :, None], shp)
    cy = np.broadcast_to(ya[:, None, None], shp)
    cz = np.broadcast_to(z[None, None, :], shp)
    return f.render_at(np.stack(np.broadcast_arrays(cx, cy, cz), axis=-1))


def test_render_grid_matches_dense_on_a_nonuniform_axis():
    xa, ya, z_tr, z_ev = _axes()
    assert np.ptp(np.diff(z_tr)) > 0, "train axis should be non-uniformly spaced"
    f = _field((xa, ya, z_tr))
    for z in (z_tr, z_ev):
        got = f.render_grid((xa, ya, z))
        ref = _dense(f, xa, ya, z)
        rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12)
        assert rel < 1e-3, f"relative error {rel:.2e}"


def test_render_grid_matches_dense_on_a_uniform_axis():
    xa, ya, _z, _e = _axes()
    z = np.linspace(0.0, 1.0, 33)
    f = _field((xa, ya, z))
    got, ref = f.render_grid((xa, ya, z)), _dense(f, xa, ya, z)
    assert np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12) < 1e-3

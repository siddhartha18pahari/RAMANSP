import numpy as np
import pytest

from ramansp.containers import SpectralImage
from ramansp.splatting import fit_image
from ramansp.splatting.model import GaussianField
from ramansp.splatting.volume import Volume, build_volume


def _tiny_volume(H=5, W=6, K=9, seed=0):
    rng = np.random.default_rng(seed)
    x_axis = np.linspace(0.0, 1.0, W)
    y_axis = np.linspace(0.0, 1.0, H)
    z_axis = np.linspace(0.0, 1.0, K)
    cx = np.broadcast_to(x_axis[None, :, None], (H, W, K))
    cy = np.broadcast_to(y_axis[:, None, None], (H, W, K))
    cz = np.broadcast_to(z_axis[None, None, :], (H, W, K))
    coords = np.stack([cx, cy, cz], axis=-1)
    V = 0.3 * rng.standard_normal((H, W, K)) + np.exp(
        -(((cx - 0.5) / 0.3) ** 2 + ((cy - 0.5) / 0.3) ** 2 + ((cz - 0.5) / 0.3) ** 2)
    )
    mask = np.ones((H, W, K), bool)
    vol = Volume(
        V=V, coords=coords, mask3d=mask, wn_train=1000 + 800 * z_axis, z_train=z_axis,
        V_eval=np.zeros((H, W, 0)), coords_eval=np.zeros((H, W, 0, 3)),
        mask_eval=np.zeros((H, W, 0), bool), wn_eval=np.zeros(0),
        wn0=1000.0, wn1=1800.0, spatial_mask=np.ones((H, W), bool),
        bbox=(0, H, 0, W), scale=1.0,
    )
    return vol, (x_axis, y_axis, z_axis)


def _random_field(axes, n=3, seed=1):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.3, 0.7, size=(n, 3))
    log_s = np.log(rng.uniform(0.12, 0.22, size=(n, 3)))
    quat = np.array([[1.0, 0, 0, 0]] * n) + rng.normal(0, 0.15, size=(n, 4))
    log_a = np.log(rng.uniform(0.4, 0.9, size=n))
    return GaussianField(mu, log_s, quat, log_a, world_axes=axes,
                         lam_a=1e-3, lam_s=1e-3, trunc=6.0)


def test_analytic_gradient_matches_finite_difference():
    vol, axes = _tiny_volume()
    field = _random_field(axes)
    _loss, g, _aux = field.loss_and_grad(vol)

    eps = 1e-6
    for name, ana in [("mu", g.mu), ("log_s", g.log_s), ("quat", g.quat), ("log_a", g.log_a)]:
        arr = getattr(field, name)
        num = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            arr[idx] += eps
            lp = field.loss_and_grad(vol)[0]
            arr[idx] -= 2 * eps
            lm = field.loss_and_grad(vol)[0]
            arr[idx] += eps
            num[idx] = (lp - lm) / (2 * eps)
        err = np.max(np.abs(ana - num)) / (1 + np.max(np.abs(num)))
        assert err < 1e-4, f"{name}: rel err {err:.2e}\nana={ana}\nnum={num}"


def test_fit_improves_reconstruction():
    rng = np.random.default_rng(0)
    H, W, K = 14, 16, 90
    wn = np.linspace(1000.0, 1800.0, K)
    base = (18 * np.exp(-((wn - 1350) / 45) ** 2) + 22 * np.exp(-((wn - 1585) / 30) ** 2))
    cube = np.zeros((H, W, K))
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.exp(-(((xx - 8) / 4.0) ** 2 + ((yy - 7) / 4.0) ** 2))
    for i in range(H):
        for j in range(W):
            cube[i, j] = blob[i, j] * base + rng.normal(0, 0.4, K)
    img = SpectralImage(cube, wn, {})

    field, vol, hist = fit_image(img, n_gaussians=250, iters=60, use_particle_mask=False,
                                 hold_out=6, densify_every=1000, log_every=20)
    assert hist.train_psnr[-1] > hist.train_psnr[0] + 3.0
    assert field.render_band(1350.0, vol).shape == vol.V.shape[:2]


def test_save_load_roundtrip(tmp_path):
    vol, axes = _tiny_volume()
    field = _random_field(axes)
    p = tmp_path / "f.npz"
    field.save(str(p))
    back = GaussianField.load(str(p))
    np.testing.assert_allclose(field.mu, back.mu)
    np.testing.assert_allclose(
        field.render_at(vol.coords.reshape(-1, 3)),
        back.render_at(vol.coords.reshape(-1, 3)),
    )

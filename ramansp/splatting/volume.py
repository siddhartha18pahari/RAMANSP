"""Turn a :class:`~ramansp.containers.SpectralImage` into a fitting target.

The target is a dense cube ``V[i, j, k]`` on a unit-cube coordinate frame:

* spatial axes ``x = j/(W-1)``, ``y = i/(H-1)`` in ``[0, 1]``
* wavenumber axis ``z = (w_k - w_0) / (w_{K-1} - w_0)`` in ``[0, 1]``

Channels are split into a *train* set and a held-out *eval* set (every
``hold_out``-th channel) so reconstruction quality is scored where the field
was never fitted. An optional spatial ``mask`` (e.g. a particle mask) restricts
both the loss and the reported metrics, and the volume is cropped to the mask's
bounding box to keep the fit sparse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..containers import SpectralImage


@dataclass
class Volume:
    V: np.ndarray                 # (H, W, Ktrain) target, NaN where masked out
    coords: np.ndarray            # (H, W, Ktrain, 3) unit-cube positions (x, y, z)
    mask3d: np.ndarray            # (H, W, Ktrain) bool -- voxels that count
    wn_train: np.ndarray          # (Ktrain,) wavenumbers of the fitted channels
    z_train: np.ndarray           # (Ktrain,) normalised wavenumbers
    V_eval: np.ndarray            # (H, W, Keval) held-out channels
    coords_eval: np.ndarray       # (H, W, Keval, 3)
    mask_eval: np.ndarray
    wn_eval: np.ndarray
    wn0: float
    wn1: float
    spatial_mask: np.ndarray      # (H, W) bool
    bbox: tuple[int, int, int, int]   # i0, i1, j0, j1 into the ORIGINAL grid
    scale: float                  # data amplitude used to normalise V
    orig_K: int = 0               # channels in the map before any binning

    @property
    def shape(self):
        return self.V.shape

    def z_of(self, wavenumber: float) -> float:
        return float((wavenumber - self.wn0) / (self.wn1 - self.wn0))

    def z_eval_axis(self) -> np.ndarray:
        """Normalised wavenumbers of the held-out channels."""
        if self.wn_eval.size == 0:
            return np.zeros(0)
        return (self.wn_eval - self.wn0) / (self.wn1 - self.wn0)


def _grid_coords(H, W, z):
    """(H, W, len(z), 3) unit-cube positions with axis order (x, y, z)."""
    x = np.linspace(0.0, 1.0, W) if W > 1 else np.zeros(1)
    y = np.linspace(0.0, 1.0, H) if H > 1 else np.zeros(1)
    z = np.asarray(z, float)
    cx = np.broadcast_to(x[None, :, None], (H, W, z.size))
    cy = np.broadcast_to(y[:, None, None], (H, W, z.size))
    cz = np.broadcast_to(z[None, None, :], (H, W, z.size))
    return np.stack([cx, cy, cz], axis=-1).astype(np.float64)


def _bin_channels(cube: np.ndarray, wn: np.ndarray, factor: int):
    if factor <= 1:
        return cube, wn
    K = wn.size
    keep = (K // factor) * factor
    cube = cube[..., :keep].reshape(*cube.shape[:-1], keep // factor, factor).mean(-1)
    wn = wn[:keep].reshape(keep // factor, factor).mean(-1)
    return cube, wn


def build_volume(
    image: SpectralImage,
    spatial_mask: np.ndarray | None = None,
    hold_out: int = 5,
    pad: int = 1,
    channel_bin: int = 1,
) -> Volume:
    if not image.is_gridded:
        raise ValueError("build_volume needs a gridded SpectralImage (H, W, K)")
    cube = image.intensities.astype(np.float64)
    wn = image.wavenumber.astype(np.float64)
    orig_K = wn.size
    cube, wn = _bin_channels(cube, wn, channel_bin)
    H, W, K = cube.shape

    if spatial_mask is None:
        spatial_mask = np.ones((H, W), bool)
    spatial_mask = np.asarray(spatial_mask, bool).reshape(H, W)
    if not spatial_mask.any():
        raise ValueError("spatial_mask is empty")

    ii, jj = np.where(spatial_mask)
    i0, i1 = max(ii.min() - pad, 0), min(ii.max() + 1 + pad, H)
    j0, j1 = max(jj.min() - pad, 0), min(jj.max() + 1 + pad, W)
    cube = cube[i0:i1, j0:j1]
    sm = spatial_mask[i0:i1, j0:j1]
    Hc, Wc = cube.shape[:2]

    # hold_out <= 1 means "withhold nothing": fit on every channel. Guard the
    # modulo explicitly rather than relying on short-circuiting, which does not
    # apply to numpy's elementwise & and would divide by zero.
    if hold_out and hold_out > 1:
        eval_idx = np.arange(K)[np.arange(K) % hold_out == hold_out // 2]
    else:
        eval_idx = np.zeros(0, dtype=int)
    train_idx = np.setdiff1d(np.arange(K), eval_idx)

    wn0, wn1 = float(wn[0]), float(wn[-1])
    span = wn1 - wn0 or 1.0
    z_all = (wn - wn0) / span

    scale = float(np.nanpercentile(np.abs(cube[sm]), 99.5)) or 1.0

    def pack(idx):
        Vc = cube[:, :, idx] / scale
        z = z_all[idx]
        coords = _grid_coords(Hc, Wc, z)
        m3 = np.broadcast_to(sm[:, :, None], Vc.shape).copy()
        Vc = np.where(m3, Vc, np.nan)
        return Vc, coords, m3, wn[idx], z

    Vtr, ctr, mtr, wntr, ztr = pack(train_idx)
    Vev, cev, mev, wnev, _ = pack(eval_idx) if eval_idx.size else (
        np.zeros((Hc, Wc, 0)), np.zeros((Hc, Wc, 0, 3)), np.zeros((Hc, Wc, 0), bool),
        np.zeros(0), np.zeros(0),
    )

    return Volume(
        V=Vtr, coords=ctr, mask3d=mtr, wn_train=wntr, z_train=ztr,
        V_eval=Vev, coords_eval=cev, mask_eval=mev, wn_eval=wnev,
        wn0=wn0, wn1=wn1, spatial_mask=sm, bbox=(i0, i1, j0, j1), scale=scale,
        orig_K=orig_K,
    )

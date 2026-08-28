"""Reconstruction and spatial-structure metrics.

Reconstruction (used by the splatting fit):
    ``psnr``   peak signal-to-noise ratio, dB
    ``ssim``   global structural similarity (Wang et al. 2004), no windowing
    ``sam``    spectral angle mapper, mean radians over spatial positions

Spatial structure (adapted from the prior project's ``raman_hyperspectral`` /
``raman_spectra`` -- kept so the corpus statistics stay honest about the fact
that adjacent map points are not independent):
    ``morans_i`` ``variogram`` ``effective_n`` ``noise_by_interleave``
    ``variance_partition``
"""

from __future__ import annotations

import numpy as np

# --- reconstruction ------------------------------------------------
def psnr(ref: np.ndarray, test: np.ndarray, data_range: float | None = None) -> float:
    ref = np.asarray(ref, float)
    test = np.asarray(test, float)
    m = np.isfinite(ref) & np.isfinite(test)
    if not m.any():
        return float("nan")
    mse = float(np.mean((ref[m] - test[m]) ** 2))
    if mse == 0:
        return float("inf")
    rng = float(ref[m].max() - ref[m].min()) if data_range is None else data_range
    return float(10.0 * np.log10(rng ** 2 / mse))


def ssim(ref: np.ndarray, test: np.ndarray, data_range: float | None = None) -> float:
    ref = np.asarray(ref, float).ravel()
    test = np.asarray(test, float).ravel()
    m = np.isfinite(ref) & np.isfinite(test)
    ref, test = ref[m], test[m]
    if ref.size < 2:
        return float("nan")
    rng = float(ref.max() - ref.min()) if data_range is None else data_range
    c1, c2 = (0.01 * rng) ** 2, (0.03 * rng) ** 2
    mu_x, mu_y = ref.mean(), test.mean()
    vx, vy = ref.var(), test.var()
    cov = float(np.mean((ref - mu_x) * (test - mu_y)))
    return float(
        ((2 * mu_x * mu_y + c1) * (2 * cov + c2))
        / ((mu_x ** 2 + mu_y ** 2 + c1) * (vx + vy + c2))
    )


def sam(ref_cube: np.ndarray, test_cube: np.ndarray) -> float:
    """Mean spectral angle (radians) between matched spectra of two cubes."""
    a = ref_cube.reshape(-1, ref_cube.shape[-1])
    b = test_cube.reshape(-1, test_cube.shape[-1])
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-12) & (nb > 1e-12)
    cos = np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])
    return float(np.mean(np.arccos(np.clip(cos, -1.0, 1.0))))


# --- spatial structure -------------------------------------------
def morans_i(image: np.ndarray) -> float:
    z = image - np.nanmean(image)
    z = np.nan_to_num(z)
    num = (z[:-1, :] * z[1:, :]).sum() + (z[:, :-1] * z[:, 1:]).sum()
    n_edges = (image.shape[0] - 1) * image.shape[1] + image.shape[0] * (image.shape[1] - 1)
    denom = (z ** 2).sum()
    if denom == 0 or n_edges == 0:
        return float("nan")
    return float(image.size * num / (n_edges * denom))


def variogram(image: np.ndarray, mask: np.ndarray | None = None, max_lag: int = 12):
    valid = np.isfinite(image) if mask is None else (mask & np.isfinite(image))
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    y, x, v = yy[valid], xx[valid], image[valid]
    d = np.sqrt((y[:, None] - y) ** 2 + (x[:, None] - x) ** 2)
    g = 0.5 * (v[:, None] - v) ** 2
    iu = np.triu_indices(len(v), 1)
    d, g = d[iu], g[iu]
    lag, gamma, npair = [], [], []
    for lo in range(max_lag):
        k = (d >= lo) & (d < lo + 1)
        if k.sum() > 20:
            lag.append(lo + 0.5)
            gamma.append(float(g[k].mean()))
            npair.append(int(k.sum()))
    return np.array(lag), np.array(gamma), np.array(npair), float(v.var())


def effective_n(image: np.ndarray, mask: np.ndarray) -> dict:
    """AR(1) effective sample size after correcting for spatial autocorrelation.

    ``n_eff = n * ((1-r)/(1+r))**2`` with ``r`` the mean rook-neighbour
    correlation *within* the mask. An order-of-magnitude correction.
    """
    valid = mask & np.isfinite(image)
    rs = []
    for a, b, ma, mb in [
        (image[:-1, :], image[1:, :], valid[:-1, :], valid[1:, :]),
        (image[:, :-1], image[:, 1:], valid[:, :-1], valid[:, 1:]),
    ]:
        k = ma & mb
        if k.sum() > 5:
            rs.append(np.corrcoef(a[k], b[k])[0, 1])
    r = float(np.mean(rs)) if rs else 0.0
    n = int(valid.sum())
    return {"r": r, "n": n, "n_eff": float(n * ((1 - r) / (1 + r)) ** 2)}


def noise_by_interleave(flat: np.ndarray, wavenumber: np.ndarray,
                        window: tuple[float, float]) -> dict:
    """Noise on an integrated area, model-free, by even/odd channel interleave.

    The even- and odd-indexed channels each integrate to the same area; real
    structure cancels in their difference, leaving measurement noise. Adapted
    from ``raman_spectra.noise_by_interleave`` (linear local baseline).
    """
    x = wavenumber
    idx = np.where((x >= window[0]) & (x <= window[1]))[0]
    halves = []
    for sub in (idx[0::2], idx[1::2]):
        w = x[sub]
        A = np.vstack([w - w.mean(), np.ones_like(w)]).T
        coef, *_ = np.linalg.lstsq(A, flat[:, sub].T, rcond=None)
        base = (A @ coef).T
        halves.append(np.trapezoid(flat[:, sub] - base, w, axis=1))
    diff = halves[0] - halves[1]
    return {"diff": diff, "sd": float(np.std(diff) / 2)}


def variance_partition(image: np.ndarray, mask: np.ndarray, noise_sd: float) -> dict:
    """Split a map's variance into noise / unresolved / resolved structure."""
    valid = mask & np.isfinite(image)
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    y, x, v = yy[valid], xx[valid], image[valid]
    d = np.sqrt((y[:, None] - y) ** 2 + (x[:, None] - x) ** 2)
    g = 0.5 * (v[:, None] - v) ** 2
    iu = np.triu_indices(len(v), 1)
    d, g = d[iu], g[iu]
    total = float(v.var())
    near = (d >= 1) & (d < 2)
    nugget = float(g[near].mean()) if near.any() else float("nan")
    noise = float(noise_sd ** 2)
    return {
        "total": total, "nugget": nugget, "noise": noise,
        "unresolved": max(0.0, nugget - noise) if np.isfinite(nugget) else float("nan"),
        "resolved": max(0.0, total - nugget) if np.isfinite(nugget) else float("nan"),
        "n": int(valid.sum()),
    }

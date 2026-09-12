"""Analysis layer: decomposition, clustering, unmixing, carbon band metrics.

Decomposition / clustering wrap scikit-learn. Unmixing implements VCA
(Nascimento & Dias 2005) + fully-constrained least-squares abundances. The
carbon band maths (local anchored baselines, Otsu particle detection, the
bounded disorder index) is adapted from the prior project's ``raman_spectra``
and ``raman_hyperspectral``.
"""

from __future__ import annotations

import numpy as np

from .containers import SpectralImage

# band -> (integration window, [anchor windows for the local baseline])
CARBON_BANDS = {
    "D1": ((1300.0, 1400.0), [(1200.0, 1270.0), (1420.0, 1460.0), (1700.0, 1740.0)]),
    "G+D2": ((1540.0, 1660.0), [(1420.0, 1480.0), (1700.0, 1760.0)]),
}
_CONTROL = ((1710.0, 1770.0), [(1680.0, 1705.0), (1775.0, 1798.0)])


# --- decomposition / clustering ------------------------------------
def decompose(image: SpectralImage, method: str = "pca", n_components: int = 4, seed: int = 0):
    """Return ``(scores (N, C), components (C, K), model)``.

    ``method`` in {``pca``, ``nmf``, ``ica``}.
    """
    from sklearn.decomposition import PCA, NMF, FastICA

    X = image.flat()
    if method == "pca":
        model = PCA(n_components=n_components, random_state=seed).fit(X)
        return model.transform(X), model.components_, model
    if method == "nmf":
        Xn = np.clip(X - X.min(), 0, None)
        model = NMF(n_components=n_components, init="nndsvda", random_state=seed,
                    max_iter=600).fit(Xn)
        return model.transform(Xn), model.components_, model
    if method == "ica":
        model = FastICA(n_components=n_components, random_state=seed, max_iter=600).fit(X)
        return model.transform(X), model.components_, model
    raise ValueError(f"unknown method {method!r}")


def cluster(image: SpectralImage, k: int | None = None, k_range=(2, 8),
            features: np.ndarray | None = None, seed: int = 0) -> dict:
    """PCA-whitened k-means over spectra (or over ``features`` if given).

    ``k=None`` selects k by silhouette over ``k_range``. Adapted from
    ``raman_hyperspectral.segment``.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    X = image.flat() if features is None else np.asarray(features, float)
    Xs = StandardScaler().fit_transform(X)
    Xs = PCA(n_components=min(10, Xs.shape[1]), random_state=seed).fit_transform(Xs)

    if k is None:
        best, k = -np.inf, k_range[0]
        for kk in range(*k_range):
            lab = KMeans(kk, n_init=10, random_state=seed).fit_predict(Xs)
            s = silhouette_score(Xs, lab)
            if s > best:
                best, k = s, kk
    labels = KMeans(k, n_init=10, random_state=seed).fit_predict(Xs)
    means = np.vstack([image.flat()[labels == c].mean(0) for c in range(k)])
    return {"labels": labels, "k": int(k), "cluster_mean_spectra": means}


# --- linear unmixing (VCA + FCLS) --------------------------------
def vca(X: np.ndarray, n_endmembers: int, seed: int = 0) -> np.ndarray:
    """Vertex Component Analysis (Nascimento & Dias 2005), compact variant.

    ``X`` is (N, K); returns the ``n_endmembers`` spectra (rows of ``X``) that
    sit at the vertices of the data simplex, shape (n_endmembers, K).
    """
    rng = np.random.default_rng(seed)
    R = int(n_endmembers)
    Y = np.asarray(X, float).T                       # (K, N)
    L, N = Y.shape
    Ud = np.linalg.svd(Y @ Y.T / N, hermitian=True)[0][:, :R]   # (L, R)
    x = Ud.T @ Y                                     # (R, N)
    u = x.mean(axis=1)                               # (R,)
    denom = u @ x                                    # (N,)
    guard = (np.max(np.abs(denom)) or 1.0) * 1e-6
    y = x / (denom + np.sign(denom + 1e-30) * guard)
    A = np.zeros((R, R))
    A[-1, 0] = 1.0
    idx = np.zeros(R, dtype=int)
    for i in range(R):
        w = rng.random(R)
        f = w - A @ np.linalg.pinv(A) @ w
        f = f / (np.linalg.norm(f) + 1e-12)
        v = f @ y                                    # (N,)
        idx[i] = int(np.argmax(np.abs(v)))
        A[:, i] = y[:, idx[i]]
    return np.asarray(X, float)[idx]


def fcls(X: np.ndarray, endmembers: np.ndarray, n_iter: int = 200) -> np.ndarray:
    """Fully-constrained (non-negative, sum-to-one) least-squares abundances.

    Projected-gradient; ``X`` (N, K), ``endmembers`` (C, K) -> abundances (N, C).
    """
    E = endmembers.T                            # (K, C)
    C = E.shape[1]
    G = E.T @ E
    step = 1.0 / (np.linalg.norm(G, 2) + 1e-9)
    A = np.full((X.shape[0], C), 1.0 / C)
    ET_X = X @ E
    for _ in range(n_iter):
        A = A - step * (A @ G - ET_X)
        A = np.clip(A, 0, None)
        A = A / np.clip(A.sum(1, keepdims=True), 1e-12, None)
    return A


def unmix(image: SpectralImage, n_endmembers: int = 4, seed: int = 0) -> dict:
    X = image.flat()
    E = vca(X, n_endmembers, seed=seed)
    A = fcls(X, E)
    return {"endmembers": E, "abundances": A, "wavenumber": image.wavenumber.copy()}


# --- carbon band metrics --------------------------------------------
def _local_baseline_area(flat, wn, window, anchors, order=2):
    mask = np.zeros(len(wn), bool)
    for lo, hi in anchors:
        mask |= (wn >= lo) & (wn <= hi)
    if mask.sum() <= order:
        return np.full(flat.shape[0], np.nan)
    xa = wn[mask]
    c, s = xa.mean(), xa.std() or 1.0
    A = np.vander((xa - c) / s, order + 1)
    coef, *_ = np.linalg.lstsq(A, flat[:, mask].T, rcond=None)
    idx = np.where((wn >= window[0]) & (wn <= window[1]))[0]
    w = wn[idx]
    base = (np.vander((w - c) / s, order + 1) @ coef).T
    return np.trapezoid(flat[:, idx] - base, w, axis=1)


def band_areas(image: SpectralImage, bands: dict | None = None, order: int = 2) -> dict:
    flat, wn = image.flat(), image.wavenumber
    bands = bands or CARBON_BANDS
    return {name: _local_baseline_area(flat, wn, win, anc, order)
            for name, (win, anc) in bands.items()}


def control_window_area(image: SpectralImage, order: int = 2) -> np.ndarray:
    """Band-free window integrated exactly like a band; ~0 if the baseline is
    trustworthy. The acceptance test carried over from the prior project.
    """
    win, anc = _CONTROL
    return _local_baseline_area(image.flat(), image.wavenumber, win, anc, order)


def disorder_index(image: SpectralImage) -> np.ndarray:
    """``A_D1 / (A_D1 + A_{G+D2})`` -- bounded in [0, 1], monotonic in D/G,
    finite at the particle edge. Report only with the preprocessing protocol.
    """
    a = band_areas(image)
    tot = a["D1"] + a["G+D2"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, a["D1"] / tot, np.nan)


def _otsu(values, n_bins=128):
    hist, edges = np.histogram(values, bins=n_bins)
    p = hist / max(hist.sum(), 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    cum, mu = np.cumsum(p), np.cumsum(p * centres)
    between = (mu[-1] * cum - mu) ** 2 / np.maximum(cum * (1 - cum), 1e-12)
    return float(centres[int(np.argmax(between))])


def band_contrast(image: SpectralImage,
                  band: tuple[float, float] = (1540.0, 1660.0),
                  free: tuple[float, float] = (1700.0, 1790.0)) -> np.ndarray:
    """Per-spectrum height of the carbon bands above a band-free window.

    A shape measure, not a brightness one, so it survives normalisation and
    says whether a spectrum actually carries Raman bands.
    """
    wn, flat = image.wavenumber, image.flat()
    g = (wn >= band[0]) & (wn <= band[1])
    f = (wn >= free[0]) & (wn <= free[1])
    if g.sum() < 2:
        return np.zeros(flat.shape[0])
    base = flat[:, f].mean(1) if f.sum() >= 2 else flat.min(1)
    return flat[:, g].mean(1) - base


def detect_particle(image: SpectralImage) -> dict:
    """Split particle from substrate, then orient the split spectroscopically.

    The intensity split is an Otsu threshold on log mean counts, which needs no
    noise estimate and so avoids using the noise to define the mask that
    defines the noise.

    Which side is the particle is *not* assumed. The original specimen studied
    on this corpus was a dark carbon grain on a bright fluorescent substrate,
    and hard-coding that ("the particle is the darker side") silently selects
    the substrate on any map where the contrast runs the other way, which does
    happen here. The side carrying the carbon bands is chosen instead, by
    comparing band contrast across the split, which is a spectroscopic
    criterion rather than a photometric one.
    """
    total = image.flat().mean(1)
    pos = total > 0
    thr = _otsu(np.log10(total[pos])) if pos.any() else 0.0
    dark = pos & (np.log10(np.clip(total, 1e-9, None)) < thr)

    contrast = band_contrast(image)
    inverted = False
    if dark.any() and (~dark).any():
        if np.median(contrast[~dark]) > np.median(contrast[dark]):
            dark = ~dark
            inverted = True
    mask = dark
    return {
        "mask": mask,
        "threshold_counts": float(10 ** thr),
        "frac_particle": float(mask.mean()),
        "orientation_flipped": bool(inverted),
        "band_contrast_particle": float(np.median(contrast[mask])) if mask.any() else float("nan"),
        "band_contrast_substrate": float(np.median(contrast[~mask])) if (~mask).any() else float("nan"),
        "median_counts_particle": float(np.median(total[mask])) if mask.any() else float("nan"),
        "median_counts_substrate": float(np.median(total[~mask])) if (~mask).any() else float("nan"),
    }

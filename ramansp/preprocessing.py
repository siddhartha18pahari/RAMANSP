"""Composable preprocessing, RamanSPy-style.

A :class:`Pipeline` is an ordered list of steps; each step maps a
:class:`~ramansp.containers.SpectralImage` to another of the same shape. Named
:func:`protocol` presets bundle the common recipes. The baseline maths is
adapted from the prior project's ``raman_spectra`` (``baseline_als``,
``baseline_arpls``, ``despike``); the point of exposing several is that the
prior work showed the carbon D/G ratio is *preprocessing-dependent*, so every
downstream number must travel with the protocol that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .containers import SpectralImage


# --- individual steps --------------------------------------------------
class Step:
    def __call__(self, image: SpectralImage) -> SpectralImage:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        kv = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({kv})"


def _map_flat(image: SpectralImage, fn) -> SpectralImage:
    flat = image.flat()
    out = fn(flat)
    return SpectralImage(out.reshape(image.intensities.shape), image.wavenumber.copy(),
                         dict(image.metadata))


@dataclass(repr=False)
class Crop(Step):
    lo: float = 1000.0
    hi: float = 1800.0

    def __call__(self, image: SpectralImage) -> SpectralImage:
        return image.spectral_slice(self.lo, self.hi)


@dataclass(repr=False)
class Despike(Step):
    threshold: float = 12.0

    def __call__(self, image: SpectralImage) -> SpectralImage:
        def fn(flat):
            out = flat.copy()
            d2 = np.abs(np.diff(flat, 2, axis=1))
            sigma = np.median(d2) / 0.6745 or 1.0
            hit = np.zeros(flat.shape, bool)
            hit[:, 1:-1] = d2 > self.threshold * sigma
            r, c = np.nonzero(hit)
            out[r, c] = 0.5 * (flat[r, c - 1] + flat[r, c + 1])
            return out
        return _map_flat(image, fn)


_DTD_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _dtd_bands(n: int):
    """Upper bands (main, +1, +2) of D^T D for the second-difference operator.

    ``diag(w) + lam * D^T D`` is symmetric pentadiagonal, so the arPLS system
    can be solved with a banded Cholesky in O(n) instead of a general sparse
    factorisation. Same matrix, same answer, far less work per spectrum.
    """
    if n not in _DTD_CACHE:
        import scipy.sparse as sp

        d = sp.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
        m = (d.T @ d).toarray()
        _DTD_CACHE[n] = (np.diag(m).copy(),
                         np.diag(m, 1).copy(),
                         np.diag(m, 2).copy())
    return _DTD_CACHE[n]


def _arpls(y: np.ndarray, lam: float, n_iter: int, tol: float) -> np.ndarray:
    from scipy.linalg import solveh_banded

    n = y.size
    if n < 5:
        return np.full(n, float(np.min(y)))
    b0, b1, b2 = _dtd_bands(n)
    ab = np.zeros((3, n))
    ab[0, 2:] = lam * b2
    ab[1, 1:] = lam * b1
    w = np.ones(n)
    z = y.copy()
    for _ in range(n_iter):
        ab[2] = w + lam * b0
        z = solveh_banded(ab, w * y, lower=False, check_finite=False)
        d = y - z
        neg = d[d < 0]
        if neg.size < 2:
            break
        mu, sigma = neg.mean(), neg.std()
        wn = 1.0 / (1.0 + np.exp(np.clip(2 * (d - (2 * sigma - mu)) / max(sigma, 1e-9), -50, 50)))
        if np.linalg.norm(wn - w) / max(np.linalg.norm(w), 1e-9) < tol:
            w = wn
            break
        w = wn
    return z


def _als(y: np.ndarray, lam: float, p: float, n_iter: int) -> np.ndarray:
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    n = y.size
    diff = sp.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    stiff = lam * (diff.T @ diff)
    w = np.ones(n)
    z = y
    for _ in range(n_iter):
        z = spsolve((sp.diags(w) + stiff).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


@dataclass(repr=False)
class BaselineARPLS(Step):
    """Asymmetrically reweighted penalised least squares (Baek et al. 2015)."""

    lam: float = 1e5
    n_iter: int = 20
    tol: float = 1e-3

    def __call__(self, image: SpectralImage) -> SpectralImage:
        def fn(flat):
            return np.vstack([row - _arpls(row, self.lam, self.n_iter, self.tol) for row in flat])
        return _map_flat(image, fn)


@dataclass(repr=False)
class BaselineALS(Step):
    """Asymmetric least squares (Eilers & Boelens). Fixed asymmetry ``p``."""

    lam: float = 1e5
    p: float = 0.01
    n_iter: int = 10

    def __call__(self, image: SpectralImage) -> SpectralImage:
        def fn(flat):
            return np.vstack([row - _als(row, self.lam, self.p, self.n_iter) for row in flat])
        return _map_flat(image, fn)


@dataclass(repr=False)
class BaselinePoly(Step):
    """Iterative low-order polynomial (ModPoly): fit, clip signal above the
    fit, refit, until the baseline stops moving. A "straight-ish" baseline in
    the sense the prior work used to bracket the D/G ratio.
    """

    order: int = 3
    n_iter: int = 24
    tol: float = 1e-3

    def __call__(self, image: SpectralImage) -> SpectralImage:
        x = image.wavenumber
        xs = (x - x.mean()) / (x.std() or 1.0)
        V = np.vander(xs, self.order + 1)

        def one(y):
            work = y.copy()
            prev = None
            for _ in range(self.n_iter):
                coef, *_ = np.linalg.lstsq(V, work, rcond=None)
                base = V @ coef
                work = np.minimum(work, base)
                if prev is not None and np.linalg.norm(base - prev) / (np.linalg.norm(prev) + 1e-9) < self.tol:
                    break
                prev = base
            return y - base

        return _map_flat(image, lambda flat: np.vstack([one(r) for r in flat]))


@dataclass(repr=False)
class SavGol(Step):
    window: int = 7
    polyorder: int = 3

    def __call__(self, image: SpectralImage) -> SpectralImage:
        from scipy.signal import savgol_filter

        w = min(self.window, image.wavenumber.size - (1 - image.wavenumber.size % 2))
        w = max(w if w % 2 else w - 1, self.polyorder + 2 | 1)
        return _map_flat(image, lambda flat: savgol_filter(flat, w, self.polyorder, axis=1))


@dataclass(repr=False)
class Normalize(Step):
    method: str = "area"          # area | vector | minmax | maxband
    band: tuple[float, float] = (1550.0, 1620.0)

    def __call__(self, image: SpectralImage) -> SpectralImage:
        x = image.wavenumber

        def fn(flat):
            if self.method == "vector":
                s = np.linalg.norm(flat, axis=1, keepdims=True)
            elif self.method == "area":
                s = np.trapezoid(np.clip(flat, 0, None), x, axis=1)[:, None]
            elif self.method == "minmax":
                lo = flat.min(1, keepdims=True)
                return (flat - lo) / np.clip(flat.max(1, keepdims=True) - lo, 1e-12, None)
            elif self.method == "maxband":
                k = (x >= self.band[0]) & (x <= self.band[1])
                s = flat[:, k].max(1, keepdims=True)
            else:
                raise ValueError(f"unknown normalise method {self.method!r}")
            return flat / np.clip(s, 1e-12, None)

        return _map_flat(image, fn)


# --- pipeline --------------------------------------------------------
class Pipeline:
    def __init__(self, *steps: Step, name: str = "custom"):
        self.steps = list(steps)
        self.name = name

    def apply(self, image: SpectralImage) -> SpectralImage:
        out = image
        for step in self.steps:
            out = step(out)
        out.metadata["preprocessing"] = repr(self)
        return out

    __call__ = apply

    def __repr__(self) -> str:
        return f"Pipeline[{self.name}](" + " -> ".join(repr(s) for s in self.steps) + ")"


_PROTOCOLS = {
    "minimal": lambda: Pipeline(Crop(), Despike(), name="minimal"),
    "carbon_dg": lambda: Pipeline(
        Crop(1000, 1800), Despike(), BaselineARPLS(lam=1e5), SavGol(7, 3),
        Normalize("area"), name="carbon_dg",
    ),
    "arpls": lambda: Pipeline(
        Crop(1000, 1800), Despike(), BaselineARPLS(lam=1e5), Normalize("vector"),
        name="arpls",
    ),
    "chord": lambda: Pipeline(
        Crop(1000, 1800), Despike(), BaselinePoly(order=1), Normalize("area"),
        name="chord",
    ),
    "poly3": lambda: Pipeline(
        Crop(1000, 1800), Despike(), BaselinePoly(order=3), Normalize("area"),
        name="poly3",
    ),
}


def protocol(name: str) -> Pipeline:
    """Return a fresh preset pipeline. Names: ``minimal``, ``carbon_dg``,
    ``arpls``, ``chord``, ``poly3``.
    """
    try:
        return _PROTOCOLS[name]()
    except KeyError:
        raise KeyError(f"unknown protocol {name!r}; have {sorted(_PROTOCOLS)}") from None


def protocol_names() -> list[str]:
    return sorted(_PROTOCOLS)

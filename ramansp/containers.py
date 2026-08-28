"""Typed spectral containers.

Mirrors RamanSPy's ``Spectrum`` / ``SpectralImage`` / ``SpectralVolume`` split.
Every container is a thin wrapper over a contiguous intensity array plus a
shared 1-D wavenumber axis and a free-form ``metadata`` dict. The last array
axis is always the spectral axis.

    Spectrum        intensities (K,)                      one measurement
    SpectralImage   intensities (H, W, K) or (N, K)       a map / a bag of spectra
    SpectralVolume  intensities (D, H, W, K)              a z-stack / true volume

Spatial coordinates, when known, live in ``metadata['x']`` / ``metadata['y']``
(1-D arrays, one entry per spectrum in flattened row-major order).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _as_axis(wavenumber) -> np.ndarray:
    wn = np.asarray(wavenumber, dtype=float).ravel()
    if wn.ndim != 1 or wn.size < 2:
        raise ValueError("wavenumber axis must be 1-D with >= 2 points")
    if not (np.all(np.diff(wn) > 0) or np.all(np.diff(wn) < 0)):
        raise ValueError("wavenumber axis must be strictly monotonic")
    return wn


@dataclass
class _Base:
    intensities: np.ndarray
    wavenumber: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.intensities = np.asarray(self.intensities, dtype=float)
        self.wavenumber = _as_axis(self.wavenumber)
        if self.intensities.shape[-1] != self.wavenumber.size:
            raise ValueError(
                f"last axis of intensities ({self.intensities.shape[-1]}) "
                f"!= wavenumber axis ({self.wavenumber.size})"
            )
        # Work internally with an ascending axis; remember if we flipped.
        if self.wavenumber[0] > self.wavenumber[-1]:
            self.wavenumber = self.wavenumber[::-1].copy()
            self.intensities = self.intensities[..., ::-1].copy()
            self.metadata.setdefault("axis_reversed_on_load", True)

    # -- shared views ----------------------------------------------------
    @property
    def spectral_axis(self) -> np.ndarray:
        return self.wavenumber

    @property
    def band_shape(self) -> tuple[int, ...]:
        """Shape of the non-spectral (spatial) axes."""
        return self.intensities.shape[:-1]

    def spectral_slice(self, lo: float, hi: float):
        """Return a copy cropped to the closed wavenumber window ``[lo, hi]``."""
        k = (self.wavenumber >= lo) & (self.wavenumber <= hi)
        if k.sum() < 2:
            raise ValueError(f"window ({lo}, {hi}) keeps < 2 channels")
        return self.__class__(
            self.intensities[..., k].copy(), self.wavenumber[k].copy(), dict(self.metadata)
        )

    def flat(self) -> np.ndarray:
        """(n_spectra, K) view, row-major over the spatial axes."""
        return self.intensities.reshape(-1, self.wavenumber.size)

    def copy(self):
        return self.__class__(
            self.intensities.copy(), self.wavenumber.copy(), dict(self.metadata)
        )


class Spectrum(_Base):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.intensities.ndim != 1:
            raise ValueError("Spectrum intensities must be 1-D (K,)")


class SpectralImage(_Base):
    """(H, W, K) gridded map, or (N, K) unstructured bag of spectra."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.intensities.ndim not in (2, 3):
            raise ValueError("SpectralImage intensities must be (H, W, K) or (N, K)")

    @property
    def is_gridded(self) -> bool:
        return self.intensities.ndim == 3

    def mean_spectrum(self) -> Spectrum:
        return Spectrum(self.flat().mean(0), self.wavenumber.copy(), dict(self.metadata))

    def as_grid(self, height: int, width: int) -> "SpectralImage":
        if self.is_gridded:
            return self
        n = self.intensities.shape[0]
        if height * width != n:
            raise ValueError(f"{height}x{width} != {n} spectra")
        cube = self.flat().reshape(height, width, self.wavenumber.size)
        return SpectralImage(cube, self.wavenumber.copy(), dict(self.metadata))


class SpectralVolume(_Base):
    """(D, H, W, K) stack of maps."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.intensities.ndim != 4:
            raise ValueError("SpectralVolume intensities must be (D, H, W, K)")

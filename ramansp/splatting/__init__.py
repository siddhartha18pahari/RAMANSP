"""Spectral 3D Gaussian Splatting for Raman hyperspectral maps.

A map ``V(x, y, wavenumber)`` is represented by a differentiable field of
anisotropic 3D Gaussians (ellipsoids). Each ellipsoid localises a chemical
domain in the two spatial axes and a band along the wavenumber axis, so the
fitted primitives are directly readable ("a band, here, this strong, this
wide"). Optimisation is plain first-order descent on **analytic** gradients --
no autodiff, no GPU -- with periodic prune/clone densification in the spirit of
Kerbl et al. (2023).

    from ramansp import io, preprocessing, splatting
    img   = preprocessing.protocol("carbon_dg").apply(io.read_matrix_map("DOE 8.txt"))
    field = splatting.fit_image(img, n_gaussians=4000, iters=300)
    dmap  = field.render_band(1350.0)          # D-band image, reconstructed
    field.save("field.npz")

The path to *millions* of Gaussians (a CUDA tile rasteriser + autodiff) is
specified, not implemented, in :mod:`ramansp.splatting.torch_backend`.
"""

from __future__ import annotations

from .model import GaussianField
from .fit import SplatConfig, fit_image
from .volume import Volume, build_volume

__all__ = ["GaussianField", "SplatConfig", "fit_image", "Volume", "build_volume"]

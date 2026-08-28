"""ramansp -- a reproducible cross-sample Raman workflow framework.

The design follows RamanSPy (Georgiev et al., *Anal. Chem.* 2024): typed
spectral containers, composable preprocessing pipelines with named protocols,
an analysis layer (decomposition / clustering / unmixing) and a plotting layer.

Two things are new here:

* ``ramansp.knowledge_graph`` -- a graph that ties many samples together by
  spectral similarity, shared unmixing endmembers and disorder-metric
  proximity, so a corpus of maps is queried as one object.
* ``ramansp.splatting`` -- a *Spectral 3D Gaussian Splatting* representation:
  a hyperspectral map V(x, y, wavenumber) is fitted by a differentiable field
  of anisotropic 3D ellipsoids, each localising a chemical domain in space and
  a band in wavenumber. Pure NumPy analytic gradients; a documented port to a
  CUDA tile rasteriser for millions of Gaussians lives in ``torch_backend``.

Quick start
-----------
>>> import ramansp as rp
>>> img = rp.io.read_matrix_map("DOE 8.txt")            # -> SpectralImage
>>> pipe = rp.preprocessing.protocol("carbon_dg")
>>> clean = pipe.apply(img)
>>> field = rp.splatting.fit_image(clean, n_gaussians=4000, iters=300)
>>> field.render_band(1350).shape                       # D-band map from the splat
"""

from __future__ import annotations

from . import analysis, io, knowledge_graph, metrics, plotting, preprocessing, splatting
from .containers import Spectrum, SpectralImage, SpectralVolume

__version__ = "0.1.0"

__all__ = [
    "Spectrum",
    "SpectralImage",
    "SpectralVolume",
    "io",
    "preprocessing",
    "analysis",
    "metrics",
    "plotting",
    "knowledge_graph",
    "splatting",
    "__version__",
]

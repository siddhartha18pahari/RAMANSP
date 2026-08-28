"""GPU backend for *millions* of Gaussians -- specified, not implemented.

The pure-NumPy engine in :mod:`ramansp.splatting.model` is O(N) per iteration
with a Python loop over Gaussians and tops out around 1e4-1e5 primitives on a
CPU. Scaling to the 1e6-1e7 range that "3D Gaussian Splatting" implies needs
three changes, none of which are algorithmic novelties -- they are the standard
Kerbl et al. (2023) rasteriser adapted from (u, v) image tiles to (x, y, k)
voxel tiles:

1. **Autodiff + GPU tensors.** Replace the four parameter arrays with
   ``torch.nn.Parameter`` tensors on CUDA. The forward is the same closed
   form; ``loss.backward()`` then replaces every hand-derived expression in
   ``loss_and_grad``. The analytic gradients here are what the tests check the
   autodiff path against.

2. **Tiled rasterisation.** Partition the volume into e.g. 16x16x16 voxel
   tiles. For each Gaussian, compute its 3-sigma AABB and the list of tiles it
   touches (a segmented sort by tile id, exactly as in 2D 3DGS but with a
   third axis). Each tile then sums only its assigned Gaussians -- a custom
   CUDA kernel, or ``torch`` scatter-add over a (tile, gaussian) COO layout for
   a pure-PyTorch version.

3. **Densification at scale.** The prune / clone / split logic in
   ``GaussianField.densify`` is already the 3DGS heuristic; on GPU it runs
   every ~100 steps over the full population with the gradient-magnitude
   threshold from the paper (tau_pos ~ 2e-4 in normalised units).

Spectral extension (optional): give each Gaussian a length-D code over a fixed
endmember dictionary ``Phi`` (from NMF/VCA of the map) so ``Phi.T @ c_i`` is a
physical Raman profile; the field value becomes
``sum_i a_i * G_i(p) * (Phi.T c_i)(k)``. Adds ``N x D`` parameters and one more
closed-form gradient block.

Install target: ``pip install "ramansp[torch]"`` then set
``RAMANSP_SPLAT_BACKEND=torch``.
"""

from __future__ import annotations


def is_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def fit_volume_torch(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "The torch/CUDA splatting backend is specified in this module's "
        "docstring but not implemented. Use ramansp.splatting.fit_volume "
        "(pure NumPy) for now."
    )

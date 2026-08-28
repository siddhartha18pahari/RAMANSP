"""Figures from a fitted :class:`GaussianField`.

Everything here is matplotlib only. The turntable GIF uses
``matplotlib.animation`` + pillow, both already dependencies.
"""

from __future__ import annotations

import numpy as np

from .model import GaussianField
from .volume import Volume


def _ellipsoid_axes(field: GaussianField):
    """Return centres (N,3), rotation matrices (N,3,3), radii (N,3) in world units."""
    from .model import _rotmat

    qn = field._qn()
    R = np.stack([_rotmat(qn[i]) for i in range(field.n)])
    return field.mu.copy(), R, field.sigma.copy()


def plot_ellipsoids(field: GaussianField, vol: Volume, ax=None, max_glyphs: int = 900,
                    elev: float = 22.0, azim: float = -60.0):
    """3-D scatter of the Gaussian centres, sized by opacity, coloured by the
    wavenumber they sit at -- i.e. which band each ellipsoid explains.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    mu, _R, s = _ellipsoid_axes(field)
    a = field.amplitude
    order = np.argsort(a)[-max_glyphs:]
    wn = vol.wn0 + mu[order, 2] * (vol.wn1 - vol.wn0)
    sizes = 8 + 240 * (a[order] / a[order].max())
    p = ax.scatter(mu[order, 0], mu[order, 1], wn, c=wn, s=sizes, cmap="turbo",
                   alpha=0.7, linewidths=0)
    ax.set_xlabel("x (norm.)"); ax.set_ylabel("y (norm.)")
    ax.set_zlabel("wavenumber (cm$^{-1}$)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{field.n} ellipsoids  (top {len(order)} by opacity)")
    return ax, p


def turntable_gif(field: GaussianField, vol: Volume, path: str, n_frames: int = 36,
                  dpi: int = 130):
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    def draw(k):
        ax.clear()
        plot_ellipsoids(field, vol, ax=ax, azim=-60 + 360 * k / n_frames)

    anim = animation.FuncAnimation(fig, draw, frames=n_frames, interval=90)
    anim.save(path, writer="pillow", dpi=dpi)
    plt.close(fig)
    return path


def band_triptych(field: GaussianField, vol: Volume, image_bands: dict, path: str | None = None):
    """Measured band map vs splat-reconstructed band map, per named band."""
    import matplotlib.pyplot as plt

    names = list(image_bands)
    fig, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 6.4), squeeze=False)
    for c, name in enumerate(names):
        meas = image_bands[name]
        recon = field.render_band(name if isinstance(name, float) else _band_centre(name),
                                  vol)
        for r, (img, tag) in enumerate([(meas, "measured"), (recon, "splat")]):
            ax = axes[r, c]
            ax.imshow(img, origin="lower", cmap="magma")
            ax.set_title(f"{name} -- {tag}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=320, bbox_inches="tight")
        plt.close(fig)
    return fig


def _band_centre(name: str) -> float:
    return {"D1": 1350.0, "D": 1350.0, "G": 1585.0, "G+D2": 1600.0, "D3": 1500.0}.get(name, 1500.0)


def spectra_panel(field: GaussianField, vol: Volume, n: int = 6, path: str | None = None,
                  seed: int = 0):
    """Reconstructed vs measured spectra at random in-mask positions, including
    the held-out channels.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    ii, jj = np.where(vol.spatial_mask)
    sel = rng.choice(len(ii), size=min(n, len(ii)), replace=False)

    wn_all = np.concatenate([vol.wn_train, vol.wn_eval])
    order = np.argsort(wn_all)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 5.2),
                             squeeze=False)
    for ax, s in zip(axes.ravel(), sel):
        i, j = ii[s], jj[s]
        meas = np.concatenate([vol.V[i, j], vol.V_eval[i, j]])[order]
        cx = np.full(wn_all.size, np.linspace(0, 1, vol.spatial_mask.shape[1])[j])
        cy = np.full(wn_all.size, np.linspace(0, 1, vol.spatial_mask.shape[0])[i])
        cz = (wn_all - vol.wn0) / (vol.wn1 - vol.wn0)
        coords = np.stack([cx, cy, cz], axis=-1)
        recon = field.render_at(coords)[order]
        ax.plot(wn_all[order], meas, lw=1.0, label="measured", color="0.35")
        ax.plot(wn_all[order], recon, lw=1.2, label="splat", color="crimson")
        ax.scatter(vol.wn_eval, [np.interp(w, wn_all[order], recon) for w in vol.wn_eval],
                   s=8, color="crimson", zorder=3)
        ax.set_xlabel("cm$^{-1}$"); ax.set_yticks([])
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=320, bbox_inches="tight")
        plt.close(fig)
    return fig

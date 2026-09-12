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


_BAND_WINDOWS = (("D1", 1300.0, 1400.0, "#D55E00"),
                 ("G+D2", 1540.0, 1660.0, "#0072B2"))


def _unit_sphere(nu: int = 16, nv: int = 11):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    return (np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)))


def _band_colour(wn):
    for _nm, lo, hi, col in _BAND_WINDOWS:
        if lo <= wn <= hi:
            return col
    return "#9a9a9a"


def plot_ellipsoids(field: GaussianField, vol: Volume, ax=None, max_glyphs: int = 900,
                    n_surface: int = 55, elev: float = 22.0, azim: float = -60.0):
    """Draw the primitives as the anisotropic ellipsoids they actually are.

    The earlier version scattered the centres as round markers, which showed
    neither the anisotropy nor the orientation that the representation exists
    to provide: a reader could not tell a sphere from a disc lying in the
    spatial plane from a needle running along the wavenumber axis. Here the
    strongest primitives are drawn as true ellipsoid surfaces from their own
    covariance, in the normalised cube the model is fitted in, so the shapes on
    the page are the shapes in the optimisation. The remaining centres stay as
    faint dots for context. Colour marks which carbon band a primitive sits on.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    mu, R, sig = _ellipsoid_axes(field)
    a = field.amplitude
    order = np.argsort(a)
    ctx = order[-max_glyphs:]
    top = order[-n_surface:]
    to_wn = lambda z: vol.wn0 + z * (vol.wn1 - vol.wn0)  # noqa: E731

    ax.scatter(mu[ctx, 0], mu[ctx, 1], mu[ctx, 2], s=5, alpha=0.16,
               c=[_band_colour(to_wn(z)) for z in mu[ctx, 2]], linewidths=0)

    sx, sy, sz = _unit_sphere()
    unit = np.stack([sx.ravel(), sy.ravel(), sz.ravel()])
    for i in top:
        pts = (R[i] @ (sig[i][:, None] * unit)) + mu[i][:, None]
        X, Y, Z = (pts[k].reshape(sx.shape) for k in range(3))
        ax.plot_surface(X, Y, Z, color=_band_colour(to_wn(mu[i, 2])),
                        alpha=0.42, linewidth=0, shade=True, antialiased=False)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    # the default 1.0 on x lands on top of the 0.0 on y at the front corner
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("x (norm.)", labelpad=-4)
    ax.set_ylabel("y (norm.)", labelpad=-4)
    zt = np.linspace(0, 1, 5)
    ax.set_zticks(zt)
    ax.set_zticklabels([f"{to_wn(z):.0f}" for z in zt])
    ax.set_zlabel("Raman shift (cm$^{-1}$)", labelpad=-2)
    ax.tick_params(labelsize=6, pad=-2)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:  # noqa: BLE001
        pass
    ax.set_title(f"{field.n} primitives, {n_surface} drawn as ellipsoids",
                 fontsize=8, y=0.98)
    return ax, None


def ellipsoid_panel(field: GaussianField, vol: Volume, path: str | None = None):
    """The flagship view: the ellipsoids, their shapes, and where they sit.

    One 3-D view alone cannot answer the two questions a reader has, namely
    whether the primitives are genuinely anisotropic and whether they land on
    the carbon bands rather than anywhere. Panels (b) and (c) answer both with
    numbers instead of an impression.
    """
    import matplotlib.pyplot as plt

    from .._style import OKABE, acs_figsize, apply_style, save
    apply_style()

    mu, _R, sig = _ellipsoid_axes(field)
    a = field.amplitude
    to_wn = lambda z: vol.wn0 + z * (vol.wn1 - vol.wn0)  # noqa: E731
    wn = to_wn(mu[:, 2])

    # a 3-D axes and two charts in a single row of 7 in leaves each panel about
    # 2.2 in, at which the 3-D tick labels and the neighbouring y-label collide.
    # The view gets its own column instead, spanning both rows.
    fig = plt.figure(figsize=acs_figsize("double", 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0],
                          wspace=0.30, hspace=0.55, left=0.015, right=0.975,
                          top=0.925, bottom=0.09)
    ax0 = fig.add_subplot(gs[:, 0], projection="3d")
    plot_ellipsoids(field, vol, ax=ax0, n_surface=70)
    ax0.text2D(-0.02, 1.02, "a", transform=ax0.transAxes, fontsize=9,
               fontweight="bold", va="top")

    ax1 = fig.add_subplot(gs[0, 1])
    aniso = sig.max(1) / np.clip(sig.min(1), 1e-12, None)
    clip = 12.0
    n_over = int((aniso > clip).sum())
    ax1.hist(np.clip(aniso, 1, clip), bins=40, color=OKABE["blue"], alpha=0.85)
    med = float(np.median(aniso))
    top1 = ax1.get_ylim()[1]
    ax1.axvline(med, color=OKABE["red"], ls="--", lw=1.4)
    # headroom above the tallest bar, so neither label sits on the data or has
    # the rule it belongs to drawn through it
    ax1.set_ylim(0, top1 * 1.20)
    ax1.text(med * 1.10, top1 * 1.05, f"median {med:.2f}", color=OKABE["red"],
             fontsize=7, va="center")
    ax1.axvline(1.0, color="#777777", lw=0.8)
    # offset in points, not data units, so the rule at 1.0 never touches the word
    ax1.annotate("isotropic", xy=(1.0, top1 * 1.14), xytext=(4, 0),
                 textcoords="offset points", fontsize=6.5, color="#777777",
                 ha="left", va="center")
    if n_over:
        ax1.text(clip, top1 * 0.55, f"{n_over} beyond {clip:.0f} (clipped)",
                 fontsize=7, color="#777777", ha="right", va="center")
    ax1.set_xlabel("axis ratio  (longest / shortest)")
    ax1.set_ylabel("primitives")
    ax1.set_title("(b) the primitives are anisotropic", fontsize=8, loc="left")

    ax2 = fig.add_subplot(gs[1, 1])
    w = a / max(a.sum(), 1e-30)
    ax2.hist(wn, bins=60, weights=w, color="#9a9a9a", alpha=0.9)
    top2 = ax2.get_ylim()[1]
    for k, (nm, lo, hi, col) in enumerate(_BAND_WINDOWS):
        ax2.axvspan(lo, hi, color=col, alpha=0.16, lw=0)
        frac = 100 * w[(wn >= lo) & (wn <= hi)].sum()
        ax2.text(0.5 * (lo + hi), top2 * (0.95 - 0.10 * k), f"{nm}: {frac:.0f}%",
                 ha="center", fontsize=8, color=col, fontweight="bold")
    ax2.set_ylim(0, top2 * 1.08)
    ax2.set_xlabel("Raman shift of primitive centre (cm$^{-1}$)")
    ax2.set_ylabel("share of total amplitude")
    ax2.set_title("(c) amplitude concentrates on the bands", fontsize=8,
                  loc="left")

    stats = {"aniso_median": float(np.median(aniso)),
             "aniso_p90": float(np.percentile(aniso, 90)),
             "amp_share": {nm: float(w[(wn >= lo) & (wn <= hi)].sum())
                           for nm, lo, hi, _c in _BAND_WINDOWS},
             "band_span_frac": float(sum(hi - lo for _n, lo, hi, _c in _BAND_WINDOWS)
                                     / max(vol.wn1 - vol.wn0, 1e-9))}
    if path:
        save(fig, path)
    return stats


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


def band_maps(field: GaussianField, vol: Volume, bands=None):
    """(measured, reconstructed) band-area maps, computed identically."""
    import numpy as _np
    from ..analysis import band_areas
    from ..containers import SpectralImage

    H, W = vol.spatial_mask.shape
    meas_img = SpectralImage(vol.V * vol.scale, vol.wn_train, {})
    xa, ya, _z = field.world_axes
    recon = field.render_grid((xa, ya, vol.z_train)) * vol.scale
    rec_img = SpectralImage(recon, vol.wn_train, {})

    ma = band_areas(meas_img, bands)
    ra = band_areas(rec_img, bands)
    out = {}
    for name in ma:
        m = _np.where(vol.spatial_mask, ma[name].reshape(H, W), _np.nan)
        r = _np.where(vol.spatial_mask, ra[name].reshape(H, W), _np.nan)
        out[name] = (m, r)
    return out


def band_triptych(field: GaussianField, vol: Volume, image_bands=None,
                  path: str | None = None, bands=None):
    """Measured, reconstructed and residual band maps on a shared colour scale.

    ``image_bands`` is accepted for backwards compatibility and ignored; both
    maps are recomputed here so they are the same quantity.

    The colour bars live in their own grid columns rather than being attached
    to the map axes. Attaching them lets matplotlib take the space out of those
    axes, which on a fixed canvas put the bar on top of the reconstructed map
    and hid the data.
    """
    import matplotlib.pyplot as plt
    import numpy as _np

    from .._style import (acs_figsize, apply_style, masked_imshow,
                          robust_limits, save)

    apply_style()
    pairs = band_maps(field, vol, bands)
    names = list(pairs)
    nr = len(names)

    # column 3 is an empty spacer: without it the band-area colour bar label is
    # set hard against the residual panel and reads as if it belonged to it
    fig = plt.figure(figsize=acs_figsize("double", 2.28 * nr))
    gs = fig.add_gridspec(nr, 6, width_ratios=[1, 1, 0.05, 0.26, 1, 0.05],
                          wspace=0.12, hspace=0.20,
                          left=0.012, right=0.905,
                          top=1 - 0.245 / nr, bottom=0.015)

    for r, name in enumerate(names):
        meas, rec = pairs[name]
        vmin, vmax = robust_limits(meas)

        ax0 = fig.add_subplot(gs[r, 0])
        ax1 = fig.add_subplot(gs[r, 1])
        masked_imshow(ax0, meas, vmin, vmax)
        im1 = masked_imshow(ax1, rec, vmin, vmax)
        ax0.set_title(f"{name}: measured", fontsize=7.5)
        ax1.set_title(f"{name}: reconstructed", fontsize=7.5)

        cax1 = fig.add_subplot(gs[r, 2])
        cb1 = fig.colorbar(im1, cax=cax1)
        cb1.ax.tick_params(labelsize=6)
        cb1.outline.set_linewidth(0.5)
        cb1.set_label("band area (a.u.)", fontsize=6.5)

        diff = rec - meas
        lim = _np.nanpercentile(_np.abs(diff), 98) or 1.0
        ax2 = fig.add_subplot(gs[r, 4])
        im2 = masked_imshow(ax2, diff, -lim, lim, cmap="RdBu_r")
        scale = max(_np.nanmedian(_np.abs(meas)), 1e-30)
        frac = _np.nanmedian(_np.abs(diff)) / scale
        # the signed median matters separately: a residual that is mostly one
        # colour is a systematic under- or over-shoot, not scatter, and saying
        # only "median |err|" would hide which of the two this is
        bias = _np.nanmedian(diff) / scale
        ax2.set_title("residual" + chr(10) +
                      f"median |err| {100 * frac:.1f}%, "
                      f"signed {100 * bias:+.1f}%", fontsize=7.5)

        cax2 = fig.add_subplot(gs[r, 5])
        cb2 = fig.colorbar(im2, cax=cax2)
        cb2.ax.tick_params(labelsize=6)
        cb2.outline.set_linewidth(0.5)
        cb2.set_label("reconstructed - measured", fontsize=6.5)

    # the residual title runs to two lines, so the banner needs real clearance
    fig.suptitle("Band maps share one colour scale per band; "
                 "grey is outside the fitted mask", fontsize=7.5, y=0.998,
                 va="top")
    if path:
        save(fig, path)
    return fig


def spectra_panel(field: GaussianField, vol: Volume, n: int = 6, path: str | None = None,
                  seed: int = 0):
    """Reconstructed vs measured spectra at random in-mask positions, including
    the held-out channels.
    """
    import matplotlib.pyplot as plt

    from .._style import acs_figsize, apply_style, save
    apply_style()

    rng = np.random.default_rng(seed)
    ii, jj = np.where(vol.spatial_mask)
    sel = rng.choice(len(ii), size=min(n, len(ii)), replace=False)

    wn_all = np.concatenate([vol.wn_train, vol.wn_eval])
    order = np.argsort(wn_all)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=acs_figsize("double", 3.5),
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
        ax.plot(wn_all[order], recon, lw=1.3, label="splat", color="crimson")
        ax.scatter(vol.wn_eval, [np.interp(w, wn_all[order], recon) for w in vol.wn_eval],
                   s=14, color="crimson", zorder=3, edgecolors="white", linewidths=0.4,
                   label="withheld channel")
        # the withheld channels are the whole point of the panel, so say how far
        # off they are instead of leaving the reader to judge by eye
        ev = np.interp(vol.wn_eval, wn_all[order], recon) - vol.V_eval[i, j]
        rng_ = np.ptp(meas) or 1.0
        ax.set_title(f"pixel ({int(j)}, {int(i)});  withheld RMSE "
                     f"{100 * np.sqrt(np.mean(ev ** 2)) / rng_:.1f}% of range",
                     fontsize=7.5, loc="left")
        ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=8)
        ax.set_yticks([])
    axes[0, 0].set_ylabel("intensity (a.u.)", fontsize=8)
    axes[0, 0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(pad=0.4)
    if path:
        save(fig, path)
    return fig

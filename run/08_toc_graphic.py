"""Stage 8 -- the table-of-contents graphic.

Analytical Chemistry requires one at 8.25 cm by 4.45 cm (3.25 in by 1.75 in).
At that size almost nothing survives, so this is not a shrunk body figure: it
carries one claim in three panels, with no type below 6 pt and no axis
furniture at all.

The middle panel went through a 3-D view of the fitted ellipsoids first. At
1.2 in wide the primitives were specks, and the only ways to rescue them were
to enlarge them beyond their fitted size or to crop to a handful, neither of
which is an honest thumbnail. It shows the field's own reconstruction of the
band map instead, which is the claim the compression number makes.

    figures/fig_toc.png (+ .pdf)
"""

from __future__ import annotations

import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import CORPUS, FIGS, SPLAT  # noqa: E402

from ramansp._style import (ACS_TOC_IN, OKABE, apply_style,  # noqa: E402
                            masked_imshow, robust_limits, save)
from ramansp.containers import SpectralImage  # noqa: E402
from ramansp.splatting.fit import SplatConfig  # noqa: E402
from ramansp.splatting.model import GaussianField  # noqa: E402
from ramansp.splatting.render import band_maps  # noqa: E402
from ramansp.splatting.volume import build_volume  # noqa: E402


def flagship():
    allm = SPLAT / "all_metrics.json"
    reps = json.loads(allm.read_text()) if allm.exists() else []
    pool = [r for r in reps if r.get("flagship")] or reps
    if not pool:
        return None
    return max(pool, key=lambda r: r.get("fitted_voxels", 0))["acq_id"]


def main():
    apply_style()
    acq = flagship()
    if acq is None:
        print("no fitted field; run run/03_splat_fit.py first")
        return

    arr = np.load(CORPUS / "arrays" / f"{acq}.npz")
    img = SpectralImage(arr["cube"].astype(float), arr["wavenumber"].astype(float), {})
    cfg = SplatConfig()
    vol = build_volume(img, spatial_mask=arr["particle_mask"],
                       hold_out=cfg.hold_out, channel_bin=cfg.channel_bin)
    field = GaussianField.load(str(SPLAT / acq / "field.npz"))
    rep = json.loads((SPLAT / acq / "metrics.json").read_text())

    pairs = band_maps(field, vol)
    name = "G+D2" if "G+D2" in pairs else list(pairs)[0]
    meas, recon = pairs[name]
    vmin, vmax = robust_limits(meas)

    fig = plt.figure(figsize=ACS_TOC_IN)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.06,
                          left=0.02, right=0.98, top=0.775, bottom=0.135)

    ax0 = fig.add_subplot(gs[0, 0])
    masked_imshow(ax0, meas, vmin, vmax)
    ax0.set_title("measured map", fontsize=6.2, pad=1.5)

    ax1 = fig.add_subplot(gs[0, 1])
    masked_imshow(ax1, recon, vmin, vmax)
    ax1.set_title(f"{rep['n_gaussians']} ellipsoids", fontsize=6.2, pad=1.5)

    ax2 = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(3)
    ang = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    px = 0.5 + 0.35 * np.cos(ang) + rng.normal(0, 0.02, 9)
    py = 0.5 + 0.35 * np.sin(ang) + rng.normal(0, 0.02, 9)
    groups = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    cols = [OKABE["blue"], OKABE["orange"], OKABE["green"]]
    for i in range(9):
        ax2.plot([0.5, px[i]], [0.5, py[i]], lw=0.4, color="#c4c4c4", zorder=1)
        for j in range(i + 1, 9):
            if groups[i] == groups[j]:
                ax2.plot([px[i], px[j]], [py[i], py[j]], lw=0.8,
                         color=cols[groups[i]], alpha=0.8, zorder=2)
    ax2.scatter(0.5, 0.5, s=58, marker="*", color="#444444", zorder=4)
    ax2.scatter(px, py, s=24, c=[cols[g] for g in groups], zorder=4,
                edgecolors="white", linewidths=0.4)
    ax2.set_xlim(0.03, 0.97)
    ax2.set_ylim(0.03, 0.97)
    ax2.set_aspect("equal")
    ax2.set_axis_off()
    ax2.grid(False)
    ax2.set_title("knowledge graph", fontsize=6.2, pad=1.5)

    fig.text(0.5, 0.99, "One map, one compact field, one corpus-wide graph",
             ha="center", va="top", fontsize=7.5, fontweight="bold",
             color="#1a1a1a")
    raw = rep.get("raw_cube_values_full", rep["raw_cube_values"])
    fig.text(0.5, 0.012,
             f"{raw:,} cube values to {rep['n_params']:,} numbers "
             f"({rep['compression_ratio_full']:.0f}x), at the map's own noise floor",
             ha="center", va="bottom", fontsize=6.2, color="#555555")
    for x in (0.345, 0.665):
        fig.text(x, 0.45, "→", ha="center", va="center", fontsize=11,
                 color="#999999")

    save(fig, FIGS / "fig_toc.png")
    print(f"wrote {FIGS / 'fig_toc.png'} for map {acq}")


if __name__ == "__main__":
    main()

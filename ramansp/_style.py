"""Shared figure style and plotting primitives.

Two rules this module exists to enforce, both of which were violated by the
first pass at these figures:

1. Panels that a reader is invited to compare must share a colour scale, and
   must show it. Independent per-panel scaling makes a faithful reconstruction
   look like a failure, or the reverse.
2. Masked-out data is not zero. It gets its own neutral colour, never the
   bottom of the colour map, which would read as "measured, and low".
"""

from __future__ import annotations

import numpy as np

# Colour-blind-safe qualitative set (Okabe-Ito), used consistently everywhere.
OKABE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000",
}
MASK_COLOUR = "#d9d9d9"     # masked / not measured
GRID = "#e8e8e8"
INK = "#1a1a1a"

# --- Analytical Chemistry artwork specification -----------------------
# Values are the journal's own, not house style, so they are named rather
# than sprinkled through the figure code:
#   single column   240 pt = 3.33 in = 8.46 cm
#   double column   300 to 504 pt = 4.167 to 7.0 in = 10.58 to 17.78 cm
#   maximum depth   660 pt = 9.167 in, caption included
#   minimum type    4.5 pt; we hold 6 pt so nothing is borderline
#   minimum rule    0.5 pt
#   resolution      300 dpi colour, 600 dpi greyscale, 1200 dpi line art
# A caption of a few lines costs roughly 1 in, so figures are capped at
# 7.6 in of art to stay inside the depth limit once set.
ACS_SINGLE_IN = 3.33
ACS_DOUBLE_IN = 7.0
ACS_MAX_DEPTH_IN = 9.167
ACS_ART_DEPTH_IN = 7.6
ACS_DPI = 600
ACS_MIN_PT = 6.0
ACS_MIN_RULE_PT = 0.5
ACS_TOC_IN = (3.25, 1.75)          # 8.25 cm by 4.45 cm
ACS_FONTS = ["Arial", "Helvetica", "Nimbus Sans", "TeX Gyre Heros",
             "Liberation Sans", "DejaVu Sans"]


def apply_style():
    """Typography and axes for every figure, to the journal's artwork spec.

    Sizes are set once here rather than per figure so that no panel can drift
    below the 4.5 pt type floor or the 0.5 pt rule floor. Sans-serif throughout
    because the journal asks for Helvetica or Arial, and Type 42 fonts in the
    PDF output so the text stays selectable and embeddable rather than being
    shipped as Type 3 bitmaps.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        # NOT "tight": a tight bounding box trims or expands the canvas, so the
        # delivered file is whatever width the content happened to need rather
        # than the column width it was declared at. Each figure fits its
        # content inside the fixed canvas instead.
        "savefig.bbox": None,
        "savefig.dpi": ACS_DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ACS_FONTS,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "image.cmap": "magma",
        "figure.dpi": 110,
    })


def acs_figsize(width="double", height_in=None, aspect=0.42):
    """Figure size in inches at an exact journal column width.

    ``width`` is "single", "double", or a number of inches, and is never
    rescaled afterwards: a figure exported at the column width and then resized
    by ``includegraphics`` is exactly how type ends up below the 4.5 pt floor.
    """
    w = (ACS_SINGLE_IN if width == "single"
         else ACS_DOUBLE_IN if width == "double" else float(width))
    h = float(height_in) if height_in else w * aspect
    return w, min(h, ACS_ART_DEPTH_IN)


def enforce_minimums(fig):
    """Raise any text below the type floor and any rule below the rule floor.

    A safety net rather than a substitute for setting sizes correctly: a
    hand-tuned ``fontsize=5`` somewhere in a panel is exactly the kind of thing
    that survives review of the code and fails review of the artwork.
    """
    lo_pt = ACS_MIN_PT
    lo_lw = ACS_MIN_RULE_PT / 72 * 72  # points, matplotlib linewidth unit
    for t in fig.findobj(match=lambda o: hasattr(o, "get_fontsize")):
        try:
            if t.get_fontsize() < lo_pt:
                t.set_fontsize(lo_pt)
        except Exception:  # noqa: BLE001
            pass
    for o in fig.findobj(match=lambda o: hasattr(o, "get_linewidth")):
        try:
            lw = o.get_linewidth()
            if lw is not None and 0 < float(lw) < lo_lw:
                o.set_linewidth(lo_lw)
        except Exception:  # noqa: BLE001
            pass
    return fig


def overflowing(fig, slack_pt: float = 1.0):
    """Artists that stick out past the canvas, as (label, points over) pairs.

    With the canvas fixed at a column width nothing is trimmed to fit any more,
    so a title or tick label wider than its panel is simply cut off in the
    output file. That is invisible in a thumbnail and obvious in print, which is
    the worst combination, so it is detected rather than eyeballed.

    Only artists that are actually drawn count. Matplotlib keeps tick labels
    beyond the view limits, and an axes with its frame off keeps ticks it never
    renders; both report extents outside the canvas and neither is a defect. A
    3-D axes is skipped outright because its projected label extents are not
    comparable with the canvas in the same way.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    tol = slack_pt * fig.dpi / 72
    out = []

    def visible_ticks(ax, axis):
        lo, hi = (ax.get_xlim() if axis == "x" else ax.get_ylim())
        lo, hi = min(lo, hi), max(lo, hi)
        pos = ax.get_xticks() if axis == "x" else ax.get_yticks()
        labs = (ax.get_xticklabels() if axis == "x" else ax.get_yticklabels())
        span = hi - lo
        return [t for v, t in zip(pos, labs)
                if lo - 1e-9 * span <= v <= hi + 1e-9 * span]

    for ax in fig.axes:
        if hasattr(ax, "get_zlim") or not getattr(ax, "axison", True):
            continue
        items = [(f"{_axname(ax)} title", ax.title),
                 (f"{_axname(ax)} xlabel", ax.xaxis.label),
                 (f"{_axname(ax)} ylabel", ax.yaxis.label)]
        items += [(f"{_axname(ax)} xtick", t) for t in visible_ticks(ax, "x")]
        items += [(f"{_axname(ax)} ytick", t) for t in visible_ticks(ax, "y")]
        for name, artist in items:
            if artist is None or not artist.get_visible():
                continue
            txt = str(getattr(artist, "get_text", lambda: "")()).strip()
            if not txt:
                continue
            try:
                bb = artist.get_window_extent(renderer)
            except Exception:  # noqa: BLE001
                continue
            over = max(fb.x0 - bb.x0, bb.x1 - fb.x1,
                       fb.y0 - bb.y0, bb.y1 - fb.y1)
            if over > tol:
                out.append((f"{name}: {txt[:40]!r}", over * 72 / fig.dpi))
    return out


def _axname(ax):
    try:
        ss = ax.get_subplotspec()
        return f"axes[{ss.num1}]"
    except Exception:  # noqa: BLE001
        return "axes"


def save(fig, path, dpi: int = ACS_DPI, also_pdf: bool = True,
         strict: bool = False):
    """Write the figure at submission resolution, plus a vector PDF sibling.

    The PDF is what the manuscript includes, because vector text cannot be
    resampled below the journal's resolution floor no matter what scale it is
    placed at; the PNG is kept for anyone assembling the figures outside LaTeX.

    Saved with an explicit bounding box, so the file on disk is exactly the size
    the figure was declared at and the journal can place it 1:1.
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    path = __import__("pathlib").Path(path)
    enforce_minimums(fig)
    w, h = fig.get_size_inches()
    if h > ACS_MAX_DEPTH_IN:
        raise ValueError(f"{path.name}: {h:.2f} in tall exceeds the "
                         f"{ACS_MAX_DEPTH_IN} in depth limit")
    bad = overflowing(fig)
    if bad:
        msg = "; ".join(f"{n} by {o:.1f} pt" for n, o in bad[:4])
        if strict:
            raise ValueError(f"{path.name}: content clipped -- {msg}")
        print(f"  [clipped] {path.name}: {msg}")
    box = Bbox.from_bounds(0, 0, w, h)
    fig.savefig(path, dpi=dpi, bbox_inches=box)
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches=box)
    plt.close(fig)
    return path


def robust_limits(*arrays, lo=2.0, hi=98.0):
    """Shared (vmin, vmax) from percentiles, so one outlier cannot set the scale.

    The prior study on this corpus hit exactly that: a single pixel with a
    ratio of 3079 against a median of 22 flattened an entire map to one shade.
    """
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(vals, [lo, hi])
    if not np.isfinite(vmin) or vmin == vmax:
        vmin, vmax = float(vals.min()), float(vals.max())
    if vmin == vmax:
        vmax = vmin + 1e-9
    return float(vmin), float(vmax)


def masked_imshow(ax, img, vmin=None, vmax=None, cmap="magma", extent=None,
                  mask_colour=MASK_COLOUR):
    """imshow with NaN rendered in a neutral colour rather than as a value."""
    import matplotlib as mpl

    cm = mpl.colormaps[cmap].copy()
    cm.set_bad(mask_colour)
    im = ax.imshow(np.ma.masked_invalid(np.asarray(img, float)), origin="lower",
                   cmap=cm, vmin=vmin, vmax=vmax, extent=extent,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#888888")
    return im


def colorbar(fig, im, ax, label="", shrink=0.85, pad=0.02):
    cb = fig.colorbar(im, ax=ax, shrink=shrink, pad=pad)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.6)
    if label:
        cb.set_label(label, fontsize=7.5)
    return cb


def panel_tag(ax, tag, dx=-0.08, dy=1.04, size=9):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=size, fontweight="bold",
            va="top", ha="left", color=INK)

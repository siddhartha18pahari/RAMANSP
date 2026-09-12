"""Stage 6 -- the framework workflow diagram.

A schematic of what the pipeline does, from the instrument's own binary through
to the manuscript. Drawn rather than photographed so it stays in sync with the
code: the stage names and output names below are the real ones.

The layout is portrait rather than landscape. The earlier landscape version
placed five text-filled boxes per row, which at the journal's 7 in column width
leaves each box 1.4 in and forces the type inside below the 4.5 pt floor. Three
boxes per row at the same width leaves 2.2 in each, which holds 6 pt
comfortably, so the diagram grew downwards instead of being shrunk sideways.

    figures/fig_workflow.png (+ .pdf)
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import FIGS  # noqa: E402

from ramansp._style import acs_figsize, apply_style, save  # noqa: E402

INK = "#1b1b1b"
COLS = {
    "input": "#c9d6e4",
    "io": "#8fb8d8",
    "prep": "#a8cbb0",
    "graph": "#e6c08a",
    "splat": "#dda0a0",
    "out": "#cbc3dd",
}
BODY = 6.2          # body type inside boxes, above the 4.5 pt floor
HEAD = 6.8          # box headings


def box(ax, x, y, w, h, text, color, fontsize=BODY, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=0.6, edgecolor=INK, facecolor=color, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, zorder=3, weight=weight,
            linespacing=1.4)


def arrow(ax, p, q, style="-|>", lw=0.8, rad=0.0, color=INK, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=7, linewidth=lw, color=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1,
        shrinkA=1.0, shrinkB=1.0))


def band(ax, y, label):
    ax.text(0.005, y, label, fontsize=6.4, style="italic", color="#8a8a8a",
            ha="left", va="bottom")


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=acs_figsize("double", 7.4))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.965, bottom=0.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    # ---- band 1: acquisition and ingest ----------------------------
    band(ax, 0.962, "1  ingest and anonymise")
    box(ax, 0.02, 0.845, 0.20, 0.105,
        "Raman microscope\n(map acquisition)", COLS["input"])
    box(ax, 0.26, 0.902, 0.22, 0.048, "vendor binary  .l6m / .l6s", COLS["input"])
    box(ax, 0.26, 0.845, 0.22, 0.048, "manual text export", "#eeeeee")
    box(ax, 0.52, 0.845, 0.21, 0.105,
        r"$\bf{io.read\_l6}$" "\nrecord walker,\nconstraint-based typing",
        COLS["io"])
    box(ax, 0.77, 0.845, 0.21, 0.105,
        "anonymiser + leak gate\nopaque codes,\nprivate key held back",
        COLS["io"])

    arrow(ax, (0.22, 0.915), (0.26, 0.926))
    arrow(ax, (0.22, 0.882), (0.26, 0.869))
    arrow(ax, (0.48, 0.926), (0.52, 0.912))
    arrow(ax, (0.48, 0.869), (0.52, 0.884), ls=":", lw=0.7)
    arrow(ax, (0.73, 0.897), (0.77, 0.897))

    ax.text(0.37, 0.957, "23 maps reach analysis", ha="center", fontsize=6.0,
            style="italic", color="#2f6b3f")
    ax.text(0.37, 0.826, "1 map reaches analysis", ha="center", fontsize=6.0,
            style="italic", color="#8a3b3b")

    box(ax, 0.20, 0.735, 0.60, 0.068,
        r"$\bf{01\_build\_corpus}$"
        "    manifest  .  cached cubes  .  endmembers  .  "
        "byte-identical duplicate detection", COLS["out"])
    arrow(ax, (0.875, 0.845), (0.70, 0.803), rad=0.10)

    # ---- band 2: the library ---------------------------------------
    band(ax, 0.690, "2  library: preprocessing, analysis, metrics")
    box(ax, 0.02, 0.560, 0.30, 0.115,
        r"$\bf{preprocessing.Pipeline}$" "\ncrop . despike . baseline\n"
        "smooth . normalise\n"
        r"$\it{named\ protocols,\ recorded\ with\ every\ number}$",
        COLS["prep"])
    box(ax, 0.35, 0.560, 0.30, 0.115,
        r"$\bf{analysis}$" "\nPCA / NMF / ICA . k-means\nVCA + FCLS unmixing\n"
        "band areas . band-contrast mask", COLS["prep"])
    box(ax, 0.68, 0.560, 0.30, 0.115,
        r"$\bf{metrics}$" "\nPSNR . SSIM . SAM\nvariogram . effective $n$\n"
        "second-difference noise", COLS["prep"])
    arrow(ax, (0.50, 0.735), (0.50, 0.675))
    arrow(ax, (0.32, 0.618), (0.35, 0.618))
    arrow(ax, (0.65, 0.618), (0.68, 0.618))

    # ---- band 3: the two cross-sample products ---------------------
    band(ax, 0.512, "3  the two cross-sample products")
    box(ax, 0.02, 0.290, 0.46, 0.195,
        r"$\bf{02\_knowledge\_graph}$" "\n\n"
        "nodes   specimen . material . config\nacquisition . endmember . splat model\n\n"
        "edges   provenance  +  SIMILAR_TO (mutual $k$NN)\n"
        "SHARES_ENDMEMBER . DISORDER_NEIGHBOUR\n"
        r"$\it{every\ metric\ edge\ carries\ its\ protocol}$", COLS["graph"])
    box(ax, 0.52, 0.290, 0.46, 0.195,
        r"$\bf{03\_splat\_fit}$   Spectral 3D Gaussian Splatting" "\n\n"
        r"$\widehat{V}(p)=\sum_i a_i\,\exp(-\frac{1}{2}(p-\mu_i)^{\top}"
        r"P_i(p-\mu_i))$" "\n\n"
        "anisotropic ellipsoids in $(x,y,\\nu)$\n"
        "closed-form gradients . separable evaluation\n"
        "Adam + prune / clone densification", COLS["splat"])
    arrow(ax, (0.25, 0.560), (0.25, 0.485))
    arrow(ax, (0.75, 0.560), (0.75, 0.485))

    # ---- band 4: benchmark and released artefacts ------------------
    band(ax, 0.245, "4  benchmark and released artefacts")
    box(ax, 0.02, 0.130, 0.30, 0.095,
        r"$\bf{07\_ml\_benchmark}$" "\ndenoiser comparison\n"
        "30-model benchmark\nWilcoxon + Benjamini-Hochberg", COLS["prep"])
    box(ax, 0.35, 0.130, 0.30, 0.095,
        "graphml + interactive view\ncommunities . duplicate groups\nqueries",
        COLS["out"])
    box(ax, 0.68, 0.130, 0.30, 0.095,
        "compact field, $11N$ numbers\nband maps . novel slices\n"
        "ellipsoid statistics", COLS["out"])
    arrow(ax, (0.17, 0.290), (0.17, 0.225))
    arrow(ax, (0.25, 0.290), (0.50, 0.225), rad=-0.10)
    arrow(ax, (0.75, 0.290), (0.83, 0.225))

    box(ax, 0.20, 0.020, 0.60, 0.062,
        r"$\bf{04\ /\ 05}$   figures, tables and LaTeX macros $\rightarrow$ "
        "manuscript\n"
        r"$\it{no\ number\ in\ the\ paper\ is\ typed\ by\ hand}$", COLS["out"])
    for x in (0.17, 0.50, 0.83):
        arrow(ax, (x, 0.130), (min(max(x, 0.28), 0.72), 0.082),
              rad=0.0 if x == 0.50 else (0.08 if x < 0.5 else -0.08))

    ax.set_title("ramansp: from the instrument's own bytes to the manuscript",
                 fontsize=8.5, weight="bold", color=INK, pad=2)
    save(fig, FIGS / "fig_workflow.png")
    print(f"wrote {FIGS / 'fig_workflow.png'}")


if __name__ == "__main__":
    main()

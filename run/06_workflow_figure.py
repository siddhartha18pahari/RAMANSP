"""Stage 6 -- the framework workflow diagram.

A schematic of what the pipeline does, from the instrument's own binary through
to the manuscript. Drawn rather than photographed so it stays in sync with the
code: the stage names and output names below are the real ones.

    figures/fig_workflow.png
"""

from __future__ import annotations

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import FIGS  # noqa: E402

INK = "#1b1b1b"
COLS = {
    "input": "#c9d6e4",
    "io": "#8fb8d8",
    "prep": "#a8cbb0",
    "graph": "#e6c08a",
    "splat": "#dda0a0",
    "out": "#cbc3dd",
}


def box(ax, x, y, w, h, text, color, fontsize=7.4, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.9, edgecolor=INK, facecolor=color, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, zorder=3, weight=weight,
            linespacing=1.35)


def arrow(ax, p, q, style="-|>", lw=1.1, rad=0.0, color=INK, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=11, linewidth=lw, color=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1,
        shrinkA=1.5, shrinkB=1.5))


def main():
    fig, ax = plt.subplots(figsize=(11.6, 6.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- row 1: acquisition and ingest -----------------------------
    box(ax, 0.015, 0.79, 0.135, 0.15,
        "Raman\nmicroscope\n(map acquisition)", COLS["input"], 7.6)
    box(ax, 0.175, 0.865, 0.155, 0.075,
        "vendor binary\n.l6m / .l6s", COLS["input"])
    box(ax, 0.175, 0.775, 0.155, 0.075,
        "manual text export\n(the usual route)", "#eeeeee")

    box(ax, 0.365, 0.79, 0.15, 0.15,
        "$\\bf{io.read\\_l6}$\nrecord walker\nconstraint-based\narray typing",
        COLS["io"], 7.2)
    box(ax, 0.545, 0.79, 0.145, 0.15,
        "anonymiser\n+ leak gate\n\nopaque codes,\nprivate key", COLS["io"], 7.2)
    box(ax, 0.72, 0.79, 0.265, 0.15,
        "$\\bf{01\\_build\\_corpus}$\n"
        "manifest . cached cubes . endmembers\n"
        "byte-identical duplicate detection", COLS["out"], 7.2)

    arrow(ax, (0.15, 0.895), (0.175, 0.9))
    arrow(ax, (0.15, 0.845), (0.175, 0.815))
    arrow(ax, (0.33, 0.9), (0.365, 0.878))
    arrow(ax, (0.33, 0.812), (0.365, 0.845), ls=":", lw=0.9)
    arrow(ax, (0.515, 0.865), (0.545, 0.865))
    arrow(ax, (0.69, 0.865), (0.72, 0.865))

    ax.text(0.253, 0.752, "1 map reaches analysis", ha="center",
            fontsize=6.3, style="italic", color="#8a3b3b")
    ax.text(0.253, 0.952, "23 maps reach analysis", ha="center",
            fontsize=6.3, style="italic", color="#2f6b3f")

    # ---- row 2: preprocessing -------------------------------------
    box(ax, 0.015, 0.545, 0.20, 0.14,
        "$\\bf{preprocessing.Pipeline}$\n"
        "crop . despike . baseline\nsmooth . normalise", COLS["prep"], 7.2)
    box(ax, 0.235, 0.545, 0.175, 0.14,
        "named protocols\n"
        "carbon_dg . arpls\nchord . poly3 . minimal\n"
        "$\\it{recorded\\ with\\ every\\ number}$", COLS["prep"], 6.9)
    box(ax, 0.43, 0.545, 0.17, 0.14,
        "$\\bf{analysis}$\nPCA / NMF / ICA\nk-means . VCA+FCLS\nband areas, Otsu mask",
        COLS["prep"], 7.0)
    box(ax, 0.62, 0.545, 0.17, 0.14,
        "$\\bf{metrics}$\nPSNR . SSIM . SAM\nvariogram . effective $n$\n"
        "interleave noise", COLS["prep"], 7.0)
    box(ax, 0.81, 0.545, 0.175, 0.14,
        "$\\bf{07\\_ml\\_benchmark}$\ndenoiser comparison\n"
        "model benchmark\nWilcoxon + BH", COLS["prep"], 7.0)

    arrow(ax, (0.852, 0.79), (0.6, 0.687), rad=0.12)
    arrow(ax, (0.215, 0.615), (0.235, 0.615))
    arrow(ax, (0.41, 0.615), (0.43, 0.615))
    arrow(ax, (0.60, 0.615), (0.62, 0.615))
    arrow(ax, (0.79, 0.615), (0.81, 0.615))

    # ---- row 3: the two analysis products -------------------------
    box(ax, 0.05, 0.235, 0.36, 0.235,
        "$\\bf{02\\_knowledge\\_graph}$\n\n"
        "nodes  specimen . material . config\n"
        "acquisition . endmember . splat model\n\n"
        "edges  provenance  +  SIMILAR_TO (mutual $k$NN)\n"
        "SHARES_ENDMEMBER . DISORDER_NEIGHBOUR\n"
        "$\\it{every\\ metric\\ edge\\ carries\\ its\\ protocol}$",
        COLS["graph"], 7.0)
    box(ax, 0.47, 0.235, 0.48, 0.235,
        "$\\bf{03\\_splat\\_fit}$   Spectral 3D Gaussian Splatting\n\n"
        r"$\widehat{V}(p)=\sum_i a_i\,\exp(-\frac{1}{2}(p-\mu_i)^{\top}P_i(p-\mu_i))$"
        "\n\nanisotropic ellipsoids in $(x,y,\\nu)$\n"
        "closed-form gradients . separable evaluation\n"
        "Adam + prune / clone densification",
        COLS["splat"], 7.0)

    arrow(ax, (0.2, 0.545), (0.2, 0.47))
    arrow(ax, (0.68, 0.545), (0.68, 0.47))

    # ---- row 4: outputs -------------------------------------------
    box(ax, 0.05, 0.05, 0.17, 0.14,
        "graphml\n+ interactive\npyvis view", COLS["out"], 7.2)
    box(ax, 0.24, 0.05, 0.17, 0.14,
        "communities\nduplicate groups\nqueries", COLS["out"], 7.2)
    box(ax, 0.47, 0.05, 0.22, 0.14,
        "compact field\n$11N$ numbers\nband maps . novel slices", COLS["out"], 7.2)
    box(ax, 0.72, 0.05, 0.23, 0.14,
        "$\\bf{04/05}$  figures, tables,\n"
        "LaTeX macros $\\rightarrow$ manuscript\n"
        "$\\it{no\\ number\\ typed\\ by\\ hand}$", COLS["out"], 7.2)

    arrow(ax, (0.135, 0.235), (0.135, 0.19))
    arrow(ax, (0.32, 0.235), (0.32, 0.19))
    arrow(ax, (0.58, 0.235), (0.58, 0.19))
    arrow(ax, (0.69, 0.12), (0.72, 0.12))
    arrow(ax, (0.41, 0.12), (0.47, 0.12))

    ax.text(0.5, 0.985, "ramansp: from the instrument's own bytes to the manuscript",
            ha="center", fontsize=10.5, weight="bold", color=INK)

    fig.savefig(FIGS / "fig_workflow.png", dpi=320, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {FIGS / 'fig_workflow.png'}")


if __name__ == "__main__":
    main()

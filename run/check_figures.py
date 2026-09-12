"""Check every manuscript figure against the journal's artwork specification.

Analytical Chemistry asks for artwork at one of its column widths, no deeper
than the page allows, at 300 dpi for colour and 600 dpi for greyscale, with no
type below 4.5 pt and no rule below 0.5 pt. Exits non-zero on any violation, so
it can gate a build the way run/check_anonymity.py gates a release.

The type and rule floors are enforced at draw time by ramansp._style, which is
the only place they can be checked reliably; what this script checks is what
can still be got wrong afterwards, namely the dimensions and resolution of the
files actually handed over, and whether a vector version exists at all.

    python run/check_figures.py
"""

from __future__ import annotations

import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import PAPER  # noqa: E402

from ramansp._style import (ACS_DOUBLE_IN, ACS_MAX_DEPTH_IN,  # noqa: E402
                            ACS_SINGLE_IN, ACS_TOC_IN)

MIN_DPI = 300
WIDTH_TOL = 0.06          # inches; bbox_inches="tight" trims a hair
TOC = "fig_toc"


def png_size_in(path):
    """(width, height) in inches and the stored dpi, from the PNG header."""
    from PIL import Image

    with Image.open(path) as im:
        dpi = im.info.get("dpi", (0, 0))
        d = float(dpi[0]) or 0.0
        if d <= 0:
            return None
        return im.width / d, im.height / d, d


def main():
    figdir = PAPER / "figures"
    # main.tex is only a driver since the preprint was added: the figures live
    # in the shared body, and reading the driver alone silently checked one
    tex = "".join((PAPER / n).read_text(encoding="utf-8")
                  for n in ("main.tex", "body.tex") if (PAPER / n).exists())
    used = sorted(set(re.findall(r"includegraphics\[[^]]*\]\{([^}]*)\}", tex)))
    if not used:
        print("no figures referenced by the manuscript")
        return 1

    problems, rows = [], []
    for stem in used:
        stem = stem.rsplit(".", 1)[0]
        png, pdf = figdir / f"{stem}.png", figdir / f"{stem}.pdf"
        if not png.exists():
            problems.append(f"{stem}: no raster version")
            continue
        if not pdf.exists():
            problems.append(f"{stem}: no vector version")
        got = png_size_in(png)
        if got is None:
            problems.append(f"{stem}: PNG carries no dpi")
            continue
        w, h, dpi = got
        targets = ([ACS_TOC_IN[0]] if stem == TOC
                   else [ACS_SINGLE_IN, ACS_DOUBLE_IN])
        if not any(abs(w - t) <= WIDTH_TOL for t in targets):
            problems.append(
                f"{stem}: {w:.2f} in wide, not a column width "
                f"({' or '.join(f'{t:.2f}' for t in targets)})")
        depth_cap = ACS_TOC_IN[1] + WIDTH_TOL if stem == TOC else ACS_MAX_DEPTH_IN
        if h > depth_cap:
            problems.append(f"{stem}: {h:.2f} in deep, over the "
                            f"{depth_cap:.2f} in limit")
        if dpi < MIN_DPI:
            problems.append(f"{stem}: {dpi:.0f} dpi, under {MIN_DPI}")
        rows.append((stem, w, h, dpi, pdf.exists()))

    width = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'figure':<{width}}  {'in':>5} {'x':>1} {'in':>5}  {'dpi':>4}  vector")
    for stem, w, h, dpi, has_pdf in rows:
        print(f"{stem:<{width}}  {w:5.2f} x {h:5.2f}  {dpi:4.0f}  "
              f"{'yes' if has_pdf else 'NO'}")

    if problems:
        print(f"\nFAIL -- {len(problems)} artwork problems")
        for line in problems:
            print(f"  {line}")
        return 1
    print(f"\nOK -- {len(rows)} figures meet the artwork specification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

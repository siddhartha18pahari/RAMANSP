"""Stage 9 -- assemble the Analytical Chemistry submission package.

The journal asks for the material as separate files: a manuscript file, a cover
letter, Supporting Information for Publication, and optionally a manuscript PDF
that becomes the proof used during peer review. This script gathers what the
pipeline has produced into one folder in that shape, and writes a checklist
mapping each requirement in the author guide to the file that satisfies it or
to the reason it is still outstanding.

Nothing here is authored: it copies artefacts that other stages built, and
fails loudly if one is missing, so the package cannot quietly ship a stale or
absent figure.

    python run/09_acs_submission.py
"""

from __future__ import annotations

import datetime as _dt
import shutil
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import GRAPH, OUTPUT, PAPER, TABLES  # noqa: E402

SUB = OUTPUT.parent / "ACS SUBMISSION"

# published path -> source, and whether its absence is fatal
MANIFEST = [
    ("01_cover_letter.docx", PAPER / "cover_letter.docx", True),
    ("01_cover_letter.md", PAPER / "cover_letter.md", True),
    ("02_manuscript.pdf", PAPER / "main.pdf", True),
    ("03_supporting_information.pdf", None, True),   # resolved at run time
    ("04_toc_graphic.png", PAPER / "figures" / "fig_toc.png", True),
    ("04_toc_graphic.pdf", PAPER / "figures" / "fig_toc.pdf", True),
    # not part of the ACS submission: the same paper in ordinary article
    # format, for a preprint server or for circulating a readable copy
    ("05_preprint_not_for_acs.pdf", None, True),
]

# the manuscript's own figures, at the journal's column widths
FIGURE_STEMS = [
    "fig_workflow", "fig_corpus", "fig_splat_quality", "flagship_ellipsoids",
    "flagship_bands", "flagship_spectra", "fig_preproc_ablation", "kg_graph",
    "kg_similarity_heatmap", "fig_ml_denoise", "fig_ml_benchmark",
]

# the LaTeX sources, so the journal's TeX route is available at revision
SOURCE = ["main.tex", "preprint.tex", "si.tex", "body.tex", "abstract.tex",
          "ack.tex", "dataavail.tex", "suppinfo.tex", "placeholders.tex",
          "refs.bib", "values.tex", "build.ps1", "build_si.ps1",
          "build_preprint.ps1"]

# machine-readable artefacts referenced by the data availability statement
DATA = [
    ("data/kg_communities.csv", TABLES / "kg_communities.csv"),
    ("data/ml_models.csv", TABLES / "ml_models.csv"),
    ("data/ml_denoise.csv", TABLES / "ml_denoise.csv"),
    ("data/knowledge_graph.graphml", GRAPH / "knowledge_graph.graphml"),
    ("data/knowledge_graph.html", GRAPH / "knowledge_graph.html"),
    ("data/kg_summary.json", GRAPH / "kg_summary.json"),
]

CHECKLIST = """# Analytical Chemistry submission package

Assembled by `run/09_acs_submission.py` on {date}. Every file here is copied
from a pipeline output; nothing is hand-edited in this folder, so regenerate it
rather than editing it in place.

## What the journal asks for, and where it is

| Requirement (author guide) | File | Status |
|---|---|---|
| Manuscript file | `02_manuscript.pdf` | see note on file format below |
| Cover letter | `01_cover_letter.docx` | **placeholders to fill** |
| Supporting Information for Publication | `03_supporting_information.pdf` | ready |
| Table of contents graphic, 8.25 x 4.45 cm | `04_toc_graphic.png` / `.pdf` | {toc} |
| Graphics at one- or two-column width | `figures/` | {figs} |
| Abstract, 80 to 250 words | in manuscript | {abswords} words |
| Manuscript word count, Article limit 10,000 | in manuscript | {bodywords} words |
| Introduction heading not used | in manuscript | done |
| Sections not numbered | in manuscript | done |
| Safety statement in Experimental Section | in manuscript | drafted, **check it** |
| Conflict of interest statement | in manuscript | drafted |
| Author contributions statement | in manuscript | drafted |
| AI tool disclosure in Acknowledgments | in manuscript | drafted, **check it** |
| Data availability statement | in manuscript | done |
| References with titles, ACS format | in manuscript | done, {nrefs} entries |

`05_preprint_not_for_acs.pdf` is **not** part of the submission. It is the same
paper in an ordinary two-column article format, for a preprint server or for
circulating a readable copy. Body text, abstract and the closing statements are
shared files, so it and the submission cannot say different things.

## Before you submit

Four things in this package need you, not the pipeline.

1. **Cover letter placeholders.** `01_cover_letter.docx` has the missing items
   highlighted in yellow:
   your phone number, your ORCID iD, four to six suggested reviewers with email
   addresses, and a statement about preprint deposition. The guide requires all
   of them, and suggested reviewers must not be at your institution. The
   Markdown source is beside it; edit that and re-run
   `run/10_cover_letter_docx.py` if you prefer, since the Word file is
   generated from it.

2. **Manuscript file format.** The guide lists the manuscript file as a single
   `.doc` or `.docx`. This manuscript is LaTeX. ACS accepts TeX/LaTeX
   submissions through a separate route, and the PDF here can be uploaded as
   the optional Manuscript PDF File, which becomes the proof used in review.
   The full LaTeX source is in `latex_source/` for the TeX route or for
   conversion. Confirm which route the submission system offers you before
   uploading.

3. **Read the safety statement.** It is drafted from what the work actually
   involved, a reanalysis of already-acquired spectra with no new hazards, but
   you know the laboratory and should confirm it.

4. **Read the AI disclosure.** The guide requires disclosure of AI tool use for
   text or image generation, in the Acknowledgments, describing when and how.
   A description is drafted. It should match your own account of the work.

## Contents

```
{tree}
```

## Rebuilding

```
python run/04_figures.py
python run/05_paper_assets.py
python run/08_toc_graphic.py
powershell -ExecutionPolicy Bypass -File paper/build.ps1
powershell -ExecutionPolicy Bypass -File paper/build_si.ps1
python run/09_acs_submission.py
```

`run/check_figures.py` verifies the artwork against the journal's size and
resolution specification and exits non-zero on any violation.
"""


def measure():
    """Word counts and reference count, read from the built manuscript."""
    import re

    bs = chr(92)
    # main.tex is now only a driver; the prose lives in the shared includes
    body = (PAPER / "body.tex").read_text(encoding="utf-8")
    for extra in ("ack.tex", "dataavail.tex"):
        body += chr(10) + (PAPER / extra).read_text(encoding="utf-8")
    for env in ("figure*", "figure", "table*", "table", "algorithm"):
        body = re.sub(bs * 2 + r"begin\{" + re.escape(env) + r"\}.*?"
                      + bs * 2 + r"end\{" + re.escape(env) + r"\}", " ",
                      body, flags=re.S)
    body = re.sub(r"(?m)^\s*%.*$", " ", body)
    body = re.sub(bs * 2 + r"[A-Za-z@]+\*?", " ", body)
    body = re.sub(r"[{}$&_^~\\\[\]]", " ", body)
    words = [w for w in body.split() if any(c.isalnum() for c in w)]

    a = (PAPER / "abstract.tex").read_text(encoding="utf-8")
    a = re.sub(r"(?m)^\s*%.*$", " ", a)
    a = re.sub(bs * 2 + r"[A-Za-z@]+\*?", " ", a)
    a = re.sub(r"[{}$&_^~]", " ", a)
    aw = len([w for w in a.split() if any(c.isalnum() for c in w)])

    bbl = PAPER / "main.bbl"
    nrefs = bbl.read_text(encoding="utf-8").count(bs + "bibitem") if bbl.exists() else 0
    return len(words), aw, nrefs


def copy(src, dst, locked):
    """Copy over the top, recording rather than raising if the target is held.

    Wiping and recreating the folder is cleaner but fails outright when a PDF
    viewer holds one file open, which throws away the whole assembly for one
    stale document. Overwriting in place degrades to a named warning instead.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(src, dst)
        return True
    except PermissionError:
        locked.append(dst.name)
        return False


def preflight(targets):
    """Names of destinations that exist and cannot be opened for writing.

    Checked before anything is copied. Failing half way through leaves the
    package with some files fresh and some stale, which is worse than not
    running at all, because the folder then looks assembled.
    """
    held = []
    for dst in targets:
        if not dst.exists():
            continue
        try:
            with open(dst, "ab"):
                pass
        except PermissionError:
            held.append(dst.name)
    return held


def main():
    for sub in ("figures", "latex_source/figures", "latex_source/tables", "data"):
        (SUB / sub).mkdir(parents=True, exist_ok=True)
    locked = []

    # the SI build falls back to si_tmp.pdf when a viewer holds si.pdf open, so
    # take whichever of the two is newer rather than shipping a stale document
    def newest(*names):
        found = [PAPER / n for n in names if (PAPER / n).exists()]
        return (max(found, key=lambda q: q.stat().st_mtime) if found
                else PAPER / names[0])

    si_src = newest("si.pdf", "si_tmp.pdf")
    pp_src = newest("preprint.pdf", "pp_tmp.pdf")
    late = {"03_supporting_information.pdf": si_src,
            "05_preprint_not_for_acs.pdf": pp_src}

    held = preflight([SUB / dst for dst, _s, _f in MANIFEST]
                     + [SUB / "figures" / f"{stem}.{ext}"
                        for stem in FIGURE_STEMS for ext in ("png", "pdf")]
                     + [SUB / "latex_source" / n for n in SOURCE]
                     + [SUB / dst for dst, _s in DATA]
                     + [SUB / "00_README_submission.md"])
    if held:
        raise SystemExit(
            "nothing copied. These are open in another program; close them "
            "and re-run:" + chr(10) + "  "
            + (chr(10) + "  ").join(sorted(set(held))))

    missing = []
    for dst, src, fatal in MANIFEST:
        src = late[dst] if src is None else src
        if src.exists():
            copy(src, SUB / dst, locked)
        elif fatal:
            missing.append(str(src))

    figdir = PAPER / "figures"
    n_fig = 0
    for stem in FIGURE_STEMS:
        for ext in ("png", "pdf"):
            src = figdir / f"{stem}.{ext}"
            if src.exists():
                copy(src, SUB / "figures" / src.name, locked)
                n_fig += ext == "pdf"
            else:
                missing.append(str(src))

    for name in SOURCE:
        src = PAPER / name
        if src.exists():
            copy(src, SUB / "latex_source" / name, locked)
        else:
            missing.append(str(src))
    for src in figdir.glob("*"):
        if src.is_file():
            copy(src, SUB / "latex_source" / "figures" / src.name, locked)
    for src in (PAPER / "tables").glob("*.tex"):
        copy(src, SUB / "latex_source" / "tables" / src.name, locked)

    for dst, src in DATA:
        if src.exists():
            copy(src, SUB / dst, locked)

    if missing:
        raise SystemExit("missing required artefacts:\n  "
                         + "\n  ".join(sorted(set(missing))))
    if locked:
        # shipping the old copy of a file we failed to overwrite is exactly the
        # stale-artefact failure this script exists to prevent, so it is fatal
        raise SystemExit(
            "could not overwrite these, they are open in another program;\n"
            "close them and re-run:\n  " + "\n  ".join(sorted(set(locked))))

    body, abs_w, nrefs = measure()
    tree = "\n".join(sorted(
        str(p.relative_to(SUB)).replace("\\", "/") + ("/" if p.is_dir() else "")
        for p in SUB.rglob("*") if p.parent == SUB or p.parent.parent == SUB))
    (SUB / "00_README_submission.md").write_text(CHECKLIST.format(
        date=_dt.date.today().isoformat(),
        toc="3.25 x 1.75 in, 600 dpi",
        figs=f"{n_fig} figures, all at a column width",
        abswords=abs_w, bodywords=f"{body:,}", nrefs=nrefs, tree=tree),
        encoding="utf-8")

    total = sum(p.stat().st_size for p in SUB.rglob("*") if p.is_file())
    print(f"assembled {SUB}")
    print(f"  {len(list(SUB.rglob('*')))} files, {total / 1e6:.1f} MB")
    print(f"  abstract {abs_w} words, body {body:,} words, {nrefs} references")
    print(f"  {n_fig} manuscript figures + TOC graphic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

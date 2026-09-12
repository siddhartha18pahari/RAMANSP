"""Stage 11 -- assemble the static site published at ramanspjs.vercel.app.

The site is a landing page plus three things that are better on the web than in
a PDF: the interactive knowledge graph, the figures at full resolution, and the
three renderings of the paper.

Every number on the page is read from the pipeline's own outputs, the same way
the manuscript's are, so the site cannot drift from the paper. Nothing here is
hand-typed except the prose.

    python run/11_build_site.py
"""

from __future__ import annotations

import html
import json
import shutil
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import FIGS, GRAPH, OUTPUT, PAPER, ROOT, SPLAT  # noqa: E402

# The site is deployed on its own, not from the repository, so it is built
# outside the working tree: Vercel gets a standalone folder and GitHub gets
# only the code and the paper.
WEB = ROOT.parent / "RAMANSP SITE"
GITHUB = "https://github.com/siddhartha18pahari/RAMANSP"
SITE = "https://ramanspjs.vercel.app"

FIGURES = [
    ("fig_workflow", "The pipeline",
     "From the instrument's own bytes to the manuscript."),
    ("fig_corpus", "The anonymised corpus",
     "What is in it, and what reading the vendor binary unlocks."),
    ("kg_graph", "Cross-sample knowledge graph",
     "Provenance backbone plus content edges earned from the data."),
    ("kg_similarity_heatmap", "Pairwise similarity",
     "Why a fixed cosine threshold is not usable on this corpus."),
    ("flagship_ellipsoids", "The fitted primitives",
     "Anisotropic ellipsoids in two spatial axes and wavenumber."),
    ("flagship_bands", "Measured against reconstructed",
     "Both integrated the same way, on a shared colour scale."),
    ("flagship_spectra", "Withheld channels",
     "The dots are wavenumbers the field never saw during fitting."),
    ("fig_splat_quality", "What limits the fit",
     "Held-out error against each map's own measurement noise."),
    ("fig_preproc_ablation", "Preprocessing dependence",
     "The bounded index survives the baseline choice; the raw ratio does not."),
    ("fig_ml_denoise", "Denoising benchmark",
     "Reported where our own representation loses, as well as where it wins."),
    ("fig_ml_benchmark", "Classifier benchmark",
     "Every accuracy against its majority-class baseline."),
]

PAPERS = [
    ("preprint.pdf", "Preprint",
     "Ordinary two-column format. Start here."),
    ("main.pdf", "Journal manuscript",
     "Analytical Chemistry submission format."),
    ("si.pdf", "Supporting Information",
     "Per-map reconstructions and the full benchmark tables."),
]


def numbers():
    """Headline figures, read from the pipeline rather than typed."""
    out = {}
    corpus = OUTPUT / "corpus" / "summary.json"
    if corpus.exists():
        out.update(json.loads(corpus.read_text()))
    kg = GRAPH / "kg_summary.json"
    if kg.exists():
        out["kg"] = json.loads(kg.read_text())
    allm = SPLAT / "all_metrics.json"
    if allm.exists():
        reps = json.loads(allm.read_text())
        out["splat"] = reps
        flag = [r for r in reps if r.get("flagship")] or reps
        out["flagship"] = max(flag, key=lambda r: r.get("fitted_voxels", 0))
    ml = OUTPUT / "ml" / "summary.json"
    if ml.exists():
        out["ml"] = json.loads(ml.read_text())
    return out


def stat_cards(n):
    cards = []
    acq = n.get("n_acquisitions")
    if acq:
        cards.append((f"{acq}", "acquisitions in the anonymised corpus"))
    kg = n.get("kg") or {}
    if kg.get("n_communities"):
        cards.append((f"{kg['n_communities']}",
                      f"graph communities at Q = {kg.get('modularity', 0):.2f}"))
    f = n.get("flagship") or {}
    if f.get("compression_ratio_full"):
        cards.append((f"{f['compression_ratio_full']:.0f}&times;",
                      "compression, at the map's own noise floor"))
    if f.get("n_gaussians"):
        cards.append((f"{f['n_gaussians']:,}",
                      "differentiable ellipsoids in the flagship field"))
    return cards


def main():
    if WEB.exists():
        shutil.rmtree(WEB)
    (WEB / "figures").mkdir(parents=True)
    (WEB / "papers").mkdir(parents=True)

    for stem, _t, _c in FIGURES:
        for ext in ("png", "pdf"):
            src = PAPER / "figures" / f"{stem}.{ext}"
            if src.exists():
                shutil.copy(src, WEB / "figures" / src.name)
    for name, _t, _c in PAPERS:
        src = PAPER / name
        alt = PAPER / (name.replace(".pdf", "_tmp.pdf"))
        src = src if src.exists() else alt
        if src.exists():
            shutil.copy(src, WEB / "papers" / name)

    graph_src = GRAPH / "knowledge_graph.html"
    has_graph = graph_src.exists()
    if has_graph:
        shutil.copy(graph_src, WEB / "graph.html")
    for extra in ("knowledge_graph.graphml", "kg_summary.json"):
        if (GRAPH / extra).exists():
            shutil.copy(GRAPH / extra, WEB / extra)

    n = numbers()
    cards = "\n".join(
        f'      <div class="stat"><b>{v}</b><span>{html.escape(k)}</span></div>'
        for v, k in stat_cards(n))
    papers = "\n".join(
        f'      <a class="card" href="papers/{name}">'
        f'<b>{html.escape(title)}</b>'
        f'<span>{html.escape(desc)}</span></a>'
        for name, title, desc in PAPERS if (WEB / "papers" / name).exists())
    figs = "\n".join(
        f'      <figure>\n'
        f'        <a href="figures/{stem}.png">'
        f'<img src="figures/{stem}.png" alt="{html.escape(title)}" loading="lazy"></a>\n'
        f'        <figcaption><b>{html.escape(title)}</b> '
        f'{html.escape(caption)}</figcaption>\n'
        f'      </figure>'
        for stem, title, caption in FIGURES
        if (WEB / "figures" / f"{stem}.png").exists())

    graph_block = (
        '      <a class="card wide" href="graph.html"><b>Open the interactive '
        'graph</b><span>Every acquisition, its provenance and its earned '
        'content edges. Drag, zoom, and hover a node for its '
        'metadata.</span></a>' if has_graph else
        '      <p class="muted">Run <code>python run/02_knowledge_graph.py</code> '
        'to generate the interactive graph.</p>')

    (WEB / "index.html").write_text(
        TEMPLATE.format(cards=cards, papers=papers, figures=figs,
                        graph=graph_block, github=GITHUB, site=SITE),
        encoding="utf-8")

    # config sits inside the site folder, so `vercel deploy --prod` run from
    # there needs no outputDirectory indirection and no repository link
    (WEB / "vercel.json").write_text(json.dumps({
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "cleanUrls": True,
        "trailingSlash": False,
        "headers": [{
            "source": "/(.*)",
            "headers": [{"key": "X-Content-Type-Options", "value": "nosniff"}],
        }],
    }, indent=2) + "\n", encoding="utf-8")

    total = sum(p.stat().st_size for p in WEB.rglob("*") if p.is_file())
    print(f"built {WEB}")
    print(f"  {len(list(WEB.rglob('*')))} files, {total / 1e6:.1f} MB")
    n_papers = papers.count("class=" + chr(34) + "card" + chr(34))
    print(f"  {figs.count('<figure>')} figures, {n_papers} papers, "
          f"interactive graph: {'yes' if has_graph else 'no'}")
    return 0


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ramansp &middot; cross-sample Raman workflow with spectral 3D Gaussian splatting</title>
<meta name="description" content="An open framework for cross-sample Raman
analysis: a reader for the undocumented vendor container, a knowledge graph
across acquisitions, and a differentiable field of anisotropic ellipsoids that
represents a hyperspectral map.">
<style>
  :root {{
    --ink: #15181c; --muted: #5d646e; --line: #e3e6ea; --bg: #ffffff;
    --accent: #0072B2; --soft: #f6f8fa;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --ink: #e8ebef; --muted: #9aa3ae; --line: #262b31; --bg: #0f1216;
      --accent: #56B4E9; --soft: #161b21;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 60rem; margin: 0 auto; padding: 0 1.25rem; }}
  header {{ border-bottom: 1px solid var(--line); padding: 3.5rem 0 2.5rem; }}
  h1 {{ font-size: 2.1rem; line-height: 1.2; margin: 0 0 .6rem; letter-spacing: -.02em; }}
  h1 span {{ color: var(--accent); }}
  .lede {{ font-size: 1.12rem; color: var(--muted); max-width: 46rem; margin: 0 0 1.5rem; }}
  .links a {{
    display: inline-block; margin: 0 .5rem .5rem 0; padding: .5rem .95rem;
    border: 1px solid var(--line); border-radius: 7px; text-decoration: none;
    color: var(--ink); font-size: .94rem; background: var(--soft);
  }}
  .links a:hover {{ border-color: var(--accent); color: var(--accent); }}
  .links a.primary {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  section {{ padding: 2.75rem 0; border-bottom: 1px solid var(--line); }}
  h2 {{ font-size: 1.25rem; margin: 0 0 1.1rem; letter-spacing: -.01em; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: 1rem; }}
  .stat {{ background: var(--soft); border: 1px solid var(--line); border-radius: 9px; padding: 1rem 1.1rem; }}
  .stat b {{ display: block; font-size: 1.75rem; line-height: 1.1; color: var(--accent); }}
  .stat span {{ display: block; color: var(--muted); font-size: .88rem; margin-top: .3rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 1rem; }}
  .card {{
    display: block; padding: 1.1rem 1.2rem; border: 1px solid var(--line);
    border-radius: 9px; text-decoration: none; color: var(--ink); background: var(--soft);
  }}
  .card:hover {{ border-color: var(--accent); }}
  .card b {{ display: block; margin-bottom: .25rem; }}
  .card span {{ color: var(--muted); font-size: .9rem; }}
  .card.wide {{ grid-column: 1 / -1; }}
  figure {{ margin: 0 0 2rem; }}
  figure img {{
    width: 100%; height: auto; border: 1px solid var(--line);
    border-radius: 8px; background: #fff;
  }}
  figcaption {{ color: var(--muted); font-size: .9rem; margin-top: .55rem; }}
  figcaption b {{ color: var(--ink); }}
  p {{ max-width: 46rem; }}
  code {{
    background: var(--soft); border: 1px solid var(--line); border-radius: 4px;
    padding: .1rem .35rem; font-size: .88em;
  }}
  pre {{
    background: var(--soft); border: 1px solid var(--line); border-radius: 8px;
    padding: 1rem; overflow-x: auto; font-size: .88rem;
  }}
  pre code {{ background: none; border: 0; padding: 0; }}
  .muted {{ color: var(--muted); }}
  footer {{ padding: 2.5rem 0 4rem; color: var(--muted); font-size: .9rem; }}
  footer a {{ color: var(--accent); }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>ramansp <span>&middot;</span> reading the whole corpus</h1>
  <p class="lede">An open framework for cross-sample Raman analysis. It reads
  the instrument's undocumented binary container directly, links every
  acquisition in a study into one queryable graph, and represents a
  hyperspectral map as a differentiable field of anisotropic
  three-dimensional ellipsoids fitted on closed-form gradients, in NumPy, on a
  CPU.</p>
  <p class="links">
    <a class="primary" href="papers/preprint.pdf">Read the preprint</a>
    <a href="{github}">Source on GitHub</a>
    <a href="graph.html">Interactive graph</a>
  </p>
</header>

<section>
  <h2>At a glance</h2>
  <div class="stats">
{cards}
  </div>
</section>

<section>
  <h2>The paper</h2>
  <div class="cards">
{papers}
  </div>
</section>

<section>
  <h2>The knowledge graph</h2>
  <p>Provenance edges are true by construction and are drawn as a backbone.
  Content edges have to be earned: spectral similarity by mutual
  <i>k</i>-nearest neighbours rather than a fixed threshold, shared unmixing
  endmembers, and disorder-metric proximity that carries the preprocessing
  protocol it was computed under.</p>
  <div class="cards">
{graph}
  </div>
</section>

<section>
  <h2>Figures</h2>
{figures}
</section>

<section>
  <h2>Run it</h2>
  <pre><code>git clone {github}.git
cd RAMANSP
pip install -e .
pytest ramansp/tests -q

python run/01_build_corpus.py     # ingest, anonymise, cache
python run/02_knowledge_graph.py  # graph + interactive view
python run/03_splat_fit.py        # fit the Gaussian field
python run/04_figures.py          # cross-cutting figures</code></pre>
  <p class="muted">The corpus is distributed in anonymised form only. The
  identity map is withheld, and a gate fails the build on any surviving
  identifier.</p>
</section>

<footer>
  <p>Siddhartha Pahari and Jainish Shailesh Solanki, University of Toronto.
  Source and issues at <a href="{github}">{github}</a>.
  This page is at <a href="{site}">{site}</a>.</p>
</footer>

</div>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

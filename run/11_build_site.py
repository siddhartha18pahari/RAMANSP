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
import re
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
    # Clear the generated content but keep anything hidden: `vercel link`
    # writes .vercel/project.json here, and wiping the folder wholesale
    # unlinks the deployment, so the next deploy tries to create a project
    # named after this directory and fails on the space in the name.
    if WEB.exists():
        for item in WEB.iterdir():
            if item.name.startswith("."):
                continue
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    (WEB / "figures").mkdir(parents=True, exist_ok=True)
    (WEB / "papers").mkdir(parents=True, exist_ok=True)

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

    # the rotating view of the fitted field: a moving graphic that is data
    turntable = None
    for cand in sorted(SPLAT.glob("*/turntable.gif"),
                       key=lambda q: q.stat().st_mtime, reverse=True):
        shutil.copy(cand, WEB / "figures" / "turntable.gif")
        turntable = "figures/turntable.gif"
        break

    graph_src = GRAPH / "knowledge_graph.html"
    has_graph = graph_src.exists()
    if has_graph:
        shutil.copy(graph_src, WEB / "graph.html")
    for extra in ("knowledge_graph.graphml", "kg_summary.json"):
        if (GRAPH / extra).exists():
            shutil.copy(GRAPH / extra, WEB / extra)

    n = numbers()
    def card(v, k):
        num = re.sub(r"[^0-9.]", "", v) or "0"
        return (f'      <div class="stat reveal"><b data-to="{num}">{v}</b>'
                f'<span>{html.escape(k)}</span></div>')
    cards = "\n".join(card(v, k) for v, k in stat_cards(n))
    papers = "\n".join(
        f'      <a class="card" href="papers/{name}">'
        f'<b>{html.escape(title)}</b>'
        f'<span>{html.escape(desc)}</span></a>'
        for name, title, desc in PAPERS if (WEB / "papers" / name).exists())
    figs = "\n".join(
        f'      <figure class=\"reveal\">\n'
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

    hero = ('      <figure class="turntable reveal">\n'
            f'        <img src="{turntable}" alt="The fitted Gaussian field, '
            'rotating">\n'
            '        <figcaption>The fitted field itself, rotating. Each '
            'ellipsoid is a primitive with a position, three semi-axes and an '
            'orientation, fitted on closed-form gradients. Colour marks the '
            'carbon band its centre falls in.</figcaption>\n'
            '      </figure>' if turntable else "")
    (WEB / "index.html").write_text(
        TEMPLATE.format(cards=cards, papers=papers, figures=figs,
                        graph=graph_block, github=GITHUB, site=SITE,
                        hero=hero),
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
    print(f"  {figs.count('<figure')} figures, {n_papers} papers, "
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
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><ellipse cx='16' cy='16' rx='13' ry='6' transform='rotate(-25 16 16)' fill='%230072B2'/></svg>">
<style>
  :root {{
    --ink: #10141a; --muted: #5b6472; --line: #e4e8ee; --bg: #ffffff;
    --accent: #0072B2; --accent2: #D55E00; --soft: #f6f8fb; --glow: rgba(0,114,178,.12);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --ink: #e9edf3; --muted: #99a3b2; --line: #232a33; --bg: #0b0e13;
      --accent: #56B4E9; --accent2: #E69F00; --soft: #12171e; --glow: rgba(86,180,233,.14);
    }}
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased; overflow-x: hidden;
  }}
  .wrap {{ max-width: 62rem; margin: 0 auto; padding: 0 1.35rem; }}

  /* ---- sticky nav ---- */
  nav {{
    position: sticky; top: 0; z-index: 40; backdrop-filter: saturate(180%) blur(12px);
    background: color-mix(in srgb, var(--bg) 82%, transparent);
    border-bottom: 1px solid transparent; transition: border-color .25s;
  }}
  nav.stuck {{ border-bottom-color: var(--line); }}
  nav .wrap {{ display: flex; align-items: center; gap: 1.25rem; height: 3.4rem; }}
  nav b {{ font-size: .98rem; letter-spacing: -.01em; }}
  nav b i {{ color: var(--accent); font-style: normal; }}
  nav a {{ color: var(--muted); text-decoration: none; font-size: .9rem; }}
  nav a:hover {{ color: var(--accent); }}
  nav .spacer {{ margin-left: auto; }}
  @media (max-width: 44rem) {{ nav .hide-sm {{ display: none; }} }}

  /* ---- hero with the animated field ---- */
  header {{ position: relative; padding: 4.5rem 0 3rem; overflow: hidden; }}
  #field {{
    position: absolute; inset: 0; width: 100%; height: 100%;
    z-index: 0; opacity: .85; pointer-events: none;
  }}
  header .wrap {{ position: relative; z-index: 1; }}
  h1 {{
    font-size: clamp(2rem, 5.2vw, 3.1rem); line-height: 1.08; margin: 0 0 .85rem;
    letter-spacing: -.033em; font-weight: 700;
  }}
  h1 em {{
    font-style: normal;
    background: linear-gradient(92deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  .lede {{ font-size: 1.14rem; color: var(--muted); max-width: 43rem; margin: 0 0 1.7rem; }}
  .links a {{
    display: inline-flex; align-items: center; gap: .45rem; margin: 0 .55rem .6rem 0;
    padding: .62rem 1.15rem; border: 1px solid var(--line); border-radius: 9px;
    text-decoration: none; color: var(--ink); font-size: .95rem; background: var(--bg);
    transition: transform .18s, box-shadow .18s, border-color .18s, color .18s;
  }}
  .links a:hover {{ transform: translateY(-2px); border-color: var(--accent);
    color: var(--accent); box-shadow: 0 8px 22px var(--glow); }}
  .links a.primary {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .links a.primary:hover {{ color: #fff; }}

  section {{ padding: 3.25rem 0; border-top: 1px solid var(--line); }}
  h2 {{ font-size: 1.32rem; margin: 0 0 1.2rem; letter-spacing: -.018em; }}
  h2 span {{ color: var(--accent); font-weight: 400; }}

  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(11.5rem, 1fr)); gap: 1rem; }}
  .stat {{
    background: var(--soft); border: 1px solid var(--line); border-radius: 11px;
    padding: 1.15rem 1.2rem; transition: transform .2s, box-shadow .2s;
  }}
  .stat:hover {{ transform: translateY(-3px); box-shadow: 0 10px 26px var(--glow); }}
  .stat b {{
    display: block; font-size: 2rem; line-height: 1.05; color: var(--accent);
    font-variant-numeric: tabular-nums; letter-spacing: -.02em;
  }}
  .stat span {{ display: block; color: var(--muted); font-size: .875rem; margin-top: .35rem; }}

  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 1rem; }}
  .card {{
    display: block; padding: 1.15rem 1.25rem; border: 1px solid var(--line);
    border-radius: 11px; text-decoration: none; color: var(--ink); background: var(--soft);
    transition: transform .2s, box-shadow .2s, border-color .2s;
  }}
  .card:hover {{ transform: translateY(-3px); border-color: var(--accent);
    box-shadow: 0 10px 26px var(--glow); }}
  .card b {{ display: block; margin-bottom: .3rem; }}
  .card span {{ color: var(--muted); font-size: .9rem; }}
  .card.wide {{ grid-column: 1 / -1; }}

  figure {{ margin: 0 0 2.25rem; }}
  figure img {{
    width: 100%; height: auto; border: 1px solid var(--line); border-radius: 10px;
    background: #fff; transition: transform .3s, box-shadow .3s;
  }}
  figure a:hover img {{ transform: scale(1.012); box-shadow: 0 14px 34px var(--glow); }}
  figcaption {{ color: var(--muted); font-size: .9rem; margin-top: .6rem; }}
  figcaption b {{ color: var(--ink); }}
  .turntable {{ max-width: 30rem; margin: 0 auto 2.25rem; }}
  .turntable img {{ background: #fff; }}

  p {{ max-width: 44rem; }}
  code {{
    background: var(--soft); border: 1px solid var(--line); border-radius: 5px;
    padding: .12rem .38rem; font-size: .88em;
  }}
  pre {{
    background: var(--soft); border: 1px solid var(--line); border-radius: 10px;
    padding: 1.1rem; overflow-x: auto; font-size: .87rem; line-height: 1.7;
  }}
  pre code {{ background: none; border: 0; padding: 0; }}
  .muted {{ color: var(--muted); }}
  footer {{ padding: 2.75rem 0 4.5rem; color: var(--muted); font-size: .9rem;
    border-top: 1px solid var(--line); }}
  footer a {{ color: var(--accent); }}

  /* ---- scroll reveal ----
     Hidden only once the script has confirmed it is running. Hiding content in
     the stylesheet and revealing it from JavaScript means any failure of the
     observer leaves the page blank, which is exactly what happened the first
     time this was written. */
  html.js .reveal {{ opacity: 0; transform: translateY(14px);
    transition: opacity .6s ease, transform .6s ease; }}
  html.js .reveal.in {{ opacity: 1; transform: none; }}

  @media (prefers-reduced-motion: reduce) {{
    html {{ scroll-behavior: auto; }}
    * {{ animation: none !important; transition: none !important; }}
    .reveal {{ opacity: 1; transform: none; }}
    #field {{ display: none; }}
  }}
</style>
</head>
<body>

<nav id="nav"><div class="wrap">
  <b>raman<i>sp</i></b>
  <a href="#glance" class="hide-sm">Results</a>
  <a href="#paper">Paper</a>
  <a href="#graph" class="hide-sm">Graph</a>
  <a href="#figures" class="hide-sm">Figures</a>
  <a href="#run" class="hide-sm">Run it</a>
  <span class="spacer"></span>
  <a href="{github}">GitHub</a>
</div></nav>

<header>
  <canvas id="field" aria-hidden="true"></canvas>
  <div class="wrap">
    <h1>Reading the <em>whole corpus</em></h1>
    <p class="lede">An open framework for cross-sample Raman analysis. It reads
    the instrument's undocumented binary container directly, links every
    acquisition in a study into one queryable graph, and represents a
    hyperspectral map as a differentiable field of anisotropic
    three-dimensional ellipsoids fitted on closed-form gradients, in NumPy, on
    a CPU.</p>
    <p class="links">
      <a class="primary" href="papers/preprint.pdf">Read the preprint</a>
      <a href="{github}">Source on GitHub</a>
      <a href="graph.html">Interactive graph</a>
    </p>
  </div>
</header>

<section id="glance"><div class="wrap">
  <h2>At a glance</h2>
  <div class="stats">
{cards}
  </div>
</div></section>

<section id="field-section"><div class="wrap">
  <h2>The representation <span>&middot; a field of ellipsoids</span></h2>
{hero}
</div></section>

<section id="paper"><div class="wrap">
  <h2>The paper</h2>
  <div class="cards">
{papers}
  </div>
</div></section>

<section id="graph"><div class="wrap">
  <h2>The knowledge graph</h2>
  <p>Provenance edges are true by construction and are drawn as a backbone.
  Content edges have to be earned: spectral similarity by mutual
  <i>k</i>-nearest neighbours rather than a fixed threshold, shared unmixing
  endmembers, and disorder-metric proximity that carries the preprocessing
  protocol it was computed under.</p>
  <div class="cards">
{graph}
  </div>
</div></section>

<section id="figures"><div class="wrap">
  <h2>Figures</h2>
{figures}
</div></section>

<section id="run"><div class="wrap">
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
</div></section>

<footer><div class="wrap">
  <p>Siddhartha Pahari and Jainish Shailesh Solanki, University of Toronto.
  Source and issues at <a href="{github}">{github}</a>.
  This page is at <a href="{site}">{site}</a>.</p>
</div></footer>

<script>
(function () {{
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var items = document.querySelectorAll('.reveal');

  function revealAll() {{
    items.forEach(function (el) {{
      el.classList.add('in');
      var n = el.querySelector && el.querySelector('b[data-to]');
      if (n && !n.dataset.done) {{ n.dataset.done = '1'; countUp(n); }}
    }});
  }}

  /* only hide things if the observer is actually available and motion is
     wanted; otherwise leave the page as the stylesheet renders it */
  if (!reduce && 'IntersectionObserver' in window) {{
    document.documentElement.classList.add('js');
  }} else {{
    setTimeout(revealAll, 0);
  }}

  /* failsafe: whatever happens, nothing stays invisible */
  setTimeout(revealAll, 2500);

  /* sticky nav hairline */
  var nav = document.getElementById('nav');
  addEventListener('scroll', function () {{
    nav.classList.toggle('stuck', scrollY > 8);
  }}, {{ passive: true }});

  /* reveal on scroll, and count the stat numbers up once visible */
  var io = new IntersectionObserver(function (entries) {{
    entries.forEach(function (e) {{
      if (!e.isIntersecting) return;
      e.target.classList.add('in');
      var num = e.target.querySelector && e.target.querySelector('b[data-to]');
      if (num && !num.dataset.done) {{ num.dataset.done = '1'; countUp(num); }}
      io.unobserve(e.target);
    }});
  }}, {{ rootMargin: '0px 0px -8% 0px' }});
  items.forEach(function (el) {{ io.observe(el); }});

  function countUp(el) {{
    var target = parseFloat(el.dataset.to), raw = el.textContent;
    /* Never animate in a background tab. requestAnimationFrame is throttled
       there, so the first frame would replace the real number with 0 and leave
       it stranded until the reader focuses the tab. */
    if (reduce || document.hidden || !isFinite(target)) return;
    var suffix = raw.replace(/[0-9.,]/g, ''), dec = (raw.split('.')[1] || '').length ? 1 : 0;
    var fmt = function (v) {{
      return (dec ? v.toFixed(1) : Math.round(v).toLocaleString()) + suffix;
    }};
    var t0 = performance.now(), dur = 1100, done = false;
    function finish() {{ if (!done) {{ done = true; el.textContent = fmt(target); }} }}
    setTimeout(finish, dur + 400);   /* the value always lands, frames or not */
    (function step(t) {{
      if (done) return;
      var k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(target * e);
      if (k < 1) requestAnimationFrame(step); else finish();
    }})(t0);
  }}

  /* hero: anisotropic ellipsoids drifting, which is what the method fits */
  var c = document.getElementById('field');
  if (!c || reduce) return;
  var ctx = c.getContext('2d'), dots = [], raf = null, w = 0, h = 0;
  var accent = getComputedStyle(document.documentElement)
                 .getPropertyValue('--accent').trim() || '#0072B2';
  var accent2 = getComputedStyle(document.documentElement)
                 .getPropertyValue('--accent2').trim() || '#D55E00';

  function resize() {{
    var r = c.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    w = r.width; h = r.height;
    c.width = w * dpr; c.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var n = Math.round(Math.min(46, Math.max(16, w / 26)));
    dots = new Array(n).fill(0).map(function () {{
      var band = Math.random();
      return {{
        x: Math.random() * w, y: Math.random() * h,
        rx: 9 + Math.random() * 30, ry: 3 + Math.random() * 8,
        a: Math.random() * Math.PI, da: (Math.random() - .5) * .0042,
        vx: (Math.random() - .5) * .16, vy: (Math.random() - .5) * .16,
        col: band < .62 ? accent : (band < .8 ? accent2 : '#9aa3ae'),
        al: .05 + Math.random() * .14
      }};
    }});
  }}

  function frame() {{
    ctx.clearRect(0, 0, w, h);
    for (var i = 0; i < dots.length; i++) {{
      var d = dots[i];
      d.x += d.vx; d.y += d.vy; d.a += d.da;
      if (d.x < -60) d.x = w + 60; if (d.x > w + 60) d.x = -60;
      if (d.y < -40) d.y = h + 40; if (d.y > h + 40) d.y = -40;
      ctx.save(); ctx.translate(d.x, d.y); ctx.rotate(d.a);
      ctx.beginPath(); ctx.ellipse(0, 0, d.rx, d.ry, 0, 0, Math.PI * 2);
      ctx.fillStyle = d.col; ctx.globalAlpha = d.al; ctx.fill();
      ctx.restore();
    }}
    raf = requestAnimationFrame(frame);
  }}

  addEventListener('resize', resize, {{ passive: true }});
  document.addEventListener('visibilitychange', function () {{
    if (document.hidden) {{ cancelAnimationFrame(raf); raf = null; }}
    else if (!raf) raf = requestAnimationFrame(frame);
  }});
  resize(); frame();
}})();
</script>

</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

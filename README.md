# ramansp

[Companion site](https://ramanspjs.vercel.app) &middot; [Preprint](https://ramanspjs.vercel.app/papers/preprint.pdf) &middot; [Interactive knowledge graph](https://ramanspjs.vercel.app/graph.html)

A small, reproducible **Raman workflow framework** for a multi-sample study,
built in the spirit of [RamanSPy](https://ramanspy.readthedocs.io)
(Georgiev *et al.*, *Anal. Chem.* 2024) and extended with three things a corpus
of maps needs:

1. **A direct reader for the vendor's binary format**, Horiba LabSpec 6
   `.l6m` / `.l6s`, reverse engineered here and verified bit-identical against
   the instrument's own export. Without it the analysable corpus is only
   whatever somebody remembered to export by hand: 1 map instead of 23.
2. **A cross-sample knowledge graph**, every acquisition in a study, linked by
   spectral similarity, shared unmixing endmembers and disorder-metric
   proximity, with each ratio edge stamped with the preprocessing protocol it
   was computed under.
3. **Spectral 3D Gaussian Splatting**, a hyperspectral map `V(x, y, ν)`
   represented by a differentiable field of anisotropic 3-D ellipsoids, each
   localising a chemical domain in space and a band in wavenumber. Pure NumPy,
   analytic gradients, prune/clone densification. A documented port to a tiled
   CUDA rasteriser (`ramansp/splatting/torch_backend.py`) scales it to millions.

```
ramansp/
  containers.py      Spectrum / SpectralImage / SpectralVolume
  io.py              LabSpec 6 binaries, matrix maps, fit tables, SPC, spectra
  preprocessing.py   Pipeline + named protocols (carbon_dg, arpls, chord, poly3, minimal)
  analysis.py        PCA/NMF/ICA, k-means, VCA+FCLS unmixing, carbon band areas
  metrics.py         PSNR / SSIM / SAM  +  variogram / effective_n / interleave noise
  knowledge_graph.py build_graph / analyse / write_graphml
  splatting/         volume.py · model.py (analytic grads) · fit.py · render.py · torch_backend.py
  datasets.py        Corpus() registry over the anonymised corpus
```

## Install

```bash
pip install -e "CLAUDE RAMAN SP"
pytest "CLAUDE RAMAN SP/ramansp/tests" -q
```

Depends only on `numpy scipy scikit-learn matplotlib networkx pandas pillow`.
No GPU, no PyTorch, no LaTeX required.

## Quick start

```python
import ramansp as rp

img   = rp.io.read_matrix_map("INPUT FILES/.../DOE 8.txt")   # -> SpectralImage
clean = rp.preprocessing.protocol("carbon_dg").apply(img)
field, vol, hist = rp.splatting.fit_image(clean, n_gaussians=1600, iters=170)
dmap  = field.render_band(1350.0, vol)        # D-band image, reconstructed from the splat
```

## The pipeline (`run/`)

| script | what it does | writes to `OUTPUT FILES/` |
|---|---|---|
| `01_build_corpus.py` | read every machine-readable export, **anonymise**, cache | `corpus/`, `_private/sample_key.csv` |
| `02_knowledge_graph.py` | build + analyse the cross-sample graph | `graph/`, `figures/kg_*` |
| `03_splat_fit.py` | fit a Gaussian field to each raw map | `splat/<id>/`, `figures/splat_*` |
| `04_figures.py` | corpus, ablation and reconstruction-quality figures | `figures/` |
| `05_paper_assets.py` | copy figures, emit LaTeX macros/tables | `paper/figures/`, `paper/values.tex` |
| `06_workflow_figure.py` | the pipeline diagram | `figures/fig_workflow.*` |
| `07_ml_benchmark.py` | denoiser comparison + 30-model benchmark | `ml/`, `tables/`, `figures/fig_ml_*` |
| `08_toc_graphic.py` | table-of-contents graphic at the journal's size | `figures/fig_toc.*` |
| `09_acs_submission.py` | assemble the journal submission package | `ACS SUBMISSION/` |
| `10_cover_letter_docx.py` | render the cover letter as Word | `paper/cover_letter.docx` |
| `11_build_site.py` | build the static site published on Vercel | `web/` |
| `check_anonymity.py` | safety gate, fails on any identifier leak | (exit code) |
| `check_figures.py` | artwork gate: column widths, depth, resolution | (exit code) |

`03_splat_fit.py --rescore` recomputes every metric from the saved fields
without refitting, which turns an hour into seconds when a new measure is
added.

```bash
cd "CLAUDE RAMAN SP"
python run/01_build_corpus.py
python run/02_knowledge_graph.py
python run/03_splat_fit.py          # add --quick for a ~90 s pass
python run/04_figures.py
python run/05_paper_assets.py
```

## Anonymisation

`INPUT FILES/` holds real specimen ids, material names, a person's name and
dated folders. `01_build_corpus.py` replaces all of them with opaque codes
(`S01`, `MAT-A`, `CFG-08`, `session-13`, `PERSON-01`, …). The forward map is
written **only** to `OUTPUT FILES/_private/sample_key.csv`, which `.gitignore`
excludes. Every figure, table, manifest and the manuscript use codes only.

Two names pass through **by design**: `glassy_carbon` and `graphite` are public
reference standards, not study identifiers, and naming them is scientifically
useful. That decision lives in `CLASS_OF` in `01_build_corpus.py` and in the
`ALLOWED` set of the gate below, it is an explicit choice, not an oversight.

Verify before sharing anything:

```bash
python run/check_anonymity.py
```

It scans every shareable file under `OUTPUT FILES/` and `paper/` (skipping
`_private/`) for specimen ids, lab material designations, method and person
names, raw `DOE n` numbers and dated sessions, and exits non-zero on a hit, so
it can sit in CI or a pre-publish hook. Two traps it exists to catch, both of
which actually occurred here: a `carbon_class` column that re-introduced
material names the path scrubber had already removed, and a 5-digit specimen
pattern matching inside a hex content digest (fixed by requiring alphanumeric
boundaries).

## Reading the vendor binaries

`io.read_l6` reads Horiba **LabSpec 6** `.l6m` / `.l6s` files directly, no
manual export step, which is what otherwise biases a corpus toward whichever
files someone remembered to export.

The container is a flat table of 24-byte records
`[type][id][tag][ptr_hi][val][extra]`; a record tagged `mat` is followed by a
contiguous little-endian `float32` array. The trap: `ptr_hi` looks like a format
constant but is the high half of a serialised 64-bit Windows pointer, so it
varies per file with ASLR (`0x7ff6`, `0x7ff7`, or 0 in the 32-bit variant), it
is calibrated per file, not hard-coded. The arrays are unlabelled, so they are
identified by constraint: the cube is the largest, the wavenumber axis is the
longest ascending array in a plausible shift range that divides it, and the
stage axes are the evenly-stepped pair with `nx*ny*K == cube.size`. Axis order
is resolved by map isotropy. **If nothing satisfies the constraint the reader
raises rather than guessing.**

Verified against the vendor: for the one map that has both a binary and the
instrument's own text export, the cube is **bit-identical** (766 080 float32
values, zero differences), see `ramansp/tests/test_l6.py`. The binary is in
fact *more* precise, since the text export rounds the axis to 6 significant
figures.

Not yet handled: the older 32-bit container variant, which stores matrices
under a different tag. Those files are almost entirely autosave / practice /
training scratch rather than experiments.

## Scope

Also read: `.txt` matrix maps and line scans, the `.txt` G/D3 curve-fit tables,
single-column spectra, calibration spectra, and evenly-spaced `.spc`.

## The manuscript

The paper exists in three renderings that share one source. The body, the
abstract and the closing statements live in `paper/body.tex`,
`paper/abstract.tex`, `paper/ack.tex`, `paper/dataavail.tex` and
`paper/suppinfo.tex`; each driver adds only its own class and front matter, so
the three cannot say different things.

| driver | build | what it is |
|---|---|---|
| `paper/preprint.tex` | `paper/build_preprint.ps1` | ordinary two-column article, for a preprint server |
| `paper/main.tex` | `paper/build.ps1` | Analytical Chemistry submission format (`achemso`) |
| `paper/si.tex` | `paper/build_si.ps1` | Supporting Information |

Every number in all three comes from `paper/values.tex`, which
`05_paper_assets.py` writes from the pipeline's own outputs. Nothing quoted in
the text is typed by hand, so the prose cannot drift from the artefacts.

## The site

`11_build_site.py` assembles the landing page, the interactive knowledge graph,
every figure at full resolution and the three PDFs from those same outputs. It
is deployed at <https://ramanspjs.vercel.app>.

The site is deliberately **not** part of this repository. It is built into a
standalone folder beside it and deployed on its own, so the Vercel project and
this repository are independent: neither one triggers or depends on the other.

```bash
python run/11_build_site.py      # writes ../RAMANSP SITE
cd "../RAMANSP SITE"
vercel deploy --prod
```

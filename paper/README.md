# Manuscript

`main.tex` + `refs.bib`, styled as a two-column `article`. **No LaTeX toolchain
is assumed on the analysis machine**, build it elsewhere:

## Overleaf
Upload `main.tex`, `refs.bib`, and the `figures/` and `tables/` folders (and
`values.tex`). Set the compiler to pdfLaTeX. Overleaf runs bibtex automatically.

## Local
```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Needs `algorithm`, `algpseudocode`, `authblk`, `siunitx`, `booktabs` (all in a
standard TeX Live).

## Regenerating the assets
`figures/`, `tables/*.tex` and `values.tex` are produced by the pipeline:

```bash
python run/01_build_corpus.py
python run/02_knowledge_graph.py
python run/03_splat_fit.py
python run/04_figures.py
python run/05_paper_assets.py      # copies figures here, writes values.tex
```

`main.tex` compiles without them too, every generated number has a
`\providecommand` placeholder and every `\input` is guarded by `\IfFileExists`.

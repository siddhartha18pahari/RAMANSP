"""Stage 5 -- copy figures into paper/ and emit numbers as LaTeX macros.

    paper/figures/*                every OUTPUT FILES/figures/* and splat GIF
    paper/values.tex              \\newcommand macros for every quoted number
    paper/tables/corpus.tex       corpus composition
    paper/tables/kg.tex           knowledge-graph summary
"""

from __future__ import annotations

import json
import shutil
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import CORPUS, FIGS, GRAPH, OUTPUT, PAPER, SPLAT, TABLES  # noqa: E402


def _num(x, nd=1):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "--"


def main():
    figdir = PAPER / "figures"
    # start clean: a stale figure from an earlier flagship must not survive and
    # be silently included by a \includegraphics that still names it
    if figdir.exists():
        shutil.rmtree(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    for p in list(FIGS.glob("*.png")) + list(SPLAT.rglob("*.gif")) + list(SPLAT.rglob("*.png")):
        shutil.copy(p, figdir / p.name)

    corpus = json.loads((CORPUS / "summary.json").read_text())
    kg = json.loads((GRAPH / "kg_summary.json").read_text()) if (GRAPH / "kg_summary.json").exists() else {}
    man = pd.read_csv(CORPUS / "manifest.csv")

    splat_reps = [json.loads(m.read_text()) for m in SPLAT.glob("*/metrics.json")]
    flagged = [r for r in splat_reps if r.get("flagship")]
    if not flagged:  # metrics.json is per-map; the flag lives in all_metrics.json
        allm = SPLAT / "all_metrics.json"
        if allm.exists():
            ids = {r["acq_id"] for r in json.loads(allm.read_text()) if r.get("flagship")}
            flagged = [r for r in splat_reps if r.get("acq_id") in ids]
    pool = flagged or splat_reps
    flagship = max(pool, key=lambda r: r.get("mask_points", 0), default={})

    # stable aliases so main.tex never hard-codes which acquisition is flagship
    fa = flagship.get("acq_id")
    if fa:
        for src, dst in (("ellipsoids.png", "flagship_ellipsoids.png"),
                         ("bands.png", "flagship_bands.png"),
                         ("spectra.png", "flagship_spectra.png"),
                         ("history.png", "flagship_history.png"),
                         ("turntable.gif", "flagship_turntable.gif")):
            p = SPLAT / fa / src
            if p.exists():
                shutil.copy(p, figdir / dst)

    macros = {
        "Nacq": corpus["n_acquisitions"],
        "Nfittable": int((man.kind == "fit_table").sum()),
        "Nrawmap": int((man.kind == "raw_map").sum()),
        "Nrawscan": int((man.kind == "raw_scan").sum()),
        "Nspecimen": corpus["n_specimens"],
        "Nconfig": corpus["n_configs"],
        "Nskipped": corpus["files_skipped"],
        "Nlsix": corpus.get("l6_binaries_read", "--"),
        "Nlsixfail": corpus.get("l6_binaries_unreadable", "--"),
        "Ndupgroups": len(corpus.get("duplicate_fit_table_groups", [])),
        "KGnodes": kg.get("n_nodes", "--"),
        "KGedges": kg.get("n_edges", "--"),
        "KGcommunities": kg.get("n_communities", "--"),
        "KGmodularity": _num(kg.get("modularity"), 2),
        "KGpurity": _num(kg.get("community_purity_vs_carbon_class"), 2),
        "SplatN": flagship.get("n_gaussians", "--"),
        "SplatParams": flagship.get("n_params", "--"),
        "SplatPSNRtrain": _num(flagship.get("psnr_train_dB")),
        "SplatPSNReval": _num(flagship.get("psnr_eval_dB")),
        "SplatSSIM": _num(flagship.get("ssim_train"), 3),
        "SplatSAM": _num(flagship.get("sam_eval_rad"), 3),
        "SplatCompression": _num(flagship.get("compression_ratio_full",
                                              flagship.get("compression_ratio"))),
        "SplatCompressionBinned": _num(flagship.get("compression_ratio")),
        "SplatSeconds": _num(flagship.get("seconds"), 0),
        "SplatRawValues": flagship.get("raw_cube_values_full",
                                       flagship.get("raw_cube_values", "--")),
        "SplatAcq": flagship.get("acq_id", "--"),
        "SplatMapNx": flagship.get("map_nx", "--"),
        "SplatMapNy": flagship.get("map_ny", "--"),
        "SplatMapK": flagship.get("map_channels", "--"),
        "SplatMaskPts": flagship.get("mask_points", "--"),
        "SplatNfits": len(splat_reps),
        "SplatResidNoise": _num(flagship.get("resid_over_noise"), 2),
        "SplatResidNoiseMin": _num(min((r.get("resid_over_noise", float("inf"))
                                        for r in splat_reps), default=float("nan")), 2),
        "SplatResidNoiseMax": _num(max((r.get("resid_over_noise", float("-inf"))
                                        for r in splat_reps), default=float("nan")), 2),
        "SplatRhoMin": _num(min((r.get("gaussians_per_1k_voxels", float("inf"))
                                 for r in splat_reps), default=float("nan")), 0),
        "SplatRhoMax": _num(max((r.get("gaussians_per_1k_voxels", float("-inf"))
                                 for r in splat_reps), default=float("nan")), 0),
        "SplatComprMin": _num(min((r.get("compression_ratio_full", float("inf"))
                                   for r in splat_reps), default=float("nan"))),
        "SplatComprMax": _num(max((r.get("compression_ratio_full", float("-inf"))
                                   for r in splat_reps), default=float("nan"))),
        "SplatPSNRevalMin": _num(min((r.get("psnr_eval_dB", float("inf"))
                                      for r in splat_reps), default=float("nan"))),
        "SplatPSNRevalMax": _num(max((r.get("psnr_eval_dB", float("-inf"))
                                      for r in splat_reps), default=float("nan"))),
        "Nrawmapfit": len(splat_reps),
    }

    # --- machine-learning benchmarks -------------------------------
    mlp = OUTPUT / "ml" / "summary.json"
    if mlp.exists():
        ml = json.loads(mlp.read_text())
        den = ml.get("denoise") or {}
        cls = ml.get("classification") or []
        if den:
            macros["MLDenoiseN"] = den.get("n_spectra", "n/a")
            macros["MLDenoiseMap"] = den.get("map", "n/a")
            macros["MLRefFilter"] = str(den.get("reference_filter", "n/a")).replace("_", " ")
            tbl = den.get("table", [])
            best = min((r for r in tbl if r["method"] != "input (noisy)"),
                       key=lambda r: r["MSE"], default=None)
            if best:
                macros["MLBestDenoiser"] = best["method"]
                inp = next((r for r in tbl if r["method"] == "input (noisy)"), None)
                if inp:
                    macros["MLDenoiseGain"] = _num(inp["MSE"] / max(best["MSE"], 1e-30), 2)
            rows = "\n".join(
                r"%s & %.3g & %.4f & %.4f \\" % (
                    r["method"].replace("_", " ").replace("&", "+"),
                    r["MSE"], r["SAD"], r["SID"])
                for r in tbl)
            (PAPER / "tables" / "ml_denoise_tex.tex").write_text(
                "\\begin{tabular}{lrrr}\n\\toprule\n"
                "method & MSE & SAD & SID \\\\\n\\midrule\n"
                + rows + "\n\\bottomrule\n\\end{tabular}\n")
        if cls:
            macros["MLNmodels"] = max(len(c.get("top5", [])) for c in cls) and \
                len(pd.read_csv(TABLES / "ml_models.csv").model.unique()) \
                if (TABLES / "ml_models.csv").exists() else "n/a"
            for c in cls:
                tag = "Particle" if "particle" in c["task"] else "Config"
                top = (c.get("top5") or [{}])[0]
                macros[f"ML{tag}Acc"] = _num(100 * top.get("accuracy", float("nan")), 2)
                macros[f"ML{tag}Model"] = str(top.get("model", "n/a")).replace("_", " ")
                macros[f"ML{tag}TrainMaps"] = c.get("n_train_maps", "n/a")
                macros[f"ML{tag}TestMaps"] = c.get("n_test_maps", "n/a")
                macros[f"ML{tag}Classes"] = len(c.get("labels") or []) or "n/a"

    figman = FIGS / "manifest.json"
    if figman.exists():
        ab = (json.loads(figman.read_text()) or {}).get("preproc_ablation_disorder_median")
        if ab:
            macros["AblMin"] = _num(min(ab.values()), 3)
            macros["AblMax"] = _num(max(ab.values()), 3)
            macros["AblSpread"] = _num(
                100 * (max(ab.values()) - min(ab.values())) / max(min(ab.values()), 1e-9), 1)
            macros["AblNproto"] = len(ab)
    (PAPER / "values.tex").write_text(
        "% auto-generated by run/05_paper_assets.py -- do not edit\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    )

    kinds = man.kind.value_counts()
    rows = "\n".join(rf"{k.replace('_', ' ')} & {v} \\" for k, v in kinds.items())
    (PAPER / "tables" / "corpus.tex").write_text(
        "\\begin{tabular}{lr}\n\\toprule\nacquisition kind & count \\\\\n\\midrule\n"
        + rows
        + f"\n\\midrule\ntotal & {len(man)} \\\\\n"
        + "\\bottomrule\n\\end{tabular}\n"
    )

    srows = "\n".join(
        r"%s & %d & %.0f & %.1f & %.1f & %.2f \\" % (
            str(r.get("acq_id", "?")).replace("acq-", "map "),
            r.get("n_gaussians", 0),
            r.get("gaussians_per_1k_voxels", float("nan")),
            r.get("compression_ratio_full", float("nan")),
            r.get("psnr_eval_dB", float("nan")),
            r.get("resid_over_noise", float("nan")),
        )
        for r in sorted(splat_reps, key=lambda r: -r.get("mask_points", 0))
    )
    (PAPER / "tables" / "splat.tex").write_text(
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "map & $N$ & $\\rho$ & compr. & PSNR (dB) & $r/\\sigma$ \\\\\n\\midrule\n"
        + srows + "\n\\bottomrule\n\\end{tabular}\n"
    )

    (PAPER / "tables" / "kg.tex").write_text(
        "\\begin{tabular}{lr}\n\\toprule\nquantity & value \\\\\n\\midrule\n"
        f"nodes & {macros['KGnodes']} \\\\\nedges & {macros['KGedges']} \\\\\n"
        f"communities & {macros['KGcommunities']} \\\\\nmodularity $Q$ & {macros['KGmodularity']} \\\\\n"
        f"community purity vs.\\ class & {macros['KGpurity']} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    print("wrote paper/values.tex and paper/tables/*.tex")
    print({k: v for k, v in macros.items()})


if __name__ == "__main__":
    main()

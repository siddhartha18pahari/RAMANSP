"""Stage 4 -- cross-cutting figures for the manuscript.

    figures/fig_corpus.png            what is in the anonymised corpus
    figures/fig_preproc_ablation.png  disorder index vs preprocessing protocol
                                      on the flagship map (the D/G-is-
                                      preprocessing-dependent result, generalised)
"""

from __future__ import annotations

import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import CORPUS, FIGS, dump_json  # noqa: E402

from ramansp import analysis, preprocessing  # noqa: E402
from ramansp.containers import SpectralImage  # noqa: E402


def fig_corpus():
    man = pd.read_csv(CORPUS / "manifest.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    vc = man.kind.value_counts()
    ax[0].bar(vc.index, vc.values, color="0.4")
    ax[0].set_title(f"anonymised corpus: {len(man)} acquisitions", fontsize=10)
    ax[0].tick_params(axis="x", rotation=20)

    for aid in man[man.kind.isin(["raw_map", "raw_scan", "spectrum"])].acq_id:
        d = np.load(CORPUS / "arrays" / f"{aid}.npz")
        if "mean_spectrum" in d:
            y = d["mean_spectrum"].astype(float)
            y = (y - y.min()) / (np.ptp(y) or 1)
            ax[1].plot(d["wavenumber"], y + 0.0, lw=1.0, label=aid)
    ax[1].set_xlim(1000, 1800)
    ax[1].set_xlabel("Raman shift (cm$^{-1}$)"); ax[1].set_ylabel("norm. intensity")
    ax[1].set_title("mean spectra of the spectral acquisitions", fontsize=10)
    ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(FIGS / "fig_corpus.png", dpi=320); plt.close(fig)


def fig_preproc_ablation():
    man = pd.read_csv(CORPUS / "manifest.csv")
    raw = man[man.kind == "raw_map"].acq_id
    if raw.empty:
        return None
    aid = raw.iloc[0]
    d = np.load(CORPUS / "arrays" / f"{aid}.npz")
    key = "cube_raw" if "cube_raw" in d else "cube"        # crop+despike only
    base = SpectralImage(d[key].astype(float), d["wavenumber"].astype(float), {})
    pmask = d["particle_mask"].ravel() if "particle_mask" in d else None

    names = ["minimal", "arpls", "chord", "poly3", "carbon_dg"]
    meds, iqrs = [], []
    for nm in names:
        steps = preprocessing.protocol(nm).steps
        # drop the leading Crop (already cropped) to avoid re-narrowing
        pipe = preprocessing.Pipeline(*[s for s in steps
                                        if not isinstance(s, preprocessing.Crop)], name=nm)
        di = analysis.disorder_index(pipe.apply(base))
        di = di[pmask] if pmask is not None else di
        di = di[np.isfinite(di)]
        meds.append(np.median(di)); iqrs.append(np.subtract(*np.percentile(di, [75, 25])))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(names, meds, yerr=np.array(iqrs) / 2, color="0.4", capsize=4)
    ax.set_ylabel(r"median disorder index  $A_{D1}/(A_{D1}+A_{G+D2})$")
    ax.set_title(f"preprocessing moves the disorder metric  (map {aid})", fontsize=10)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIGS / "fig_preproc_ablation.png", dpi=320); plt.close(fig)
    return dict(zip(names, [float(x) for x in meds]))


def fig_splat_quality():
    """What actually limits reconstruction: the acquisition's own noise."""
    import json
    from _common import SPLAT

    reps = [json.loads(p.read_text()) for p in SPLAT.glob("*/metrics.json")]
    reps = [r for r in reps if "resid_over_noise" in r]
    if not reps:
        return None
    reps.sort(key=lambda r: r["gaussians_per_1k_voxels"])
    rho = np.array([r["gaussians_per_1k_voxels"] for r in reps])
    psnr = np.array([r["psnr_eval_dB"] for r in reps])
    ratio = np.array([r["resid_over_noise"] for r in reps])
    comp = np.array([r.get("compression_ratio_full", np.nan) for r in reps])
    lab = [r["acq_id"].replace("acq-", "") for r in reps]

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
    ax[0].scatter(rho, psnr, c="crimson", zorder=3)
    for x, y, t in zip(rho, psnr, lab):
        ax[0].annotate(t, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[0].set_xlabel("primitives per 1000 fitted voxels")
    ax[0].set_ylabel("held-out PSNR (dB)")
    ax[0].set_title("apparent quality vs. budget", fontsize=9)

    ax[1].scatter(comp, psnr, c="0.3", zorder=3)
    for x, y, t in zip(comp, psnr, lab):
        ax[1].annotate(t, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel(r"compression ratio ($\times$)")
    ax[1].set_ylabel("held-out PSNR (dB)")
    ax[1].set_title("rate-distortion", fontsize=9)

    order = np.argsort(ratio)
    ax[2].barh(np.arange(len(reps)), ratio[order], color="steelblue")
    ax[2].axvline(1.0, color="crimson", lw=1.5, ls="--")
    ax[2].set_yticks(np.arange(len(reps)))
    ax[2].set_yticklabels([lab[i] for i in order], fontsize=7)
    ax[2].set_xlabel("held-out residual / measurement noise")
    ax[2].set_title("1.0 = at the noise floor", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_splat_quality.png", dpi=320)
    plt.close(fig)
    return {"resid_over_noise": dict(zip(lab, ratio.tolist()))}


def main():
    fig_corpus()
    sq = fig_splat_quality()
    ab = fig_preproc_ablation()
    figs = sorted(p.name for p in FIGS.glob("*.png"))
    dump_json({"figures": figs, "preproc_ablation_disorder_median": ab,
               "splat_quality": sq}, FIGS / "manifest.json")
    print(f"{len(figs)} figures in {FIGS}")
    if ab:
        print("preproc ablation (median disorder):",
              {k: round(v, 3) for k, v in ab.items()})


if __name__ == "__main__":
    main()

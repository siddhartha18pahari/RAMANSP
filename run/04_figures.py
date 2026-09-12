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
    """What the corpus contains, and what its spectra look like.

    The first attempt drew one labelled line per acquisition, which produced a
    60-entry legend covering the axes and a mass of overlapping curves. Neither
    the composition nor the spectral shape survived. Here composition is split
    by how the file was read, and the spectra are summarised by their spread
    with a few exemplars named, which is the quantity a reader can actually use.
    """
    from ramansp._style import OKABE, acs_figsize, apply_style, save
    apply_style()

    man = pd.read_csv(CORPUS / "manifest.csv")
    # fit tables and calibration scans carry no source_format, and a plain
    # groupby silently dropped them, so panel (a) accounted for 60 of the 108
    man["source_format"] = man["source_format"].fillna("derived_text")
    fig, ax = plt.subplots(1, 3, figsize=acs_figsize("double", 2.55),
                           gridspec_kw={"width_ratios": [1.0, 0.62, 1.55]})

    # (a) composition, split by source format
    vc = man.groupby(["kind", "source_format"]).size().unstack(fill_value=0)
    order = vc.sum(1).sort_values(ascending=False).index
    vc = vc.loc[order]
    cols = {"labspec6": OKABE["blue"], "matrix_text": OKABE["orange"],
            "derived_text": "#9e9e9e"}
    bottom = np.zeros(len(vc))
    for c in vc.columns:
        ax[0].bar(range(len(vc)), vc[c], bottom=bottom,
                  color=cols.get(c, "#9e9e9e"),
                  label=("vendor binary" if c == "labspec6"
                         else "text export" if c == "matrix_text"
                         else "derived table"))
        bottom += vc[c].values
    ax[0].set_xticks(range(len(vc)))
    ax[0].set_xticklabels([k.replace("_", " ") for k in vc.index],
                          rotation=25, ha="right")
    ax[0].set_ylabel("acquisitions")
    ax[0].set_title(f"(a) corpus: {int(vc.values.sum())} acquisitions", loc="left")
    ax[0].set_ylim(0, float(vc.sum(1).max()) * 1.30)
    ax[0].legend(loc="upper right", handlelength=1.1, handletextpad=0.5,
                 borderaxespad=0.2)

    # (b) how many maps the two routes deliver
    n_bin = int((man[man.kind == "raw_map"].source_format == "labspec6").sum())
    n_txt = int((man[man.kind == "raw_map"].source_format == "matrix_text").sum())
    ax[1].bar([0, 1], [n_txt, n_txt + n_bin],
              color=["#b0b0b0", OKABE["blue"]], width=0.62)
    for x, v in zip((0, 1), (n_txt, n_txt + n_bin)):
        ax[1].text(x, v + 0.4, str(v), ha="center", fontweight="bold")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["text exports", "+ vendor binaries"], rotation=12)
    ax[1].set_ylabel("hyperspectral maps")
    ax[1].set_ylim(0, (n_txt + n_bin) * 1.25)
    ax[1].set_title("(b) what the reader unlocks", loc="left")

    # (c) spectral envelope rather than spaghetti
    common = np.arange(1000.0, 1800.0, 2.0)
    curves, labels, dis = [], [], []
    for _, row in man[man.kind.isin(["raw_map", "raw_scan", "spectrum"])].iterrows():
        f = CORPUS / "arrays" / f"{row.acq_id}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        if "mean_spectrum" not in d:
            continue
        y = np.interp(common, d["wavenumber"].astype(float),
                      d["mean_spectrum"].astype(float))
        rng_ = np.ptp(y)
        if rng_ <= 0:
            continue
        curves.append((y - y.min()) / rng_)
        labels.append(row.acq_id)
        dis.append(row.get("feat_disorder_median", np.nan))
    if curves:
        C = np.vstack(curves)
        lo, med, hi = np.percentile(C, [10, 50, 90], axis=0)
        ax[2].fill_between(common, lo, hi, color=OKABE["blue"], alpha=0.20,
                           lw=0, label=f"10-90th pct ({len(C)} spectra)")
        ax[2].plot(common, med, color=OKABE["blue"], lw=1.8, label="median")
        # name the extremes of the disorder axis, not whichever files sorted
        # first, so the three exemplars say something about the corpus
        dv = np.array(dis, float)
        rank = np.argsort(np.where(np.isfinite(dv), dv, np.inf))
        rank = rank[np.isfinite(dv[rank])]
        picks = ((rank[0], "lowest disorder"), (rank[len(rank) // 2], "median disorder"),
                 (rank[-1], "highest disorder")) if len(rank) >= 3 else ()
        for (k, tag), c in zip(picks, (OKABE["green"], OKABE["purple"], OKABE["red"])):
            ax[2].plot(common, C[k], lw=1.0, color=c, alpha=0.95,
                       label=f"{labels[k]} ({tag})")
        for x, nm in ((1350, "D"), (1585, "G")):
            ax[2].axvline(x, color="#bbbbbb", lw=0.8, ls=":", zorder=0)
            ax[2].text(x, 1.02, nm, ha="center", fontsize=8, color="#777777")
    ax[2].set_xlim(1000, 1800)
    ax[2].set_xlabel("Raman shift (cm$^{-1}$)")
    ax[2].set_ylabel("normalised intensity")
    ax[2].set_title("(c) spectral envelope of the corpus", loc="left")
    ax[2].set_ylim(-0.04, 1.42)
    ax[2].legend(loc="upper left", ncol=2, handlelength=1.2, handletextpad=0.5,
                 columnspacing=1.0, borderaxespad=0.15)

    fig.tight_layout(pad=0.4, w_pad=1.1)
    save(fig, FIGS / "fig_corpus.png")


def _protocol_pipelines():
    names = ["minimal", "arpls", "chord", "poly3", "carbon_dg"]
    out = {}
    for nm in names:
        steps = preprocessing.protocol(nm).steps
        # the cached cube is already cropped, so dropping the leading Crop
        # avoids narrowing the window a second time
        out[nm] = preprocessing.Pipeline(
            *[st for st in steps if not isinstance(st, preprocessing.Crop)], name=nm)
    return out


def _peak_dg(image):
    """Raw peak-height D/G, the ratio the bounded index is meant to replace."""
    wn, flat = image.wavenumber, image.flat()
    d = (wn >= 1300) & (wn <= 1400)
    g = (wn >= 1540) & (wn <= 1660)
    if d.sum() < 2 or g.sum() < 2:
        return np.full(flat.shape[0], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        gg = flat[:, g].max(1)
        return np.where(np.abs(gg) > 1e-12, flat[:, d].max(1) / gg, np.nan)


def fig_preproc_ablation(max_maps: int = 8, max_spectra: int = 900, seed: int = 0):
    """Is the bounded disorder index stable under the baseline choice?

    The previous version answered this on a single map and titled itself
    "preprocessing moves the disorder metric", which is the opposite of what its
    own five near-identical bars showed, and it started the axis at zero so the
    differences it did have were invisible. The question is a comparison, so it
    is now asked across maps and against the quantity the bounded index exists
    to replace: the raw peak-height D/G ratio. Each map contributes one paired
    line, since what matters is whether a protocol moves a given map, not
    whether maps differ from each other.
    """
    from ramansp._style import OKABE, acs_figsize, apply_style, save
    apply_style()

    man = pd.read_csv(CORPUS / "manifest.csv")
    raw = man[man.kind == "raw_map"]
    if raw.empty:
        return None
    raw = raw.sort_values("feat_n_points").head(max_maps)
    pipes = _protocol_pipelines()
    names = list(pipes)
    rng = np.random.default_rng(seed)

    di_rows, dg_rows, used = [], [], []
    for aid in raw.acq_id:
        f = CORPUS / "arrays" / f"{aid}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        key = "cube_raw" if "cube_raw" in d else "cube"
        cube = d[key].astype(float)
        H, W, _K = cube.shape
        pmask = (d["particle_mask"].ravel() if "particle_mask" in d
                 else np.ones(H * W, bool))
        idx = np.flatnonzero(pmask)
        if idx.size == 0:
            continue
        if idx.size > max_spectra:
            idx = rng.choice(idx, max_spectra, replace=False)
        sub = SpectralImage(cube.reshape(H * W, -1)[idx][None, :, :],
                            d["wavenumber"].astype(float), {})
        di, dg = [], []
        for nm in names:
            proc = pipes[nm].apply(sub)
            v = analysis.disorder_index(proc)
            di.append(np.nanmedian(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan)
            r = _peak_dg(proc)
            dg.append(np.nanmedian(r[np.isfinite(r)]) if np.isfinite(r).any() else np.nan)
        di_rows.append(di); dg_rows.append(dg); used.append(aid)

    if not di_rows:
        return None
    DI = np.array(di_rows, float)
    DG = np.array(dg_rows, float)

    def rel_spread(M):
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100 * (np.nanmax(M, 1) - np.nanmin(M, 1)) / np.abs(np.nanmedian(M, 1))

    sp_di, sp_dg = rel_spread(DI), rel_spread(DG)

    fig, ax = plt.subplots(1, 3, figsize=acs_figsize("double", 2.75),
                           gridspec_kw={"width_ratios": [1.1, 1.1, 0.8]})
    x = np.arange(len(names))
    for a, M, lab, ttl in ((ax[0], DI, "bounded disorder index", "(a) bounded index"),
                           (ax[1], DG, "raw peak-height $I_D/I_G$", "(b) raw ratio")):  # noqa: E501
        for row in M:
            a.plot(x, row, "-o", ms=3.4, lw=1.0, color=OKABE["blue"], alpha=0.55)
        a.set_xticks(x); a.set_xticklabels(names, rotation=18, ha="right")
        a.set_ylabel(lab)
        a.set_title(ttl + f", {len(M)} maps", loc="left", fontsize=8)
    # a shared y-span makes the two panels comparable at a glance
    for a, M in ((ax[0], DI), (ax[1], DG)):
        lo, hi = np.nanmin(M), np.nanmax(M)
        pad = 0.12 * max(hi - lo, 1e-9)
        a.set_ylim(lo - pad, hi + pad)

    ax[2].boxplot([sp_di[np.isfinite(sp_di)], sp_dg[np.isfinite(sp_dg)]],
                  tick_labels=["bounded" + chr(10) + "index", "raw" + chr(10) + "$I_D/I_G$"],
                  widths=0.55)
    ax[2].set_ylabel("spread across protocols (% of map median)")
    # this panel is the narrowest of the three, so its title has to fit about
    # 1.5 in; the previous one ran off the right edge of the figure
    ax[2].set_title("(c) per-map spread" + chr(10) +
                    f"median {np.nanmedian(sp_di):.1f}% vs "
                    f"{np.nanmedian(sp_dg):.1f}%",
                    loc="left", fontsize=8)
    fig.tight_layout(pad=0.4, w_pad=1.1)
    save(fig, FIGS / "fig_preproc_ablation.png")
    return {"protocols": names, "maps": used,
            "disorder_median": {nm: float(v) for nm, v in
                                zip(names, np.nanmedian(DI, axis=0))},
            "spread_pct_bounded_median": float(np.nanmedian(sp_di)),
            "spread_pct_bounded_max": float(np.nanmax(sp_di)),
            "spread_pct_rawdg_median": float(np.nanmedian(sp_dg)),
            "n_maps": len(used)}


def _map_snr(acq_id, _noise_sd=None):
    """Signal spread over measurement noise, both measured on the same array.

    The stored ``noise_sd`` lives in the fit's normalised volume units while the
    cached cube is in its own, so dividing one by the other produced a ratio
    below one and meant nothing. Both terms are therefore computed here from the
    cube: the numerator is the spread of masked values, and the denominator is a
    second-difference estimate along wavenumber, which is what the reconstruction
    report uses and is valid because a band spans tens of channels and noise
    does not.
    """
    f = CORPUS / "arrays" / f"{acq_id}.npz"
    if not f.exists():
        return np.nan
    d = np.load(f)
    if "cube" not in d:
        return np.nan
    cube = d["cube"].astype(float)
    H, W, K = cube.shape
    m = d["particle_mask"].ravel() if "particle_mask" in d else np.ones(H * W, bool)
    v = cube.reshape(H * W, K)[m]
    if v.shape[0] < 2 or K < 5:
        return np.nan
    d2 = np.diff(v, n=2, axis=1)
    noise = float(np.std(d2) / np.sqrt(6.0))     # var(2nd diff) = 6 sigma^2
    return float(np.std(v) / noise) if noise > 0 else np.nan


def fig_splat_quality():
    """What actually limits reconstruction: the acquisition's own noise.

    The middle panel used to repeat the left one with a different x axis, so the
    figure asserted the confound in its caption without ever showing it. It now
    plots held-out PSNR against each map's own signal-to-noise, which is the
    claim itself: the spread in PSNR across maps tracks how noisily each map was
    measured, not how well the field represents it.
    """
    import json
    from _common import SPLAT

    from ramansp._style import OKABE, acs_figsize, apply_style, save
    apply_style()

    reps = [json.loads(p.read_text()) for p in SPLAT.glob("*/metrics.json")]
    reps = [r for r in reps if "resid_over_noise" in r]
    if not reps:
        return None
    reps.sort(key=lambda r: r["gaussians_per_1k_voxels"])
    rho = np.array([r["gaussians_per_1k_voxels"] for r in reps])
    psnr = np.array([r["psnr_eval_dB"] for r in reps])
    ratio = np.array([r["resid_over_noise"] for r in reps])
    comp = np.array([r.get("compression_ratio_full", np.nan) for r in reps])
    snr = np.array([_map_snr(r["acq_id"], r.get("noise_sd")) for r in reps])
    lab = [r["acq_id"].replace("acq-", "") for r in reps]

    fig, ax = plt.subplots(1, 3, figsize=acs_figsize("double", 2.55))

    def annotate(a, xs, ys):
        for x, y, t in zip(xs, ys, lab):
            if np.isfinite(x) and np.isfinite(y):
                a.annotate(t, (x, y), fontsize=6.5, xytext=(4, 4),
                           textcoords="offset points", color="0.35")

    ax[0].scatter(rho, psnr, c=OKABE["blue"], zorder=3, s=44)
    annotate(ax[0], rho, psnr)
    ax[0].set_xlabel("primitives per 1000 fitted voxels")
    ax[0].set_ylabel("held-out PSNR (dB)")
    ax[0].set_title("(a) the naive reading: quality vs budget", fontsize=9, loc="left")

    ok = np.isfinite(snr) & np.isfinite(psnr)
    ax[1].scatter(snr[ok], psnr[ok], c=OKABE["red"], zorder=3, s=44)
    annotate(ax[1], snr, psnr)
    if ok.sum() >= 3:
        rr = float(np.corrcoef(snr[ok], psnr[ok])[0, 1])
        b, a0 = np.polyfit(snr[ok], psnr[ok], 1)
        xs = np.linspace(snr[ok].min(), snr[ok].max(), 20)
        ax[1].plot(xs, a0 + b * xs, color="#999999", lw=1.1, ls="--", zorder=2)
        ax[1].text(0.03, 0.94, f"Pearson r = {rr:.2f}", transform=ax[1].transAxes,
                   fontsize=8.5, va="top", color="0.3")
    ax[1].set_xlabel("signal / noise of the map itself")
    ax[1].set_ylabel("held-out PSNR (dB)")
    ax[1].set_title("(b) PSNR tracks how the map was measured", fontsize=9, loc="left")

    order = np.argsort(ratio)
    cols = [OKABE["blue"] if ratio[i] <= 1.0 else "#c9c9c9" for i in order]
    ax[2].barh(np.arange(len(reps)), ratio[order], color=cols)
    ax[2].axvline(1.0, color=OKABE["red"], lw=1.5, ls="--")
    ax[2].set_yticks(np.arange(len(reps)))
    ax[2].set_yticklabels([lab[i] for i in order], fontsize=7.5)
    ax[2].set_xlabel("held-out residual / measurement noise")
    n_at = int((ratio <= 1.0).sum())
    ax[2].set_title(f"(c) 1.0 is the noise floor; {n_at} of {len(reps)} at or below",
                    fontsize=9, loc="left")
    ax[2].grid(axis="x", alpha=0.35); ax[2].set_axisbelow(True)

    fig.tight_layout(pad=0.4, w_pad=1.1)
    save(fig, FIGS / "fig_splat_quality.png")
    return {"resid_over_noise": dict(zip(lab, ratio.tolist())),
            "snr": dict(zip(lab, snr.tolist())),
            "compression": dict(zip(lab, comp.tolist())),
            "psnr_snr_pearson_r": (float(np.corrcoef(snr[ok], psnr[ok])[0, 1])
                                   if ok.sum() >= 3 else None)}


def main():
    fig_corpus()
    sq = fig_splat_quality()
    ab = fig_preproc_ablation()
    figs = sorted(p.name for p in FIGS.glob("*.png"))
    dump_json({"figures": figs, "preproc_ablation": ab,
               "preproc_ablation_disorder_median": (ab or {}).get("disorder_median"),
               "splat_quality": sq}, FIGS / "manifest.json")
    print(f"{len(figs)} figures in {FIGS}")
    if ab:
        print(f"preproc ablation over {ab['n_maps']} maps: bounded index moves "
              f"{ab['spread_pct_bounded_median']:.1f}% (max "
              f"{ab['spread_pct_bounded_max']:.1f}%), raw D/G moves "
              f"{ab['spread_pct_rawdg_median']:.1f}%")


if __name__ == "__main__":
    main()

"""Stage 3 -- fit a Spectral 3D Gaussian field to a raw hyperspectral map.

    python run/03_splat_fit.py                 # flagship map, default budget
    python run/03_splat_fit.py --acq acq-0012 --gaussians 3000 --iters 220
    python run/03_splat_fit.py --all --quick   # every raw map, small budget

Outputs per map (under ``OUTPUT FILES/splat/<acq_id>/``):
    field.npz              the fitted Gaussian field
    metrics.json          PSNR / SSIM / SAM / compression ratio
    history.png           loss and held-out PSNR vs iteration
    ellipsoids.png        3-D view of the fitted primitives, coloured by band
    bands.png             measured vs splat-reconstructed band maps
    spectra.png           measured vs splat spectra (incl. held-out channels)
    turntable.gif         rotating view of the ellipsoid cloud
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import types

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import CORPUS, FIGS, SPLAT, dump_json  # noqa: E402

from ramansp.containers import SpectralImage  # noqa: E402
from ramansp.splatting import fit_image  # noqa: E402
from ramansp.splatting.fit import SplatConfig, reconstruction_report  # noqa: E402
from ramansp.splatting.volume import build_volume  # noqa: E402
from ramansp.splatting.render import (  # noqa: E402
    band_triptych, ellipsoid_panel, spectra_panel, turntable_gif,
)


def raw_map_table():
    import pandas as pd
    m = pd.read_csv(CORPUS / "manifest.csv")
    m = m[m.kind == "raw_map"].copy()
    # information content actually available to the fit: masked points x channels
    m["budget"] = m.feat_n_points.fillna(0) * m.feat_n_channels.fillna(0)
    return m.sort_values("budget", ascending=False)


def raw_map_acqs():
    return list(raw_map_table().acq_id)


def pick_targets(n: int, max_budget: float = 1.5e6):
    """The n largest maps that still fit in a CPU budget, biggest first.

    Deterministic and recorded, so "the flagship map" means something specific
    rather than "whichever file sorted first".
    """
    t = raw_map_table()
    t = t[t.budget <= max_budget]
    # one map per acquisition config, so the set spans recipes rather than
    # re-fitting the same experiment several times
    t = t.drop_duplicates(subset=["config"], keep="first")
    return list(t.acq_id[:n])


def history_figure(hist, path, acq_id):
    """Loss and held-out PSNR against iteration.

    The gap that opens between the two PSNR curves is the substance of this
    figure, not the convergence: the training channels keep improving while the
    withheld ones stop, which is the optimiser reaching the point where the only
    thing left to fit is that map's own noise. Annotating the plateau makes that
    readable instead of leaving it as two lines.
    """
    from ramansp._style import OKABE, acs_figsize, apply_style, save
    apply_style()

    it = np.asarray(hist.iters, float)
    tr = np.asarray(hist.train_psnr, float)
    ev = np.asarray(hist.eval_psnr, float)

    fig, ax = plt.subplots(1, 2, figsize=acs_figsize("double", 2.5))
    ax[0].plot(it, hist.loss, "-o", ms=3.5, color=OKABE["blue"])
    ax[0].set_yscale("log")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("masked squared error + priors")
    ax[0].set_title("(a) the analytic-gradient fit converges", fontsize=8, loc="left")
    ax[0].grid(alpha=0.35); ax[0].set_axisbelow(True)

    ax[1].plot(it, tr, "-o", ms=3.5, color=OKABE["blue"], label="fitted channels")
    ax[1].plot(it, ev, "-s", ms=3.5, color=OKABE["orange"], label="withheld channels")
    if ev.size >= 3:
        # first iteration within 0.2 dB of the final held-out value
        near = np.flatnonzero(np.abs(ev - ev[-1]) <= 0.2)
        if near.size:
            k = int(near[0])
            ax[1].axvline(it[k], color="#999999", ls=":", lw=1.1)
            ax[1].annotate(f"withheld PSNR flat from iteration {int(it[k])}",
                           xy=(it[k], ev[k]), xytext=(6, -26),
                           textcoords="offset points", fontsize=7.5, color="0.35")
        ax[1].annotate("", xy=(it[-1], tr[-1]), xytext=(it[-1], ev[-1]),
                       arrowprops=dict(arrowstyle="<->", lw=0.9, color="0.45"))
        ax[1].text(it[-1], 0.5 * (tr[-1] + ev[-1]), f"  {tr[-1] - ev[-1]:.1f} dB",
                   fontsize=7.5, va="center", ha="right", color="0.35")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("PSNR (dB)")
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_title("(b) the gap is the noise the field declines to fit",
                    fontsize=8, loc="left")
    ax[1].grid(alpha=0.35); ax[1].set_axisbelow(True)
    fig.suptitle(f"map {acq_id}", fontsize=7, color="0.4", y=1.02)
    fig.tight_layout(pad=0.4, w_pad=1.1)
    save(fig, path)


def fit_one(acq_id: str, cfg: SplatConfig, make_gif: bool = True,
            per_1k_voxels: float = 40.0):
    arr = np.load(CORPUS / "arrays" / f"{acq_id}.npz")
    img = SpectralImage(arr["cube"].astype(float), arr["wavenumber"].astype(float),
                        {"acq_id": acq_id})
    pmask = arr["particle_mask"] if "particle_mask" in arr else None

    cfg = copy.copy(cfg)
    if cfg.auto_gaussians:
        # hold the primitive *density* roughly constant so PSNR is comparable
        # across maps of very different size, rather than penalising big maps
        vox = int((pmask.sum() if pmask is not None else np.prod(img.band_shape))
                  * (img.wavenumber.size / max(cfg.channel_bin, 1)))
        cfg.n_gaussians = int(np.clip(round(per_1k_voxels * vox / 1000), 600, 2600))
        cfg.max_gaussians = int(cfg.n_gaussians * 1.5)

    print(f"[{acq_id}] fitting {cfg.n_gaussians} Gaussians x {cfg.iters} iters "
          f"(bin {cfg.channel_bin}) ...")
    field, vol, hist = fit_image(img, spatial_mask=pmask, config=cfg)
    rep = reconstruction_report(field, vol)
    rep["seconds"] = hist.seconds
    rep["acq_id"] = acq_id
    rep["map_ny"], rep["map_nx"] = int(img.band_shape[0]), int(img.band_shape[1])
    rep["map_channels"] = int(img.wavenumber.size)
    rep["mask_points"] = int(vol.spatial_mask.sum())
    rep["fitted_voxels"] = int(vol.mask3d.sum())
    rep["gaussians_per_1k_voxels"] = round(
        1000 * field.n / max(int(vol.mask3d.sum()), 1), 2)

    outdir = SPLAT / acq_id
    outdir.mkdir(parents=True, exist_ok=True)
    field.save(str(outdir / "field.npz"))
    dump_json(rep, outdir / "metrics.json")

    # the history is cheap to keep and impossible to recover from the field
    # alone, so it is stored and the figure can be redrawn without refitting
    dump_json({"iters": list(map(int, hist.iters)),
               "loss": [float(x) for x in hist.loss],
               "train_psnr": [float(x) for x in hist.train_psnr],
               "eval_psnr": [float(x) for x in hist.eval_psnr]},
              outdir / "history.json")
    history_figure(hist, outdir / "history.png", acq_id)

    # ellipsoids: the 3-D view plus the shape and band-placement statistics
    rep["ellipsoids"] = ellipsoid_panel(field, vol, path=str(outdir / "ellipsoids.png"))
    dump_json(rep, outdir / "metrics.json")

    # measured vs reconstructed band maps, both integrated the same way
    band_triptych(field, vol, path=str(outdir / "bands.png"))

    spectra_panel(field, vol, n=6, path=str(outdir / "spectra.png"))
    if make_gif:
        try:
            turntable_gif(field, vol, str(outdir / "turntable.gif"), n_frames=22)
        except Exception as e:  # noqa: BLE001
            print(f"  (gif skipped: {e})")

    # copy the two hero figures into the shared figures dir
    for name in ("ellipsoids.png", "bands.png"):
        (FIGS / f"splat_{acq_id}_{name}").write_bytes((outdir / name).read_bytes())

    print(f"  PSNR train {rep['psnr_train_dB']:.1f} dB | held-out "
          f"{rep.get('psnr_eval_dB', float('nan')):.1f} dB | "
          f"compression x{rep['compression_ratio']:.1f} | {hist.seconds:.0f} s")
    return rep


def refigure(cfg: SplatConfig):
    """Redraw every figure from the saved fields, without refitting."""
    from ramansp.splatting.model import GaussianField

    for mp in sorted(SPLAT.glob("*/metrics.json")):
        acq = mp.parent.name
        arr = np.load(CORPUS / "arrays" / f"{acq}.npz")
        img = SpectralImage(arr["cube"].astype(float), arr["wavenumber"].astype(float),
                            {"acq_id": acq})
        vol = build_volume(img, spatial_mask=arr.get("particle_mask"),
                           hold_out=cfg.hold_out, channel_bin=cfg.channel_bin)
        field = GaussianField.load(str(mp.parent / "field.npz"))
        rep = json.loads(mp.read_text())
        rep["ellipsoids"] = ellipsoid_panel(field, vol,
                                            path=str(mp.parent / "ellipsoids.png"))
        dump_json(rep, mp)
        hp = mp.parent / "history.json"
        if hp.exists():
            h = json.loads(hp.read_text())
            history_figure(types.SimpleNamespace(**h), mp.parent / "history.png", acq)
        band_triptych(field, vol, path=str(mp.parent / "bands.png"))
        spectra_panel(field, vol, n=6, path=str(mp.parent / "spectra.png"))
        for name in ("ellipsoids.png", "bands.png"):
            (FIGS / f"splat_{acq}_{name}").write_bytes((mp.parent / name).read_bytes())
        print(f"  redrew {acq}")


def rescore(cfg: SplatConfig):
    """Recompute metrics for every saved field -- no refitting.

    The fields are the expensive artefact; the scores are cheap. This keeps
    metrics current when a new measure is added without spending another hour.
    """
    from ramansp.splatting.model import GaussianField

    reps = []
    old = json.loads((SPLAT / "all_metrics.json").read_text()) \
        if (SPLAT / "all_metrics.json").exists() else []
    flags = {r["acq_id"]: r.get("flagship", False) for r in old}
    secs = {r["acq_id"]: r.get("seconds") for r in old}

    for mp in sorted(SPLAT.glob("*/metrics.json")):
        acq = mp.parent.name
        arr = np.load(CORPUS / "arrays" / f"{acq}.npz")
        img = SpectralImage(arr["cube"].astype(float), arr["wavenumber"].astype(float),
                            {"acq_id": acq})
        vol = build_volume(img, spatial_mask=arr.get("particle_mask"),
                           hold_out=cfg.hold_out, channel_bin=cfg.channel_bin)
        field = GaussianField.load(str(mp.parent / "field.npz"))
        rep = reconstruction_report(field, vol)
        rep.update(acq_id=acq, flagship=flags.get(acq, False), seconds=secs.get(acq),
                   map_ny=int(img.band_shape[0]), map_nx=int(img.band_shape[1]),
                   map_channels=int(img.wavenumber.size),
                   mask_points=int(vol.spatial_mask.sum()),
                   fitted_voxels=int(vol.mask3d.sum()),
                   gaussians_per_1k_voxels=round(
                       1000 * field.n / max(int(vol.mask3d.sum()), 1), 2))
        dump_json(rep, mp)
        reps.append(rep)
        print(f"  {acq}  PSNR held-out {rep.get('psnr_eval_dB', float('nan')):.1f} dB | "
              f"resid/noise {rep.get('resid_over_noise', float('nan')):.2f}")
    dump_json(reps, SPLAT / "all_metrics.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acq", default=None, help="acquisition id (default: pick_targets)")
    ap.add_argument("--all", action="store_true", help="fit every raw map")
    ap.add_argument("--top", type=int, default=1,
                    help="fit the N largest maps within budget, one per config")
    ap.add_argument("--gaussians", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--bin", type=int, default=None, help="wavenumber binning factor")
    ap.add_argument("--quick", action="store_true", help="small budget for a fast pass")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute metrics from saved fields, without refitting")
    ap.add_argument("--refigure", action="store_true",
                    help="redraw figures from saved fields, without refitting")
    args = ap.parse_args()

    cfg = SplatConfig()
    if args.quick:
        cfg.n_gaussians, cfg.iters, cfg.channel_bin = 1200, 90, 4
    if args.gaussians:
        cfg.n_gaussians = args.gaussians
    if args.iters:
        cfg.iters = args.iters
    if args.bin:
        cfg.channel_bin = args.bin

    if args.acq:
        targets = [a.strip() for a in args.acq.split(",") if a.strip()]
    elif args.all:
        targets = raw_map_acqs()
    else:
        targets = pick_targets(args.top)
    if not targets:
        print("no raw maps in the corpus -- run run/01_build_corpus.py first")
        return

    if args.rescore:
        rescore(cfg)
        return
    if args.refigure:
        refigure(cfg)
        return

    print(f"targets: {targets}")
    reps = []
    for i, t in enumerate(targets):
        rep = fit_one(t, cfg, make_gif=(not args.no_gif) and i == 0)
        rep["flagship"] = (i == 0)
        reps.append(rep)
    dump_json(reps, SPLAT / "all_metrics.json")


if __name__ == "__main__":
    main()

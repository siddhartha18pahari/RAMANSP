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

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import CORPUS, FIGS, SPLAT, dump_json  # noqa: E402

from ramansp import analysis  # noqa: E402
from ramansp.containers import SpectralImage  # noqa: E402
from ramansp.splatting import fit_image  # noqa: E402
from ramansp.splatting.fit import SplatConfig, reconstruction_report  # noqa: E402
from ramansp.splatting.volume import build_volume  # noqa: E402
from ramansp.splatting.render import (  # noqa: E402
    band_triptych, plot_ellipsoids, spectra_panel, turntable_gif,
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

    # history
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(hist.iters, hist.loss, "-o", ms=3); ax[0].set_yscale("log")
    ax[0].set_xlabel("iteration"); ax[0].set_title("loss", fontsize=9)
    ax[1].plot(hist.iters, hist.train_psnr, "-o", ms=3, label="train channels")
    ax[1].plot(hist.iters, hist.eval_psnr, "-s", ms=3, label="held-out channels")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("PSNR (dB)")
    ax[1].legend(fontsize=8); ax[1].set_title("reconstruction", fontsize=9)
    fig.tight_layout(); fig.savefig(outdir / "history.png", dpi=320); plt.close(fig)

    # ellipsoids
    fig = plt.figure(figsize=(6.5, 5.5))
    axp = fig.add_subplot(111, projection="3d")
    plot_ellipsoids(field, vol, ax=axp)
    fig.savefig(outdir / "ellipsoids.png", dpi=320, bbox_inches="tight"); plt.close(fig)

    # measured vs reconstructed band maps
    clean = SpectralImage(vol.V.transpose(0, 1, 2) * vol.scale, vol.wn_train, {})
    ba = analysis.band_areas(clean)
    H, W = vol.spatial_mask.shape
    meas = {"D1 (1350)": np.where(vol.spatial_mask, ba["D1"].reshape(H, W), np.nan),
            "G+D2 (1600)": np.where(vol.spatial_mask, ba["G+D2"].reshape(H, W), np.nan)}
    band_triptych(field, vol, {"D1": meas["D1 (1350)"], "G+D2": meas["G+D2 (1600)"]},
                  path=str(outdir / "bands.png"))

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

    print(f"targets: {targets}")
    reps = []
    for i, t in enumerate(targets):
        rep = fit_one(t, cfg, make_gif=(not args.no_gif) and i == 0)
        rep["flagship"] = (i == 0)
        reps.append(rep)
    dump_json(reps, SPLAT / "all_metrics.json")


if __name__ == "__main__":
    main()

"""Fit a :class:`GaussianField` to a :class:`~ramansp.containers.SpectralImage`.

Plain Adam on the analytic gradients, with per-group step sizes (centres move
slowly, amplitudes quickly -- the 3DGS convention) and periodic
prune/clone densification. Reconstruction is scored on the held-out channels
that ``build_volume`` set aside.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..analysis import detect_particle
from ..containers import SpectralImage
from ..metrics import psnr, sam, ssim
from .model import GaussianField
from .volume import Volume, build_volume


@dataclass
class SplatConfig:
    n_gaussians: int = 1300
    iters: int = 140
    hold_out: int = 5
    channel_bin: int = 4
    seed: int = 0
    lr_mu: float = 4e-4
    lr_log_s: float = 6e-3
    lr_quat: float = 3e-3
    lr_log_a: float = 3e-2
    lam_a: float = 2e-4
    lam_s: float = 1e-4
    trunc: float = 3.0
    sigma_vox_min: float = 0.45     # primitive width bounds, in voxels
    sigma_vox_max: float = 2.6
    densify_every: int = 40
    densify_until: float = 0.5      # fraction of iters
    max_gaussians: int = 1900
    prune_a_frac: float = 5e-3      # prune a_i below this * median(a)
    use_particle_mask: bool = True
    auto_gaussians: bool = True     # scale n_gaussians with the masked volume
    log_every: int = 20


@dataclass
class FitHistory:
    iters: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    train_psnr: list[float] = field(default_factory=list)
    eval_psnr: list[float] = field(default_factory=list)
    n_gaussians: list[int] = field(default_factory=list)
    seconds: float = 0.0


class _Adam:
    def __init__(self, b1=0.9, b2=0.999, eps=1e-8):
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m: dict = {}
        self.v: dict = {}
        self.t = 0

    def step(self, name, param, grad, lr):
        if name not in self.m or self.m[name].shape != param.shape:
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)
        self.t += 1
        self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * grad
        self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * grad * grad
        mhat = self.m[name] / (1 - self.b1 ** self.t)
        vhat = self.v[name] / (1 - self.b2 ** self.t)
        return param - lr * mhat / (np.sqrt(vhat) + self.eps)


def _report_psnr(field: GaussianField, vol: Volume, which: str) -> float:
    if which == "eval" and vol.V_eval.shape[-1] == 0:
        return float("nan")
    tgt = vol.V if which == "train" else vol.V_eval
    msk = vol.mask3d if which == "train" else vol.mask_eval
    sel = msk.reshape(-1)
    pred = field.render_volume(vol, which).reshape(-1)[sel]
    return psnr(tgt.reshape(-1)[sel], pred, data_range=1.0)


def fit_volume(vol: Volume, cfg: SplatConfig) -> tuple[GaussianField, FitHistory]:
    field = GaussianField.from_volume(
        vol, cfg.n_gaussians, seed=cfg.seed,
        lam_a=cfg.lam_a, lam_s=cfg.lam_s, trunc=cfg.trunc,
    )
    opt = _Adam()
    hist = FitHistory()
    t0 = time.time()
    stop_densify = int(cfg.densify_until * cfg.iters)

    # Bound each primitive's width in *voxels*, not in absolute normalised
    # units: that keeps both the physics (a primitive should not be much wider
    # than the sampling) and the cost (the 3-sigma box stays a fixed number of
    # voxels) identical across maps of very different pixel counts.
    xa, ya, za = field.world_axes
    spacing = np.array([
        float(np.median(np.diff(xa))) if xa.size > 1 else 1.0,
        float(np.median(np.diff(ya))) if ya.size > 1 else 1.0,
        float(np.median(np.diff(za))) if za.size > 1 else 1.0,
    ])
    lo_s = np.log(cfg.sigma_vox_min * spacing)[None, :]
    hi_s = np.log(cfg.sigma_vox_max * spacing)[None, :]
    np.clip(field.log_s, lo_s, hi_s, out=field.log_s)

    for it in range(1, cfg.iters + 1):
        loss, g, aux = field.loss_and_grad(vol)
        if cfg.log_every and (it == 1 or it % cfg.log_every == 0):
            print(f"    it {it:4d}/{cfg.iters}  loss {loss:.3g}  N {field.n}", flush=True)
        field.mu = opt.step("mu", field.mu, g.mu, cfg.lr_mu)
        field.log_s = opt.step("log_s", field.log_s, g.log_s, cfg.lr_log_s)
        field.quat = opt.step("quat", field.quat, g.quat, cfg.lr_quat)
        field.log_a = opt.step("log_a", field.log_a, g.log_a, cfg.lr_log_a)
        np.clip(field.mu, -0.05, 1.05, out=field.mu)
        np.clip(field.log_s, lo_s, hi_s, out=field.log_s)

        if it % cfg.densify_every == 0 and it <= stop_densify:
            a_med = float(np.median(field.amplitude))
            field.densify(aux["pos_grad"], cfg.max_gaussians, seed=cfg.seed + it)
            field.prune(cfg.prune_a_frac * a_med)
            opt.m.clear(); opt.v.clear()

        # log_every == 0 means "never log"; guard the modulo explicitly
        due = cfg.log_every and it % cfg.log_every == 0
        if due or it == 1 or it == cfg.iters:
            hist.iters.append(it)
            hist.loss.append(float(loss))
            hist.train_psnr.append(_report_psnr(field, vol, "train"))
            hist.eval_psnr.append(_report_psnr(field, vol, "eval"))
            hist.n_gaussians.append(field.n)

    hist.seconds = time.time() - t0
    return field, hist


def fit_image(image: SpectralImage, n_gaussians: int = 4000, iters: int = 300,
              spatial_mask: np.ndarray | None = None, config: SplatConfig | None = None,
              **overrides):
    """Convenience wrapper: build the volume (optionally on a particle mask) and fit."""
    cfg = config or SplatConfig(n_gaussians=n_gaussians, iters=iters)
    for k, v in overrides.items():
        setattr(cfg, k, v)

    if spatial_mask is None and cfg.use_particle_mask and image.is_gridded:
        det = detect_particle(image)
        H, W = image.band_shape
        m = det["mask"].reshape(H, W)
        # only use it if it actually selects a sensible sub-region
        spatial_mask = m if 0.03 < m.mean() < 0.97 else None

    vol = build_volume(image, spatial_mask=spatial_mask, hold_out=cfg.hold_out,
                       channel_bin=cfg.channel_bin)
    field, hist = fit_volume(vol, cfg)
    return field, vol, hist


def reconstruction_report(field: GaussianField, vol: Volume) -> dict:
    sel = vol.mask3d.reshape(-1)
    pred_tr = field.render_volume(vol, "train").reshape(-1)[sel]
    tgt_tr = vol.V.reshape(-1)[sel]
    out = {
        "n_gaussians": field.n,
        "n_params": field.n_params(),
        "raw_cube_values": int(vol.spatial_mask.sum() *
                               (vol.wn_train.size + vol.wn_eval.size)),
        "psnr_train_dB": psnr(tgt_tr, pred_tr, data_range=1.0),
        "ssim_train": ssim(tgt_tr, pred_tr, data_range=1.0),
    }
    out["compression_ratio"] = out["raw_cube_values"] / max(out["n_params"], 1)
    if getattr(vol, "orig_K", 0):
        out["raw_cube_values_full"] = int(vol.spatial_mask.sum() * vol.orig_K)
        out["compression_ratio_full"] = out["raw_cube_values_full"] / max(out["n_params"], 1)
    if vol.V_eval.shape[-1]:
        se = vol.mask_eval.reshape(-1)
        full_ev = field.render_volume(vol, "eval")
        pred_ev = full_ev.reshape(-1)[se]
        tgt_ev = vol.V_eval.reshape(-1)[se]
        out["psnr_eval_dB"] = psnr(tgt_ev, pred_ev, data_range=1.0)

        # PSNR against a noisy target confounds the representation with the
        # acquisition. The decisive comparison is the held-out residual against
        # an *independent* estimate of that map's own measurement noise, taken
        # from second differences along wavenumber (a band spans tens of
        # channels; noise does not). A ratio near 1 means the field is at the
        # noise floor and no representation could do better on this data.
        flat = vol.V.reshape(-1, vol.V.shape[-1])[vol.spatial_mask.reshape(-1)]
        noise_sd = float(np.median(np.std(np.diff(flat, 2, axis=1), axis=1) / np.sqrt(6)))
        resid_sd = float((tgt_ev - pred_ev).std())
        out["resid_sd"] = resid_sd
        out["noise_sd"] = noise_sd
        out["resid_over_noise"] = resid_sd / max(noise_sd, 1e-12)
        # SAM over spatial positions, on the held-out channels
        H, W, Ke = vol.V_eval.shape
        sp = vol.spatial_mask.reshape(-1)
        out["sam_eval_rad"] = sam(vol.V_eval.reshape(H * W, Ke)[sp],
                                  full_ev.reshape(H * W, Ke)[sp])
    return out

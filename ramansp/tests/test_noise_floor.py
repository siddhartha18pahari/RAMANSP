"""The field should recover signal, not noise.

This guards the interpretation placed on the corpus results: that a low
held-out PSNR on a noisy acquisition reflects the acquisition's own SNR rather
than a failure of the representation. The decisive test is not the residual
against the noisy target -- it is the reconstruction against the *clean* signal
that target was drawn from.
"""

import numpy as np

from ramansp.containers import SpectralImage
from ramansp.splatting import fit_image
from ramansp.splatting.fit import reconstruction_report
from ramansp.splatting.volume import build_volume

H, W, K = 12, 13, 64
NOISE = 0.10


def _cubes(seed=0):
    rng = np.random.default_rng(seed)
    wn = np.linspace(1000.0, 1800.0, K)
    band = np.exp(-((wn - 1350) / 50) ** 2) + 1.2 * np.exp(-((wn - 1585) / 32) ** 2)
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.exp(-(((xx - 6) / 3.5) ** 2 + ((yy - 6) / 3.5) ** 2))
    clean = blob[..., None] * band
    return wn, clean, clean + NOISE * rng.standard_normal((H, W, K))


def _fit(img):
    return fit_image(img, n_gaussians=400, iters=90, channel_bin=1,
                     use_particle_mask=False, densify_every=10_000,
                     auto_gaussians=False)


def test_field_reconstructs_signal_better_than_the_noisy_data_it_was_given():
    wn, clean, noisy = _cubes()
    field, vol, _h = _fit(SpectralImage(noisy, wn, {}))
    vol_c = build_volume(SpectralImage(clean, wn, {}), hold_out=vol.V_eval.shape[-1] and 5,
                         channel_bin=1)

    # back to original units on the held-out channels
    pred = field.render_volume(vol, "eval") * vol.scale
    truth = vol_c.V_eval * vol_c.scale
    err_vs_clean = float(np.sqrt(np.mean((pred - truth) ** 2)))

    # the data the fit actually saw was this far from the clean signal
    err_of_data = float(np.sqrt(np.mean(
        (vol.V_eval * vol.scale - truth) ** 2)))

    assert err_vs_clean < err_of_data, (
        f"reconstruction {err_vs_clean:.4f} should beat the noisy data "
        f"{err_of_data:.4f} it was fitted to")


def test_report_exposes_noise_floor_fields():
    wn, _clean, noisy = _cubes(seed=1)
    field, vol, _h = _fit(SpectralImage(noisy, wn, {}))
    rep = reconstruction_report(field, vol)
    for k in ("resid_sd", "noise_sd", "resid_over_noise"):
        assert k in rep and np.isfinite(rep[k])
    assert rep["resid_over_noise"] > 0

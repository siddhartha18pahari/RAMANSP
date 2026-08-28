"""Minimal end-to-end example on a synthetic map (no corpus needed).

    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

import ramansp as rp


def synthetic_map(H=18, W=20, K=120, seed=0):
    rng = np.random.default_rng(seed)
    wn = np.linspace(1000.0, 1800.0, K)
    D = np.exp(-((wn - 1350) / 55) ** 2)
    G = np.exp(-((wn - 1585) / 32) ** 2)
    yy, xx = np.mgrid[0:H, 0:W]
    core = np.exp(-(((xx - 10) / 4.0) ** 2 + ((yy - 9) / 4.0) ** 2))     # a "particle"
    dg = 0.3 + 0.5 * core                                               # D/G varies radially
    cube = (core[..., None] * (dg[..., None] * D + (1 - dg[..., None]) * G) * 40
            + 6 + 0.4 * rng.standard_normal((H, W, K)))
    return rp.SpectralImage(cube, wn, {"synthetic": True})


def main():
    img = synthetic_map()
    print("raw:", img.intensities.shape)

    clean = rp.preprocessing.protocol("carbon_dg").apply(img)
    areas = rp.analysis.band_areas(clean)
    print("median D1 area:", float(np.nanmedian(areas["D1"])))

    um = rp.analysis.unmix(clean, n_endmembers=3)
    print("endmembers:", um["endmembers"].shape)

    field, vol, hist = rp.splatting.fit_image(
        clean, n_gaussians=500, iters=80, channel_bin=2, use_particle_mask=False,
        densify_every=1000,
    )
    from ramansp.splatting.fit import reconstruction_report
    rep = reconstruction_report(field, vol)
    print(f"splat: N={field.n}  PSNR(held-out)={rep.get('psnr_eval_dB', float('nan')):.1f} dB  "
          f"compression x{rep['compression_ratio']:.1f}")

    dmap = field.render_band(1350.0, vol)
    print("reconstructed D-band map:", dmap.shape)


if __name__ == "__main__":
    main()

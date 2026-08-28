import numpy as np
import pytest

from ramansp import io
from ramansp.containers import SpectralImage


def _write_matrix_map(path, H=4, W=5, K=60):
    wn = np.linspace(1000.0, 1800.0, K)
    rng = np.random.default_rng(0)
    xs = np.arange(W) * 0.99
    ys = np.arange(H) * 0.99
    lines = ["\t\t" + "\t".join(f"{v:.4f}" for v in wn)]
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            spec = 100 + 20 * np.exp(-((wn - 1350) / 40) ** 2) + rng.normal(0, 1, K)
            lines.append("\t".join([f"{x:.4f}", f"{y:.4f}", *[f"{v:.4f}" for v in spec]]))
    path.write_text("\n".join(lines))
    return wn, H, W


def test_read_matrix_map_roundtrips_grid(tmp_path):
    p = tmp_path / "map.txt"
    wn, H, W = _write_matrix_map(p)
    img = io.read_matrix_map(str(p))
    assert isinstance(img, SpectralImage)
    assert img.is_gridded and img.intensities.shape == (H, W, len(wn))
    np.testing.assert_allclose(img.wavenumber, wn, rtol=1e-4)
    assert img.metadata["nx"] == W and img.metadata["ny"] == H


def test_read_matrix_map_rejects_position_axis(tmp_path):
    p = tmp_path / "bad.txt"
    pos = np.linspace(0, 30, 40)                      # microns, not cm-1
    rows = ["\t\t" + "\t".join(f"{v}" for v in pos)]
    for i in range(6):
        rows.append("\t".join([f"{i}", f"{0}", *[f"{np.sin(v)}" for v in pos]]))
    p.write_text("\n".join(rows))
    with pytest.raises(ValueError):
        io.read_matrix_map(str(p))


def test_read_fit_table(tmp_path):
    p = tmp_path / "fit.txt"
    hdr = "\tG Intensity\tG Width\tG Area\tD3 Intensity\tD3 Width\tD3 Area\tRatio Intensity\tRatio Width\tRatio Area\t"
    rows = [hdr]
    for i in range(1, 11):
        rows.append("\t".join(str(x) for x in
                    [i, 100 + i, 25, 3000, 3500, 5 + i, 200, 900, 1500, 0.2]))
    p.write_text("\n".join(rows))
    t = io.read_fit_table(str(p))
    assert t["params"].shape == (10, 9)
    assert t["contiguous"] is True
    qc = io.fit_table_qc(t)
    assert qc["g_ok"].all() and qc["ratio_ok"].all()

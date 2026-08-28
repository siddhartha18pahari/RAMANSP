"""The LabSpec 6 reader, checked against the vendor's own text export.

These tests are skipped when the corpus is not present, so the suite still runs
on a clean checkout.
"""

import os

import numpy as np
import pytest

from ramansp import io
from ramansp.containers import SpectralImage

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "INPUT FILES")
PAIR = os.path.join(ROOT, "Results", "31 JULY 2026", "DOE 8")
L6M = os.path.join(PAIR, "SINGLE MAPPING 31 JULY ATE 34069.l6m")
TXT = os.path.join(PAIR, "DOE 8.txt")

needs_corpus = pytest.mark.skipif(
    not (os.path.exists(L6M) and os.path.exists(TXT)),
    reason="instrument corpus not present",
)


@needs_corpus
def test_l6m_matches_vendor_text_export_exactly():
    """The strongest available check: same map, two independent routes."""
    txt = io.read_matrix_map(TXT)
    binr = io.read_l6(L6M)
    assert isinstance(binr, SpectralImage)
    assert binr.intensities.shape == txt.intensities.shape
    # the export is written from float32, so this is exact, not approximate
    assert np.array_equal(
        binr.intensities.astype(np.float32), txt.intensities.astype(np.float32)
    )
    # The text export rounds the axis to 6 significant figures (1002.79); the
    # binary carries the full float32 (1002.793396). The binary is the more
    # precise of the two, so this compares at the export's precision.
    np.testing.assert_allclose(binr.wavenumber, txt.wavenumber, rtol=2e-5)
    assert (binr.metadata["nx"], binr.metadata["ny"]) == (
        txt.metadata["nx"], txt.metadata["ny"])
    np.testing.assert_allclose(binr.metadata["extent"], txt.metadata["extent"], rtol=1e-5)


@needs_corpus
def test_l6m_recovers_carbon_bands():
    """Physics, not just self-consistency: D and G land where they must."""
    from ramansp import preprocessing

    img = io.read_l6(L6M)
    mean = img.intensities.reshape(-1, img.wavenumber.size).mean(0)
    one = SpectralImage(mean[None, :], img.wavenumber, {})
    clean = preprocessing.Pipeline(
        preprocessing.Crop(1000, 1800), preprocessing.BaselineARPLS()
    ).apply(one)
    wn, y = clean.wavenumber, clean.flat()[0]
    d = wn[(wn > 1300) & (wn < 1400)][np.argmax(y[(wn > 1300) & (wn < 1400)])]
    g = wn[(wn > 1560) & (wn < 1620)][np.argmax(y[(wn > 1560) & (wn < 1620)])]
    assert 1320 < d < 1390, f"D band at {d}"
    assert 1565 < g < 1600, f"G band at {g}"


def test_rejects_non_labspec_file(tmp_path):
    p = tmp_path / "nope.l6m"
    p.write_bytes(b"NOTLABSPEC" + b"\x00" * 100)
    with pytest.raises(ValueError, match="not a LabSpec 6 file"):
        io.read_l6(str(p))


def test_refuses_to_guess_when_inconsistent(tmp_path):
    """A file with the magic but no usable arrays must raise, not invent a cube."""
    p = tmp_path / "empty.l6m"
    p.write_bytes(b"LabSpec6" + b"\x00" * 512)
    with pytest.raises(ValueError):
        io.read_l6(str(p))

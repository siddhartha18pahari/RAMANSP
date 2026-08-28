"""Readers for the machine-readable Raman exports in this corpus.

Supported now
-------------
``read_matrix_map``   LabSpec "export map as matrix": row 1 is the spectral
                      axis, every later row is ``[x, (y,) intensities...]``.
                      Covers both the 2-coordinate map export (``DOE 8.txt``)
                      and 1-coordinate line scans.
``read_fit_table``    the 9-column G/D3 curve-fit result export (no spectra,
                      no coordinates, 2 of 5 carbon bands) -- adapted from the
                      prior project's ``raman_hyperspectral.load_fit_table``.
``read_single``       a two-column ``[axis, intensity]`` spectrum.
``read_spc``          minimal Galactic SPC (new LSB format, evenly spaced).
``read_quickmap``     a single-channel intensity image exported as a matrix.
``read_l6``           the proprietary LabSpec 6 binary container (``.l6m`` maps
                      and ``.l6s`` spectra), read directly -- no vendor export
                      step. Verified byte-exact against the vendor's own text
                      export; see the format note above ``_L6_MAGIC``.
"""

from __future__ import annotations

import bisect
import struct
import warnings

import numpy as np

from .containers import SpectralImage, Spectrum

# A value axis is accepted as "spectral" only inside this envelope. Rayleigh-
# adjacent starts (a few cm-1) are fine; positions in microns are not.
_SHIFT_MIN, _SHIFT_MAX = -300.0, 4500.0


def _read_header_floats(path: str) -> list[float]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline().rstrip("\n")
    out = []
    for tok in first.split("\t"):
        tok = tok.strip()
        if tok == "":
            continue
        try:
            out.append(float(tok))
        except ValueError:
            return []  # a labelled header (fit table) -- not a matrix map
    return out


def _plausible_spectral_axis(a: np.ndarray) -> bool:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size < 16:
        return False
    da = np.diff(a)
    monotonic = bool(np.all(da > 0) or np.all(da < 0))
    step = float(np.median(np.abs(da)))
    span = float(a.max() - a.min())
    return (
        monotonic
        and span > 100.0
        and 0.05 < step < 60.0
        and a.min() > _SHIFT_MIN
        and a.max() < _SHIFT_MAX
    )


def read_matrix_map(path: str) -> SpectralImage:
    """Read a LabSpec matrix-format map/line-scan into a :class:`SpectralImage`.

    Layout: line 1 holds the wavenumber axis (with 1-2 leading empty cells);
    each subsequent line is ``x [y] i0 i1 ... i_{K-1}``. The number of leading
    coordinate columns is inferred as ``n_cols - K``.
    """
    axis = np.asarray(_read_header_floats(path), float)
    if axis.size == 0:
        raise ValueError(f"{path}: line 1 is not a numeric spectral axis")
    body = np.loadtxt(path, delimiter="\t", skiprows=1, ndmin=2)
    k = axis.size
    n_coord = body.shape[1] - k
    if n_coord not in (1, 2):
        raise ValueError(
            f"{path}: header has {k} channels but rows carry {body.shape[1]} "
            f"columns ({n_coord} coordinate columns is not 1 or 2)"
        )
    if not _plausible_spectral_axis(axis):
        raise ValueError(
            f"{path}: line-1 axis (span {axis.max() - axis.min():.0f}, "
            f"step {np.median(np.abs(np.diff(axis))):.2f}) is not a plausible "
            "Raman-shift axis; refusing to guess"
        )

    coords = body[:, :n_coord]
    intensity = body[:, n_coord:]
    meta: dict = {"source": path, "layout": "matrix_map", "n_coord_cols": n_coord}

    if n_coord == 2:
        x, y = coords[:, 0], coords[:, 1]
        ux, uy = np.unique(x), np.unique(y)
        if ux.size * uy.size == x.size and ux.size > 1 and uy.size > 1:
            xi = np.searchsorted(ux, x)
            yi = np.searchsorted(uy, y)
            grid = np.full((uy.size, ux.size, axis.size), np.nan)
            seen = np.zeros((uy.size, ux.size), bool)
            grid[yi, xi] = intensity
            seen[yi, xi] = True
            if seen.all():                       # every cell filled exactly once
                meta.update(
                    x=x, y=y, nx=int(ux.size), ny=int(uy.size),
                    step_x=float(np.diff(ux).mean()), step_y=float(np.diff(uy).mean()),
                    extent=(float(ux[0]), float(ux[-1]), float(uy[0]), float(uy[-1])),
                )
                return SpectralImage(grid, axis, meta)
        warnings.warn(f"{path}: coordinates are not a clean raster; kept as (N, K) bag")
        meta.update(x=x, y=y)
    else:
        meta["position"] = coords[:, 0]

    return SpectralImage(intensity, axis, meta)


# --- fit-result table ----------------------------------------------------
FIT_COLUMNS = ["G_I", "G_W", "G_A", "D3_I", "D3_W", "D3_A", "R_I", "R_W", "R_A"]
FIT_BOUNDS = {"G_W": (10.0, 35.0), "D3_W": (40.0, 200.0)}


def read_fit_table(path: str) -> dict:
    """Read the tab-separated 5-band (G/D3) fit export.

    Returns ``{'params': (N, 9) float, 'columns': FIT_COLUMNS, 'contiguous':
    bool, 'source': path}``. Adapted from the prior project's
    ``raman_hyperspectral.load_fit_table``; the contiguity requirement is
    relaxed to a flag so partial exports still load.
    """
    raw = np.genfromtxt(path, delimiter="\t", skip_header=1)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] < 10:
        raise ValueError(f"{path}: expected >= 10 columns, got {raw.shape}")
    idx = raw[:, 0]
    params = raw[:, 1:10].astype(float)
    good = np.isfinite(params).all(1)
    params = params[good]
    contiguous = bool(
        np.array_equal(idx[good], np.arange(1, good.sum() + 1))
    )
    return {
        "params": params,
        "columns": list(FIT_COLUMNS),
        "contiguous": contiguous,
        "n_points": int(params.shape[0]),
        "source": path,
    }


def fit_table_qc(table: dict) -> dict:
    """Per-point masks: which bands the fit actually delivered, and clamping.

    Adapted from ``raman_hyperspectral.qc_masks``.
    """
    p = {c: table["params"][:, i] for i, c in enumerate(table["columns"])}
    g_ok = p["G_I"] > 0
    d3_ok = p["D3_I"] > 0
    band_ok = {"G_W": g_ok, "D3_W": d3_ok}
    pinned = np.zeros(len(g_ok), bool)
    for name, (lo, hi) in FIT_BOUNDS.items():
        pinned |= ((p[name] == lo) | (p[name] == hi)) & band_ok[name]
    return {"g_ok": g_ok, "d3_ok": d3_ok, "ratio_ok": g_ok & d3_ok, "pinned": pinned}


# --- simple spectra ---------------------------------------------------
def read_single(path: str, kind: str = "spectrum") -> Spectrum:
    """Two-column ``[axis, intensity]`` file -> :class:`Spectrum`."""
    arr = np.loadtxt(path, ndmin=2)
    if arr.shape[1] < 2:
        raise ValueError(f"{path}: need >= 2 columns")
    axis, inten = arr[:, 0], arr[:, 1]
    order = np.argsort(axis)
    return Spectrum(inten[order], axis[order], {"source": path, "kind": kind})


def read_autocal(path: str) -> Spectrum:
    """An AutoCalibration lamp spectrum -- provenance / instrument QC only."""
    return read_single(path, kind="autocalibration")


def read_quickmap(path: str) -> np.ndarray:
    """A single-channel intensity image exported as a value matrix.

    Line 1 = x positions, column 0 = y positions, the rest = the image. Used
    only for a thumbnail; returns the 2-D array.
    """
    body = np.loadtxt(path, delimiter="\t", skiprows=1, ndmin=2)
    return body[:, 1:]


def read_spc(path: str) -> Spectrum:
    """Minimal Galactic SPC reader: new-format (fversn 0x4B), evenly spaced,
    single subfile, Y as 32-bit float. Raises on anything outside that.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < 512:
        raise ValueError(f"{path}: too short for an SPC header")
    ftflgs, fversn = raw[0], raw[1]
    if fversn != 0x4B:
        raise NotImplementedError(f"{path}: SPC fversn 0x{fversn:02X} not supported")
    # new-format main header (LSB): see the Thermo Galactic SPC spec
    fnpts = struct.unpack_from("<i", raw, 4)[0]
    ffirst = struct.unpack_from("<d", raw, 8)[0]
    flast = struct.unpack_from("<d", raw, 16)[0]
    if fnpts <= 1:
        raise ValueError(f"{path}: fnpts={fnpts}")
    if ftflgs & 0x80:  # TXYXYS / non-even -- not handled
        raise NotImplementedError(f"{path}: uneven-spacing SPC not supported")
    axis = np.linspace(ffirst, flast, fnpts)
    # main header is 512 bytes; subheader is 32 bytes; then fnpts float32 Y
    y0 = 512 + 32
    y = np.frombuffer(raw, dtype="<f4", count=fnpts, offset=y0).astype(float)
    return Spectrum(y, axis, {"source": path, "kind": "spectrum", "format": "spc"})


# --- proprietary LabSpec 6 binary --------------------------------------
#
# Format, reverse-engineered from this corpus and verified byte-exact against
# the vendor's own text export (see ``tests/test_l6.py``):
#
#   bytes 0..7            ASCII magic ``LabSpec6``
#   then a flat table of 24-byte records:
#       [u32 type][u32 id][4-byte tag][u32 ptr_hi][u32 val][u32 extra]
#   ``ptr_hi`` is the high half of a serialised 64-bit Windows pointer, so it
#   is a per-file constant (0x00007ff6, 0x00007ff7, ...) set by ASLR at save
#   time -- NOT a format constant. It is calibrated per file from the bytes
#   following a known tag, then used to walk the record table.
#   Tags seen: ``mat`` (matrix), ``typ``/``typm``, ``uni`` (unit), ``axi``
#   (axis), ``fil`` (file), ``tem`` (template). The byte preceding a tag's
#   ASCII is a type/hash byte and varies.
#   A ``mat`` record is followed -- at record start + 24 -- by a contiguous
#   little-endian float32 array running up to the next record.
#
# A map file carries several such arrays: the intensity cube (largest), the
# wavenumber axis, and the two stage-coordinate axes. They are identified by
# the self-consistency requirement ``nx * ny * n_channels == cube.size``, which
# is a strong enough constraint to make the parse checkable rather than
# guessed; ``read_l6`` raises if it cannot be satisfied.
_L6_MAGIC = b"LabSpec6"
_L6_MAT = b"tam"
_L6_TAGS = (b"tam", b"typm", b"uni", b"axi", b"temx")
_L6_RECORD = 24


def _l6_pointer_word(raw: bytes) -> bytes | None:
    """Per-file high half of the serialised pointers, from known tag sites."""
    seen: dict[bytes, int] = {}
    for tag in _L6_TAGS:
        start = 0
        while True:
            t = raw.find(tag, start)
            if t < 0:
                break
            start = t + 1
            w = raw[t + 3:t + 7]
            if len(w) == 4 and w[2:] == b"\x00\x00" and w != b"\x00\x00\x00\x00":
                seen[w] = seen.get(w, 0) + 1
    if not seen:
        return None
    return max(seen, key=seen.get)


def _l6_records(raw: bytes, word: bytes) -> list[int]:
    """Offsets of every 24-byte record, found via the per-file pointer word."""
    out, start = [], 0
    while True:
        i = raw.find(word, start)
        if i < 0:
            break
        if i >= 12:
            out.append(i - 12)
        start = i + 1
    return out


def _l6_arrays(raw: bytes) -> list[tuple[int, np.ndarray]]:
    """Every float32 array in the file, as ``(offset, values)``."""
    word = _l6_pointer_word(raw)
    if word is None:
        return []
    recs = _l6_records(raw, word)
    rec_set = sorted(recs)
    out = []
    for p in recs:
        if raw[p + 9:p + 12] != _L6_MAT:
            continue
        data = p + _L6_RECORD
        j = bisect.bisect_right(rec_set, data)
        end = rec_set[j] if j < len(rec_set) else len(raw)
        n = (end - data) // 4
        if n >= 2:
            out.append((data, np.frombuffer(raw, dtype="<f4", count=n, offset=data)))
    return out


def _l6_is_axis(a: np.ndarray) -> bool:
    return bool(
        a.size >= 2
        and np.all(np.isfinite(a))
        and np.all(np.diff(a) > 0)
        and np.all(np.abs(a) < 1e6)
    )


def _l6_is_stage_axis(a: np.ndarray, tol: float = 0.05) -> bool:
    """A stage axis is monotonic AND evenly stepped -- a raster, not a curve.

    Without this, any long ascending run inside the intensity block can be
    mistaken for a spatial axis and the cube silently reshapes wrong.
    """
    if not _l6_is_axis(a):
        return False
    d = np.diff(a)
    return bool(d.mean() > 0 and d.std() <= tol * abs(d.mean()))


def _l6_anisotropy(cube: np.ndarray) -> float:
    """Mean |vertical difference| / |horizontal difference| of a band image.

    A map read at its true row length is an isotropic 2-D field; read at the
    wrong one, "vertical" neighbours are unrelated and their difference
    inflates. Same test the prior project used to recover an unlabelled grid.
    """
    img = cube.mean(axis=2)
    dv = np.abs(np.diff(img, axis=0)).mean()
    dh = np.abs(np.diff(img, axis=1)).mean()
    return float(dv / dh) if dh > 0 else float("inf")


def read_l6(path: str):
    """Read a LabSpec 6 binary (``.l6m`` map / ``.l6s`` spectrum) directly.

    Returns a :class:`~ramansp.containers.SpectralImage` for a map and a
    :class:`~ramansp.containers.Spectrum` for a single spectrum. Raises
    ``ValueError`` if the arrays found cannot be reconciled into a consistent
    cube -- this reader refuses to guess.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw.startswith(_L6_MAGIC):
        raise ValueError(f"{path}: not a LabSpec 6 file (magic {raw[:8]!r})")

    arrays = _l6_arrays(raw)
    if not arrays:
        raise ValueError(f"{path}: no matrix records found")

    cube_off, payload = max(arrays, key=lambda t: t[1].size)
    if payload.size < 8 or not np.all(np.isfinite(payload)):
        raise ValueError(f"{path}: largest matrix is not usable spectral data")

    axes = [(o, a) for o, a in arrays if o != cube_off and _l6_is_axis(a)]
    # spectral axis: the longest plausible Raman-shift axis that divides the cube
    spec = [(o, a) for o, a in axes
            if a.size >= 16 and _SHIFT_MIN < a[0] and a[-1] < _SHIFT_MAX
            and payload.size % a.size == 0]
    if not spec:
        raise ValueError(f"{path}: no wavenumber axis consistent with the payload")
    wn_off, wn = max(spec, key=lambda t: t[1].size)
    K = wn.size
    n_points = payload.size // K

    meta = {"source": path, "layout": "labspec6", "format": path.rsplit(".", 1)[-1]}

    if n_points == 1:
        return Spectrum(payload.astype(float), wn.astype(float), meta)

    # spatial axes: a pair of evenly-stepped stage axes multiplying to n_points
    cands = {}
    for o, a in axes:
        if o != wn_off and a.size <= n_points and _l6_is_stage_axis(a):
            cands.setdefault(a.size, a)
    # Which axis is the outer (slow) one is not labelled, and neither is which
    # pair is right when several fit; score every option by map isotropy -- the
    # wrong row length scrambles rows and inflates vertical differences.
    best = None
    for na, aa in sorted(cands.items()):
        for nb, bb in sorted(cands.items()):
            if na * nb != n_points:
                continue
            cube = payload.reshape(na, nb, K).transpose(1, 0, 2)
            score = abs(np.log(max(_l6_anisotropy(cube), 1e-9)))
            if best is None or score < best[0]:
                best = (score, cube, aa, bb)
    if best is None:
        raise ValueError(
            f"{path}: found {n_points} spectra x {K} channels but no pair of "
            f"evenly-stepped stage axes multiplying to {n_points} "
            f"(have {sorted(cands)})"
        )
    _score, cube, x_ax, y_ax = best

    meta.update(
        x=np.repeat(x_ax, y_ax.size).astype(float),
        y=np.tile(y_ax, x_ax.size).astype(float),
        nx=int(x_ax.size), ny=int(y_ax.size),
        step_x=float(np.diff(x_ax).mean()), step_y=float(np.diff(y_ax).mean()),
        extent=(float(x_ax[0]), float(x_ax[-1]), float(y_ax[0]), float(y_ax[-1])),
    )
    return SpectralImage(cube.astype(float), wn.astype(float), meta)


def load_l6m(path: str):
    """Backwards-compatible alias for :func:`read_l6`."""
    return read_l6(path)

"""A tiny registry over the anonymised corpus built by ``run/01_build_corpus.py``.

RamanSPy ships curated datasets behind ``.load()``; here the "dataset" is the
local corpus of exported acquisitions, already stripped of identifiers.

>>> ds = ramansp.datasets.Corpus("OUTPUT FILES/corpus")
>>> ds.manifest.head()
>>> img = ds.load("acq-0007")          # -> SpectralImage  (for raw maps)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import io as _io
from .containers import SpectralImage, Spectrum


class Corpus:
    def __init__(self, root: str):
        self.root = root
        mpath = os.path.join(root, "manifest.csv")
        if not os.path.exists(mpath):
            raise FileNotFoundError(f"{mpath} -- run run/01_build_corpus.py first")
        self.manifest = pd.read_csv(mpath)

    def list(self, kind: str | None = None):
        m = self.manifest
        return list(m[m.kind == kind].acq_id if kind else m.acq_id)

    def record(self, acq_id: str) -> dict:
        row = self.manifest.set_index("acq_id").loc[acq_id]
        return row.to_dict() | {"acq_id": acq_id}

    def load(self, acq_id: str):
        row = self.record(acq_id)
        npz = os.path.join(self.root, "arrays", f"{acq_id}.npz")
        if os.path.exists(npz):
            d = np.load(npz, allow_pickle=True)
            if row["kind"] == "raw_map" and "cube" in d:
                return SpectralImage(d["cube"], d["wavenumber"], {"acq_id": acq_id})
            if "mean_spectrum" in d:
                return Spectrum(d["mean_spectrum"], d["wavenumber"], {"acq_id": acq_id})
        # fall back to the (already anonymised path) source pointer if present
        src = row.get("source_rel")
        raise FileNotFoundError(f"no arrays cached for {acq_id} (source {src})")

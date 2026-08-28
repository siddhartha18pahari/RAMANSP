"""Shared paths and helpers for the run scripts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "INPUT FILES"
OUTPUT = ROOT / "OUTPUT FILES"
CORPUS = OUTPUT / "corpus"
ARRAYS = CORPUS / "arrays"
FIGS = OUTPUT / "figures"
GRAPH = OUTPUT / "graph"
TABLES = OUTPUT / "tables"
SPLAT = OUTPUT / "splat"
PRIVATE = OUTPUT / "_private"
PAPER = ROOT / "paper"

for d in (OUTPUT, CORPUS, ARRAYS, FIGS, GRAPH, TABLES, SPLAT, PRIVATE):
    d.mkdir(parents=True, exist_ok=True)


def dump_json(obj, path: Path) -> None:
    def default(o):
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=default))


def rel_to_input(p: Path) -> str:
    return str(p.relative_to(INPUT)).replace("\\", "/")

"""Safety gate: no study identifier may appear outside ``_private/``.

Run after the pipeline, before sharing anything. Exits non-zero on a leak, so
it can sit in CI or a pre-publish hook.

    python run/check_anonymity.py

Two names are allowed through by design: ``glassy_carbon`` and ``graphite``
are public reference standards, not study identifiers (see ``CLASS_OF`` in
01_build_corpus.py). Everything else -- specimen ids, the lab's own material
designations, person names, dated sessions, raw DOE numbers -- must be opaque.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN = [ROOT / "OUTPUT FILES", ROOT / "paper"]
SUFFIXES = {".csv", ".json", ".tex", ".graphml", ".md", ".txt", ".bib"}

ALLOWED = {"glassy_carbon", "graphite"}

PATTERNS = {
    # alphanumeric boundaries, not just digit ones: a bare 5-digit id must not
    # match inside a hex content digest such as "f1a27d34676b"
    "specimen id": r"ATE[\s_-]?\d{4,6}|(?<![0-9A-Za-z])34\d{3}(?![0-9A-Za-z])",
    "material name": r"\bGQS\b|\bQ[\s_-]?CARBON\b",
    "method name": r"\bSWIFT\b",
    "person": r"\bpeng\b",
    "reference material": r"glass[y]?\s*carbon|\bgraphite\b",
    "raw DOE number": r"\bDOE[\s_-]?\d",
    # a code that spells out the initials of what it replaces is not opaque
    "non-opaque code": r"\bMAT-(?!A\b|B\b)[A-Z]{2,}\b|\bMETH-(?!A\b)[A-Z]{2,}\b",
    "dated session": (r"\b\d{1,2}\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
                      r"[A-Z]*\s*\d{0,4}\b"),
}

# A bibliography is third-party citation metadata, not study output. It
# legitimately contains hundreds of author surnames and material words, so a
# bare-surname or "graphite" rule fires there on cited authors and titles only.
# False alarms are not harmless: they train the reader to ignore the gate.
# Structural identifiers are still checked in the bibliography, because a real
# leak would take that form.
PER_FILE_EXEMPT = {
    "refs.bib": {"person", "reference material"},
}


def main() -> int:
    leaks: list[tuple[str, str, int, str]] = []
    n_files = 0
    for root in SCAN:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in SUFFIXES:
                continue
            if "_private" in p.parts:
                continue
            n_files += 1
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            exempt = PER_FILE_EXEMPT.get(p.name, set())
            for line_no, line in enumerate(text.splitlines(), 1):
                for kind, rx in PATTERNS.items():
                    if kind in exempt:
                        continue
                    for m in re.finditer(rx, line, re.IGNORECASE):
                        if m.group().strip().lower().replace(" ", "_") in ALLOWED:
                            continue
                        leaks.append((str(p.relative_to(ROOT)), kind, line_no,
                                      m.group()[:40]))

    print(f"scanned {n_files} shareable files under {[str(s.name) for s in SCAN]}")
    if not leaks:
        print("CLEAN -- no study identifiers outside _private/")
        return 0
    print(f"LEAK -- {len(leaks)} occurrence(s):")
    seen = set()
    for path, kind, line_no, tok in leaks:
        key = (path, kind, tok)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {path}:{line_no}  [{kind}]  {tok!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

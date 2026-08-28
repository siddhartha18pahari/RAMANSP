"""Build and apply the sample-identity map.

The instrument corpus carries real specimen identifiers (``ATE34068`` ...),
material names, a person's name and dated session folders. For a publishable
artefact none of that may appear in any figure, table, manifest or the
manuscript. This module scans the input tree once, assigns opaque codes and
exposes :func:`scrub` to rewrite arbitrary strings.

The forward map (code -> original token) is written by the corpus builder to
``OUTPUT FILES/_private/sample_key.csv`` and to nowhere else; ``_private/`` is
git-ignored and the file carries a NOT-FOR-DISTRIBUTION banner.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# token regex -> (kind, code prefix). Order matters: first match wins.
_SPECIMEN_RE = re.compile(r"ATE[\s_-]?0*(\d{4,6})", re.IGNORECASE)
_BARE_ID_RE = re.compile(r"(?<!\d)(34\d{3})(?!\d)")  # 34068 / 34150 / 34448 seen bare

# Codes must be *opaque*: "MAT-QC" would still spell out the initials of the
# material it replaces, which is not anonymisation. Only the two public
# reference standards keep a mnemonic, and even they resolve to a name solely
# through the private key.
_KEYWORDS: list[tuple[str, str, str]] = [
    # regex, kind, code
    (r"glass[y]?\s*carbon", "reference", "REF-01"),
    (r"\bgraphite\b", "reference", "REF-02"),
    (r"\bGQS\b", "material", "MAT-A"),
    (r"\bQ[\s_-]?CARBON\b", "material", "MAT-B"),
    (r"\bSWIFT\b", "method", "METH-A"),
    (r"\bpeng\b", "person", "PERSON-01"),
]

_MONTHS = {
    m: i
    for i, m in enumerate(
        "JANUARY FEBRUARY MARCH APRIL MAY JUNE JULY AUGUST SEPTEMBER OCTOBER "
        "NOVEMBER DECEMBER".split(),
        start=1,
    )
}
_DATE_DIR_RE = re.compile(r"^(\d{1,2})\s+([A-Z]+)\s+(\d{4})$", re.IGNORECASE)
# NB: a trailing \b would fail on "DOE 17_04" -- '_' is a word character, so
# there is no boundary between "17" and "_". Use a not-a-digit lookahead.
_DOE_RE = re.compile(r"\bDOE[\s_-]?(\d{1,3})(?!\d)", re.IGNORECASE)
# Dates embedded in file names ("... 17 JULY SPECTRA", "AUG 5") are identifying
# even when they are not a session directory of their own.
_MONTH_ALT = ("JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|"
              "JUL(?:Y)?|AUG(?:UST)?|SEP(?:T|TEMBER)?|OCT(?:OBER)?|"
              "NOV(?:EMBER)?|DEC(?:EMBER)?")
_LOOSE_DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}\s*(?:{_MONTH_ALT})|(?:{_MONTH_ALT})\s*\d{{1,2}})"
    rf"(?:\s*,?\s*\d{{4}})?\b",
    re.IGNORECASE,
)


@dataclass
class IdentityMap:
    """Deterministic, reproducible mapping from raw tokens to opaque codes."""

    specimen: dict[str, str] = field(default_factory=dict)   # numeric id -> S##
    keyword: dict[str, str] = field(default_factory=dict)    # regex      -> code
    config: dict[str, str] = field(default_factory=dict)     # DOE number -> CFG-##
    session: dict[str, str] = field(default_factory=dict)    # raw dirname -> session-##

    # ---- construction -------------------------------------------------
    @classmethod
    def from_tree(cls, root: str) -> "IdentityMap":
        m = cls()
        ids_seen: list[str] = []
        does_seen: list[int] = []
        sessions: list[tuple[tuple[int, int, int], str]] = []

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            filenames.sort()
            rel = os.path.relpath(dirpath, root)
            for part in rel.replace("\\", "/").split("/"):
                dm = _DATE_DIR_RE.match(part.strip())
                if dm:
                    d, mon, yr = int(dm[1]), _MONTHS.get(dm[2].upper(), 0), int(dm[3])
                    key = ((yr, mon, d), part.strip())
                    if key not in sessions:
                        sessions.append(key)
            for name in [rel] + filenames:
                for mt in _SPECIMEN_RE.finditer(name):
                    if mt[1] not in ids_seen:
                        ids_seen.append(mt[1])
                for mt in _BARE_ID_RE.finditer(name):
                    if mt[1] not in ids_seen:
                        ids_seen.append(mt[1])
                for mt in _DOE_RE.finditer(name):
                    if int(mt[1]) not in does_seen:
                        does_seen.append(int(mt[1]))

        for i, raw in enumerate(sorted(ids_seen, key=int), start=1):
            m.specimen[raw] = f"S{i:02d}"
        for rx, _kind, code in _KEYWORDS:
            m.keyword[rx] = code
        for n in sorted(does_seen):
            m.config[str(n)] = f"CFG-{n:02d}"
        for i, (_srt, raw) in enumerate(sorted(sessions), start=1):
            m.session[raw] = f"session-{i:02d}"
        return m

    # ---- application ------------------------------------------------
    def scrub(self, text: str) -> str:
        """Replace every known identifier in ``text`` with its opaque code."""
        if not text:
            return text
        out = text
        for raw, code in sorted(self.specimen.items(), key=lambda kv: -len(kv[0])):
            out = re.sub(rf"ATE[\s_-]?0*{raw}\b", code, out, flags=re.IGNORECASE)
            out = re.sub(rf"(?<!\d){raw}(?!\d)", code, out)
        for rx, code in self.keyword.items():
            out = re.sub(rx, code, out, flags=re.IGNORECASE)
        out = _DOE_RE.sub(lambda mt: self.config.get(mt[1], f"CFG-{int(mt[1]):02d}"), out)
        # longest raw first, and never start a match mid-number ("6 AUGUST 2026"
        # must not fire inside "26 AUGUST 2026")
        for raw in sorted(self.session, key=len, reverse=True):
            out = re.sub(rf"(?<!\d){re.escape(raw)}", self.session[raw], out)
        # any remaining loose date in a file name
        out = _LOOSE_DATE_RE.sub("date", out)
        return out

    def code_for_path(self, rel_path: str) -> dict[str, str | None]:
        """Best-effort (specimen, material, config, session) codes for one file."""
        text = rel_path.replace("\\", "/")
        specimen = material = config = session = None
        mt = _SPECIMEN_RE.search(text) or _BARE_ID_RE.search(text)
        if mt:
            key = mt[1] if mt.re is _SPECIMEN_RE else mt[1]
            specimen = self.specimen.get(re.sub(r"\D", "", key))
        for rx, kind, code in _KEYWORDS:
            if re.search(rx, text, re.IGNORECASE):
                if kind in ("reference", "material"):
                    material = code
                elif kind == "method" and material is None:
                    material = code
        dm = _DOE_RE.search(text)
        if dm:
            config = self.config.get(dm[1], f"CFG-{int(dm[1]):02d}")
        for raw in sorted(self.session, key=len, reverse=True):
            if re.search(rf"(?<!\d){re.escape(raw)}", text):
                session = self.session[raw]
                break
        return {"specimen": specimen, "material": material, "config": config, "session": session}

    def key_rows(self) -> list[dict[str, str]]:
        rows = [{"code": c, "kind": "specimen", "original_token": f"ATE{r}"}
                for r, c in sorted(self.specimen.items(), key=lambda kv: kv[1])]
        rows += [{"code": c, "kind": "keyword", "original_token": rx}
                 for rx, c in self.keyword.items()]
        rows += [{"code": c, "kind": "config", "original_token": f"DOE {n}"}
                 for n, c in sorted(self.config.items(), key=lambda kv: int(kv[0]))]
        rows += [{"code": c, "kind": "session", "original_token": r}
                 for r, c in sorted(self.session.items(), key=lambda kv: kv[1])]
        return rows

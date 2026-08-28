import re

from ramansp._anonymize import IdentityMap

FORBIDDEN = re.compile(
    r"ate\s?34|swift|\bpeng\b|glassy\s*carbon|q\s?carbon|\bGQS\b|"
    r"\d{1,2}\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)",
    re.IGNORECASE,
)


def _map():
    m = IdentityMap()
    m.specimen = {"34068": "S01", "34448": "S04"}
    m.keyword = {r"glass[y]?\s*carbon": "REF-01", r"\bGQS\b": "MAT-GQS",
                 r"\bSWIFT\b": "METH-SW", r"\bpeng\b": "PERSON-01"}
    m.config = {"8": "CFG-08", "17": "CFG-17"}
    m.session = {"31 JULY 2026": "session-13"}
    return m


def test_scrubs_specimen_material_and_config():
    m = _map()
    out = m.scrub("Results/31 JULY 2026/DOE 8/ATE34068 glassy carbon SWIFT.l6m")
    assert "S01" in out and "CFG-08" in out and "session-13" in out
    assert not FORBIDDEN.search(out), out


def test_doe_number_followed_by_underscore():
    """`DOE 17_04` -- a trailing \\b fails here because '_' is a word char."""
    out = _map().scrub("DOE 17_04 ALL SPECTRA.l6m")
    assert out.startswith("CFG-17_04"), out
    assert "DOE" not in out


def test_loose_date_in_filename_is_scrubbed():
    out = _map().scrub("D_4 VRM 17 JULY SPECTRA.l6m")
    assert not FORBIDDEN.search(out), out


def test_scrub_is_idempotent():
    m = _map()
    once = m.scrub("ATE34448 DOE 8 31 JULY 2026")
    assert m.scrub(once) == once

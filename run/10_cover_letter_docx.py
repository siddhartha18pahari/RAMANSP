"""Stage 10 -- render the cover letter as a Word document.

The submission system takes the cover letter pasted in or attached as a file,
and .docx is the format it expects. The source of truth stays
``paper/cover_letter.md``; this converts it rather than duplicating it, so the
two cannot drift apart.

The converter handles only the Markdown this letter uses: ATX headings, block
quotes, bullet and numbered lists, horizontal rules, and bold spans. Anything
richer would need pandoc, which is not assumed here.

    python run/10_cover_letter_docx.py
"""

from __future__ import annotations

import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import PAPER  # noqa: E402

BODY_PT = 11
PLACEHOLDER = re.compile(r"\[\[(.+?)\]\]", re.S)
BOLD = re.compile(r"\*\*(.+?)\*\*")
CODE = re.compile(r"`([^`]+)`")


INLINE = re.compile(r"\*\*(.+?)\*\*|\[\[(.+?)\]\]|`([^`]+)`|\*(.+?)\*",
                    re.S)


def add_runs(par, text, highlight_placeholders=True):
    """Write text into a paragraph, marking bold, italic and placeholder spans.

    Whitespace is normalised once for the whole paragraph rather than per run,
    because collapsing it inside each run deletes the single space that
    separates a bold span from the words after it.
    """
    from docx.enum.text import WD_COLOR_INDEX
    from docx.shared import Pt

    # normalise each line separately so that a hard break, carried here as a
    # newline, is not swallowed along with the wrapping whitespace
    text = "\n".join(" ".join(seg.split()) for seg in text.split("\n"))
    parts, pos = [], 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            parts.append((text[pos:m.start()], None))
        if m.group(1) is not None:
            parts.append((m.group(1), "bold"))
        elif m.group(2) is not None:
            parts.append(("[" + m.group(2) + "]", "todo"))
        elif m.group(3) is not None:
            parts.append((m.group(3), "code"))
        else:
            parts.append((m.group(4), "italic"))
        pos = m.end()
    if pos < len(text):
        parts.append((text[pos:], None))

    for chunk, kind in parts:
        if not chunk:
            continue
        segments = chunk.split("\n")
        run = par.add_run(segments[0])
        for seg in segments[1:]:
            run.add_break()
            run.add_text(seg)
        run.font.size = Pt(BODY_PT)
        if kind == "bold":
            run.bold = True
        elif kind == "italic":
            run.italic = True
        elif kind == "code":
            run.font.name = "Consolas"
            run.font.size = Pt(BODY_PT - 1)
        elif kind == "todo" and highlight_placeholders:
            run.bold = True
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def blocks(lines):
    """Group Markdown lines into blocks, joining soft-wrapped paragraph lines.

    Yields (kind, text) where kind is one of title, heading, quote, bullet,
    number, rule, para.
    """
    buf, kind = [], None

    def flush():
        nonlocal buf, kind
        if buf:
            joined = ""
            for i, piece in enumerate(buf):
                if i and not joined.endswith("\n"):
                    joined += " "
                joined += piece
            yield_val = (kind or "para", joined)
            buf, kind = [], None
            return yield_val
        buf, kind = [], None
        return None

    for raw in list(lines) + [""]:
        hard = raw.endswith("  ") and raw.strip() != ""
        line = raw.rstrip()
        stripped = line.strip() + ("\n" if hard else "")

        if not stripped or stripped == "---":
            out = flush()
            if out:
                yield out
            continue

        if stripped.startswith("#"):
            out = flush()
            if out:
                yield out
            level = len(stripped) - len(stripped.lstrip("#"))
            yield ("title" if level == 1 else "heading", stripped[level:].strip())
            continue

        if stripped.startswith(">") and kind == "quote":
            buf.append(stripped.lstrip("> "))
            continue

        starts = (stripped.startswith("> "), stripped.startswith("- "),
                  stripped.startswith("* ") and not stripped.startswith("**"),
                  bool(re.match(r"^\d+\.\s", stripped)))
        if any(starts):
            out = flush()
            if out:
                yield out
            if starts[0]:
                kind, body = "quote", stripped[2:]
            elif starts[1] or starts[2]:
                kind, body = "bullet", stripped[2:]
            else:
                kind, body = "number", re.sub(r"^\d+\.\s+", "", stripped)
            buf = [body]
            continue

        # a continuation line belongs to whatever block is open
        buf.append(stripped)

    out = flush()
    if out:
        yield out


def main():
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    src = PAPER / "cover_letter.md"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    lines = src.read_text(encoding="utf-8").split("\n")

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(BODY_PT)
    style.paragraph_format.space_after = Pt(8)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    n_todo = 0
    for kind, text in blocks(lines):
        n_todo += len(PLACEHOLDER.findall(text))

        if kind == "title":
            add_runs(doc.add_heading(level=0), text)
        elif kind == "heading":
            h = doc.add_heading(level=1)
            add_runs(h, text)
            for run in h.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(BODY_PT + 1)
        elif kind == "quote":
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Inches(0.3)
            par.paragraph_format.space_after = Pt(12)
            add_runs(par, text)
            for run in par.runs:
                run.italic = True
        elif kind == "bullet":
            add_runs(doc.add_paragraph(style="List Bullet"), text)
        elif kind == "number":
            add_runs(doc.add_paragraph(style="List Number"), text)
        else:
            par = doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT
            add_runs(par, text)

    out = PAPER / "cover_letter.docx"
    doc.save(out)
    print(f"wrote {out}")
    print(f"  {n_todo} placeholders, highlighted in yellow, still to fill")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

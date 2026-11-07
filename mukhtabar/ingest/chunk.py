"""Page text -> paragraphs -> chunks of roughly a target size.

Lines are joined into paragraphs: a paragraph ends at a line ending in sentence
punctuation, at a short line (a heading, list item, or a paragraph's last line),
or before a table row or a line opening with a list marker (MCQ option, section
number, bullet). A plain number or a dash only starts a paragraph after a line
that already ended one, since wrapped lines often begin with them.

Paragraphs are packed into chunks without splitting them, unless one is too long
on its own, in which case it's split at sentence ends. A chunk is never closed
below a minimum size, and never ends on a heading or on an intro ending in ":".
Chunks may cross page breaks and record the pages they start and end on.
"""

import re
import statistics
from dataclasses import dataclass

TARGET_CHARS = 1000
MIN_CHARS = 200
TERMINAL = tuple(".؟?!:")
SENTENCE_END = re.compile(r"(?<=[.؟?!])\s+")
SHORT_LINE = 0.7     # lines shorter than this share of a typical full line
TABLE_SEPARATOR = " | "
# A line opening with one of these always starts a new paragraph: an MCQ option
# letter ("أ ", "أ.", "أ)"), a section number ("1-3", "14-1-3"), "1." or "1)", or a
# bullet symbol. Plain numbers and dashes aren't here: a wrapped line in the middle
# of a sentence can start with a number or a dash.
LIST_MARKER = re.compile(r"([أابجده][.)]?|\d+(?:-\d+)+|\d+[.)]|[•▪◦●■★⚠✓])\s")


@dataclass
class Paragraph:
    text: str
    page: int
    page_end: int
    heading: bool = False


@dataclass
class Chunk:
    text: str
    page: int
    page_end: int
    position: int


def _is_table_row(line: str) -> bool:
    return TABLE_SEPARATOR in f" {line} "


def paragraphs(pages: list[tuple[int, str]]) -> list[Paragraph]:
    """pages: (page number, text) in document order."""
    lines = [(n, l.strip()) for n, text in pages for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    full = statistics.quantiles([len(l) for _, l in lines], n=4)[2] if len(lines) > 1 else len(lines[0][1])
    out: list[Paragraph] = []
    current: list[str] = []
    start = end = 0

    def close(heading=False):
        nonlocal current
        if current:
            out.append(Paragraph(" ".join(current), start, end, heading))
            current = []

    for page, line in lines:
        short = len(line) < SHORT_LINE * full
        if _is_table_row(line):
            close()
            out.append(Paragraph(line, page, page))
            continue
        if LIST_MARKER.match(line):
            close()
        if not current:
            start = page
        current.append(line)
        end = page
        if line.endswith(TERMINAL) or short:
            # A short line that is a whole paragraph by itself and isn't a sentence is a heading.
            close(heading=len(current) == 1 and short and not line.endswith(TERMINAL))
    close()
    return out


def _split_long(p: Paragraph, limit: int) -> list[Paragraph]:
    if len(p.text) <= limit:
        return [p]
    parts, buf = [], ""
    for sentence in SENTENCE_END.split(p.text):
        if buf and len(buf) + 1 + len(sentence) > limit:
            parts.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    parts.append(buf)
    # Page boundaries inside a paragraph aren't tracked; each part keeps the paragraph's range.
    return [Paragraph(t, p.page, p.page_end) for t in parts]


def chunk_paragraphs(paras: list[Paragraph], target: int = TARGET_CHARS) -> list[Chunk]:
    limit = int(target * 1.5)
    items = [part for p in paras for part in _split_long(p, limit)]
    chunks: list[Chunk] = []
    buf: list[Paragraph] = []

    def leads_in(p: Paragraph) -> bool:
        # Headings and intros ending in ":" belong with the text after them.
        return p.heading or p.text.endswith(":")

    def size() -> int:
        return sum(len(b.text) + 1 for b in buf)

    def emit(paras: list[Paragraph]) -> None:
        chunks.append(Chunk("\n".join(p.text for p in paras), paras[0].page,
                            max(p.page_end for p in paras), len(chunks)))

    for p in items:
        if buf and size() + len(p.text) > target:
            # Trailing headings/intros move to the next chunk; close this one only
            # if what stays behind is still big enough to stand alone.
            keep = len(buf)
            while keep > 1 and leads_in(buf[keep - 1]):
                keep -= 1
            if sum(len(b.text) + 1 for b in buf[:keep]) >= MIN_CHARS:
                emit(buf[:keep])
                buf = buf[keep:]
        buf.append(p)
    if buf:
        if chunks and size() < MIN_CHARS:
            # A tiny tail joins the previous chunk instead of standing alone.
            last = chunks.pop()
            prev = Paragraph(last.text, last.page, last.page_end)
            emit([prev] + buf)
        else:
            emit(buf)
    return chunks


def chunk_pages(pages: list[tuple[int, str]], target: int = TARGET_CHARS) -> list[Chunk]:
    return chunk_paragraphs(paragraphs(pages), target)

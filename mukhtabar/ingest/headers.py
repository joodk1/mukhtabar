"""Remove running headers/footers: page numbers and lines repeated at a page edge.

Deliberately conservative. A line is removed only if it sits in the outer band
of the page and is either a page number, or the same text at the same height on
a large share of the document's pages while being the outermost line on that
edge (so a badge under a heading at the top of a box isn't mistaken for one).
Labels of study boxes that happen to start at the top of a page, and per-chapter
running titles, stay in; the latter are reported as kept so they can be reviewed.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

from mukhtabar.ingest.extract import Page

EDGE = 0.08            # outer share of the page height, top and bottom
MIN_SHARE = 0.20       # share of pages a repeated line must appear on
MIN_REPEATS = 3        # ...and never fewer pages than this, so short documents keep their text
MAX_Y_SPREAD = 0.02    # repeated lines must sit at a consistent height
MIN_REPORTED = 3       # repeats below the threshold are reported from this count
SAME_Y = 2.0           # points; lines this close to the edge-most one count as edge-most

DIGITS = re.compile(r"[0-9٠-٩۰-۹]+")
# Front matter is numbered with lowercase roman numerals ("iv"), up to 89 here
# (letters i, v, x, l), so words like "mix" (1009) aren't read as numbers.
# Uppercase is left alone so a lone "C" or "D" at a page edge isn't mistaken either.
ROMAN = re.compile(r"(?<![A-Za-z])(?=[ivxl])(?:xl|l?x{0,3})(?:ix|iv|v?i{0,3})(?![A-Za-z])")
PAGE_NUMBER = re.compile(r"[#\s/|.\-–—]*#[#\s/|.\-–—]*")
# MCQ options ("أ ...", "ب ...") repeat a lot in question banks; never touch them.
OPTION = re.compile(r"[أابجده]\s")


@dataclass
class Removal:
    page: int        # 1-based
    side: str        # "top" or "bottom"
    text: str


@dataclass
class EdgeReport:
    removed: list[Removal]
    kept: list[tuple[str, str, int]]   # (side, text pattern, pages) repeated but below the threshold


def _key(text: str) -> str:
    return ROMAN.sub("#", DIGITS.sub("#", text)).strip()


def _side(line, height: float) -> str | None:
    if line.bottom <= EDGE * height:
        return "top"
    if line.top >= (1 - EDGE) * height:
        return "bottom"
    return None


def _edge_lines(page: Page) -> list[tuple[object, str, str, bool]]:
    """(line, side, key, outermost) for each non-empty line in the edge bands.
    Outermost means no line other than a page number is closer to that edge."""
    found = []
    for line in page.lines:
        side = _side(line, page.height)
        if side and line.text.strip():
            found.append((line, side, _key(line.text)))
    text_lines = [(l, s) for l, s, k in found if not PAGE_NUMBER.fullmatch(k)]
    top = min((l.top for l, s in text_lines if s == "top"), default=None)
    bottom = max((l.bottom for l, s in text_lines if s == "bottom"), default=None)
    return [(l, s, k, l.top <= top + SAME_Y if s == "top" else l.bottom >= bottom - SAME_Y)
            if not PAGE_NUMBER.fullmatch(k) else (l, s, k, True)
            for l, s, k in found]


def remove_headers_footers(pages: list[Page]) -> tuple[list[Page], EdgeReport]:
    """Return new pages with running headers/footers removed, and what was removed."""
    seen: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for i, page in enumerate(pages):
        if not page.lines or not page.height:
            continue
        for line, side, key, outermost in _edge_lines(page):
            if outermost:
                seen[(side, key)].append((i, line.top / page.height))

    n_pages = sum(1 for p in pages if p.text.strip())
    remove_keys, kept = set(), []
    for (side, key), hits in seen.items():
        distinct = len({i for i, _ in hits})
        ys = [y for _, y in hits]
        if OPTION.match(key):
            continue
        if PAGE_NUMBER.fullmatch(key):
            remove_keys.add((side, key))
        elif (distinct >= max(MIN_REPEATS, MIN_SHARE * n_pages)
              and max(ys) - min(ys) <= MAX_Y_SPREAD):
            remove_keys.add((side, key))
        elif distinct >= MIN_REPORTED:
            kept.append((side, key, distinct))

    out, removed = [], []
    for i, page in enumerate(pages):
        if not page.lines or not page.height:
            out.append(page)
            continue
        drop = {id(line): side for line, side, key, outermost in _edge_lines(page)
                if outermost and (side, key) in remove_keys}
        lines = []
        for line in page.lines:
            if id(line) in drop:
                removed.append(Removal(i + 1, drop[id(line)], line.text))
            else:
                lines.append(line)
        out.append(Page("\n".join(l.text for l in lines), page.flags, lines, page.height))
    kept.sort(key=lambda k: -k[2])
    return out, EdgeReport(removed, kept)

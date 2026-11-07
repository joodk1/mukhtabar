"""Rebuild reading order from glyph positions, for Arabic (right-to-left) pages.

PDFs store Arabic text either in logical (reading) order or in visual order, and
plain extractors break on one or the other. Ordering glyphs by their position on
the page sidesteps that: lines read right to left, while runs of Latin letters
and digits read left to right. Glyph objects are kept whole, so a ligature like
"لا" stored as one object is never split or flipped.

Brackets in right-to-left runs are stored by drawn shape by some producers
(Word) and by meaning by others (InDesign), so they are emitted as placeholders
and resolved per page by resolve_brackets().

Handles one or two columns. Pages that look multi-column but have no clear
gutter are laid out as one column and flagged.
"""

import re
import statistics

ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
LTR_CHAR = re.compile(r"[A-Za-z0-9٠-٩۰-۹]")
MARKS = set(chr(c) for c in range(0x064B, 0x0653)) | {"ٰ"}
BRACKETS = "()[]{}<>«»"
MIRRORED = ")(][}{><»«"
# Private-use placeholders for brackets drawn in a right-to-left run.
PLACEHOLDERS = "".join(chr(0xE000 + i) for i in range(len(BRACKETS)))
TO_PLACEHOLDER = str.maketrans(BRACKETS, PLACEHOLDERS)
KEEP = str.maketrans(PLACEHOLDERS, BRACKETS)
MIRROR = str.maketrans(PLACEHOLDERS, MIRRORED)
OPEN_PH, CLOSE_PH = PLACEHOLDERS[0], PLACEHOLDERS[1]

FLAG_COLUMNS = "possible-columns-not-split"

# Column detection, in PDF points.
MIN_GUTTER = 6
WIDE_GAP = 25
SIDE_MARGIN = 0.05  # share of the page width
SPAN_GAP_RATIO = 2.0  # a gap this many times a line's usual word gap separates columns


def _reverse_runs(items: list, in_run) -> list:
    """Reverse each maximal run of items for which in_run(item) is true, in place order."""
    out, run = [], []
    for item in items:
        if in_run(item):
            run.append(item)
            continue
        out += run[::-1]
        run = []
        out.append(item)
    return out + run[::-1]


def _word_text(chars: list[dict]) -> str:
    text = "".join(c["text"] for c in chars)
    if not ARABIC.search(text):
        return "".join(c["text"] for c in sorted(chars, key=lambda c: c["x0"]))
    parts = [c["text"] for c in sorted(chars, key=lambda c: -c["x0"])]
    parts = _reverse_runs(parts, lambda t: bool(LTR_CHAR.fullmatch(t)))
    return "".join(p if LTR_CHAR.fullmatch(p) else p.translate(TO_PLACEHOLDER) for p in parts)


def _ltr_run_text(tokens: list[str]) -> str:
    """A run of non-Arabic tokens (given in visual left-to-right order) inside a
    right-to-left line. The core, from the first to the last letter or digit,
    reads left to right. Punctuation at its edges belongs to the surrounding
    Arabic, so it moves to the other side and its brackets become placeholders,
    except a bracket that closes one opened inside the core, as in "(X − M)"."""
    s = " ".join(tokens)
    idx = [i for i, ch in enumerate(s) if LTR_CHAR.match(ch)]
    if not idx:
        return s[::-1].translate(TO_PLACEHOLDER)
    i, j = idx[0], idx[-1] + 1
    while j < len(s) and s[j] == ")" and s[:j].count("(") > s[:j].count(")"):
        j += 1
    while i > 0 and s[i - 1] == "(" and s[i:].count(")") > s[i:].count("("):
        i -= 1
    lead, core, trail = s[:i], s[i:j], s[j:]
    return trail[::-1].translate(TO_PLACEHOLDER) + core + lead[::-1].translate(TO_PLACEHOLDER)


def _line_text(words: list[dict]) -> str:
    tokens = [_word_text(w["chars"]) for w in sorted(words, key=lambda w: -w["x1"])]
    out, run = [], []
    for t in tokens + [None]:
        if t is not None and not ARABIC.search(t):
            run.append(t)
            continue
        if run:
            out.append(_ltr_run_text(run[::-1]))
            run = []
        if t is not None:
            out.append(t)
    return " ".join(out)


def _baseline(w: dict) -> float:
    """Distance from the page top to the word's baseline. Glyph boxes differ by
    font (an Arabic font's are much taller than a Latin one's); baselines don't."""
    return -w["chars"][0]["matrix"][5]


def _cluster_lines(words: list[dict]) -> list[list[dict]]:
    if not words:
        return []
    tol = max(2.0, 0.4 * statistics.median(w["bottom"] - w["top"] for w in words))
    lines: list[list[dict]] = []
    ref = None
    for w in sorted(words, key=_baseline):
        if ref is not None and abs(_baseline(w) - ref) < tol:
            lines[-1].append(w)
        else:
            lines.append([w])
            ref = _baseline(w)
    return lines


def _find_gutter(words: list[dict], width: float, n_lines: int) -> tuple[float, float] | None:
    """A vertical strip in the middle of the page that almost no word crosses."""
    if len(words) < 20:
        return None
    cover = [0] * (int(width) + 2)
    for w in words:
        for x in range(max(int(w["x0"]), 0), min(int(w["x1"]) + 1, len(cover))):
            cover[x] += 1
    tol = max(2, int(0.1 * n_lines))  # a few full-width headings may cross it
    lo, hi = int(0.3 * width), int(0.7 * width)
    best, start = None, None
    for x in range(lo, hi + 1):
        if cover[x] <= tol:
            start = x if start is None else start
            if best is None or x - start > best[1] - best[0]:
                best = (start, x)
        else:
            start = None
    if best is None or best[1] - best[0] < MIN_GUTTER:
        return None
    mid = (best[0] + best[1]) / 2
    right = sum((w["x0"] + w["x1"]) / 2 > mid for w in words)
    if min(right, len(words) - right) < 0.15 * len(words):
        return None
    return best


def _word_gaps(line: list[dict]) -> list[float]:
    xs = sorted(line, key=lambda w: w["x0"])
    return [b["x0"] - a["x1"] for a, b in zip(xs, xs[1:])]


def _spans(line: list[dict], mid: float, page_gap: float) -> bool:
    """Whether a line runs across the gutter (a full-width heading or sentence)
    rather than being two column lines that share a baseline. A word may cover
    the gutter's middle, or the middle may fall in an ordinary gap between words.
    "Ordinary" is judged against the line's own gaps (justified lines stretch
    them evenly), or the page's when the line has too few words; two columns
    leave a gap there that is clearly wider than the rest."""
    if any(w["x0"] < mid < w["x1"] for w in line):
        return True
    left = [w["x1"] for w in line if w["x1"] <= mid]
    right = [w["x0"] for w in line if w["x0"] >= mid]
    if not (left and right):
        return False
    cross = min(right) - max(left)
    others = [g for g in _word_gaps(line) if abs(g - cross) > 1e-6]
    typical = statistics.median(others) if len(others) >= 2 else page_gap
    return cross < min(WIDE_GAP, SPAN_GAP_RATIO * max(typical, 1.0))


def _looks_multicolumn(lines: list[list[dict]]) -> bool:
    multi = [l for l in lines if len(l) > 1]
    if len(multi) < 5:
        return False
    def max_gap(line):
        xs = sorted(line, key=lambda w: w["x0"])
        return max(b["x0"] - a["x1"] for a, b in zip(xs, xs[1:]))
    return sum(max_gap(l) > WIDE_GAP for l in multi) >= 0.3 * len(multi)


def _bracket_votes(text: str) -> tuple[int, int]:
    """Lines whose first right-to-left parenthesis is "(" (keep) vs ")" (mirror)."""
    keep = mirror = 0
    for line in text.splitlines():
        for ch in line:
            if ch == OPEN_PH:
                keep += 1
                break
            if ch == CLOSE_PH:
                mirror += 1
                break
    return keep, mirror


def resolve_brackets(texts: list[str]) -> list[str]:
    """Replace bracket placeholders, deciding per page whether glyphs were stored
    by shape (mirror) or by meaning (keep). Pages without evidence follow the
    document majority."""
    votes = [_bracket_votes(t) for t in texts]
    doc_mirror = sum(m for _, m in votes) > sum(k for k, _ in votes)
    out = []
    for text, (k, m) in zip(texts, votes):
        mirror = m > k if k != m else doc_mirror
        out.append(text.translate(MIRROR if mirror else KEEP))
    return out


def _inside(w: dict, bbox) -> bool:
    x0, top, x1, bottom = bbox
    cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom


def _positioned(words: list[dict]) -> tuple[str, float, float]:
    return _line_text(words), min(w["top"] for w in words), max(w["bottom"] for w in words)


def _table_lines(table, words: list[dict]) -> list[tuple[str, float, float]]:
    """One line per row: each cell's text kept whole, cells right to left."""
    rows = []
    for row in table.rows:
        cells = sorted((c for c in row.cells if c), key=lambda c: -c[2])
        texts = [" ".join(_line_text(l) for l in _cluster_lines([w for w in words if _inside(w, c)]))
                 for c in cells]
        if any(texts):
            rows.append((" | ".join(texts), row.bbox[1], row.bbox[3]))
    return rows


def layout_page(page) -> tuple[list[tuple[str, float, float]], list[str]]:
    """Lines of a pdfplumber page in reading order, as (text, top, bottom), with
    bracket placeholders still unresolved (see resolve_brackets), plus any flags.

    Tables are laid out cell by cell; the text between them is laid out in bands."""
    page = page.filter(lambda o: not (
        o.get("object_type") == "char" and o["text"] and all(ch in MARKS for ch in o["text"])
    ))
    words = page.extract_words(return_chars=True, x_tolerance=1.5)
    # Words entirely in the side margins (e.g. a chapter-number tab) can share a
    # baseline with body text; lay them out on their own, after the page body.
    margin = [w for w in words
              if not SIDE_MARGIN * page.width <= (w["x0"] + w["x1"]) / 2 <= (1 - SIDE_MARGIN) * page.width]
    words = [w for w in words if w not in margin]
    tables = sorted(page.find_tables(), key=lambda t: t.bbox[1])
    out: list[tuple[str, float, float]] = []
    flags: list[str] = []
    top = 0.0
    for table in tables + [None]:
        bottom = table.bbox[1] if table else float("inf")
        band = [w for w in words if top <= (w["top"] + w["bottom"]) / 2 < bottom
                and not any(_inside(w, t.bbox) for t in tables)]
        lines, band_flags = _layout_flow(band, page.width)
        out += lines
        flags += [f for f in band_flags if f not in flags]
        if table:
            out += _table_lines(table, [w for w in words if _inside(w, table.bbox)])
            top = table.bbox[3]
    out += [_positioned(l) for l in _cluster_lines(margin)]
    return out, flags


def _layout_flow(words: list[dict], width: float) -> tuple[list[tuple[str, float, float]], list[str]]:
    """Running text outside tables, in one or two columns."""
    lines = _cluster_lines(words)
    gutter = _find_gutter(words, width, len(lines))
    if gutter is None:
        flags = [FLAG_COLUMNS] if _looks_multicolumn(lines) else []
        return [_positioned(l) for l in lines], flags

    g0, g1 = gutter
    mid = (g0 + g1) / 2
    page_gap = statistics.median([g for l in lines for g in _word_gaps(l)] or [WIDE_GAP])
    ordered: list[list[dict]] = []
    right: list[dict] = []
    left: list[dict] = []
    for line in lines:
        if _spans(line, mid, page_gap):
            # A full-width line (e.g. a heading) closes the columns above it.
            ordered += _cluster_lines(right) + _cluster_lines(left)
            right, left = [], []
            ordered.append(line)
        else:
            for w in line:
                (right if (w["x0"] + w["x1"]) / 2 > mid else left).append(w)
    ordered += _cluster_lines(right) + _cluster_lines(left)
    return [_positioned(l) for l in ordered], []

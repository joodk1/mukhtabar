"""Generate the synthetic Arabic test PDFs from content.py.

    python tests/fixtures/make_fixtures.py --font NotoNaskhArabic-Regular.ttf

Needs fpdf2 (dev only) and Noto Naskh Arabic (SIL OFL 1.1, see OFL.txt),
from https://github.com/notofonts/arabic. The font isn't committed; the PDFs
embed a subset of it.

Two PDFs with the same content, mimicking the two kinds of real Arabic PDFs:

- visual-order.pdf: like Word. Each word is pre-shaped into presentation forms
  and written as one string in visual (left-to-right) glyph order; words are
  written left to right; brackets are stored by drawn shape. Naive extraction
  reverses the word order.
- logical-order.pdf: like InDesign. Every glyph is written separately, in
  reading order, at its visual position; brackets are stored by meaning.

Only the character codes and positions matter for extraction, so the logical
file draws brackets facing the "wrong" way; that's expected. Characters the
Arabic font lacks (Latin letters, brackets) use the built-in Helvetica.
"""

import argparse
import re
import unicodedata
from pathlib import Path

from fontTools.ttLib import TTFont
from fpdf import FPDF

import content

HERE = Path(__file__).parent
ARABIC = re.compile(r"[؀-ۿ]")
MIRROR = str.maketrans("()", ")(")
SIZE = 16
LINE = 10  # mm

# Presentation forms from Unicode data: base letter -> {form: code point}.
FORMS: dict[str, dict[str, str]] = {}
for _cp in range(0xFE70, 0xFEFD):
    _d = unicodedata.decomposition(chr(_cp)).split()
    if len(_d) == 2 and _d[0] in ("<isolated>", "<final>", "<initial>", "<medial>"):
        FORMS.setdefault(chr(int(_d[1], 16)), {})[_d[0][1:-1]] = chr(_cp)
LAM_ALEF = {"ا": "ﻻ", "أ": "ﻷ", "إ": "ﻹ", "آ": "ﻵ"}  # isolated; +1 is final


def _joins_next(ch: str) -> bool:
    return "medial" in FORMS.get(ch, {})  # dual-joining letters have a medial form


def shape(word: str) -> list[str]:
    """Logical-order word -> presentation-form glyph characters, logical order."""
    out, i = [], 0
    while i < len(word):
        ch = word[i]
        prev_joins = i > 0 and _joins_next(word[i - 1])
        if ch == "ل" and word[i + 1:i + 2] in LAM_ALEF:
            lig = LAM_ALEF[word[i + 1]]
            out.append(chr(ord(lig) + 1) if prev_joins else lig)
            i += 2
            continue
        forms = FORMS.get(ch)
        if not forms:
            out.append(ch)
            i += 1
            continue
        next_joins = _joins_next(ch) and word[i + 1:i + 2] in FORMS
        form = {(True, True): "medial", (True, False): "final",
                (False, True): "initial", (False, False): "isolated"}[(prev_joins, next_joins)]
        out.append(forms.get(form) or forms["isolated"])
        i += 1
    return out


def _runs(tokens: list[str]) -> list[list[str]]:
    """Group a line's tokens into runs: each Arabic token alone, consecutive others together."""
    runs: list[list[str]] = []
    for t in tokens:
        if not ARABIC.search(t) and runs and not ARABIC.search(runs[-1][0]):
            runs[-1].append(t)
        else:
            runs.append([t])
    return runs


class Writer:
    def __init__(self, pdf: FPDF, font: Path):
        self.pdf = pdf
        self.covered = set(TTFont(font).getBestCmap())
        pdf.add_font("naskh", fname=str(font))

    def font_for(self, text: str) -> str:
        return "naskh" if all(ord(c) in self.covered for c in text) else "helvetica"

    def width(self, text: str, font: str) -> float:
        self.pdf.set_font(font, size=SIZE)
        return self.pdf.get_string_width(text)

    def _segments(self, glyphs: list[str]) -> list[str]:
        """Consecutive glyphs drawn with the same font, logical order."""
        segs: list[str] = []
        for g in glyphs:
            if segs and self.font_for(segs[-1]) == self.font_for(g):
                segs[-1] += g
            else:
                segs.append(g)
        return segs

    def line_ops(self, line: str, right: float, visual: bool) -> list[tuple[float, str, str]]:
        """Text operations (x, text, font) for one right-aligned line, in stream order."""
        space = self.width(" ", "naskh")
        ops: list[tuple[float, str, str]] = []
        x = right
        for run in _runs(line.split()):
            if ARABIC.search(run[0]):
                for seg in self._segments(shape(run[0])):
                    font = self.font_for(seg)
                    if visual:
                        w = self.width(seg, font)
                        ops.append((x - w, seg[::-1].translate(MIRROR), font))
                        x -= w
                    else:
                        for g in seg:
                            x -= self.width(g, font)
                            ops.append((x, g, font))
            else:
                text = " ".join(run)  # a left-to-right run, e.g. "S&P 500"
                font = self.font_for(text)
                x -= self.width(text, font)
                ops.append((x, text, font))
            x -= space
            ops.append((x, " ", "naskh"))
        ops.pop()  # no trailing space
        return sorted(ops) if visual else ops

    def lines(self, lines: list[str], right: float, y: float, visual: bool) -> float:
        for line in lines:
            self.ops(self.line_ops(line, right, visual), y)
            y += LINE
        return y

    def ops(self, ops: list[tuple[float, str, str]], y: float) -> None:
        for x, text, font in ops:
            self.pdf.set_font(font, size=SIZE)
            self.pdf.text(x, y, text)


def write_table(w: Writer, rows: list[list[str]], y: float, visual: bool) -> None:
    pdf = w.pdf
    left, right = pdf.l_margin, pdf.w - pdf.r_margin
    width = (right - left) / len(rows[0])
    for r, row in enumerate(rows):
        top = y + r * LINE
        for c, cell in enumerate(row):  # cells right to left
            cell_right = right - c * width
            pdf.rect(cell_right - width, top, width, LINE)
            w.ops(w.line_ops(cell, cell_right - 2, visual), top + LINE * 0.7)


def build(path: Path, font: Path, visual: bool) -> None:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(False)
    w = Writer(pdf, font)
    right = pdf.w - pdf.r_margin
    mid = pdf.w / 2

    pdf.add_page()
    w.lines([content.TITLE] + content.BODY, right, 30, visual)

    pdf.add_page()
    end = w.lines(content.RIGHT_COLUMN, right, 30, visual)
    w.lines(content.LEFT_COLUMN, mid - 8, 30, visual)
    write_table(w, content.TABLE, end + LINE, visual)

    pdf.output(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", required=True, type=Path, help="NotoNaskhArabic-Regular.ttf")
    args = parser.parse_args()
    for name, visual in (("visual-order.pdf", True), ("logical-order.pdf", False)):
        build(HERE / name, args.font, visual)
        print(HERE / name, (HERE / name).stat().st_size, "bytes")


if __name__ == "__main__":
    main()

"""Arabic for terminals that don't shape or reorder it themselves (many Windows
consoles show Arabic letters unjoined and in the wrong order).

visual(line) returns the line as such a terminal should draw it: Arabic words in
their joined presentation forms, laid out right to left; runs of Latin letters and
digits stay left to right. Use it only for display, never for stored text.
"""

import re
import unicodedata

ARABIC = re.compile(r"[؀-ۿ]")

# Presentation forms from Unicode data: base letter -> {form: code point}.
FORMS: dict[str, dict[str, str]] = {}
for _cp in range(0xFE70, 0xFEFD):
    _d = unicodedata.decomposition(chr(_cp)).split()
    if len(_d) == 2 and _d[0] in ("<isolated>", "<final>", "<initial>", "<medial>"):
        FORMS.setdefault(chr(int(_d[1], 16)), {})[_d[0][1:-1]] = chr(_cp)
LAM_ALEF = {"ا": "ﻻ", "أ": "ﻷ", "إ": "ﻹ", "آ": "ﻵ"}  # isolated; +1 is final
MIRROR = str.maketrans("()[]{}<>«»", ")(][}{><»«")


def _joins_next(ch: str) -> bool:
    return "medial" in FORMS.get(ch, {})


def shape(word: str) -> str:
    """A logical-order Arabic word in presentation forms (still logical order)."""
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
    return "".join(out)


LTR = re.compile(r"[A-Za-z0-9٠-٩۰-۹]")     # Latin letters and digits (Arabic-Indic too)
OPEN, CLOSE = "([{", ")]}"
UNITS = "%٪$€£"


def _is_rtl(ch: str) -> bool:
    return bool(ARABIC.match(ch)) and not LTR.match(ch)


def _ltr_spans(line: str) -> list[tuple[int, int]]:
    """Left-to-right runs: from the first to the last Latin letter or digit between
    two Arabic letters, so "Discount (T+2)" or "S&P 500" stays one run. A bracket
    opened inside a run is closed inside it too."""
    spans, i, n = [], 0, len(line)
    while i < n:
        if _is_rtl(line[i]):
            i += 1
            continue
        j = i
        while j < n and not _is_rtl(line[j]):
            j += 1
        strong = [k for k in range(i, j) if LTR.match(line[k])]
        if strong:
            a, b = strong[0], strong[-1] + 1
            while b < j and line[b] in UNITS:                          # 12.5% or $100 stay with the number
                b += 1
            while a > i and line[a - 1] in UNITS:
                a -= 1
            depth = sum(line[a:b].count(c) for c in OPEN) - sum(line[a:b].count(c) for c in CLOSE)
            k = b
            while depth > 0 and k < j:
                if line[k] in CLOSE:
                    depth, b = depth - 1, k + 1
                k += 1
            k = a - 1
            while depth < 0 and k >= i:
                if line[k] in OPEN:
                    depth, a = depth + 1, k
                k -= 1
            spans.append((a, b))
        i = j
    return spans


def visual(line: str) -> str:
    if not ARABIC.search(line):
        return line
    pieces, pos = [], 0
    for a, b in _ltr_spans(line):
        if pos < a:
            pieces.append(shape(line[pos:a])[::-1].translate(MIRROR))
        pieces.append(line[a:b])
        pos = b
    if pos < len(line):
        pieces.append(shape(line[pos:])[::-1].translate(MIRROR))
    return "".join(reversed(pieces))

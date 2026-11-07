"""PDF -> per-page text, with pluggable extractor backends.

Each backend is a function that takes a PDF path and returns one Page per page.
Backends import their library lazily, so only the one you use needs installing.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NamedTuple


class Line(NamedTuple):
    text: str
    top: float
    bottom: float


@dataclass
class Page:
    text: str
    flags: list[str] = field(default_factory=list)
    # Only backends that know positions fill these; header/footer removal needs them.
    lines: list[Line] | None = None
    height: float | None = None


Backend = Callable[[Path], list[Page]]

BACKENDS: dict[str, Backend] = {}
LICENSES: dict[str, str] = {}

TATWEEL = "ـ"
HARAKAT = re.compile(r"[ً-ْٰ]")


def register(name: str, license: str):
    def wrap(fn: Backend) -> Backend:
        BACKENDS[name] = fn
        LICENSES[name] = license
        return fn
    return wrap


@register("pdfplumber-rtl", "MIT")
def _pdfplumber_rtl(path: Path) -> list[Page]:
    import pdfplumber

    from mukhtabar.ingest.rtl_layout import layout_page, resolve_brackets

    with pdfplumber.open(path) as pdf:
        laid_out = [(*layout_page(page), page.height) for page in pdf.pages]
    texts = resolve_brackets(["\n".join(t for t, _, _ in lines) for lines, _, _ in laid_out])
    pages = []
    for text, (lines, flags, height) in zip(texts, laid_out):
        resolved = [Line(t, top, bottom) for t, (_, top, bottom) in zip(text.split("\n"), lines)]
        pages.append(Page(text, flags, resolved, height))
    return pages


@register("pypdf", "BSD-3-Clause")
def _pypdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    return [Page(page.extract_text() or "") for page in PdfReader(path).pages]


@register("pdfplumber", "MIT")
def _pdfplumber(path: Path) -> list[Page]:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return [Page(page.extract_text() or "") for page in pdf.pages]


@register("pypdfium2", "Apache-2.0 / BSD-3-Clause")
def _pypdfium2(path: Path) -> list[Page]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    pages = []
    try:
        for page in pdf:
            textpage = page.get_textpage()
            pages.append(Page(textpage.get_text_bounded()))
            textpage.close()
            page.close()
    finally:
        pdf.close()
    return pages


@register("pymupdf", "AGPL-3.0")
def _pymupdf(path: Path) -> list[Page]:
    import pymupdf

    with pymupdf.open(path) as doc:
        return [Page(page.get_text()) for page in doc]


def normalize(text: str) -> str:
    """Map Arabic presentation forms (shaped glyphs, ligatures like ﻻ) back to
    base letters, then strip harakat and tatweel. Alef/hamza/ya/ta marbuta are
    left as written. Not suitable for Quranic text or poetry, where harakat matter."""
    text = unicodedata.normalize("NFKC", text)
    text = HARAKAT.sub("", text.replace(TATWEEL, ""))
    text = re.sub(r"[ \t]+", " ", text)
    # Glyphs from a second font (e.g. Latin brackets) can come out as separate words.
    # Only tighten brackets that sit where an opening/closing bracket belongs, so a
    # misoriented one never glues two words together.
    text = re.sub(r"(^|(?<= ))([(\[«]) ", r"\2", text, flags=re.M)
    text = re.sub(r" ([)\]»])(?=$|[ .,،؛:؟!])", r"\1", text, flags=re.M)
    return "\n".join(line.strip() for line in text.splitlines())


DEFAULT_BACKEND = "pdfplumber-rtl"


def extract_pages(path: str | Path, backend: str = DEFAULT_BACKEND, clean: bool = True) -> list[Page]:
    pages = BACKENDS[backend](Path(path))
    if clean:
        for p in pages:
            if p.lines is not None:
                p.lines = [Line(normalize(l.text), l.top, l.bottom) for l in p.lines]
                p.text = "\n".join(l.text for l in p.lines)
            else:
                p.text = normalize(p.text)
    return pages

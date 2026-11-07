"""PDF -> per-page text, with pluggable extractor backends.

Each backend is a function that takes a PDF path and returns one Page per page.
Backends import their library lazily, so only the one you use needs installing.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Page:
    text: str
    flags: list[str] = field(default_factory=list)


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
        laid_out = [layout_page(page) for page in pdf.pages]
    texts = resolve_brackets([text for text, _ in laid_out])
    return [Page(text, flags) for text, (_, flags) in zip(texts, laid_out)]


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
    return "\n".join(line.strip() for line in text.splitlines())


DEFAULT_BACKEND = "pdfplumber-rtl"


def extract_pages(path: str | Path, backend: str = DEFAULT_BACKEND, clean: bool = True) -> list[Page]:
    pages = BACKENDS[backend](Path(path))
    if clean:
        for p in pages:
            p.text = normalize(p.text)
    return pages

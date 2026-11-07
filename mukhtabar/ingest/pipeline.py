"""PDF -> pages -> header/footer removal -> chunks -> SQLite."""

import hashlib
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from mukhtabar import db
from mukhtabar.ingest.chunk import TARGET_CHARS, chunk_pages
from mukhtabar.ingest.extract import extract_pages
from mukhtabar.ingest.headers import remove_headers_footers

ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓ]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
ENGLISH_WORDS = {"the", "and", "of", "to", "in", "is", "a", "for", "that", "with"}
ARABIC_SHARE = 0.2   # Arabic with English terms mixed in still counts as Arabic
ENGLISH_SHARE = 0.03


@dataclass
class IngestResult:
    filename: str
    skipped: bool = False         # already in this study set
    language: str = ""
    pages: int = 0
    blank_pages: int = 0
    flagged_pages: list[int] = field(default_factory=list)
    removed_lines: int = 0
    kept_repeats: list[tuple[str, str, int]] = field(default_factory=list)
    chunks: int = 0


def guess_language(text: str) -> str:
    """"ar" if a meaningful share of letters is Arabic, "en" for English, else "und"."""
    arabic = len(ARABIC_LETTER.findall(text))
    latin = len(LATIN_LETTER.findall(text))
    if arabic + latin == 0:
        return "und"
    if arabic >= ARABIC_SHARE * (arabic + latin):
        return "ar"
    words = re.findall(r"[A-Za-z]+", text.lower())
    if words and sum(w in ENGLISH_WORDS for w in words) >= ENGLISH_SHARE * len(words):
        return "en"
    return "und"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest_pdf(conn: sqlite3.Connection, set_name: str, path: str | Path,
               language: str | None = None, target_chars: int = TARGET_CHARS) -> IngestResult:
    """Add one PDF to a study set, creating the set if needed. A file already in
    the set (same content) is skipped. language overrides the guess."""
    path = Path(path)
    # Store names in NFC: Arabic names can be composed differently on disk and when typed.
    result = IngestResult(unicodedata.normalize("NFC", path.name))
    sha = _sha256(path)
    study_set = db.get_study_set(conn, set_name)
    if study_set and db.find_document(conn, study_set["id"], sha):
        result.skipped = True
        return result

    pages = extract_pages(path)
    pages, report = remove_headers_footers(pages)
    removed_by_page: dict[int, list[str]] = {}
    for r in report.removed:
        removed_by_page.setdefault(r.page, []).append(r.text)

    with conn:
        if study_set is None:
            result.language = language or guess_language("\n".join(p.text for p in pages))
            set_id = db.create_study_set(conn, set_name, result.language)
        else:
            set_id = study_set["id"]
            result.language = language or study_set["language"]
            if language and language != study_set["language"]:
                db.set_language(conn, set_id, language)
        doc_id = db.add_document(conn, set_id, result.filename, sha, len(pages))
        db.add_pages(conn, doc_id, [(i + 1, p.text, p.flags, removed_by_page.get(i + 1, []))
                                    for i, p in enumerate(pages)])
        chunks = chunk_pages([(i + 1, p.text) for i, p in enumerate(pages)], target_chars)
        db.replace_chunks(conn, set_id, doc_id, chunks)

    result.pages = len(pages)
    result.blank_pages = sum(not p.text.strip() for p in pages)
    result.flagged_pages = [i + 1 for i, p in enumerate(pages) if p.flags]
    result.removed_lines = len(report.removed)
    result.kept_repeats = report.kept
    result.chunks = len(chunks)
    return result


def rechunk(conn: sqlite3.Connection, set_name: str, target_chars: int = TARGET_CHARS) -> dict[str, int]:
    """Rebuild every document's chunks from the stored pages (no re-extraction).
    Returns chunk counts by filename."""
    study_set = db.get_study_set(conn, set_name)
    if study_set is None:
        raise ValueError(f"No study set named {set_name!r}")
    counts = {}
    with conn:
        for doc in db.documents(conn, study_set["id"]):
            stored = [(p["page"], p["text"]) for p in db.pages(conn, doc["id"])]
            chunks = chunk_pages(stored, target_chars)
            db.replace_chunks(conn, study_set["id"], doc["id"], chunks)
            counts[doc["filename"]] = len(chunks)
    return counts

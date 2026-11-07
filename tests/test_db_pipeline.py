import json

import pytest

from mukhtabar import db
from mukhtabar.ingest import pipeline
from mukhtabar.ingest.extract import Line, Page
from mukhtabar.ingest.pipeline import guess_language, ingest_pdf, rechunk

SENTENCE = "يشرح هذا الدرس طريقة حساب العائد على الاستثمار خطوة بخطوة."


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "test.db")


@pytest.fixture
def fake_pdf(tmp_path, monkeypatch):
    """A file on disk for hashing, with extraction replaced by fixed pages."""
    def make(name: str, content: bytes = b"%PDF-fake", n_pages: int = 3):
        path = tmp_path / name
        path.write_bytes(content)
        pages = []
        for n in range(1, n_pages + 1):
            lines = [Line(SENTENCE, 300 + 12 * k, 310 + 12 * k) for k in range(8)] + [Line(str(n), 785, 795)]
            pages.append(Page("\n".join(l.text for l in lines), lines=lines, height=800))
        monkeypatch.setattr(pipeline, "extract_pages", lambda p: pages)
        return path
    return make


def test_migrate_sets_version_and_is_idempotent(tmp_path):
    path = tmp_path / "v.db"
    db.connect(path).close()
    conn = db.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS)


def test_ingest_stores_document_pages_and_chunks(conn, fake_pdf):
    r = ingest_pdf(conn, "Set A", fake_pdf("book.pdf"))
    assert (r.language, r.pages, r.removed_lines) == ("ar", 3, 3)
    assert r.chunks > 0
    doc = db.documents(conn, db.get_study_set(conn, "Set A")["id"])[0]
    stored = db.pages(conn, doc["id"])
    assert [p["page"] for p in stored] == [1, 2, 3]
    assert json.loads(stored[0]["removed"]) == ["1"]
    chunks = db.chunks(conn, doc["id"])
    assert len(chunks) == r.chunks
    assert all(c["page"] <= c["page_end"] for c in chunks)


def test_same_file_twice_is_skipped(conn, fake_pdf):
    path = fake_pdf("book.pdf")
    ingest_pdf(conn, "Set A", path)
    assert ingest_pdf(conn, "Set A", path).skipped


def test_filename_stored_in_nfc(conn, fake_pdf):
    import unicodedata
    # Shadda before fatha: a valid but non-canonical order that NFC rewrites.
    decomposed = "المعدَّل.pdf"
    assert unicodedata.normalize("NFC", decomposed) != decomposed
    r = ingest_pdf(conn, "Set A", fake_pdf(decomposed))
    assert r.filename == unicodedata.normalize("NFC", decomposed)


def test_language_override_and_update(conn, fake_pdf):
    r = ingest_pdf(conn, "S", fake_pdf("a.pdf", b"a"), language="en")
    assert r.language == "en"
    ingest_pdf(conn, "S", fake_pdf("b.pdf", b"b"), language="ar")
    assert db.get_study_set(conn, "S")["language"] == "ar"


def test_rechunk_uses_stored_pages(conn, fake_pdf, monkeypatch):
    ingest_pdf(conn, "Set A", fake_pdf("book.pdf"))
    monkeypatch.setattr(pipeline, "extract_pages", lambda p: pytest.fail("rechunk must not re-extract"))
    small = rechunk(conn, "Set A", target_chars=150)
    large = rechunk(conn, "Set A", target_chars=5000)
    assert small["book.pdf"] > large["book.pdf"] == 1


@pytest.mark.parametrize("text, expected", [
    ("يشرح هذا الدرس طريقة حساب العائد خطوة بخطوة", "ar"),
    ("يقارن الدرس بين مؤشرين (ABC) و XYZ 100 في سوقين مختلفتين", "ar"),
    ("The market is a place for buyers and sellers of the assets", "en"),
    ("Le marché financier est un lieu de rencontre", "und"),
    ("12 34", "und"),
])
def test_guess_language(text, expected):
    assert guess_language(text) == expected


def test_llm_call_rejections_are_stored(conn):
    sid = db.create_study_set(conn, "R", "ar")
    db.log_llm_call(conn, sid, {"purpose": "p", "model": "m", "attempts": 2, "ok": True,
                                "rejections": ["question 1: \"evidence\" must be copied word for word"]})
    stored = conn.execute("SELECT rejections FROM llm_calls").fetchone()[0]
    assert json.loads(stored) == ["question 1: \"evidence\" must be copied word for word"]

"""Extraction on the synthetic Arabic PDFs (see tests/fixtures/make_fixtures.py)."""

import sys
from pathlib import Path

import pytest

from mukhtabar.ingest.extract import extract_pages

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))
import content  # noqa: E402

PDFS = ["visual-order.pdf", "logical-order.pdf"]


@pytest.fixture(scope="module", params=PDFS)
def pages(request):
    return extract_pages(FIXTURES / request.param)


def test_running_text_in_reading_order(pages):
    # Covers لا ligatures, brackets stored by shape (visual) or meaning (logical),
    # and an English/number run inside an Arabic line.
    assert pages[0].text.splitlines() == [content.TITLE] + content.BODY


def test_two_columns_then_table(pages):
    table = [" | ".join(row) for row in content.TABLE]
    assert pages[1].text.splitlines() == content.RIGHT_COLUMN + content.LEFT_COLUMN + table


def test_no_pages_flagged(pages):
    assert all(not p.flags for p in pages)


# The fixtures are only useful if they behave like real files under a naive extractor.

def test_visual_fixture_reproduces_reversed_word_order():
    pypdf = pytest.importorskip("pypdf")
    from mukhtabar.ingest.extract import normalize
    lines = normalize(pypdf.PdfReader(FIXTURES / "visual-order.pdf").pages[0].extract_text()).splitlines()
    assert lines[1].split() == content.BODY[0].split()[::-1]


def test_logical_fixture_reads_correctly_with_naive_extraction():
    pypdf = pytest.importorskip("pypdf")
    from mukhtabar.ingest.extract import normalize
    lines = normalize(pypdf.PdfReader(FIXTURES / "logical-order.pdf").pages[0].extract_text()).splitlines()
    assert lines[1] == content.BODY[0]

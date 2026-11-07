"""Build a visual review pack: rendered PDF pages next to their stored extracted text.

    python spikes/review_pack.py DB PDF_DIR OUT_DIR [--page FILENAME:N ...]

For each document in DB (found by filename in PDF_DIR), picks about 8 pages: a
two-column page, a table page, a page with English terms or numbers, an MCQ page,
up to 3 flagged pages, any pages asked for with --page, and evenly spaced pages to
fill. Writes OUT_DIR/index.html and OUT_DIR/img/*.png. Everything stays local;
keep OUT_DIR somewhere private, since the pack shows your material.
"""

import argparse
import html
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mukhtabar.ingest import rtl_layout  # noqa: E402

PAGES_PER_DOC = 8
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
NUMBER = re.compile(r"\d")
OPTION_LINE = re.compile(r"^[أبجد][ .)\-]", re.M)


def gutter_pages(pdf_path: Path) -> set[int]:
    """1-based pages where the layout split two columns."""
    found: set[int] = set()
    original = rtl_layout._find_gutter
    current = {"page": 0}

    def spy(*args):
        g = original(*args)
        if g is not None:
            found.add(current["page"])
        return g

    rtl_layout._find_gutter = spy
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                current["page"] = i
                rtl_layout.layout_page(page)
    finally:
        rtl_layout._find_gutter = original
    return found


def pick_pages(pages: list[sqlite3.Row], gutters: set[int], requested: list[int]) -> dict[int, list[str]]:
    """page number -> reasons it was picked."""
    texts = {p["page"]: p["text"] for p in pages}
    nonblank = [n for n, t in texts.items() if len(t.strip()) > 200]
    picks: dict[int, list[str]] = {}

    def add(n, reason):
        if n is not None:
            picks.setdefault(n, []).append(reason)

    def best(score, minimum=1):
        scored = [(score(texts[n]), n) for n in nonblank]
        top = max(scored, default=(0, None))
        return top[1] if top[0] >= minimum else None

    add(max((n for n in gutters if n in nonblank), key=lambda n: len(texts[n]), default=None), "two columns")
    add(best(lambda t: t.count(" | "), 2), "table")
    add(best(lambda t: len(LATIN_WORD.findall(t)) * 3 + len(NUMBER.findall(t)), 5), "English terms / numbers")
    add(best(lambda t: len(OPTION_LINE.findall(t)), 4), "MCQ")
    for p in [p for p in pages if json.loads(p["flags"])][:3]:
        add(p["page"], "flagged: " + ", ".join(json.loads(p["flags"])))
    for n in requested:
        if n in texts:
            add(n, "requested")
    # Fill with evenly spaced pages so ordinary pages are reviewed too.
    step = max(len(nonblank) // PAGES_PER_DOC, 1)
    for n in nonblank[step // 2::step]:
        if len(picks) >= PAGES_PER_DOC:
            break
        if n not in picks:
            add(n, "ordinary page")
    return dict(sorted(picks.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=Path)
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--page", action="append", default=[], metavar="FILENAME:N",
                        help="always include this page of this document (repeatable)")
    args = parser.parse_args()
    db_path, pdf_dir, out = args.db, args.pdf_dir, args.out
    requested: dict[str, list[int]] = {}
    for item in args.page:
        name, _, n = item.rpartition(":")
        requested.setdefault(unicodedata.normalize("NFC", name), []).append(int(n))
    (out / "img").mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    on_disk = {unicodedata.normalize("NFC", p.name): p for p in pdf_dir.glob("*.pdf")}

    sections = []
    for d, doc in enumerate(conn.execute("SELECT * FROM documents ORDER BY id"), start=1):
        pdf_path = on_disk[doc["filename"]]
        pages = conn.execute("SELECT * FROM pages WHERE document_id = ? ORDER BY page", (doc["id"],)).fetchall()
        by_number = {p["page"]: p for p in pages}
        print(f"{doc['filename']}: finding two-column pages...", flush=True)
        picks = pick_pages(pages, gutter_pages(pdf_path), requested.get(doc["filename"], []))
        rendered = pdfium.PdfDocument(str(pdf_path))
        cards = []
        for n, reasons in picks.items():
            img = f"img/doc{d}_p{n}.png"
            rendered[n - 1].render(scale=1.4).to_pil().save(out / img)
            page = by_number[n]
            removed = json.loads(page["removed"])
            cards.append(f"""
<section class="card">
  <h3>Page {n} <span class="tags">{' · '.join(html.escape(r) for r in reasons)}</span></h3>
  <div class="pair">
    <a href="{img}" target="_blank"><img src="{img}" alt="page {n}" loading="lazy"></a>
    <div class="text">
      <pre dir="rtl" lang="ar">{html.escape(page['text']) or '<em>(no text)</em>'}</pre>
      {'<p class="removed">Removed as header/footer: ' + ', '.join(html.escape(r) for r in removed) + '</p>' if removed else ''}
    </div>
  </div>
</section>""")
        sections.append(f"<h2>{html.escape(doc['filename'])}</h2>" + "".join(cards))
        print(f"  pages: {list(picks)}", flush=True)

    (out / "index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Extraction review</title>
<style>
  :root {{ --bg: #fafaf8; --fg: #1d1d1b; --muted: #6b6b66; --card: #fff; --line: #e2e2dc; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg: #161615; --fg: #ececea; --muted: #a0a09a; --card: #1f1f1d; --line: #33332f; }} }}
  body {{ margin: 0; padding: 16px; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; }}
  h1 {{ font-size: 1.4rem; }} h2 {{ margin-top: 2.5rem; border-bottom: 1px solid var(--line); padding-bottom: .3rem; }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin: 16px 0; }}
  .card h3 {{ margin: 0 0 8px; font-size: 1rem; }} .tags {{ color: var(--muted); font-weight: normal; }}
  .pair {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; align-items: start; }}
  @media (max-width: 800px) {{ .pair {{ grid-template-columns: 1fr; }} }}
  img {{ width: 100%; height: auto; border: 1px solid var(--line); background: #fff; }}
  pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; font: 16px/1.7 "Segoe UI", "Noto Naskh Arabic", Tahoma, sans-serif;
        max-height: 1100px; overflow: auto; }}
  .removed {{ color: var(--muted); font-size: .85rem; }}
</style></head><body>
<h1>Extraction review</h1>
<p>Each page is shown as rendered (left) and as stored after extraction, normalization and header/footer
removal (right). Click an image to open it full size. Harakat and tatweel are stripped by design.</p>
{''.join(sections)}
</body></html>""", encoding="utf-8")
    print(f"Wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()

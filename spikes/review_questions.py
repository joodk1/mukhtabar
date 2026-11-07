"""Review generated questions next to their source: chunk text with the evidence
highlighted, and the rendered page of the PDF.

    python spikes/review_questions.py DB PDF_DIR OUT_DIR (--topic ID ... | --question ID ...)

--topic shows all of a topic's questions; --question shows just those questions,
grouped by topic.

Writes OUT_DIR/index.html and OUT_DIR/img/*.png. The page shows your material, so
keep OUT_DIR somewhere private.
"""

import argparse
import html
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

import pypdfium2 as pdfium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mukhtabar.questions import canon  # noqa: E402

LETTERS = "أبجد"


def highlight(text: str, evidence: str) -> str:
    """Escape text and wrap the evidence in <mark>, matching it the way the
    grounding check does (ignoring spacing and punctuation)."""
    words = canon(evidence).split()
    if words:
        pattern = r"[\W_]*".join(re.escape(w) for w in words)
        m = re.search(pattern, text, re.I)
        if m:
            return (html.escape(text[:m.start()]) + "<mark>" + html.escape(m.group(0)) + "</mark>"
                    + html.escape(text[m.end():]))
    return html.escape(text) + '<p class="warn">(evidence not located for highlighting)</p>'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=Path)
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--topic", type=int, action="append", default=[])
    parser.add_argument("--question", type=int, action="append", default=[])
    args = parser.parse_args()
    (args.out / "img").mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    pdfs = {unicodedata.normalize("NFC", p.name): p for p in args.pdf_dir.glob("*.pdf")}
    rendered: dict[str, pdfium.PdfDocument] = {}
    e = html.escape
    sections = []
    wanted = set(args.question)
    topic_ids = list(args.topic) + [r[0] for r in conn.execute(
        f"""SELECT DISTINCT q.topic_id FROM questions q JOIN topics t ON t.id = q.topic_id
            WHERE q.id IN ({','.join('?' * len(wanted)) or 'NULL'}) ORDER BY t.position""", sorted(wanted))
        if r[0] not in args.topic]
    if not topic_ids:
        parser.error("give --topic or --question")
    for tid in topic_ids:
        topic = conn.execute("SELECT * FROM topics WHERE id = ?", (tid,)).fetchone()
        qs = [q for q in conn.execute("SELECT * FROM questions WHERE topic_id = ? ORDER BY id", (tid,)).fetchall()
              if tid in args.topic or q["id"] in wanted]
        cards = []
        for n, q in enumerate(qs, start=1):
            c = conn.execute("""SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id = c.document_id
                                WHERE c.id = ?""", (q["chunk_id"],)).fetchone()
            options = json.loads(q["options"])
            answer = "<ol class=opts>" + "".join(
                f"<li class='{'right' if i == int(q['answer']) else ''}'>{LETTERS[i]}. {e(o)}"
                f"{' ✓' if i == int(q['answer']) else ''}</li>" for i, o in enumerate(options)) + "</ol>"
            img = ""
            if c is not None and c["filename"] in pdfs:
                name = f"img/{tid}_{q['id']}_p{c['page']}.png"
                doc = rendered.setdefault(c["filename"], pdfium.PdfDocument(str(pdfs[c["filename"]])))
                doc[c["page"] - 1].render(scale=1.3).to_pil().save(args.out / name)
                img = f'<a href="{name}" target="_blank"><img src="{name}" alt="page {c["page"]}" loading="lazy"></a>'
            source = (f"<p class=muted>chunk {c['id']} · {e(c['filename'][:24])}… · page {c['page']}–{c['page_end']}</p>"
                      f"<div class=chunk dir=rtl>{highlight(c['text'], q['evidence'])}</div>") if c else "<p>(source chunk missing)</p>"
            cards.append(f"""
<section class=card>
  <h3>{n}. <span class=tag>question {q['id']}</span></h3>
  <div class=grid>
    <div dir=rtl class=q>
      <p class=prompt>{e(q['prompt'])}</p>{answer}
      <p><b>الشرح:</b> {e(q['explanation'])}</p>
      <p><b>الدليل:</b> <span class=evidence>{e(q['evidence'])}</span></p>
      <p class=judge>Check: correct per the book? ☐ yes ☐ no · grounded in the chunk? ☐ yes ☐ no · new (not copied)? ☐ yes ☐ no</p>
    </div>
    <div>{source}</div>
    <div>{img}</div>
  </div>
</section>""")
        sections.append(f"<h2 dir=rtl>{e(topic['title'])} <span class=muted>(topic {tid}, {e(topic['chapter'] or '')})</span></h2>"
                        + ("".join(cards) or "<p>No questions yet.</p>"))
    (args.out / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Question review</title>
<style>
 :root {{ --bg:#fafaf8; --fg:#1d1d1b; --muted:#6b6b66; --line:#e2e2dc; --card:#fff; --mark:#fff2a8; --right:#1a7f37; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --fg:#ececea; --muted:#a0a09a; --line:#33332f; --card:#1f1f1d; --mark:#5c5200; --right:#56d364; }} }}
 body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:15px/1.6 system-ui, "Segoe UI", Tahoma, sans-serif; }}
 .card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin:14px 0; }}
 .grid {{ display:grid; grid-template-columns:minmax(0,1.1fr) minmax(0,1fr) minmax(0,.9fr); gap:14px; align-items:start; }}
 @media (max-width:1000px) {{ .grid {{ grid-template-columns:1fr; }} }}
 .prompt {{ font-size:17px; font-weight:600; }} .q, .chunk {{ font-size:16px; }}
 .chunk {{ white-space:pre-wrap; border-inline-start:3px solid var(--line); padding-inline-start:10px; max-height:520px; overflow:auto; }}
 mark {{ background:var(--mark); color:inherit; }} .right {{ color:var(--right); font-weight:600; }}
 .muted, .judge, .tag {{ color:var(--muted); font-size:.85rem; }} .warn {{ color:#b35900; }}
 img {{ width:100%; height:auto; border:1px solid var(--line); background:#fff; }}
 .evidence {{ background:var(--mark); }}
</style></head><body>
<h1>Question review</h1>
<p>Each question (left) is shown with its source chunk, the evidence highlighted (middle), and the book page (right).
Check that the answer is right according to the book, that the evidence supports it, and that the question is new.</p>
{''.join(sections)}
</body></html>""", encoding="utf-8")
    print(f"Wrote {args.out / 'index.html'}")


if __name__ == "__main__":
    main()

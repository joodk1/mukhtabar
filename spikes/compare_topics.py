"""Compare topic maps built by different models on the same chunks.

    python spikes/compare_topics.py OUT.html DB_A DB_B [DB_C ...]

Each DB is a copy of the same ingested database with a topic map built by one
model (chunk ids match across copies). Writes one local HTML page with, per
model: usage and validity from llm_calls, topic counts and sizes, excluded
chunks, titles in the wrong language; the outlines side by side; and the same
10 randomly sampled chunks with the topic each model put them in, for you to
judge. The page shows your material, so keep OUT.html somewhere private.
"""

import html
import random
import re
import sqlite3
import statistics
import sys
from pathlib import Path

ARABIC = re.compile(r"[ء-ي]")
LATIN = re.compile(r"[A-Za-z]")
SAMPLE = 10


def load(path: Path) -> dict:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    calls = conn.execute("SELECT * FROM llm_calls ORDER BY id").fetchall()
    topics = conn.execute(
        """SELECT t.*, COUNT(tc.chunk_id) AS n FROM topics t LEFT JOIN topic_chunks tc ON tc.topic_id = t.id
           GROUP BY t.id ORDER BY t.position""").fetchall()
    links = {r["chunk_id"]: r["topic_id"] for r in conn.execute("SELECT * FROM topic_chunks WHERE is_primary = 1")}
    excluded = {r["id"]: r["excluded_reason"] for r in conn.execute("SELECT id, excluded_reason FROM chunks WHERE excluded = 1")}
    spans = {}
    for t in topics:
        rows = conn.execute(
            """SELECT d.filename, MIN(c.page) AS a, MAX(c.page_end) AS b, COUNT(*) AS n FROM topic_chunks tc
               JOIN chunks c ON c.id = tc.chunk_id JOIN documents d ON d.id = c.document_id
               WHERE tc.topic_id = ? GROUP BY d.id ORDER BY d.id""", (t["id"],)).fetchall()
        spans[t["id"]] = "; ".join(f"{r['filename'][:18]}… p{r['a']}–{r['b']} ({r['n']})" for r in rows)
    chunks = {r["id"]: r for r in conn.execute(
        "SELECT c.id, c.page, c.page_end, c.text, d.filename FROM chunks c JOIN documents d ON d.id = c.document_id")}
    model = calls[0]["model"] if calls else path.stem
    return {"path": path, "model": model, "calls": calls, "topics": topics, "links": links,
            "excluded": excluded, "spans": spans, "chunks": chunks}


def wrong_language(text: str) -> bool:
    a, l = len(ARABIC.findall(text)), len(LATIN.findall(text))
    return a + l > 0 and a < 0.5 * (a + l)


def metrics(m: dict) -> list[tuple[str, str]]:
    calls, topics = m["calls"], m["topics"]
    sizes = [t["n"] for t in topics] or [0]
    cost = sum(c["cost_usd"] or 0 for c in calls)
    return [
        ("Model calls", str(len(calls))),
        ("Valid on first attempt", f"{sum(1 for c in calls if c['ok'] and c['attempts'] == 1)} / {len(calls)}"),
        ("Retries used", str(sum(max(c["attempts"] - 1, 0) for c in calls))),
        ("Failed calls", str(sum(1 for c in calls if not c["ok"]))),
        ("Tokens in / out", f"{sum(c['input_tokens'] or 0 for c in calls):,} / {sum(c['output_tokens'] or 0 for c in calls):,}"),
        ("Cost (LiteLLM's pricing)", f"${cost:.4f}"),
        ("Time", f"{sum(c['seconds'] or 0 for c in calls):.0f} s"),
        ("Topics", str(len(topics))),
        ("Chunks per topic (min / median / max)", f"{min(sizes)} / {statistics.median(sizes):g} / {max(sizes)}"),
        ("Excluded as not-study", str(len(m["excluded"]))),
        ("Titles/summaries not in Arabic", str(sum(wrong_language(t["title"] + " " + t["summary"]) for t in topics))),
    ]


def main() -> None:
    out = Path(sys.argv[1])
    models = [load(Path(p)) for p in sys.argv[2:]]
    titles = [{t["id"]: t["title"] for t in m["topics"]} for m in models]
    # Sample among chunks every model placed in a topic, the same ones for all.
    common = set.intersection(*(set(m["links"]) for m in models)) if models else set()
    sample = sorted(random.Random(7).sample(sorted(common), min(SAMPLE, len(common))))

    e = html.escape
    head = "".join(f"<th>{e(m['model'])}</th>" for m in models)
    metric_rows = "".join(
        f"<tr><th>{e(name)}</th>{''.join(f'<td>{e(metrics(m)[i][1])}</td>' for m in models)}</tr>"
        for i, (name, _) in enumerate(metrics(models[0])))
    outlines = "".join(
        f"""<div class="col"><h3>{e(m['model'])}</h3><ol>{''.join(
            f"<li><b>{e(t['title'])}</b> <span class=muted>[{t['n']} chunks · {e(t['chapter'] or '')}]</span>"
            f"<br>{e(t['summary'])}<br><span class=muted>{e(m['spans'][t['id']])}</span></li>"
            for t in m['topics'])}</ol>
            <p class=muted>Excluded: {e(', '.join(f"chunk {k} ({v})" for k, v in m['excluded'].items()) or 'none')}</p></div>"""
        for m in models)
    sample_rows = ""
    for cid in sample:
        c = models[0]["chunks"][cid]
        cells = "".join(f"<td dir=rtl>{e(titles[i].get(m['links'][cid], '?'))}<br>"
                        f"<span class=judge>right topic? ☐ yes ☐ no</span></td>" for i, m in enumerate(models))
        sample_rows += (f"<tr><td dir=rtl class=chunk><span class=muted>chunk {cid} · {e(c['filename'][:18])}… "
                        f"p{c['page']}–{c['page_end']}</span><br>{e(c['text'][:420])}…</td>{cells}</tr>")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Topic map comparison</title>
<style>
 :root {{ --bg:#fafaf8; --fg:#1d1d1b; --muted:#6b6b66; --line:#e2e2dc; --card:#fff; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --fg:#ececea; --muted:#a0a09a; --line:#33332f; --card:#1f1f1d; }} }}
 body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui, "Segoe UI", Tahoma, sans-serif; }}
 table {{ border-collapse:collapse; width:100%; background:var(--card); }}
 th, td {{ border:1px solid var(--line); padding:6px 8px; vertical-align:top; text-align:start; }}
 .cols {{ display:grid; grid-template-columns:repeat({len(models)}, minmax(0,1fr)); gap:16px; }}
 @media (max-width:800px) {{ .cols {{ grid-template-columns:1fr; }} }}
 .col {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 12px; }}
 ol {{ padding-inline-start:1.4em; }} li {{ margin-bottom:.6em; }}
 [dir=rtl], .col li {{ font-size:16px; }} .col {{ direction:rtl; }} .col h3 {{ direction:ltr; }}
 .muted {{ color:var(--muted); font-size:.85em; }} .chunk {{ width:45%; }} .judge {{ color:var(--muted); font-size:.85em; }}
</style></head><body>
<h1>Topic map comparison</h1>
<h2>Numbers</h2><table><tr><th></th>{head}</tr>{metric_rows}</table>
<h2>Same {len(sample)} chunks, the topic each model chose</h2>
<p class=muted>Judge each: is the chunk in the right topic? The checkboxes are for your notes only.</p>
<table><tr><th>Chunk</th>{head}</tr>{sample_rows}</table>
<h2>Outlines</h2>
<p class=muted>Check: are topics study-sized (not a whole chapter, not one fact)? Any duplicates? Any topic that isn't in the material?</p>
<div class="cols">{outlines}</div>
</body></html>""", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

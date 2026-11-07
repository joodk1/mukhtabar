"""Audit stored chunks for anomalies.

    python spikes/audit_chunks.py DB OUT.md

Checks: very short / very long chunks, chunks that start or end mid-sentence,
leftover presentation forms, bracket placeholders, harakat or tatweel, signs of
reversed text, and exact or near-duplicate chunks. Writes counts plus a few
examples of each to OUT.md (which contains your text, so keep it private).
"""

import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mukhtabar.ingest.chunk import LIST_MARKER, TARGET_CHARS, TERMINAL  # noqa: E402

SHORT = 200
LONG = int(TARGET_CHARS * 1.5)
PRES_FORM = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
PLACEHOLDER = re.compile(r"[-]")
HARAKAT_TATWEEL = re.compile(r"[ً-ْٰـ]")
ARABIC_WORD = re.compile(r"[ء-ي]+")
EXAMPLES = 4


def excerpt(text: str, where: str = "start", n: int = 160) -> str:
    text = text.replace("\n", " ⏎ ")
    return (text[:n] + " …") if where == "start" else ("… " + text[-n:])


def reversal_signs(text: str) -> tuple[int, int, int]:
    words = ARABIC_WORD.findall(text)
    taa_start = sum(w.startswith("ة") for w in words)
    al_start = sum(w.startswith("ال") and len(w) > 3 for w in words)
    la_end = sum(w.endswith("لا") and len(w) > 3 for w in words)
    return taa_start, al_start, la_end


def shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.sub(r"[\d\W_]+", " ", text).split()
    return {tuple(words[i:i + n]) for i in range(max(len(words) - n + 1, 1))}


def main() -> None:
    db_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id = c.document_id ORDER BY c.document_id, c.position"
    ).fetchall()
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r["filename"]].append(r)

    findings: dict[str, list[str]] = defaultdict(list)

    def note(kind: str, r: sqlite3.Row, detail: str) -> None:
        findings[kind].append(f"{r['filename']} · chunk {r['position']} · p{r['page']}–{r['page_end']} · {detail}")

    for doc_rows in by_doc.values():
        for prev, r in zip([None] + doc_rows[:-1], doc_rows):
            t = r["text"]
            if len(t) < SHORT:
                note("very short (< %d chars)" % SHORT, r, f"{len(t)} chars: {excerpt(t)}")
            if len(t) > LONG:
                note("very long (> %d chars)" % LONG, r, f"{len(t)} chars: {excerpt(t)}")
            last_line = t.rstrip().split("\n")[-1]
            if not t.rstrip().endswith(TERMINAL) and " | " not in last_line:
                note("ends without sentence punctuation", r, excerpt(t, "end"))
            first_line = t.split("\n")[0]
            if (prev is not None and not prev["text"].rstrip().endswith(TERMINAL)
                    and " | " not in prev["text"].rstrip().split("\n")[-1]
                    and not LIST_MARKER.match(first_line) and len(first_line) > 60):
                note("starts mid-sentence (previous chunk unterminated, long first line)", r, excerpt(t))
            if PRES_FORM.search(t):
                note("presentation-form characters left", r, "".join(sorted(set(PRES_FORM.findall(t)))))
            if PLACEHOLDER.search(t):
                note("unresolved bracket placeholders", r, excerpt(t))
            if HARAKAT_TATWEEL.search(t):
                note("harakat/tatweel left", r, excerpt(t))
            taa_start, al_start, la_end = reversal_signs(t)
            if taa_start or (la_end > 3 and la_end > al_start):
                note("signs of reversed text", r, f"ة-initial words {taa_start}, ال…/…لا {al_start}/{la_end}: {excerpt(t)}")

    # Exact and near duplicates across the whole set.
    seen: dict[str, sqlite3.Row] = {}
    for r in rows:
        key = re.sub(r"\s+", " ", r["text"]).strip()
        if key in seen:
            note("exact duplicate", r, f"same as {seen[key]['filename']} chunk {seen[key]['position']}: {excerpt(r['text'])}")
        else:
            seen[key] = r
    sh = [(r, shingles(r["text"])) for r in rows]
    for i, (a, sa) in enumerate(sh):
        for b, sb in sh[i + 1:]:
            if len(sa) < 10 or len(sb) < 10:
                continue
            overlap = len(sa & sb) / min(len(sa), len(sb))
            if overlap >= 0.8 and re.sub(r"\s+", " ", a["text"]) != re.sub(r"\s+", " ", b["text"]):
                note("near duplicate (≥ 80% shared 5-word runs)", b,
                     f"{overlap:.0%} of {a['filename']} chunk {a['position']}: {excerpt(b['text'])}")

    sizes = [len(r["text"]) for r in rows]
    lines = ["# Chunk audit", "",
             f"Database: `{db_path.as_posix()}` · {len(rows)} chunks in {len(by_doc)} documents · "
             f"size min {min(sizes)}, median {sorted(sizes)[len(sizes) // 2]}, max {max(sizes)} chars", "",
             "| check | chunks |", "|---|---|"]
    checks = ["very short (< %d chars)" % SHORT, "very long (> %d chars)" % LONG,
              "ends without sentence punctuation", "starts mid-sentence (previous chunk unterminated, long first line)",
              "presentation-form characters left", "unresolved bracket placeholders", "harakat/tatweel left",
              "signs of reversed text", "exact duplicate", "near duplicate (≥ 80% shared 5-word runs)"]
    for c in checks:
        lines.append(f"| {c} | {len(findings[c])} |")
    for c in checks:
        if findings[c]:
            per_doc = defaultdict(int)
            for f in findings[c]:
                per_doc[f.split(" · ")[0]] += 1
            lines += ["", f"## {c} — {len(findings[c])}", "",
                      "By document: " + ", ".join(f"{k}: {v}" for k, v in per_doc.items()), ""]
            lines += [f"- {f}" for f in findings[c][:EXAMPLES]]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:len(checks) + 6]))


if __name__ == "__main__":
    main()

"""Diagnostics and the weakness report, in the terminal (the UI comes in milestone 5).

    python -m mukhtabar.quiz diagnostic --set NAME [--mode fixed|adaptive] [--n 30] [--visual]
    python -m mukhtabar.quiz report     --set NAME [--top 15] [--visual]

Answer each question with أ/ب/ج/د, a-d or 1-4; "q" stops early (answers so far are
kept). After each answer you see whether it was right, the explanation, and the
passage it comes from. No model calls.

--visual is for terminals that show Arabic unjoined or reversed.
"""

import argparse
import json
import sys

from mukhtabar import db, grading, mastery, selection
from mukhtabar.display import visual as _visual

LETTERS = "أبجد"
KEYS = {**{l: i for i, l in enumerate(LETTERS)}, **{c: i for i, c in enumerate("abcd")},
        **{str(i + 1): i for i in range(4)}}


def _out(text: str, visual: bool) -> None:
    for line in text.split("\n"):
        print(_visual(line) if visual else line)


def ask(q, number: int, total: int, visual: bool, read=input) -> int | None:
    options = json.loads(q["options"])
    _out(f"\n[{number}/{total}] {q['chapter'] or ''} · {q['topic_title']}\n{q['prompt']}", visual)
    for i, o in enumerate(options):
        _out(f"  {LETTERS[i]}) {o}", visual)
    while True:
        try:
            answer = read("> ").strip().lower()
        except EOFError:
            return None
        if answer in ("q", "quit"):
            return None
        if answer in KEYS:
            return KEYS[answer]
        print("Answer with أ/ب/ج/د, a-d or 1-4 (q to stop).")


def progress(score: float, results: list[float]) -> str:
    """How a topic stands after an answer: the count while it's provisional, then mastery."""
    right = sum(1 for r in results if r)
    if mastery.is_provisional(len(results)):
        return f"{right}/{len(results)} correct · provisional"
    return f"mastery {score:.0%} ({right}/{len(results)} correct)"


def source(conn, chunk_id) -> str:
    """ " (document, ص page)" for the passage a question comes from, or ""."""
    if not chunk_id:
        return ""
    row = conn.execute("SELECT c.page, d.filename FROM chunks c JOIN documents d ON d.id = c.document_id "
                       "WHERE c.id = ?", (chunk_id,)).fetchone()
    if row is None:
        return ""
    name = row["filename"].rsplit(".", 1)[0] if row["filename"].lower().endswith(".pdf") else row["filename"]
    return f" ({name}، ص {row['page']})"


def run_diagnostic(conn, study_set_id: int, mode: str, length: int, *, visual: bool = False,
                   read=input, seed: int | None = None) -> dict:
    d = selection.Diagnostic(conn, study_set_id, mode, length, seed=seed)
    planned = min(length, len(d.plan)) if d.plan is not None else length
    with conn:
        session = db.start_session(conn, study_set_id, mode, planned)
    rows = {r["id"]: r for r in db.set_questions(conn, study_set_id)}
    right = total = 0
    while (q := d.next_question()) is not None:
        q = rows[q["id"]]
        chosen = ask(q, total + 1, planned, visual, read)
        if chosen is None:
            break
        score = grading.grade(q, chosen)
        with conn:
            new = mastery.record(conn, q, chosen, score, session)
        total += 1
        right += int(score)
        options = json.loads(q["options"])
        verdict = "✓ صحيح" if score else f"✗ خطأ — الإجابة الصحيحة: {LETTERS[int(q['answer'])]}) {options[int(q['answer'])]}"
        _out(f"{verdict}\n{q['explanation']}\nمن المادة{source(conn, q['chunk_id'])}: «{q['evidence']}»", visual)
        results = [a["score"] for a in db.topic_attempts(conn, q["topic_id"])]
        print(f"   topic so far: {progress(new, results)}")
    with conn:
        db.finish_session(conn, session)
    print(f"\nDone: {right}/{total} correct.")
    return {"session": session, "answered": total, "correct": right}


def print_report(conn, study_set_id: int, top: int, visual: bool) -> None:
    rows = selection.weakness_report(conn, study_set_id)
    has_weights = bool(db.chapter_weights(conn, study_set_id))
    print(f"Topics ranked by exam points to gain ({'using your chapter weights' if has_weights else 'no chapter weights: equal per topic'};"
          f" mastery {mastery.PRIOR:.0%} = not yet answered; * = fewer than {mastery.PROVISIONAL_BELOW} answers):\n")
    print(f"{'rank':>4}  {'gain':>5}  {'points':>6}  {'mastery':>8}  {'answers':>7}  topic")
    for i, r in enumerate(rows[:top], start=1):
        flag = "*" if r["provisional"] else " "
        line = f"{r['title']}  ({r['chapter']})"
        print(f"{i:4}  {r['gain']:5.2f}  {r['exam_points']:6.2f}  {r['mastery']:7.0%}{flag}  {r['answers']:7}  "
              + (_visual(line) if visual else line))
    answered = sum(1 for r in rows if r["answers"])
    print(f"\n{answered} of {len(rows)} topics answered so far; "
          f"total points still to gain: {sum(r['gain'] for r in rows):.1f} of {selection.EXAM_QUESTIONS}.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mukhtabar.quiz")
    parser.add_argument("--db", default=db.DEFAULT_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("diagnostic", help="answer a diagnostic quiz")
    d.add_argument("--set", required=True)
    d.add_argument("--mode", choices=["fixed", "adaptive"], default="fixed")
    d.add_argument("--n", type=int, default=selection.DEFAULT_LENGTH, help="number of questions")
    d.add_argument("--visual", action="store_true", help="for terminals that show Arabic unjoined or reversed")
    r = sub.add_parser("report", help="weakness report")
    r.add_argument("--set", required=True)
    r.add_argument("--top", type=int, default=15)
    r.add_argument("--visual", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    conn = db.connect(args.db)
    s = db.get_study_set(conn, args.set)
    if s is None:
        raise SystemExit(f"No study set named {args.set!r}")
    if args.command == "diagnostic":
        run_diagnostic(conn, s["id"], args.mode, args.n, visual=args.visual)
    else:
        print_report(conn, s["id"], args.top, args.visual)


if __name__ == "__main__":
    main()

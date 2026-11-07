"""Quizzes in the terminal (the UI comes in milestone 5). No model calls.

    python -m mukhtabar.quiz diagnostic --set NAME [--mode fixed|adaptive] [--n 30]
    python -m mukhtabar.quiz practice   --set NAME [--n 20]
    python -m mukhtabar.quiz mock       --set NAME [--minutes 60]
    python -m mukhtabar.quiz report     --set NAME [--top 15]
    python -m mukhtabar.quiz due        --set NAME

Answer each question with أ/ب/ج/د, a-d or 1-4; "q" stops early (answers so far are
kept). In a diagnostic or practice you see after each answer whether it was right,
the explanation and its source; in the mock exam only at the end.

Every command takes --visual, for terminals that show Arabic unjoined or reversed.
"""

import argparse
import json
import sys
import time

from mukhtabar import db, grading, leitner, mastery, selection
from mukhtabar.display import visual as _visual

LETTERS = "أبجد"
KEYS = {**{l: i for i, l in enumerate(LETTERS)}, **{c: i for i, c in enumerate("abcd")},
        **{str(i + 1): i for i in range(4)}}


def _out(text: str, visual: bool) -> None:
    for line in text.split("\n"):
        print(_visual(line) if visual else line)


def ask(q, number: int, total: int, visual: bool, read=input, note: str = "") -> int | None:
    options = json.loads(q["options"])
    print(f"\n[{number}/{total}{note}]")
    _out(f"{q['chapter'] or ''} · {q['topic_title']}\n{q['prompt']}", visual)
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
    """ " (document، ص page)" for the passage a question comes from, or ""."""
    if not chunk_id:
        return ""
    row = conn.execute("SELECT c.page, d.filename FROM chunks c JOIN documents d ON d.id = c.document_id "
                       "WHERE c.id = ?", (chunk_id,)).fetchone()
    if row is None:
        return ""
    name = row["filename"].rsplit(".", 1)[0] if row["filename"].lower().endswith(".pdf") else row["filename"]
    return f" ({name}، ص {row['page']})"


def correct_answer(q) -> str:
    options = json.loads(q["options"])
    return f"{LETTERS[int(q['answer'])]}) {options[int(q['answer'])]}"


def run_session(conn, study_set_id: int, kind: str, next_question, planned: int, *, feedback: bool = True,
                minutes: float | None = None, visual: bool = False, read=input, clock=time.monotonic) -> dict:
    """Ask questions until next_question() returns None, the user stops, or time runs out.
    Records every answer, then moves the answered topics' Leitner boxes."""
    with conn:
        session = db.start_session(conn, study_set_id, kind, planned)
    rows = {r["id"]: r for r in db.set_questions(conn, study_set_id)}
    deadline = clock() + minutes * 60 if minutes else None
    answers = []                                           # (question row, chosen, score)
    while True:
        if deadline is not None and clock() >= deadline:
            print("\nTime is up.")
            break
        q = next_question()
        if q is None:
            break
        q = rows[q["id"]]
        note = f" · {max(0, deadline - clock()) / 60:.0f} min left" if deadline is not None else ""
        chosen = ask(q, len(answers) + 1, planned, visual, read, note)
        if chosen is None:
            break
        score = grading.grade(q, chosen)
        with conn:
            new = mastery.record(conn, q, chosen, score, session)
        answers.append((q, chosen, score))
        if feedback:
            verdict = "✓ صحيح" if score else f"✗ خطأ — الإجابة الصحيحة: {correct_answer(q)}"
            _out(f"{verdict}\n{q['explanation']}\nمن المادة{source(conn, q['chunk_id'])}: «{q['evidence']}»", visual)
            results = [a["score"] for a in db.topic_attempts(conn, q["topic_id"])]
            print(f"   topic so far: {progress(new, results)}")
    with conn:
        db.finish_session(conn, session)
        moves = leitner.update(conn, study_set_id, db.session_topics(conn, session))
    right = sum(int(s) for _, _, s in answers)
    print(f"\nDone: {right}/{len(answers)} correct.")
    if moves:
        up = sum(1 for _, old, new in moves if old is not None and new > old)
        down = sum(1 for _, old, new in moves if old is not None and new < old)
        print(f"Review boxes: {sum(1 for _, old, _ in moves if old is None)} topics added, {up} up, {down} down.")
    return {"session": session, "answered": len(answers), "correct": right, "answers": answers}


def run_diagnostic(conn, study_set_id: int, mode: str, length: int, *, visual: bool = False,
                   read=input, seed: int | None = None) -> dict:
    d = selection.Diagnostic(conn, study_set_id, mode, length, seed=seed)
    planned = min(length, len(d.plan)) if d.plan is not None else length
    return run_session(conn, study_set_id, mode, d.next_question, planned, visual=visual, read=read)


def run_practice(conn, study_set_id: int, length: int, *, visual: bool = False, read=input,
                 seed: int | None = None, today=None) -> dict:
    p = selection.Practice(conn, study_set_id, length, seed=seed, today=today)
    if p.resting:
        print("No topic is due for review; practising the weakest topics anyway.")
    return run_session(conn, study_set_id, "practice", p.next_question, length, visual=visual, read=read)


def run_mock(conn, study_set_id: int, *, minutes: float | None = None, visual: bool = False, read=input,
             seed: int | None = None, clock=time.monotonic) -> dict:
    questions = selection.mock_exam(conn, study_set_id, seed=seed)
    pending = iter(questions)
    if minutes:
        print(f"Mock exam: {len(questions)} questions, {minutes:g} minutes.")
    result = run_session(conn, study_set_id, "mock", lambda: next(pending, None), len(questions),
                         feedback=False, minutes=minutes, visual=visual, read=read, clock=clock)
    print_mock_result(conn, study_set_id, questions, result["answers"], visual)
    return result


def mock_scores(conn, study_set_id: int, questions, answers) -> dict[str, list[int]]:
    """Chapter -> [right, questions], in chapter order; unanswered questions count as wrong."""
    chapter_of = {r["id"]: r["chapter"] for r in db.set_questions(conn, study_set_id)}
    per = {c: [0, 0] for c in db.chapters(conn, study_set_id)}
    for q in questions:
        per.setdefault(chapter_of[q["id"]], [0, 0])[1] += 1
    for q, _, score in answers:
        per[chapter_of[q["id"]]][0] += int(score)
    return {c: v for c, v in per.items() if v[1]}


def print_mock_result(conn, study_set_id: int, questions, answers, visual: bool) -> None:
    per = mock_scores(conn, study_set_id, questions, answers)
    right = sum(v[0] for v in per.values())
    print(f"\nScore: {right}/{len(questions)} ({right / len(questions):.0%})"
          + (f"; {len(questions) - len(answers)} unanswered, counted wrong" if len(answers) < len(questions) else ""))
    print(f"\n{'right':>5}  {'of':>3}  chapter")
    for c, (r, n) in per.items():
        print(f"{r:5}  {n:3}  " + (_visual(c) if visual else c))
    wrong = [(q, chosen) for q, chosen, score in answers if not score]
    if wrong:
        print("\nWrong answers:")
        for q, chosen in wrong:
            options = json.loads(q["options"])
            _out(f"\n{q['prompt']}\n  your answer: {LETTERS[chosen]}) {options[chosen]}\n"
                 f"  correct: {correct_answer(q)}{source(conn, q['chunk_id'])}", visual)


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


def print_due(conn, study_set_id: int, visual: bool, today=None) -> None:
    from datetime import date
    today = (today or date.today()).isoformat()
    reviews = db.reviews(conn, study_set_id)
    titles = {t["id"]: t["title"] for t in db.topics(conn, study_set_id)}
    due = sorted((r["due"], tid) for tid, r in reviews.items() if r["due"] <= today)
    print(f"Due today ({len(due)}):")
    for _, tid in due:
        line = f"  box {reviews[tid]['box']}  {titles[tid]}"
        print(_visual(line) if visual else line)
    later: dict[str, int] = {}
    for r in reviews.values():
        if r["due"] > today:
            later[r["due"]] = later.get(r["due"], 0) + 1
    if later:
        print("\nComing up: " + ", ".join(f"{d} ({n})" for d, n in sorted(later.items())))
    boxes = [sum(1 for r in reviews.values() if r["box"] == b) for b in leitner.INTERVALS]
    print("\nIn boxes 1-4: " + ", ".join(map(str, boxes))
          + f"; not in a box yet (fewer than {leitner.MIN_ANSWERS} answers): {len(titles) - len(reviews)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mukhtabar.quiz")
    parser.add_argument("--db", default=db.DEFAULT_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "diagnostic": sub.add_parser("diagnostic", help="answer a diagnostic quiz"),
        "practice": sub.add_parser("practice", help="practise the weakest topics"),
        "mock": sub.add_parser("mock", help="a full mock exam"),
        "report": sub.add_parser("report", help="weakness report"),
        "due": sub.add_parser("due", help="topics due for review"),
    }
    for p in commands.values():
        p.add_argument("--set", required=True)
        p.add_argument("--visual", action="store_true", help="for terminals that show Arabic unjoined or reversed")
    commands["diagnostic"].add_argument("--mode", choices=["fixed", "adaptive"], default="fixed")
    commands["diagnostic"].add_argument("--n", type=int, default=selection.DEFAULT_LENGTH, help="number of questions")
    commands["practice"].add_argument("--n", type=int, default=20, help="number of questions")
    commands["mock"].add_argument("--minutes", type=float, help="time limit (none by default)")
    commands["report"].add_argument("--top", type=int, default=15)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    conn = db.connect(args.db)
    s = db.get_study_set(conn, args.set)
    if s is None:
        raise SystemExit(f"No study set named {args.set!r}")
    if args.command == "diagnostic":
        run_diagnostic(conn, s["id"], args.mode, args.n, visual=args.visual)
    elif args.command == "practice":
        run_practice(conn, s["id"], args.n, visual=args.visual)
    elif args.command == "mock":
        run_mock(conn, s["id"], minutes=args.minutes, visual=args.visual)
    elif args.command == "report":
        print_report(conn, s["id"], args.top, args.visual)
    else:
        print_due(conn, s["id"], args.visual)


if __name__ == "__main__":
    main()

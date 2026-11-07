"""Spaced repetition: a Leitner box per topic, moved by the topic's mastery score.

A topic enters the boxes once it has MIN_ANSWERS answers. At the end of each
session, every topic answered in it is moved once:

- mastery below DEMOTE_BELOW: back to box 1 (review tomorrow);
- mastery at least PROMOTE_AT, and the topic was due: up one box;
- otherwise it stays; if it was due, its next review is set from its box.

Answering a topic before it's due can send it down but never up, so cramming
doesn't skip reviews. Intervals are short to fit an exam a couple of weeks away.
"""

from datetime import date, timedelta

from mukhtabar import db, mastery

INTERVALS = {1: 1, 2: 2, 3: 4, 4: 7}     # box -> days until the next review
MIN_ANSWERS = 3
PROMOTE_AT = 0.8
DEMOTE_BELOW = 0.6


def _due(today: date, box: int) -> str:
    return (today + timedelta(days=INTERVALS[box])).isoformat()


def move(box: int | None, due: str | None, score: float, today: date) -> tuple[int, str]:
    """The new (box, due date) for a topic; box None means it isn't in a box yet."""
    if box is None:
        box = 2 if score >= PROMOTE_AT else 1
        return box, _due(today, box)
    was_due = due <= today.isoformat()
    if score < DEMOTE_BELOW:
        return 1, _due(today, 1)
    if score >= PROMOTE_AT and was_due:
        box = min(box + 1, max(INTERVALS))
        return box, _due(today, box)
    return box, (_due(today, box) if was_due else due)


def update(conn, study_set_id: int, topic_ids, today: date | None = None) -> list[tuple[int, int | None, int]]:
    """Move each of these topics once (call at the end of a session).
    Returns (topic id, old box or None if new, new box) for each topic placed."""
    today = today or date.today()
    known = mastery.get(conn, study_set_id)
    current = db.reviews(conn, study_set_id)
    moves = []
    for tid in topic_ids:
        score, n = known.get(tid, (mastery.PRIOR, 0))
        if n < MIN_ANSWERS:
            continue
        row = current.get(tid)
        box, due = move(row["box"] if row else None, row["due"] if row else None, score, today)
        db.set_review(conn, tid, box, due)
        moves.append((tid, row["box"] if row else None, box))
    return moves


def due_topics(conn, study_set_id: int, today: date | None = None) -> set[int]:
    """Topics in a box whose review date has come."""
    today = (today or date.today()).isoformat()
    return {tid for tid, r in db.reviews(conn, study_set_id).items() if r["due"] <= today}

"""Mastery per topic: a recency-weighted average of answers, plus how many there are.

Each answer moves the score towards its result: score = (1 - ALPHA) * score + ALPHA
* result, starting from PRIOR, so recent answers count more. A score based on few
answers is shown as provisional.
"""

from mukhtabar import db

ALPHA = 0.3
PRIOR = 0.5            # before any answer: unknown, not zero
PROVISIONAL_BELOW = 4  # answers needed before a score is shown as settled


def score_from(results: list[float]) -> float:
    score = PRIOR
    for r in results:
        score = (1 - ALPHA) * score + ALPHA * r
    return score


def record(conn, question, chosen: int, score: float, session_id: int | None) -> float:
    """Store an answer and update its topic's mastery. Returns the new mastery."""
    db.add_attempt(conn, question["id"], str(chosen), score, session_id)
    results = [a["score"] for a in db.topic_attempts(conn, question["topic_id"])]
    new = score_from(results)
    db.set_mastery(conn, question["topic_id"], new, len(results))
    return new


def get(conn, study_set_id: int) -> dict[int, tuple[float, int]]:
    """Topic id -> (mastery, answers); topics never answered aren't included."""
    return {tid: (r["score"], r["attempts"]) for tid, r in db.mastery_rows(conn, study_set_id).items()}


def is_provisional(answers: int) -> bool:
    return answers < PROVISIONAL_BELOW

"""What to ask next. The fixed and adaptive diagnostics share one engine
(Diagnostic.next_question); only the strategy that picks the next topic differs.

- Fixed: the number of questions per chapter is set up front in proportion to the
  chapter's exam weight; within a chapter, topics take turns, fewest answers first.
- Adaptive: the next topic is the one with the highest (1 - mastery) x the topic's
  exam points, boosted while it has few answers; it stops once every topic has
  MIN_ANSWERS_PER_TOPIC answers, or at the planned length.

A topic's exam points are its share of an EXAM_QUESTIONS-question exam: its
chapter's weight, split across the chapter's topics by their share of its study
chunks. (Weighting by the whole chapter's weight instead let a heavy chapter with
many topics take every question.) The weakness report ranks by the same points.

Within a topic, unseen questions come first, preferring a source chunk not used yet
in the session; then the least recently answered. Without exam weights, each
chapter counts in proportion to its topics, so each topic gets an equal share.
"""

import math
import random
from dataclasses import dataclass, field

from mukhtabar import db, mastery

DEFAULT_LENGTH = 30
MIN_ANSWERS_PER_TOPIC = 2
EXAM_QUESTIONS = 50        # for the weakness report's "exam points"


def chapter_weights(conn, study_set_id: int, by_topics: bool = False) -> dict[str, float]:
    """Chapter -> weight. Chapters without a stored weight get the average of the
    others; with no weights at all, each chapter weighs 1 (or, with by_topics, its
    number of topics)."""
    chapters = db.chapters(conn, study_set_id)
    stored = db.chapter_weights(conn, study_set_id)
    if not stored:
        if by_topics:
            counts = {c: 0 for c in chapters}
            for t in db.topics(conn, study_set_id):
                if t["chapter"] in counts:
                    counts[t["chapter"]] += 1
            return counts
        return {c: 1.0 for c in chapters}
    mean = sum(stored.values()) / len(stored)
    return {c: stored.get(c, mean) for c in chapters}


def quotas(weights: dict[str, float], n: int) -> dict[str, int]:
    """Split n questions across chapters in proportion to weight (largest remainder;
    ties go to the heavier chapter, then the earlier one)."""
    total = sum(weights.values())
    if not total:
        return {c: 0 for c in weights}
    exact = {c: n * w / total for c, w in weights.items()}
    out = {c: math.floor(x) for c, x in exact.items()}
    order = list(weights)
    for c in sorted(order, key=lambda c: (-(exact[c] - out[c]), -weights[c], order.index(c)))[:n - sum(out.values())]:
        out[c] += 1
    return out


@dataclass
class Diagnostic:
    conn: object
    study_set_id: int
    mode: str                       # "fixed" or "adaptive"
    length: int = DEFAULT_LENGTH
    seed: int | None = None
    asked: list[int] = field(default_factory=list)
    used_chunks: set[int] = field(default_factory=set)
    plan: list[int] | None = None   # fixed mode: topic order, decided up front

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.answered = db.answered_question_ids(self.conn, self.study_set_id)
        self.topics = [t for t in db.topics(self.conn, self.study_set_id) if db.questions(self.conn, t["id"])]
        if self.mode == "fixed":
            self.plan = self._fixed_plan()
        elif self.mode != "adaptive":
            raise ValueError(f"unknown mode {self.mode!r}")

    # --- Strategies -------------------------------------------------------------

    def _fixed_plan(self) -> list[int]:
        answers = {tid: n for tid, (_, n) in mastery.get(self.conn, self.study_set_id).items()}
        plan = []
        for chapter, k in quotas(chapter_weights(self.conn, self.study_set_id, by_topics=True), self.length).items():
            ts = sorted((t for t in self.topics if t["chapter"] == chapter),
                        key=lambda t: (answers.get(t["id"], 0), t["position"]))
            if ts:
                plan += [ts[i % len(ts)]["id"] for i in range(k)]
        return plan

    def priority(self, topic, points, known) -> float:
        """(1 - mastery) x the topic's exam points, boosted while it has few answers."""
        score, n = known.get(topic["id"], (mastery.PRIOR, 0))
        return (1 - score) * points.get(topic["id"], 0.0) * (1 + 1 / (1 + n))

    def _adaptive_topic(self) -> int | None:
        known = mastery.get(self.conn, self.study_set_id)
        if all(known.get(t["id"], (0, 0))[1] >= MIN_ANSWERS_PER_TOPIC for t in self.topics):
            return None
        points = topic_points(self.conn, self.study_set_id)
        open_topics = [t for t in self.topics if self._candidates(t["id"])]
        if not open_topics:
            return None
        best = max(self.priority(t, points, known) for t in open_topics)
        return self.rng.choice([t["id"] for t in open_topics if abs(self.priority(t, points, known) - best) < 1e-9])

    # --- Engine -------------------------------------------------------------------

    def _candidates(self, topic_id: int) -> list:
        return [q for q in db.questions(self.conn, topic_id) if q["id"] not in self.asked]

    def _pick(self, topic_id: int):
        qs = self._candidates(topic_id)
        if not qs:
            return None
        self.rng.shuffle(qs)
        qs.sort(key=lambda q: (q["id"] in self.answered,             # unseen first
                               q["chunk_id"] in self.used_chunks,     # then a source not used yet
                               self.answered.get(q["id"], "")))       # then least recently answered
        return qs[0]

    def next_question(self):
        """The next question to ask, or None when the diagnostic is over."""
        if len(self.asked) >= self.length:
            return None
        if self.mode == "fixed":
            while len(self.asked) < len(self.plan):
                q = self._pick(self.plan[len(self.asked)])
                if q is not None:
                    break
                self.plan.pop(len(self.asked))      # that topic ran out of questions
            else:
                return None
        else:
            tid = self._adaptive_topic()
            q = self._pick(tid) if tid is not None else None
            if q is None:
                return None
        self.asked.append(q["id"])
        self.used_chunks.add(q["chunk_id"])
        return q


def topic_points(conn, study_set_id: int) -> dict[int, float]:
    """Topic id -> its share of an EXAM_QUESTIONS-question exam: the chapter's
    weight, split across the chapter's topics by their share of its study chunks.
    Topics outside any weighted chapter get 0."""
    weights = chapter_weights(conn, study_set_id, by_topics=True)
    total_w = sum(weights.values()) or 1
    topics_ = db.topics(conn, study_set_id)
    primary = {t["id"]: conn.execute("""SELECT COUNT(*) FROM topic_chunks tc JOIN chunks c ON c.id = tc.chunk_id
                                         WHERE tc.topic_id = ? AND tc.is_primary = 1 AND c.excluded = 0""",
                                      (t["id"],)).fetchone()[0] for t in topics_}
    per_chapter: dict = {}
    for t in topics_:
        per_chapter[t["chapter"]] = per_chapter.get(t["chapter"], 0) + primary[t["id"]]
    out = {}
    for t in topics_:
        if t["chapter"] not in weights or not per_chapter[t["chapter"]]:
            out[t["id"]] = 0.0
            continue
        out[t["id"]] = EXAM_QUESTIONS * weights[t["chapter"]] / total_w * primary[t["id"]] / per_chapter[t["chapter"]]
    return out


def weakness_report(conn, study_set_id: int) -> list[dict]:
    """Topics ranked by estimated exam points to gain: the topic's exam points
    (see topic_points) times (1 - mastery). Topics never answered count at the prior."""
    weights = chapter_weights(conn, study_set_id, by_topics=True)
    known = mastery.get(conn, study_set_id)
    points = topic_points(conn, study_set_id)
    rows = []
    for t in db.topics(conn, study_set_id):
        if t["chapter"] not in weights:
            continue
        score, n = known.get(t["id"], (mastery.PRIOR, 0))
        rows.append({"topic_id": t["id"], "title": t["title"], "chapter": t["chapter"], "mastery": score,
                     "answers": n, "provisional": mastery.is_provisional(n), "exam_points": points[t["id"]],
                     "gain": points[t["id"]] * (1 - score)})
    return sorted(rows, key=lambda r: -r["gain"])

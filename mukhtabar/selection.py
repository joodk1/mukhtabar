"""What to ask next: the diagnostics, practice (class Practice) and the mock exam
(mock_exam). The fixed and adaptive diagnostics share one engine
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
from datetime import date

from mukhtabar import db, leitner, mastery

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
        return pick(self._candidates(topic_id), self.answered, self.used_chunks, self.rng)

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


def pick(qs, answered: dict, used_chunks: set, rng, last: dict | None = None):
    """The best of these questions: unseen first; then, if `last` (question id ->
    last score) is given, those answered wrong last time; then a source chunk not
    used yet in the session; then the least recently answered."""
    if not qs:
        return None
    qs = list(qs)
    rng.shuffle(qs)
    qs.sort(key=lambda q: (q["id"] in answered,
                           bool(last is not None and last.get(q["id"], 0) > 0),
                           q["chunk_id"] in used_chunks,
                           answered.get(q["id"], "")))
    return qs[0]


@dataclass
class Practice:
    """Weakest topics first, by exam points to gain, recomputed after every answer.

    Topics resting in a Leitner box until a later date are left out; if every
    topic is resting, all of them are used (self.resting is then True). A topic
    gets at most MAX_PER_TOPIC questions in one session."""
    conn: object
    study_set_id: int
    length: int = 20
    seed: int | None = None
    today: date | None = None
    asked: list[int] = field(default_factory=list)
    used_chunks: set[int] = field(default_factory=set)
    resting: bool = False

    MAX_PER_TOPIC = 4

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.answered = db.answered_question_ids(self.conn, self.study_set_id)
        self.last = db.last_scores(self.conn, self.study_set_id)
        self.points = topic_points(self.conn, self.study_set_id)
        self.per_topic: dict[int, int] = {}
        reviews = db.reviews(self.conn, self.study_set_id)
        due = leitner.due_topics(self.conn, self.study_set_id, self.today)
        topics_ = [t for t in db.topics(self.conn, self.study_set_id) if db.questions(self.conn, t["id"])]
        self.topics = [t for t in topics_ if t["id"] not in reviews or t["id"] in due]
        if not self.topics:
            self.topics, self.resting = topics_, True

    def gain(self, topic_id: int, known) -> float:
        score, _ = known.get(topic_id, (mastery.PRIOR, 0))
        return self.points.get(topic_id, 0.0) * (1 - score)

    def next_question(self):
        if len(self.asked) >= self.length:
            return None
        known = mastery.get(self.conn, self.study_set_id)
        options = [(t["id"], qs) for t in self.topics
                   if self.per_topic.get(t["id"], 0) < self.MAX_PER_TOPIC
                   and (qs := [q for q in db.questions(self.conn, t["id"]) if q["id"] not in self.asked])]
        if not options:
            return None
        best = max(self.gain(tid, known) for tid, _ in options)
        tid, qs = self.rng.choice([o for o in options if abs(self.gain(o[0], known) - best) < 1e-9])
        q = pick(qs, self.answered, self.used_chunks, self.rng, self.last)
        self.asked.append(q["id"])
        self.used_chunks.add(q["chunk_id"])
        self.per_topic[tid] = self.per_topic.get(tid, 0) + 1
        return q


def mock_exam(conn, study_set_id: int, n: int = EXAM_QUESTIONS, seed: int | None = None) -> list:
    """n questions split across chapters by exam weight; within a chapter, across
    topics by their exam points (a topic short of questions passes the rest on).
    Unseen questions first, then the least recently answered. Shuffled."""
    rng = random.Random(seed)
    answered = db.answered_question_ids(conn, study_set_id)
    points = topic_points(conn, study_set_id)
    chosen = []
    for chapter, k in quotas(chapter_weights(conn, study_set_id, by_topics=True), n).items():
        pools = {t["id"]: list(db.questions(conn, t["id"]))
                 for t in db.topics(conn, study_set_id) if t["chapter"] == chapter}
        pools = {tid: qs for tid, qs in pools.items() if qs}
        counts = {tid: 0 for tid in pools}
        while k > 0:
            open_ = {tid: points.get(tid, 0) or 1 for tid in pools if counts[tid] < len(pools[tid])}
            if not open_:
                break                                    # the chapter has fewer questions than its share
            share = quotas(open_, k)
            given = 0
            for tid, extra in share.items():
                extra = min(extra, len(pools[tid]) - counts[tid])
                counts[tid] += extra
                given += extra
            k -= given
            if not given:
                break
        used: set = set()
        for tid, c in counts.items():
            qs = pools[tid]
            for _ in range(c):
                q = pick(qs, answered, used, rng)
                qs.remove(q)
                used.add(q["chunk_id"])
                chosen.append(q)
    rng.shuffle(chosen)
    return chosen


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

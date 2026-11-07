import json
from types import SimpleNamespace

import pytest

from mukhtabar import db, display, grading, mastery, quiz, selection

CHAPTERS = {"الفصل أ": 3, "الفصل ب": 2, "الفصل ج": 1}   # chapter -> number of topics
WEIGHTS = {"الفصل أ": 50, "الفصل ب": 30, "الفصل ج": 20}


@pytest.fixture
def conn(tmp_path):
    """A set with 3 chapters, 6 topics, 4 questions per topic (each on its own chunk)."""
    conn = db.connect(tmp_path / "quiz.db")
    with conn:
        sid = db.create_study_set(conn, "Set Z", "ar")
        doc = db.add_document(conn, sid, "b.pdf", "b", 30)
        pos = 0
        for chapter, n_topics in CHAPTERS.items():
            for k in range(n_topics):
                tid = db.add_topic(conn, sid, f"{chapter} موضوع {k}", "", chapter, pos)
                pos += 1
                for j in range(4):
                    conn.execute("INSERT INTO chunks (study_set_id, document_id, page, page_end, text, position) "
                                 "VALUES (?, ?, 1, 1, ?, ?)", (sid, doc, f"نص {tid}-{j}", tid * 10 + j))
                    cid = conn.execute("SELECT MAX(id) FROM chunks").fetchone()[0]
                    db.link_chunks(conn, tid, [cid])
                    db.add_question(conn, tid, {"type": "mcq", "prompt": f"سؤال {tid}-{j}", "chunk_id": cid,
                                                "options": ["أ", "ب", "ج", "د"], "answer": j % 4,
                                                "explanation": "شرح", "evidence": f"نص {tid}-{j}"}, "fake")
    return conn


def sid(conn):
    return conn.execute("SELECT id FROM study_sets").fetchone()[0]


def set_weights(conn):
    with conn:
        for c, w in WEIGHTS.items():
            db.set_chapter_weight(conn, sid(conn), c, w)


def test_grading_is_exact_match():
    q = {"answer": "2"}
    assert grading.grade(q, 2) == 1.0 and grading.grade(q, 1) == 0.0


def test_quotas_follow_the_weights():
    cme = dict(zip("abcdefghij", [6, 8, 18, 14, 10, 8, 12, 10, 8, 6]))
    assert list(selection.quotas(cme, 50).values()) == [3, 4, 9, 7, 5, 4, 6, 5, 4, 3]
    thirty = selection.quotas(cme, 30)
    assert sum(thirty.values()) == 30 and thirty["c"] == max(thirty.values())


def test_mastery_is_a_recency_weighted_average():
    assert mastery.score_from([]) == mastery.PRIOR
    assert mastery.score_from([1.0]) == pytest.approx(0.65)
    # Recent answers count more: the same results in a different order give different scores.
    assert mastery.score_from([0, 0, 1, 1]) > mastery.score_from([1, 1, 0, 0])
    assert mastery.is_provisional(3) and not mastery.is_provisional(4)


def test_fixed_diagnostic_splits_questions_by_chapter_weight(conn):
    set_weights(conn)
    d = selection.Diagnostic(conn, sid(conn), "fixed", 10, seed=1)
    asked = []
    while (q := d.next_question()) is not None:
        asked.append(q)
    chapters = [conn.execute("SELECT chapter FROM topics WHERE id = ?", (q["topic_id"],)).fetchone()[0] for q in asked]
    assert [chapters.count(c) for c in WEIGHTS] == [5, 3, 2]
    assert len({q["id"] for q in asked}) == len(asked)           # no repeats
    assert len({q["topic_id"] for q in asked if chapters[asked.index(q)] == "الفصل أ"}) == 3   # topics take turns


def test_fixed_diagnostic_without_weights_gives_each_topic_an_equal_share(conn):
    d = selection.Diagnostic(conn, sid(conn), "fixed", 12, seed=1)
    asked = [q for q in iter(d.next_question, None)]
    per_topic = {}
    for q in asked:
        per_topic[q["topic_id"]] = per_topic.get(q["topic_id"], 0) + 1
    assert sorted(per_topic.values()) == [2] * 6


def answer(conn, q, right: bool):
    chosen = int(q["answer"]) if right else (int(q["answer"]) + 1) % 4
    with conn:
        mastery.record(conn, q, chosen, grading.grade(q, chosen), None)


def test_unseen_questions_come_first(conn):
    first = selection.Diagnostic(conn, sid(conn), "fixed", 6, seed=1)
    seen = [q for q in iter(first.next_question, None)]
    for q in seen:
        answer(conn, q, True)
    second = selection.Diagnostic(conn, sid(conn), "fixed", 6, seed=1)
    again = [q for q in iter(second.next_question, None)]
    assert not {q["id"] for q in seen} & {q["id"] for q in again}


def test_adaptive_priority_is_one_minus_mastery_times_the_topics_exam_points(conn):
    set_weights(conn)
    s = sid(conn)
    d = selection.Diagnostic(conn, s, "adaptive", 20, seed=1)
    points = selection.topic_points(conn, s)
    # Chapter أ (50%) has 3 equal topics, ب (30%) 2, ج (20%) 1: 50 x 0.5 / 3, 50 x 0.3 / 2, 50 x 0.2.
    t = {r["chapter"]: r["id"] for r in reversed(db.topics(conn, s))}
    assert points[t["الفصل أ"]] == pytest.approx(25 / 3) and points[t["الفصل ب"]] == pytest.approx(7.5)
    assert points[t["الفصل ج"]] == pytest.approx(10)
    topic = db.topics(conn, s)[0]
    assert d.priority(topic, points, {}) == pytest.approx(0.5 * points[topic["id"]] * 2)   # unanswered: prior, n = 0
    assert d.priority(topic, points, {topic["id"]: (0.8, 3)}) == pytest.approx(0.2 * points[topic["id"]] * 1.25)


def test_adaptive_prefers_the_weaker_of_two_topics_with_the_same_points(conn):
    set_weights(conn)
    s = sid(conn)
    strong, weak = db.topics(conn, s)[0], db.topics(conn, s)[1]       # same chapter, same size
    for q in db.questions(conn, strong["id"])[:2]:
        answer(conn, dict(q), True)
    for q in db.questions(conn, weak["id"])[:2]:
        answer(conn, dict(q), False)
    d = selection.Diagnostic(conn, s, "adaptive", 20, seed=1)
    points, known = selection.topic_points(conn, s), mastery.get(conn, s)
    assert d.priority(weak, points, known) > d.priority(strong, points, known)


def test_adaptive_reaches_every_chapter_rather_than_only_the_heaviest(conn):
    set_weights(conn)
    s = sid(conn)
    d = selection.Diagnostic(conn, s, "adaptive", 12, seed=1)
    chapters = []
    for k, q in enumerate(iter(d.next_question, None)):
        answer(conn, q, k % 3 != 0)                                       # two right, one wrong, ...
        chapters.append(conn.execute("SELECT chapter FROM topics WHERE id = ?", (q["topic_id"],)).fetchone()[0])
    assert set(chapters) == set(WEIGHTS)


def test_adaptive_stops_once_every_topic_has_two_answers(conn):
    s = sid(conn)
    for t in db.topics(conn, s):
        for q in db.questions(conn, t["id"])[:2]:
            answer(conn, dict(q), True)
    assert selection.Diagnostic(conn, s, "adaptive", 20, seed=1).next_question() is None


def test_weakness_report_ranks_by_exam_points_to_gain(conn):
    set_weights(conn)
    s = sid(conn)
    strong, weak = db.topics(conn, s)[0]["id"], db.topics(conn, s)[1]["id"]   # same chapter, same size
    for q in db.questions(conn, strong):
        answer(conn, dict(q), True)
    for q in db.questions(conn, weak):
        answer(conn, dict(q), False)
    rows = {r["topic_id"]: r for r in selection.weakness_report(conn, s)}
    chapter_points = selection.EXAM_QUESTIONS * 50 / 100
    assert rows[weak]["exam_points"] == pytest.approx(chapter_points / 3)
    assert rows[weak]["gain"] > rows[strong]["gain"]
    assert rows[weak]["gain"] == pytest.approx(rows[weak]["exam_points"] * (1 - rows[weak]["mastery"]))
    ordered = selection.weakness_report(conn, s)
    assert [r["gain"] for r in ordered] == sorted((r["gain"] for r in ordered), reverse=True)


def test_run_diagnostic_records_answers_and_mastery(conn, capsys):
    s = sid(conn)
    answers = iter(["أ", "b", "3", "x", "4", "q"])                   # "x" is rejected and asked again
    result = quiz.run_diagnostic(conn, s, "fixed", 6, read=lambda _: next(answers), seed=2)
    assert result["answered"] == 4
    assert conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 4
    assert conn.execute("SELECT finished_at IS NOT NULL FROM sessions").fetchone()[0] == 1
    assert sum(n for _, n in mastery.get(conn, s).values()) == 4
    out = capsys.readouterr().out
    assert "Done: " in out
    assert "(b، ص 1)" in out                                          # document name and page
    assert "correct · provisional" in out


def test_progress_shows_counts_until_four_answers_then_mastery():
    assert quiz.progress(mastery.score_from([1]), [1.0]) == "1/1 correct · provisional"
    assert quiz.progress(mastery.score_from([1, 0, 1]), [1.0, 0.0, 1.0]) == "2/3 correct · provisional"
    four = [1.0, 0.0, 1.0, 1.0]
    assert quiz.progress(mastery.score_from(four), four) == f"mastery {mastery.score_from(four):.0%} (3/4 correct)"


def test_visual_display_joins_letters_and_lays_out_right_to_left():
    shown = display.visual("سهم S&P 500")
    assert shown.startswith("S&P 500 ")                              # the Latin run stays left to right, on the left
    assert "ﻢ" in shown or "ﻣ" in shown                    # meem drawn in a joined form
    assert display.visual("plain English") == "plain English"


def arabic_free(shown: str) -> str:
    return "".join(ch for ch in shown if display.LTR.match(ch) or not (display.ARABIC.match(ch) or "ﹰ" <= ch <= "﻿")).strip()


@pytest.mark.parametrize("line, kept", [
    ("الخصم Discount هنا", "Discount"),
    ("ما هو Discount؟", "Discount"),                                   # Arabic punctuation right after it
    ("الخصم،Discount، والعائد", "Discount"),
    ("سعر Present Value للسند", "Present Value"),                      # several Latin words stay in order
    ("التسوية (T+2) للأسهم", "(T+2)"),
    ("نظام Discount (T+2) في السوق", "Discount (T+2)"),
    ("العائد 12.5% سنويا", "12.5%"),
    ("بلغ ١٢٣ سهما", "١٢٣"),
])
def test_visual_display_keeps_latin_words_numbers_and_brackets_left_to_right(line, kept):
    shown = display.visual(line)
    assert kept in shown
    assert arabic_free(shown).replace("  ", " ").strip("،؟ ") == kept


def test_visual_display_puts_the_first_arabic_word_on_the_right():
    shown = display.visual("الخصم Discount")
    assert shown.startswith("Discount ") and shown.endswith(display.shape("الخصم")[0])

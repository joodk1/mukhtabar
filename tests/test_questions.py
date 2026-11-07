import json
import re
from types import SimpleNamespace

import pytest

from mukhtabar import db, questions
from mukhtabar.llm import Invalid

STUDY = [
    "السند أداة دين يصدرها المقترض للمستثمر. ويحصل حامل السند على فائدة دورية تسمى القسيمة حتى تاريخ الاستحقاق.",
    "العائد الجاري يساوي القسيمة السنوية مقسومة على السعر السوقي للسند. وكلما ارتفع السعر انخفض العائد الجاري.",
    "تصدر الحكومات سندات لتمويل الإنفاق العام. وتعد السندات الحكومية عادة أقل مخاطرة من سندات الشركات.",
]
PRACTICE = ("4 ما الفرق بين السند الحكومي وسند الشركة من حيث مستوى المخاطرة المتوقعة للمستثمر؟\n"
            "مرجع الإجابة: القسم 2-1")


@pytest.fixture
def conn(tmp_path):
    conn = db.connect(tmp_path / "q.db")
    with conn:
        sid = db.create_study_set(conn, "Set Q", "ar")
        doc = db.add_document(conn, sid, "book.pdf", "b", 4)
        texts = STUDY + [PRACTICE]
        db.replace_chunks(conn, sid, doc, [SimpleNamespace(text=t, page=i + 1, page_end=i + 1, position=i)
                                           for i, t in enumerate(texts)])
        ids = [r["id"] for r in db.chunks(conn, doc)]
        tid = db.add_topic(conn, sid, "خصائص السندات", "ملخص قصير", None, 0)
        db.link_chunks(conn, tid, ids[:3])
        db.link_chunks(conn, tid, ids[3:], primary=False)
    return conn


def topic_id(conn):
    return conn.execute("SELECT id FROM topics").fetchone()[0]


def reply(obj):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(obj, ensure_ascii=False)))],
                           usage=SimpleNamespace(prompt_tokens=500, completion_tokens=300))


def good_questions(user: str, n=6, offset=0):
    study = re.findall(r'<chunk id="(\d+)"[^>]*kind="study">\n(.*?)\n</chunk>', user, re.S)
    qs = []
    for i in range(offset, offset + n):
        cid, text = study[i % len(study)]
        qs.append({"prompt": f"سؤال تجريبي رقم {i} عن فكرة مختلفة في النص {cid} وما يترتب عليها",
                   "options": [f"خيار {i}-{k}" for k in range(4)], "answer_index": i % 4,
                   "explanation": "لأن النص يذكر ذلك صراحة.", "chunk_id": int(cid),
                   "evidence": text.split(". ")[0]})
    return {"questions": qs}


def fake(mutate=None):
    calls = []

    def completion(messages, **kwargs):
        calls.append(messages)
        n = int(re.search(r"Write exactly (\d+) new", messages[0]["content"]).group(1))
        obj = good_questions(messages[1]["content"], n, offset=len(calls) * 10)
        if mutate and len(messages) == 2:  # only the first attempt is broken
            mutate(obj, messages[1]["content"])
        return reply(obj)
    completion.calls = calls
    return completion


def test_generates_grounded_multiple_choice_questions(conn):
    tid = topic_id(conn)
    added = questions.generate(conn, "Set Q", "fake", [tid], completion=fake())
    assert added == {tid: 6}
    qs = db.questions(conn, tid)
    assert {q["type"] for q in qs} == {"mcq"} and all(len(json.loads(q["options"])) == 4 for q in qs)
    study_ids = {r["id"] for r in db.topic_sources(conn, tid) if r["is_primary"]}
    for q in qs:
        assert q["chunk_id"] in study_ids
        source = db.get_chunk(conn, q["chunk_id"])["text"]
        assert questions.canon(q["evidence"]) in questions.canon(source)


def test_short_answer_questions_are_rejected(conn):
    completion = fake(lambda o, u: o["questions"][0].update(type="short"))
    questions.generate(conn, "Set Q", "fake", [topic_id(conn)], completion=completion)
    assert "multiple-choice questions only" in completion.calls[1][-1]["content"]


def test_topic_is_topped_up_without_repeating_existing_questions(conn):
    tid = topic_id(conn)
    questions.generate(conn, "Set Q", "fake", [tid], n=4, completion=fake())
    kept = [q["prompt"] for q in db.questions(conn, tid)]
    completion = fake()
    assert questions.generate(conn, "Set Q", "fake", [tid], n=6, completion=completion) == {tid: 2}
    system = completion.calls[0][0]["content"]
    assert "Write exactly 2 new" in system and all(p in system for p in kept)   # shown so they aren't repeated
    assert [q["prompt"] for q in db.questions(conn, tid)][:4] == kept and len(db.questions(conn, tid)) == 6
    # A reply that repeats an existing question is rejected.
    with pytest.raises(Invalid, match="repeats"):
        questions.validate_questions(
            {"questions": [{"prompt": kept[0], "options": ["أ", "ب", "ج", "د"], "answer_index": 0,
                            "explanation": "شرح", "chunk_id": db.questions(conn, tid)[0]["chunk_id"],
                            "evidence": db.questions(conn, tid)[0]["evidence"]}]},
            [db.get_chunk(conn, db.questions(conn, tid)[0]["chunk_id"])], 1, "ar", set(), kept)


def test_practice_chunks_are_sent_as_style_examples_only(conn):
    completion = fake()
    questions.generate(conn, "Set Q", "fake", [topic_id(conn)], completion=completion)
    user = completion.calls[0][1]["content"]
    assert 'kind="practice"' in user and user.count('kind="study"') == 3


@pytest.mark.parametrize("mutate, message", [
    (lambda o, u: o["questions"][0].update(evidence="جملة لا توجد في أي نص من النصوص المرسلة إطلاقا"), "word for word"),
    (lambda o, u: o["questions"][0].update(chunk_id=int(re.findall(r'<chunk id="(\d+)"[^>]*kind="practice"', u)[0])),
     "style examples only"),
    (lambda o, u: o["questions"][0].update(prompt="ما الفرق بين السند الحكومي وسند الشركة من حيث مستوى المخاطرة المتوقعة للمستثمر؟"),
     "copies an existing practice question"),
    (lambda o, u: o["questions"].pop(), "exactly 6"),
    (lambda o, u: o["questions"][0].update(options=["أ", "أ", "ب", "ج"]), "4 different"),
    (lambda o, u: o["questions"][1].update(prompt=o["questions"][0]["prompt"]), "repeats"),
])
def test_bad_questions_are_rejected_and_retried(conn, mutate, message):
    completion = fake(mutate)
    questions.generate(conn, "Set Q", "fake", [topic_id(conn)], completion=completion)
    assert len(completion.calls) == 2
    assert message in completion.calls[1][-1]["content"]


def test_topic_with_questions_is_skipped(conn):
    tid = topic_id(conn)
    questions.generate(conn, "Set Q", "fake", [tid], completion=fake())
    again = fake()
    assert questions.generate(conn, "Set Q", "fake", [tid], completion=again) == {}
    assert again.calls == []


def test_sources_fit_the_budget_and_spread_over_the_topic():
    rows = [{"id": i, "text": "س" * 3000} for i in range(20)] + [{"id": 99, "text": PRACTICE}] * 5
    study, practice = questions.select_sources(rows)
    assert sum(len(r["text"]) for r in study) <= questions.SOURCE_CHARS
    assert study[0]["id"] == 0 and study[-1]["id"] > 10   # spread, not just the first chunks
    assert len(practice) == questions.MAX_PRACTICE_CHUNKS


def test_estimate_makes_no_call(conn):
    e = questions.estimate(conn, 1, [topic_id(conn)], "gemini/gemini-2.5-flash", "ar", 6, token_counter=len)
    assert e.calls == 1 and e.input_tokens > 0 and e.output_tokens == 6 * questions.OUTPUT_TOKENS_PER_QUESTION
    questions.generate(conn, "Set Q", "fake", [topic_id(conn)], completion=fake())
    done = questions.estimate(conn, 1, [topic_id(conn)], "gemini/gemini-2.5-flash", "ar", 6, token_counter=len)
    assert done.calls == 0   # topics that already have enough questions cost nothing

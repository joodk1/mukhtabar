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
    calls, kwargs_seen = [], []

    def completion(messages, **kwargs):
        calls.append(messages)
        kwargs_seen.append(kwargs)
        n = int(re.search(r"Write exactly (\d+) new", messages[0]["content"]).group(1))
        obj = good_questions(messages[1]["content"], n, offset=len(calls) * 10)
        if mutate and len(messages) == 2:  # only the first attempt is broken
            mutate(obj, messages[1]["content"])
        return reply(obj)
    completion.calls = calls
    completion.kwargs = kwargs_seen
    return completion


def test_generates_grounded_multiple_choice_questions(conn):
    tid = topic_id(conn)
    completion = fake()
    added, failed = questions.generate(conn, "Set Q", "fake", [tid], completion=completion)
    assert added == {tid: 6} and failed == {}
    # Three questions per call, with an explicit output limit.
    assert len(completion.calls) == 2
    assert all("Write exactly 3 new" in c[0]["content"] for c in completion.calls)
    assert all(k["max_tokens"] == questions.MAX_OUTPUT_TOKENS for k in completion.kwargs)
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
    assert questions.generate(conn, "Set Q", "fake", [tid], n=6, completion=completion) == ({tid: 2}, {})
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
    (lambda o, u: o["questions"][0].update(evidence="جملة لا توجد في أي نص من النصوص المرسلة إطلاقا"),
     'you gave: "جملة لا توجد في أي نص من النصوص المرسلة إطلاقا"'),   # the quote is logged with the rejection
    (lambda o, u: o["questions"][0].update(chunk_id=int(re.findall(r'<chunk id="(\d+)"[^>]*kind="practice"', u)[0])),
     "style examples only"),
    (lambda o, u: o["questions"][0].update(prompt="ما الفرق بين السند الحكومي وسند الشركة من حيث مستوى المخاطرة المتوقعة للمستثمر؟"),
     "copies an existing practice question"),
    (lambda o, u: o["questions"].pop(), "exactly 3"),
    (lambda o, u: o["questions"][0].update(evidence="س" * 450), "at most about 400 characters"),
    (lambda o, u: o["questions"][0].update(options=["أ", "أ", "ب", "ج"]), "4 different"),
    (lambda o, u: o["questions"][1].update(prompt=o["questions"][0]["prompt"]), "repeats"),
])
def test_bad_questions_are_rejected_and_retried(conn, mutate, message):
    completion = fake(mutate)
    questions.generate(conn, "Set Q", "fake", [topic_id(conn)], n=3, completion=completion)
    assert len(completion.calls) == 2
    assert message in completion.calls[1][-1]["content"]


def test_topic_with_questions_is_skipped(conn):
    tid = topic_id(conn)
    questions.generate(conn, "Set Q", "fake", [tid], completion=fake())
    again = fake()
    assert questions.generate(conn, "Set Q", "fake", [tid], completion=again) == ({}, {})
    assert again.calls == []


def test_sources_fit_the_budget_and_spread_over_the_topic():
    rows = [{"id": i, "text": "س" * 3000} for i in range(20)] + [{"id": 99, "text": PRACTICE}] * 5
    study, practice = questions.select_sources(rows)
    assert sum(len(r["text"]) for r in study) <= questions.SOURCE_CHARS
    assert study[0]["id"] == 0 and study[-1]["id"] > 10   # spread, not just the first chunks
    assert len(practice) == questions.MAX_PRACTICE_CHUNKS


def test_estimate_makes_no_call(conn):
    e = questions.estimate(conn, 1, [topic_id(conn)], "gemini/gemini-2.5-flash", "ar", 6, token_counter=len)
    assert e.calls == 2 and e.input_tokens > 0 and e.output_tokens == 6 * questions.OUTPUT_TOKENS_PER_QUESTION
    questions.generate(conn, "Set Q", "fake", [topic_id(conn)], completion=fake())
    done = questions.estimate(conn, 1, [topic_id(conn)], "gemini/gemini-2.5-flash", "ar", 6, token_counter=len)
    assert done.calls == 0   # topics that already have enough questions cost nothing


@pytest.fixture
def two_topics(conn):
    """A second topic sharing the same chunks."""
    with conn:
        sid = conn.execute("SELECT id FROM study_sets").fetchone()[0]
        second = db.add_topic(conn, sid, "العائد الجاري للسندات", "", None, 1)
        ids = [r[0] for r in conn.execute("SELECT id FROM chunks ORDER BY id")]
        db.link_chunks(conn, second, ids[:3], primary=False)
    return topic_id(conn), second


def test_a_failed_topic_is_skipped_and_the_run_goes_on(conn, two_topics):
    first, second = two_topics
    good = fake()
    asked = []

    def failing_first_topic(messages, **kwargs):
        asked.append(1)
        if "Topic: خصائص السندات" in messages[0]["content"]:
            return reply({"questions": []})          # invalid every time
        return good(messages, **kwargs)
    added, failed = questions.generate(conn, "Set Q", "fake", [first, second], completion=failing_first_topic)
    assert first in failed and "write exactly 3" in failed[first]
    assert added == {second: 6} and db.questions(conn, first) == []


def test_exhausted_quota_stops_the_whole_run(conn, two_topics):
    from mukhtabar.llm import QuotaExhausted
    first, second = two_topics

    def quota(messages, **kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded, limit per day")
    with pytest.raises(QuotaExhausted):
        questions.generate(conn, "Set Q", "fake", [first, second], completion=quota)


def test_evidence_of_a_full_long_sentence_is_accepted(conn):
    long_sentence = ("يحصل حامل السند على فائدة دورية تسمى القسيمة " * 7).strip()
    assert 300 < len(long_sentence) <= questions.MAX_EVIDENCE_CHARS
    study = [{"id": 1, "text": f"مقدمة قصيرة. {long_sentence}. خاتمة."}]
    q = {"prompt": "ما الذي يحصل عليه حامل السند بشكل دوري حتى الاستحقاق؟", "options": ["قسيمة", "سهم", "عقد", "خيار"],
         "answer_index": 0, "explanation": "لأن النص يذكر ذلك.", "chunk_id": 1, "evidence": long_sentence}
    assert questions.validate_questions({"questions": [q]}, study, 1, "ar", set(), [])[0]["evidence"] == long_sentence


def one_question(chunk_id, evidence):
    return {"questions": [{"prompt": "ما الذي يحصل عليه حامل السند بشكل دوري حتى تاريخ الاستحقاق؟",
                           "options": ["القسيمة", "الأرباح الموزعة", "رأس المال", "العائد الجاري"], "answer_index": 0,
                           "explanation": "لأن النص يذكر ذلك.", "chunk_id": chunk_id, "evidence": evidence}]}


STUDY_ROWS = [{"id": 1, "text": STUDY[0]}, {"id": 2, "text": STUDY[1]}, {"id": 3, "text": STUDY[2]}]
QUOTE_IN_1 = "ويحصل حامل السند على فائدة دورية تسمى القسيمة حتى تاريخ الاستحقاق"


def test_chunk_id_written_as_text_is_accepted():
    q = questions.validate_questions(one_question("1", QUOTE_IN_1), STUDY_ROWS, 1, "ar", set(), [])[0]
    assert q["chunk_id"] == 1 and not q["chunk_corrected"]
    with pytest.raises(Invalid, match="not one of the study chunks"):
        questions.validate_questions(one_question("chunk 1", QUOTE_IN_1), STUDY_ROWS, 1, "ar", set(), [])


def test_wrong_chunk_is_corrected_when_the_quote_is_in_exactly_one_study_chunk():
    q = questions.validate_questions(one_question(2, QUOTE_IN_1), STUDY_ROWS, 1, "ar", set(), [])[0]
    assert q["chunk_id"] == 1 and q["chunk_corrected"]


def test_wrong_chunk_is_not_corrected_when_the_quote_is_in_two_chunks():
    rows = STUDY_ROWS + [{"id": 4, "text": "تكرار: " + STUDY[0]}]
    with pytest.raises(Invalid, match="word for word"):
        questions.validate_questions(one_question(2, QUOTE_IN_1), rows, 1, "ar", set(), [])


def test_never_corrected_to_a_practice_chunk(conn):
    # Only the study chunks sent are candidates: a quote found only in practice material is rejected.
    practice_quote = "ما الفرق بين السند الحكومي وسند الشركة من حيث مستوى المخاطرة"
    with pytest.raises(Invalid, match="word for word"):
        questions.validate_questions(one_question(2, practice_quote), STUDY_ROWS, 1, "ar", set(), [])
    # And a question claiming a practice chunk is still rejected, even if its quote is in a study chunk.
    tid = topic_id(conn)
    practice_id = [r["id"] for r in db.topic_sources(conn, tid) if not r["is_primary"]][0]
    study = [r for r in db.topic_sources(conn, tid) if r["is_primary"]]
    with pytest.raises(Invalid, match="not one of the study chunks"):
        questions.validate_questions(one_question(practice_id, QUOTE_IN_1), study, 1, "ar", set(), [])


def test_corrected_questions_are_stored_against_the_real_chunk(conn):
    tid = topic_id(conn)
    sources = [r for r in db.topic_sources(conn, tid) if r["is_primary"]]
    wrong = sources[1]["id"]
    real = sources[0]["id"]

    def misattributing(messages, **kwargs):
        obj = good_questions(messages[1]["content"], 3)
        for q in obj["questions"]:
            if q["chunk_id"] == real:
                q["chunk_id"] = wrong
        return reply(obj)
    added, failed = questions.generate(conn, "Set Q", "fake", [tid], n=3, completion=misattributing)
    assert added == {tid: 3} and failed == {}
    stored = db.questions(conn, tid)
    assert all(questions.canon(q["evidence"]) in questions.canon(db.get_chunk(conn, q["chunk_id"])["text"])
               for q in stored)


# --- Exam logistics --------------------------------------------------------------

@pytest.mark.parametrize("prompt", [
    "كم عدد الأسئلة المتوقع أن يتضمنها الفصل الأول في الاختبار؟",
    "ما وزن الفصل الثالث في الاختبار؟",
    "ما الذي تنص عليه خطة المنهاج لهذا الفصل؟",
    "How many exam questions come from chapter 2?",
])
def test_questions_about_the_exam_itself_are_rejected(prompt):
    q = one_question(1, QUOTE_IN_1)
    q["questions"][0]["prompt"] = prompt
    with pytest.raises(Invalid, match="about the exam itself"):
        questions.validate_questions(q, STUDY_ROWS, 1, "ar" if "ا" in prompt else "en", set(), [])


def test_subject_questions_that_mention_a_test_are_fine():
    q = one_question(1, QUOTE_IN_1)
    q["questions"][0]["prompt"] = "ما الغرض من اختبار الملاءمة الذي يجريه المستشار المالي قبل التوصية بسند معين؟"
    assert questions.validate_questions(q, STUDY_ROWS, 1, "ar", set(), [])


# --- Arithmetic ------------------------------------------------------------------

def test_evaluate_is_safe_and_reads_arabic_digits_and_operators():
    assert questions.evaluate("(2.30 - 2.00) / 2.00 * 100") == pytest.approx(15)
    assert questions.evaluate("٢٥٬٠٠٠ ÷ ٠٫٠٤".replace("٬", ",")) == pytest.approx(625000)
    assert questions.evaluate("1,875 × 10") == pytest.approx(18750)
    for bad in ("__import__('os')", "abs(-2)", "x + 1"):
        with pytest.raises(ValueError):
            questions.evaluate(bad)


CALC_SOURCE = "مثال محلول: إذا كان سعر تداول الوحدة 2.30 وصافي قيمة الأصول 2.00 فإن العلاوة = (2.30 - 2.00) ÷ 2.00 × 100 = 15%."


def check(expression, result, options, answer=0, prompt="سعر الوحدة 2.30 وصافي قيمة الأصول 2.00، كم نسبة العلاوة؟"):
    return questions.check_calculation({"expression": expression, "result": result}, 1, prompt, options, answer,
                                       CALC_SOURCE)


def test_a_correct_calculation_passes():
    assert check("(2.30 - 2.00) / 2.00 * 100", 15, ["15%", "13%", "30%", "0.30%"])["result"] == pytest.approx(15)
    assert check("(2.30 - 2.00) / 2.00", 0.15, ["15%", "13%", "30%", "1.5%"])   # a fraction shown as a percent


@pytest.mark.parametrize("args, message", [
    (("(2.30 - 2.00) / 2.00 * 100", 20, ["15%", "13%", "30%", "0.3%"]), "not the stated result 20"),
    (("(2.30 - 2.00) / 2.00 * 100", 15, ["13%", "15%", "30%", "0.3%"]), "correct option doesn't show"),
    (("(2.30 - 2.00) / 2.00 * 100", 15, ["15%", "15.0%", "30%", "0.3%"]), "a wrong option also shows"),
    (("(2.75 - 2.00) / 2.00 * 100", 37.5, ["37.5%", "13%", "30%", "0.3%"]), "aren't in the question"),
    (("abs(2.30)", 2.3, ["2.3", "13", "30", "0.3"]), "can't be evaluated"),
])
def test_a_calculation_that_doesnt_check_out_is_rejected(args, message):
    with pytest.raises(Invalid, match=message):
        check(*args)


# --- Calculation mode ------------------------------------------------------------

@pytest.fixture
def calc_conn(conn):
    """Adds a worked-example chunk to the topic."""
    with conn:
        sid = conn.execute("SELECT id FROM study_sets").fetchone()[0]
        doc = conn.execute("SELECT id FROM documents").fetchone()[0]
        conn.execute("INSERT INTO chunks (study_set_id, document_id, page, page_end, text, position) VALUES (?, ?, 9, 9, ?, 99)",
                     (sid, doc, CALC_SOURCE))
        cid = conn.execute("SELECT MAX(id) FROM chunks").fetchone()[0]
        db.link_chunks(conn, topic_id(conn), [cid])
    return conn


def calc_model(bad_first=False):
    calls = []

    def completion(messages, **kwargs):
        calls.append(messages)
        n = int(re.search(r"Write exactly (\d+) new", messages[0]["content"]).group(1))
        cid = int(re.findall(r'<chunk id="(\d+)"[^>]*kind="study"', messages[1]["content"])[0])
        prompts = ["إذا كان سعر الوحدة 2.30 وصافي قيمة الأصول 2.00، احسب نسبة العلاوة",
                   "صندوق مغلق تتداول وحدته عند 2.30 بينما صافي قيمة أصوله للوحدة 2.00؛ ما مقدار علاوته بالنسبة المئوية؟",
                   "مستثمر يقارن بين السعر السوقي 2.30 والقيمة الصافية 2.00 لوحدة صندوق؛ كم تبلغ العلاوة؟"]
        qs = [{"prompt": prompts[i],
               "options": ["15%", "13%", "30%", "0.30%"], "answer_index": 0, "explanation": "الفرق مقسوما على صافي القيمة.",
               "chunk_id": cid, "evidence": CALC_SOURCE,
               "calculation": {"expression": "(2.30 - 2.00) / 2.00 * 100", "result": 15}} for i in range(n)]
        if bad_first and len(messages) == 2:
            qs[0]["calculation"]["result"] = 20
        return reply({"questions": qs})
    completion.calls = calls
    return completion


def test_calc_topics_are_those_with_worked_calculations(conn, calc_conn):
    sid = conn.execute("SELECT id FROM study_sets").fetchone()[0]
    assert questions.calc_topics(calc_conn, sid) == [topic_id(calc_conn)]
    study, _ = questions.select_sources(db.topic_sources(calc_conn, topic_id(calc_conn)), calc=True)
    assert [c["text"] for c in study] == [CALC_SOURCE]


def test_calc_mode_adds_two_checked_calculation_questions_up_to_the_cap(calc_conn):
    tid = topic_id(calc_conn)
    questions.generate(calc_conn, "Set Q", "fake", [tid], completion=fake())          # 6 regular questions
    model = calc_model(bad_first=True)
    added, failed = questions.generate(calc_conn, "Set Q", "fake", [tid], calc=True, completion=model)
    assert added == {tid: 2} and failed == {}
    assert "not the stated result" in model.calls[1][-1]["content"]                  # bad arithmetic was rejected
    stored = [q for q in db.questions(calc_conn, tid) if q["calculation"]]
    assert len(stored) == 2 and json.loads(stored[0]["calculation"])["result"] == pytest.approx(15)
    assert len(db.questions(calc_conn, tid)) == 8
    # Already at its calculation quota (and the cap): nothing more.
    assert questions.generate(calc_conn, "Set Q", "fake", [tid], calc=True, completion=calc_model()) == ({}, {})


def test_calc_mode_respects_the_cap(calc_conn):
    tid = topic_id(calc_conn)
    questions.generate(calc_conn, "Set Q", "fake", [tid], n=7, completion=fake())
    assert questions.missing(calc_conn, tid, 6, calc=True) == 1                       # 7 + 1 = MAX_PER_TOPIC

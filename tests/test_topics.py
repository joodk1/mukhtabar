import json
import math
import re
from types import SimpleNamespace

import pytest

from mukhtabar import db, topics
from mukhtabar.ingest.pipeline import RechunkRefused, rechunk
from mukhtabar.llm import Invalid, LLMError

CH1, CH2 = "الفصل الأول: الادخار", "الفصل الثاني: الاستثمار"
BODY = "نص تعليمي عن الموضوع يشرح فكرة واحدة بوضوح ويكفي لكتابة عدة أسئلة عنها."


def page_texts():
    """A 14-page "book": two chapters with running titles, plus a one-off repeated heading."""
    first_lines = {1: "فهرس المحتويات", 2: "تمرين", 8: "تمرين", 10: "تمرين",
                   **{p: CH1 for p in range(3, 8)}, **{p: CH2 for p in range(11, 15)}}
    return [(p, f"{first_lines.get(p, 'افتتاحية الفصل')}\n{BODY} صفحة {p}.") for p in range(1, 15)]


@pytest.fixture
def conn(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    with conn:
        set_id = db.create_study_set(conn, "Set A", "ar")
        for name, pages in (("book.pdf", page_texts()), ("summary.pdf", [(p, f"{BODY} ملخص {p}.") for p in (1, 2, 3)])):
            doc = db.add_document(conn, set_id, name, name, len(pages))
            db.add_pages(conn, doc, [(p, t, [], []) for p, t in pages])
            db.replace_chunks(conn, set_id, doc, [SimpleNamespace(text=t, page=p, page_end=p, position=i)
                                                  for i, (p, t) in enumerate(pages)])
    return conn


def reply(obj):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(obj, ensure_ascii=False)))],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def fake_model(drop_first_map_id=False):
    """Replies like a well-behaved model, reading ids from the prompt. Optionally
    leaves a chunk out of its first map reply, to exercise the retry."""
    state = {"map": 0}

    def completion(messages, **kwargs):
        system, user = messages[0]["content"], messages[1]["content"]
        if "Group the chunks" in system:
            state["map"] += 1
            chunks = re.findall(r'<chunk id="(\d+)"[^>]*>\n(.*?)\n</chunk>', user, re.S)
            non_study = [int(i) for i, t in chunks if "فهرس المحتويات" in t]
            ids = [int(i) for i, _ in chunks if int(i) not in non_study]
            if drop_first_map_id and state["map"] == 1 and len(messages) == 2:
                ids = ids[:-1]
            groups = [ids[i:i + 2] for i in range(0, len(ids), 2)]
            return reply({"topics": [{"title": f"موضوع رقم {g[0]}", "summary": "ملخص قصير للموضوع", "chunk_ids": g}
                                     for g in groups],
                          "non_study": [{"chunk_id": i, "reason": "فهرس"} for i in non_study]})
        if "name the chapter whose subject it belongs to" in system:
            local = re.findall(r'"id": "(b\d+-t\d+)"', user)
            chapters = re.findall(r"^- (.+)$", system.split("Chapters:")[1], re.M)
            return reply({"routes": [{"local_topic_id": l, "chapter": chapters[0]} for l in local]})
        if "Merge them into the final study topics" in system:
            state["reduce"] = state.get("reduce", 0) + 1
            local = re.findall(r'"id": "(b\d+-t\d+)"', user)
            counts = [int(n) for n in re.findall(r"(?:between|exactly) (\d+)(?: and (\d+))?", system)[0] if n]
            n = max(counts[0], min(counts[-1], math.ceil(len(local) / 2)))
            groups = [local[i::n] for i in range(n)]
            return reply({"topics": [{"title": f"الموضوع النهائي {state['reduce']}-{k}", "summary": "ملخص",
                                      "local_topic_ids": g} for k, g in enumerate(groups, 1)]})
        if "Place each new local topic" in system:
            existing = [int(i) for i in re.findall(r'"id": (\d+)', user.split("New local topics")[0])]
            local = re.findall(r'"id": "(b\d+-t\d+)"', user)
            return reply({"placements": [{"local_topic_id": local[0], "existing_topic_id": existing[0]}] +
                          [{"local_topic_id": l, "new_title": "موضوع جديد", "new_summary": "ملخص"} for l in local[1:]]})
        raise AssertionError("unexpected prompt")
    return completion


def set_id(conn):
    return db.get_study_set(conn, "Set A")["id"]


def test_chapter_hints_come_from_running_titles_only(conn):
    book = db.documents(conn, set_id(conn))[0]["id"]
    hints = topics.chapter_hints(conn, book)
    assert [h.title for h in hints] == [CH1, CH2]          # "تمرين" repeats, but never 3 pages in a row
    assert (hints[0].first_page, hints[0].last_page) == (3, 8)
    assert (hints[1].first_page, hints[1].last_page) == (9, 14)  # two opener pages before the first title


def test_a_label_scattered_across_a_document_is_not_a_chapter(conn):
    # A box label that opens several pages far apart (e.g. "must memorize") isn't a running title.
    with conn:
        doc = db.add_document(conn, set_id(conn), "notes.pdf", "notes", 20)
        db.add_pages(conn, doc, [(p, f"{'★ مهم' if p in (2, 7, 12, 17) else 'عنوان'} {p}\n{BODY}" if p not in (2, 7, 12, 17)
                                  else f"★ مهم\n{BODY}", [], []) for p in range(1, 21)])
    assert topics.chapter_hints(conn, doc) == []


def test_running_title_on_every_other_page_is_a_chapter(conn):
    with conn:
        doc = db.add_document(conn, set_id(conn), "alt.pdf", "alt", 10)
        db.add_pages(conn, doc, [(p, f"{'الفصل الثالث' if p % 2 else f'قسم {p}'}\n{BODY}", [], []) for p in range(1, 11)])
    assert [h.title for h in topics.chapter_hints(conn, doc)] == ["الفصل الثالث"]


def test_batches_respect_budget_documents_and_chapters(conn):
    chunks = topics.select_chunks(conn, set_id(conn))
    # Each chapter here is ~600 characters: over 20% of this budget, so batches cut at chapters.
    batches = topics.make_batches(chunks, topics.set_hints(conn, set_id(conn)), budget=1_000)
    assert len({c["document_id"] for b in batches for c in b.chunks[:1]}) == 2
    assert all(len({c["document_id"] for c in b.chunks}) == 1 for b in batches)
    assert [b.chapters for b in batches if b.chapters] == [[CH1], [CH2]]
    # A chapter change in a batch that's still small doesn't force a cut.
    assert len(topics.make_batches(chunks, topics.set_hints(conn, set_id(conn)), budget=10_000)) == 2
    small = topics.make_batches(chunks, [], budget=300)
    assert all(sum(len(c["text"]) for c in b.chunks) <= 300 or len(b.chunks) == 1 for b in small)


def test_build_links_every_study_chunk_once_and_excludes_non_study(conn):
    sid = set_id(conn)
    chunks = topics.select_chunks(conn, sid)
    ids = topics.build(conn, "Set A", "fake", chunks, completion=fake_model())
    assert ids
    linked = conn.execute("SELECT chunk_id FROM topic_chunks WHERE is_primary = 1").fetchall()
    excluded = conn.execute("SELECT id FROM chunks WHERE excluded = 1").fetchall()
    assert len(linked) + len(excluded) == len(chunks)
    assert len({r[0] for r in linked}) == len(linked)
    assert len(excluded) == 1 and db.unassigned_chunks(conn, sid) == []
    # Book topics keep their chapter; the summary (no running titles) is routed to one.
    assert {t["chapter"] for t in db.topics(conn, sid)} <= {CH1, CH2}
    calls = [c[0] for c in conn.execute("SELECT purpose, ok FROM llm_calls").fetchall()]
    assert calls.count("topics.route") == 1
    assert calls.count("topics.reduce") <= topics.planned_calls(chunks, topics.set_hints(conn, sid)) - 1


def test_one_primary_topic_per_chunk_is_enforced_by_the_database(conn):
    topics.build(conn, "Set A", "fake", topics.select_chunks(conn, set_id(conn)), completion=fake_model())
    t1, t2 = [t["id"] for t in db.topics(conn, set_id(conn))][:2]
    chunk = db.topic_chunk_rows(conn, t1)[0]["id"]
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        db.link_chunks(conn, t2, [chunk])


def test_invalid_map_reply_is_retried(conn):
    topics.build(conn, "Set A", "fake", topics.select_chunks(conn, set_id(conn)),
                 completion=fake_model(drop_first_map_id=True))
    first = conn.execute("SELECT attempts, ok FROM llm_calls ORDER BY id LIMIT 1").fetchone()
    assert tuple(first) == (2, 1)


def test_build_refuses_to_replace_a_map_without_force(conn):
    chunks = topics.select_chunks(conn, set_id(conn))
    topics.build(conn, "Set A", "fake", chunks, completion=fake_model())
    with pytest.raises(RuntimeError, match="--force"):
        topics.build(conn, "Set A", "fake", chunks, completion=fake_model())
    before = len(db.topics(conn, set_id(conn)))
    topics.build(conn, "Set A", "fake", chunks, force=True, completion=fake_model())
    assert len(db.topics(conn, set_id(conn))) == before


def test_chapter_selection_limits_the_chunks(conn):
    chunks = topics.select_chunks(conn, set_id(conn), chapter=2)
    assert {c["page"] for c in chunks} == set(range(9, 15))
    only = topics.select_chunks(conn, set_id(conn), only=topics.parse_only(["summary.pdf:2-3"]))
    assert [(c["filename"], c["page"]) for c in only] == [("summary.pdf", 2), ("summary.pdf", 3)]


def test_rename_merge_delete(conn):
    sid = set_id(conn)
    topics.build(conn, "Set A", "fake", topics.select_chunks(conn, sid), completion=fake_model())
    t = db.topics(conn, sid)
    keep, other = t[0]["id"], t[1]["id"]
    n_keep, n_other = t[0]["n_chunks"], t[1]["n_chunks"]
    topics.rename(conn, keep, "عنوان جديد", "ملخص جديد")
    topics.merge(conn, keep, [other])
    merged = db.get_topic(conn, keep)
    assert merged["title"] == "عنوان جديد" and merged["edited"] == 1
    assert len(db.topic_chunk_rows(conn, keep)) == n_keep + n_other and db.get_topic(conn, other) is None
    assert topics.delete(conn, keep) == n_keep + n_other
    assert len(db.unassigned_chunks(conn, sid)) == n_keep + n_other


def test_assign_places_unassigned_chunks(conn):
    sid = set_id(conn)
    topics.build(conn, "Set A", "fake", topics.select_chunks(conn, sid), completion=fake_model())
    victim = db.topics(conn, sid)[-1]["id"]
    freed = topics.delete(conn, victim)
    before = len(db.topics(conn, sid))
    placed, new = topics.assign(conn, "Set A", "fake", completion=fake_model())
    assert placed == freed and db.unassigned_chunks(conn, sid) == []
    assert len(db.topics(conn, sid)) == before + new


def test_rechunk_refuses_while_topics_are_linked_and_force_removes_links(conn):
    sid = set_id(conn)
    topics.build(conn, "Set A", "fake", topics.select_chunks(conn, sid), completion=fake_model())
    with pytest.raises(RechunkRefused, match="--force"):
        rechunk(conn, "Set A")
    rechunk(conn, "Set A", force=True)
    assert conn.execute("SELECT COUNT(*) FROM topic_chunks").fetchone()[0] == 0
    assert len(db.empty_topics(conn, sid)) == len(db.topics(conn, sid)) > 0


def test_validation_rejects_bad_replies(conn):
    sid = set_id(conn)
    chunks = topics.select_chunks(conn, sid)
    batch = topics.make_batches(chunks, [], budget=10_000)[0]
    ids = [c["id"] for c in batch.chunks]
    good = {"title": "موضوع", "summary": "ملخص", "chunk_ids": ids}
    topics.validate_map({"topics": [good]}, batch, "ar")
    for bad, message in [
        ({"topics": [{**good, "chunk_ids": ids[:-1]}]}, "missing"),
        ({"topics": [{**good, "chunk_ids": ids + [99999]}]}, "not one of this batch"),
        ({"topics": [good, {**good, "title": "آخر"}]}, "more than once"),
        ({"topics": [{**good, "title": "Savings basics"}]}, "Arabic"),
    ]:
        with pytest.raises(Invalid, match=message):
            topics.validate_map(bad, batch, "ar")
    local = [topics.LocalTopic(f"b1-t{i}", "موضوع", "", [i], "book.pdf", "1-1") for i in range(3)]
    with pytest.raises(Invalid, match="over the limit"):
        topics.validate_reduce({"topics": [{"title": f"موضوع {i}", "local_topic_ids": [f"b1-t{i}"]} for i in range(3)]},
                               local, [], cap=2, language="ar")


def test_topic_cap(conn):
    sid = set_id(conn)
    hints = topics.set_hints(conn, sid)
    one_chapter = topics.select_chunks(conn, sid, chapter=1)
    assert topics.topic_cap(one_chapter, hints) == topics.TOPICS_PER_CHAPTER
    everything = topics.select_chunks(conn, sid)
    assert topics.topic_cap(everything, hints) == 2 * topics.TOPICS_PER_CHAPTER + topics.OUTSIDE_CHAPTERS
    assert topics.topic_cap(everything, []) == 1  # 14 chunks in the largest document / 15, rounded up


def test_estimate_makes_no_model_call(conn):
    chunks = topics.select_chunks(conn, set_id(conn))
    batches = topics.make_batches(chunks, topics.set_hints(conn, set_id(conn)))
    e = topics.estimate("gemini/gemini-2.5-flash", batches, "ar", token_counter=lambda s: len(s) // 3)
    assert e.calls == len(batches) + 1 and e.input_tokens > 0 and e.output_tokens > 0
    assert e.input_price and e.cost() > 0
    assert "Model: gemini/gemini-2.5-flash" in topics.describe(e)


def test_interrupted_build_resumes_without_repeating_finished_batches(conn):
    from mukhtabar.llm import LLMError
    sid = set_id(conn)
    chunks = topics.select_chunks(conn, sid)
    batches = topics.make_batches(chunks, topics.set_hints(conn, sid), budget=1_000)
    assert len(batches) >= 3
    good = fake_model()
    map_calls = []

    def quota_on_second_map(messages, **kwargs):
        if "Group the chunks" in messages[0]["content"]:
            map_calls.append(1)
            if len(map_calls) == 2:
                raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded, limit per day")
        return good(messages, **kwargs)

    with pytest.raises(LLMError):
        topics.build(conn, "Set A", "fake", chunks, completion=quota_on_second_map, batch_chars=1_000)
    assert db.topics(conn, sid) == []  # nothing half-built
    assert topics.cached_batches(conn, "fake", batches, "ar") == 1

    resumed = []

    def counting(messages, **kwargs):
        if "Group the chunks" in messages[0]["content"]:
            resumed.append(1)
        return good(messages, **kwargs)
    topics.build(conn, "Set A", "fake", chunks, completion=counting, batch_chars=1_000)
    assert len(resumed) == len(batches) - 1  # the finished first batch wasn't asked again
    assert db.topics(conn, sid)
    # Kept after success, so a rebuild only reruns routing and merging.
    assert conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == len(batches)


def test_estimate_counts_only_unfinished_batches(conn):
    chunks = topics.select_chunks(conn, set_id(conn))
    batches = topics.make_batches(chunks, topics.set_hints(conn, set_id(conn)), budget=1_000)
    full = topics.estimate("gemini/gemini-2.5-flash", batches, "ar", token_counter=len)
    resumed = topics.estimate("gemini/gemini-2.5-flash", batches, "ar", token_counter=len, saved=1)
    assert resumed.calls == full.calls - 1 and resumed.input_tokens < full.input_tokens
    assert "Resuming: 1 of" in topics.describe(resumed)


def purposes(conn):
    return [r[0] for r in conn.execute("SELECT purpose FROM llm_calls ORDER BY id")]


def test_rebuild_reuses_saved_map_replies_and_reruns_only_the_merge(conn):
    sid = set_id(conn)
    chunks = topics.select_chunks(conn, sid)
    topics.build(conn, "Set A", "fake", chunks, completion=fake_model())
    first = len(purposes(conn))
    topics.build(conn, "Set A", "fake", chunks, force=True, completion=fake_model())
    again = purposes(conn)[first:]
    assert "topics.map" not in again and again  # only routing / merging ran
    with conn:
        db.cache_clear(conn, sid)
    topics.build(conn, "Set A", "fake", chunks, force=True, completion=fake_model())
    assert "topics.map" in purposes(conn)[first + len(again):]


def local_topic(i, n_chunks, chapter="ch", title=None):
    return topics.LocalTopic(f"b1-t{i}", title or f"موضوع {i}", "", list(range(i * 100, i * 100 + n_chunks)),
                             "book.pdf", "1-1", chapter)


def test_each_chapter_gets_a_floor_and_a_cap():
    # 90 chunks in 12 local topics: floor 90/18 = 5, cap max(6, 5) = 6.
    groups = topics.plan_groups([local_topic(i, 9 if i < 6 else 6) for i in range(12)], ["ch"], True)
    assert [(g.chapter, g.floor, g.cap) for g in groups] == [("ch", 5, 6)]
    # A chapter with more material than the usual cap allows raises the cap to its floor.
    big = topics.plan_groups([local_topic(i, 15) for i in range(10)], ["ch"], True)[0]
    assert (big.floor, big.cap) == (9, 9)


def test_floor_never_exceeds_the_local_topics_and_caps_fit_the_global_limit():
    small = topics.plan_groups([local_topic(0, 60), local_topic(1, 60)], ["ch"], True)[0]
    assert (small.floor, small.cap) == (2, 2)
    chapters = [f"ch{c}" for c in range(15)]
    many = [local_topic(c * 10 + i, 3, chapters[c]) for c in range(15) for i in range(10)]
    groups = topics.plan_groups(many, chapters, True)
    assert sum(g.cap for g in groups) <= topics.MAX_TOPICS
    assert all(g.cap >= g.floor for g in groups)


def test_same_title_local_topics_are_merged_before_the_merge_call():
    groups = topics.plan_groups([local_topic(0, 3, title="أ"), local_topic(1, 4, title="أ"),
                                 local_topic(2, 2, title="ب")], ["ch"], True)
    assert [len(t.chunk_ids) for t in groups[0].local] == [7, 2]


def test_merge_reply_with_too_few_topics_is_rejected():
    local = [local_topic(i, 9) for i in range(6)]
    with pytest.raises(Invalid, match="too few"):
        topics.validate_reduce({"topics": [{"title": "كل شيء", "local_topic_ids": [t.id for t in local]}]},
                               local, floor=3, cap=6, language="ar")


def test_review_questions_and_glossary_entries_cannot_be_excluded(conn):
    assert topics.protected_kind("7 ما المقصود بالتضخم؟\nمرجع الإجابة: القسم 1-5") == "questions"
    assert topics.protected_kind("ما الصحيح؟\nأ. الأول\nب. الثاني") == "questions"
    assert topics.protected_kind("الجمعية العمومية السنوية Annual General Meeting AGM\n"
                                 "نظام الشركة الأساسي Articles of Association") == "glossary"
    assert topics.protected_kind("فهرس المحتويات\nمسرد المصطلحات 199\nأسئلة متعددة الخيارات 207") is None
    with conn:
        db.replace_chunks(conn, set_id(conn), db.documents(conn, set_id(conn))[1]["id"],
                          [SimpleNamespace(text=BODY, page=1, page_end=1, position=0),
                           SimpleNamespace(text="3 ما الفرق بين المفهومين؟\nمرجع الإجابة: القسم 2-1",
                                           page=2, page_end=2, position=1)])
    batch = topics.make_batches([c for c in topics.select_chunks(conn, set_id(conn)) if c["filename"] == "summary.pdf"],
                                [], budget=10_000)[0]
    body, questions = (c["id"] for c in batch.chunks)
    with pytest.raises(Invalid, match="study material"):
        topics.validate_map({"topics": [{"title": "موضوع", "chunk_ids": [body]}],
                             "non_study": [{"chunk_id": questions}]}, batch, "ar")


def test_include_and_exclude_are_kept_across_rebuilds(conn):
    sid = set_id(conn)
    chunks = topics.select_chunks(conn, sid)
    topics.build(conn, "Set A", "fake", chunks, completion=fake_model())
    toc = conn.execute("SELECT id FROM chunks WHERE excluded = 1").fetchone()[0]
    some = db.topic_chunk_rows(conn, db.topics(conn, sid)[0]["id"])[0]["id"]
    with conn:
        db.set_user_choice(conn, toc, "include")
        db.set_user_choice(conn, some, "exclude", "not needed")
    assert db.get_chunk(conn, toc)["excluded"] == 0 and db.get_chunk(conn, some)["excluded"] == 1
    assert conn.execute("SELECT COUNT(*) FROM topic_chunks WHERE chunk_id = ?", (some,)).fetchone()[0] == 0
    # A rebuild must not exclude the included chunk again (this fake keeps trying,
    # so every attempt is rejected), nor bring back the excluded one.
    with pytest.raises(LLMError, match="marked as study material"):
        topics.build(conn, "Set A", "fake", topics.select_chunks(conn, sid), force=True, completion=fake_model())
    assert some not in {c["id"] for c in topics.select_chunks(conn, sid)}
    assert db.get_chunk(conn, some)["user_choice"] == "exclude"

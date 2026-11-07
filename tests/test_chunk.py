from mukhtabar.ingest.chunk import MIN_CHARS, chunk_pages, paragraphs

FULL = "هذا سطر طويل من النص العربي يملأ عرض الصفحة كاملا تقريبا ولا ينتهي بعلامة"


def texts(text: str) -> list[str]:
    return [p.text for p in paragraphs([(1, text)])]


def test_lines_join_until_sentence_end():
    paras = paragraphs([(1, f"{FULL}\n{FULL}\nوهنا تنتهي الفقرة.")])
    assert len(paras) == 1
    assert paras[0].text.endswith("تنتهي الفقرة.")


def test_short_line_alone_is_a_heading():
    paras = paragraphs([(1, f"مقدمة\n{FULL}\n{FULL}.")])
    assert paras[0].text == "مقدمة" and paras[0].heading
    assert not paras[1].heading


def test_mcq_options_stay_separate_even_when_wrapped():
    text = "\n".join([
        "1. أي العبارات الآتية صحيحة؟",
        f"أ {FULL}",
        "وتكملة الخيار الأول ✓",
        "ب الخيار الثاني",
        "ج الخيار الثالث",
        "د الخيار الرابع",
    ])
    t = texts(text)
    assert t[1].startswith("أ ") and "ب " not in t[1]
    assert t[2:] == ["ب الخيار الثاني", "ج الخيار الثالث", "د الخيار الرابع"]


def test_options_with_a_dot_or_bracket_are_list_items():
    t = texts(f"{FULL}\nأ. الخيار الأول\nب) الخيار الثاني")
    assert t[1:] == ["أ. الخيار الأول", "ب) الخيار الثاني"]


def test_wrapped_line_starting_with_a_number_continues_the_sentence():
    t = texts(f"{FULL} سوى\n100 وحدة في نهاية المدة.")
    assert len(t) == 1 and "سوى 100 وحدة" in t[0]


def test_wrapped_line_starting_with_a_dash_continues_the_sentence():
    t = texts(f"{FULL}\n– سواء في البداية أو في النهاية - وتستمر الجملة.")
    assert len(t) == 1


def test_plain_number_after_a_finished_sentence_starts_a_new_paragraph():
    t = texts(f"{FULL}.\n12 ما المقصود بالمصطلح؟")
    assert t[1] == "12 ما المقصود بالمصطلح؟"


def test_section_number_starts_a_paragraph_even_after_an_unfinished_line():
    t = texts(f"{FULL}\n3-2 عنوان القسم الجديد\n{FULL}.")
    assert t[1] == "3-2 عنوان القسم الجديد"


def test_table_rows_are_their_own_paragraphs():
    paras = paragraphs([(1, f"{FULL}\nالعمود الأول | العمود الثاني\n{FULL}.")])
    assert [p.text for p in paras][1] == "العمود الأول | العمود الثاني"


def test_paragraph_crossing_a_page_break_records_both_pages():
    chunks = chunk_pages([(3, f"{FULL}\n{FULL}"), (4, "وتنتهي هنا.")])
    assert len(chunks) == 1
    assert (chunks[0].page, chunks[0].page_end) == (3, 4)


def test_chunks_respect_target_and_positions_are_sequential():
    sentence = "جملة قصيرة تنتهي بنقطة."
    pages = [(n, "\n".join([sentence] * 20)) for n in range(1, 6)]
    chunks = chunk_pages(pages, target=300)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks[:-1])
    # The last chunk may have absorbed a tail shorter than the minimum.
    assert len(chunks[-1].text) < 300 + MIN_CHARS
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_overlong_paragraph_splits_at_sentence_ends():
    long_para = " ".join(["هذه جملة في فقرة طويلة جدا."] * 60)
    chunks = chunk_pages([(1, long_para)], target=300)
    assert len(chunks) > 1
    assert all(c.text.endswith(".") for c in chunks)


def test_chunk_does_not_end_on_a_heading():
    body = "جملة قصيرة تنتهي بنقطة. " * 9
    pages = [(1, f"{body.strip()}\nالفصل الثاني\n{body.strip()}")]
    for c in chunk_pages(pages, target=250):
        assert c.text.split("\n")[-1] != "الفصل الثاني"


def test_intro_ending_in_colon_stays_with_what_follows():
    body = "جملة قصيرة تنتهي بنقطة. " * 9
    row = " | ".join(["خلية فيها نص طويل نسبيا لوصف البند"] * 4)
    pages = [(1, f"{body.strip()}\nيبين الجدول الآتي المقارنة:\n{row}\n{row}")]
    chunks = chunk_pages(pages, target=300)
    assert all(not c.text.endswith(":") for c in chunks)
    assert any(c.text.startswith("يبين الجدول الآتي المقارنة:\n") for c in chunks)


def test_no_chunk_below_minimum_before_a_large_paragraph():
    big = " ".join(["جملة ضمن فقرة كبيرة تنتهي بنقطة."] * 25)
    pages = [(1, f"جملة تمهيدية قصيرة.\n{big}\n{big}")]
    chunks = chunk_pages(pages, target=1000)
    assert all(len(c.text) >= MIN_CHARS for c in chunks)
    assert chunks[0].text.startswith("جملة تمهيدية قصيرة.")


def test_carrying_an_intro_never_leaves_a_tiny_chunk_behind():
    # A one-character bullet followed by a long intro: moving the intro to the next
    # chunk must not leave the bullet alone.
    intro = " ".join(["جملة تمهيدية طويلة تشرح ما يأتي بعدها"] * 8) + ":"
    big = " ".join(["جملة ضمن فقرة كبيرة تنتهي بنقطة."] * 25)
    chunks = chunk_pages([(1, f"•\n{intro}\n{big}\n{big}")], target=1000)
    assert all(len(c.text) >= MIN_CHARS for c in chunks)


def test_tiny_tail_joins_the_previous_chunk():
    big = " ".join(["جملة ضمن فقرة كبيرة تنتهي بنقطة."] * 25)
    chunks = chunk_pages([(1, big), (2, "خاتمة قصيرة.")], target=1000)
    assert len(chunks) == 1
    assert chunks[0].text.endswith("خاتمة قصيرة.") and chunks[0].page_end == 2


def test_blank_pages_produce_nothing():
    assert chunk_pages([(1, ""), (2, "  \n ")]) == []

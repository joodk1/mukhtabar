from mukhtabar.ingest.chunk import chunk_pages, paragraphs

FULL = "هذا سطر طويل من النص العربي يملأ عرض الصفحة كاملا تقريبا ولا ينتهي بعلامة"


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
        "1 ما الفرق بين السند الاستثماري والسند غير الاستثماري؟",
        f"أ {FULL}",
        "وتكملة الخيار الأول ✓",
        "ب الخيار الثاني",
        "ج الخيار الثالث",
        "د الخيار الرابع",
    ])
    texts = [p.text for p in paragraphs([(1, text)])]
    assert texts[1].startswith("أ ") and "ب " not in texts[1]
    assert texts[2:] == ["ب الخيار الثاني", "ج الخيار الثالث", "د الخيار الرابع"]


def test_table_rows_are_their_own_paragraphs():
    paras = paragraphs([(1, f"{FULL}\nالعقارات السكنية | العقارات التجارية\n{FULL}.")])
    assert [p.text for p in paras][1] == "العقارات السكنية | العقارات التجارية"


def test_paragraph_crossing_a_page_break_records_both_pages():
    chunks = chunk_pages([(3, f"{FULL}\n{FULL}"), (4, "وتنتهي هنا.")])
    assert len(chunks) == 1
    assert (chunks[0].page, chunks[0].page_end) == (3, 4)


def test_chunks_respect_target_and_positions_are_sequential():
    sentence = "جملة قصيرة تنتهي بنقطة."
    pages = [(n, "\n".join([sentence] * 20)) for n in range(1, 6)]
    chunks = chunk_pages(pages, target=300)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
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


def test_blank_pages_produce_nothing():
    assert chunk_pages([(1, ""), (2, "  \n ")]) == []

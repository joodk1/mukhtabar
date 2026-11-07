from mukhtabar.ingest.rtl_layout import _spans


def words(*spans: tuple[float, float]) -> list[dict]:
    return [{"x0": a, "x1": b} for a, b in spans]


MID = 300.0


def test_word_covering_the_middle_spans():
    assert _spans(words((100, 200), (205, 320), (325, 400)), MID, page_gap=3.5)


def test_narrow_gutter_between_column_lines_does_not_span():
    # Ordinary 3-4pt word gaps on each side, a 12pt gap at the middle: two columns.
    line = words((200, 240), (243.5, 290), (302, 340), (343.4, 380))
    assert not _spans(line, MID, page_gap=3.4)


def test_justified_line_with_stretched_gaps_spans():
    # Every gap stretched to ~9pt, including the one at the middle: one full-width line.
    line = words((180, 230), (239, 291), (300.5, 350), (359, 400))
    assert _spans(line, MID, page_gap=3.4)


def test_two_word_line_uses_the_page_gap():
    assert _spans(words((250, 297), (301, 340)), MID, page_gap=3.4)
    assert not _spans(words((250, 290), (310, 340)), MID, page_gap=3.4)

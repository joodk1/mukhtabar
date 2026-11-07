from mukhtabar.ingest.extract import Line, Page
from mukhtabar.ingest.headers import remove_headers_footers

H = 800.0


def page(*lines: tuple[str, float]) -> Page:
    ls = [Line(text, top, top + 10) for text, top in lines]
    return Page("\n".join(l.text for l in ls), lines=ls, height=H)


def body(n: int) -> tuple[str, float]:
    return (f"نص المحتوى في الصفحة رقم {n} وهو مختلف في كل صفحة", 300)


def test_removes_page_numbers_and_repeated_footer():
    pages = [page(body(n), ("Course Notes — Unit 1", 770), (str(n), 785)) for n in range(1, 11)]
    out, report = remove_headers_footers(pages)
    assert all(p.text == body(n)[0] for n, p in enumerate(out, start=1))
    assert len(report.removed) == 20


def test_keeps_badge_under_a_heading_even_if_repeated():
    # "جديد" sits under a heading at the top of many pages: content, not a header.
    headings = ["الأسهم", "السندات", "المشتقات", "الصناديق", "العملات",
                "التأمين", "المصارف", "الضرائب", "التقاعد", "العقارات"]
    pages = [page((h, 8), ("جديد", 20), body(n)) for n, h in enumerate(headings, start=1)]
    out, report = remove_headers_footers(pages)
    assert all("جديد" in p.text for p in out)
    assert report.removed == []


def test_keeps_rare_running_titles_and_reports_them():
    pages = [page(("السندات" if n < 4 else f"عنوان {n}", 20), body(n)) for n in range(1, 21)]
    out, report = remove_headers_footers(pages)
    assert sum("السندات" in p.text for p in out) == 3
    assert ("top", "السندات", 3) in report.kept


def test_never_removes_mcq_options():
    pages = [page(body(n), ("ب خطأ", 785)) for n in range(1, 11)]
    out, report = remove_headers_footers(pages)
    assert all("ب خطأ" in p.text for p in out)


def test_ignores_lines_away_from_the_edges():
    pages = [page(("ملاحظة مهمة", 400), body(n)) for n in range(1, 11)]
    out, _ = remove_headers_footers(pages)
    assert all("ملاحظة مهمة" in p.text for p in out)


def test_pages_without_positions_pass_through():
    pages = [Page("نص بلا مواقع")]
    out, report = remove_headers_footers(pages)
    assert out[0].text == "نص بلا مواقع" and report.removed == []

"""Milestone 0 spike: compare PDF extractors on real Arabic material.

Usage:
    python spikes/compare_extractors.py [input_dir] [out_dir]

Defaults: samples_private/ in, samples_private/out/ out (both git-ignored).

Optional accuracy scoring: put a hand-copied passage in
    <input_dir>/reference/<pdf name without .pdf>.p<N>.txt
where N is the 1-based page it was copied from.
"""

import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mukhtabar.ingest.extract import BACKENDS, LICENSES, normalize  # noqa: E402

ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓ]")
PRES_FORM = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
DIACRITIC = re.compile(r"[ً-ْٰ]")
ARABIC_WORD = re.compile(r"[ء-يٱ-ۓً-ْٰ]+")
EMPTY_PAGE_CHARS = 20


def metrics(pages: list[str]) -> dict:
    text = "\n".join(pages)
    arabic = len(ARABIC_LETTER.findall(text))
    pres = len(PRES_FORM.findall(text))
    # Shape checks run on normalized, undiacritized words so presentation forms don't hide them.
    words = [DIACRITIC.sub("", w) for w in ARABIC_WORD.findall(normalize(text))]
    n = max(len(words), 1)
    taa_end = sum(w.endswith("ة") for w in words)
    taa_start = sum(w.startswith("ة") for w in words)
    al_start = sum(w.startswith("ال") and len(w) > 3 for w in words)
    la_end = sum(w.endswith("لا") and len(w) > 3 for w in words)
    return {
        "empty_pages": sum(len(p.strip()) < EMPTY_PAGE_CHARS for p in pages),
        "arabic_letters": arabic,
        "pres_form_pct": 100 * pres / max(arabic + pres, 1),
        # ة only ever ends a word; starting one means the word is reversed.
        "taa_reversed_pct": 100 * taa_start / max(taa_start + taa_end, 1),
        # "ال..." is far more common than "...لا"; a flipped ratio means reversal.
        "al_vs_la": f"{al_start}/{la_end}",
        "one_letter_pct": 100 * sum(len(w) == 1 for w in words) / n,
        "long_word_pct": 100 * sum(len(w) > 15 for w in words) / n,
        "avg_word_len": sum(map(len, words)) / n,
        "paren_ok_pct": _paren_ok_pct(text),
    }


def _paren_ok_pct(text: str) -> float:
    """Of lines with a parenthesis, the share whose first one is "(" (as it should be)."""
    ok = bad = 0
    for line in text.splitlines():
        i, j = line.find("("), line.find(")")
        if i >= 0 and (j < 0 or i < j):
            ok += 1
        elif j >= 0:
            bad += 1
    return 100 * ok / max(ok + bad, 1)


def _canon(s: str) -> str:
    s = DIACRITIC.sub("", normalize(s))
    return re.sub(r"\s+", " ", s).strip()


def accuracy(reference: str, page: str) -> tuple[float, float]:
    """Best char similarity of any window of the page vs the reference, and
    the share of reference words found anywhere on the page."""
    ref, pg = _canon(reference), _canon(page)
    if not ref or not pg:
        return 0.0, 0.0
    w = len(ref)
    best = 0.0
    step = max(w // 10, 1)
    for i in range(0, max(len(pg) - w, 0) + 1, step):
        sm = SequenceMatcher(None, ref, pg[i:i + w], autojunk=False)
        if sm.quick_ratio() > best:
            best = max(best, sm.ratio())
    page_words = set(pg.split())
    ref_words = ref.split()
    recall = sum(x in page_words for x in ref_words) / len(ref_words)
    return 100 * best, 100 * recall


def find_references(input_dir: Path, stem: str) -> list[tuple[int, str]]:
    # Compare NFC forms: Arabic filenames typed by hand may compose marks differently than on disk.
    want = unicodedata.normalize("NFC", stem)
    refs = []
    for f in sorted((input_dir / "reference").glob("*.txt")):
        m = re.fullmatch(r"(.+)\.p(\d+)\.txt", unicodedata.normalize("NFC", f.name))
        if m and m.group(1) == want:
            refs.append((int(m.group(2)), f.read_text(encoding="utf-8")))
    return refs


def sample_page(pages: list[str]) -> int:
    """Index of the page with the most Arabic text, for the side-by-side view."""
    counts = [len(ARABIC_LETTER.findall(normalize(p))) for p in pages]
    return max(range(len(pages)), key=counts.__getitem__) if pages else 0


def run_pdf(pdf: Path, input_dir: Path, out_dir: Path) -> list[str]:
    doc_out = out_dir / pdf.stem
    doc_out.mkdir(parents=True, exist_ok=True)
    refs = find_references(input_dir, pdf.stem)
    results = {}
    for name, backend in BACKENDS.items():
        t0 = time.perf_counter()
        try:
            extracted = backend(pdf)
        except Exception as e:  # one broken backend shouldn't stop the comparison
            print(f"  {name}: FAILED ({e!r})")
            continue
        secs = time.perf_counter() - t0
        raw = [p.text for p in extracted]
        clean = [normalize(p) for p in raw]
        flagged = [i + 1 for i, p in enumerate(extracted) if p.flags]
        results[name] = (raw, clean, secs, flagged)
        for mode, pages in (("raw", raw), ("clean", clean)):
            (doc_out / f"{name}.{mode}.txt").write_text(
                "".join(f"\n\n===== page {i + 1} =====\n{p}" for i, p in enumerate(pages)),
                encoding="utf-8",
            )
        print(f"  {name}: {len(raw)} pages in {secs:.1f}s")

    lines = [f"## {pdf.name}", ""]
    header = ("| backend | license | mode | secs | empty pages | flagged pages | pres-form % | ة-reversed % "
              "| ال/لا | 1-letter % | >15-letter % | avg len | '(' first % |")
    if refs:
        header += " ref sim % | ref word recall % |"
    lines += [header, header_rule(header)]
    for name, (raw, clean, secs, flagged) in results.items():
        for mode, pages in (("raw", raw), ("clean", clean)):
            m = metrics(pages)
            row = (f"| {name} | {LICENSES[name]} | {mode} | {secs:.1f} | {m['empty_pages']}/{len(pages)} "
                   f"| {len(flagged)} | {m['pres_form_pct']:.1f} | {m['taa_reversed_pct']:.1f} | {m['al_vs_la']} "
                   f"| {m['one_letter_pct']:.1f} | {m['long_word_pct']:.1f} | {m['avg_word_len']:.2f} "
                   f"| {m['paren_ok_pct']:.0f} |")
            if refs:
                scores = [accuracy(text, pages[p - 1]) if p <= len(pages) else (0.0, 0.0) for p, text in refs]
                sim = sum(s for s, _ in scores) / len(scores)
                rec = sum(r for _, r in scores) / len(scores)
                row += f" {sim:.1f} | {rec:.1f} |"
            lines.append(row)
    lines.append("")
    for name, (*_, flagged) in results.items():
        if flagged:
            lines.append(f"Flagged by {name} (possible columns, not split): pages {flagged}")
            lines.append("")

    if results:
        first_raw = next(iter(results.values()))[0]
        idx = refs[0][0] - 1 if refs else sample_page(first_raw)
        side = [f"# {pdf.name} — page {idx + 1}, every extractor (clean)", ""]
        for name, (_, clean, *_) in results.items():
            excerpt = clean[idx][:1500] if idx < len(clean) else ""
            side += [f"## {name}", "", "```", excerpt, "```", ""]
        (doc_out / "side_by_side.md").write_text("\n".join(side), encoding="utf-8")
        lines.append(f"Side-by-side of page {idx + 1}: `{(doc_out / 'side_by_side.md').as_posix()}`")
        lines.append("")
    return lines


def header_rule(header: str) -> str:
    return "|" + "---|" * (header.count("|") - 1)


def main() -> None:
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples_private")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_dir / "out"
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {input_dir}")
    report = ["# Extractor comparison", ""]
    for pdf in pdfs:
        print(pdf.name)
        report += run_pdf(pdf, input_dir, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nReport: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

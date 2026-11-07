# Mukhtabar (مختبر) — Plan

A study-buddy app that quizzes you on your own material and works with you on your weakest spots.

Arabic-first, but it isn't tied to one language. Works with any model provider.

## Working agreement (for Claude Code)

- Discuss and agree on an approach before building a milestone. Small plan first, code second.
- One milestone at a time. Stop at the end of each one and show what works.
- Keep it simple. No feature that isn't in this plan without asking.

## What it does

1. **Ingest** a study set (text PDFs to start) and split it into chunks.
2. **Topic map:** the model outlines the material into topics, each linked to its source chunks.
3. **Diagnostic:** a broad test across every topic.
4. **Weakness profile:** a mastery score per topic, visible in the UI.
5. **Targeted practice:** drill weak topics, explain mistakes using the user's own material.
6. **Re-test over time** (spaced repetition) so scores reflect what is actually retained.

## Not doing (for now)

- Scanned PDFs / OCR, slides, Word docs
- User accounts, multi-user hosting
- Mobile app

## Decisions already made

| Area | Choice | Why |
|---|---|---|
| UI | Streamlit | Pure Python, easy RTL with a little CSS, easy charts |
| Models | LiteLLM | One interface for Claude, OpenAI, Gemini, local models. User supplies model name + key |
| Storage | SQLite | Mastery history, due dates, per-question records are easy to query |
| Language | Follows the material | Arabic material gets Arabic questions; same for any language |
| Diagnostic | Fixed **and** adaptive | One engine, two selection strategies (see below) |
| Questions | Grounded | Every question stores the source chunk it came from |
| PDF extraction | pdfplumber + own RTL layout (MIT) | Only option correct on both logical- and visual-order Arabic PDFs; PyMuPDF (AGPL) breaks every لا (milestone 0) |
| Text normalization | NFKC, strip harakat and tatweel only | Alef/hamza/ya/ta marbuta left as written. **Unsuitable for Quranic text or poetry**, where harakat carry meaning |

## Extraction (milestone 0 findings)

- Arabic PDFs store text in logical order (e.g. InDesign) or visual order (e.g. Word). Plain extractors each fail on one kind, so reading order is rebuilt from glyph positions.
- Brackets are stored by shape in some PDFs and by meaning in others; this is decided per page.
- Two-column pages are split at a central gutter; ruled tables are read cell by cell. Pages that look multi-column without a clear gutter are laid out as one column and flagged.
- Known limits: digit/punctuation sequences such as `10-1-1` or `24 ÷ 6 = 4` keep visual order; a grid of boxes is read column by column rather than row by row; tables without ruling lines aren't detected.
- Ingest must compare filenames in NFC form: Arabic names typed by hand can be composed differently from the names on disk.

## Testing

- `samples_private/` won't exist for other people, so public tests use tiny synthetic Arabic PDFs in `tests/fixtures/`, one stored in logical order and one in visual order, covering لا ligatures, brackets, English/number runs, two columns, and a table.

## Folder layout

```
mukhtabar/
  app.py                 # Streamlit entry point
  mukhtabar/
    llm.py               # LiteLLM wrapper, JSON-output helper with retry
    ingest/
      extract.py         # PDF -> per-page text; pluggable backends, default pdfplumber-rtl
      rtl_layout.py      # reading order from glyph positions: RTL lines, columns, tables
      chunk.py           # text -> chunks
    topics.py            # topic map from chunks
    questions.py         # generate MCQ + short-answer, with source refs
    grading.py           # MCQ exact match, short-answer graded by model
    mastery.py           # per-topic scores + spaced-repetition schedule
    selection.py         # fixed and adaptive "what to ask next"
    db.py                # SQLite schema + queries
  tests/                 # run with a mocked LLM, no API calls
    fixtures/            # tiny synthetic Arabic PDFs, logical and visual order
  sample/                # tiny original text so the repo works out of the box
  spikes/                # throwaway experiments (milestone 0 extractor comparison)
  README.md  PLAN.md  LICENSE  .gitignore  pyproject.toml
```

Each study set lives in its own folder/record: its material, chunks, topics, questions, and progress.

## Data model (SQLite, sketch)

- `study_sets` — id, name, language, created_at
- `documents` — id, study_set_id, filename, page_count, added_at
- `chunks` — id, study_set_id, document_id, page, text, position
- `topics` — id, study_set_id, title, summary
- `topic_chunks` — topic_id, chunk_id
- `questions` — id, topic_id, chunk_id, type, prompt, options, answer, explanation
- `attempts` — id, question_id, correct (or 0–1 score), answered_at
- `mastery` — topic_id, score (0–1), attempts_count, due_at

## Mastery model

- Each topic has a score from 0 to 1, updated after every answer with an exponentially weighted average, so recent answers count more.
- Track attempt count too. A score from 2 answers is less trustworthy than one from 20.
- Spaced repetition: simple Leitner / SM-2-lite. Correct answers push the next review further out, wrong answers pull it back.
- Scheduling is **per topic, by design**, not per question. The goal is retained understanding of a topic, and questions are drawn fresh (or from cache) each review, so a per-question schedule isn't needed.
- Short-answer questions give partial credit (0–1) instead of just right/wrong.

## Diagnostic modes (one engine)

Both modes call `selection.next_question(state)`. Only the strategy differs.

- **Fixed:** walks through every topic round-robin until each has *k* questions. Guarantees full coverage.
- **Adaptive:** picks the next topic by weakness and uncertainty (low score, few attempts, or overdue). Zooms in on weak spots faster.

## Question generation rules

- Generate from specific chunks, never from the model's general knowledge.
- Store the source chunk with each question so answers can be verified.
- Validate model JSON (4 options, valid answer index) and retry on failure.
- Deduplicate against questions already asked.
- Cache generated questions so repeat sessions don't re-spend tokens.

## Milestones

0. **Repo setup + extractor spike.** Compare pypdf, pdfplumber, pypdfium2 and PyMuPDF on real Arabic material (each document type separately), raw and with Unicode normalization. Check for reversed or disconnected letters, lost spaces, and scanned pages. Make the extractor pluggable and prefer a permissive license (PyMuPDF is AGPL) unless PyMuPDF is clearly better. Everything after depends on this.
1. **Ingest + SQLite.** Upload a PDF, extract, chunk, store.
2. **Topic map.** Generate topics from chunks. Let the user rename, merge, or delete topics, since model outlines can be rough.
3. **Question generation.** MCQ and short-answer, grounded, validated, cached.
4. **Quiz engine + mastery.** Both diagnostic modes, scoring, mastery updates.
5. **Streamlit UI.** Upload, choose study set, take a quiz, see a weak-topics dashboard. RTL support when the language is Arabic.
6. **Practice loop.** Targeted drills on weak topics, explanations from the source, spaced-repetition scheduling.
7. **Polish.** README with a screenshot, tests, license, sample study set.

## Public-repo checklist

- [ ] `.gitignore`: API keys / `.env`, the database file, uploaded PDFs, user study sets
- [ ] `LICENSE` (MIT)
- [ ] README disclaimer: unofficial study aid, not affiliated with any exam body, AI-generated questions can be wrong, verify against official materials
- [ ] Keys read from environment or the UI session only, never stored in the repo

## Open questions

- Chunk size that works best for Arabic text
- How many questions to generate per call vs. cost
- How to auto-detect language, or just ask the user when creating a study set
- Whether weak-topic explanations should be a chat or a one-shot card
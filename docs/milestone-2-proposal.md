# Milestone 2 proposal: topic map

Status: proposal for review. Nothing here is implemented yet.

## Goal

Turn a study set's chunks into a list of topics, each linked to the chunks it came
from, using whichever model the user configures. Then let the user fix the outline by
renaming, merging and deleting topics. Topics are what mastery (milestone 4) is scored
on, so they should be study-sized: a concept area you can write several questions
about. A single sentence is too small and a whole chapter is too big.

The first real study set has three documents of different types (a textbook, a
revision summary and a question bank). Together they have about 870 chunks, of which
about 590 come from the textbook, and about 740k characters (about 120k words). All
documents in a set map onto **one** set of topics. A topic links to chunks from the
textbook, the summary and the question bank together, which later lets questions and
explanations draw on all three.

## Decisions taken

- **Rechunking:** refuse once a topic map exists; `--force` deletes that document's
  topic links (section 6).
- **One primary topic per chunk**, with `topic_chunks` kept many-to-many so secondary
  links can be added later (section 5).
- **No fixed topic target.** The document's own chapter structure is used as a hint,
  with a cap derived from it (section 2).

## 1. LiteLLM wrapper (`mukhtabar/llm.py`)

One function does all model calls:

```python
def complete_json(
    messages: list[dict],
    *,
    validate: Callable[[Any], T],   # parses the reply and raises ValueError with a readable message
    model: str,                     # any LiteLLM model string, e.g. "anthropic/...", "ollama/..."
    purpose: str,                   # "topics.map", "topics.reduce", ... for the usage log
    max_attempts: int = 3,
    temperature: float = 0.0,
    completion: Callable = litellm.completion,  # injectable, so tests never call a model
) -> T:
```

- **Transport errors** (rate limits, timeouts, 5xx) are retried with backoff by
  LiteLLM itself (`num_retries=2`). They don't count as attempts.
- **JSON mode:** `response_format={"type": "json_object"}` is requested when LiteLLM
  reports the model supports it. Otherwise the prompt alone asks for JSON. Either way
  the reply is parsed defensively: strip code fences, take the outermost `{...}`, then
  `json.loads`.
- **Validation:** every reply goes through `validate`. It covers both structural
  checks (a pydantic model; pydantic already comes with LiteLLM) and semantic checks
  (the rules in section 2).
- **Retry on invalid output:** the invalid reply is appended as the assistant turn,
  followed by "Your reply was invalid: <error>. Reply again with corrected JSON only."
  Up to `max_attempts` attempts in total. After that the call raises `LLMError`, which
  carries the last error and the raw reply.
- **Usage log:** each call records the model, input/output tokens, cost (when LiteLLM
  can compute it), attempts and success in an `llm_calls` table. After the first run,
  that gives real numbers in place of the estimate in section 3.
- **Keys:** LiteLLM reads the provider's usual environment variables
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, ...). Keys are never
  written to the database or the repo. The model is chosen with `--model` or
  `MUKHTABAR_MODEL`. There is deliberately no default, so nothing is sent anywhere by
  accident.
- **Tests:** a fake `completion` returns scripted replies, for example an invalid one
  and then a valid one. No network.

## 2. Building the map: hints, map, then reduce

A textbook is too long for one call on many models, especially local ones, so the map
is built in passes.

**Chapter hints (no model call).**

- **Where they come from:** running titles, the chapter name printed at the top of
  most pages. Ingest already detects these as "repeated at a page edge but kept".
  Milestone 2 recomputes them from the stored pages: a line that opens at least 3
  pages of a document is a candidate, ordered by first appearance, with the page
  range it covers.
- **Fallback:** for documents without running titles (the summary and the question
  bank have none), section headings that start with a section number such as `3-1`
  are used instead. If neither exists, there are no hints.
- **How they're used:**
  - map batches are cut at chapter boundaries where possible;
  - each batch is told which chapter it's in;
  - the reduce step gets the chapter list as the expected top-level structure.

**Map (per batch of consecutive chunks, per document).**

- **Batch size:** batches are cut by a character budget (default about 25k
  characters, roughly 8–12k tokens; `--batch-chars` for small-context local models),
  preferring chapter boundaries.
- **Input:** each chunk is sent as `<chunk id="417" pages="153-154">...</chunk>`,
  together with the chapter hint and the previous batch's last topic title, so a topic
  continuing across the boundary keeps its name.
- **Output:**

  ```json
  {"topics": [{"title": "...", "summary": "...", "chunk_ids": [417, 418]}],
   "non_study": [{"chunk_id": 12, "reason": "table of contents"}]}
  ```

- **Validation rules:**
  - every chunk id is from this batch;
  - every chunk in the batch appears exactly once, either in one topic or in
    `non_study` (this gives the primary topic);
  - 1–12 topics per batch;
  - titles are non-empty, at most about 10 words, with no duplicates within the batch;
  - when the set's language is `ar`, titles and summaries are mostly Arabic.

**Reduce (once per study set).**

- **Input:** all local topics as `{id, title, summary, document, pages, n_chunks}`
  plus the chapter list. Titles and summaries only, no chunk text. For the first set
  that's roughly 300 local topics, about 20k tokens.
- **Output:**

  ```json
  {"topics": [{"title": "...", "summary": "...", "chapter": "...", "local_topic_ids": ["b3-t2", "b17-t1"]}]}
  ```

- **Validation rules:**
  - every local topic id is used exactly once;
  - final titles are unique;
  - `chapter` is one of the hints, or empty when there are none;
  - the count is within the cap below.
- **Chunks follow deterministically:** chunk links come from the local topics, so the
  model never re-types chunk ids in this pass.
- **If the local-topic list is too big** for one call, reduce per document first, then
  across documents.

**Topic cap (no fixed target).**

The model chooses how many topics the material needs, guided by the chapters. Only
the upper limit is enforced. Above the cap, the reply is invalid, and the retry asks
it to merge the closest topics.

- **With chapter hints:** at most **6 topics per chapter** (for example, 10 chapters
  give at most 60), plus up to 5 for material outside any chapter. The prompt suggests
  "usually 2–5 per chapter".
- **Without hints:** at most `ceil(study chunks of the largest document / 15)`, capped
  at 60. The largest document is used, not the total, because a summary and a question
  bank repeat the textbook's content rather than adding new topics.
- **No minimum.** A short document can legitimately have 3 topics.
- **Why these numbers:**
  - 6 per chapter keeps a topic at roughly 5–15 textbook pages, big enough for
    several questions;
  - 60 keeps the mastery dashboard readable;
  - both are settings, and the comparison run (section 8) will show whether they fit.

`non_study` chunks (table of contents, copyright page, exam logistics) are marked
excluded. They're kept in the database but skipped by question generation, and the
user can un-exclude them.

**Prompt approach.**

- **Role:** the system prompt describes the task: outline study material for exam
  revision.
- **Grounding:** use only the provided text; never add topics from general knowledge.
- **Language:** write titles and summaries in the study set's language.
- **Granularity:** give the size definition, with one good and one bad example
  (original examples, not taken from any user's material).
- **Structure:** follow the chapter hints.
- **Output:** return JSON only.
- **Settings:** temperature 0. Prompts live in `mukhtabar/prompts/` as plain text
  files, so they can be reviewed without reading code.

## 3. Rough size and cost

Estimates only; the `llm_calls` log will give real numbers. Arabic costs more tokens
per word than English. I'm assuming 1.5–3 tokens per word, which isn't measured,
because counting tokens needs an API call.

| | Calls | Input tokens | Output tokens |
|---|---|---|---|
| Map: textbook (~20 batches), summary (~7), question bank (~3) | ~30 | 180k–360k text + ~20k prompts | ~30k |
| Reduce | 1–2 | ~20k | ~4k |
| Retries (assume 10–20%) | +3–6 | +20k–80k | +5k |
| **Total** | **~35** | **~240k–480k** | **~40k** |

- **Cost formula:** cost ≈ input tokens × input price + output tokens × output price.
- **Illustrative example:** at **$3 per million input tokens and $15 per million
  output**, that comes to roughly **$1.30–$2.10** for the whole set. These rates are
  placeholders, not a quote. Check your provider's current pricing for the model you
  pick.
- **One-off cost:** the map is built once per study set and stored. The one-chapter
  comparison in section 8 costs about a tenth of this per model.

## 4. Editing topics (command line until the UI exists)

```
python -m mukhtabar.topics build   --set NAME --model ... [--force]
python -m mukhtabar.topics list    --set NAME [--unassigned] [--empty]
python -m mukhtabar.topics show    TOPIC_ID            # title, summary, linked chunks with pages
python -m mukhtabar.topics rename  TOPIC_ID "new title" [--summary "..."]
python -m mukhtabar.topics merge   KEEP_ID OTHER_ID [OTHER_ID ...]
python -m mukhtabar.topics delete  TOPIC_ID
```

- **merge:** moves the other topics' chunk links onto `KEEP_ID` and deletes the others.
  It keeps `KEEP_ID`'s title and summary; use `rename` to change them. No model call.
- **delete:** removes the topic. Its chunks become unassigned, and `list --unassigned`
  shows them. Unassigned chunks are skipped by question generation, the same as
  excluded ones.
- **Edits are protected:** every edit marks the topic `edited`. `build` refuses when a
  map already exists, and `build --force` warns how many edited topics will be lost.
- **Sets stay separate:** a topic can only merge with topics from the same study set.

## 5. Schema additions (migration 2)

```sql
CREATE TABLE topics (
    id           INTEGER PRIMARY KEY,
    study_set_id INTEGER NOT NULL REFERENCES study_sets(id) ON DELETE CASCADE,
    title        TEXT NOT NULL,
    summary      TEXT NOT NULL DEFAULT '',
    chapter      TEXT,                       -- the chapter hint it falls under, if any
    position     INTEGER NOT NULL,           -- outline order (first appearance)
    edited       INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE topic_chunks (
    topic_id   INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    chunk_id   INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    is_primary INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (topic_id, chunk_id)
);
-- At most one primary topic per chunk; secondary links can be added later.
CREATE UNIQUE INDEX one_primary_topic_per_chunk ON topic_chunks (chunk_id) WHERE is_primary = 1;
ALTER TABLE chunks ADD COLUMN excluded INTEGER NOT NULL DEFAULT 0;
ALTER TABLE chunks ADD COLUMN excluded_reason TEXT;
CREATE TABLE llm_calls (
    id            INTEGER PRIMARY KEY,
    study_set_id  INTEGER REFERENCES study_sets(id) ON DELETE SET NULL,
    purpose       TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL,
    attempts      INTEGER NOT NULL,
    ok            INTEGER NOT NULL,
    error         TEXT,
    created_at    TEXT NOT NULL
);
```

In milestone 2 every link is primary. `merge` keeps the rule by construction: a
chunk's single primary link just moves to the kept topic.

## 6. Rechunking once topics exist (decided: option 1)

**Rule:** `rechunk` refuses when any of the document's chunks are linked to a topic:

```
Refusing: 412 chunks of book.pdf are linked to 31 topics.
Rechunking replaces chunks and would remove those links.
Re-run with --force to delete this document's topic links and rechunk.
```

With `--force`:

1. Delete the document's topic links. The `ON DELETE CASCADE` on `topic_chunks` does
   this when the chunks are replaced.
2. Rechunk.
3. Report topics that are now empty, and that the document's new chunks are
   unassigned.

Empty topics aren't deleted automatically. You decide, with `list --empty` and `delete`.

**Why this is the right default:**

- **Nothing is lost silently.** Without the guard, the cascade would remove links with
  no warning.
- **It's the simplest rule that's always correct.** Remapping by page range would be
  approximate and could quietly attach the wrong chunks to topics you edited.
- **Rechunking is mostly a pre-topic activity** (tuning chunk size), so the guard will
  rarely get in the way.

**Concerns:**

1. **New chunks need a home after `--force`, and the same applies to adding a document
   to an existing set.** Both leave unassigned chunks. A small `topics assign --set S`
   step would fix that: map the unassigned chunks, then one call that places each new
   local topic into an existing topic or proposes a new one. It's the same machinery
   as the reduce step. Without it, the only remedy is `build --force`, which throws
   away your edits. I'd include it in milestone 2; it's open below.
2. **Milestone 3 raises the stakes.** Questions will store `chunk_id`, and attempts
   reference questions. `--force` must then also refuse, or say exactly what it
   deletes, when questions or attempts depend on those chunks. I'd make
   `questions.chunk_id` `ON DELETE SET NULL`, so answer history survives.
3. **Deleting a topic later loses mastery history** once attempts exist (milestone 4).
   At that point "delete" should probably become "archive". Not needed in milestone 2.

## 7. Chunk quality going in

Both chunker fixes from the extraction audit are applied before milestone 2, so no
rechunk (and no `--force`) is needed after the first topic map:

- **Wrapped lines:** a wrapped line that begins with a plain number or a dash no longer
  starts a new paragraph, which had caused the few mid-sentence chunk breaks.
- **Short chunks:** chunks are never closed below 200 characters, and an intro line
  ending in `:` stays with what it introduces.

## 8. Comparing two models on one chapter before the full run

**What runs:**

- **One chapter only:** a mid-sized chapter of the textbook (about 50–70 chunks), plus
  the summary and question-bank chunks under the same chapter hint, so cross-document
  merging is tested too. About a tenth of the full cost per model.
- **Both models on identical input:** same batches, same prompts, temperature 0.
- **Its own database:** a scratch copy under `samples_private/`, so nothing touches the
  real set.

**Measured automatically (from `llm_calls` and validation):**

- valid replies on the first attempt, retries used, failures;
- input/output tokens, cost, wall time;
- number of topics, chunks per topic (smallest, median, largest), excluded chunks;
- titles in the wrong language;
- how often the same chunk lands in a different topic when a model is run twice,
  which shows stability.

**Judged by you (about 15 minutes per model):**

- **Side by side:** a local HTML page with both outlines (title, summary, page ranges),
  using the same approach as the extraction review pack.
- **Granularity:** for each outline, are the topics study-sized? Count too broad, too
  narrow, and duplicates.
- **Assignments:** 10 randomly sampled chunks (the same 10 for both models), each
  shown with the topic it was put in. Right topic: yes or no.
- **Faithfulness:** any topic that isn't actually in the chapter, which would be
  invented from general knowledge.

**How to decide:** the cheaper model wins unless the stronger one is clearly better on
assignments and granularity. Ties go to fewer retries.

## Decisions still needed

1. **Should `topics assign`** (for adding documents and recovering after
   `rechunk --force`) be part of milestone 2? I recommend yes.
2. **Are the cap numbers acceptable:** 6 per chapter, or 1 per 15 chunks, with a
   ceiling of 60?
3. **Which provider and models** for the comparison. The material itself is sent to
   that provider.

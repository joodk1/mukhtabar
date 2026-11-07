"""Topic map: chunks -> study topics, with chapter hints, map/reduce, assign, and editing.

    python -m mukhtabar.topics hints    --set NAME
    python -m mukhtabar.topics estimate --set NAME --model MODEL [--chapter N] [--only FILE:FIRST-LAST ...]
    python -m mukhtabar.topics build    --set NAME --model MODEL [--chapter N] [--only ...] [--force] [--yes]
    python -m mukhtabar.topics assign   --set NAME --model MODEL [--yes]
    python -m mukhtabar.topics list     --set NAME [--unassigned] [--empty]
    python -m mukhtabar.topics show     TOPIC_ID
    python -m mukhtabar.topics rename   TOPIC_ID "new title" [--summary "..."]
    python -m mukhtabar.topics merge    KEEP_ID OTHER_ID [OTHER_ID ...]
    python -m mukhtabar.topics delete   TOPIC_ID
    python -m mukhtabar.topics calls    --set NAME

Model calls happen only in build and assign, and only after the estimate is
shown and confirmed (interactively, or with --yes).
"""

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from mukhtabar import db
from mukhtabar.llm import Invalid, LLMError, complete_json

PROMPTS = Path(__file__).parent / "prompts"
LANGUAGE_NAMES = {"ar": "Arabic", "en": "English"}

BATCH_CHARS = 25_000        # map batch budget (characters of chunk text)
MAP_MAX_TOPICS = 12         # per map batch
TOPICS_PER_CHAPTER = 6      # cap with chapter hints
OUTSIDE_CHAPTERS = 5        # extra topics allowed for material outside any chapter
CHUNKS_PER_TOPIC = 15       # cap without hints: largest document's chunks / this
MAX_TOPICS = 60
TITLE_WORDS = 12            # the prompt asks for 10; a little slack avoids needless retries
MIN_HINT_PAGES = 3          # a running title must open this many pages...
MIN_HINT_RUN = 3            # ...including this many in a row
HINT_MAX_GAP = 2            # "in a row" allows the title on every other page
OPENER_PAGES = 2            # pages just before a chapter's first running title belong to it
MIN_LANGUAGE_SHARE = 0.5    # share of letters in the set's script, for titles and summaries

ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓ]")
LATIN_LETTER = re.compile(r"[A-Za-z]")


# --- Chapter hints ---------------------------------------------------------

@dataclass
class Hint:
    title: str
    document_id: int
    first_page: int
    last_page: int


def _runs(seq: list[tuple[int, str]]) -> list[list]:
    """[title, first page, last page, pages] for each run of the same title on
    nearby pages (running titles may appear on every other page only)."""
    runs: list[list] = []
    for page, title in seq:
        if runs and runs[-1][0] == title and page - runs[-1][2] <= HINT_MAX_GAP:
            runs[-1][2] = page
            runs[-1][3] += 1
        else:
            runs.append([title, page, page, 1])
    return runs


def chapter_hints(conn: sqlite3.Connection, document_id: int) -> list[Hint]:
    """Chapters from running titles: the first line of a page, repeated on at least
    MIN_HINT_PAGES pages including a run of MIN_HINT_RUN in a row (one-off repeats
    such as an "example" box heading don't qualify). No model call."""
    pages = db.pages(conn, document_id)
    if not pages:
        return []
    first = [(p["page"], p["text"].strip().split("\n")[0].strip()) for p in pages if p["text"].strip()]
    counts = Counter(t for _, t in first)
    candidates = {t for t, n in counts.items()
                  if n >= MIN_HINT_PAGES and (ARABIC_LETTER.search(t) or LATIN_LETTER.search(t))}
    seq = [(p, t) for p, t in first if t in candidates]
    keep = {r[0] for r in _runs(seq) if r[3] >= MIN_HINT_RUN}
    starts: list[tuple[str, int]] = []
    for title, a, _b, _n in _runs([(p, t) for p, t in seq if t in keep]):
        if not starts or starts[-1][0] != title:
            starts.append((title, a))
    last_page = pages[-1]["page"]
    hints = []
    for i, (title, a) in enumerate(starts):
        first_page = a - OPENER_PAGES if i else a
        nxt = starts[i + 1][1] - OPENER_PAGES - 1 if i + 1 < len(starts) else last_page
        hints.append(Hint(title, document_id, max(first_page, 1), max(nxt, a)))
    return hints


def set_hints(conn: sqlite3.Connection, study_set_id: int) -> list[Hint]:
    return [h for d in db.documents(conn, study_set_id) for h in chapter_hints(conn, d["id"])]


def chapter_of(chunk: sqlite3.Row, hints: list[Hint]) -> str | None:
    for h in hints:
        if h.document_id == chunk["document_id"] and h.first_page <= chunk["page"] <= h.last_page:
            return h.title
    return None


# --- Selecting chunks --------------------------------------------------------

def parse_only(items: list[str]) -> list[tuple[str, int, int]]:
    """"FILE:FIRST-LAST" -> (NFC filename, first, last)."""
    out = []
    for item in items:
        name, _, span = item.rpartition(":")
        a, _, b = span.partition("-")
        out.append((unicodedata.normalize("NFC", name), int(a), int(b or a)))
    return out


def select_chunks(conn, study_set_id: int, only: list[tuple[str, int, int]] | None = None,
                  chapter: int | None = None) -> list[sqlite3.Row]:
    """Chunks of a set for a build, optionally limited to page ranges and/or one
    chapter (1-based, as listed by `hints`). A chunk is in a range if it starts in
    it. Previously excluded chunks are included: a build decides afresh."""
    rows = db.study_chunks(conn, study_set_id)
    if only is None and chapter is None:
        return rows
    ranges = list(only or [])
    if chapter is not None:
        hints = set_hints(conn, study_set_id)
        if not 1 <= chapter <= len(hints):
            raise ValueError(f"No chapter {chapter}; `hints` lists {len(hints)}")
        h = hints[chapter - 1]
        name = conn.execute("SELECT filename FROM documents WHERE id = ?", (h.document_id,)).fetchone()[0]
        ranges.append((name, h.first_page, h.last_page))
    return [c for c in rows if any(c["filename"] == f and a <= c["page"] <= b for f, a, b in ranges)]


# --- Batching ----------------------------------------------------------------

@dataclass
class Batch:
    index: int
    chunks: list[sqlite3.Row]
    chapters: list[str] = field(default_factory=list)


def make_batches(chunks: list[sqlite3.Row], hints: list[Hint], budget: int = BATCH_CHARS) -> list[Batch]:
    """Consecutive chunks of one document per batch, within a character budget,
    cut at chapter boundaries unless the batch so far is small."""
    batches: list[Batch] = []
    current: list[sqlite3.Row] = []
    size = 0
    last_doc = last_chapter = None

    def close():
        nonlocal current, size
        if current:
            chapters = list(dict.fromkeys(c for c in (chapter_of(x, hints) for x in current) if c))
            batches.append(Batch(len(batches) + 1, current, chapters))
        current, size = [], 0

    for c in chunks:
        chapter = chapter_of(c, hints)
        new_doc = last_doc is not None and c["document_id"] != last_doc
        new_chapter = chapter != last_chapter and size >= 0.2 * budget
        if current and (new_doc or new_chapter or size + len(c["text"]) > budget):
            close()
        current.append(c)
        size += len(c["text"])
        last_doc, last_chapter = c["document_id"], chapter
    close()
    return batches


# --- Prompts and validation --------------------------------------------------

def _prompt(name: str, **values) -> str:
    text = (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def _language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def _check_language(text: str, language: str, what: str) -> None:
    if language != "ar":
        return
    arabic, latin = len(ARABIC_LETTER.findall(text)), len(LATIN_LETTER.findall(text))
    if arabic + latin and arabic < MIN_LANGUAGE_SHARE * (arabic + latin):
        raise Invalid(f"{what} must be written in Arabic: {text[:60]!r}")


def _check_title(title, language: str) -> str:
    if not isinstance(title, str) or not title.strip():
        raise Invalid("every topic needs a non-empty title")
    title = title.strip()
    if len(title.split()) > TITLE_WORDS:
        raise Invalid(f"title too long (more than {TITLE_WORDS} words): {title[:60]!r}")
    _check_language(title, language, "titles")
    return title


def _check_summary(summary, language: str) -> str:
    summary = (summary or "").strip() if isinstance(summary, (str, type(None))) else ""
    if summary:
        _check_language(summary, language, "summaries")
    return summary


@dataclass
class LocalTopic:
    id: str
    title: str
    summary: str
    chunk_ids: list[int]
    document: str
    pages: str


@dataclass
class MapResult:
    topics: list[LocalTopic]
    non_study: dict[int, str]


def validate_map(obj, batch: Batch, language: str) -> MapResult:
    if not isinstance(obj, dict) or not isinstance(obj.get("topics"), list):
        raise Invalid('expected {"topics": [...], "non_study": [...]}')
    expected = {c["id"]: c for c in batch.chunks}
    seen: dict[int, str] = {}
    topics, titles = [], set()
    if not 1 <= len(obj["topics"]) <= MAP_MAX_TOPICS:
        raise Invalid(f"return between 1 and {MAP_MAX_TOPICS} topics, not {len(obj['topics'])}")
    for i, t in enumerate(obj["topics"], start=1):
        if not isinstance(t, dict):
            raise Invalid("each topic must be an object")
        title = _check_title(t.get("title"), language)
        if title in titles:
            raise Invalid(f"duplicate title {title!r}")
        titles.add(title)
        ids = t.get("chunk_ids")
        if not isinstance(ids, list) or not ids:
            raise Invalid(f"topic {title!r} has no chunk_ids")
        for cid in ids:
            if not isinstance(cid, int) or cid not in expected:
                raise Invalid(f"chunk id {cid!r} is not one of this batch's chunks")
            if cid in seen:
                raise Invalid(f"chunk {cid} appears more than once ({seen[cid]!r} and {title!r})")
            seen[cid] = title
        rows = [expected[c] for c in ids]
        topics.append(LocalTopic(f"b{batch.index}-t{i}", title, _check_summary(t.get("summary"), language),
                                 list(ids), rows[0]["filename"],
                                 f"{min(r['page'] for r in rows)}-{max(r['page_end'] for r in rows)}"))
    non_study: dict[int, str] = {}
    for n in obj.get("non_study") or []:
        cid = n.get("chunk_id") if isinstance(n, dict) else None
        if not isinstance(cid, int) or cid not in expected:
            raise Invalid(f"non_study chunk id {cid!r} is not one of this batch's chunks")
        if cid in seen:
            raise Invalid(f"chunk {cid} is both in a topic and in non_study")
        seen[cid] = "non_study"
        non_study[cid] = str(n.get("reason") or "not study material")[:200]
    missing = sorted(set(expected) - set(seen))
    if missing:
        raise Invalid(f"these chunk ids are missing: {missing}; put each in one topic or in non_study")
    return MapResult(topics, non_study)


@dataclass
class FinalTopic:
    title: str
    summary: str
    chapter: str | None
    local_ids: list[str]


def validate_reduce(obj, local: list[LocalTopic], chapters: list[str], cap: int, language: str) -> list[FinalTopic]:
    if not isinstance(obj, dict) or not isinstance(obj.get("topics"), list) or not obj["topics"]:
        raise Invalid('expected {"topics": [...]} with at least one topic')
    if len(obj["topics"]) > cap:
        raise Invalid(f"{len(obj['topics'])} topics is over the limit of {cap}; merge the closest topics")
    known = {t.id for t in local}
    used: dict[str, str] = {}
    out, titles = [], set()
    for t in obj["topics"]:
        if not isinstance(t, dict):
            raise Invalid("each topic must be an object")
        title = _check_title(t.get("title"), language)
        if title in titles:
            raise Invalid(f"duplicate title {title!r}")
        titles.add(title)
        chapter = (t.get("chapter") or "").strip() or None
        if chapter and chapters and chapter not in chapters:
            raise Invalid(f"chapter {chapter!r} is not one of the listed chapters")
        ids = t.get("local_topic_ids")
        if not isinstance(ids, list) or not ids:
            raise Invalid(f"topic {title!r} has no local_topic_ids")
        for lid in ids:
            if lid not in known:
                raise Invalid(f"unknown local topic id {lid!r}")
            if lid in used:
                raise Invalid(f"local topic {lid} is used twice ({used[lid]!r} and {title!r})")
            used[lid] = title
        out.append(FinalTopic(title, _check_summary(t.get("summary"), language), chapter, list(ids)))
    missing = sorted(known - set(used))
    if missing:
        raise Invalid(f"these local topic ids are missing: {missing}")
    return out


@dataclass
class Placement:
    local_id: str
    existing_id: int | None
    new_title: str | None
    new_summary: str


def validate_assign(obj, local: list[LocalTopic], existing: dict[int, str], language: str) -> list[Placement]:
    if not isinstance(obj, dict) or not isinstance(obj.get("placements"), list):
        raise Invalid('expected {"placements": [...]}')
    known = {t.id for t in local}
    used, out = set(), []
    for p in obj["placements"]:
        lid = p.get("local_topic_id") if isinstance(p, dict) else None
        if lid not in known:
            raise Invalid(f"unknown local topic id {lid!r}")
        if lid in used:
            raise Invalid(f"local topic {lid} placed twice")
        used.add(lid)
        eid = p.get("existing_topic_id")
        if eid is not None:
            if eid not in existing:
                raise Invalid(f"existing topic id {eid!r} doesn't exist")
            out.append(Placement(lid, eid, None, ""))
        else:
            out.append(Placement(lid, None, _check_title(p.get("new_title"), language),
                                 _check_summary(p.get("new_summary"), language)))
    missing = sorted(known - used)
    if missing:
        raise Invalid(f"these local topic ids are missing: {missing}")
    return out


# --- Cap and estimate -----------------------------------------------------------

def topic_cap(chunks: list[sqlite3.Row], hints: list[Hint]) -> int:
    in_scope = {chapter_of(c, hints) for c in chunks}
    chapters = {c for c in in_scope if c}
    if chapters:
        outside = OUTSIDE_CHAPTERS if None in in_scope else 0
        return min(MAX_TOPICS, TOPICS_PER_CHAPTER * len(chapters) + outside)
    per_doc = Counter(c["document_id"] for c in chunks)
    return min(MAX_TOPICS, max(1, math.ceil(max(per_doc.values(), default=0) / CHUNKS_PER_TOPIC)))


def _chunk_xml(c: sqlite3.Row) -> str:
    return f'<chunk id="{c["id"]}" pages="{c["page"]}-{c["page_end"]}">\n{c["text"]}\n</chunk>'


def _map_messages(batch: Batch, language: str, previous_title: str) -> list[dict]:
    system = _prompt("topics_map", language=_language_name(language), max_topics=MAP_MAX_TOPICS,
                     chapters=", ".join(batch.chapters) or "(none detected)",
                     previous_title=previous_title or "(none; this is the first batch)")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(_chunk_xml(c) for c in batch.chunks)}]


@dataclass
class Estimate:
    model: str
    chunks: int
    batches: int
    calls: int
    input_tokens: int
    output_tokens: int
    input_price: float | None    # USD per token
    output_price: float | None
    thinking_per_call: int

    def cost(self, retry_share: float = 0.0, thinking: bool = False) -> float | None:
        if self.input_price is None or self.output_price is None:
            return None
        out = self.output_tokens + (self.thinking_per_call * self.calls if thinking else 0)
        return (1 + retry_share) * (self.input_tokens * self.input_price + out * self.output_price)


THINKING_ALLOWANCE = 3000   # output tokens per call a reasoning model may spend thinking (rough)


def estimate(model: str, batches: list[Batch], language: str, extra_calls: int = 1,
             token_counter: Callable[[str], int] | None = None) -> Estimate:
    """Token and cost estimate with no model call. Tokens are counted locally with
    LiteLLM's tokenizer for the model (an approximation for Gemini); output is
    estimated from the batch sizes."""
    import litellm

    if token_counter is None:
        def token_counter(text: str) -> int:
            return litellm.token_counter(model=model, text=text)

    inp = out = 0
    local_topics = 0
    for b in batches:
        msgs = _map_messages(b, language, "")
        inp += sum(token_counter(m["content"]) for m in msgs)
        topics = min(MAP_MAX_TOPICS, max(1, len(b.chunks) // 4))
        local_topics += topics
        out += 30 + topics * 70 + len(b.chunks) * 4
    # Reduce (and assign's placement call): local topic lines in, final topics out.
    for _ in range(extra_calls):
        inp += token_counter(_prompt("topics_reduce", language="", chapter_rule="", max_topics=0, chapters="")) \
            + local_topics * 80
        out += max(1, local_topics // 2) * 70 + local_topics * 6
    info = litellm.model_cost.get(model) or litellm.model_cost.get(model.split("/", 1)[-1]) or {}
    thinking = THINKING_ALLOWANCE if info.get("supports_reasoning") else 0
    return Estimate(model, sum(len(b.chunks) for b in batches), len(batches), len(batches) + extra_calls,
                    inp, out, info.get("input_cost_per_token"), info.get("output_cost_per_token"), thinking)


def describe(e: Estimate) -> str:
    def usd(x):
        return "unknown (model not in LiteLLM's price table)" if x is None else f"${x:.2f}"
    lines = [
        f"Model: {e.model}",
        f"Chunks: {e.chunks} in {e.batches} batches; model calls: {e.calls} (+ retries if a reply is invalid)",
        f"Estimated tokens: ~{e.input_tokens:,} in, ~{e.output_tokens:,} out",
    ]
    if e.input_price is not None:
        lines.append(f"Prices (LiteLLM's table; check your provider's page): "
                     f"${e.input_price * 1e6:.3f}/M in, ${e.output_price * 1e6:.3f}/M out")
    lines.append(f"Estimated cost: {usd(e.cost())} without retries; up to {usd(e.cost(0.2))} with 20% retries")
    if e.thinking_per_call:
        lines.append(f"Reasoning model: thinking tokens are billed as output. Allowing ~{e.thinking_per_call:,} "
                     f"per call: up to {usd(e.cost(0.2, thinking=True))}")
    return "\n".join(lines)


# --- Build, assign ------------------------------------------------------------------

Logger = Callable[[dict], None]


def _run_map(batches, language, model, log, completion, extra) -> tuple[list[LocalTopic], dict[int, str]]:
    local: list[LocalTopic] = []
    non_study: dict[int, str] = {}
    previous = ""
    for b in batches:
        print(f"  map batch {b.index}/{len(batches)}: {len(b.chunks)} chunks", flush=True)
        result, _ = complete_json(
            _map_messages(b, language, previous), model=model, purpose="topics.map", log=log,
            validate=lambda obj, b=b: validate_map(obj, b, language), completion=completion, extra=extra)
        local += result.topics
        non_study.update(result.non_study)
        previous = result.topics[-1].title
    return local, non_study


def _local_lines(local: list[LocalTopic]) -> str:
    return "\n".join(json.dumps({"id": t.id, "title": t.title, "summary": t.summary, "document": t.document,
                                 "pages": t.pages, "chunks": len(t.chunk_ids)}, ensure_ascii=False)
                     for t in local)


def build(conn: sqlite3.Connection, set_name: str, model: str, chunks: list[sqlite3.Row], *,
          force: bool = False, completion: Callable | None = None, extra: dict | None = None,
          batch_chars: int = BATCH_CHARS) -> list[int]:
    """Build (or with force, rebuild) the topic map from the given chunks. Returns topic ids."""
    study_set = _get_set(conn, set_name)
    set_id, language = study_set["id"], study_set["language"]
    if db.topics(conn, set_id) and not force:
        raise RuntimeError("This set already has a topic map; use --force to replace it (edits are lost).")
    hints = set_hints(conn, set_id)
    batches = make_batches(chunks, hints, batch_chars)
    if not batches:
        raise RuntimeError("No chunks selected.")
    cap = topic_cap(chunks, hints)

    def log(record):
        db.log_llm_call(conn, set_id, record)

    local, non_study = _run_map(batches, language, model, log, completion, extra)
    chapters = list(dict.fromkeys(h.title for h in hints if any(chapter_of(c, [h]) for c in chunks)))
    system = _prompt(
        "topics_reduce", language=_language_name(language), max_topics=cap,
        chapters="\n".join(f"- {c}" for c in chapters) or "(none detected)",
        chapter_rule=("Set each topic's \"chapter\" to the listed chapter it belongs to, or \"\" if none fits."
                      if chapters else "Set \"chapter\" to \"\"."))
    print(f"  reduce: {len(local)} local topics -> at most {cap}", flush=True)
    final, _ = complete_json(
        [{"role": "system", "content": system}, {"role": "user", "content": _local_lines(local)}],
        model=model, purpose="topics.reduce", log=log, completion=completion, extra=extra,
        validate=lambda obj: validate_reduce(obj, local, chapters, cap, language))

    by_id = {t.id: t for t in local}
    ids = []
    with conn:
        if force:
            db.clear_topic_map(conn, set_id)
        for cid, reason in non_study.items():
            db.set_excluded(conn, cid, reason)
        for pos, t in enumerate(final):
            tid = db.add_topic(conn, set_id, t.title, t.summary, t.chapter, pos)
            db.link_chunks(conn, tid, [c for lid in t.local_ids for c in by_id[lid].chunk_ids])
            ids.append(tid)
    return ids


def assign(conn: sqlite3.Connection, set_name: str, model: str, *, completion: Callable | None = None,
           extra: dict | None = None, batch_chars: int = BATCH_CHARS) -> tuple[int, int]:
    """Place unassigned study chunks into existing topics or new ones.
    Returns (chunks placed, new topics)."""
    study_set = _get_set(conn, set_name)
    set_id, language = study_set["id"], study_set["language"]
    existing = {t["id"]: t["title"] for t in db.topics(conn, set_id)}
    if not existing:
        raise RuntimeError("No topic map yet; run build first.")
    chunks = db.unassigned_chunks(conn, set_id)
    if not chunks:
        return 0, 0
    hints = set_hints(conn, set_id)

    def log(record):
        db.log_llm_call(conn, set_id, record)

    local, non_study = _run_map(make_batches(chunks, hints, batch_chars), language, model, log, completion, extra)
    existing_lines = "\n".join(json.dumps({"id": t["id"], "title": t["title"], "summary": t["summary"]},
                                          ensure_ascii=False) for t in db.topics(conn, set_id))
    placements, _ = complete_json(
        [{"role": "system", "content": _prompt("topics_assign", language=_language_name(language))},
         {"role": "user", "content": f"Existing topics:\n{existing_lines}\n\nNew local topics:\n{_local_lines(local)}"}],
        model=model, purpose="topics.assign", log=log, completion=completion, extra=extra,
        validate=lambda obj: validate_assign(obj, local, existing, language))

    by_id = {t.id: t for t in local}
    new_topics: dict[str, int] = {}
    placed = 0
    with conn:
        for cid, reason in non_study.items():
            db.set_excluded(conn, cid, reason)
        position = max((t["position"] for t in db.topics(conn, set_id)), default=-1) + 1
        for p in placements:
            tid = p.existing_id
            if tid is None:
                if p.new_title not in new_topics:
                    new_topics[p.new_title] = db.add_topic(conn, set_id, p.new_title, p.new_summary, None, position)
                    position += 1
                tid = new_topics[p.new_title]
            db.link_chunks(conn, tid, by_id[p.local_id].chunk_ids)
            placed += len(by_id[p.local_id].chunk_ids)
    return placed, len(new_topics)


# --- Editing ---------------------------------------------------------------------

def rename(conn, topic_id: int, title: str, summary: str | None = None) -> None:
    fields = {"title": title.strip()}
    if summary is not None:
        fields["summary"] = summary.strip()
    with conn:
        db.update_topic(conn, topic_id, **fields)


def merge(conn, keep_id: int, other_ids: list[int]) -> None:
    keep = db.get_topic(conn, keep_id)
    if keep is None:
        raise ValueError(f"No topic {keep_id}")
    with conn:
        for oid in other_ids:
            other = db.get_topic(conn, oid)
            if other is None or oid == keep_id:
                raise ValueError(f"Can't merge topic {oid}")
            if other["study_set_id"] != keep["study_set_id"]:
                raise ValueError(f"Topic {oid} is in a different study set")
            db.move_links(conn, oid, keep_id)
            db.delete_topic(conn, oid)
        db.update_topic(conn, keep_id)


def delete(conn, topic_id: int) -> int:
    """Delete a topic; its chunks become unassigned. Returns how many."""
    n = len(db.topic_chunk_rows(conn, topic_id))
    with conn:
        db.delete_topic(conn, topic_id)
    return n


# --- CLI ---------------------------------------------------------------------------

def _get_set(conn, name: str) -> sqlite3.Row:
    s = db.get_study_set(conn, name)
    if s is None:
        raise SystemExit(f"No study set named {name!r}")
    return s


def _confirm(e: Estimate, yes: bool) -> None:
    print(describe(e))
    if yes:
        return
    if not sys.stdin.isatty():
        raise SystemExit("Not confirmed: re-run with --yes to make these model calls.")
    if input("Make these model calls? [y/N] ").strip().lower() != "y":
        raise SystemExit("Cancelled.")


def _model(args) -> str:
    model = args.model or os.environ.get("MUKHTABAR_MODEL")
    if not model:
        raise SystemExit("Choose a model with --model or MUKHTABAR_MODEL (e.g. gemini/<model-name>).")
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mukhtabar.topics")
    parser.add_argument("--db", default=db.DEFAULT_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    def with_set(p):
        p.add_argument("--set", required=True)
        return p

    def with_model(p):
        p.add_argument("--model")
        p.add_argument("--chapter", type=int, help="only this chapter (number from `hints`)")
        p.add_argument("--only", action="append", metavar="FILE:FIRST-LAST", help="only chunks starting on these pages")
        p.add_argument("--batch-chars", type=int, default=BATCH_CHARS)
        p.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                       help="passed to the model; lower means fewer billed thinking tokens")
        p.add_argument("--yes", action="store_true", help="don't ask before making model calls")
        return p

    with_set(sub.add_parser("hints", help="list chapter hints (no model call)"))
    with_model(with_set(sub.add_parser("estimate", help="estimate calls, tokens and cost (no model call)")))
    b = with_model(with_set(sub.add_parser("build", help="build the topic map")))
    b.add_argument("--force", action="store_true", help="replace an existing topic map")
    a = with_set(sub.add_parser("assign", help="place unassigned chunks into topics"))
    a.add_argument("--model")
    a.add_argument("--batch-chars", type=int, default=BATCH_CHARS)
    a.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    a.add_argument("--yes", action="store_true")
    ls = with_set(sub.add_parser("list"))
    ls.add_argument("--unassigned", action="store_true")
    ls.add_argument("--empty", action="store_true")
    sh = sub.add_parser("show")
    sh.add_argument("topic_id", type=int)
    rn = sub.add_parser("rename")
    rn.add_argument("topic_id", type=int)
    rn.add_argument("title")
    rn.add_argument("--summary")
    mg = sub.add_parser("merge")
    mg.add_argument("keep_id", type=int)
    mg.add_argument("other_ids", type=int, nargs="+")
    dl = sub.add_parser("delete")
    dl.add_argument("topic_id", type=int)
    with_set(sub.add_parser("calls", help="model usage log"))

    args = parser.parse_args(argv)
    conn = db.connect(args.db)

    if args.command == "hints":
        s = _get_set(conn, args.set)
        for i, h in enumerate(set_hints(conn, s["id"]), start=1):
            name = conn.execute("SELECT filename FROM documents WHERE id = ?", (h.document_id,)).fetchone()[0]
            n = sum(1 for c in db.study_chunks(conn, s["id"]) if chapter_of(c, [h]))
            print(f"{i:3}. {h.title}  ({name}, pages {h.first_page}-{h.last_page}, {n} chunks)")
    elif args.command in ("estimate", "build"):
        s = _get_set(conn, args.set)
        model = _model(args)
        chunks = select_chunks(conn, s["id"], parse_only(args.only) if args.only else None, args.chapter)
        batches = make_batches(chunks, set_hints(conn, s["id"]), args.batch_chars)
        e = estimate(model, batches, s["language"])
        if args.command == "estimate":
            print(describe(e))
            return
        if db.topics(conn, s["id"]) and not args.force:
            raise SystemExit("This set already has a topic map; use --force to replace it.")
        if args.force:
            edited = sum(t["edited"] for t in db.topics(conn, s["id"]))
            if edited:
                print(f"Warning: --force replaces the map, losing {edited} edited topics.")
        _confirm(e, args.yes)
        extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else None
        try:
            ids = build(conn, args.set, model, chunks, force=args.force, extra=extra, batch_chars=args.batch_chars)
        except LLMError as e:  # message only: the chained error could echo a request URL
            raise SystemExit(f"Stopped, nothing saved: {e}") from None
        print(f"Built {len(ids)} topics.")
    elif args.command == "assign":
        s = _get_set(conn, args.set)
        model = _model(args)
        chunks = db.unassigned_chunks(conn, s["id"])
        if not chunks:
            print("No unassigned chunks.")
            return
        e = estimate(model, make_batches(chunks, set_hints(conn, s["id"]), args.batch_chars), s["language"])
        _confirm(e, args.yes)
        extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else None
        try:
            placed, new = assign(conn, args.set, model, extra=extra, batch_chars=args.batch_chars)
        except LLMError as e:
            raise SystemExit(f"Stopped, nothing saved: {e}") from None
        print(f"Placed {placed} chunks; {new} new topics.")
    elif args.command == "list":
        s = _get_set(conn, args.set)
        if args.unassigned:
            for c in db.unassigned_chunks(conn, s["id"]):
                print(f"chunk {c['id']}  {c['filename']} p{c['page']}-{c['page_end']}  {c['text'][:70]!r}")
            return
        for t in db.topics(conn, s["id"]):
            if args.empty and t["n_chunks"]:
                continue
            mark = " (edited)" if t["edited"] else ""
            print(f"{t['id']:5}  {t['title']}  [{t['n_chunks']} chunks]{mark}  {t['chapter'] or ''}")
    elif args.command == "show":
        t = db.get_topic(conn, args.topic_id)
        if t is None:
            raise SystemExit(f"No topic {args.topic_id}")
        print(f"{t['title']}\n{t['summary']}\nchapter: {t['chapter'] or '-'}")
        for c in db.topic_chunk_rows(conn, t["id"]):
            print(f"  chunk {c['id']}  {c['filename']} p{c['page']}-{c['page_end']}")
    elif args.command == "rename":
        rename(conn, args.topic_id, args.title, args.summary)
    elif args.command == "merge":
        merge(conn, args.keep_id, args.other_ids)
    elif args.command == "delete":
        print(f"Deleted; {delete(conn, args.topic_id)} chunks are now unassigned.")
    elif args.command == "calls":
        s = _get_set(conn, args.set)
        rows = conn.execute("SELECT * FROM llm_calls WHERE study_set_id = ? ORDER BY id", (s["id"],)).fetchall()
        for r in rows:
            cost = "?" if r["cost_usd"] is None else f"${r['cost_usd']:.4f}"
            print(f"{r['created_at']}  {r['purpose']:14} {r['model']}  in {r['input_tokens']} out {r['output_tokens']}"
                  f"  {cost}  attempts {r['attempts']}  {'ok' if r['ok'] else 'FAILED: ' + (r['error'] or '')}")
        total = sum(r["cost_usd"] or 0 for r in rows)
        print(f"{len(rows)} calls, ${total:.4f} total (where LiteLLM could price them)")


if __name__ == "__main__":
    main()

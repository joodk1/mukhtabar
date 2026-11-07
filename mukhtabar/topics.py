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
    python -m mukhtabar.topics include  CHUNK_ID [CHUNK_ID ...]
    python -m mukhtabar.topics exclude  CHUNK_ID [CHUNK_ID ...] [--reason "..."]
    python -m mukhtabar.topics clear-cache --set NAME

Model calls happen only in build and assign, and only after the estimate is
shown and confirmed (interactively, or with --yes).
"""

import argparse
import hashlib
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
FLOOR_CHUNKS_PER_TOPIC = 18 # floor per chapter: at least one topic per this many chunks
MAX_TOPICS = 60
TITLE_WORDS = 12            # the prompt asks for 10; a little slack avoids needless retries
MIN_HINT_PAGES = 3          # a running title must open this many pages...
MIN_HINT_RUN = 3            # ...including this many in a row
HINT_MAX_GAP = 2            # "in a row" allows the title on every other page
OPENER_PAGES = 2            # pages just before a chapter's first running title belong to it
MIN_LANGUAGE_SHARE = 0.5    # share of letters in the set's script, for titles and summaries

ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓ]")
LATIN_LETTER = re.compile(r"[A-Za-z]")

# Study material that must never be excluded, recognised from the text itself:
# review questions point to where the answer is ("مرجع الإجابة"), MCQs have
# option lines, and glossary entries pair a term with its English name.
ANSWER_REF = re.compile(r"مرجع الإجابة|answer reference", re.I)
OPTION_LINE = re.compile(r"^(?:[أبجد]|[A-D])[ .)\-]\s*\S", re.M)
GLOSSARY_LINE = re.compile(r"^[ء-ي][ء-ي\s/()\-]{2,50}\s[A-Z][A-Za-z'’\- ]{2,60}(?:\s[A-Z]{2,6})?$", re.M)


def protected_kind(text: str) -> str | None:
    """"questions" or "glossary" if the chunk holds material that's always study
    material, else None."""
    if ANSWER_REF.search(text) or len(OPTION_LINE.findall(text)) >= 2:
        return "questions"
    if len(GLOSSARY_LINE.findall(text)) >= 2:
        return "glossary"
    return None


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
    it. The model's earlier exclusions are reconsidered (a build decides afresh);
    chunks you excluded yourself are left out."""
    rows = [c for c in db.study_chunks(conn, study_set_id) if c["user_choice"] != "exclude"]
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
    chapter: str | None = None
    also_chunk_ids: list[int] = field(default_factory=list)   # secondary links


@dataclass
class MapResult:
    topics: list[LocalTopic]
    non_study: dict[int, str]
    duplicates: int = 0   # chunks listed in a second topic, kept there as secondary links


def validate_map(obj, batch: Batch, language: str) -> MapResult:
    if not isinstance(obj, dict) or not isinstance(obj.get("topics"), list):
        raise Invalid('expected {"topics": [...], "non_study": [...]}')
    expected = {c["id"]: c for c in batch.chunks}
    seen: dict[int, str] = {}
    topics, titles = [], set()
    duplicates = 0
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
        primary, also = [], []
        for cid in ids:
            if not isinstance(cid, int) or cid not in expected:
                raise Invalid(f"chunk id {cid!r} is not one of this batch's chunks")
            if cid in seen:
                # Listed in an earlier topic too: that one stays primary, this one
                # becomes a secondary link. Repeats within one topic are ignored.
                if seen[cid] != title and cid not in also:
                    also.append(cid)
                    duplicates += 1
                continue
            seen[cid] = title
            primary.append(cid)
        rows = [expected[c] for c in ids]
        topics.append(LocalTopic(f"b{batch.index}-t{i}", title, _check_summary(t.get("summary"), language),
                                 primary, rows[0]["filename"],
                                 f"{min(r['page'] for r in rows)}-{max(r['page_end'] for r in rows)}",
                                 also_chunk_ids=also))
    non_study: dict[int, str] = {}
    for n in obj.get("non_study") or []:
        cid = n.get("chunk_id") if isinstance(n, dict) else None
        if not isinstance(cid, int) or cid not in expected:
            raise Invalid(f"non_study chunk id {cid!r} is not one of this batch's chunks")
        if cid in seen:
            raise Invalid(f"chunk {cid} is both in a topic and in non_study")
        if expected[cid]["user_choice"] == "include":
            raise Invalid(f"chunk {cid} is marked as study material by the student; put it in a topic")
        kind = protected_kind(expected[cid]["text"])
        if kind:
            what = "review or exam questions" if kind == "questions" else "glossary entries"
            raise Invalid(f"chunk {cid} contains {what}, which are study material; put it in a topic, "
                          "not in non_study")
        seen[cid] = "non_study"
        non_study[cid] = str(n.get("reason") or "not study material")[:200]
    missing = sorted(set(expected) - set(seen))
    if missing:
        raise Invalid(f"these chunk ids are missing: {missing}; put each in one topic or in non_study")
    return MapResult(topics, non_study, duplicates)


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
    saved: int = 0               # batches reused from an interrupted run

    def cost(self, retry_share: float = 0.0, thinking: bool = False) -> float | None:
        if self.input_price is None or self.output_price is None:
            return None
        out = self.output_tokens + (self.thinking_per_call * self.calls if thinking else 0)
        return (1 + retry_share) * (self.input_tokens * self.input_price + out * self.output_price)


THINKING_ALLOWANCE = 3000   # output tokens per call a reasoning model may spend thinking (rough)


PLACE_TOPIC_LIST_TOKENS = 3000  # the topic list sent with each placement call (rough)


def estimate(model: str, batches: list[Batch], language: str, extra_calls: int = 1,
             token_counter: Callable[[str], int] | None = None, saved: int = 0,
             place_chunks: list | None = None) -> Estimate:
    """Token and cost estimate with no model call. Tokens are counted locally with
    LiteLLM's tokenizer for the model (an approximation for Gemini); output is
    estimated from the batch sizes. The first `saved` batches already have a saved
    reply (an interrupted run) and cost nothing. place_chunks are practice and
    glossary chunks placed into topics after the merge."""
    import litellm

    if token_counter is None:
        def token_counter(text: str) -> int:
            return litellm.token_counter(model=model, text=text)

    inp = out = 0
    local_topics = 0
    for b in batches:
        topics = min(MAP_MAX_TOPICS, max(1, len(b.chunks) // 4))
        local_topics += topics
        if b.index <= saved:
            continue
        msgs = _map_messages(b, language, "")
        inp += sum(token_counter(m["content"]) for m in msgs)
        out += 30 + topics * 70 + len(b.chunks) * 4
    # Calls after the map (routing, per-chapter merges, or assign's placement): each
    # local topic line is sent about twice in total, final topics come back.
    if extra_calls:
        inp += extra_calls * token_counter(_prompt("topics_reduce", language="", scope="", count_rule="")) \
            + local_topics * 80 * 2
        out += max(1, local_topics // 3) * 70 + local_topics * 12
    place_chunks = place_chunks or []
    place_calls = math.ceil(len(place_chunks) / PLACE_BATCH_CHUNKS)
    if place_calls:
        inp += sum(token_counter(_chunk_xml(c)) for c in place_chunks) \
            + place_calls * (PLACE_TOPIC_LIST_TOKENS + token_counter(_prompt("topics_place", topics="")))
        out += len(place_chunks) * 25
    info = litellm.model_cost.get(model) or litellm.model_cost.get(model.split("/", 1)[-1]) or {}
    thinking = THINKING_ALLOWANCE if info.get("supports_reasoning") else 0
    return Estimate(model, sum(len(b.chunks) for b in batches) + len(place_chunks), len(batches),
                    len(batches) - saved + extra_calls + place_calls,
                    inp, out, info.get("input_cost_per_token"), info.get("output_cost_per_token"), thinking, saved)


def describe(e: Estimate) -> str:
    def usd(x):
        return "unknown (model not in LiteLLM's price table)" if x is None else f"${x:.2f}"
    lines = [
        f"Model: {e.model}",
        f"Chunks: {e.chunks} in {e.batches} batches; model calls: {e.calls} (+ retries if a reply is invalid)",
        f"Estimated tokens: ~{e.input_tokens:,} in, ~{e.output_tokens:,} out",
    ]
    if e.saved:
        lines.insert(2, f"Resuming: {e.saved} of {e.batches} batches already done in an earlier run "
                        "(no new calls for them)")
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


def _cache_key(model: str, messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps([model, messages], ensure_ascii=False).encode("utf-8")).hexdigest()


def _cached_map(conn, model, batch, language, previous) -> MapResult | None:
    reply = db.cache_get(conn, _cache_key(model, _map_messages(batch, language, previous)))
    return validate_map(json.loads(reply), batch, language) if reply else None


def cached_batches(conn, model: str, batches: list[Batch], language: str) -> int:
    """How many leading batches already have a saved reply (no model call)."""
    previous, n = "", 0
    for b in batches:
        result = _cached_map(conn, model, b, language, previous)
        if result is None:
            break
        previous, n = result.topics[-1].title, n + 1
    return n


def _run_map(conn, set_id, batches, language, model, log, completion, extra,
             hints: list[Hint]) -> tuple[list[LocalTopic], dict[int, str], int]:
    """Map every batch. A batch whose exact prompt was answered before (an
    interrupted or earlier run) reuses the saved reply instead of calling the model
    again. Each local topic gets the chapter most of its chunks are in. Also
    returns how many duplicate chunk listings became secondary links."""
    local: list[LocalTopic] = []
    non_study: dict[int, str] = {}
    duplicates = 0
    previous = ""
    for b in batches:
        messages = _map_messages(b, language, previous)
        result = _cached_map(conn, model, b, language, previous)
        if result is not None:
            print(f"  map batch {b.index}/{len(batches)}: reused the saved reply", flush=True)
        else:
            print(f"  map batch {b.index}/{len(batches)}: {len(b.chunks)} chunks", flush=True)
            raw: list = []

            def validate(obj, b=b, raw=raw):
                result = validate_map(obj, b, language)
                raw.append(obj)
                return result
            result, _ = complete_json(messages, model=model, purpose="topics.map", log=log,
                                      validate=validate, completion=completion, extra=extra)
            _check(log)
            db.cache_put(conn, _cache_key(model, messages), set_id, "topics.map",
                         json.dumps(raw[-1], ensure_ascii=False))
        if result.duplicates:
            print(f"    {result.duplicates} chunk(s) listed in two topics: kept as secondary links", flush=True)
        rows = {c["id"]: c for c in b.chunks}
        for t in result.topics:
            chapters = Counter(chapter_of(rows[c], hints) for c in (t.chunk_ids or t.also_chunk_ids))
            t.chapter = chapters.most_common(1)[0][0]
        local += result.topics
        non_study.update(result.non_study)
        duplicates += result.duplicates
        previous = result.topics[-1].title
    return local, non_study, duplicates


def _local_lines(local: list[LocalTopic]) -> str:
    return "\n".join(json.dumps({"id": t.id, "title": t.title, "summary": t.summary, "document": t.document,
                                 "pages": t.pages, "chunks": len(t.chunk_ids)}, ensure_ascii=False)
                     for t in local)


@dataclass
class Group:
    """Local topics merged together: one chapter, or material outside any chapter."""
    chapter: str | None
    local: list[LocalTopic]
    floor: int = 1
    cap: int = 1

    @property
    def n_chunks(self) -> int:
        return sum(len(t.chunk_ids) for t in self.local)


def _premerge(local: list[LocalTopic]) -> list[LocalTopic]:
    """Merge local topics with the same title (a topic continued across batches)."""
    by_title: dict[str, LocalTopic] = {}
    out = []
    for t in local:
        if t.title in by_title:
            by_title[t.title].chunk_ids += t.chunk_ids
            by_title[t.title].also_chunk_ids += t.also_chunk_ids
        else:
            by_title[t.title] = LocalTopic(t.id, t.title, t.summary, list(t.chunk_ids), t.document, t.pages,
                                           t.chapter, list(t.also_chunk_ids))
            out.append(by_title[t.title])
    return out


def plan_groups(local: list[LocalTopic], chapters: list[str], has_hints: bool) -> list[Group]:
    """One group per chapter (in chapter order), then material outside any chapter.
    Each gets a floor of one topic per FLOOR_CHUNKS_PER_TOPIC chunks (never more
    than its local topics) and a cap of TOPICS_PER_CHAPTER (raised to the floor if
    needed). Caps are trimmed, never below a floor, so the total stays within
    MAX_TOPICS."""
    groups = [Group(c, [t for t in local if t.chapter == c]) for c in chapters]
    groups.append(Group(None, [t for t in local if t.chapter not in chapters]))
    groups = [g for g in groups if g.local]
    for g in groups:
        g.local = _premerge(g.local)
        g.floor = min(len(g.local), max(1, math.ceil(g.n_chunks / FLOOR_CHUNKS_PER_TOPIC)))
        if g.chapter is not None:
            base = TOPICS_PER_CHAPTER
        elif has_hints:
            base = OUTSIDE_CHAPTERS
        else:
            base = min(MAX_TOPICS, max(1, math.ceil(g.n_chunks / CHUNKS_PER_TOPIC)))
        g.cap = min(len(g.local), max(base, g.floor))
    while sum(g.cap for g in groups) > MAX_TOPICS:
        slack = [g for g in groups if g.cap > g.floor]
        if not slack:
            break
        max(slack, key=lambda g: g.cap - g.floor).cap -= 1
    return groups


@dataclass
class FinalTopic:
    title: str
    summary: str
    chapter: str | None
    local_ids: list[str]


def validate_reduce(obj, local: list[LocalTopic], floor: int, cap: int, language: str) -> list[FinalTopic]:
    if not isinstance(obj, dict) or not isinstance(obj.get("topics"), list) or not obj["topics"]:
        raise Invalid('expected {"topics": [...]} with at least one topic')
    n = len(obj["topics"])
    if n > cap:
        raise Invalid(f"{n} topics is over the limit of {cap}; merge the closest topics")
    if n < floor:
        raise Invalid(f"{n} topics is too few for this much material; return at least {floor}, "
                      "splitting broad topics into their distinct ideas")
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
        ids = t.get("local_topic_ids")
        if not isinstance(ids, list) or not ids:
            raise Invalid(f"topic {title!r} has no local_topic_ids")
        for lid in ids:
            if lid not in known:
                raise Invalid(f"unknown local topic id {lid!r}")
            if lid in used:
                raise Invalid(f"local topic {lid} is used twice ({used[lid]!r} and {title!r})")
            used[lid] = title
        out.append(FinalTopic(title, _check_summary(t.get("summary"), language), None, list(ids)))
    missing = sorted(known - set(used))
    if missing:
        raise Invalid(f"these local topic ids are missing: {missing}")
    return out


def validate_route(obj, local: list[LocalTopic], chapters: list[str]) -> dict[str, str | None]:
    if not isinstance(obj, dict) or not isinstance(obj.get("routes"), list):
        raise Invalid('expected {"routes": [...]}')
    known = {t.id for t in local}
    out: dict[str, str | None] = {}
    for r in obj["routes"]:
        lid = r.get("local_topic_id") if isinstance(r, dict) else None
        if lid not in known:
            raise Invalid(f"unknown local topic id {lid!r}")
        if lid in out:
            raise Invalid(f"local topic {lid} routed twice")
        chapter = (r.get("chapter") or "").strip() or None
        if chapter is not None and chapter not in chapters:
            raise Invalid(f"chapter {chapter!r} is not one of the listed chapters; use one of them or \"\"")
        out[lid] = chapter
    missing = sorted(known - set(out))
    if missing:
        raise Invalid(f"these local topic ids are missing: {missing}")
    return out


def _route(local, chapters, language, model, log, completion, extra) -> None:
    """Give chapterless local topics (documents without running titles) a chapter."""
    loose = [t for t in local if t.chapter is None]
    if not loose or not chapters:
        return
    print(f"  route: {len(loose)} local topics from documents without chapters", flush=True)
    system = _prompt("topics_route", chapters="\n".join(f"- {c}" for c in chapters))
    routes, _ = complete_json(
        [{"role": "system", "content": system}, {"role": "user", "content": _local_lines(loose)}],
        model=model, purpose="topics.route", log=log, completion=completion, extra=extra,
        validate=lambda obj: validate_route(obj, loose, chapters))
    for t in loose:
        t.chapter = routes[t.id]


def _reduce(group: Group, language, model, log, completion, extra) -> list[FinalTopic]:
    if len(group.local) <= group.floor:
        # As many local topics as the floor asks for: nothing to merge, no call.
        return [FinalTopic(t.title, t.summary, group.chapter, [t.id]) for t in group.local]
    scope = f'the chapter "{group.chapter}"' if group.chapter else "material outside the listed chapters"
    count = (f"Return between {group.floor} and {group.cap} topics." if group.floor < group.cap
             else f"Return exactly {group.cap} topics.")
    system = _prompt("topics_reduce", language=_language_name(language), scope=scope, count_rule=count)
    print(f"  reduce {group.chapter or '(outside chapters)'}: {len(group.local)} local topics, "
          f"{group.n_chunks} chunks -> {group.floor}-{group.cap}", flush=True)
    final, _ = complete_json(
        [{"role": "system", "content": system}, {"role": "user", "content": _local_lines(group.local)}],
        model=model, purpose="topics.reduce", log=log, completion=completion, extra=extra,
        validate=lambda obj: validate_reduce(obj, group.local, group.floor, group.cap, language))
    for t in final:
        t.chapter = group.chapter
    return final


PRACTICE_SHARE = 0.25       # a chapter this share of question chunks is a practice section
PLACE_BATCH_CHUNKS = 15     # chunks per placement call
MAX_ALSO = 3                # secondary topics per placed chunk
RETRY_BUDGET = 20           # stop a build once more than this many replies were invalid


class RetryBudgetExceeded(LLMError):
    pass


class Budget:
    """Logs every call, and stops the build once invalid replies pile up."""

    def __init__(self, conn, set_id: int, limit: int = RETRY_BUDGET):
        self.conn, self.set_id = conn, set_id
        self.limit = limit
        self.retries = 0

    def __call__(self, record: dict) -> None:
        db.log_llm_call(self.conn, self.set_id, record)
        self.retries += max(record["attempts"] - 1, 0)

    def check(self) -> None:
        if self.retries > self.limit:
            raise RetryBudgetExceeded(f"stopped after {self.retries} invalid replies (limit {self.limit}); "
                                      "see `calls` for what was rejected")


def _check(log) -> None:
    getattr(log, "check", lambda: None)()


def practice_chapters(chunks: list[sqlite3.Row], hints: list[Hint]) -> set[str]:
    """Chapters that are mostly practice questions (e.g. an MCQ appendix)."""
    out = set()
    for h in hints:
        cs = [c for c in chunks if chapter_of(c, [h])]
        if cs and sum(protected_kind(c["text"]) == "questions" for c in cs) >= PRACTICE_SHARE * len(cs):
            out.add(h.title)
    return out


def cross_cutting(chunks: list[sqlite3.Row], hints: list[Hint]) -> list[sqlite3.Row]:
    """Chunks placed into subject topics instead of forming topics of their own:
    those in a practice chapter, and every glossary chunk."""
    practice = practice_chapters(chunks, hints)
    return [c for c in chunks if chapter_of(c, hints) in practice or protected_kind(c["text"]) == "glossary"]


@dataclass
class ChunkPlacement:
    chunk_id: int
    topic: int            # index into the final topic list
    also: list[int]


def validate_place(obj, batch: list[sqlite3.Row], n_topics: int) -> tuple[list[ChunkPlacement], dict[int, str]]:
    if not isinstance(obj, dict) or not isinstance(obj.get("placements"), list):
        raise Invalid('expected {"placements": [...], "non_study": [...]}')
    expected = {c["id"]: c for c in batch}
    seen, out, non_study = set(), [], {}

    def topic_index(tid):
        if not isinstance(tid, int) or not 1 <= tid <= n_topics:
            raise Invalid(f"topic id {tid!r} is not in the list (1-{n_topics})")
        return tid - 1

    for p in obj["placements"]:
        cid = p.get("chunk_id") if isinstance(p, dict) else None
        if cid not in expected:
            raise Invalid(f"chunk id {cid!r} is not one of the given chunks")
        if cid in seen:
            raise Invalid(f"chunk {cid} placed twice")
        seen.add(cid)
        primary = topic_index(p.get("topic_id"))
        also = [topic_index(t) for t in (p.get("also") or [])]
        if primary in also:
            raise Invalid(f"chunk {cid}: \"also\" repeats its topic_id")
        if len(set(also)) != len(also) or len(also) > MAX_ALSO:
            raise Invalid(f"chunk {cid}: \"also\" must list at most {MAX_ALSO} different topics")
        out.append(ChunkPlacement(cid, primary, also))
    for n in obj.get("non_study") or []:
        cid = n.get("chunk_id") if isinstance(n, dict) else None
        if cid not in expected or cid in seen:
            raise Invalid(f"non_study chunk id {cid!r} is unknown or already placed")
        if expected[cid]["user_choice"] == "include" or protected_kind(expected[cid]["text"]):
            raise Invalid(f"chunk {cid} holds questions or glossary entries, which are study material; place it")
        seen.add(cid)
        non_study[cid] = str(n.get("reason") or "not study material")[:200]
    missing = sorted(set(expected) - seen)
    if missing:
        raise Invalid(f"these chunk ids are missing: {missing}")
    return out, non_study


def _place(chunks, final: list[FinalTopic], model, log, completion, extra):
    lines = "\n".join(json.dumps({"id": i, "title": t.title, "summary": t.summary, "chapter": t.chapter or ""},
                                 ensure_ascii=False) for i, t in enumerate(final, start=1))
    system = _prompt("topics_place", topics=lines)
    placements, non_study = [], {}
    batches = [chunks[i:i + PLACE_BATCH_CHUNKS] for i in range(0, len(chunks), PLACE_BATCH_CHUNKS)]
    for k, batch in enumerate(batches, start=1):
        print(f"  place {k}/{len(batches)}: {len(batch)} practice/glossary chunks", flush=True)
        (p, n), _ = complete_json(
            [{"role": "system", "content": system},
             {"role": "user", "content": "\n\n".join(_chunk_xml(c) for c in batch)}],
            model=model, purpose="topics.place", log=log, completion=completion, extra=extra,
            validate=lambda obj, batch=batch: validate_place(obj, batch, len(final)))
        _check(log)
        placements += p
        non_study.update(n)
    return placements, non_study


def build(conn: sqlite3.Connection, set_name: str, model: str, chunks: list[sqlite3.Row], *,
          force: bool = False, completion: Callable | None = None, extra: dict | None = None,
          batch_chars: int = BATCH_CHARS, retry_budget: int = RETRY_BUDGET) -> list[int]:
    """Build (or with force, rebuild) the topic map from the given chunks. Returns topic ids.

    Subject chunks are mapped and merged per chapter; practice-question and
    glossary chunks are then placed into those topics (with secondary links for
    other topics they cover). Map replies are saved and reused, so a rebuild with
    the same chunks and prompts reruns only routing, merging and placement (see
    `clear-cache`). Nothing is stored unless every step succeeds, and the build
    stops if invalid replies pile up."""
    study_set = _get_set(conn, set_name)
    set_id, language = study_set["id"], study_set["language"]
    if db.topics(conn, set_id) and not force:
        raise RuntimeError("This set already has a topic map; use --force to replace it (edits are lost).")
    hints = set_hints(conn, set_id)
    cross = cross_cutting(chunks, hints)
    cross_ids = {c["id"] for c in cross}
    subject = [c for c in chunks if c["id"] not in cross_ids]
    batches = make_batches(subject, hints, batch_chars)
    if not batches:
        raise RuntimeError("No chunks selected.")
    log = Budget(conn, set_id, retry_budget)

    local, non_study, duplicates = _run_map(conn, set_id, batches, language, model, log, completion, extra, hints)
    if duplicates:
        print(f"  duplicates converted to secondary links: {duplicates}", flush=True)
    practice = practice_chapters(chunks, hints)
    chapters = list(dict.fromkeys(h.title for h in hints
                                  if h.title not in practice and any(chapter_of(c, [h]) for c in subject)))
    _route(local, chapters, language, model, log, completion, extra)
    _check(log)
    groups = plan_groups(local, chapters, bool(hints))
    final_by_group = []
    for g in groups:
        final_by_group.append((g, _reduce(g, language, model, log, completion, extra)))
        _check(log)
    final = [t for _, ts in final_by_group for t in ts]
    placements, place_non_study = _place(cross, final, model, log, completion, extra) if cross else ([], {})

    ids = []
    with conn:
        if force:
            db.clear_topic_map(conn, set_id)
        for cid, reason in {**non_study, **place_non_study}.items():
            db.set_excluded(conn, cid, reason)
        for g, ts in final_by_group:
            by_id = {t.id: t for t in g.local}
            for t in ts:
                tid = db.add_topic(conn, set_id, t.title, t.summary, t.chapter, len(ids))
                primary = [c for lid in t.local_ids for c in by_id[lid].chunk_ids]
                db.link_chunks(conn, tid, primary)
                also = {c for lid in t.local_ids for c in by_id[lid].also_chunk_ids} - set(primary)
                db.link_chunks(conn, tid, sorted(also), primary=False)
                ids.append(tid)
        for p in placements:
            db.link_chunks(conn, ids[p.topic], [p.chunk_id])
            for a in p.also:
                db.link_chunks(conn, ids[a], [p.chunk_id], primary=False)
    return ids


def planned_calls(chunks: list[sqlite3.Row], hints: list[Hint]) -> int:
    """Calls after the map: one route call if some subject chunks have no chapter
    while others do, one merge per chapter (practice chapters excepted) and one for
    material outside chapters, then placement calls for practice/glossary chunks."""
    cross = {c["id"] for c in cross_cutting(chunks, hints)}
    practice = practice_chapters(chunks, hints)
    subject = [c for c in chunks if c["id"] not in cross]
    chapters = {chapter_of(c, hints) for c in subject}
    named = {c for c in chapters if c and c not in practice}
    route = 1 if named and None in chapters else 0
    return route + len(named) + (1 if None in chapters else 0) + math.ceil(len(cross) / PLACE_BATCH_CHUNKS)


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

    local, non_study, _ = _run_map(conn, set_id, make_batches(chunks, hints, batch_chars), language, model, log,
                                completion, extra, hints)
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
            db.link_chunks(conn, tid, by_id[p.local_id].also_chunk_ids, primary=False)
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
    try:
        answer = input("Make these model calls? [y/N] ")
    except EOFError:  # input closed (some terminals report a closed stdin as a tty)
        answer = ""
    if answer.strip().lower() != "y":
        raise SystemExit("Not confirmed: no model calls made (re-run with --yes to make them).")


def _stopped(e: Exception) -> str:
    return (f"Stopped: {e}\n"
            "Finished batches are saved; run the same command again to continue without repeating them.")


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
    b.add_argument("--fresh", action="store_true", help="ignore saved map replies and ask the model again")
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
    inc = sub.add_parser("include", help="mark chunks as study material (never excluded again)")
    inc.add_argument("chunk_ids", type=int, nargs="+")
    exc = sub.add_parser("exclude", help="exclude chunks from topics (kept across rebuilds)")
    exc.add_argument("chunk_ids", type=int, nargs="+")
    exc.add_argument("--reason")
    with_set(sub.add_parser("clear-cache", help="forget saved map replies, so the next build asks the model again"))

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
        hints = set_hints(conn, s["id"])
        # Mirror build(): practice/glossary chunks are placed, the rest is mapped.
        cross = cross_cutting(chunks, hints)
        cross_ids = {c["id"] for c in cross}
        batches = make_batches([c for c in chunks if c["id"] not in cross_ids], hints, args.batch_chars)
        if args.command == "build" and args.fresh:
            with conn:
                db.cache_clear(conn, s["id"])
        place_calls = math.ceil(len(cross) / PLACE_BATCH_CHUNKS)
        e = estimate(model, batches, s["language"], extra_calls=planned_calls(chunks, hints) - place_calls,
                     saved=cached_batches(conn, model, batches, s["language"]), place_chunks=cross)
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
            raise SystemExit(_stopped(e)) from None
        print(f"Built {len(ids)} topics.")
    elif args.command == "assign":
        s = _get_set(conn, args.set)
        model = _model(args)
        chunks = db.unassigned_chunks(conn, s["id"])
        if not chunks:
            print("No unassigned chunks.")
            return
        batches = make_batches(chunks, set_hints(conn, s["id"]), args.batch_chars)
        e = estimate(model, batches, s["language"], saved=cached_batches(conn, model, batches, s["language"]))
        _confirm(e, args.yes)
        extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else None
        try:
            placed, new = assign(conn, args.set, model, extra=extra, batch_chars=args.batch_chars)
        except LLMError as e:
            raise SystemExit(_stopped(e)) from None
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
    elif args.command in ("include", "exclude"):
        with conn:
            for cid in args.chunk_ids:
                c = db.get_chunk(conn, cid)
                if c is None:
                    raise SystemExit(f"No chunk {cid}")
                db.set_user_choice(conn, cid, args.command, getattr(args, "reason", None))
        n = len(args.chunk_ids)
        if args.command == "include":
            print(f"Included {n} chunks. They have no topic yet: run `assign` (or rebuild) to place them.")
        else:
            print(f"Excluded {n} chunks; their topic links were removed.")
    elif args.command == "clear-cache":
        s = _get_set(conn, args.set)
        with conn:
            db.cache_clear(conn, s["id"])
        print("Saved map replies cleared.")
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

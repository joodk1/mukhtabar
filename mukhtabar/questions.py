"""Question generation: grounded multiple-choice questions per topic.

    python -m mukhtabar.questions estimate --set NAME --model MODEL (--topic ID ... | --all) [--calc]
    python -m mukhtabar.questions generate --set NAME --model MODEL (--topic ID ... | --all) [--calc] [--yes]
    python -m mukhtabar.questions calc-topics --set NAME
    python -m mukhtabar.questions list     --topic ID

Each topic gets up to QUESTIONS_PER_TOPIC questions; a topic with fewer is topped
up, and the model is shown the existing ones so it doesn't repeat them. Each
question is based on one study chunk of its topic and carries an evidence quote
that must appear in that chunk. Practice questions (from a question bank or an
MCQ appendix) are shown to the model as style examples only; a generated question
that copies one is rejected, as is one about the exam itself (its questions,
weights, syllabus or format).

--calc adds up to CALC_PER_TOPIC calculation questions to topics whose chunks show
a worked calculation, up to MAX_PER_TOPIC questions in all. The model must give
the arithmetic as an expression; the code evaluates it and checks that it gives
the stated result, that the correct option shows that result and no other option
does, and that the numbers used come from the question or its source.

Model calls happen only after the estimate is shown and confirmed (interactively,
or with --yes).
"""

import argparse
import ast
import math
import operator
import re
import sqlite3
from typing import Callable

from mukhtabar import db, topics
from mukhtabar.ingest.extract import normalize
from mukhtabar.llm import Invalid, LLMError, QuotaExhausted, complete_json

QUESTIONS_PER_TOPIC = 6
CALC_PER_TOPIC = 2           # calculation questions added by --calc
MAX_PER_TOPIC = 8            # --calc never takes a topic above this
QUESTIONS_PER_CALL = 3       # short replies are less likely to be cut off
MAX_OUTPUT_TOKENS = 8192     # explicit and generous, so the provider's default never truncates
MAX_EVIDENCE_CHARS = 420     # "about 400 characters": a full Arabic textbook sentence fits
SOURCE_CHARS = 20_000        # study text sent per topic
MAX_PRACTICE_CHUNKS = 3      # practice-question chunks sent as style examples
MIN_EVIDENCE_CHARS = 20
COPY_SHARE = 0.6             # share of a question's 4-word runs found in practice questions = copied
DUPLICATE_SHARE = 0.8        # ...found in another question of the topic = duplicate
SHINGLE = 4
OUTPUT_TOKENS_PER_QUESTION = 220   # Arabic question + options + explanation + evidence (rough)

PUNCT = re.compile(r"[^\w\s]|_", re.U)

# Questions about the exam itself rather than the subject: how many questions it
# has, chapter weights, the syllabus or learning objectives, duration, pass mark.
LOGISTICS = re.compile(
    r"(?:عدد|كم)\s+(?:من\s+)?(?:ال)?أسئلة"
    r"|(?:أسئلة|سؤالا?)[^.؟?]{0,40}(?:الاختبار|الامتحان)"
    r"|(?:وزن|نسبة)\s+(?:هذا\s+)?(?:ال)?فصل"
    r"|خطة\s+المنهاج|المنهاج\s+الدراسي|أهداف?\s+التعلم"
    r"|(?:مدة|زمن|درجة\s+النجاح|علامة\s+النجاح)\s+(?:ال)?(?:اختبار|امتحان)"
    r"|\bexam(?:ination)?\s+(?:questions?|structure|format|duration|weight)|\bsyllabus\b|\blearning\s+objectives?\b",
    re.I)

# Worked calculations in a chunk: a number joined to another by ×, ÷, * or =, or a
# chunk marked as a worked example.
_NUM = r"\d[\d,.]*\s*[%٪]?"
CALC_LINE = re.compile(rf"{_NUM}\s*[×÷*]\s*{_NUM}|{_NUM}\s*=\s*[(]?\s*{_NUM}|{_NUM}\s*[(]?\s*=\s*{_NUM}")
WORKED_EXAMPLE = re.compile(r"مثال\s+محلول|worked\s+example", re.I)
MIN_CALC_LINES = 2


def canon(text: str) -> str:
    """For matching quotes: normalized letters (no harakat or tatweel), no
    punctuation, single spaces, lowercase Latin."""
    return re.sub(r"\s+", " ", PUNCT.sub(" ", normalize(text or ""))).strip().lower()


def shingles(text: str) -> set[tuple[str, ...]]:
    words = canon(text).split()
    return {tuple(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def overlap(a: set, b: set) -> float:
    return len(a & b) / len(a) if a else 0.0


def is_practice(chunk) -> bool:
    return topics.protected_kind(chunk["text"]) == "questions"


def has_worked_calculation(chunk) -> bool:
    text = chunk["text"]
    return bool(WORKED_EXAMPLE.search(text)) or sum(bool(CALC_LINE.search(l)) for l in text.split("\n")) >= MIN_CALC_LINES


def select_sources(rows: list[sqlite3.Row], calc: bool = False) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    """(study chunks within SOURCE_CHARS, spread evenly over the topic; up to
    MAX_PRACTICE_CHUNKS practice chunks as style examples). With calc, only the
    primary study chunks that show a worked calculation."""
    study = [r for r in rows if not is_practice(r)]
    if calc:
        study = [r for r in study if r["is_primary"] and has_worked_calculation(r)]
    practice = [r for r in rows if is_practice(r)][:MAX_PRACTICE_CHUNKS]
    if sum(len(r["text"]) for r in study) > SOURCE_CHARS:
        avg = sum(len(r["text"]) for r in study) / len(study)
        k = max(1, int(SOURCE_CHARS // avg))
        step = len(study) / k
        study = [study[int(i * step)] for i in range(k)]
    return study, practice


def _chunk_xml(c, kind: str) -> str:
    return f'<chunk id="{c["id"]}" pages="{c["page"]}-{c["page_end"]}" kind="{kind}">\n{c["text"]}\n</chunk>'


def messages(topic, study, practice, language: str, n: int, existing: list[str] | None = None,
             calc: bool = False) -> list[dict]:
    asked = "\n".join(f"- {p}" for p in existing or []) or "(none yet)"
    system = topics._prompt("questions_calculate" if calc else "questions_generate", topic=topic["title"],
                            summary=f"Topic summary: {topic['summary']}" if topic["summary"] else "",
                            n=n, existing=asked, language=topics._language_name(language))
    body = "\n\n".join([_chunk_xml(c, "study") for c in study] + [_chunk_xml(c, "practice") for c in practice])
    return [{"role": "system", "content": system}, {"role": "user", "content": body}]


# --- Arithmetic ---------------------------------------------------------------

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫", "01234567890123456789.")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
# Numbers a calculation may use without them appearing in the question: units and periods.
CONSTANTS = {0, 1, 2, 4, 10, 12, 100, 360, 365, 1000}


def _plain(text: str) -> str:
    """Western digits and operators, no thousands separators."""
    text = text.translate(ARABIC_DIGITS).replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)


def evaluate(expression: str) -> float:
    """Safely evaluate plain arithmetic (+ - * / ** and brackets on numbers)."""
    tree = ast.parse(_plain(expression), mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers, + - * / ** and brackets are allowed")
    return ev(tree)


def numbers_in(text: str) -> list[float]:
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", _plain(text or ""))]


def close(a: float, b: float) -> bool:
    """Equal after rounding to the precision an exam option would show."""
    return abs(a - b) <= max(0.006, 0.005 * abs(b))


def shows(option: str, result: float) -> bool:
    """Whether an option states the result (also as a percentage of a fraction)."""
    nums = numbers_in(option)
    return any(close(n, result) or (abs(result) < 1 and close(n, result * 100)) for n in nums)


def check_calculation(calc, i: int, prompt: str, options: list[str], answer: int, source: str) -> dict:
    if not isinstance(calc, dict) or not isinstance(calc.get("expression"), str):
        raise Invalid(f"question {i}: a calculation question needs \"calculation\": "
                      "{\"expression\": \"...\", \"result\": number}")
    try:
        value = evaluate(calc["expression"])
        stated = float(calc.get("result"))
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError) as e:
        raise Invalid(f"question {i}: calculation: the expression can't be evaluated ({e}); "
                      "use only numbers, + - * / and brackets") from None
    if not close(value, stated):
        raise Invalid(f"question {i}: calculation: the expression gives {value:g}, not the stated result {stated:g}")
    if not shows(options[answer], value):
        raise Invalid(f"question {i}: calculation: the correct option doesn't show the result {value:g}")
    if any(shows(o, value) for k, o in enumerate(options) if k != answer):
        raise Invalid(f"question {i}: calculation: a wrong option also shows the result {value:g}")
    known = numbers_in(prompt) + numbers_in(source)
    foreign = [n for n in numbers_in(calc["expression"])
               if n not in CONSTANTS and not any(close(n, k) for k in known)]
    if foreign:
        raise Invalid(f"question {i}: calculation: the expression uses numbers that aren't in the question "
                      f"or its source ({', '.join(f'{n:g}' for n in foreign)})")
    return {"expression": calc["expression"], "result": value}


# --- Validation -----------------------------------------------------------------

def validate_questions(obj, study: list, n: int, language: str, practice_shingles: set,
                       existing: list[str], calc: bool = False) -> list[dict]:
    if not isinstance(obj, dict) or not isinstance(obj.get("questions"), list):
        raise Invalid('expected {"questions": [...]}')
    qs = obj["questions"]
    if len(qs) != n:
        raise Invalid(f"write exactly {n} questions, not {len(qs)}")
    texts = {c["id"]: canon(c["text"]) for c in study}
    raw = {c["id"]: c["text"] for c in study}
    seen = [shingles(p) for p in existing]
    out = []
    for i, q in enumerate(qs, start=1):
        if not isinstance(q, dict) or q.get("type", "mcq") != "mcq":
            raise Invalid(f"question {i}: write multiple-choice questions only")
        prompt = (q.get("prompt") or "").strip()
        if not prompt:
            raise Invalid(f"question {i} has no prompt")
        topics._check_language(prompt, language, f"question {i}")
        cid = q.get("chunk_id")
        if isinstance(cid, str) and cid.strip().isdigit():
            cid = int(cid)   # an id written as text ("287") is the same id
        if cid not in texts:
            raise Invalid(f"question {i}: chunk_id {cid!r} is not one of the study chunks "
                          "(practice chunks are style examples only)")
        evidence = (q.get("evidence") or "").strip()
        if len(evidence) > MAX_EVIDENCE_CHARS:
            raise Invalid(f"question {i}: \"evidence\" must be one sentence of at most about 400 characters")
        quote = canon(evidence)
        corrected = False
        if len(quote) < MIN_EVIDENCE_CHARS:
            raise Invalid(f"question {i}: \"evidence\" must be copied word for word from chunk {cid} "
                          f"(you gave: \"{evidence[:MAX_EVIDENCE_CHARS]}\")")
        if quote not in texts[cid]:
            # Attributed to the wrong chunk? Accept it only if the quote is word for word in
            # exactly one of the study chunks sent (never a practice chunk), and use that one.
            matches = [k for k, t in texts.items() if quote in t]
            if len(matches) != 1:
                # The quote is included so the rejection log shows exactly what didn't match.
                raise Invalid(f"question {i}: \"evidence\" must be copied word for word from chunk {cid} "
                              f"(you gave: \"{evidence[:MAX_EVIDENCE_CHARS]}\")")
            cid, corrected = matches[0], True
        explanation = (q.get("explanation") or "").strip()
        if not explanation:
            raise Invalid(f"question {i} has no explanation")
        options = q.get("options")
        if (not isinstance(options, list) or len(options) != 4
                or not all(isinstance(o, str) and o.strip() for o in options)
                or len({canon(o) for o in options}) != 4):
            raise Invalid(f"question {i}: needs exactly 4 different, non-empty options")
        if q.get("answer_index") not in (0, 1, 2, 3):
            raise Invalid(f"question {i}: answer_index must be 0, 1, 2 or 3")
        options = [o.strip() for o in options]
        if LOGISTICS.search(prompt) or LOGISTICS.search(options[q["answer_index"]]):
            raise Invalid(f"question {i} is about the exam itself (its questions, weights, syllabus or format); "
                          "test the subject instead")
        own = shingles(prompt)
        if (overlap(own, practice_shingles) >= COPY_SHARE
                or overlap(shingles(prompt + " " + options[q["answer_index"]]), practice_shingles) >= COPY_SHARE):
            raise Invalid(f"question {i} copies an existing practice question; write a new one in its style")
        if any(overlap(own, s) >= DUPLICATE_SHARE for s in seen):
            raise Invalid(f"question {i} repeats another question; test a different point")
        item = {"type": "mcq", "prompt": prompt, "chunk_id": cid, "evidence": evidence,
                "explanation": explanation, "options": options, "answer": q["answer_index"],
                "chunk_corrected": corrected}
        if calc:
            item["calculation"] = check_calculation(q.get("calculation"), i, prompt, options,
                                                    q["answer_index"], raw[cid])
        seen.append(own)
        out.append(item)
    return out


def practice_shingles_for_set(conn, study_set_id: int) -> set:
    """4-word runs of every practice question in the set (question bank, appendix,
    end-of-chapter questions), for the copy check."""
    out: set = set()
    for c in db.study_chunks(conn, study_set_id):
        if is_practice(c):
            out |= shingles(c["text"])
    return out


# --- Planning -------------------------------------------------------------------

def calc_count(conn, topic_id: int) -> int:
    return sum(1 for q in db.questions(conn, topic_id) if q["calculation"])


def calc_topics(conn, study_set_id: int) -> list[int]:
    """Topics whose primary study chunks show a worked calculation."""
    return [t["id"] for t in db.topics(conn, study_set_id)
            if select_sources(db.topic_sources(conn, t["id"]), calc=True)[0]]


def target(conn, topic_id: int, n: int, calc: bool) -> int:
    """How many questions the topic should end up with."""
    have = len(db.questions(conn, topic_id))
    if not calc:
        return max(have, n)
    return min(MAX_PER_TOPIC, have + max(0, CALC_PER_TOPIC - calc_count(conn, topic_id)))


def missing(conn, topic_id: int, n: int, calc: bool = False) -> int:
    """How many questions a topic still needs (in calc mode: calculation questions)."""
    return max(0, target(conn, topic_id, n, calc) - len(db.questions(conn, topic_id)))


def estimate(conn, set_id: int, topic_ids: list[int], model: str, language: str, n: int,
             token_counter: Callable[[str], int] | None = None, calc: bool = False) -> topics.Estimate:
    import litellm

    if token_counter is None:
        def token_counter(text: str) -> int:
            return litellm.token_counter(model=model, text=text)
    inp = out = calls = n_chunks = n_topics = 0
    for tid in topic_ids:
        need = missing(conn, tid, n, calc)
        if not need:
            continue
        study, practice = select_sources(db.topic_sources(conn, tid), calc)
        existing = [q["prompt"] for q in db.questions(conn, tid)]
        n_chunks += len(study) + len(practice)
        per_call = sum(token_counter(m["content"]) for m in messages(
            db.get_topic(conn, tid), study, practice, language, min(need, QUESTIONS_PER_CALL), existing, calc))
        k = math.ceil(need / QUESTIONS_PER_CALL)
        inp += k * per_call   # the study text is resent with every call
        out += need * OUTPUT_TOKENS_PER_QUESTION
        calls += k
        n_topics += 1
    info = litellm.model_cost.get(model) or {}
    thinking = topics.THINKING_ALLOWANCE if info.get("supports_reasoning") else 0
    return topics.Estimate(model, n_chunks, n_topics, calls, inp, out,
                           info.get("input_cost_per_token"), info.get("output_cost_per_token"), thinking)


def generate(conn, set_name: str, model: str, topic_ids: list[int], *, n: int = QUESTIONS_PER_TOPIC,
             calc: bool = False, completion: Callable | None = None, extra: dict | None = None,
             retry_budget: int = topics.RETRY_BUDGET) -> tuple[dict[int, int], dict[int, str]]:
    """Bring each topic up to its target (n questions; with calc, CALC_PER_TOPIC
    calculation questions, never above MAX_PER_TOPIC), at most QUESTIONS_PER_CALL
    per call, saving each call's questions as soon as it succeeds. A topic whose
    call fails all its attempts is skipped and the run goes on; exhausted quota or
    too many invalid replies overall still stop it. Returns (questions added per
    topic, failed topics with the reason)."""
    study_set = topics._get_set(conn, set_name)
    set_id, language = study_set["id"], study_set["language"]
    log = topics.Budget(conn, set_id, retry_budget)
    practice_all = practice_shingles_for_set(conn, set_id)
    extra = {"max_tokens": MAX_OUTPUT_TOKENS, **(extra or {})}
    added: dict[int, int] = {}
    failed: dict[int, str] = {}
    for k, tid in enumerate(topic_ids, start=1):
        topic = db.get_topic(conn, tid)
        if topic is None or topic["study_set_id"] != set_id:
            raise ValueError(f"Topic {tid} isn't in study set {set_name!r}")
        existing = [q["prompt"] for q in db.questions(conn, tid)]
        goal = target(conn, tid, n, calc)
        if len(existing) >= goal:
            print(f"  topic {tid}: already has {len(existing)} questions, skipped", flush=True)
            continue
        study, practice = select_sources(db.topic_sources(conn, tid), calc)
        if not study:
            failed[tid] = "no worked calculation in its chunks" if calc else "no study text"
            print(f"  topic {tid}: {failed[tid]}, skipped", flush=True)
            continue
        print(f"  [{k}/{len(topic_ids)}] topic {tid}: {goal - len(existing)} new{' calculation' if calc else ''} "
              f"({len(existing)} existing), {len(study)} study + {len(practice)} practice chunks", flush=True)
        while len(existing) < goal:
            want = min(QUESTIONS_PER_CALL, goal - len(existing))
            try:
                qs, _ = complete_json(
                    messages(topic, study, practice, language, want, existing, calc), model=model,
                    purpose="questions.calculate" if calc else "questions.generate", log=log,
                    completion=completion, extra=extra,
                    validate=lambda obj, want=want, existing=list(existing): validate_questions(
                        obj, study, want, language, practice_all, existing, calc))
            except QuotaExhausted:
                raise
            except LLMError as err:
                log.check()  # too many invalid replies overall: stop the run
                failed[tid] = str(err)
                print(f"    failed, skipping this topic: {err}", flush=True)
                break
            log.check()
            with conn:
                for q in qs:
                    db.add_question(conn, tid, q, model)
            existing += [q["prompt"] for q in qs]
            fixed = sum(q["chunk_corrected"] for q in qs)
            if fixed:
                print(f"    {fixed} question(s) attributed to the wrong chunk: corrected to the chunk the quote is in",
                      flush=True)
            added[tid] = added.get(tid, 0) + len(qs)
    return added, failed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mukhtabar.questions")
    parser.add_argument("--db", default=db.DEFAULT_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("estimate", "generate"):
        p = sub.add_parser(name)
        p.add_argument("--set", required=True)
        p.add_argument("--model")
        p.add_argument("--topic", type=int, action="append", default=[])
        p.add_argument("--all", action="store_true", help="every topic that needs questions")
        p.add_argument("--n", type=int, default=QUESTIONS_PER_TOPIC, help="target questions per topic")
        p.add_argument("--calc", action="store_true",
                       help=f"add up to {CALC_PER_TOPIC} calculation questions to topics with worked calculations "
                            f"(at most {MAX_PER_TOPIC} questions per topic)")
        p.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
        p.add_argument("--yes", action="store_true")
    ct = sub.add_parser("calc-topics", help="topics whose chunks show a worked calculation (no model call)")
    ct.add_argument("--set", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--topic", type=int, required=True)
    args = parser.parse_args(argv)
    conn = db.connect(args.db)

    if args.command == "list":
        for q in db.questions(conn, args.topic):
            print(f"{q['id']:5} {'[calc] ' if q['calculation'] else ''}{q['prompt']}  (chunk {q['chunk_id']})")
        return
    s = topics._get_set(conn, args.set)
    if args.command == "calc-topics":
        for tid in calc_topics(conn, s["id"]):
            t = db.get_topic(conn, tid)
            chunks = select_sources(db.topic_sources(conn, tid), calc=True)[0]
            print(f"topic {tid:3}  {t['title']}  [{t['chapter'] or ''}]  {len(db.questions(conn, tid))} questions, "
                  f"{calc_count(conn, tid)} calculation; worked-calculation chunks: {[c['id'] for c in chunks]}")
        return
    model = topics._model(args)
    pool = calc_topics(conn, s["id"]) if args.calc else [t["id"] for t in db.topics(conn, s["id"])]
    ids = [t for t in (pool if args.all else args.topic) if missing(conn, t, args.n, args.calc)]
    if not ids:
        raise SystemExit("Nothing to do: the chosen topics already have enough questions.")
    e = estimate(conn, s["id"], ids, model, s["language"], args.n, calc=args.calc)
    if args.command == "estimate":
        print(topics.describe(e).replace(" batches;", " topics;"))
        return
    topics._confirm(e, args.yes)
    extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else None
    try:
        added, failed = generate(conn, args.set, model, ids, n=args.n, calc=args.calc, extra=extra)
    except LLMError as err:
        raise SystemExit(f"Stopped: {err}\nQuestions finished before the stop are saved; "
                         "run the same command again to continue.") from None
    print(f"Added {sum(added.values())} questions to {len(added)} topics.")
    if failed:
        print(f"Failed topics ({len(failed)}), skipped; run the same command again to retry only these:")
        for tid, reason in failed.items():
            print(f"  topic {tid}: {reason}")


if __name__ == "__main__":
    main()

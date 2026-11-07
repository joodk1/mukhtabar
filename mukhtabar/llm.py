"""Model calls through LiteLLM, with JSON validation and retry.

Keys come from the environment; a .env file in the working directory (or a
parent) is loaded first. LiteLLM reads each provider's usual variable, e.g.
GEMINI_API_KEY for "gemini/..." models. Keys are never stored by this package.
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T")

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")
RETRY_PROMPT = "Your reply was invalid: {error}\nReply again with the corrected JSON only."

# Waits (seconds) before retrying a call that hit a per-minute rate limit, or a
# temporary server/connection error. Exhausted quota (e.g. a daily limit) and
# other errors (bad key, unknown model, bad request) are never retried.
RATE_LIMIT_WAITS = (10, 30, 60)
TRANSIENT_WAITS = (5, 15)
MAX_WAIT = 90
# Only an explicitly daily limit counts as exhausted quota. Providers word
# per-minute and per-day errors alike ("You exceeded your current quota"), so
# anything else is treated as a rate limit, which still gets only a few waits.
QUOTA = re.compile(r"per[ _-]?day|perday|daily", re.I)
RETRY_DELAY = re.compile(r"retry(?:Delay)?\D{0,12}(\d+(?:\.\d+)?)\s*s", re.I)
TRANSIENT_NAMES = ("Timeout", "APIConnectionError", "InternalServerError", "ServiceUnavailableError")
# Error messages can echo a request URL that carries the key (Gemini uses ?key=...).
SECRETS = re.compile(r"(key=)[^&\s\"']+|AIza[0-9A-Za-z_\-]{20,}|sk-[0-9A-Za-z_\-]{20,}", re.I)


def redact(text: str) -> str:
    return SECRETS.sub(lambda m: (m.group(1) or "") + "[REDACTED]", text)


class Invalid(ValueError):
    """Raised by a validator; the message is shown to the model on retry."""


class LLMError(RuntimeError):
    def __init__(self, message: str, last_reply: str | None = None):
        super().__init__(message)
        self.last_reply = last_reply


class QuotaExhausted(LLMError):
    """The provider says the quota is used up; retrying now won't help."""


def classify_error(e: Exception) -> str:
    """"quota", "rate", "transient" or "fatal"."""
    name, msg = type(e).__name__, str(e)
    if "RateLimit" in name or "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "quota" if QUOTA.search(msg) else "rate"
    if any(n in name for n in TRANSIENT_NAMES) or re.search(r"\b(500|502|503|504)\b", msg):
        return "transient"
    return "fatal"


def _call_with_backoff(completion: Callable, sleep: Callable[[float], None], **kwargs):
    rate_tries = transient_tries = 0
    while True:
        try:
            return completion(**kwargs)
        except Exception as e:
            kind = classify_error(e)
            if kind == "rate" and rate_tries < len(RATE_LIMIT_WAITS):
                hinted = RETRY_DELAY.search(str(e))
                wait = max(RATE_LIMIT_WAITS[rate_tries], float(hinted.group(1)) if hinted else 0)
                rate_tries += 1
            elif kind == "transient" and transient_tries < len(TRANSIENT_WAITS):
                wait = TRANSIENT_WAITS[transient_tries]
                transient_tries += 1
            else:
                raise
            wait = min(wait, MAX_WAIT)
            print(f"  {kind} error ({type(e).__name__}); waiting {wait:.0f}s before retrying", flush=True)
            sleep(wait)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    attempts: int = 0
    seconds: float = 0.0


def parse_json(text: str) -> Any:
    """The JSON object in a reply, tolerating code fences and surrounding prose."""
    text = FENCE.sub("", text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise Invalid("no JSON object found")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise Invalid(f"not valid JSON ({e.msg} at position {e.pos})") from None


def _json_mode_supported(model: str) -> bool:
    import litellm

    try:
        return "response_format" in (litellm.get_supported_openai_params(model=model) or [])
    except Exception:
        return False


def _cost(response) -> float | None:
    import litellm

    try:
        return float(litellm.completion_cost(completion_response=response))
    except Exception:
        return None


def complete_json(
    messages: list[dict],
    *,
    validate: Callable[[Any], T],
    model: str,
    purpose: str,
    log: Callable[[dict], None] | None = None,
    max_attempts: int = 3,
    temperature: float = 0.0,
    completion: Callable | None = None,
    extra: dict | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, Usage]:
    """Call the model until validate(parsed reply) succeeds or attempts run out.

    Rate limits and temporary errors are retried with backoff (see
    RATE_LIMIT_WAITS) and don't count as attempts; exhausted quota raises
    QuotaExhausted at once. An invalid reply is shown back to the model with the
    error. Every call is reported to log() once, successful or not."""
    real = completion is None
    if real:
        import litellm

        completion = litellm.completion
    kwargs = {"model": model, "temperature": temperature, **(extra or {})}
    if real:
        kwargs["num_retries"] = 0  # retries are handled here, so quota errors stop cleanly
        if _json_mode_supported(model):
            kwargs["response_format"] = {"type": "json_object"}

    usage = Usage()
    msgs = list(messages)
    last_reply, error, ok = None, None, False
    rejections: list[str] = []   # why each invalid reply was rejected
    started = time.perf_counter()
    try:
        for attempt in range(1, max_attempts + 1):
            usage.attempts = attempt
            response = _call_with_backoff(completion, sleep, messages=msgs, **kwargs)
            u = getattr(response, "usage", None)
            usage.input_tokens += getattr(u, "prompt_tokens", 0) or 0
            usage.output_tokens += getattr(u, "completion_tokens", 0) or 0
            if real:
                c = _cost(response)
                usage.cost_usd = None if c is None else (usage.cost_usd or 0.0) + c
            last_reply = response.choices[0].message.content
            try:
                result = validate(parse_json(last_reply))
            except Invalid as e:
                error = str(e)
                rejections.append(error)
                msgs += [{"role": "assistant", "content": last_reply or ""},
                         {"role": "user", "content": RETRY_PROMPT.format(error=error)}]
                continue
            ok = True
            return result, usage
        raise LLMError(f"{purpose}: no valid reply after {max_attempts} attempts; last error: {error}", last_reply)
    except LLMError:
        raise
    except Exception as e:  # after backoff: quota, auth, unknown model, bad request, ...
        error = redact(f"{type(e).__name__}: {str(e)[:2000]}")
        if classify_error(e) == "quota":
            raise QuotaExhausted(f"{purpose}: quota exhausted, stopping ({error})") from e
        raise LLMError(f"{purpose}: {error}") from e
    finally:
        usage.seconds = time.perf_counter() - started
        if log:
            log({"purpose": purpose, "model": model, "input_tokens": usage.input_tokens,
                 "output_tokens": usage.output_tokens, "cost_usd": usage.cost_usd,
                 "attempts": usage.attempts, "ok": ok, "error": None if ok else error,
                 "rejections": rejections, "seconds": round(usage.seconds, 2)})

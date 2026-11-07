from types import SimpleNamespace

import pytest

from mukhtabar.llm import Invalid, LLMError, complete_json, parse_json


def reply(text, prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))


def scripted(*texts):
    calls = []

    def completion(messages, **kwargs):
        calls.append(messages)
        return reply(texts[len(calls) - 1])
    completion.calls = calls
    return completion


def needs_answer(obj):
    if "answer" not in obj:
        raise Invalid('missing "answer"')
    return obj["answer"]


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    with pytest.raises(Invalid):
        parse_json("no json here")


def test_invalid_reply_is_retried_with_the_error_and_then_succeeds():
    log = []
    completion = scripted("not json", '{"wrong": 1}', '{"answer": 42}')
    result, usage = complete_json([{"role": "user", "content": "q"}], validate=needs_answer, model="fake",
                                  purpose="test", log=log.append, completion=completion)
    assert result == 42 and usage.attempts == 3
    assert (usage.input_tokens, usage.output_tokens) == (30, 15)
    # The second retry shows the model its previous reply and what was wrong with it.
    last_messages = completion.calls[2]
    assert last_messages[-2] == {"role": "assistant", "content": '{"wrong": 1}'}
    assert 'missing "answer"' in last_messages[-1]["content"]
    assert len(log) == 1 and log[0]["ok"] and log[0]["attempts"] == 3 and log[0]["error"] is None


def test_gives_up_after_max_attempts_and_logs_the_failure():
    log = []
    with pytest.raises(LLMError, match='missing "answer"'):
        complete_json([], validate=needs_answer, model="fake", purpose="test", log=log.append,
                      completion=scripted("{}", "{}", "{}"))
    assert len(log) == 1 and not log[0]["ok"] and log[0]["attempts"] == 3


def test_transport_error_becomes_llm_error_and_is_logged():
    log = []

    def broken(messages, **kwargs):
        raise ConnectionError("network down")
    with pytest.raises(LLMError, match="network down"):
        complete_json([], validate=needs_answer, model="fake", purpose="test", log=log.append, completion=broken)
    assert not log[0]["ok"] and "network down" in log[0]["error"]


class RateLimitError(Exception):
    pass


def failing_then(errors, final='{"answer": 1}'):
    calls = []

    def completion(messages, **kwargs):
        calls.append(1)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return reply(final)
    completion.calls = calls
    return completion


def test_rate_limit_is_retried_with_backoff_using_the_hinted_delay():
    waits = []
    completion = failing_then([RateLimitError('429 RESOURCE_EXHAUSTED "retryDelay": "37s"'),
                               RateLimitError("429 Too Many Requests")])
    result, usage = complete_json([], validate=needs_answer, model="fake", purpose="t",
                                  completion=completion, sleep=waits.append)
    assert result == 1 and usage.attempts == 1  # rate-limit retries don't count as attempts
    assert waits == [37.0, 30]


def test_rate_limit_gives_up_after_the_last_wait():
    waits = []
    errors = [RateLimitError("429 Too Many Requests")] * 4
    with pytest.raises(LLMError):
        complete_json([], validate=needs_answer, model="fake", purpose="t",
                      completion=failing_then(errors), sleep=waits.append)
    assert waits == [10, 30, 60]


def test_per_minute_limit_worded_as_quota_is_waited_out():
    # Gemini words its free-tier per-minute limit like a quota error.
    waits = []
    per_minute = RateLimitError("429 You exceeded your current quota, please check your plan and billing details. "
                                "Quota exceeded for metric: generate_content_free_tier_requests, limit: 15, "
                                "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier, retryDelay: 41s")
    result, _ = complete_json([], validate=needs_answer, model="fake", purpose="t",
                              completion=failing_then([per_minute]), sleep=waits.append)
    assert result == 1 and waits == [41.0]


def test_exhausted_quota_stops_at_once():
    from mukhtabar.llm import QuotaExhausted
    waits = []
    quota = RateLimitError("429 RESOURCE_EXHAUSTED: Quota exceeded for metric "
                           "generate_content_free_tier_requests, limit: 250, PerDay")
    with pytest.raises(QuotaExhausted):
        complete_json([], validate=needs_answer, model="fake", purpose="t",
                      completion=failing_then([quota]), sleep=waits.append)
    assert waits == []


def test_errors_never_carry_the_api_key():
    from mukhtabar.llm import redact
    log = []

    # Built at runtime so no key-shaped literal sits in the repo for secret scanners.
    fake_key = "AI" + "za" + "x" * 35

    def leaky(messages, **kwargs):
        raise RuntimeError(f"400 Bad Request for url https://example.test/v1/models/x:generate?key={fake_key}&alt=sse")
    with pytest.raises(LLMError) as info:
        complete_json([], validate=needs_answer, model="fake", purpose="t", log=log.append, completion=leaky)
    assert fake_key not in str(info.value) and fake_key not in log[0]["error"]
    assert redact("in the text " + fake_key) == "in the text [REDACTED]"
    assert redact("token sk-" + "a" * 30) == "token [REDACTED]"


def test_rejection_reasons_are_logged_even_when_a_later_attempt_succeeds():
    log = []
    complete_json([], validate=needs_answer, model="fake", purpose="t", log=log.append,
                  completion=scripted("{}", '{"answer": 2}'))
    assert log[0]["ok"] and log[0]["rejections"] == ['missing "answer"']

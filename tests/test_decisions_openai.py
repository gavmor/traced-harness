"""Tests for the OpenAI Decisions backend.

Every request is served by ``httpx2.MockTransport``: no network, and the only
credential in play is the literal sentinel ``test-key-not-a-real-key``. The
live endpoint has never been called by this repository, so these tests pin what
the client *sends* and what it *accepts*, not confirmed server behaviour.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx2
import pytest

from traced_harness.decisions import (
    Choice,
    DecisionBackendError,
    DecisionQuestion,
    DecisionsConfigError,
    Level,
    annotate_trace,
)
from traced_harness.decisions_openai import (
    DEFAULT_DECISIONS_MODEL,
    OPENAI_DECISIONS_URL,
    OpenAIDecisionsBackend,
)

SENTINEL_KEY = "test-key-not-a-real-key"
FIXTURES = Path(__file__).parent / "fixtures" / "decisions"
TRACES = Path(__file__).parent / "fixtures" / "traces"

PREDICATE = DecisionQuestion(
    name="answered_the_question",
    instructions="Did the output answer the input?",
)
CHOICE = DecisionQuestion(
    name="tool_use",
    instructions="Judge the tool calls.",
    type="choice",
    choices=(
        Choice("necessary", "needed"),
        Choice("unnecessary"),
        Choice("missing"),
    ),
)
SCORE = DecisionQuestion(
    name="answer_quality",
    instructions="Rate the answer.",
    type="score",
    levels=(Level("Poor"), Level("Adequate"), Level("Good"), Level("Excellent")),
)
BOOL_CHOICE = DecisionQuestion(
    name="verdict",
    instructions="Boolean and string options must stay distinct.",
    type="choice",
    choices=(Choice(True), Choice("true")),
)


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


class _Recorder:
    """A MockTransport handler that records requests and replays one response."""

    def __init__(
        self,
        body: Any = None,
        status_code: int = 200,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.requests: list[httpx2.Request] = []
        self._body = body
        self._status = status_code
        self._content = content
        self._headers = headers

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self._content is not None:
            return httpx2.Response(
                self._status, content=self._content, headers=self._headers
            )
        return httpx2.Response(
            self._status, json=self._body, headers=self._headers
        )


def _backend(recorder: _Recorder, **kwargs: Any) -> OpenAIDecisionsBackend:
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(recorder), follow_redirects=False
    )
    kwargs.setdefault("api_key", SENTINEL_KEY)
    return OpenAIDecisionsBackend(client=client, **kwargs)


def _ask(recorder: _Recorder, questions: list[DecisionQuestion], **kwargs: Any) -> Any:
    backend = _backend(recorder, **kwargs)
    return asyncio.run(backend.ask("### Input\nhi\n\n### Output\nhello", questions))


# ---------------------------------------------------------------------------
# T12 — request shape.
# ---------------------------------------------------------------------------


def test_request_payload_shape() -> None:
    recorder = _Recorder(_fixture("predicate_response.json"))
    _ask(recorder, [PREDICATE])

    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == OPENAI_DECISIONS_URL

    payload = json.loads(request.content)
    assert payload == {
        "model": DEFAULT_DECISIONS_MODEL,
        "input": "### Input\nhi\n\n### Output\nhello",
        "questions": [
            {
                "name": "answered_the_question",
                "instructions": "Did the output answer the input?",
                "type": "predicate",
            }
        ],
    }
    # Absent fields are omitted, never sent as null.
    assert "choices" not in payload["questions"][0]
    assert "levels" not in payload["questions"][0]

    assert request.headers["authorization"] == f"Bearer {SENTINEL_KEY}"
    assert request.headers["content-type"] == "application/json"


def test_request_payload_carries_choices_and_levels() -> None:
    recorder = _Recorder(_fixture("choice_response.json"))
    _ask(recorder, [CHOICE])
    questions = json.loads(recorder.requests[0].content)["questions"]
    assert questions[0]["choices"] == [
        {"value": "necessary", "description": "needed"},
        {"value": "unnecessary"},
        {"value": "missing"},
    ]
    assert "levels" not in questions[0]

    recorder = _Recorder(_fixture("score_response.json"))
    _ask(recorder, [SCORE])
    questions = json.loads(recorder.requests[0].content)["questions"]
    assert questions[0]["levels"] == [
        {"label": "Poor"},
        {"label": "Adequate"},
        {"label": "Good"},
        {"label": "Excellent"},
    ]
    assert "choices" not in questions[0]


def test_base_url_override_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACED_DECISIONS_OPENAI_URL", "https://gateway.invalid/v1/d")
    recorder = _Recorder(_fixture("predicate_response.json"))
    _ask(recorder, [PREDICATE])
    assert str(recorder.requests[0].url) == "https://gateway.invalid/v1/d"


def test_timeout_default_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert OpenAIDecisionsBackend(api_key=SENTINEL_KEY).timeout == 60.0
    monkeypatch.setenv("TRACED_DECISIONS_TIMEOUT", "4.5")
    assert OpenAIDecisionsBackend(api_key=SENTINEL_KEY).timeout == 4.5
    monkeypatch.setenv("TRACED_DECISIONS_TIMEOUT", "soon")
    with pytest.raises(DecisionsConfigError, match="TRACED_DECISIONS_TIMEOUT"):
        OpenAIDecisionsBackend(api_key=SENTINEL_KEY)


# ---------------------------------------------------------------------------
# T13 — all three answer types map onto the harness record.
# ---------------------------------------------------------------------------


def test_all_three_answer_types() -> None:
    predicate = _ask(_Recorder(_fixture("predicate_response.json")), [PREDICATE])
    answer = predicate.answers[0]
    assert (answer.name, answer.type) == ("answered_the_question", "predicate")
    assert answer.probability == 0.93
    assert answer.confidence is None  # predicate has no confidence upstream
    assert answer.choice is None and answer.score is None
    assert answer.probabilities == ()
    assert answer.backend == "openai"
    assert predicate.model == "gpt-6-luna"
    assert predicate.usage == {"input_tokens": 812, "output_tokens": 14}
    assert predicate.raw == {}

    choice = _ask(_Recorder(_fixture("choice_response.json")), [CHOICE])
    answer = choice.answers[0]
    assert answer.type == "choice"
    assert answer.choice == "necessary"
    assert answer.confidence == 0.88
    assert [p.value for p in answer.probabilities] == [
        "necessary",
        "unnecessary",
        "missing",
    ]
    assert all(p.label is None for p in answer.probabilities)
    assert answer.score is None and answer.probability is None
    # The resolved model overwrites the requested one; unknown usage keys and
    # unknown top-level keys are passed through.
    assert choice.model == "gpt-6-luna-2026-01-30"
    assert choice.usage["input_tokens_details"] == {"cached_tokens": 128}
    assert choice.raw == {"service_tier": "default"}

    score = _ask(_Recorder(_fixture("score_response.json")), [SCORE])
    answer = score.answers[0]
    assert answer.type == "score"
    assert answer.score == 2.53
    assert answer.confidence == 0.63
    assert [(p.value, p.label) for p in answer.probabilities] == [
        (0, "Poor"),
        (1, "Adequate"),
        (2, "Good"),
        (3, "Excellent"),
    ]
    assert answer.choice is None and answer.probability is None


def test_bool_and_str_choice_values_stay_distinct() -> None:
    result = _ask(_Recorder(_fixture("boolean_choice_response.json")), [BOOL_CHOICE])
    answer = result.answers[0]
    assert answer.choice is True
    assert [p.value for p in answer.probabilities] == [True, "true"]
    assert [type(p.value) for p in answer.probabilities] == [bool, str]


def test_oversized_raw_passthrough_is_dropped(tmp_path: Path) -> None:
    body = _fixture("predicate_response.json")
    body["debug"] = "x" * 5000
    recorder = _Recorder(body)
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(recorder), follow_redirects=False
    )
    backend = OpenAIDecisionsBackend(api_key=SENTINEL_KEY, client=client)
    out = asyncio.run(
        annotate_trace(
            TRACES / "clean_run.jsonl",
            [PREDICATE],
            backend,
            out_file=tmp_path / "out.jsonl",
        )
    )
    record = json.loads(out.read_text().splitlines()[0])["additional_metadata"][
        "decisions"
    ]
    assert record["raw"] == {"_truncated": True}


# ---------------------------------------------------------------------------
# T14 — a refusal is legal for any question type.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question", [PREDICATE, CHOICE, SCORE])
def test_refusal_accepted_for_any_question_type(question: DecisionQuestion) -> None:
    body = {
        "model": "gpt-6-luna",
        "answers": [{"name": question.name, "type": "refusal"}],
        "usage": {"input_tokens": 5, "output_tokens": 0},
    }
    result = _ask(_Recorder(body), [question])
    answer = result.answers[0]
    assert answer.refused is True
    assert answer.type == "refusal"
    assert answer.probability is None
    assert answer.choice is None
    assert answer.score is None
    assert answer.confidence is None
    assert answer.probabilities == ()


# ---------------------------------------------------------------------------
# T15 — response validation.
# ---------------------------------------------------------------------------


def _predicate_body(**overrides: Any) -> dict[str, Any]:
    body = _fixture("predicate_response.json")
    body.update(overrides)
    return body


def _with_answer(**overrides: Any) -> dict[str, Any]:
    body = _fixture("predicate_response.json")
    body["answers"][0].update(overrides)
    return body


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("answer count", _predicate_body(answers=[])),
        (
            "answer count high",
            _predicate_body(
                answers=[
                    {"name": "answered_the_question", "type": "predicate", "probability": 0.5},
                    {"name": "extra", "type": "predicate", "probability": 0.5},
                ]
            ),
        ),
        ("name mismatch", _with_answer(name="something_else")),
        ("type mismatch", _with_answer(type="choice", choice="a")),
        ("unknown answer type", _with_answer(type="vibes")),
        ("probability out of range", _with_answer(probability=1.4)),
        ("probability not a number", _with_answer(probability="high")),
        ("blank model", _predicate_body(model="   ")),
        ("missing model", _predicate_body(model=None)),
        ("missing answers", {"model": "m", "usage": {"input_tokens": 1, "output_tokens": 1}}),
        ("missing usage", {"model": "m", "answers": []}),
        ("bool usage", _predicate_body(usage={"input_tokens": True, "output_tokens": 1})),
        ("float usage", _predicate_body(usage={"input_tokens": 1.5, "output_tokens": 1})),
        ("negative usage", _predicate_body(usage={"input_tokens": -1, "output_tokens": 1})),
        ("partial usage", _predicate_body(usage={"input_tokens": 1})),
        ("empty object", {}),
        ("array body", []),
    ],
)
def test_response_validation_errors(label: str, body: Any) -> None:
    with pytest.raises(DecisionBackendError):
        _ask(_Recorder(body), [PREDICATE])


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("duplicate keys", b'{"model": "a", "model": "b", "answers": [], "usage": {}}'),
        ("nan", b'{"model": "a", "answers": [], "usage": {"input_tokens": NaN}}'),
        ("infinity", b'{"model": "a", "answers": [], "usage": {"input_tokens": Infinity}}'),
        ("invalid utf-8", b'{"model": "\xff\xfe"}'),
        ("not json", b"<html>503</html>"),
    ],
)
def test_response_body_parsing_errors(label: str, content: bytes) -> None:
    with pytest.raises(DecisionBackendError):
        _ask(_Recorder(content=content), [PREDICATE])


def test_choice_answer_needs_a_distribution() -> None:
    body = {
        "model": "gpt-6-luna",
        "answers": [
            {
                "name": "tool_use",
                "type": "choice",
                "choice": "necessary",
                "confidence": 0.9,
                "probabilities": [{"value": "necessary", "probability": 0.9}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    with pytest.raises(DecisionBackendError, match="at least 2 probability"):
        _ask(_Recorder(body), [CHOICE])


# ---------------------------------------------------------------------------
# T16 — HTTP error statuses.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [302, 401, 422, 429, 500])
def test_http_error_statuses(status: int) -> None:
    headers = {"location": "https://evil.invalid/v1/decisions"} if status == 302 else None
    recorder = _Recorder(
        content=b'{"error": "nope"}', status_code=status, headers=headers
    )
    with pytest.raises(DecisionBackendError, match=f"HTTP {status}"):
        _ask(recorder, [PREDICATE])

    # The redirect was never followed: exactly one request reached the transport.
    assert len(recorder.requests) == 1
    assert str(recorder.requests[0].url) == OPENAI_DECISIONS_URL


def test_transport_failure_becomes_a_backend_error() -> None:
    def explode(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(explode), follow_redirects=False
    )
    backend = OpenAIDecisionsBackend(api_key=SENTINEL_KEY, client=client)
    with pytest.raises(DecisionBackendError, match="failed"):
        asyncio.run(backend.ask("ctx", [PREDICATE]))


def test_error_message_truncates_the_body() -> None:
    recorder = _Recorder(content=b"E" * 5000, status_code=500)
    with pytest.raises(DecisionBackendError) as caught:
        _ask(recorder, [PREDICATE])
    assert "E" * 200 in str(caught.value)
    assert "E" * 201 not in str(caught.value)


# ---------------------------------------------------------------------------
# T17 — the credential never leaks.
# ---------------------------------------------------------------------------


def test_missing_key_raises_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    recorder = _Recorder(_fixture("predicate_response.json"))
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(recorder), follow_redirects=False
    )
    backend = OpenAIDecisionsBackend(client=client)
    with pytest.raises(DecisionsConfigError, match="OPENAI_API_KEY"):
        asyncio.run(backend.ask("ctx", [PREDICATE]))
    assert recorder.requests == []

    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(DecisionsConfigError, match="OPENAI_API_KEY"):
        asyncio.run(OpenAIDecisionsBackend(client=client).ask("ctx", [PREDICATE]))
    assert recorder.requests == []


def test_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_KEY)
    recorder = _Recorder(_fixture("predicate_response.json"))
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(recorder), follow_redirects=False
    )
    asyncio.run(OpenAIDecisionsBackend(client=client).ask("ctx", [PREDICATE]))
    assert recorder.requests[0].headers["authorization"] == f"Bearer {SENTINEL_KEY}"


_LEAKY_CASES: list[tuple[str, Any, int, bytes | None]] = [
    ("success", _fixture("predicate_response.json"), 200, None),
    ("name mismatch", _with_answer(name="nope"), 200, None),
    ("bool usage", _predicate_body(usage={"input_tokens": True, "output_tokens": 0}), 200, None),
    ("http 302", None, 302, b'{"error": "redirect"}'),
    ("http 401", None, 401, b'{"error": "unauthorized"}'),
    ("http 500", None, 500, b'{"error": "server"}'),
    ("not json", None, 200, b"<html>503</html>"),
]


@pytest.mark.parametrize(("label", "body", "status", "content"), _LEAKY_CASES)
def test_api_key_never_leaks(
    label: str,
    body: Any,
    status: int,
    content: bytes | None,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = _Recorder(body=body, status_code=status, content=content)
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(recorder), follow_redirects=False
    )
    backend = OpenAIDecisionsBackend(api_key=SENTINEL_KEY, client=client)

    caplog.set_level(logging.DEBUG)
    out = tmp_path / f"{label.replace(' ', '_')}.jsonl"
    try:
        asyncio.run(
            annotate_trace(
                TRACES / "clean_run.jsonl", [PREDICATE], backend, out_file=out
            )
        )
    except Exception as exc:  # noqa: BLE001 - the point is to inspect it
        assert SENTINEL_KEY not in str(exc)
        assert SENTINEL_KEY not in repr(exc)

    assert SENTINEL_KEY not in repr(backend)
    assert SENTINEL_KEY not in caplog.text
    for record in caplog.records:
        assert SENTINEL_KEY not in record.getMessage()
    if out.is_file():
        assert SENTINEL_KEY not in out.read_text()

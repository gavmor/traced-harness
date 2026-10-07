"""Tests for the backend-neutral decisions vocabulary and the annotation driver.

No network, no credentials, no live backend. Backends here are plain fakes
defined in this file, following ``tests/test_session_runner.py``: structural
stand-ins rather than mocks, so the assertions describe behaviour instead of
call bookkeeping.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from traced_harness.decisions import (
    DECISIONS_METADATA_KEY,
    DECISIONS_SCHEMA_VERSION,
    Choice,
    DecisionAnswer,
    DecisionBackend,
    DecisionBackendError,
    DecisionQuestion,
    DecisionResult,
    DecisionsConfigError,
    Level,
    Probability,
    ReplayDecisionBackend,
    annotate_trace,
    build_turn_context,
    decision_id,
    decision_span,
    load_questions_file,
    questions_from_json_schema,
)
from traced_harness.eval import ToolExecutionData, TraceTurn, evaluate_trace, load_trace

FIXTURES = Path(__file__).parent / "fixtures" / "traces"
CLEAN_RUN = FIXTURES / "clean_run.jsonl"

PREDICATE = DecisionQuestion(
    name="answered_the_question",
    instructions="Did the output answer the input?",
)
CHOICE = DecisionQuestion(
    name="tool_use",
    instructions="Judge the tool calls.",
    type="choice",
    choices=(Choice("necessary"), Choice("unnecessary"), Choice("missing")),
)
SCORE = DecisionQuestion(
    name="answer_quality",
    instructions="Rate the answer.",
    type="score",
    levels=(Level("Poor"), Level("Good")),
)


# ---------------------------------------------------------------------------
# Fakes.
# ---------------------------------------------------------------------------


class _FakeBackend(DecisionBackend):
    """Answers every question affirmatively; records the contexts it saw."""

    name = "fake"

    def __init__(self, refuse: bool = False) -> None:
        self.contexts: list[str] = []
        self.refuse = refuse

    async def ask(
        self, context: str, questions: Sequence[DecisionQuestion]
    ) -> DecisionResult:
        self.contexts.append(context)
        answers = []
        for question in questions:
            if self.refuse:
                answers.append(DecisionAnswer(question.name, "refusal", backend=self.name))
            elif question.type == "predicate":
                answers.append(
                    DecisionAnswer(
                        question.name, "predicate", probability=0.5, backend=self.name
                    )
                )
            elif question.type == "choice":
                assert question.choices is not None
                answers.append(
                    DecisionAnswer(
                        question.name,
                        "choice",
                        choice=question.choices[0].value,
                        confidence=0.5,
                        probabilities=(
                            Probability(question.choices[0].value, None, 0.5),
                            Probability(question.choices[1].value, None, 0.5),
                        ),
                        backend=self.name,
                    )
                )
            else:
                answers.append(
                    DecisionAnswer(
                        question.name, "score", score=1.0, confidence=0.5, backend=self.name
                    )
                )
        return DecisionResult(
            answers=tuple(answers),
            model="fake-1",
            usage={"input_tokens": 10, "output_tokens": 2},
        )


class _ExplodingBackend(DecisionBackend):
    """Fails every call with the error class the driver is allowed to swallow."""

    name = "exploding"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or DecisionBackendError("boom: upstream said no")
        self.calls = 0

    async def ask(
        self, context: str, questions: Sequence[DecisionQuestion]
    ) -> DecisionResult:
        self.calls += 1
        raise self.error


def _turn(
    index: int = 0,
    text_input: str = "in",
    output: str = "out",
    tools: list[ToolExecutionData] | None = None,
    metadata: dict | None = None,
) -> TraceTurn:
    return TraceTurn(
        index=index,
        input=text_input,
        actual_output=output,
        tools_called=tools or [],
        additional_metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# T1 — question validation.
# ---------------------------------------------------------------------------


def test_question_validation() -> None:
    with pytest.raises(ValueError, match="at least 2 choices"):
        DecisionQuestion("q", "i", "choice", choices=(Choice("only"),))

    with pytest.raises(ValueError, match="duplicate choice value"):
        DecisionQuestion(
            "q", "i", "choice", choices=(Choice("yes"), Choice("no"), Choice("yes"))
        )

    with pytest.raises(ValueError, match="must not carry levels"):
        DecisionQuestion(
            "q",
            "i",
            "choice",
            choices=(Choice("a"), Choice("b")),
            levels=(Level("Low"), Level("High")),
        )

    with pytest.raises(ValueError, match="must not carry choices"):
        DecisionQuestion(
            "q",
            "i",
            "score",
            choices=(Choice("a"), Choice("b")),
            levels=(Level("Low"), Level("High")),
        )

    with pytest.raises(ValueError, match="at least 2 levels"):
        DecisionQuestion("q", "i", "score", levels=(Level("Low"),))

    with pytest.raises(ValueError, match="name must be a non-blank"):
        DecisionQuestion("   ", "i")

    with pytest.raises(ValueError, match="instructions must be a non-blank"):
        DecisionQuestion("q", "  ")

    with pytest.raises(ValueError, match="neither choices nor levels"):
        DecisionQuestion("q", "i", "predicate", choices=(Choice("a"), Choice("b")))

    with pytest.raises(ValueError, match="unknown type"):
        DecisionQuestion("q", "i", "ranking")  # type: ignore[arg-type]


def test_choice_values_are_not_coerced_across_types() -> None:
    """``True`` and ``"true"`` are distinct options, never deduplicated."""
    question = DecisionQuestion(
        "verdict", "i", "choice", choices=(Choice(True), Choice("true"))
    )
    assert [c.value for c in question.choices or ()] == [True, "true"]


# ---------------------------------------------------------------------------
# T2 — JSON Schema mapping.
# ---------------------------------------------------------------------------


def test_json_schema_mapping() -> None:
    schema = {
        "type": "object",
        "properties": {
            "answered": {"type": "boolean", "description": "Did it answer?"},
            "tool_use": {
                "type": "string",
                "enum": ["necessary", "unnecessary", "missing"],
                "description": "Judge the tools.",
            },
            "quality": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Rate 1-5.",
            },
            "graded": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2,
                "x-levels": ["Bad", "OK", "Good"],
                "description": "Graded rating.",
            },
        },
    }
    questions = questions_from_json_schema(schema)

    # Property order is preserved.
    assert [q.name for q in questions] == ["answered", "tool_use", "quality", "graded"]

    assert questions[0].type == "predicate"
    assert questions[0].instructions == "Did it answer?"

    assert questions[1].type == "choice"
    assert [c.value for c in questions[1].choices or ()] == [
        "necessary",
        "unnecessary",
        "missing",
    ]

    assert questions[2].type == "score"
    assert [lv.label for lv in questions[2].levels or ()] == ["1", "2", "3", "4", "5"]

    assert [lv.label for lv in questions[3].levels or ()] == ["Bad", "OK", "Good"]


@pytest.mark.parametrize(
    "prop",
    [
        {"type": "number", "description": "A ratio."},
        {"type": "string", "description": "Free text."},
        {"type": "array", "items": {"type": "string"}, "description": "A list."},
        {"type": "object", "description": "Nested."},
    ],
)
def test_json_schema_unsupported_property_raises(prop: dict) -> None:
    with pytest.raises(DecisionsConfigError, match="ratio"):
        questions_from_json_schema({"type": "object", "properties": {"ratio": prop}})


def test_json_schema_requires_description_and_level_count() -> None:
    with pytest.raises(DecisionsConfigError, match="answered"):
        questions_from_json_schema(
            {"type": "object", "properties": {"answered": {"type": "boolean"}}}
        )

    with pytest.raises(DecisionsConfigError, match="exactly 3 labels"):
        questions_from_json_schema(
            {
                "type": "object",
                "properties": {
                    "graded": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                        "x-levels": ["Bad", "Good"],
                        "description": "Graded.",
                    }
                },
            }
        )


def test_questions_file_accepts_both_forms(tmp_path: Path) -> None:
    native = tmp_path / "native.json"
    native.write_text(
        json.dumps(
            [
                {"name": "a", "instructions": "i"},
                {
                    "name": "b",
                    "instructions": "i",
                    "type": "choice",
                    "choices": ["x", {"value": "y", "description": "why"}],
                },
                {
                    "name": "c",
                    "instructions": "i",
                    "type": "score",
                    "levels": ["Low", {"label": "High", "description": "best"}],
                },
            ]
        )
    )
    questions = load_questions_file(native)
    assert [q.type for q in questions] == ["predicate", "choice", "score"]
    assert (questions[1].choices or ())[1].description == "why"

    schema_file = tmp_path / "schema.json"
    schema_file.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"a": {"type": "boolean", "description": "d"}},
            }
        )
    )
    assert [q.name for q in load_questions_file(schema_file)] == ["a"]

    bogus = tmp_path / "bogus.json"
    bogus.write_text(json.dumps("not a questions document"))
    with pytest.raises(DecisionsConfigError):
        load_questions_file(bogus)

    with pytest.raises(DecisionsConfigError, match="not found"):
        load_questions_file(tmp_path / "missing.json")


def test_shipped_example_questions_load() -> None:
    """The questions file the README and E1 use is valid."""
    examples = Path(__file__).parents[1] / "docs" / "examples" / "decision-questions.json"
    questions = load_questions_file(examples)
    assert [q.name for q in questions] == [
        "answered_the_question",
        "tool_use",
        "answer_quality",
    ]


# ---------------------------------------------------------------------------
# T3 — answer records are JSON-serializable.
# ---------------------------------------------------------------------------


def test_answer_record_is_json_serializable() -> None:
    answers = [
        DecisionAnswer("p", "predicate", probability=0.25),
        DecisionAnswer(
            "c",
            "choice",
            choice=True,
            confidence=0.5,
            probabilities=(Probability(True, None, 0.5), Probability("true", None, 0.5)),
        ),
        DecisionAnswer(
            "s",
            "score",
            score=1.75,
            confidence=0.4,
            probabilities=(Probability(0, "Poor", 0.25), Probability(1, "Good", 0.75)),
        ),
        DecisionAnswer("r", "refusal"),
    ]
    for answer in answers:
        record = answer.as_record()
        assert json.loads(json.dumps(record)) == record

    refusal = answers[-1].as_record()
    assert refusal["refused"] is True
    assert refusal["probability"] is None
    assert refusal["choice"] is None
    assert refusal["score"] is None
    assert refusal["confidence"] is None
    assert refusal["probabilities"] == []

    assert answers[0].refused is False


@pytest.mark.parametrize("bad", [-0.01, 1.01, "0.5", True])
def test_probability_outside_unit_interval_raises(bad: object) -> None:
    with pytest.raises(ValueError):
        DecisionAnswer("p", "predicate", probability=bad)  # type: ignore[arg-type]


def test_absent_probability_is_legal() -> None:
    assert DecisionAnswer("r", "refusal").probability is None


# ---------------------------------------------------------------------------
# T4 — the context builder is byte-stable.
# ---------------------------------------------------------------------------


def test_context_builder_is_deterministic() -> None:
    turn = load_trace(CLEAN_RUN)[0]
    expected = (
        "### Input\n"
        "What is the health and armor of the Colonial Spatha tank?\n"
        "\n"
        "### Output\n"
        "The Spatha has 3650 HP and Tier 2 Heavy Vehicle armor.\n"
        "\n"
        "### Tools called\n"
        '- get_vehicle_stats({"vehicle_name": "Spatha"}) -> '
        '{"title": "Spatha", "hp": 3650, "armor_type": "Tier 2 Heavy Vehicle"}'
    )
    assert build_turn_context(turn) == expected
    assert not build_turn_context(turn).endswith("\n")

    # Parameters are key-sorted regardless of insertion order.
    unsorted = _turn(tools=[ToolExecutionData("t", {"b": 2, "a": 1}, "ok")])
    resorted = _turn(tools=[ToolExecutionData("t", {"a": 1, "b": 2}, "ok")])
    assert build_turn_context(unsorted) == build_turn_context(resorted)
    assert '- t({"a": 1, "b": 2}) -> ok' in build_turn_context(unsorted)

    # Tool output is truncated at 2000 characters.
    long_turn = _turn(tools=[ToolExecutionData("t", {}, "x" * 5000)])
    assert "x" * 2000 in build_turn_context(long_turn)
    assert "x" * 2001 not in build_turn_context(long_turn)

    # The section vanishes entirely when no tools were called.
    assert build_turn_context(_turn()) == "### Input\nin\n\n### Output\nout"
    assert "Tools called" not in build_turn_context(_turn())


# ---------------------------------------------------------------------------
# T5 — decision ids are deterministic.
# ---------------------------------------------------------------------------


def test_decision_id_stable() -> None:
    expected = hashlib.sha256(
        b"clean_run:0:answered_the_question"
    ).hexdigest()[:16]
    assert expected == "f261ff1032b66d89"
    assert decision_id("clean_run", 0, "answered_the_question") == expected
    assert decision_id("clean_run", 0, "answered_the_question") == decision_id(
        "clean_run", 0, "answered_the_question"
    )

    assert decision_id("other_run", 0, "answered_the_question") != expected
    assert decision_id("clean_run", 1, "answered_the_question") != expected
    assert decision_id("clean_run", 0, "other_question") != expected
    assert len(expected) == 16


# ---------------------------------------------------------------------------
# T6/T7/T8 — the annotation driver.
# ---------------------------------------------------------------------------


def _annotate(
    tmp_path: Path,
    backend: DecisionBackend,
    source: Path = CLEAN_RUN,
    questions: Sequence[DecisionQuestion] = (PREDICATE, CHOICE, SCORE),
    **kwargs: object,
) -> Path:
    out = tmp_path / f"{source.stem}.decided.jsonl"
    return asyncio.run(
        annotate_trace(source, questions, backend, out_file=out, **kwargs)  # type: ignore[arg-type]
    )


def test_annotate_trace_with_fake_backend(tmp_path: Path) -> None:
    backend = _FakeBackend()
    out = _annotate(tmp_path, backend)

    original = [json.loads(line) for line in CLEAN_RUN.read_text().splitlines() if line.strip()]
    annotated = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(annotated) == len(original) == 2
    assert len(backend.contexts) == 2

    for before, after in zip(original, annotated):
        record = after["additional_metadata"].pop(DECISIONS_METADATA_KEY)
        # Everything else round-trips identically; only the slot was added.
        assert after == before

        assert record["schema_version"] == DECISIONS_SCHEMA_VERSION
        assert record["backend"] == "fake"
        assert record["model"] == "fake-1"
        assert record["trace_id"] == "clean_run"
        assert record["session_id"] == "clean_session"
        assert record["usage"] == {"input_tokens": 10, "output_tokens": 2}
        assert record["raw"] == {}
        assert [a["name"] for a in record["answers"]] == [
            "answered_the_question",
            "tool_use",
            "answer_quality",
        ]
        assert record["measure"]["decision.backend"] == "fake"
        assert record["measure"]["decision.question_count"] == 3
        assert record["measure"]["decision.refusal_count"] == 0
        assert "wall_seconds" in record["measure"]

    assert annotated[0]["additional_metadata"] == original[0]["additional_metadata"]
    first = json.loads(out.read_text().splitlines()[0])
    ids = [
        a["decision_id"]
        for a in first["additional_metadata"][DECISIONS_METADATA_KEY]["answers"]
    ]
    assert ids[0] == decision_id("clean_run", 0, "answered_the_question")
    assert len(set(ids)) == 3


def test_annotate_trace_default_output_path(tmp_path: Path) -> None:
    source = tmp_path / "trace_demo.jsonl"
    source.write_text(CLEAN_RUN.read_text())
    out = asyncio.run(annotate_trace(source, [PREDICATE], _FakeBackend()))
    assert out == tmp_path / "trace_demo.decided.jsonl"
    assert out.is_file()


def test_annotate_trace_does_not_mutate_input(tmp_path: Path) -> None:
    before = hashlib.sha256(CLEAN_RUN.read_bytes()).hexdigest()
    _annotate(tmp_path, _FakeBackend())
    assert hashlib.sha256(CLEAN_RUN.read_bytes()).hexdigest() == before


def test_annotate_trace_overwrites_an_existing_slot(tmp_path: Path) -> None:
    seeded = tmp_path / "seeded.jsonl"
    lines = []
    for line in CLEAN_RUN.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        obj["additional_metadata"][DECISIONS_METADATA_KEY] = {"stale": True}
        lines.append(json.dumps(obj))
    seeded.write_text("\n".join(lines) + "\n")

    out = _annotate(tmp_path, _FakeBackend(), source=seeded)
    for line in out.read_text().splitlines():
        record = json.loads(line)["additional_metadata"][DECISIONS_METADATA_KEY]
        assert "stale" not in record
        assert record["schema_version"] == DECISIONS_SCHEMA_VERSION


@pytest.mark.parametrize(
    "name", ["clean_run.jsonl", "failing_peripheral.jsonl", "multi_turn_cascade.jsonl"]
)
def test_annotated_trace_still_loads_and_evaluates(tmp_path: Path, name: str) -> None:
    """Decision records must never be mistaken for peripheral failures."""
    source = FIXTURES / name
    before = evaluate_trace(source)

    for backend in (_FakeBackend(), _FakeBackend(refuse=True), _ExplodingBackend()):
        out = _annotate(tmp_path / backend.name, backend, source=source)
        assert len(load_trace(out)) == before.total_turns

        after = evaluate_trace(out)
        assert after.total_turns == before.total_turns
        assert after.evaluated_turns == before.evaluated_turns
        assert len(after.failed_tool_calls) == len(before.failed_tool_calls)
        assert after.first_failure_turn == before.first_failure_turn

        # The record lives in additional_metadata, never in tools_called.
        for turn in load_trace(out):
            assert DECISIONS_METADATA_KEY in turn.additional_metadata
            for tool in turn.tools_called:
                assert DECISIONS_METADATA_KEY not in tool.output


# ---------------------------------------------------------------------------
# T9 — the decision span.
# ---------------------------------------------------------------------------


def test_decision_span_records_measurement() -> None:
    with decision_span(backend="openai", question_count=3) as measurement:
        measurement.attributes["decision.refusal_count"] = 1
    record = measurement.as_record()

    assert record["wall_seconds"] >= 0.0
    assert record["cpu_seconds"] >= 0.0
    assert record["decision.backend"] == "openai"
    assert record["decision.question_count"] == 3
    assert record["decision.refusal_count"] == 1
    assert json.loads(json.dumps(record)) == record


# ---------------------------------------------------------------------------
# T10 — the replay backend.
# ---------------------------------------------------------------------------


def _replay_file(tmp_path: Path, bodies: list[dict]) -> Path:
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(bodies))
    return path


def _body(probability: float = 0.5) -> dict:
    return {
        "model": "gpt-6-luna",
        "answers": [
            {
                "name": "answered_the_question",
                "type": "predicate",
                "probability": probability,
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def test_replay_backend(tmp_path: Path) -> None:
    backend = ReplayDecisionBackend(_replay_file(tmp_path, [_body(0.1), _body(0.9)]))

    first = asyncio.run(backend.ask("ctx", [PREDICATE]))
    second = asyncio.run(backend.ask("ctx", [PREDICATE]))
    assert first.answers[0].probability == 0.1
    assert second.answers[0].probability == 0.9
    assert first.model == "gpt-6-luna"

    with pytest.raises(DecisionBackendError, match="exhausted"):
        asyncio.run(backend.ask("ctx", [PREDICATE]))


def test_replay_backend_runs_the_real_validation_path(tmp_path: Path) -> None:
    """A recorded body is checked exactly like a live one."""
    broken = _body()
    broken["answers"][0]["name"] = "wrong_name"
    backend = ReplayDecisionBackend(_replay_file(tmp_path, [broken]))
    with pytest.raises(DecisionBackendError, match="does not match question"):
        asyncio.run(backend.ask("ctx", [PREDICATE]))

    non_strict = ReplayDecisionBackend(_replay_file(tmp_path, [_body()]), strict=False)
    assert asyncio.run(non_strict.ask("ctx", [PREDICATE])).answers[0].probability == 0.5
    assert asyncio.run(non_strict.ask("ctx", [PREDICATE])).answers[0].probability == 0.5


def test_replay_backend_rejects_a_bad_file(tmp_path: Path) -> None:
    with pytest.raises(DecisionsConfigError, match="not found"):
        ReplayDecisionBackend(tmp_path / "nope.json")

    not_a_list = tmp_path / "obj.json"
    not_a_list.write_text(json.dumps({"model": "x"}))
    with pytest.raises(DecisionsConfigError, match="list of recorded"):
        ReplayDecisionBackend(not_a_list)


def test_shipped_example_replay_file_annotates_the_clean_run(tmp_path: Path) -> None:
    """The committed example files are the ones the docs tell you to run."""
    examples = Path(__file__).parents[1] / "docs" / "examples"
    questions = load_questions_file(examples / "decision-questions.json")
    backend = ReplayDecisionBackend(examples / "recorded-decisions.json")
    out = tmp_path / "clean_run.decided.jsonl"
    asyncio.run(annotate_trace(CLEAN_RUN, questions, backend, out_file=out))

    records = [
        json.loads(line)["additional_metadata"][DECISIONS_METADATA_KEY]
        for line in out.read_text().splitlines()
    ]
    assert len(records) == 2
    assert records[0]["model"] == "gpt-6-luna"
    assert [a["type"] for a in records[0]["answers"]] == [
        "predicate",
        "choice",
        "score",
    ]


# ---------------------------------------------------------------------------
# T11 — error policy.
# ---------------------------------------------------------------------------


def test_on_error_record_vs_raise(tmp_path: Path) -> None:
    backend = _ExplodingBackend()
    out = _annotate(tmp_path, backend)

    # record: every turn still processed, the error lands in the slot.
    assert backend.calls == 2
    for line in out.read_text().splitlines():
        record = json.loads(line)["additional_metadata"][DECISIONS_METADATA_KEY]
        assert record["error"].startswith("DecisionBackendError: boom")
        assert record["schema_version"] == DECISIONS_SCHEMA_VERSION
        assert record["backend"] == "exploding"
        assert "answers" not in record
        assert "wall_seconds" in record["measure"]

    # raise: the first failure propagates.
    raiser = _ExplodingBackend()
    with pytest.raises(DecisionBackendError, match="boom"):
        _annotate(tmp_path / "raise", raiser, on_error="raise")
    assert raiser.calls == 1


def test_config_errors_propagate_in_both_modes(tmp_path: Path) -> None:
    for mode in ("record", "raise"):
        backend = _ExplodingBackend(DecisionsConfigError("missing credential"))
        with pytest.raises(DecisionsConfigError, match="missing credential"):
            _annotate(tmp_path / mode, backend, on_error=mode)

    with pytest.raises(DecisionsConfigError, match="no questions"):
        _annotate(tmp_path / "empty", _FakeBackend(), questions=[])

    with pytest.raises(DecisionsConfigError, match="on_error"):
        _annotate(tmp_path / "mode", _FakeBackend(), on_error="explode")

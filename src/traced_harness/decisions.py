"""Calibrated decision backends as a first-class harness capability.

A *decision* here is a calibrated answer to one explicit question asked about
one recorded turn: a predicate with a probability, a choice with a confidence,
or a score over ordered levels. The point is to turn free text into numbers
without sampling a model's prose, so repeated runs over the same trace produce
the same record.

Layering
--------
The harness ships the **mechanism** and never the **rubric**:

* the backend-neutral vocabulary — :class:`DecisionQuestion` / :class:`Choice` /
  :class:`Level` / :class:`DecisionAnswer` / :class:`DecisionResult`;
* the :class:`DecisionBackend` ABC every provider implements;
* the response parser and its validation rules, shared by every backend so a
  replayed body is checked exactly like a live one;
* the trace slot (:data:`DECISIONS_METADATA_KEY`), the
  :func:`decision_span` measurement, and the :func:`annotate_trace` driver;
* one offline reading device, :class:`ReplayDecisionBackend`.

*Which* questions to ask is the study's business and is always supplied by the
caller. There is deliberately no default question set, no built-in quality
rubric, no scoring and no pass/fail opinion anywhere in this package — the same
line :mod:`traced_harness.telemetry` draws between measuring and interpreting.

Decisions are a **post-hoc annotation over recorded traces**: :func:`annotate_trace`
reads a finished JSONL trace and writes an annotated copy beside it. Nothing
here runs during a live agent turn, and no module in the harness core imports
this one.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from opentelemetry import trace

from traced_harness.eval import TraceTurn, load_trace
from traced_harness.telemetry import SpanMeasurement, measured_span

__all__ = [
    "DECISIONS_METADATA_KEY",
    "DECISIONS_SCHEMA_VERSION",
    "DECISION_SPAN",
    "Choice",
    "DecisionAnswer",
    "DecisionBackend",
    "DecisionBackendError",
    "DecisionQuestion",
    "DecisionResult",
    "DecisionsConfigError",
    "DecisionsError",
    "Level",
    "Probability",
    "ReplayDecisionBackend",
    "annotate_trace",
    "build_decision_backend",
    "build_turn_context",
    "decision_id",
    "decision_span",
    "load_questions_file",
    "loads_strict",
    "parse_decision_response",
    "questions_from_document",
    "questions_from_json_schema",
]

logger = logging.getLogger(__name__)

#: Slot in a turn's ``additional_metadata`` where decision records are written.
DECISIONS_METADATA_KEY = "decisions"
#: Bumped on any change to the record shape written into that slot.
DECISIONS_SCHEMA_VERSION = 1

DECISION_TRACER_NAME = "traced.harness.decisions"
DECISION_SPAN = "decision.turn"

DECISION_BACKEND_ATTR = "decision.backend"
DECISION_QUESTION_COUNT_ATTR = "decision.question_count"
DECISION_REFUSAL_COUNT_ATTR = "decision.refusal_count"

#: ``raw`` provider passthrough is dropped wholesale beyond this many characters.
RAW_PASSTHROUGH_LIMIT = 4096
#: Tool output truncation in :func:`build_turn_context`, matching ``agent.py``.
TOOL_OUTPUT_LIMIT = 2000


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


class DecisionsError(Exception):
    """Base class for every error raised by the decisions subsystem."""


class DecisionsConfigError(DecisionsError):
    """The run is misconfigured. Always raised *before* any request is made."""


class DecisionBackendError(DecisionsError):
    """A backend failed: transport, non-2xx status, or an unusable response."""


# ---------------------------------------------------------------------------
# Question side.
# ---------------------------------------------------------------------------

QuestionType = Literal["predicate", "choice", "score"]
_QUESTION_TYPES = ("predicate", "choice", "score")


@dataclass(frozen=True)
class Choice:
    """One selectable option of a ``choice`` question."""

    value: str | bool
    description: str | None = None


@dataclass(frozen=True)
class Level:
    """One rung of a ``score`` question's ordered ladder, low to high."""

    label: str
    description: str | None = None


@dataclass(frozen=True)
class DecisionQuestion:
    """One question asked about one turn.

    Validation mirrors the upstream OpenAI Decisions client's cross-field rules
    so a question this harness accepts is a question the endpoint accepts.
    """

    name: str
    instructions: str
    type: QuestionType = "predicate"
    choices: tuple[Choice, ...] | None = None
    levels: tuple[Level, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("question name must be a non-blank string")
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError(
                f"question {self.name!r}: instructions must be a non-blank string"
            )
        if self.type not in _QUESTION_TYPES:
            raise ValueError(
                f"question {self.name!r}: unknown type {self.type!r}; "
                f"expected one of {_QUESTION_TYPES}"
            )

        if self.type == "choice":
            if not self.choices or len(self.choices) < 2:
                raise ValueError(
                    f"question {self.name!r}: a choice question needs at least 2 choices"
                )
            seen: set[tuple[type, Any]] = set()
            for choice in self.choices:
                key = (type(choice.value), choice.value)
                if key in seen:
                    raise ValueError(
                        f"question {self.name!r}: duplicate choice value "
                        f"{choice.value!r} ({type(choice.value).__name__})"
                    )
                seen.add(key)
            if self.levels is not None:
                raise ValueError(
                    f"question {self.name!r}: a choice question must not carry levels"
                )
        elif self.type == "score":
            if not self.levels or len(self.levels) < 2:
                raise ValueError(
                    f"question {self.name!r}: a score question needs at least 2 levels"
                )
            if self.choices is not None:
                raise ValueError(
                    f"question {self.name!r}: a score question must not carry choices"
                )
        else:  # predicate
            if self.choices is not None or self.levels is not None:
                raise ValueError(
                    f"question {self.name!r}: a predicate question takes neither "
                    f"choices nor levels"
                )

    def as_payload(self) -> dict[str, Any]:
        """Wire form of the question: absent fields are omitted, never ``null``."""
        payload: dict[str, Any] = {
            "name": self.name,
            "instructions": self.instructions,
            "type": self.type,
        }
        if self.choices is not None:
            payload["choices"] = [_option_payload("value", c.value, c.description) for c in self.choices]
        if self.levels is not None:
            payload["levels"] = [_option_payload("label", lv.label, lv.description) for lv in self.levels]
        return payload


def _option_payload(key: str, value: Any, description: str | None) -> dict[str, Any]:
    option: dict[str, Any] = {key: value}
    if description is not None:
        option["description"] = description
    return option


# ---------------------------------------------------------------------------
# Answer side.
# ---------------------------------------------------------------------------

AnswerType = Literal["predicate", "choice", "score", "refusal"]


def _unit_interval(value: Any, label: str) -> float:
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_number:
        raise ValueError(f"{label} must be a number, got {value!r}")
    as_float = float(value)
    if not 0.0 <= as_float <= 1.0:
        raise ValueError(f"{label} must be within [0.0, 1.0], got {as_float!r}")
    return as_float


@dataclass(frozen=True)
class Probability:
    """One entry of a calibrated distribution over a question's options."""

    #: Choice value, or the 0-based level index for a score question.
    value: str | bool | int | None
    #: Level label for a score question; ``None`` for a choice question.
    label: str | None
    probability: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "probability", _unit_interval(self.probability, "probability")
        )


@dataclass
class DecisionAnswer:
    """One calibrated answer, normalized across backends."""

    name: str
    type: AnswerType
    probability: float | None = None
    choice: str | bool | None = None
    score: float | None = None
    confidence: float | None = None
    probabilities: tuple[Probability, ...] = ()
    backend: str = ""

    def __post_init__(self) -> None:
        if self.probability is not None:
            self.probability = _unit_interval(self.probability, "probability")
        if self.confidence is not None:
            self.confidence = _unit_interval(self.confidence, "confidence")

    @property
    def refused(self) -> bool:
        return self.type == "refusal"

    def as_record(self) -> dict[str, Any]:
        """JSON-serializable answer record. ``decision_id`` is added by the driver."""
        return {
            "name": self.name,
            "type": self.type,
            "probability": self.probability,
            "choice": self.choice,
            "score": self.score,
            "confidence": self.confidence,
            "probabilities": [
                {"value": p.value, "label": p.label, "probability": p.probability}
                for p in self.probabilities
            ],
            "refused": self.refused,
        }


@dataclass
class DecisionResult:
    """What one :meth:`DecisionBackend.ask` call returned."""

    answers: tuple[DecisionAnswer, ...]
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class DecisionBackend(ABC):
    """A device that turns (context, questions) into calibrated answers.

    ``ask`` is ``async`` because the harness is async end to end and a remote
    backend does network I/O. A synchronous engine implements it as an ``async
    def`` that does its work in :func:`asyncio.to_thread`.
    """

    #: Short backend id, recorded in the trace and on the span.
    name: str = "decisions"

    @abstractmethod
    async def ask(
        self, context: str, questions: Sequence[DecisionQuestion]
    ) -> DecisionResult:
        """Answer every question about ``context``, in order."""


# ---------------------------------------------------------------------------
# Strict JSON and the shared response parser.
#
# Every backend parses through here, so a replayed body is validated exactly
# like a live one and a replay run exercises the real parser.
# ---------------------------------------------------------------------------


def _reject_constant(name: str) -> Any:
    raise ValueError(f"response contains the non-JSON constant {name}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"response contains duplicate JSON key {key!r}")
        seen.add(key)
    return dict(pairs)


def loads_strict(raw: bytes | str) -> Any:
    """Parse a provider body, rejecting duplicate keys, ``NaN`` and ``Infinity``."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            raise DecisionBackendError(
                "response body is not valid UTF-8"
            ) from None
    try:
        return json.loads(
            raw, parse_constant=_reject_constant, object_pairs_hook=_no_duplicate_keys
        )
    except ValueError as exc:  # JSONDecodeError is a ValueError
        raise DecisionBackendError(f"response body is not valid JSON: {exc}") from None


def _require_int(value: Any, label: str) -> int:
    # ``type(value) is int`` deliberately: True would pass isinstance(_, int).
    if type(value) is not int:
        raise DecisionBackendError(f"{label} must be an integer, got {value!r}")
    if value < 0:
        raise DecisionBackendError(f"{label} must be non-negative, got {value!r}")
    return value


def _require_unit(value: Any, label: str) -> float:
    try:
        return _unit_interval(value, label)
    except ValueError as exc:
        raise DecisionBackendError(str(exc)) from None


def _answer_from_payload(
    payload: dict[str, Any],
    question: DecisionQuestion,
    backend: str,
) -> DecisionAnswer:
    kind = payload.get("type")
    if kind == "refusal":
        return DecisionAnswer(name=question.name, type="refusal", backend=backend)

    if kind == "predicate":
        return DecisionAnswer(
            name=question.name,
            type="predicate",
            probability=_require_unit(
                payload.get("probability"), f"answer {question.name!r}: probability"
            ),
            backend=backend,
        )

    if kind == "choice":
        value = payload.get("choice")
        if not isinstance(value, (str, bool)):
            raise DecisionBackendError(
                f"answer {question.name!r}: choice must be a string or boolean, "
                f"got {value!r}"
            )
        entries = payload.get("probabilities")
        if not isinstance(entries, list) or len(entries) < 2:
            raise DecisionBackendError(
                f"answer {question.name!r}: a choice answer needs at least 2 "
                f"probability entries"
            )
        return DecisionAnswer(
            name=question.name,
            type="choice",
            choice=value,
            confidence=_require_unit(
                payload.get("confidence"), f"answer {question.name!r}: confidence"
            ),
            probabilities=tuple(
                Probability(
                    value=e.get("value") if isinstance(e, dict) else None,
                    label=None,
                    probability=_require_unit(
                        e.get("probability") if isinstance(e, dict) else None,
                        f"answer {question.name!r}: probabilities[].probability",
                    ),
                )
                for e in entries
            ),
            backend=backend,
        )

    if kind == "score":
        raw_score = payload.get("score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise DecisionBackendError(
                f"answer {question.name!r}: score must be a number, got {raw_score!r}"
            )
        entries = payload.get("probabilities") or []
        if not isinstance(entries, list):
            raise DecisionBackendError(
                f"answer {question.name!r}: probabilities must be a list"
            )
        return DecisionAnswer(
            name=question.name,
            type="score",
            score=float(raw_score),
            confidence=_require_unit(
                payload.get("confidence"), f"answer {question.name!r}: confidence"
            ),
            probabilities=tuple(
                Probability(
                    value=e.get("value") if isinstance(e, dict) else None,
                    label=e.get("label") if isinstance(e, dict) else None,
                    probability=_require_unit(
                        e.get("probability") if isinstance(e, dict) else None,
                        f"answer {question.name!r}: probabilities[].probability",
                    ),
                )
                for e in entries
            ),
            backend=backend,
        )

    raise DecisionBackendError(
        f"answer {question.name!r}: unknown answer type {kind!r}"
    )


def parse_decision_response(
    body: Any,
    questions: Sequence[DecisionQuestion],
    backend: str,
) -> DecisionResult:
    """Validate a provider response body and map it onto harness types.

    Enforces the five invariants every decisions backend must satisfy: one
    answer per question in order, matching names, matching types (or a
    refusal), a non-blank resolved model, and non-negative integer token usage.
    """
    if not isinstance(body, dict):
        raise DecisionBackendError(
            f"response must be a JSON object, got {type(body).__name__}"
        )

    answers = body.get("answers")
    if not isinstance(answers, list):
        raise DecisionBackendError("response is missing an 'answers' list")
    if len(answers) != len(questions):
        raise DecisionBackendError(
            f"response has {len(answers)} answers for {len(questions)} questions"
        )

    parsed: list[DecisionAnswer] = []
    for payload, question in zip(answers, questions):
        if not isinstance(payload, dict):
            raise DecisionBackendError(
                f"answer for {question.name!r} is not a JSON object"
            )
        if payload.get("name") != question.name:
            raise DecisionBackendError(
                f"answer name {payload.get('name')!r} does not match question "
                f"{question.name!r}"
            )
        kind = payload.get("type")
        if kind != question.type and kind != "refusal":
            raise DecisionBackendError(
                f"answer {question.name!r} has type {kind!r}; expected "
                f"{question.type!r} or 'refusal'"
            )
        parsed.append(_answer_from_payload(payload, question, backend))

    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise DecisionBackendError("response 'model' must be a non-blank string")

    usage_payload = body.get("usage")
    if not isinstance(usage_payload, dict):
        raise DecisionBackendError("response is missing a 'usage' object")
    usage: dict[str, Any] = dict(usage_payload)
    for key in ("input_tokens", "output_tokens"):
        if key not in usage:
            raise DecisionBackendError(f"response usage is missing {key!r}")
        usage[key] = _require_int(usage[key], f"usage.{key}")

    raw = {k: v for k, v in body.items() if k not in ("answers", "model", "usage")}
    return DecisionResult(
        answers=tuple(parsed), model=model, usage=usage, raw=raw
    )


# ---------------------------------------------------------------------------
# Telemetry.
# ---------------------------------------------------------------------------


@contextmanager
def decision_span(
    backend: str = "",
    question_count: int = 0,
) -> Iterator[SpanMeasurement]:
    """Measure one turn's decision call: wall + CPU time, with decision vocabulary.

    Decision naming over the generic
    :func:`~traced_harness.telemetry.measured_span`, the same relationship
    :func:`traced_harness.memory.consolidation_span` has to it. The caller sets
    ``decision.refusal_count`` on the yielded handle once answers are back.
    """
    with measured_span(
        DECISION_SPAN,
        {
            DECISION_BACKEND_ATTR: backend,
            DECISION_QUESTION_COUNT_ATTR: question_count,
            DECISION_REFUSAL_COUNT_ATTR: 0,
        },
        tracer_name=DECISION_TRACER_NAME,
    ) as measurement:
        yield measurement


def _active_otel_ids() -> dict[str, str] | None:
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return {
        "trace_id": format(context.trace_id, "032x"),
        "span_id": format(context.span_id, "016x"),
    }


# ---------------------------------------------------------------------------
# Context and identity.
# ---------------------------------------------------------------------------


def build_turn_context(turn: TraceTurn) -> str:
    """Render the text a turn's questions are asked about.

    The template is fixed so annotations are reproducible byte for byte: tool
    parameters are key-sorted, tool output is truncated at
    :data:`TOOL_OUTPUT_LIMIT` characters, the tools section is omitted entirely
    when the turn called none, and there is no trailing newline.
    """
    sections = [
        f"### Input\n{turn.input}",
        f"### Output\n{turn.actual_output}",
    ]
    if turn.tools_called:
        lines = ["### Tools called"]
        for tool in turn.tools_called:
            params = json.dumps(tool.input_parameters, sort_keys=True)
            lines.append(f"- {tool.name}({params}) -> {tool.output[:TOOL_OUTPUT_LIMIT]}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def decision_id(trace_id: str, turn_index: int, question_name: str) -> str:
    """Stable 16-hex-char id for one answer. No UUIDs, no timestamps."""
    seed = f"{trace_id}:{turn_index}:{question_name}".encode()
    return hashlib.sha256(seed).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Questions from JSON Schema and from file.
# ---------------------------------------------------------------------------


def questions_from_json_schema(schema: dict[str, Any]) -> list[DecisionQuestion]:
    """Map an object JSON Schema's ``properties`` onto questions, in order.

    This is the "same json_schema in" contract decision backends share. An
    unsupported property type raises rather than being dropped: a missing
    question would silently break the answer-count parity check.
    """
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise DecisionsConfigError(
            "questions schema must be a JSON Schema object with \"type\": \"object\""
        )
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise DecisionsConfigError("questions schema has no 'properties' to map")

    questions: list[DecisionQuestion] = []
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            raise DecisionsConfigError(f"property {name!r} is not a schema object")
        instructions = str(prop.get("description") or "").strip()
        if not instructions:
            raise DecisionsConfigError(
                f"property {name!r} has no 'description' to use as instructions"
            )
        prop_type = prop.get("type")

        if prop_type == "boolean":
            questions.append(
                DecisionQuestion(name=name, instructions=instructions, type="predicate")
            )
            continue

        if prop_type == "string" and isinstance(prop.get("enum"), list):
            enum = prop["enum"]
            try:
                questions.append(
                    DecisionQuestion(
                        name=name,
                        instructions=instructions,
                        type="choice",
                        choices=tuple(Choice(value=value) for value in enum),
                    )
                )
            except ValueError as exc:
                raise DecisionsConfigError(f"property {name!r}: {exc}") from None
            continue

        if (
            prop_type == "integer"
            and type(prop.get("minimum")) is int
            and type(prop.get("maximum")) is int
        ):
            low, high = prop["minimum"], prop["maximum"]
            if high <= low:
                raise DecisionsConfigError(
                    f"property {name!r}: 'maximum' must exceed 'minimum'"
                )
            labels = [str(i) for i in range(low, high + 1)]
            custom = prop.get("x-levels")
            if custom is not None:
                if not isinstance(custom, list) or len(custom) != len(labels):
                    raise DecisionsConfigError(
                        f"property {name!r}: 'x-levels' must list exactly "
                        f"{len(labels)} labels"
                    )
                labels = [str(label) for label in custom]
            try:
                questions.append(
                    DecisionQuestion(
                        name=name,
                        instructions=instructions,
                        type="score",
                        levels=tuple(Level(label=label) for label in labels),
                    )
                )
            except ValueError as exc:
                raise DecisionsConfigError(f"property {name!r}: {exc}") from None
            continue

        raise DecisionsConfigError(
            f"property {name!r} has unsupported schema type {prop_type!r}; "
            f"supported: boolean, string+enum, integer with minimum and maximum"
        )
    return questions


def _choice_from_entry(entry: Any) -> Choice:
    if isinstance(entry, dict):
        if "value" not in entry:
            raise DecisionsConfigError("a choice object needs a 'value' key")
        return Choice(value=entry["value"], description=entry.get("description"))
    return Choice(value=entry)


def _level_from_entry(entry: Any) -> Level:
    if isinstance(entry, dict):
        if "label" not in entry:
            raise DecisionsConfigError("a level object needs a 'label' key")
        return Level(label=str(entry["label"]), description=entry.get("description"))
    return Level(label=str(entry))


def _question_from_entry(entry: Any) -> DecisionQuestion:
    if not isinstance(entry, dict):
        raise DecisionsConfigError(
            f"each question must be an object, got {type(entry).__name__}"
        )
    choices = entry.get("choices")
    levels = entry.get("levels")
    try:
        return DecisionQuestion(
            name=str(entry.get("name", "")),
            instructions=str(entry.get("instructions", "")),
            type=entry.get("type", "predicate"),
            choices=(
                tuple(_choice_from_entry(c) for c in choices)
                if choices is not None
                else None
            ),
            levels=(
                tuple(_level_from_entry(lv) for lv in levels)
                if levels is not None
                else None
            ),
        )
    except ValueError as exc:
        raise DecisionsConfigError(str(exc)) from None
    except TypeError as exc:
        raise DecisionsConfigError(f"malformed question entry: {exc}") from None


def questions_from_document(document: Any) -> list[DecisionQuestion]:
    """Route a parsed questions document to the native or JSON-Schema reader."""
    if isinstance(document, list):
        if not document:
            raise DecisionsConfigError("questions file contains an empty list")
        return [_question_from_entry(entry) for entry in document]
    if isinstance(document, dict) and document.get("type") == "object":
        return questions_from_json_schema(document)
    raise DecisionsConfigError(
        "questions file must be a list of questions or a JSON Schema object "
        "with \"type\": \"object\""
    )


def load_questions_file(path: str | Path) -> list[DecisionQuestion]:
    """Read ``--questions FILE``: a native question list or a JSON Schema."""
    file_path = Path(path)
    if not file_path.is_file():
        raise DecisionsConfigError(f"questions file not found: {file_path}")
    try:
        document = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise DecisionsConfigError(f"questions file is unreadable: {exc}") from None
    except json.JSONDecodeError as exc:
        raise DecisionsConfigError(
            f"questions file is not valid JSON: {exc}"
        ) from None
    return questions_from_document(document)


# ---------------------------------------------------------------------------
# The offline backend.
# ---------------------------------------------------------------------------


class ReplayDecisionBackend(DecisionBackend):
    """Answer from recorded provider response bodies, in order.

    A first-class feature rather than test scaffolding: deterministic
    re-annotation and keyless CI or demo runs. ``path`` holds a JSON list of
    bodies shaped exactly as a live provider would return, and each one is run
    through :func:`parse_decision_response` — the same validation a live
    response gets.
    """

    name = "replay"

    def __init__(self, path: str | Path, strict: bool = True) -> None:
        file_path = Path(path)
        if not file_path.is_file():
            raise DecisionsConfigError(f"replay file not found: {file_path}")
        try:
            bodies = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise DecisionsConfigError(f"replay file is unreadable: {exc}") from None
        except json.JSONDecodeError as exc:
            raise DecisionsConfigError(
                f"replay file is not valid JSON: {exc}"
            ) from None
        if not isinstance(bodies, list):
            raise DecisionsConfigError(
                "replay file must contain a list of recorded response bodies"
            )
        self.path = file_path
        self.strict = strict
        self._bodies = bodies
        self._cursor = 0

    async def ask(
        self, context: str, questions: Sequence[DecisionQuestion]
    ) -> DecisionResult:
        if self._cursor >= len(self._bodies):
            if self.strict:
                raise DecisionBackendError(
                    f"replay file {self.path.name} is exhausted after "
                    f"{len(self._bodies)} response(s)"
                )
            self._cursor = 0
        body = self._bodies[self._cursor]
        self._cursor += 1
        return parse_decision_response(body, questions, self.name)


def build_decision_backend(
    name: str,
    model: str | None = None,
    replay_file: str | Path | None = None,
) -> DecisionBackend:
    """CLI backend factory. Imports the OpenAI backend lazily, never at import time."""
    if name == "replay":
        if not replay_file:
            raise DecisionsConfigError(
                "the replay backend requires --replay-file FILE"
            )
        return ReplayDecisionBackend(replay_file)
    if name == "openai":
        from traced_harness.decisions_openai import (
            DEFAULT_DECISIONS_MODEL,
            OpenAIDecisionsBackend,
        )

        return OpenAIDecisionsBackend(model=model or DEFAULT_DECISIONS_MODEL)
    raise DecisionsConfigError(
        f"unknown decisions backend {name!r}; expected 'openai' or 'replay'"
    )


# ---------------------------------------------------------------------------
# The driver.
# ---------------------------------------------------------------------------


def _capped_raw(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        encoded = json.dumps(raw)
    except (TypeError, ValueError):
        return {"_truncated": True}
    if len(encoded) > RAW_PASSTHROUGH_LIMIT:
        return {"_truncated": True}
    return raw


def _base_record(
    backend: str, trace_id: str, turn: TraceTurn
) -> dict[str, Any]:
    return {
        "schema_version": DECISIONS_SCHEMA_VERSION,
        "backend": backend,
        "trace_id": trace_id,
        "turn_index": turn.index,
        "session_id": turn.additional_metadata.get("session_id"),
    }


def _read_raw_lines(source: Path) -> list[dict[str, Any]]:
    """Re-read a trace's JSON lines verbatim.

    :class:`~traced_harness.eval.TraceTurn` is lossy for ``additional_metadata``
    subkeys it does not model, so the annotated output is built from these raw
    objects rather than from the parsed turns.
    """
    raw_lines: list[dict[str, Any]] = []
    with open(source, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw_lines.append(json.loads(line))
    return raw_lines


def _write_raw_lines(destination: Path, raw_lines: list[dict[str, Any]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(raw_line) + "\n" for raw_line in raw_lines)


async def annotate_trace(
    trace_file: str | Path,
    questions: Sequence[DecisionQuestion],
    backend: DecisionBackend,
    out_file: str | Path | None = None,
    context_builder: Callable[[TraceTurn], str] = build_turn_context,
    on_error: Literal["record", "raise"] = "record",
) -> Path:
    """Annotate every turn of a recorded JSONL trace, writing a new file.

    The input is never modified. Turns are processed sequentially — determinism
    and rate-limit safety beat speed here. With ``on_error="record"`` a backend
    failure is written into that turn's slot and processing continues; with
    ``"raise"`` it propagates. A :class:`DecisionsConfigError` always
    propagates: it means the run was misconfigured, not that one turn failed.
    """
    if on_error not in ("record", "raise"):
        raise DecisionsConfigError(
            f"on_error must be 'record' or 'raise', got {on_error!r}"
        )
    if not questions:
        raise DecisionsConfigError("no questions supplied")

    source = Path(trace_file)
    turns = load_trace(source)
    raw_lines = _read_raw_lines(source)
    if len(raw_lines) != len(turns):
        raise DecisionsConfigError(
            f"trace {source.name}: read {len(raw_lines)} raw lines but "
            f"{len(turns)} turns"
        )

    destination = Path(out_file) if out_file else source.with_suffix(".decided.jsonl")
    trace_id = source.stem
    overwritten = 0

    for raw_line, turn in zip(raw_lines, turns):
        context = context_builder(turn)
        measurement: SpanMeasurement | None = None
        record: dict[str, Any]
        try:
            with decision_span(
                backend=backend.name, question_count=len(questions)
            ) as handle_measurement:
                measurement = handle_measurement
                otel = _active_otel_ids()
                result = await backend.ask(context, questions)
                handle_measurement.attributes[DECISION_REFUSAL_COUNT_ATTR] = sum(
                    1 for answer in result.answers if answer.refused
                )
        except DecisionBackendError as exc:
            if on_error == "raise":
                raise
            record = _base_record(backend.name, trace_id, turn)
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["measure"] = (
                measurement.as_record() if measurement is not None else {}
            )
            logger.warning(
                "decision backend failed on turn %s of %s: %s",
                turn.index,
                source.name,
                type(exc).__name__,
            )
        else:
            record = _base_record(backend.name, trace_id, turn)
            record["model"] = result.model
            record["answers"] = [
                {
                    "decision_id": decision_id(trace_id, turn.index, answer.name),
                    **answer.as_record(),
                }
                for answer in result.answers
            ]
            record["usage"] = dict(result.usage)
            record["measure"] = measurement.as_record() if measurement else {}
            if otel is not None:
                record["otel"] = otel
            record["raw"] = _capped_raw(result.raw)

        metadata = raw_line.setdefault("additional_metadata", {})
        if DECISIONS_METADATA_KEY in metadata:
            overwritten += 1
        metadata[DECISIONS_METADATA_KEY] = record

    if overwritten:
        logger.info(
            "overwrote %s pre-existing %r record(s) in %s",
            overwritten,
            DECISIONS_METADATA_KEY,
            source.name,
        )

    _write_raw_lines(destination, raw_lines)
    return destination

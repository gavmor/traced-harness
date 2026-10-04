"""Data models for whole-trace qualitative evaluation and failure taxonomies."""

from __future__ import annotations

import datetime
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field


class StepKind(StrEnum):
    USER_INPUT = "user_input"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    COMPLETION = "completion"


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


class Step(BaseModel):
    step_index: int
    kind: StepKind
    name: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    text: str = ""
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))


class Turn(BaseModel):
    turn_index: int
    input_text: str
    steps: list[Step] = Field(default_factory=list)
    output_text: str = ""
    verdict: Verdict | None = None


class Trace(BaseModel):
    trace_id: str
    session_id: str
    agent_name: str = "dspy_react"
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))
    turns: list[Turn] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaxonomyCategory(BaseModel):
    key: str
    name: str
    description: str
    examples: list[str] = Field(default_factory=list)


class Annotation(BaseModel):
    annotation_id: str
    trace_id: str
    turn_index: int
    verdict: Verdict
    first_failure_step_index: int | None = None
    open_code: str = ""
    failure_modes: dict[str, bool] = Field(default_factory=dict)
    annotator: str = "human"
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))

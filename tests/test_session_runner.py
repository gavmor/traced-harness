"""Tests for the multi-session SessionRunner orchestration (fake executor).

The runner is peripheral-agnostic: these tests supply a plain fake satisfying
the ``PeripheralLifecycle`` protocol, with no base class from the harness.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traced_harness.session_runner import (
    PeripheralLifecycle,
    Scenario,
    Session,
    SessionRunner,
)


class _FakeLifecycle:
    """Structurally satisfies PeripheralLifecycle — no harness base class."""

    name = "fake"

    def __init__(self) -> None:
        self.actions: list[str] = []

    def setup(self, workspace_dir: Path) -> dict[str, Any]:
        self.actions.append("setup")
        return {"store_paths": []}

    def between_sessions(self) -> None:
        self.actions.append("between_sessions")

    def teardown(self) -> None:
        self.actions.append("teardown")


@dataclass
class _FakeTurn:
    prompt: str
    output: str
    session_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tools_called: list[Any] = field(default_factory=list)
    timestamp: str = "2026-01-01T00:00:00Z"


def _executor():
    async def _exec(prompt: str, session_id: str, peripheral: str) -> _FakeTurn:
        return _FakeTurn(
            prompt=prompt,
            output=f"ans:{prompt}",
            session_id=session_id,
            metadata={"injection": {"overhead_tokens": 10}, "generated_tokens": 5},
        )

    return _exec


def _scenario() -> Scenario:
    return Scenario(
        name="belief_revision",
        sessions=[
            Session("s1", ["I drive a Honda", "ok"]),
            Session("s2", ["Actually I drive a Tesla now"]),
            Session("s3", ["What do I drive?"]),
        ],
    )


def test_fake_satisfies_protocol_structurally():
    assert isinstance(_FakeLifecycle(), PeripheralLifecycle)


def test_runner_orchestrates_sessions_and_between_hooks(tmp_path):
    lifecycle = _FakeLifecycle()
    runner = SessionRunner(
        _executor(), tmp_path, lifecycle=lifecycle, metadata_key="memory"
    )
    result = asyncio.run(runner.run_scenario(_scenario()))

    assert lifecycle.actions.count("setup") == 1
    # The between-sessions hook fires between sessions only: N-1 = 2 times.
    assert lifecycle.actions.count("between_sessions") == 2
    assert len(result.between_session_records) == 2

    # One JSONL line per turn (2 + 1 + 1 = 4).
    lines = result.trace_file.read_text().strip().splitlines()
    assert len(lines) == 4

    entries = [json.loads(line) for line in lines]
    session_ids = [e["additional_metadata"]["session_id"] for e in entries]
    assert session_ids == ["s1", "s1", "s2", "s3"]

    # The record is attached to the FIRST turn of the session it precedes.
    s1_first = entries[0]["additional_metadata"]["memory"]
    s2_first = entries[2]["additional_metadata"]["memory"]
    s3_first = entries[3]["additional_metadata"]["memory"]
    assert "between_sessions" not in s1_first
    assert "between_sessions" in s2_first
    assert "between_sessions" in s3_first


def test_metadata_key_is_caller_supplied(tmp_path):
    """The harness writes under whatever slot the caller names."""
    runner = SessionRunner(
        _executor(), tmp_path, lifecycle=_FakeLifecycle(), metadata_key="widget"
    )
    result = asyncio.run(runner.run_scenario(_scenario()))
    first = json.loads(result.trace_file.read_text().splitlines()[0])
    assert "widget" in first["additional_metadata"]
    assert "memory" not in first["additional_metadata"]
    # Turn-level domain measurements pass through uninterpreted.
    assert first["additional_metadata"]["widget"]["generated_tokens"] == 5


def test_runner_works_without_a_lifecycle(tmp_path):
    """A plain multi-session run needs no peripheral at all."""
    runner = SessionRunner(_executor(), tmp_path)
    result = asyncio.run(runner.run_scenario(_scenario()))
    assert len(result.turn_results) == 4
    assert result.between_session_records == []
    assert result.trace_file.exists()


def test_reset_suite_tears_down(tmp_path):
    lifecycle = _FakeLifecycle()
    runner = SessionRunner(_executor(), tmp_path, lifecycle=lifecycle)
    asyncio.run(runner.run_scenario(_scenario()))
    runner.reset_suite()
    assert lifecycle.actions[-1] == "teardown"


def test_distinct_session_ids_isolate_history(tmp_path):
    """Each session_id is distinct so in-context history cannot leak across."""
    seen: list[str] = []

    async def _exec(prompt, session_id, peripheral):
        seen.append(session_id)
        return _FakeTurn(prompt, "ok", session_id)

    runner = SessionRunner(_exec, tmp_path, lifecycle=_FakeLifecycle())
    asyncio.run(runner.run_scenario(_scenario()))
    assert set(seen) == {"s1", "s2", "s3"}

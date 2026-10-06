"""Tests for the multi-session SessionRunner orchestration (fake executor)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traced_harness.plugins import MemoryPluginAdapter, MemoryToolContract
from traced_harness.session_runner import Scenario, Session, SessionRunner


class _FakeAdapter(MemoryPluginAdapter):
    name = "fake"

    def setup(self, workspace_dir: Path) -> dict[str, Any]:
        self.store_paths = []
        self._record("setup", str(workspace_dir))
        return {"env": {}, "contract": self.contract(), "store_paths": []}

    def trigger_consolidation(self) -> None:
        self._record("consolidate", None)

    def teardown(self) -> None:
        self._record("teardown", None)

    def contract(self) -> MemoryToolContract:
        return MemoryToolContract(provider="fake", tools=["recall"])


@dataclass
class _FakeTurn:
    prompt: str
    output: str
    session_id: str
    memory: dict[str, Any] = field(default_factory=dict)
    tools_called: list[Any] = field(default_factory=list)
    timestamp: str = "2026-01-01T00:00:00Z"


def _executor():
    async def _exec(prompt: str, session_id: str, provider: str) -> _FakeTurn:
        return _FakeTurn(
            prompt=prompt,
            output=f"ans:{prompt}",
            session_id=session_id,
            memory={"injection": {"overhead_tokens": 10}, "generated_tokens": 5},
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


def test_runner_orchestrates_sessions_and_consolidation(tmp_path):
    adapter = _FakeAdapter()
    runner = SessionRunner(adapter, _executor(), tmp_path)
    result = asyncio.run(runner.run_scenario(_scenario()))

    kinds = [a[0] for a in adapter.actions]
    assert kinds.count("setup") == 1
    # Consolidation fires between sessions only: N-1 = 2 times.
    assert kinds.count("consolidate") == 2
    assert len(result.consolidation_records) == 2

    # One JSONL line per turn (2 + 1 + 1 = 4).
    lines = result.trace_file.read_text().strip().splitlines()
    assert len(lines) == 4

    entries = [json.loads(line) for line in lines]
    session_ids = [e["additional_metadata"]["session_id"] for e in entries]
    assert session_ids == ["s1", "s1", "s2", "s3"]

    # Consolidation is attached to the FIRST turn of the session it precedes.
    s1_first = entries[0]["additional_metadata"]["memory"]
    s2_first = entries[2]["additional_metadata"]["memory"]
    s3_first = entries[3]["additional_metadata"]["memory"]
    assert "consolidation" not in s1_first
    assert "consolidation" in s2_first
    assert "consolidation" in s3_first


def test_reset_suite_tears_down_and_purges(tmp_path):
    adapter = _FakeAdapter()
    runner = SessionRunner(adapter, _executor(), tmp_path)
    asyncio.run(runner.run_scenario(_scenario()))
    runner.reset_suite()
    assert adapter.actions[-1][0] == "teardown"


def test_distinct_session_ids_isolate_history(tmp_path):
    """Each session_id is distinct so in-context history cannot leak across."""
    adapter = _FakeAdapter()
    seen: list[str] = []

    async def _exec(prompt, session_id, provider):
        seen.append(session_id)
        return _FakeTurn(prompt, "ok", session_id)

    runner = SessionRunner(adapter, _exec, tmp_path)
    asyncio.run(runner.run_scenario(_scenario()))
    assert set(seen) == {"s1", "s2", "s3"}

"""Multi-session scenario runner for memory-provider evaluations.

Standard harness turns are isolated. Memory benchmarks instead need an ordered
sequence of *sessions* with an inter-session consolidation ("sleep") pass:

    Session 1 (Fact Ingress) --> consolidate --> Session 2 (Fact Revision)
                                                      |
                                                 consolidate
                                                      |
                                               Session 3 (Probe Turn)

Each session uses a distinct ``session_id`` so the agent's in-context history
does NOT carry across sessions (Agno keys history by ``session_id``). That is
deliberate: cross-session recall must come from the *memory provider*, not from
conversation history still sitting in the context window — otherwise the
benchmark measures the model's context, not the plugin.

The runner is decoupled from the agent/model: it takes an async
``turn_executor`` so orchestration (setup once, consolidate between sessions,
teardown + purge on suite reset, enriched trace logging) is unit-testable with
a fake executor and a fake adapter. :func:`make_agno_turn_executor` wires the
real Agno agent when you want a live run.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from traced_harness.plugins import MemoryPluginAdapter
from traced_harness.telemetry import consolidation_span


class TurnLike(Protocol):
    """Structural type the runner needs from a turn result."""

    prompt: str
    output: str
    session_id: str


# (prompt, session_id, provider) -> awaitable turn result
TurnExecutor = Callable[[str, str, str], Awaitable[Any]]


@dataclass
class Session:
    """One session: an ordered list of prompts sharing a ``session_id``."""

    session_id: str
    turns: list[str]


@dataclass
class Scenario:
    """An ordered sequence of sessions defining a memory benchmark."""

    name: str
    sessions: list[Session]


@dataclass
class SessionRunResult:
    scenario: str
    provider: str
    trace_file: Path
    turn_results: list[Any] = field(default_factory=list)
    consolidation_records: list[dict[str, Any]] = field(default_factory=list)


class SessionRunner:
    """Drives a memory provider across multi-session scenarios."""

    def __init__(
        self,
        adapter: MemoryPluginAdapter,
        turn_executor: TurnExecutor,
        workspace_dir: str | Path,
        trace_dir: str | Path | None = None,
    ) -> None:
        self.adapter = adapter
        self.turn_executor = turn_executor
        self.workspace_dir = Path(workspace_dir)
        self.trace_dir = Path(trace_dir) if trace_dir else self.workspace_dir / "sessions"
        self._store_paths: list[str | Path] = []

    async def run_scenario(self, scenario: Scenario) -> SessionRunResult:
        """Run every session in order, consolidating between sessions."""
        provider = self.adapter.name
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        setup_info = self.adapter.setup(self.workspace_dir)
        self._store_paths = list(setup_info.get("store_paths", []))

        trace_file = self.trace_dir / f"trace_{scenario.name}_{provider}.jsonl"
        if trace_file.exists():
            trace_file.unlink()

        result = SessionRunResult(
            scenario=scenario.name, provider=provider, trace_file=trace_file
        )
        pending_consolidation: dict[str, Any] | None = None
        last_index = len(scenario.sessions) - 1

        for i, session in enumerate(scenario.sessions):
            for turn_no, prompt in enumerate(session.turns):
                turn = await self.turn_executor(
                    prompt, session.session_id, provider
                )
                result.turn_results.append(turn)
                # Attach any pending inter-session consolidation to the first
                # turn of the session it precedes (the "sleep before next turn").
                attach = pending_consolidation if turn_no == 0 else None
                self._log_turn(
                    trace_file, turn, session.session_id, provider, attach
                )
                if turn_no == 0:
                    pending_consolidation = None

            if i < last_index:
                with consolidation_span(provider, self._store_paths) as meas:
                    self.adapter.trigger_consolidation()
                record = meas.as_record()
                result.consolidation_records.append(record)
                pending_consolidation = record

        return result

    def reset_suite(self) -> None:
        """Ephemeral reset between benchmark suites.

        Tears the provider down and purges its SQLite caches / vector indices so
        the next suite starts from an empty store. Trace files are preserved as
        results.
        """
        self.adapter.teardown()
        MemoryPluginAdapter._purge(*self._store_paths)
        self._store_paths = []

    # -- internal ---------------------------------------------------------
    @staticmethod
    def _log_turn(
        trace_file: Path,
        turn: Any,
        session_id: str,
        provider: str,
        consolidation: dict[str, Any] | None,
    ) -> None:
        tools = []
        for t in getattr(turn, "tools_called", []) or []:
            tools.append(
                {
                    "name": getattr(t, "name", ""),
                    "input_parameters": getattr(t, "input_parameters", {}),
                    "output": getattr(t, "output", ""),
                }
            )
        memory: dict[str, Any] = dict(getattr(turn, "memory", {}) or {})
        memory["provider"] = provider
        memory["session_id"] = session_id
        if consolidation is not None:
            memory["consolidation"] = consolidation

        entry = {
            "input": getattr(turn, "prompt", ""),
            "actual_output": getattr(turn, "output", ""),
            "tools_called": tools,
            "additional_metadata": {
                "session_id": session_id,
                "timestamp": getattr(
                    turn,
                    "timestamp",
                    datetime.datetime.now(datetime.UTC).isoformat(),
                ),
                "agent": "traced_agno",
                "memory": memory,
            },
        }
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def make_agno_turn_executor(agent: Any) -> TurnExecutor:
    """Build a turn executor backed by a live Agno agent.

    Lazy-imports the agent module so ``session_runner`` stays importable (and
    unit-testable) without pulling the agno/model stack.
    """
    from traced_harness.agent import execute_turn

    async def _executor(prompt: str, session_id: str, provider: str) -> Any:
        return await execute_turn(prompt, agent, session_id)

    return _executor

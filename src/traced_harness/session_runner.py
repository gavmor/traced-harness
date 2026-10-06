"""Multi-session scenario runner for peripheral evaluations.

Standard harness turns are isolated. Some benchmarks instead need an ordered
sequence of *sessions* with a hook that runs between them:

    Session 1 --> between_sessions --> Session 2 --> between_sessions --> Session 3

Each session uses a distinct ``session_id`` so the agent's in-context history
does NOT carry across sessions (Agno keys history by ``session_id``). That is
deliberate: cross-session behaviour must come from the *peripheral under test*,
not from conversation history still sitting in the context window — otherwise
the benchmark measures the model's context rather than the peripheral.

The runner is agnostic about what the peripheral is. It needs only an object
satisfying :class:`PeripheralLifecycle` (setup / between_sessions / teardown)
and an async ``turn_executor``, so orchestration is unit-testable with a fake
executor and a fake lifecycle. :func:`make_agno_turn_executor` wires the real
Agno agent when you want a live run.

Domain vocabulary stays with the caller: ``metadata_key`` names the slot in
each turn's ``additional_metadata`` where the runner writes the lifecycle
record, and the between-sessions measurement is recorded under whatever span
name the caller supplies.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from traced_harness.telemetry import measured_span


class TurnLike(Protocol):
    """Structural type the runner needs from a turn result."""

    prompt: str
    output: str
    session_id: str


@runtime_checkable
class PeripheralLifecycle(Protocol):
    """What the runner needs from the peripheral under test.

    Any object with these members works — the harness defines no base class
    and knows nothing about what the peripheral actually does.
    """

    #: Short id for the peripheral, recorded in trace metadata.
    name: str

    def setup(self, workspace_dir: Path) -> dict[str, Any]:
        """Initialize state; may return ``{"store_paths": [...]}`` to measure."""
        ...

    def between_sessions(self) -> None:
        """Run between consecutive sessions (compaction, sleep passes, ...)."""
        ...

    def teardown(self) -> None:
        """Release resources."""
        ...


# (prompt, session_id, peripheral_name) -> awaitable turn result
TurnExecutor = Callable[[str, str, str], Awaitable[Any]]


@dataclass
class Session:
    """One session: an ordered list of prompts sharing a ``session_id``."""

    session_id: str
    turns: list[str]


@dataclass
class Scenario:
    """An ordered sequence of sessions defining a benchmark."""

    name: str
    sessions: list[Session]


@dataclass
class SessionRunResult:
    scenario: str
    peripheral: str
    trace_file: Path
    turn_results: list[Any] = field(default_factory=list)
    between_session_records: list[dict[str, Any]] = field(default_factory=list)


class SessionRunner:
    """Drives a peripheral across multi-session scenarios."""

    def __init__(
        self,
        turn_executor: TurnExecutor,
        workspace_dir: str | Path,
        trace_dir: str | Path | None = None,
        lifecycle: PeripheralLifecycle | None = None,
        metadata_key: str = "peripheral",
        between_sessions_span: str = "peripheral.between_sessions",
    ) -> None:
        self.turn_executor = turn_executor
        self.lifecycle = lifecycle
        self.workspace_dir = Path(workspace_dir)
        self.trace_dir = (
            Path(trace_dir) if trace_dir else self.workspace_dir / "sessions"
        )
        self.metadata_key = metadata_key
        self.between_sessions_span = between_sessions_span
        self._store_paths: list[str | Path] = []

    @property
    def peripheral_name(self) -> str:
        return getattr(self.lifecycle, "name", "") if self.lifecycle else ""

    async def run_scenario(self, scenario: Scenario) -> SessionRunResult:
        """Run every session in order, invoking the between-sessions hook."""
        name = self.peripheral_name
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        if self.lifecycle is not None:
            setup_info = self.lifecycle.setup(self.workspace_dir) or {}
            self._store_paths = list(setup_info.get("store_paths", []))

        trace_file = self.trace_dir / f"trace_{scenario.name}_{name or 'run'}.jsonl"
        if trace_file.exists():
            trace_file.unlink()

        result = SessionRunResult(
            scenario=scenario.name, peripheral=name, trace_file=trace_file
        )
        pending_record: dict[str, Any] | None = None
        last_index = len(scenario.sessions) - 1

        for i, session in enumerate(scenario.sessions):
            for turn_no, prompt in enumerate(session.turns):
                turn = await self.turn_executor(prompt, session.session_id, name)
                result.turn_results.append(turn)
                # Attach any pending between-sessions record to the first turn
                # of the session it precedes.
                attach = pending_record if turn_no == 0 else None
                self._log_turn(trace_file, turn, session.session_id, name, attach)
                if turn_no == 0:
                    pending_record = None

            if i < last_index and self.lifecycle is not None:
                with measured_span(
                    self.between_sessions_span,
                    {f"{self.metadata_key}.name": name},
                    watch_paths=self._store_paths,
                ) as meas:
                    self.lifecycle.between_sessions()
                record = meas.as_record()
                result.between_session_records.append(record)
                pending_record = record

        return result

    def reset_suite(self) -> None:
        """Ephemeral reset between benchmark suites.

        Tears the peripheral down. Purging any on-disk state is the
        peripheral's own responsibility — the harness does not know what its
        stores mean. Trace files are preserved as results.
        """
        if self.lifecycle is not None:
            self.lifecycle.teardown()
        self._store_paths = []

    # -- internal ---------------------------------------------------------
    def _log_turn(
        self,
        trace_file: Path,
        turn: Any,
        session_id: str,
        peripheral: str,
        between_sessions: dict[str, Any] | None,
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
        # A turn result may carry domain measurements under `.metadata`; the
        # harness passes them through without interpreting them.
        payload: dict[str, Any] = dict(getattr(turn, "metadata", {}) or {})
        payload["provider"] = peripheral
        payload["session_id"] = session_id
        if between_sessions is not None:
            payload["between_sessions"] = between_sessions

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
                self.metadata_key: payload,
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

    async def _executor(prompt: str, session_id: str, peripheral: str) -> Any:
        return await execute_turn(prompt, agent, session_id)

    return _executor

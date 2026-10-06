"""Trace loading and peripheral-health analysis for traced-harness.

Implements the first-failure evaluation principle from 'Evals for AI Engineers'
(Husain & Shankar, Ch. 3 & Ch. 8): multi-turn agent errors cascade forward.
Subsequent turns after the first observed failure are polluted by the upstream
failure and should be pruned to avoid cataloging symptoms or blaming downstream
components.

The harness deliberately has **no** evaluation-framework dependency: it loads
traces and reports peripheral health. Converting traces into DeepEval test
cases, and scoring them, belongs to the study repo that consumes this library
(see ``memory-provider-evals``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class ToolExecutionData:
    name: str
    input_parameters: dict[str, Any] = field(default_factory=dict)
    output: str = ""

    @property
    def has_error(self) -> bool:
        """Heuristic detection of tool error responses (4xx, 5xx, or error dicts)."""
        raw = self.output.strip()
        if not raw:
            return False
        # Check if output is JSON with error key
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "error" in parsed:
                return True
        except (json.JSONDecodeError, TypeError):
            pass
        lower = raw.lower()
        return "error" in lower and (
            "404" in lower or "500" in lower or "failed" in lower
        )

    @property
    def error_message(self) -> str | None:
        """Extract a readable error message if present."""
        raw = self.output.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "error" in parsed:
                return str(parsed["error"])
        except (json.JSONDecodeError, TypeError):
            pass
        if self.has_error:
            return raw[:150]
        return None


@dataclass
class TraceTurn:
    index: int
    input: str
    actual_output: str
    tools_called: list[ToolExecutionData] = field(default_factory=list)
    additional_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_peripheral_errors(self) -> bool:
        return any(t.has_error for t in self.tools_called)

    @property
    def tool_errors(self) -> list[ToolExecutionData]:
        return [t for t in self.tools_called if t.has_error]


@dataclass
class TraceReport:
    file_path: Path
    turns: list[TraceTurn]
    total_turns: int = 0
    evaluated_turns: int = 0
    total_tool_calls: int = 0
    failed_tool_calls: list[tuple[int, ToolExecutionData]] = field(default_factory=list)
    first_failure_turn: int | None = None
    polluted_turns_count: int = 0
    stop_at_first_failure: bool = True

    @property
    def error_rate(self) -> float:
        if self.total_tool_calls == 0:
            return 0.0
        return len(self.failed_tool_calls) / self.total_tool_calls


def load_trace(trace_file: str | Path) -> list[TraceTurn]:
    """Load a session JSONL trace file into a list of TraceTurns."""
    path = Path(trace_file)
    if not path.is_file():
        raise FileNotFoundError(f"Trace file not found: {path}")

    turns: list[TraceTurn] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_str = line.strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            tools: list[ToolExecutionData] = []
            for t in data.get("tools_called", []):
                tools.append(
                    ToolExecutionData(
                        name=t.get("name", ""),
                        input_parameters=t.get("input_parameters", {}),
                        output=t.get("output", ""),
                    )
                )
            turns.append(
                TraceTurn(
                    index=idx,
                    input=data.get("input", ""),
                    actual_output=data.get("actual_output", ""),
                    tools_called=tools,
                    additional_metadata=data.get("additional_metadata", {}),
                )
            )
    return turns


def evaluate_trace(
    trace_file: str | Path,
    stop_at_first_failure: bool = True,
) -> TraceReport:
    """Analyze a trace file for peripheral health, tool calls, and error rates.

    By default (stop_at_first_failure=True), stops at the first failing turn to avoid
    evaluating subsequent turns polluted by upstream cascading errors.
    """
    turns = load_trace(trace_file)
    total_tool_calls = 0
    failed_tool_calls: list[tuple[int, ToolExecutionData]] = []
    first_failure_turn: int | None = None
    polluted_turns_count = 0
    evaluated_turns = 0

    for turn in turns:
        if stop_at_first_failure and first_failure_turn is not None:
            polluted_turns_count += 1
            continue

        evaluated_turns += 1
        turn_has_error = False
        for tool in turn.tools_called:
            total_tool_calls += 1
            if tool.has_error:
                failed_tool_calls.append((turn.index, tool))
                turn_has_error = True

        if turn_has_error and first_failure_turn is None:
            first_failure_turn = turn.index

    return TraceReport(
        file_path=Path(trace_file),
        turns=turns,
        total_turns=len(turns),
        evaluated_turns=evaluated_turns,
        total_tool_calls=total_tool_calls,
        failed_tool_calls=failed_tool_calls,
        first_failure_turn=first_failure_turn,
        polluted_turns_count=polluted_turns_count,
        stop_at_first_failure=stop_at_first_failure,
    )


def display_trace_report(report: TraceReport, console: Console | None = None) -> None:
    """Render a Rich summary table of the trace evaluation."""
    con = console or Console()

    status_color = "red" if report.failed_tool_calls else "green"
    if report.failed_tool_calls:
        status_text = (
            f"[bold red]FIRST FAILURE AT TURN {report.first_failure_turn} "
            f"({len(report.failed_tool_calls)} error(s))[/]"
        )
    else:
        status_text = "[bold green]PASS (0 tool errors)[/]"

    header = (
        f"[bold]Trace Evaluation:[/] {report.file_path.name}\n"
        f"Total Turns: {report.total_turns} | Evaluated: {report.evaluated_turns}"
        + (
            f" ([yellow]{report.polluted_turns_count} polluted turns pruned[/])"
            if report.polluted_turns_count
            else ""
        )
        + f" | Tool Calls: {report.total_tool_calls} | "
        f"Status: {status_text} | Error Rate: {report.error_rate:.1%}"
    )
    con.print(Panel(header, border_style=status_color))

    table = Table(title="Turn-by-Turn Peripheral Telemetry", show_header=True)
    table.add_column("Turn", justify="right", style="cyan", width=6)
    table.add_column("Prompt / Input", style="white", max_width=42, overflow="fold")
    table.add_column("Tools Called", style="dim", max_width=32, overflow="fold")
    table.add_column("Status", justify="center", width=20)

    for turn in report.turns:
        is_polluted = (
            report.stop_at_first_failure
            and report.first_failure_turn is not None
            and turn.index > report.first_failure_turn
        )

        tool_names = ", ".join(t.name for t in turn.tools_called) or "None"
        if is_polluted:
            status = "[dim yellow]PRUNED (polluted)[/]"
        elif turn.has_peripheral_errors:
            err_tools = ", ".join(t.name for t in turn.tool_errors)
            status = f"[bold red]FAIL ({err_tools})[/]"
        else:
            status = "[green]OK[/]" if turn.tools_called else "[dim]NO_TOOLS[/]"

        prompt_snip = turn.input[:75] + "..." if len(turn.input) > 75 else turn.input
        table.add_row(str(turn.index), prompt_snip, tool_names, status)

    con.print(table)

    if report.failed_tool_calls:
        err_table = Table(
            title="[bold red]First Observed Failure: Peripheral Tool Error[/]",
            border_style="red",
            show_header=True,
        )
        err_table.add_column("Turn", style="cyan", width=6)
        err_table.add_column("Tool", style="yellow", width=22)
        err_table.add_column(
            "Input Arguments", style="white", max_width=35, overflow="fold"
        )
        err_table.add_column(
            "Tool Error / Payload", style="red", max_width=45, overflow="fold"
        )

        for turn_idx, tool in report.failed_tool_calls:
            err_table.add_row(
                str(turn_idx),
                tool.name,
                json.dumps(tool.input_parameters),
                tool.error_message or tool.output[:120],
            )
        con.print(err_table)

"""Trace evaluation and DeepEval import utilities for traced-harness.

Implements the first-failure evaluation principle from 'Evals for AI Engineers'
(Husain & Shankar, Ch. 3 & Ch. 8): multi-turn agent errors cascade forward.
Subsequent turns after the first observed failure are polluted by the upstream
failure and should be pruned to avoid cataloging symptoms or blaming downstream
components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from deepeval.test_case import LLMTestCase, ToolCall

    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False
    LLMTestCase = Any  # type: ignore
    ToolCall = Any  # type: ignore


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

    def to_deepeval(self) -> Any:
        """Convert this turn into a DeepEval LLMTestCase if deepeval is installed."""
        if not HAS_DEEPEVAL:
            raise ImportError(
                "deepeval is not installed. Install it with: uv add --group dev deepeval"
            )
        kwargs: dict[str, Any] = {
            "input": self.input,
            "actual_output": self.actual_output,
            "tools_called": [
                ToolCall(
                    name=t.name,
                    input_parameters=t.input_parameters,
                    output=t.output,
                )
                for t in self.tools_called
            ],
            "metadata": self.additional_metadata,
        }
        return LLMTestCase(**kwargs)


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


def to_deepeval_test_cases(
    trace_file: str | Path,
    stop_at_first_failure: bool = True,
) -> list[Any]:
    """Import an exact session trace directly as DeepEval LLMTestCase objects.

    If stop_at_first_failure is True, evaluation stops at the first failed turn,
    omitting subsequent polluted turns from the test dataset.
    """
    turns = load_trace(trace_file)
    test_cases = []
    for turn in turns:
        test_cases.append(turn.to_deepeval())
        if stop_at_first_failure and turn.has_peripheral_errors:
            break
    return test_cases


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


# ===========================================================================
# Memory-trace evaluation (text retrieval only).
#
# Operates on traces enriched by ``telemetry.py`` memory helpers and the
# multi-session ``session_runner``. A turn's memory metadata is expected under
# ``additional_metadata["memory"]`` with the shape produced by the telemetry
# ``*.as_record()`` / ``record_memory_injection`` helpers:
#
#   {
#     "provider": "cashew",
#     "session_id": "s2",
#     "injection": {"base_tokens", "injected_tokens", "overhead_tokens"},
#     "retrievals": [{"query", "passage_count", "passages", "latency_ms"}],
#     "consolidation": {"wall_seconds", "cpu_seconds", "db_growth_bytes", ...},
#     "generated_tokens": int,
#   }
# ===========================================================================

MEMORY_META_KEY = "memory"

# Tool names across the four providers that perform memory retrieval. Used to
# treat a tool call's output as retrieved context even when the harness did not
# emit an explicit retrieval span (e.g. the agent called the tool directly).
DEFAULT_MEMORY_TOOL_NAMES = frozenset(
    {
        "cashew_query",
        "cashew_recall",
        "recall",
        "chronicle_recall",
        "memex8_search",
        "memex8_recall",
        "nachos_memory_recall",
        "memory",
    }
)


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def _word_set(text: str) -> set[str]:
    return {w.strip(".,;:!?\"'()[]") for w in _normalize(text).split()} - {""}


def text_contains(haystack: str, needle: str, threshold: float = 0.6) -> bool:
    """True if ``needle`` appears in ``haystack`` by substring or word overlap.

    Substring match (after whitespace/case normalization) is authoritative.
    Otherwise, returns True when the fraction of ``needle`` content words also
    present in ``haystack`` meets ``threshold`` — tolerant of paraphrase/order.
    """
    h = _normalize(haystack)
    n = _normalize(needle)
    if not n:
        return False
    if n in h:
        return True
    needle_words = _word_set(needle)
    if not needle_words:
        return False
    hay_words = _word_set(haystack)
    overlap = len(needle_words & hay_words) / len(needle_words)
    return overlap >= threshold


def _token_f1(reference: str, candidate: str) -> float:
    """Standard QA token-level F1 between a reference and candidate string."""
    ref = _word_set(reference)
    cand = _word_set(candidate)
    if not ref and not cand:
        return 1.0
    if not ref or not cand:
        return 0.0
    common = ref & cand
    if not common:
        return 0.0
    precision = len(common) / len(cand)
    recall = len(common) / len(ref)
    return 2 * precision * recall / (precision + recall)


@dataclass
class TraceRecord:
    """A full (possibly multi-session) memory trace loaded from JSONL.

    Thin view over :class:`TraceTurn` list that surfaces the memory metadata
    the :class:`MemoryEvalSuite` judges consume.
    """

    turns: list[TraceTurn]
    file_path: Path | None = None
    memory_tool_names: frozenset[str] = DEFAULT_MEMORY_TOOL_NAMES

    @classmethod
    def from_file(cls, trace_file: str | Path) -> TraceRecord:
        turns = load_trace(trace_file)
        return cls(turns=turns, file_path=Path(trace_file))

    # -- memory metadata accessors ----------------------------------------
    @staticmethod
    def _mem(turn: TraceTurn) -> dict[str, Any]:
        meta = turn.additional_metadata or {}
        mem = meta.get(MEMORY_META_KEY)
        return mem if isinstance(mem, dict) else {}

    def session_id(self, turn: TraceTurn) -> str:
        mem = self._mem(turn)
        if mem.get("session_id"):
            return str(mem["session_id"])
        return str((turn.additional_metadata or {}).get("session_id", ""))

    @property
    def final_turn(self) -> TraceTurn | None:
        return self.turns[-1] if self.turns else None

    def retrieval_records(self) -> list[dict[str, Any]]:
        """All retrieval span records across every turn, tagged with session."""
        records: list[dict[str, Any]] = []
        for turn in self.turns:
            sid = self.session_id(turn)
            for rec in self._mem(turn).get("retrievals", []) or []:
                item = dict(rec)
                item.setdefault("session_id", sid)
                records.append(item)
        return records

    def retrieved_texts(self) -> list[str]:
        """Every passage injected via retrieval spans or memory-tool outputs."""
        texts: list[str] = []
        for rec in self.retrieval_records():
            texts.extend(str(p) for p in rec.get("passages", []) or [])
        for turn in self.turns:
            for tool in turn.tools_called:
                if tool.name in self.memory_tool_names and tool.output:
                    texts.append(tool.output)
        return texts

    def consolidation_records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for turn in self.turns:
            cons = self._mem(turn).get("consolidation")
            if isinstance(cons, dict):
                out.append(cons)
        return out

    def injected_tokens_total(self) -> int:
        total = 0
        for turn in self.turns:
            inj = self._mem(turn).get("injection") or {}
            total += int(inj.get("overhead_tokens", 0) or 0)
        return total

    def generated_tokens_total(self) -> int:
        total = 0
        for turn in self.turns:
            gen = self._mem(turn).get("generated_tokens")
            if gen is None:
                # Fall back to a heuristic on the produced answer text.
                gen = max(1, round(len(turn.actual_output) / 4)) if (
                    turn.actual_output
                ) else 0
            total += int(gen)
        return total


class MemoryEvalSuite:
    """Assertion judges for head-to-head memory-provider traces.

    All judges are text-only: they read retrieved passages, memory-tool
    outputs, and generated answers. No visual/multimodal signals are used.
    """

    def __init__(self, match_threshold: float = 0.6) -> None:
        self.match_threshold = match_threshold

    def _present_in_context(self, trace: TraceRecord, fact: str) -> bool:
        if any(
            text_contains(t, fact, self.match_threshold)
            for t in trace.retrieved_texts()
        ):
            return True
        return any(
            text_contains(turn.actual_output, fact, self.match_threshold)
            for turn in trace.turns
        )

    def eval_precision_recall(
        self, trace: TraceRecord, ground_truth: str
    ) -> float:
        """Token-F1 that the agent surfaced the target fact without noise.

        Measured against the final generated answer: recall rewards recalling
        the ground-truth fact, precision penalizes padding the answer with
        unrelated (potentially hallucinated) content.
        """
        final = trace.final_turn
        if final is None:
            return 0.0
        return _token_f1(ground_truth, final.actual_output)

    def eval_temporal_invalidation(
        self, trace: TraceRecord, stale_fact: str
    ) -> bool:
        """True iff an outdated/superseded fact was NOT recalled anywhere.

        Checks both retrieved context spans and generated answers.
        """
        return not self._present_in_context(trace, stale_fact)

    def eval_token_overhead_ratio(self, trace: TraceRecord) -> float:
        """Ratio of memory-injection overhead tokens to generated tokens."""
        generated = trace.generated_tokens_total()
        if generated == 0:
            return 0.0
        return trace.injected_tokens_total() / generated

    def eval_contradiction_rejection(
        self, trace: TraceRecord, superseded_facts: list[str]
    ) -> bool:
        """True iff NO superseded fact appears in the final answer or context.

        Scoped to the final belief-revision turn: its generated answer plus the
        retrieval spans / memory-tool outputs that supported it. (Whole-trace
        recall of a now-stale fact is judged by ``eval_temporal_invalidation``.)
        """
        final = trace.final_turn
        if final is None:
            return False
        contexts = [final.actual_output]
        for rec in trace._mem(final).get("retrievals", []) or []:
            contexts.extend(str(p) for p in rec.get("passages", []) or [])
        for tool in final.tools_called:
            if tool.name in trace.memory_tool_names and tool.output:
                contexts.append(tool.output)
        for fact in superseded_facts:
            if any(
                text_contains(c, fact, self.match_threshold) for c in contexts
            ):
                return False
        return True

    def eval_multi_hop(
        self, trace: TraceRecord, entities: list[str]
    ) -> bool:
        """True iff retrievals connect entities introduced in different sessions.

        Requires (a) every entity to appear in some retrieved passage and
        (b) the connecting retrievals to span at least two distinct sessions —
        evidence of a cross-session graph/passage hop rather than single-turn
        recall.
        """
        if len(entities) < 2:
            return False
        records = trace.retrieval_records()
        if not records:
            return False
        sessions_for_entity: dict[str, set[str]] = {}
        for ent in entities:
            hits: set[str] = set()
            for rec in records:
                passages = " ".join(
                    str(p) for p in rec.get("passages", []) or []
                )
                blob = f"{rec.get('query', '')} {passages}"
                if text_contains(blob, ent, self.match_threshold):
                    hits.add(str(rec.get("session_id", "")))
            sessions_for_entity[ent] = hits
        if any(not hits for hits in sessions_for_entity.values()):
            return False
        all_sessions: set[str] = set()
        for hits in sessions_for_entity.values():
            all_sessions |= hits
        return len(all_sessions) >= 2

    def cost_accuracy_point(
        self, trace: TraceRecord, accuracy: float
    ) -> tuple[float, float]:
        """One (context_token_overhead, accuracy) point for the frontier plot.

        Pair across providers to compare retrieval cost vs accuracy.
        """
        return (float(trace.injected_tokens_total()), float(accuracy))

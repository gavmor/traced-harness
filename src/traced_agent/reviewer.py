"""Interactive trace review CLI for rendering execution graphs and recording first-failure annotations."""

from __future__ import annotations

import json
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.tree import Tree

from traced_agent.models import Annotation, StepKind, TaxonomyCategory, Trace, Turn, Verdict
from traced_agent.storage import TraceStorage


class TraceReviewer:
    def __init__(self, storage: TraceStorage, console: Console | None = None) -> None:
        self.storage = storage
        self.console = console or Console()

    def render_turn_graph(self, turn: Turn) -> None:
        tree = Tree(f"[bold cyan]Turn {turn.turn_index}:[/bold cyan] [white]{turn.input_text}[/white]")

        for step in turn.steps:
            if step.kind == StepKind.USER_INPUT:
                continue
            elif step.kind == StepKind.REASONING:
                tree.add(f"[dim italic]Step {step.step_index} [Reasoning]:[/dim italic] {step.text[:200]}")
            elif step.kind == StepKind.TOOL_CALL:
                args_str = json.dumps(step.payload, indent=2)
                node = tree.add(f"[bold yellow]Step {step.step_index} [Tool Call]:[/bold yellow] [bold]{step.name}[/bold]")
                node.add(f"[dim]{args_str}[/dim]")
            elif step.kind == StepKind.TOOL_OUTPUT:
                out_snippet = step.text[:300] + ("..." if len(step.text) > 300 else "")
                tree.add(f"[dim green]Step {step.step_index} [Tool Output]:[/dim green] {out_snippet}")
            elif step.kind == StepKind.COMPLETION:
                tree.add(f"[bold green]Step {step.step_index} [Completion]:[/bold green] {step.text}")

        self.console.print(Panel(tree, title=f"Execution Graph: Turn {turn.turn_index}", border_style="blue"))

    def annotate_turn(self, trace: Trace, turn: Turn, taxonomy: list[TaxonomyCategory]) -> Annotation:
        self.render_turn_graph(turn)

        is_pass = Confirm.ask("\n[bold]Holistic Judgment: Did this turn PASS (acceptable)?[/bold]", default=True)
        verdict = Verdict.PASS if is_pass else Verdict.FAIL

        first_failure_idx = None
        open_code = ""
        failure_modes: dict[str, bool] = {cat.key: False for cat in taxonomy}

        if verdict == Verdict.FAIL:
            step_prompt = Prompt.ask(
                "[bold red]Enter the step index of the FIRST FAILURE (upstream root cause)[/bold red]",
                default="0",
            )
            try:
                first_failure_idx = int(step_prompt)
            except ValueError:
                first_failure_idx = 0

            open_code = Prompt.ask("[bold yellow]Open Code note (freeform description of divergence)[/bold yellow]")

            self.console.print("\n[bold]Select applicable failure modes from taxonomy (1=Present, 0=Absent):[/bold]")
            for cat in taxonomy:
                present = Confirm.ask(f"  - [bold]{cat.name}[/bold] ({cat.description[:60]}...)?", default=False)
                failure_modes[cat.key] = present

        ann = Annotation(
            annotation_id=uuid.uuid4().hex[:12],
            trace_id=trace.trace_id,
            turn_index=turn.turn_index,
            verdict=verdict,
            first_failure_step_index=first_failure_idx,
            open_code=open_code,
            failure_modes=failure_modes,
        )
        self.storage.save_annotation(ann)
        self.console.print(f"[bold green]✓ Annotation recorded for Turn {turn.turn_index}[/bold green]\n")
        return ann

    def review_trace(self, trace: Trace) -> None:
        self.console.print(f"\n[bold magenta]════ Reviewing Trace {trace.trace_id} (Session {trace.session_id}) ════[/bold magenta]")
        taxonomy = self.storage.get_taxonomy()
        for turn in trace.turns:
            self.annotate_turn(trace, turn, taxonomy)

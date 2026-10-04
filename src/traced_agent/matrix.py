"""Matrix labeling and failure prevalence rate calculations."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from traced_agent.models import Annotation, TaxonomyCategory, Verdict


class MatrixAnalysis:
    def __init__(self, annotations: list[Annotation], taxonomy: list[TaxonomyCategory]) -> None:
        self.annotations = annotations
        self.taxonomy = taxonomy
        self.taxonomy_keys = [cat.key for cat in taxonomy]

    def calculate_prevalence(self) -> dict[str, dict[str, float | int]]:
        total = len(self.annotations)
        if total == 0:
            return {}

        results: dict[str, dict[str, float | int]] = {}
        for cat in self.taxonomy:
            count = sum(1 for a in self.annotations if a.failure_modes.get(cat.key, False))
            rate = (count / total) * 100.0
            results[cat.key] = {
                "name": cat.name,
                "count": count,
                "prevalence_pct": round(rate, 1),
            }
        return results

    def render_summary_table(self, console: Console | None = None) -> None:
        console = console or Console()
        total = len(self.annotations)
        passes = sum(1 for a in self.annotations if a.verdict == Verdict.PASS)
        fails = sum(1 for a in self.annotations if a.verdict == Verdict.FAIL)

        pass_rate = (passes / total * 100.0) if total > 0 else 0.0

        console.print(f"\n[bold]Total Evaluated Traces / Turns:[/bold] {total}")
        console.print(f"[bold green]Passes:[/bold green] {passes} ({pass_rate:.1f}%) | [bold red]Fails:[/bold red] {fails} ({100.0 - pass_rate:.1f}%)")

        prevalence = self.calculate_prevalence()

        table = Table(title="Failure Taxonomy Prevalence Matrix", show_header=True, header_style="bold magenta")
        table.add_column("Key", style="dim")
        table.add_column("Category Name", style="bold")
        table.add_column("Failures Present (1)", justify="right")
        table.add_column("Prevalence Rate", justify="right")

        for key, stats in prevalence.items():
            cnt = stats["count"]
            pct = stats["prevalence_pct"]
            color = "red" if cnt > 0 else "green"
            table.add_row(key, str(stats["name"]), f"[{color}]{cnt}[/{color}]", f"[{color}]{pct}%[/{color}]")

        console.print(table)

"""CLI interface for traced-agent: ingestion, review, matrix analysis, and judge compilation."""

from __future__ import annotations

import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table

from traced_agent.judge_compiler import JudgeCompiler
from traced_agent.matrix import MatrixAnalysis
from traced_agent.reviewer import TraceReviewer
from traced_agent.storage import TraceStorage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="traced-agent: Whole-trace qualitative evaluation, failure taxonomy matrix labeling, and LLM-as-judge compiler."
    )
    parser.add_argument("--db", default="traces.db", help="Path to SQLite trace database (default: traces.db)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest
    ingest_p = subparsers.add_parser("ingest", help="Ingest sortie JSONL files into the trace database")
    ingest_p.add_argument("path", help="Path to a sortie JSONL file or directory containing JSONL files")

    # List
    subparsers.add_parser("list", help="List stored traces and their annotation status")

    # Review
    review_p = subparsers.add_parser("review", help="Review and annotate a trace interactively")
    review_p.add_argument("trace_id", nargs="?", default=None, help="Trace ID to review (default: most recent unreviewed)")

    # Matrix
    subparsers.add_parser("matrix", help="Display failure taxonomy prevalence matrix")

    # Taxonomy
    subparsers.add_parser("taxonomy", help="Show active failure taxonomy categories")

    # Compile Judges
    compile_p = subparsers.add_parser("compile-judges", help="Compile tagged failure modes into DeepEval GEval suites")
    compile_p.add_argument("-o", "--output", default="tests/evals/test_compiled_evals.py", help="Output pytest test suite file")

    args = parser.parse_args()
    console = Console()
    storage = TraceStorage(db_path=args.db)

    if args.command == "ingest":
        target_path = Path(args.path)
        files = [target_path] if target_path.is_file() else list(target_path.glob("sortie_*.jsonl"))
        if not files:
            console.print(f"[bold red]No sortie JSONL files found at:[/bold red] {target_path}")
            return
        for f in files:
            t = storage.ingest_sortie_jsonl(f)
            console.print(f"[bold green]✓ Ingested trace {t.trace_id}[/bold green] (Session {t.session_id}, {len(t.turns)} turns)")

    elif args.command == "list":
        traces = storage.list_traces()
        if not traces:
            console.print("[dim]No traces stored. Run 'traced-agent ingest <path>' to load sortie logs.[/dim]")
            return
        table = Table(title="Traced Agent Sessions", show_header=True, header_style="bold cyan")
        table.add_column("Trace ID", style="bold")
        table.add_column("Session ID")
        table.add_column("Turns", justify="right")
        table.add_column("Annotations", justify="right")
        table.add_column("Created At", style="dim")

        for t in traces:
            anns = storage.get_annotations_for_trace(t.trace_id)
            table.add_row(t.trace_id, t.session_id, str(len(t.turns)), str(len(anns)), t.created_at.strftime("%Y-%m-%d %H:%M"))
        console.print(table)

    elif args.command == "review":
        traces = storage.list_traces()
        if not traces:
            console.print("[bold red]No traces found to review.[/bold red]")
            return
        target_trace = None
        if args.trace_id:
            target_trace = storage.get_trace(args.trace_id)
        else:
            # Pick first unreviewed or most recent
            for t in traces:
                if len(storage.get_annotations_for_trace(t.trace_id)) < len(t.turns):
                    target_trace = t
                    break
            if not target_trace:
                target_trace = traces[0]

        if not target_trace:
            console.print(f"[bold red]Trace {args.trace_id} not found.[/bold red]")
            return
        reviewer = TraceReviewer(storage=storage, console=console)
        reviewer.review_trace(target_trace)

    elif args.command == "matrix":
        annotations = storage.list_all_annotations()
        taxonomy = storage.get_taxonomy()
        analysis = MatrixAnalysis(annotations=annotations, taxonomy=taxonomy)
        analysis.render_summary_table(console=console)

    elif args.command == "taxonomy":
        taxonomy = storage.get_taxonomy()
        table = Table(title="Failure Taxonomy Categories", show_header=True, header_style="bold green")
        table.add_column("Key", style="bold yellow")
        table.add_column("Name", style="bold")
        table.add_column("Description")
        for cat in taxonomy:
            table.add_row(cat.key, cat.name, cat.description)
        console.print(table)

    elif args.command == "compile-judges":
        annotations = storage.list_all_annotations()
        taxonomy = storage.get_taxonomy()
        traces = storage.list_traces()
        compiler = JudgeCompiler(taxonomy=taxonomy, annotations=annotations, traces=traces)
        out_path = Path(args.output)
        compiler.compile_pytest_suite(out_path)
        console.print(f"[bold green]✓ Successfully compiled LLM-as-judge evaluation suite to:[/bold green] {out_path}")


if __name__ == "__main__":
    main()

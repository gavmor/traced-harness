"""Interactive REPL for traced-agent powered by Agno."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from agno.agent import Agent
from mcp.client import Client
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from traced_harness.agent import DEFAULT_MODEL, TurnResult, create_agent, execute_turn
from traced_harness.skills import Skill
from traced_harness.telemetry import record_skill_activation_span

console = Console()


def print_banner(
    session_id: str,
    session_file: Path,
    mcp_label: str,
    tool_count: int,
    model_name: str,
    skills_count: int = 0,
    active_skills_count: int = 0,
) -> None:
    header = (
        f"[bold green]🔍 traced-harness (Agno + OpenTelemetry + MCP + Skills)[/]\n"
        f"[dim]MCP Server : {mcp_label} ({tool_count} tools available)[/dim]\n"
        f"[dim]Skills     : {skills_count} discovered ({active_skills_count} active)[/dim]\n"
        f"[dim]Session ID : {session_id}[/dim]\n"
        f"[dim]Trace Log  : {session_file}[/dim]\n"
        f"[dim]Model      : {model_name}[/dim]\n"
        f"[dim]Commands   : /new \\[name], /status, /tools, /skills, /skill <name>, /clear, or 'exit'/'quit'[/dim]"
    )
    console.print(Panel(header, border_style="green"))


def display_turn(res: TurnResult) -> None:
    """Format and display turn execution using Rich panels."""
    if res.tools_called:
        calls = "\n".join(
            f"• [bold cyan]{t.name}[/]({t.input_parameters})" for t in res.tools_called
        )
        console.print(Panel(calls, title="Tool Calls", border_style="blue"))
    console.print(Panel(Markdown(res.output), title="Response", border_style="green"))


async def run_repl(
    client: Client | None = None,
    mcp_label: str = "none",
    session_id: str | None = None,
    session_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    initial_prompt: str | None = None,
    skills: list[Skill] | None = None,
) -> None:
    """Run interactive REPL connected to an MCP server and/or skills."""
    skills = skills or []
    session_id = session_id or uuid.uuid4().hex[:12]
    session_dir = session_dir or (Path.cwd() / "sessions")
    session_dir.mkdir(parents=True, exist_ok=True)
    session_file = session_dir / f"trace_{session_id}.jsonl"

    agent: Agent = await create_agent(client, model_name=model_name, skills=skills)
    tool_count = 0
    tools_list_cache = []
    if client is not None:
        tools_resp = await client.list_tools()
        tools_list_cache = tools_resp.tools
        tool_count = len(tools_list_cache)

    active_count = sum(1 for s in skills if s.active)
    print_banner(
        session_id,
        session_file,
        mcp_label,
        tool_count,
        model_name,
        len(skills),
        active_count,
    )

    if initial_prompt:
        console.print(f"[bold blue]>>> {initial_prompt}[/]")
        try:
            res = await execute_turn(
                initial_prompt,
                agent=agent,
                session_id=session_id,
                session_file=session_file,
                mcp_label=mcp_label,
                skills=skills,
            )
            display_turn(res)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]Error: {exc}[/]\n")

    loop = asyncio.get_running_loop()
    while True:
        try:
            raw_input = await loop.run_in_executor(None, input, "agent> ")
        except (EOFError, KeyboardInterrupt):
            console.print(
                "\n[dim]Ending session. OpenTelemetry traces preserved.[/dim]"
            )
            break

        query = raw_input.strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", ":q", "q"):
            console.print("[dim]Ending session. OpenTelemetry traces preserved.[/dim]")
            break

        if query.startswith(("/new", "/reset")):
            parts = query.split(maxsplit=1)
            new_id = (
                parts[1].strip()
                if len(parts) > 1 and parts[1].strip()
                else uuid.uuid4().hex[:12]
            )
            session_id = new_id
            session_file = session_dir / f"trace_{session_id}.jsonl"
            # Recreate agent to reset in-memory conversation history
            agent = await create_agent(client, model_name=model_name, skills=skills)
            msg = (
                f"[bold green]🔄 Started new session[/]\n"
                f"[dim]Session ID : {session_id}[/dim]\n"
                f"[dim]Trace Log  : {session_file}[/dim]"
            )
            console.print(Panel(msg, border_style="green"))
            continue

        if query == "/status":
            active_skills_count = sum(1 for s in skills if s.active)
            info = (
                f"[bold]MCP Server[/] : {mcp_label}\n"
                f"[bold]MCP Tools[/]  : {tool_count} loaded\n"
                f"[bold]Skills[/]     : {len(skills)} discovered ({active_skills_count} active)\n"
                f"[bold]Session ID[/] : {session_id}\n"
                f"[bold]Trace Log[/]  : {session_file}\n"
                f"[bold]Model[/]      : {model_name}\n"
                f"[bold]Framework[/]  : Agno (async MCP + Skills)"
            )
            console.print(Panel(info, title="Session Status", border_style="cyan"))
            continue

        if query == "/tools":
            if not tools_list_cache:
                console.print(
                    Panel(
                        "[dim]No MCP tools available.[/dim]",
                        title="Available MCP Tools (0)",
                        border_style="blue",
                    )
                )
            else:
                formatted_tools = []
                for t in tools_list_cache:
                    desc = (t.description or "").strip().split("\n")[0]
                    formatted_tools.append(f"• [bold]{t.name}[/]: {desc}")
                console.print(
                    Panel(
                        "\n".join(formatted_tools),
                        title=f"Available MCP Tools ({tool_count})",
                        border_style="blue",
                    )
                )
            continue

        if query == "/skills":
            if not skills:
                console.print(
                    Panel(
                        "[dim]No skills discovered.[/dim]",
                        title="Skills (0)",
                        border_style="cyan",
                    )
                )
            else:
                table = Table(
                    title=f"Discovered Skills ({len(skills)})", border_style="cyan"
                )
                table.add_column("Name", style="bold cyan")
                table.add_column("Status", style="bold")
                table.add_column("Description")
                for s in skills:
                    status = (
                        "[bold green]Active[/]" if s.active else "[dim]Available[/]"
                    )
                    table.add_row(s.name, status, s.description)
                console.print(table)
            continue

        if query.startswith("/skill"):
            parts = query.split(maxsplit=2)
            if len(parts) < 2:
                console.print("[bold red]Usage:[/] /skill <name> [toggle|view]")
                continue

            target_name = parts[1].strip()
            subaction = parts[2].strip().lower() if len(parts) > 2 else None

            matched_skill = next(
                (s for s in skills if s.name.lower() == target_name.lower()), None
            )
            if not matched_skill:
                avail_names = ", ".join(s.name for s in skills) or "None"
                console.print(
                    f"[bold red]Skill '{target_name}' not found.[/] Discovered skills: {avail_names}"
                )
                continue

            if subaction == "view":
                status_label = (
                    "[bold green]Active[/]"
                    if matched_skill.active
                    else "[dim]Available[/]"
                )
            else:
                # Default behavior or 'toggle': toggle active state
                matched_skill.active = not matched_skill.active
                if matched_skill.active:
                    record_skill_activation_span(
                        name=matched_skill.name,
                        path=matched_skill.path,
                        chars_loaded=len(matched_skill.content),
                        mode="preload",
                    )
                    status_label = "[bold green]Active (Preloaded)[/]"
                else:
                    status_label = "[dim]Available (Deactivated)[/]"

                # Recreate agent to update system prompt instructions
                agent = await create_agent(client, model_name=model_name, skills=skills)

            skill_display = (
                f"**Status:** {status_label}\n\n"
                f"**Path:** `{matched_skill.path}`\n\n"
                f"**Description:** {matched_skill.description}\n\n"
                f"---\n\n"
                f"### Content\n\n{matched_skill.content}"
            )
            console.print(
                Panel(
                    Markdown(skill_display),
                    title=f"Skill: {matched_skill.name}",
                    border_style="cyan",
                )
            )
            continue

        if query == "/clear":
            console.clear()
            active_count = sum(1 for s in skills if s.active)
            print_banner(
                session_id,
                session_file,
                mcp_label,
                tool_count,
                model_name,
                len(skills),
                active_count,
            )
            continue

        if query in ("/help", "/?"):
            help_text = (
                "• [bold]/new [id][/]        - Reset context and rotate session trace log\n"
                "• [bold]/status[/]          - Show current session configuration, MCP target, and skills\n"
                "• [bold]/tools[/]           - List all discovered tools from the MCP server\n"
                "• [bold]/skills[/]          - List all discovered skills and their activation status\n"
                "• [bold]/skill <name>[/]    - Inspect skill markdown and toggle activation\n"
                "• [bold]/clear[/]           - Clear terminal screen\n"
                "• [bold]exit / quit[/]      - Exit REPL"
            )
            console.print(
                Panel(help_text, title="Available Commands", border_style="cyan")
            )
            continue

        try:
            res = await execute_turn(
                query,
                agent=agent,
                session_id=session_id,
                session_file=session_file,
                mcp_label=mcp_label,
                skills=skills,
            )
            display_turn(res)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]Error during execution: {exc}[/]\n")

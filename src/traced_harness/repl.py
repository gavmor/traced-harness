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

from traced_harness.agent import DEFAULT_MODEL, TurnResult, create_agent, execute_turn

console = Console()


def print_banner(
    session_id: str,
    session_file: Path,
    mcp_label: str,
    tool_count: int,
    model_name: str,
) -> None:
    header = (
        f"[bold green]🔍 traced-harness (Agno + OpenTelemetry + MCP)[/]\n"
        f"[dim]MCP Server : {mcp_label} ({tool_count} tools available)[/dim]\n"
        f"[dim]Session ID : {session_id}[/dim]\n"
        f"[dim]Trace Log  : {session_file}[/dim]\n"
        f"[dim]Model      : {model_name}[/dim]\n"
        f"[dim]Commands   : /new [name], /status, /tools, /clear, or 'exit'/'quit'[/dim]"
    )
    console.print(Panel(header, border_style="green"))


def display_turn(res: TurnResult) -> None:
    """Format and display turn execution using Rich panels."""
    if res.tools_called:
        calls = "\n".join(f"• [bold cyan]{t.name}[/]({t.input_parameters})" for t in res.tools_called)
        console.print(Panel(calls, title="Tool Calls", border_style="blue"))
    console.print(Panel(Markdown(res.output), title="Response", border_style="green"))


async def run_repl(
    client: Client,
    mcp_label: str,
    session_id: str | None = None,
    session_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    initial_prompt: str | None = None,
) -> None:
    """Run interactive REPL connected to an MCP server."""
    session_id = session_id or uuid.uuid4().hex[:12]
    session_dir = session_dir or (Path.cwd() / "sessions")
    session_dir.mkdir(parents=True, exist_ok=True)
    session_file = session_dir / f"trace_{session_id}.jsonl"

    agent: Agent = await create_agent(client, model_name=model_name)
    tools_resp = await client.list_tools()
    tool_count = len(tools_resp.tools)

    print_banner(session_id, session_file, mcp_label, tool_count, model_name)

    if initial_prompt:
        console.print(f"[bold blue]>>> {initial_prompt}[/]")
        try:
            res = await execute_turn(
                initial_prompt,
                agent=agent,
                session_id=session_id,
                session_file=session_file,
                mcp_label=mcp_label,
            )
            display_turn(res)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]Error: {exc}[/]\n")

    loop = asyncio.get_running_loop()
    while True:
        try:
            raw_input = await loop.run_in_executor(None, input, "agent> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Ending session. OpenTelemetry traces preserved.[/dim]")
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
                parts[1].strip() if len(parts) > 1 and parts[1].strip() else uuid.uuid4().hex[:12]
            )
            session_id = new_id
            session_file = session_dir / f"trace_{session_id}.jsonl"
            # Recreate agent to reset in-memory conversation history
            agent = await create_agent(client, model_name=model_name)
            msg = (
                f"[bold green]🔄 Started new session[/]\n"
                f"[dim]Session ID : {session_id}[/dim]\n"
                f"[dim]Trace Log  : {session_file}[/dim]"
            )
            console.print(Panel(msg, border_style="green"))
            continue

        if query == "/status":
            info = (
                f"[bold]MCP Server[/] : {mcp_label}\n"
                f"[bold]Tools[/]      : {tool_count} loaded\n"
                f"[bold]Session ID[/] : {session_id}\n"
                f"[bold]Trace Log[/]  : {session_file}\n"
                f"[bold]Model[/]      : {model_name}\n"
                f"[bold]Framework[/]  : Agno (async MCP)"
            )
            console.print(Panel(info, title="Session Status", border_style="cyan"))
            continue

        if query == "/tools":
            tools_list = []
            for t in tools_resp.tools:
                desc = (t.description or "").strip().split("\n")[0]
                tools_list.append(f"• [bold]{t.name}[/]: {desc}")
            console.print(Panel("\n".join(tools_list), title=f"Available MCP Tools ({tool_count})", border_style="blue"))
            continue

        if query == "/clear":
            console.clear()
            print_banner(session_id, session_file, mcp_label, tool_count, model_name)
            continue

        if query in ("/help", "/?"):
            help_text = (
                "• [bold]/new [id][/]   - Reset context and rotate session trace log\n"
                "• [bold]/status[/]     - Show current session configuration and MCP target\n"
                "• [bold]/tools[/]      - List all discovered tools from the MCP server\n"
                "• [bold]/clear[/]      - Clear terminal screen\n"
                "• [bold]exit / quit[/] - Exit REPL"
            )
            console.print(Panel(help_text, title="Available Commands", border_style="cyan"))
            continue

        try:
            res = await execute_turn(
                query,
                agent=agent,
                session_id=session_id,
                session_file=session_file,
                mcp_label=mcp_label,
            )
            display_turn(res)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]Error during execution: {exc}[/]\n")

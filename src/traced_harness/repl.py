"""Interactive REPL for traced-harness, built on cmd2.

Command dispatch, history, tab completion, ``help``, and ``quit`` come from
``cmd2``. What stays here is what is actually specific to this harness: the
Rich rendering of sessions/tools/skills, and the bridge that lets a synchronous
command loop drive the async agent.

Slash-prefixed commands (``/status``) are accepted as an alias for the bare
form (``status``) via a postparsing hook, so the REPL keeps the interface it
had before cmd2 took over dispatch.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import cmd2
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
        f"[dim]Commands   : type 'help' (or '?'); 'quit' to exit[/dim]"
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


class HarnessShell(cmd2.Cmd):
    """The REPL. One ``do_*`` per command; ``default`` sends input to the agent."""

    prompt = "agent> "

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        agent: Agent,
        client: Client | None,
        mcp_label: str,
        session_id: str,
        session_dir: Path,
        model_name: str,
        skills: list[Skill],
        tools: list[Any],
    ) -> None:
        super().__init__(allow_cli_args=False, include_ipy=False)
        self._loop = loop
        self.agent = agent
        self.client = client
        self.mcp_label = mcp_label
        self.session_id = session_id
        self.session_dir = session_dir
        self.model_name = model_name
        self.skills = skills
        self.tools = tools
        self.register_postparsing_hook(self._allow_slash_prefix)

    # -- plumbing ---------------------------------------------------------
    def _allow_slash_prefix(
        self, data: cmd2.plugin.PostparsingData
    ) -> cmd2.plugin.PostparsingData:
        """Treat ``/status`` as ``status``, preserving the original interface."""
        raw = data.statement.raw
        if raw.startswith("/"):
            data.statement = self.statement_parser.parse(raw[1:])
        return data

    def _await(self, coro: Any) -> Any:
        """Run a coroutine on the REPL's loop from cmd2's synchronous thread."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _rebuild_agent(self) -> None:
        """Recreate the agent, resetting in-context history and instructions."""
        self.agent = self._await(
            create_agent(self.client, model_name=self.model_name, skills=self.skills)
        )

    @property
    def session_file(self) -> Path:
        return self.session_dir / f"trace_{self.session_id}.jsonl"

    @property
    def _active_skills(self) -> int:
        return sum(1 for s in self.skills if s.active)

    def _banner(self) -> None:
        print_banner(
            self.session_id,
            self.session_file,
            self.mcp_label,
            len(self.tools),
            self.model_name,
            len(self.skills),
            self._active_skills,
        )

    # -- commands ---------------------------------------------------------
    def do_new(self, args: cmd2.Statement) -> None:
        """Reset context and rotate the session trace log.  Usage: new [id]"""
        self.session_id = args.args.strip() or uuid.uuid4().hex[:12]
        self._rebuild_agent()
        console.print(
            Panel(
                f"[bold green]🔄 Started new session[/]\n"
                f"[dim]Session ID : {self.session_id}[/dim]\n"
                f"[dim]Trace Log  : {self.session_file}[/dim]",
                border_style="green",
            )
        )

    do_reset = do_new

    def do_status(self, _args: cmd2.Statement) -> None:
        """Show session configuration, MCP target, and skills."""
        console.print(
            Panel(
                f"[bold]MCP Server[/] : {self.mcp_label}\n"
                f"[bold]MCP Tools[/]  : {len(self.tools)} loaded\n"
                f"[bold]Skills[/]     : {len(self.skills)} discovered "
                f"({self._active_skills} active)\n"
                f"[bold]Session ID[/] : {self.session_id}\n"
                f"[bold]Trace Log[/]  : {self.session_file}\n"
                f"[bold]Model[/]      : {self.model_name}\n"
                f"[bold]Framework[/]  : Agno (async MCP + Skills)",
                title="Session Status",
                border_style="cyan",
            )
        )

    def do_tools(self, _args: cmd2.Statement) -> None:
        """List all tools discovered from the MCP server."""
        body = (
            "\n".join(
                f"• [bold]{t.name}[/]: {(t.description or '').strip().splitlines()[0]}"
                if (t.description or "").strip()
                else f"• [bold]{t.name}[/]"
                for t in self.tools
            )
            or "[dim]No MCP tools available.[/dim]"
        )
        console.print(
            Panel(
                body,
                title=f"Available MCP Tools ({len(self.tools)})",
                border_style="blue",
            )
        )

    def do_skills(self, _args: cmd2.Statement) -> None:
        """List discovered skills and their activation status."""
        if not self.skills:
            console.print(
                Panel(
                    "[dim]No skills discovered.[/dim]",
                    title="Skills (0)",
                    border_style="cyan",
                )
            )
            return
        table = Table(
            title=f"Discovered Skills ({len(self.skills)})", border_style="cyan"
        )
        table.add_column("Name", style="bold cyan")
        table.add_column("Status", style="bold")
        table.add_column("Description")
        for s in self.skills:
            table.add_row(
                s.name,
                "[bold green]Active[/]" if s.active else "[dim]Available[/]",
                s.description,
            )
        console.print(table)

    def do_skill(self, args: cmd2.Statement) -> None:
        """Inspect a skill's markdown and toggle activation.

        Usage: skill <name> [toggle|view]
        """
        parts = args.args.split(maxsplit=1)
        if not parts:
            console.print("[bold red]Usage:[/] skill <name> [toggle|view]")
            return
        name = parts[0]
        view_only = len(parts) > 1 and parts[1].strip().lower() == "view"

        skill = next((s for s in self.skills if s.name.lower() == name.lower()), None)
        if skill is None:
            available = ", ".join(s.name for s in self.skills) or "None"
            console.print(
                f"[bold red]Skill '{name}' not found.[/] Discovered skills: {available}"
            )
            return

        if view_only:
            status = "[bold green]Active[/]" if skill.active else "[dim]Available[/]"
        else:
            skill.active = not skill.active
            if skill.active:
                record_skill_activation_span(
                    name=skill.name,
                    path=skill.path,
                    chars_loaded=len(skill.content),
                    mode="preload",
                )
                status = "[bold green]Active (Preloaded)[/]"
            else:
                status = "[dim]Available (Deactivated)[/]"
            self._rebuild_agent()

        console.print(
            Panel(
                Markdown(
                    f"**Status:** {status}\n\n"
                    f"**Path:** `{skill.path}`\n\n"
                    f"**Description:** {skill.description}\n\n"
                    f"---\n\n### Content\n\n{skill.content}"
                ),
                title=f"Skill: {skill.name}",
                border_style="cyan",
            )
        )

    def do_clear(self, _args: cmd2.Statement) -> None:
        """Clear the terminal screen."""
        console.clear()
        self._banner()

    def do_exit(self, _args: cmd2.Statement) -> bool:
        """Exit the REPL."""
        return True

    do_q = do_exit

    def default(self, statement: cmd2.Statement) -> None:
        """Anything that is not a command is a prompt for the agent."""
        self.run_turn(statement.raw.strip())

    def run_turn(self, prompt: str) -> None:
        """Execute one agent turn and render it."""
        try:
            display_turn(
                self._await(
                    execute_turn(
                        prompt,
                        agent=self.agent,
                        session_id=self.session_id,
                        session_file=self.session_file,
                        mcp_label=self.mcp_label,
                        skills=self.skills,
                    )
                )
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]Error during execution: {exc}[/]\n")


async def run_repl(
    client: Client | None = None,
    mcp_label: str = "none",
    session_id: str | None = None,
    session_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    initial_prompt: str | None = None,
    skills: list[Skill] | None = None,
) -> None:
    """Run the interactive REPL against an MCP server and/or skills."""
    skills = skills or []
    session_dir = session_dir or (Path.cwd() / "sessions")
    session_dir.mkdir(parents=True, exist_ok=True)

    agent = await create_agent(client, model_name=model_name, skills=skills)
    tools = list((await client.list_tools()).tools) if client is not None else []

    shell = HarnessShell(
        loop=asyncio.get_running_loop(),
        agent=agent,
        client=client,
        mcp_label=mcp_label,
        session_id=session_id or uuid.uuid4().hex[:12],
        session_dir=session_dir,
        model_name=model_name,
        skills=skills,
        tools=tools,
    )
    shell._banner()

    if initial_prompt:
        console.print(f"[bold blue]>>> {initial_prompt}[/]")
        shell.run_turn(initial_prompt)

    # cmd2's own cmdloop insists on the main thread (it installs SIGINT/SIGHUP
    # handlers) and reads via prompt_toolkit, so it cannot host an async agent.
    # It documents overriding it, and `onecmd_plus_hooks` is the entry point
    # cmdloop itself calls — so parsing, dispatch, hooks, history and `help`
    # still come from cmd2; only the read loop is ours. Dispatch runs off-loop
    # so commands can submit coroutines back to it.
    loop = asyncio.get_running_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, input, shell.prompt)
        except (EOFError, KeyboardInterrupt):
            break
        if not line.strip():
            continue
        if await asyncio.to_thread(shell.onecmd_plus_hooks, line):
            break

    console.print("\n[dim]Ending session. OpenTelemetry traces preserved.[/dim]")

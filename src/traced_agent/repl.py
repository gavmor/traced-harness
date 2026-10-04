"""Interactive REPL for traced-agent with session management and live inspection."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from mcp.client import Client

from traced_agent.agent import DEFAULT_MODEL, execute_turn


def print_banner(
    session_id: str,
    session_file: Path,
    mcp_label: str,
    tool_count: int,
    model_name: str,
) -> None:
    print("\033[1;32m═══════════════════════════════════════════════════════════════\033[0m")
    print("\033[1;32m🔍 traced-agent (DSPy ReAct + OpenTelemetry + MCP)\033[0m")
    print(f"\033[2mMCP Server : {mcp_label} ({tool_count} tools available)\033[0m")
    print(f"\033[2mSession ID : {session_id}\033[0m")
    print(f"\033[2mTrace Log  : {session_file}\033[0m")
    print(f"\033[2mModel      : {model_name}\033[0m")
    print("\033[2mCommands   : /new [name], /status, /tools, /clear, or 'exit'/'quit'\033[0m")
    print("\033[1;32m═══════════════════════════════════════════════════════════════\033[0m\n")


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

    tools_resp = await client.list_tools()
    tool_count = len(tools_resp.tools)

    print_banner(session_id, session_file, mcp_label, tool_count, model_name)

    if initial_prompt:
        print(f"\033[1;34m>>> {initial_prompt}\033[0m")
        try:
            res = await execute_turn(
                initial_prompt,
                client=client,
                session_id=session_id,
                session_file=session_file,
                mcp_label=mcp_label,
                model_name=model_name,
            )
            if res.tools_called:
                print(f"\033[2m🔧 Tools invoked: {', '.join(t.name for t in res.tools_called)}\033[0m")
            print("\n" + res.output + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"\033[1;31mError: {exc}\033[0m\n")

    loop = asyncio.get_running_loop()
    while True:
        try:
            raw_input = await loop.run_in_executor(None, input, "\033[1;36magent>\033[0m ")
        except (EOFError, KeyboardInterrupt):
            print("\n\033[2mEnding session. OpenTelemetry traces preserved.\033[0m")
            break

        query = raw_input.strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", ":q", "q"):
            print("\033[2mEnding session. OpenTelemetry traces preserved.\033[0m")
            break

        if query.startswith(("/new", "/reset")):
            parts = query.split(maxsplit=1)
            new_id = (
                parts[1].strip() if len(parts) > 1 and parts[1].strip() else uuid.uuid4().hex[:12]
            )
            session_id = new_id
            session_file = session_dir / f"trace_{session_id}.jsonl"
            print(
                "\n\033[1;32m═══════════════════════════════════════════════════════════════\033[0m"
            )
            print("\033[1;32m🔄 Started new session\033[0m")
            print(f"\033[2mSession ID : {session_id}\033[0m")
            print(f"\033[2mTrace Log  : {session_file}\033[0m")
            print(
                "\033[1;32m═══════════════════════════════════════════════════════════════\033[0m\n"
            )
            continue

        if query == "/status":
            print("\n\033[1;34m[Session Status]\033[0m")
            print(f"  MCP Server : {mcp_label}")
            print(f"  Tools      : {tool_count} loaded")
            print(f"  Session ID : {session_id}")
            print(f"  Trace Log  : {session_file}")
            print(f"  Model      : {model_name}")
            print("  Tracer     : OpenTelemetry (dspy)\n")
            continue

        if query == "/tools":
            print(f"\n\033[1;34m[Available MCP Tools ({tool_count})]\033[0m")
            for t in tools_resp.tools:
                desc = (t.description or "").strip().split("\n")[0]
                print(f"  • \033[1m{t.name}\033[0m: {desc}")
            print()
            continue

        if query == "/clear":
            print("\033[2J\033[H", end="")
            print_banner(session_id, session_file, mcp_label, tool_count, model_name)
            continue

        if query in ("/help", "/?"):
            print("\n\033[1;34m[Available Commands]\033[0m")
            print("  /new [id]   - Reset context and rotate session trace log")
            print("  /status     - Show current session configuration and MCP target")
            print("  /tools      - List all discovered tools from the MCP server")
            print("  /clear      - Clear terminal screen")
            print("  exit / quit - Exit REPL\n")
            continue

        print("\033[2mQuerying tools and formulating response...\033[0m")
        try:
            res = await execute_turn(
                query,
                client=client,
                session_id=session_id,
                session_file=session_file,
                mcp_label=mcp_label,
                model_name=model_name,
            )
            if res.tools_called:
                print(f"\033[2m🔧 Tools invoked: {', '.join(t.name for t in res.tools_called)}\033[0m")
            print("\n" + res.output + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"\033[1;31mError during execution: {exc}\033[0m\n")

"""CLI entrypoint for traced-agent powered by Agno."""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from traced_harness.agent import DEFAULT_MODEL, create_agent, execute_turn
from traced_harness.client import connect_mcp
from traced_harness.repl import display_turn, run_repl
from traced_harness.telemetry import setup_telemetry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="traced-harness",
        description="Minimal Agno agent harness instrumented with OpenTelemetry for any MCP server.",
    )
    parser.add_argument("prompt", nargs="?", default=None, help="Optional one-shot query")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive REPL mode (default if no prompt provided)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name for the agent (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--mcp",
        default=None,
        help="Command to launch an MCP server over stdio (e.g. 'uv run foxhole mcp')",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="Python import spec for in-process MCP server (e.g. 'foxhole.server:server')",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="HTTP/SSE URL for remote MCP server (e.g. 'http://localhost:8000/sse')",
    )
    parser.add_argument(
        "-s",
        "--session-id",
        default=None,
        help="Custom session identifier for tracking traces",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Directory to store JSONL session traces (default: ./sessions)",
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> None:
    setup_telemetry()
    session_id = args.session_id or uuid.uuid4().hex[:12]
    session_dir = args.session_dir or (Path.cwd() / "sessions")
    interactive_mode = args.interactive or (args.prompt is None)

    async with connect_mcp(
        mcp_cmd=args.mcp,
        server_spec=args.server,
        url=args.url,
    ) as (client, label):
        if interactive_mode:
            await run_repl(
                client=client,
                mcp_label=label,
                session_id=session_id,
                session_dir=session_dir,
                model_name=args.model,
                initial_prompt=args.prompt,
            )
        else:
            session_file = session_dir / f"trace_{session_id}.jsonl"
            print(f"\033[1;34m[{label}]\033[0m Query: {args.prompt}")
            agent = await create_agent(client, model_name=args.model)
            res = await execute_turn(
                args.prompt,
                agent=agent,
                session_id=session_id,
                session_file=session_file,
                mcp_label=label,
            )
            display_turn(res)
            print(f"\n\033[2mTrace recorded to {session_file}\033[0m")


def main() -> None:
    args = parse_args(sys.argv[1:])
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()

"""MCP client connection helper supporting stdio, in-process Python servers, and URLs."""

from __future__ import annotations

import importlib
import json
import os
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters


def parse_mcp_target(
    mcp_cmd: str | None = None,
    server_spec: str | None = None,
    url: str | None = None,
) -> tuple[Any, str]:
    """Resolve an MCP target into a Client-compatible argument and a display label."""
    if mcp_cmd:
        parts = shlex.split(mcp_cmd)
        if not parts:
            raise ValueError("Empty MCP command string.")
        params = StdioServerParameters(command=parts[0], args=parts[1:])
        return params, f"stdio:{mcp_cmd}"

    if server_spec:
        mod_name, _, attr_name = server_spec.partition(":")
        if not attr_name:
            mod_name, _, attr_name = server_spec.rpartition(".")
        mod = importlib.import_module(mod_name)
        server_obj = getattr(mod, attr_name)
        return server_obj, f"in-process:{server_spec}"

    if url:
        return url, f"url:{url}"

    # Auto-detection from environment
    env_mcp = os.environ.get("TRACED_AGENT_MCP") or os.environ.get("MCP_SERVER")
    if env_mcp:
        if env_mcp.startswith(("http://", "https://")):
            return env_mcp, f"url:{env_mcp}"
        if ":" in env_mcp and not (" " in env_mcp or "/" in env_mcp):
            return parse_mcp_target(server_spec=env_mcp)
        return parse_mcp_target(mcp_cmd=env_mcp)

    # Auto-detection from local .mcp.json
    mcp_json_path = Path.cwd() / ".mcp.json"
    if not mcp_json_path.exists():
        mcp_json_path = Path.cwd() / "mcp.json"
    if mcp_json_path.exists():
        try:
            data = json.loads(mcp_json_path.read_text("utf-8"))
            servers = data.get("mcpServers", {})
            if servers:
                _name, cfg = next(iter(servers.items()))
                cmd = cfg.get("command")
                args = cfg.get("args", [])
                if cmd:
                    full_cmd = f"{cmd} {' '.join(args)}".strip()
                    return parse_mcp_target(mcp_cmd=full_cmd)
        except (json.JSONDecodeError, OSError):
            pass

    # Auto-detection for Foxhole development environment
    try:
        mod = importlib.import_module("foxhole.server")
        server_obj = getattr(mod, "server", None)
        if server_obj is not None:
            return server_obj, "in-process:foxhole.server:server"
    except (ImportError, AttributeError):
        pass

    raise ValueError(
        "No MCP server specified. Provide --mcp '<cmd>', --server '<module:server>', "
        "or define MCP_SERVER / .mcp.json in the current directory."
    )


@asynccontextmanager
async def connect_mcp(
    mcp_cmd: str | None = None,
    server_spec: str | None = None,
    url: str | None = None,
) -> AsyncIterator[tuple[Client, str]]:
    """Establish connection to the resolved MCP server."""
    target, label = parse_mcp_target(mcp_cmd, server_spec, url)
    async with Client(target) as client:
        yield client, label

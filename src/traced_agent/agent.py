"""DSPy ReAct agent implementation instrumented with OpenTelemetry and session logging."""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import dspy
from mcp.client import Client
from opentelemetry import trace

from traced_agent.telemetry import get_tracer

tracer = get_tracer("traced.agent.dspy")

DEFAULT_MODEL = os.environ.get("AGENT_MODEL_NAME", "gemini/gemini-3.1-flash-lite-preview")


@dataclass
class ToolExecution:
    name: str
    input_parameters: dict[str, Any] = field(default_factory=dict)
    output: str = ""


@dataclass
class TurnResult:
    prompt: str
    output: str
    tools_called: list[ToolExecution] = field(default_factory=list)
    session_id: str = ""
    timestamp: str = ""


def configure_lm(model_name: str = DEFAULT_MODEL) -> dspy.LM:
    """Configure DSPy with the appropriate LM and credentials."""
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    dspy_model = model_name
    if not any(dspy_model.startswith(p) for p in ("gemini/", "openai/", "anthropic/", "ollama/")):
        dspy_model = f"gemini/{model_name}"

    lm = dspy.LM(dspy_model, api_key=api_key)
    dspy.configure(lm=lm)
    return lm


def log_turn_to_session(
    turn: TurnResult,
    session_file: Path,
    mcp_label: str = "",
) -> None:
    """Durably record the turn to a JSONL trace file."""
    session_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "input": turn.prompt,
        "actual_output": turn.output,
        "tools_called": [
            {
                "name": t.name,
                "input_parameters": t.input_parameters,
                "output": t.output,
            }
            for t in turn.tools_called
        ],
        "additional_metadata": {
            "session_id": turn.session_id,
            "timestamp": turn.timestamp,
            "agent": "traced_dspy_react",
            "mcp_server": mcp_label,
        },
    }
    with open(session_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


async def execute_turn(
    prompt: str,
    client: Client,
    session_id: str,
    session_file: Path | None = None,
    mcp_label: str = "",
    model_name: str = DEFAULT_MODEL,
    max_iters: int = 6,
) -> TurnResult:
    """Execute a single DSPy ReAct turn against the connected MCP server wrapped in OTEL spans."""
    configure_lm(model_name)

    # Discover and convert tools from the MCP server
    tools_resp = await client.list_tools()
    mcp_tools = tools_resp.tools
    dspy_tools = [dspy.Tool.from_mcp_tool(client.session, t) for t in mcp_tools]

    react = dspy.ReAct(
        cast(Any, "question -> answer"),
        tools=cast(Any, dspy_tools),
        max_iters=max_iters,
    )

    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    with tracer.start_as_current_span(
        "agent.turn",
        kind=trace.SpanKind.INTERNAL,
    ) as span:
        span.set_attribute("gen_ai.system", "dspy")
        span.set_attribute("gen_ai.agent.name", "traced-agent")
        span.set_attribute("gen_ai.session.id", session_id)
        span.set_attribute("gen_ai.prompt", prompt)
        span.set_attribute("gen_ai.model", model_name)
        span.set_attribute("mcp.server.target", mcp_label)

        pred = None
        for attempt in range(5):
            try:
                pred = await react.acall(question=prompt)
                break
            except Exception as exc:
                err_str = str(exc)
                is_transient = (
                    "429" in err_str
                    or "503" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "UNAVAILABLE" in err_str
                    or "quota" in err_str.lower()
                )
                if is_transient and attempt < 4:
                    m_retry = re.search(r"retry\s+in\s+([0-9\.]+)\s*s", err_str, re.IGNORECASE)
                    wait_time = float(m_retry.group(1)) + 1.0 if m_retry else (5.0 * (attempt + 1))
                    await asyncio.sleep(wait_time)
                    continue
                raise

        output_text = str(pred.answer) if hasattr(pred, "answer") and pred.answer else ""
        span.set_attribute("gen_ai.completion", output_text)

        # Parse tool calls from DSPy trajectory
        traj = pred.get("trajectory", {}) if hasattr(pred, "get") else {}
        tools_called: list[ToolExecution] = []
        idx = 0
        while f"tool_name_{idx}" in traj:
            t_name = traj[f"tool_name_{idx}"]
            t_args = traj.get(f"tool_args_{idx}", {})
            t_obs = traj.get(f"observation_{idx}", "")
            if t_name != "finish":
                tools_called.append(
                    ToolExecution(
                        name=t_name,
                        input_parameters=t_args if isinstance(t_args, dict) else {},
                        output=str(t_obs),
                    )
                )
            idx += 1

        span.set_attribute("gen_ai.tool_calls.count", len(tools_called))

        # Record child tool spans
        for tc in tools_called:
            with tracer.start_as_current_span(
                f"tool_call.{tc.name}",
                kind=trace.SpanKind.CLIENT,
            ) as tool_span:
                tool_span.set_attribute("gen_ai.tool.name", tc.name)
                tool_span.set_attribute("gen_ai.tool.parameters", json.dumps(tc.input_parameters))
                tool_span.set_attribute("gen_ai.tool.output", str(tc.output)[:2000])

        turn = TurnResult(
            prompt=prompt,
            output=output_text,
            tools_called=tools_called,
            session_id=session_id,
            timestamp=now_iso,
        )

        if session_file:
            log_turn_to_session(turn, session_file, mcp_label=mcp_label)

        return turn

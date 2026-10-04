"""Agno agent implementation instrumented with OpenTelemetry and session logging."""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.mcp import MCPTools
from mcp.client import Client
from opentelemetry import trace

from traced_harness.skills import (
    Skill,
    activate_skill,
    build_skill_instructions,
    get_registered_skills,
    register_skills,
)
from traced_harness.telemetry import get_tracer

tracer = get_tracer("traced.harness.agno")

DEFAULT_MODEL = os.environ.get("AGENT_MODEL_NAME", "gemini-3.1-flash-lite-preview")


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
    skills_active: list[str] = field(default_factory=list)


def get_model(model_name: str = DEFAULT_MODEL) -> Gemini:
    """Initialize model provider with environment API keys."""
    if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    clean_name = model_name.removeprefix("gemini/")
    return Gemini(id=clean_name, api_key=api_key)


async def create_agent(
    client: Client | None = None,
    model_name: str = DEFAULT_MODEL,
    skills: list[Skill] | None = None,
) -> Agent:
    """Create an Agno agent wired to optional MCP client and discovered skills."""
    tools: list[Any] = []

    if client is not None:
        mcp_tools = MCPTools(session=client.session)
        await mcp_tools.build_tools()
        mcp_tools._initialized = True
        tools.append(mcp_tools)

    instructions_list: list[str] = []
    if skills:
        register_skills(skills)
        tools.append(activate_skill)
        skill_prompt = build_skill_instructions(skills)
        if skill_prompt:
            instructions_list.append(skill_prompt)

    model = get_model(model_name)
    return Agent(
        model=model,
        tools=tools if tools else None,
        instructions=instructions_list if instructions_list else None,
        markdown=True,
    )


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
            "agent": "traced_agno",
            "mcp_server": mcp_label,
            "skills_active": turn.skills_active,
        },
    }
    with open(session_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


async def execute_turn(
    prompt: str,
    agent: Agent,
    session_id: str,
    session_file: Path | None = None,
    mcp_label: str = "",
    skills: list[Skill] | None = None,
) -> TurnResult:
    """Execute a turn with OpenTelemetry root and child tool spans."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    model_name = getattr(agent.model, "id", DEFAULT_MODEL)

    with tracer.start_as_current_span(
        "agent.turn",
        kind=trace.SpanKind.INTERNAL,
    ) as span:
        span.set_attribute("gen_ai.system", "agno")
        span.set_attribute("gen_ai.agent.name", "traced-harness")
        span.set_attribute("gen_ai.session.id", session_id)
        span.set_attribute("gen_ai.prompt", prompt)
        span.set_attribute("gen_ai.model", model_name)
        span.set_attribute("mcp.server.target", mcp_label)

        resp = await agent.arun(prompt, session_id=session_id)
        output_text = str(resp.content) if resp and resp.content else ""
        span.set_attribute("gen_ai.completion", output_text)

        tools_called: list[ToolExecution] = []
        for t in resp.tools or []:
            tools_called.append(
                ToolExecution(
                    name=t.tool_name or "",
                    input_parameters=t.tool_args or {},
                    output=str(t.result) if t.result is not None else "",
                )
            )

        span.set_attribute("gen_ai.tool_calls.count", len(tools_called))

        # Record child tool spans
        for tc in tools_called:
            with tracer.start_as_current_span(
                f"tool_call.{tc.name}",
                kind=trace.SpanKind.CLIENT,
            ) as tool_span:
                tool_span.set_attribute("gen_ai.tool.name", tc.name)
                tool_span.set_attribute(
                    "gen_ai.tool.parameters", json.dumps(tc.input_parameters)
                )
                tool_span.set_attribute("gen_ai.tool.output", tc.output[:2000])

        skills_pool = (
            skills if skills is not None else list(get_registered_skills().values())
        )
        skills_active = [s.name for s in skills_pool if s.active]

        turn = TurnResult(
            prompt=prompt,
            output=output_text,
            tools_called=tools_called,
            session_id=session_id,
            timestamp=now_iso,
            skills_active=skills_active,
        )

        if session_file:
            log_turn_to_session(turn, session_file, mcp_label=mcp_label)

        return turn

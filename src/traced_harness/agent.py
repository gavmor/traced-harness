"""Agno agent implementation instrumented with OpenTelemetry and session logging."""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.google import Gemini
from agno.tools.mcp import MCPTools
from mcp.client import Client
from openinference.instrumentation import using_attributes

from traced_harness.skills import (
    Skill,
    activate_skill,
    build_skill_instructions,
    get_registered_skills,
    register_skills,
)
from traced_harness.telemetry import (
    estimate_tokens,
    turn_telemetry,
)

DEFAULT_MODEL = os.environ.get("AGENT_MODEL_NAME", "gemini-3.1-flash-lite-preview")

#: Turn-metadata key for the model's own output-token count.
GENERATED_TOKENS_KEY = "generated_tokens"
#: Turn-metadata key for the model's own input-token count.
PROMPT_TOKENS_KEY = "prompt_tokens"


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
    #: Measurements contributed by whatever ran during the turn — the model's
    #: own token counts, plus anything a peripheral recorded through
    #: :func:`~traced_harness.telemetry.record_turn_metadata`. The harness
    #: carries this through to the trace without interpreting it.
    metadata: dict[str, Any] = field(default_factory=dict)


def _model_token_counts(resp: Any) -> tuple[int | None, int | None]:
    """Real (prompt, generated) token counts the model provider reported.

    Returns ``None`` for a count the provider did not supply, so the caller
    can tell "the model said zero" from "the model said nothing" and fall back
    to an estimate only in the latter case.
    """
    metrics = getattr(resp, "metrics", None)
    if metrics is None:
        return None, None
    prompt = getattr(metrics, "input_tokens", None)
    generated = getattr(metrics, "output_tokens", None)
    return (
        int(prompt) if prompt else None,
        int(generated) if generated else None,
    )


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
    memory_adapter: Any | None = None,
    extra_instructions: list[str] | None = None,
) -> Agent:
    """Create an Agno agent wired to optional MCP client, skills, and memory.

    When ``memory_adapter`` is supplied (any
    :class:`~traced_harness.memory.MemoryProviderAdapter`), its tool/context-hook
    contract is registered into the dispatch registry and injected into the
    system prompt, so the agent knows which memory tools (``cashew_query``,
    ``recall``, ...) or context-engine hooks are active. The concrete tool
    implementations arrive over the MCP ``client``.

    ``extra_instructions`` are appended verbatim — the generic escape hatch for
    peripherals the harness has no first-class support for.
    """
    tools: list[Any] = []
    tool_hooks: list[Any] = []

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

    if memory_adapter is not None:
        # Imported lazily: the harness core must not depend on the memory module.
        from traced_harness.memory import (
            build_memory_instructions,
            make_memory_tool_hook,
            register_memory_tools,
        )

        contract = memory_adapter.contract()
        register_memory_tools(contract.tools, contract.provider)
        memory_prompt = build_memory_instructions(
            contract.provider,
            contract.tools,
            contract.context_hooks,
            contract.system_prompt,
        )
        if memory_prompt:
            instructions_list.append(memory_prompt)
        # Measure the provider's recall calls where they actually happen, so
        # retrieval latency is a timed number rather than a post-hoc zero.
        tool_hooks.append(make_memory_tool_hook(contract))

    for instruction in extra_instructions or []:
        if instruction:
            instructions_list.append(instruction)

    model = get_model(model_name)
    return Agent(
        model=model,
        db=InMemoryDb(),
        tools=tools if tools else None,
        tool_hooks=tool_hooks if tool_hooks else None,
        instructions=instructions_list if instructions_list else None,
        markdown=True,
        add_history_to_context=True,
        num_history_runs=6,
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
            **(turn.metadata or {}),
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

    # Anything running under this turn — an MCP memory tool, a context hook —
    # records its measurements into `collected`; see
    # `traced_harness.telemetry.turn_telemetry`. The agent/LLM/tool spans
    # themselves come from AgnoInstrumentor (see `instrument_agno`), not from
    # hand-written set_attribute calls here.
    with (
        using_attributes(
            session_id=session_id, metadata={"mcp.server.target": mcp_label}
        ),
        turn_telemetry() as collected,
    ):
        resp = await agent.arun(prompt, session_id=session_id)

    output_text = str(resp.content) if resp and resp.content else ""

    # Prefer the provider's own tokenizer counts; `estimate_tokens` is the
    # documented fallback for a model that reports none.
    prompt_tokens, generated_tokens = _model_token_counts(resp)
    collected[PROMPT_TOKENS_KEY] = (
        prompt_tokens if prompt_tokens is not None else estimate_tokens(prompt)
    )
    collected[GENERATED_TOKENS_KEY] = (
        generated_tokens
        if generated_tokens is not None
        else estimate_tokens(output_text)
    )

    tools_called: list[ToolExecution] = [
        ToolExecution(
            name=t.tool_name or "",
            input_parameters=t.tool_args or {},
            output=str(t.result) if t.result is not None else "",
        )
        for t in resp.tools or []
    ]

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
        metadata=collected,
    )

    if session_file:
        log_turn_to_session(turn, session_file, mcp_label=mcp_label)

    return turn

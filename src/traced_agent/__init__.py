"""traced-agent: Minimal DSPy ReAct agent with OpenTelemetry and MCP integration."""

from traced_agent.agent import TurnResult, execute_turn
from traced_agent.client import connect_mcp
from traced_agent.telemetry import get_tracer, setup_telemetry

__all__ = [
    "TurnResult",
    "connect_mcp",
    "execute_turn",
    "get_tracer",
    "setup_telemetry",
]

"""traced-harness: Minimal Agno agent harness with OpenTelemetry, MCP, and Agent Skills."""

from traced_harness.agent import TurnResult, create_agent, execute_turn
from traced_harness.client import connect_mcp
from traced_harness.skills import (
    Skill,
    activate_skill,
    discover_skills,
    load_skill_file,
    scan_skills_dirs,
)
from traced_harness.telemetry import get_tracer, setup_telemetry

#: Lazily re-exported from :mod:`traced_harness.decisions`. Decisions are an
#: opt-in post-hoc annotation: importing the harness (or its CLI) must not drag
#: the decisions modules into ``sys.modules``, so these resolve on first access
#: via PEP 562 rather than at import time.
_LAZY_DECISIONS = (
    "DECISIONS_METADATA_KEY",
    "DecisionAnswer",
    "DecisionBackend",
    "DecisionQuestion",
    "annotate_trace",
)

__all__ = [
    "DECISIONS_METADATA_KEY",
    "DecisionAnswer",
    "DecisionBackend",
    "DecisionQuestion",
    "Skill",
    "TurnResult",
    "activate_skill",
    "annotate_trace",
    "connect_mcp",
    "create_agent",
    "discover_skills",
    "execute_turn",
    "get_tracer",
    "load_skill_file",
    "scan_skills_dirs",
    "setup_telemetry",
]


def __getattr__(name: str) -> object:
    if name in _LAZY_DECISIONS:
        import traced_harness.decisions as _decisions

        return getattr(_decisions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

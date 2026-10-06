"""Peripheral-health analysis over recorded traces, and agent context retention.

Validates that:
1. The Husain & Shankar first-failure principle (Ch. 3 & 8) is enforced by
   ``evaluate_trace``: it stops at the first observed peripheral failure and
   prunes cascaded downstream turns rather than cataloging symptoms.
2. Failing peripheral calls are identified and surfaced with a usable message.
3. The agent harness genuinely retains conversational context across turns.

Converting these traces into DeepEval test cases, and scoring them, lives in
the study repo that consumes this library (``memory-provider-evals``). The
harness itself has no evaluation-framework dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from traced_harness.eval import evaluate_trace

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "traces"


def test_first_failure_cascade_pruning() -> None:
    """Stop at the first failure; prune turns polluted by the upstream error."""
    trace_path = FIXTURES_DIR / "multi_turn_cascade.jsonl"

    pruned = evaluate_trace(trace_path, stop_at_first_failure=True)
    assert pruned.total_turns == 4
    assert pruned.evaluated_turns == 2
    assert pruned.polluted_turns_count == 2
    assert pruned.first_failure_turn == 1
    assert len(pruned.failed_tool_calls) == 1
    assert pruned.failed_tool_calls[0][0] == 1
    assert pruned.failed_tool_calls[0][1].name == "get_map_intel"

    # Unpruned evaluation catalogs downstream symptoms instead of the root cause.
    unpruned = evaluate_trace(trace_path, stop_at_first_failure=False)
    assert unpruned.total_turns == 4
    assert unpruned.evaluated_turns == 4
    assert unpruned.polluted_turns_count == 0
    assert len(unpruned.failed_tool_calls) == 2


def test_failing_peripheral_detection_and_contract() -> None:
    """Failing peripheral calls are identified and flagged."""
    report = evaluate_trace(
        FIXTURES_DIR / "failing_peripheral.jsonl", stop_at_first_failure=True
    )
    assert report.total_turns == 1
    assert report.evaluated_turns == 1
    assert report.first_failure_turn == 0
    assert len(report.failed_tool_calls) == 1

    failed_turn_idx, failed_tool = report.failed_tool_calls[0]
    assert failed_turn_idx == 0
    assert failed_tool.name == "get_map_intel"
    assert failed_tool.has_error is True
    assert "404 Not Found" in (failed_tool.error_message or "")


def test_clean_trace_full_pass() -> None:
    """A clean multi-turn run has 0% error rate and no failure turns."""
    report = evaluate_trace(
        FIXTURES_DIR / "clean_run.jsonl", stop_at_first_failure=True
    )
    assert report.total_turns == 2
    assert report.evaluated_turns == 2
    assert report.polluted_turns_count == 0
    assert report.first_failure_turn is None
    assert len(report.failed_tool_calls) == 0
    assert report.error_rate == 0.0


@pytest.mark.asyncio
async def test_multi_turn_conversational_context_retention() -> None:
    """The agent harness retains conversational context across turns."""
    from agno.models.google import Gemini
    from agno.models.response import ModelResponse

    from traced_harness.agent import create_agent

    class ContextAwareEvaluatorModel(Gemini):
        id: str = "context-aware-evaluator"

        async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            messages = kwargs.get("messages", [])
            prior_user_prompts = [
                getattr(m, "content", "")
                for m in messages
                if getattr(m, "role", "") == "user"
            ]
            full_context = " ".join(prior_user_prompts)
            if "Spatha" in full_context and len(prior_user_prompts) > 1:
                return ModelResponse(
                    content="The operation vehicle is the Colonial Spatha.",
                    role="assistant",
                )
            elif "Spatha" in full_context:
                return ModelResponse(
                    content="Understood, Colonial Spatha noted.",
                    role="assistant",
                )
            return ModelResponse(
                content="I do not know what vehicle we are discussing.",
                role="assistant",
            )

    agent = await create_agent()
    agent.model = ContextAwareEvaluatorModel(id="test", api_key="fake")

    # Turn 0: establish context
    await agent.arun(
        "The focus of our logistical plan is the Colonial Spatha.",
        session_id="eval-sess-1",
    )

    # Turn 1: context-dependent question without repeating the subject
    res = await agent.arun(
        "What vehicle did I say is the focus of this operation?",
        session_id="eval-sess-1",
    )
    output = str(res.content)

    assert "Spatha" in output, (
        f"Expected agent to recall 'Spatha' from Turn 0 context, but got: {output}"
    )

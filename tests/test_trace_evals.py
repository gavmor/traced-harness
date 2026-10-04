"""Trace evaluation suite implementing DeepEval ingestion and cascade pruning.

Validates that:
1. Recorded session traces are ingested directly into DeepEval LLMTestCase objects
   without re-running model inference.
2. The Husain & Shankar first-failure evaluation principle (Ch. 3 & 8) is enforced:
   stops evaluation at the first observed peripheral failure and prunes cascaded downstream turns.
3. Test cases are evaluated with DeepEval metrics (e.g. ToolCorrectnessMetric) and
   peripheral contract checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

from traced_harness.eval import evaluate_trace, to_deepeval_test_cases

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "traces"


class LocalDeterministicEvaluationLLM(DeepEvalBaseLLM):
    """Local, deterministic LLM stub for executing DeepEval metrics without network calls."""

    def load_model(self) -> Any:
        return None

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return "Deterministic evaluation check passed."

    async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return self.generate(prompt, *args, **kwargs)

    def get_model_name(self) -> str:
        return "local-deterministic-evaluator"


def test_trace_ingestion_into_deepeval_test_cases() -> None:
    """Validate recorded traces are converted directly into LLMTestCase instances without inference."""
    trace_path = FIXTURES_DIR / "clean_run.jsonl"
    test_cases = to_deepeval_test_cases(trace_path, stop_at_first_failure=True)

    assert len(test_cases) == 2
    for tc in test_cases:
        assert isinstance(tc, LLMTestCase)
        assert tc.input
        assert tc.actual_output
        assert len(tc.tools_called) >= 1
        for tool in tc.tools_called:
            assert isinstance(tool, ToolCall)
            assert tool.name
            assert isinstance(tool.input_parameters, dict)
            assert tool.output

    # Verify field fidelity for Turn 0
    turn0 = test_cases[0]
    assert turn0.input == "What is the health and armor of the Colonial Spatha tank?"
    assert "3650 HP" in turn0.actual_output
    assert turn0.tools_called[0].name == "get_vehicle_stats"
    assert turn0.tools_called[0].input_parameters == {"vehicle_name": "Spatha"}
    meta = getattr(turn0, "metadata", None) or getattr(turn0, "additional_metadata", {})
    assert meta.get("session_id") == "clean_session"
    assert meta.get("agent") == "traced_agno"


def test_husain_shankar_first_failure_cascade_pruning() -> None:
    """Enforce the first-failure stopping rule to avoid multi-turn cascade error pollution."""
    trace_path = FIXTURES_DIR / "multi_turn_cascade.jsonl"

    # 1. With stop_at_first_failure=True (default), stop at turn 1 and prune turns 2 & 3
    report_pruned = evaluate_trace(trace_path, stop_at_first_failure=True)
    assert report_pruned.total_turns == 4
    assert report_pruned.evaluated_turns == 2
    assert report_pruned.polluted_turns_count == 2
    assert report_pruned.first_failure_turn == 1
    assert len(report_pruned.failed_tool_calls) == 1
    assert report_pruned.failed_tool_calls[0][0] == 1
    assert report_pruned.failed_tool_calls[0][1].name == "get_map_intel"

    test_cases_pruned = to_deepeval_test_cases(trace_path, stop_at_first_failure=True)
    assert len(test_cases_pruned) == 2
    assert test_cases_pruned[0].tools_called[0].name == "get_vehicle_stats"
    assert test_cases_pruned[1].tools_called[0].name == "get_map_intel"

    # 2. With stop_at_first_failure=False, unpruned evaluation catalogs downstream symptoms
    report_unpruned = evaluate_trace(trace_path, stop_at_first_failure=False)
    assert report_unpruned.total_turns == 4
    assert report_unpruned.evaluated_turns == 4
    assert report_unpruned.polluted_turns_count == 0
    # Catalogs two failures instead of isolating the root cause
    assert len(report_unpruned.failed_tool_calls) == 2

    test_cases_unpruned = to_deepeval_test_cases(
        trace_path, stop_at_first_failure=False
    )
    assert len(test_cases_unpruned) == 4


def test_failing_peripheral_detection_and_contract() -> None:
    """Ensure failing peripheral calls are identified and flagged."""
    trace_path = FIXTURES_DIR / "failing_peripheral.jsonl"

    report = evaluate_trace(trace_path, stop_at_first_failure=True)
    assert report.total_turns == 1
    assert report.evaluated_turns == 1
    assert report.first_failure_turn == 0
    assert len(report.failed_tool_calls) == 1

    failed_turn_idx, failed_tool = report.failed_tool_calls[0]
    assert failed_turn_idx == 0
    assert failed_tool.name == "get_map_intel"
    assert failed_tool.has_error is True
    assert "404 Not Found" in (failed_tool.error_message or "")

    test_cases = to_deepeval_test_cases(trace_path, stop_at_first_failure=True)
    assert len(test_cases) == 1
    assert test_cases[0].tools_called[0].name == "get_map_intel"


def test_deepeval_tool_correctness_metric_evaluation() -> None:
    """Evaluate ingested trace test cases using DeepEval's ToolCorrectnessMetric."""
    trace_path = FIXTURES_DIR / "clean_run.jsonl"
    test_cases = to_deepeval_test_cases(trace_path, stop_at_first_failure=True)
    local_model = LocalDeterministicEvaluationLLM()

    # Turn 0: expected tools match actual ingested tools
    case0 = test_cases[0]
    case0.expected_tools = [
        ToolCall(
            name="get_vehicle_stats",
            input_parameters={"vehicle_name": "Spatha"},
        )
    ]

    metric = ToolCorrectnessMetric(
        model=local_model,
        should_exact_match=True,
        async_mode=False,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
    )
    metric.measure(case0)

    assert metric.score == 1.0
    assert metric.is_successful() is True

    # Turn 1: test parameter mismatch fails ToolCorrectnessMetric
    case1 = test_cases[1]
    case1.expected_tools = [
        ToolCall(
            name="get_production_cost",
            input_parameters={
                "item_name": "DifferentTank"
            },  # Intentional parameter mismatch
        )
    ]

    metric_mismatch = ToolCorrectnessMetric(
        model=local_model,
        should_exact_match=True,
        async_mode=False,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
    )
    metric_mismatch.measure(case1)

    assert metric_mismatch.score == 0.0
    assert metric_mismatch.is_successful() is False


def test_clean_trace_full_pass() -> None:
    """Verify that a clean multi-turn run has 0% error rate and no failure turns."""
    trace_path = FIXTURES_DIR / "clean_run.jsonl"
    report = evaluate_trace(trace_path, stop_at_first_failure=True)

    assert report.total_turns == 2
    assert report.evaluated_turns == 2
    assert report.polluted_turns_count == 0
    assert report.first_failure_turn is None
    assert len(report.failed_tool_calls) == 0
    assert report.error_rate == 0.0


@pytest.mark.asyncio
async def test_agent_history_in_context_configured() -> None:
    """Validate that create_agent configures Agno to maintain conversation history in context."""
    from traced_harness.agent import create_agent

    agent = await create_agent()
    assert agent.add_history_to_context is True, (
        "Agent must have add_history_to_context=True to maintain context across multi-turn sessions"
    )

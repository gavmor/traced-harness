"""Auto-generated LLM-as-judge evaluation suite compiled from tagged trace failures."""

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

WRONG_TOOL_JUDGE = GEval(
    name="Wrong Tool Selection",
    criteria="""Determine if the assistant committed a 'Wrong Tool Selection' failure.
Description: Agent selected an inappropriate tool or bypassed a dedicated tool in favor of generic queries/raw commands.""",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.TOOLS_CALLED,
    ],
    evaluation_steps=[
        "Inspect the user input for specific operational, geographic, or logistical constraints.",
        "Analyze the ordered sequence of tool calls and parameter choices before the final response.",
        "Check whether the agent used appropriate MCP tools or bypassed them with hallucinated mechanisms.",
        "Determine if the final response claims actions, mechanics, or routes that contradict ground-truth context.",
        "Assign a score of 1.0 if the failure is ABSENT (Pass), or 0.0 if the failure is PRESENT (Fail)."
    ],
    threshold=1.0,
)


HALLUCINATED_UI_JUDGE = GEval(
    name="Hallucinated Context / UI",
    criteria="""Determine if the assistant committed a 'Hallucinated Context / UI' failure.
Description: Agent invented nonexistent UI menus, interactions, buttons, or context actions.
Observed failure patterns from trace reviews:
- Agent advised uncrating in Seaport facility menu which is wholesale crated storage only""",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.TOOLS_CALLED,
    ],
    evaluation_steps=[
        "Inspect the user input for specific operational, geographic, or logistical constraints.",
        "Analyze the ordered sequence of tool calls and parameter choices before the final response.",
        "Check whether the agent used appropriate MCP tools or bypassed them with hallucinated mechanisms.",
        "Determine if the final response claims actions, mechanics, or routes that contradict ground-truth context.",
        "Assign a score of 1.0 if the failure is ABSENT (Pass), or 0.0 if the failure is PRESENT (Fail)."
    ],
    threshold=1.0,
)


TOPOLOGICAL_INVERSION_JUDGE = GEval(
    name="Topological / Spatial Inversion",
    criteria="""Determine if the assistant committed a 'Topological / Spatial Inversion' failure.
Description: Agent inverted spatial, geographic, or waterway topology instead of following actual map corridors.""",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.TOOLS_CALLED,
    ],
    evaluation_steps=[
        "Inspect the user input for specific operational, geographic, or logistical constraints.",
        "Analyze the ordered sequence of tool calls and parameter choices before the final response.",
        "Check whether the agent used appropriate MCP tools or bypassed them with hallucinated mechanisms.",
        "Determine if the final response claims actions, mechanics, or routes that contradict ground-truth context.",
        "Assign a score of 1.0 if the failure is ABSENT (Pass), or 0.0 if the failure is PRESENT (Fail)."
    ],
    threshold=1.0,
)


ANCHORING_DRIFT_JUDGE = GEval(
    name="Multi-Turn Anchoring / Drift",
    criteria="""Determine if the assistant committed a 'Multi-Turn Anchoring / Drift' failure.
Description: Agent committed early to an incorrect assumption and failed to revise it across subsequent turns.""",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.TOOLS_CALLED,
    ],
    evaluation_steps=[
        "Inspect the user input for specific operational, geographic, or logistical constraints.",
        "Analyze the ordered sequence of tool calls and parameter choices before the final response.",
        "Check whether the agent used appropriate MCP tools or bypassed them with hallucinated mechanisms.",
        "Determine if the final response claims actions, mechanics, or routes that contradict ground-truth context.",
        "Assign a score of 1.0 if the failure is ABSENT (Pass), or 0.0 if the failure is PRESENT (Fail)."
    ],
    threshold=1.0,
)


MISSING_CONSTRAINT_JUDGE = GEval(
    name="Missing Constraint",
    criteria="""Determine if the assistant committed a 'Missing Constraint' failure.
Description: Agent dropped or ignored explicit constraints specified in the prompt or ground truth context.""",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.TOOLS_CALLED,
    ],
    evaluation_steps=[
        "Inspect the user input for specific operational, geographic, or logistical constraints.",
        "Analyze the ordered sequence of tool calls and parameter choices before the final response.",
        "Check whether the agent used appropriate MCP tools or bypassed them with hallucinated mechanisms.",
        "Determine if the final response claims actions, mechanics, or routes that contradict ground-truth context.",
        "Assign a score of 1.0 if the failure is ABSENT (Pass), or 0.0 if the failure is PRESENT (Fail)."
    ],
    threshold=1.0,
)


ALL_COMPILED_JUDGES = [
    WRONG_TOOL_JUDGE,
    HALLUCINATED_UI_JUDGE,
    TOPOLOGICAL_INVERSION_JUDGE,
    ANCHORING_DRIFT_JUDGE,
    MISSING_CONSTRAINT_JUDGE
]

@pytest.mark.parametrize("metric", ALL_COMPILED_JUDGES)
def test_compiled_trace_evaluators(metric):
    """Run compiled trace evaluators against regression test cases."""
    # Example placeholder test case
    test_case = LLMTestCase(
        input="Check my recent Foxhole play session timeline.",
        actual_output="Synchronized logs via archive_game_logs and retrieved timeline.",
        tools_called=[],
    )
    assert_test(test_case, [metric])

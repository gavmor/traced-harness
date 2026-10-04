"""Judge Compiler: Translates tagged failure modes, open codes, and traces into automated LLM-as-judge evaluators."""

from __future__ import annotations

from pathlib import Path
from traced_agent.models import Annotation, TaxonomyCategory, Trace


class JudgeCompiler:
    def __init__(self, taxonomy: list[TaxonomyCategory], annotations: list[Annotation], traces: list[Trace]) -> None:
        self.taxonomy = taxonomy
        self.annotations = annotations
        self.traces = {t.trace_id: t for t in traces}

    def compile_category_geval(self, category: TaxonomyCategory) -> str:
        """Generate a DeepEval G-Eval definition tailored to this failure mode and annotated exemplars."""
        cat_annotations = [a for a in self.annotations if a.failure_modes.get(category.key, False)]
        open_codes = [a.open_code for a in cat_annotations if a.open_code]
        unique_open_codes = list(set(open_codes))[:5]

        exemplar_notes = ""
        if unique_open_codes:
            exemplar_notes = "\nObserved failure patterns from trace reviews:\n" + "\n".join(
                f"- {code}" for code in unique_open_codes
            )

        var_name = f"{category.key.upper()}_JUDGE"
        return f'''{var_name} = GEval(
    name="{category.name}",
    criteria="""Determine if the assistant committed a '{category.name}' failure.
Description: {category.description}{exemplar_notes}""",
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
'''

    def compile_pytest_suite(self, output_file: Path) -> None:
        """Compile all active failure categories into an executable DeepEval pytest suite."""
        lines = [
            '"""Auto-generated LLM-as-judge evaluation suite compiled from tagged trace failures."""',
            "",
            "import pytest",
            "from deepeval import assert_test",
            "from deepeval.metrics import GEval",
            "from deepeval.test_case import LLMTestCase, LLMTestCaseParams",
            "",
        ]

        # Compile GEval instances
        for cat in self.taxonomy:
            lines.append(self.compile_category_geval(cat))
            lines.append("")

        # Create test harness template
        lines.append(
            '''ALL_COMPILED_JUDGES = [
'''
            + ",\n".join(f"    {cat.key.upper()}_JUDGE" for cat in self.taxonomy)
            + """
]

@pytest.mark.parametrize("metric", ALL_COMPILED_JUDGES)
def test_compiled_trace_evaluators(metric):
    \"\"\"Run compiled trace evaluators against regression test cases.\"\"\"
    # Example placeholder test case
    test_case = LLMTestCase(
        input="Check my recent Foxhole play session timeline.",
        actual_output="Synchronized logs via archive_game_logs and retrieved timeline.",
        tools_called=[],
    )
    assert_test(test_case, [metric])
"""
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

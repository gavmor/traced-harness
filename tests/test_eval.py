"""Unit tests for trace evaluation and DeepEval test case import."""

from __future__ import annotations

import json
from pathlib import Path

from traced_harness.eval import (
    ToolExecutionData,
    evaluate_trace,
    load_trace,
)
from traced_harness.main import parse_args


def test_tool_execution_error_detection():
    clean_tool = ToolExecutionData(
        name="get_structure_stats",
        input_parameters={"structure_name": "Shipyard"},
        output='{"title": "Shipyard", "hp": 5000}',
    )
    assert not clean_tool.has_error
    assert clean_tool.error_message is None

    err_tool = ToolExecutionData(
        name="get_map_intel",
        input_parameters={"map_name": "BlackcoatHex"},
        output='{"error": "Could not retrieve map telemetry for \'BlackcoatHex\' on shard \'live-1."}',
    )
    assert err_tool.has_error
    assert "Could not retrieve map telemetry" in (err_tool.error_message or "")


def test_load_and_evaluate_trace(tmp_path: Path):
    trace_file = tmp_path / "test_trace.jsonl"
    entry1 = {
        "input": "Where is the shipyard?",
        "actual_output": "The shipyard is at Gutter.",
        "tools_called": [
            {
                "name": "get_structure_stats",
                "input_parameters": {"structure_name": "Shipyard"},
                "output": '{"hp": 5000}',
            }
        ],
        "additional_metadata": {"session_id": "s1"},
    }
    entry2 = {
        "input": "Check Blackcoat status",
        "actual_output": "I am checking.",
        "tools_called": [
            {
                "name": "get_map_intel",
                "input_parameters": {"map_name": "BlackcoatHex"},
                "output": '{"error": "404 Not Found"}',
            }
        ],
        "additional_metadata": {"session_id": "s1"},
    }
    with open(trace_file, "w") as f:
        f.write(json.dumps(entry1) + "\n")
        f.write(json.dumps(entry2) + "\n")

    turns = load_trace(trace_file)
    assert len(turns) == 2
    assert not turns[0].has_peripheral_errors
    assert turns[1].has_peripheral_errors

    report = evaluate_trace(trace_file)
    assert report.total_turns == 2
    assert report.total_tool_calls == 2
    assert len(report.failed_tool_calls) == 1
    assert report.failed_tool_calls[0][1].name == "get_map_intel"
    assert report.error_rate == 0.5


def test_parse_args_eval_subcommand(tmp_path: Path):
    dummy_trace = tmp_path / "dummy.jsonl"
    dummy_trace.touch()

    args1 = parse_args(["eval", str(dummy_trace)])
    assert args1.eval_trace == dummy_trace

    args2 = parse_args(["--eval", str(dummy_trace), "--fail-on-errors"])
    assert args2.eval_trace == dummy_trace
    assert args2.fail_on_errors is True

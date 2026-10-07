"""Tests for the ``decide`` CLI surface and for default behaviour staying put.

The feature is opt-in: with none of the new flags, ``traced-harness`` behaves
exactly as it did before decisions existed — including not importing the
decisions modules at all.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from traced_harness.main import async_main, parse_args

REPO_ROOT = Path(__file__).parents[1]
TRACES = REPO_ROOT / "tests" / "fixtures" / "traces"
EXAMPLES = REPO_ROOT / "docs" / "examples"
CLEAN_RUN = TRACES / "clean_run.jsonl"

_DECIDE_FIELDS = (
    "decide_trace",
    "decide_questions",
    "decide_backend",
    "decide_model",
    "decide_replay",
    "decide_out",
    "decide_fail_on_error",
)


def test_cli_and_default_behaviour_unchanged() -> None:
    subcommand = parse_args(
        ["decide", "trace.jsonl", "--questions", "q.json", "--decisions-backend", "replay",
         "--replay-file", "r.json"]
    )
    flag = parse_args(
        ["--decide", "trace.jsonl", "--questions", "q.json", "--decisions-backend", "replay",
         "--replay-file", "r.json"]
    )
    assert vars(subcommand) == vars(flag)
    assert subcommand.decide_trace == Path("trace.jsonl")
    assert subcommand.decide_questions == Path("q.json")
    assert subcommand.decide_backend == "replay"
    assert subcommand.decide_replay == Path("r.json")

    # Defaults: a bare invocation is untouched by any of this.
    bare = parse_args([])
    assert bare.decide_trace is None
    assert bare.decide_questions is None
    assert bare.decide_replay is None
    assert bare.decide_out is None
    assert bare.decide_fail_on_error is False
    assert bare.decide_backend == "openai"
    assert bare.decide_model == "gpt-6-luna"
    assert bare.prompt is None
    assert bare.eval_trace is None
    assert bare.model  # the pre-existing agent model default survives

    # The pre-existing eval forms still parse.
    assert parse_args(["eval", "t.jsonl"]).eval_trace == Path("t.jsonl")
    assert parse_args(["--eval", "t.jsonl"]).eval_trace == Path("t.jsonl")


def test_importing_the_cli_does_not_import_decisions() -> None:
    """Asserted in a clean interpreter: this session has already imported them."""
    probe = (
        "import sys; import traced_harness.main; "
        "assert 'traced_harness.decisions' not in sys.modules, 'decisions imported'; "
        "assert 'traced_harness.decisions_openai' not in sys.modules, 'openai imported'; "
        "import traced_harness; "
        "assert 'traced_harness.decisions' not in sys.modules, 'package import leaked'; "
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_lazy_reexports_resolve_on_access() -> None:
    import traced_harness

    assert traced_harness.DECISIONS_METADATA_KEY == "decisions"
    assert traced_harness.DecisionQuestion.__name__ == "DecisionQuestion"
    assert traced_harness.DecisionAnswer.__name__ == "DecisionAnswer"
    assert traced_harness.DecisionBackend.__name__ == "DecisionBackend"
    assert callable(traced_harness.annotate_trace)
    # The lazily-imported OpenAI backend is deliberately not re-exported.
    assert not hasattr(traced_harness, "OpenAIDecisionsBackend")
    with pytest.raises(AttributeError):
        traced_harness.nope  # noqa: B018


@pytest.mark.parametrize(
    "argv",
    [
        ["--decide", "t.jsonl"],
        ["decide", "t.jsonl"],
        ["--decide", "t.jsonl", "--questions", "q.json", "--decisions-backend", "replay"],
    ],
)
def test_incomplete_decide_invocation_exits_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        parse_args(argv)
    assert caught.value.code == 2


def test_help_lists_the_new_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--help"])
    help_text = capsys.readouterr().out
    for flag in (
        "--decide",
        "--questions",
        "--decisions-backend",
        "--decisions-model",
        "--replay-file",
        "--decisions-out",
        "--decisions-fail-on-error",
    ):
        assert flag in help_text


def test_decide_end_to_end_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = tmp_path / "clean_run.decided.jsonl"
    args = parse_args(
        [
            "decide",
            str(CLEAN_RUN),
            "--questions",
            str(EXAMPLES / "decision-questions.json"),
            "--decisions-backend",
            "replay",
            "--replay-file",
            str(EXAMPLES / "recorded-decisions.json"),
            "--decisions-out",
            str(out),
        ]
    )
    asyncio.run(async_main(args))
    assert str(out) in capsys.readouterr().out

    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 2
    for index, line in enumerate(lines):
        record = line["additional_metadata"]["decisions"]
        assert record["schema_version"] == 1
        assert record["backend"] == "replay"
        assert record["trace_id"] == "clean_run"
        assert record["turn_index"] == index
        assert record["session_id"] == "clean_session"
        assert len(record["answers"]) == 3


def test_decide_rejects_an_unknown_backend(tmp_path: Path) -> None:
    args = parse_args(
        [
            "decide",
            str(CLEAN_RUN),
            "--questions",
            str(EXAMPLES / "decision-questions.json"),
            "--decisions-backend",
            "mediapipe",
        ]
    )
    with pytest.raises(SystemExit) as caught:
        asyncio.run(async_main(args))
    assert caught.value.code == 2


def test_decide_fail_on_error_exits_1(tmp_path: Path) -> None:
    """An exhausted replay file is a backend error; the flag turns it into exit 1."""
    short_replay = tmp_path / "one.json"
    bodies = json.loads((EXAMPLES / "recorded-decisions.json").read_text())
    short_replay.write_text(json.dumps(bodies[:1]))

    base = [
        "decide",
        str(CLEAN_RUN),
        "--questions",
        str(EXAMPLES / "decision-questions.json"),
        "--decisions-backend",
        "replay",
        "--replay-file",
        str(short_replay),
        "--decisions-out",
        str(tmp_path / "out.jsonl"),
    ]

    with pytest.raises(SystemExit) as caught:
        asyncio.run(async_main(parse_args([*base, "--decisions-fail-on-error"])))
    assert caught.value.code == 1

    # Without the flag the failure is recorded and the run succeeds.
    asyncio.run(async_main(parse_args(base)))
    records = [
        json.loads(line)["additional_metadata"]["decisions"]
        for line in (tmp_path / "out.jsonl").read_text().splitlines()
    ]
    assert "answers" in records[0]
    assert records[1]["error"].startswith("DecisionBackendError:")

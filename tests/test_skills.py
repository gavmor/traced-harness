"""Comprehensive tests for Agent Skill support, OpenTelemetry tracing, and REPL commands."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traced_harness.agent import (
    ToolExecution,
    TurnResult,
    create_agent,
    execute_turn,
    log_turn_to_session,
)
from traced_harness.main import parse_args
from traced_harness.skills import (
    Skill,
    activate_skill,
    build_skill_instructions,
    clear_skills_registry,
    discover_skills,
    get_registered_skills,
    load_skill_file,
    register_skills,
    scan_skills_dirs,
)
from traced_harness.telemetry import (
    SKILL_ACTIVATE_SPAN,
    SKILL_CHARS_LOADED_ATTR,
    SKILL_DISCOVERY_SPAN,
    SKILL_MODE_ATTR,
    SKILL_NAME_ATTR,
    SKILL_PATH_ATTR,
    setup_telemetry,
)


@pytest.fixture
def memory_tracer():
    """Setup an in-memory tracer to inspect OpenTelemetry spans."""
    provider = setup_telemetry("test-harness")
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    yield exporter


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure clean global skill registry for each test."""
    clear_skills_registry()
    yield
    clear_skills_registry()


# ==========================================
# 1. Format & Parsing Tests
# ==========================================


def test_load_skill_file_with_frontmatter(tmp_path: Path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: test-skill\n"
        "description: A test skill for code inspection.\n"
        "version: 1.0.0\n"
        "custom_field: hello\n"
        "---\n\n"
        "# Detailed Instructions\n"
        "Perform deep inspection of files.\n",
        encoding="utf-8",
    )

    skill = load_skill_file(skill_file)
    assert skill.name == "test-skill"
    assert skill.description == "A test skill for code inspection."
    assert "Perform deep inspection of files." in skill.content
    assert skill.metadata["version"] == "1.0.0"
    assert skill.metadata["custom_field"] == "hello"
    assert skill.path == skill_file.resolve()
    assert not skill.active
    assert skill.chars_loaded == len(skill.content)


def test_load_skill_file_fallback_name(tmp_path: Path):
    skill_dir = tmp_path / "my-cool-skill"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\ndescription: Fallback name from directory.\n---\nBody content here.\n",
        encoding="utf-8",
    )

    skill = load_skill_file(skill_file)
    assert skill.name == "my-cool-skill"
    assert skill.description == "Fallback name from directory."
    assert skill.content == "Body content here."


def test_load_skill_from_directory(tmp_path: Path):
    skill_dir = tmp_path / "bundle-skill"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: bundle-skill\n"
        "description: Bundled skill with folders.\n"
        "---\n"
        "Bundle instructions.\n",
        encoding="utf-8",
    )
    # Add bundle folders
    (skill_dir / "scripts").mkdir()
    (skill_dir / "references").mkdir()
    (skill_dir / "examples").mkdir()

    skill = load_skill_file(skill_dir)
    assert skill.name == "bundle-skill"
    assert "scripts_dir" in skill.metadata
    assert "references_dir" in skill.metadata
    assert "examples_dir" in skill.metadata


def test_load_skill_file_invalid_yaml(tmp_path: Path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        "---\n: invalid : : yaml :\n---\nInstructions\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid YAML frontmatter"):
        load_skill_file(skill_file)


def test_load_skill_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_skill_file(tmp_path / "nonexistent")


# ==========================================
# 2. Discovery & Hierarchy Tests
# ==========================================


def test_scan_skills_dirs(tmp_path: Path, memory_tracer: InMemorySpanExporter):
    dir1 = tmp_path / "skills1"
    dir1.mkdir()
    sub1 = dir1 / "skill-a"
    sub1.mkdir()
    (sub1 / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: Skill A\n---\nContent A\n"
    )

    dir2 = tmp_path / "skills2"
    dir2.mkdir()
    sub2 = dir2 / "skill-b"
    sub2.mkdir()
    (sub2 / "SKILL.md").write_text(
        "---\nname: skill-b\ndescription: Skill B\n---\nContent B\n"
    )

    # Direct markdown file
    (dir2 / "skill-c.md").write_text(
        "---\nname: skill-c\ndescription: Skill C\n---\nContent C\n"
    )

    skills = scan_skills_dirs([dir1, dir2])
    skill_names = {s.name for s in skills}
    assert skill_names == {"skill-a", "skill-b", "skill-c"}

    # Verify skill.discovery span was recorded
    spans = memory_tracer.get_finished_spans()
    discovery_spans = [s for s in spans if s.name == SKILL_DISCOVERY_SPAN]
    assert len(discovery_spans) == 1
    assert discovery_spans[0].attributes["skills.discovered_count"] == 3


def test_discovery_hierarchy_priority(tmp_path: Path):
    # Priority: explicit skills_dir > local .skills/skills > global ~/.agents/skills
    base_dir = tmp_path / "project"
    base_dir.mkdir()
    local_dir = base_dir / "skills"
    local_dir.mkdir()
    sub_local = local_dir / "review"
    sub_local.mkdir()
    (sub_local / "SKILL.md").write_text(
        "---\nname: review\ndescription: Local project review\n---\nLocal review content\n"
    )

    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir()
    sub_explicit = explicit_dir / "review"
    sub_explicit.mkdir()
    (sub_explicit / "SKILL.md").write_text(
        "---\nname: review\ndescription: Explicit override review\n---\nExplicit review content\n"
    )

    skills = discover_skills(
        skills_dir=explicit_dir,
        base_dir=base_dir,
        global_dir=tmp_path / "empty_global",
    )
    assert len(skills) == 1
    assert skills[0].name == "review"
    assert skills[0].description == "Explicit override review"
    assert skills[0].content == "Explicit review content"


def test_discover_skills_no_skills_flag(tmp_path: Path):
    base_dir = tmp_path / "project"
    base_dir.mkdir()
    local_dir = base_dir / "skills"
    local_dir.mkdir()
    sub_local = local_dir / "local-skill"
    sub_local.mkdir()
    (sub_local / "SKILL.md").write_text(
        "---\nname: local-skill\ndescription: Local\n---\nContent\n"
    )

    skills = discover_skills(
        no_skills=True,
        base_dir=base_dir,
    )
    assert len(skills) == 0


def test_discover_skills_preload_by_name(
    tmp_path: Path, memory_tracer: InMemorySpanExporter
):
    base_dir = tmp_path / "project"
    base_dir.mkdir()
    local_dir = base_dir / "skills"
    local_dir.mkdir()
    sub_local = local_dir / "my-skill"
    sub_local.mkdir()
    (sub_local / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Preload me\n---\nPreloaded instructions\n"
    )

    skills = discover_skills(
        skill_args=["my-skill"],
        base_dir=base_dir,
        global_dir=tmp_path / "empty_global",
    )
    assert len(skills) == 1
    assert skills[0].name == "my-skill"
    assert skills[0].active is True

    # Verify preload span
    spans = memory_tracer.get_finished_spans()
    activate_spans = [s for s in spans if s.name == SKILL_ACTIVATE_SPAN]
    assert len(activate_spans) == 1
    span = activate_spans[0]
    assert span.attributes[SKILL_NAME_ATTR] == "my-skill"
    assert span.attributes[SKILL_MODE_ATTR] == "preload"
    assert span.attributes[SKILL_CHARS_LOADED_ATTR] == len(skills[0].content)


def test_discover_skills_preload_by_path(
    tmp_path: Path, memory_tracer: InMemorySpanExporter
):
    skill_file = tmp_path / "custom-skill.md"
    skill_file.write_text(
        "---\nname: custom-path-skill\ndescription: Custom file path\n---\nInstructions by path\n"
    )

    skills = discover_skills(
        skill_args=[str(skill_file)],
        no_skills=True,
    )
    assert len(skills) == 1
    assert skills[0].name == "custom-path-skill"
    assert skills[0].active is True

    spans = memory_tracer.get_finished_spans()
    activate_spans = [s for s in spans if s.name == SKILL_ACTIVATE_SPAN]
    assert len(activate_spans) == 1
    assert activate_spans[0].attributes[SKILL_MODE_ATTR] == "preload"


def test_discover_skills_preload_missing_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="was not found"):
        discover_skills(skill_args=["nonexistent-skill"], no_skills=True)


# ==========================================
# 3. Dynamic Tool Invocation & Agno Integration
# ==========================================


def test_activate_skill_dynamic_tool(memory_tracer: InMemorySpanExporter):
    skill = Skill(
        name="formatter",
        description="Formats code.",
        content="Always indent with 4 spaces.",
        path=Path("/tmp/formatter/SKILL.md"),
        active=False,
    )
    register_skills([skill])

    result = activate_skill("formatter")
    assert "Always indent with 4 spaces." in result
    assert skill.active is True

    spans = memory_tracer.get_finished_spans()
    activate_spans = [s for s in spans if s.name == SKILL_ACTIVATE_SPAN]
    assert len(activate_spans) == 1
    span = activate_spans[0]
    assert span.attributes[SKILL_NAME_ATTR] == "formatter"
    assert span.attributes[SKILL_PATH_ATTR] == str(skill.path)
    assert span.attributes[SKILL_MODE_ATTR] == "dynamic_tool"
    assert span.attributes[SKILL_CHARS_LOADED_ATTR] == len(skill.content)


def test_activate_skill_not_found():
    register_skills([])
    result = activate_skill("unknown")
    assert "Error: Skill 'unknown' not found" in result


def test_build_skill_instructions():
    s1 = Skill(
        name="active-skill",
        description="Active skill desc",
        content="Active skill content",
        path=Path("/tmp/s1"),
        active=True,
    )
    s2 = Skill(
        name="available-skill",
        description="Available skill desc",
        content="Available skill content",
        path=Path("/tmp/s2"),
        active=False,
    )

    instructions = build_skill_instructions([s1, s2])
    assert "# Pre-Activated Skills" in instructions
    assert "## Skill: active-skill" in instructions
    assert "Active skill content" in instructions
    assert "# Available Skills (Dynamic Loading)" in instructions
    assert "- **available-skill**: Available skill desc" in instructions
    # Available skill content should not be dumped upfront
    assert "Available skill content" not in instructions


@pytest.mark.asyncio
async def test_create_agent_with_skills():
    skill = Skill(
        name="unit-tester",
        description="Generates unit tests.",
        content="Write thorough tests.",
        path=Path("/tmp/test/SKILL.md"),
        active=False,
    )
    agent = await create_agent(client=None, skills=[skill])
    assert agent is not None
    assert agent.tools is not None
    assert activate_skill in agent.tools
    assert agent.instructions is not None
    assert "unit-tester" in str(agent.instructions)
    registered = get_registered_skills()
    assert "unit-tester" in registered


# ==========================================
# 4. Session Trace Logging Tests
# ==========================================


def test_log_turn_to_session_includes_skills(tmp_path: Path):
    session_file = tmp_path / "trace_test.jsonl"
    turn = TurnResult(
        prompt="Write a review",
        output="Review complete",
        tools_called=[
            ToolExecution(
                name="activate_skill",
                input_parameters={"name": "reviewer"},
                output="Instructions loaded",
            )
        ],
        session_id="sess-123",
        timestamp="2026-10-03T12:00:00Z",
        skills_active=["reviewer"],
    )

    log_turn_to_session(turn, session_file, mcp_label="test-mcp")

    assert session_file.exists()
    line = session_file.read_text(encoding="utf-8").strip()
    data = json.loads(line)

    assert data["input"] == "Write a review"
    assert data["actual_output"] == "Review complete"
    assert len(data["tools_called"]) == 1
    assert data["tools_called"][0]["name"] == "activate_skill"
    assert data["additional_metadata"]["skills_active"] == ["reviewer"]
    assert data["additional_metadata"]["session_id"] == "sess-123"


# ==========================================
# 5. CLI Arguments Tests
# ==========================================


def test_parse_args_skills_options():
    args = parse_args(
        [
            "--skills-dir",
            "/tmp/skills",
            "--skill",
            "skill1",
            "--skill",
            "/path/to/skill2/SKILL.md",
            "--no-skills",
        ]
    )
    assert args.skills_dir == Path("/tmp/skills")
    assert args.skill == ["skill1", "/path/to/skill2/SKILL.md"]
    assert args.skills == ["skill1", "/path/to/skill2/SKILL.md"]
    assert args.no_skills is True


# ==========================================
# 6. REPL Commands Tests
# ==========================================

from traced_harness.repl import run_repl


@pytest.mark.asyncio
async def test_repl_commands_toggle_and_status(tmp_path: Path):
    skill = Skill(
        name="repl-skill",
        description="A skill for testing REPL.",
        content="REPL instructions.",
        path=tmp_path / "SKILL.md",
        active=False,
    )
    commands = iter(["/skills", "/skill repl-skill", "/status", "exit"])

    with patch("builtins.input", lambda prompt: next(commands)):
        await run_repl(
            client=None,
            mcp_label="none",
            session_dir=tmp_path,
            skills=[skill],
        )

    # After running /skill repl-skill, it should have been toggled to True
    assert skill.active is True


@pytest.mark.asyncio
async def test_repl_command_view_does_not_toggle(tmp_path: Path):
    skill = Skill(
        name="view-skill",
        description="View only.",
        content="View content.",
        path=tmp_path / "SKILL.md",
        active=False,
    )
    commands = iter(["/skill view-skill view", "exit"])

    with patch("builtins.input", lambda prompt: next(commands)):
        await run_repl(
            client=None,
            mcp_label="none",
            session_dir=tmp_path,
            skills=[skill],
        )

    assert skill.active is False


@pytest.mark.asyncio
async def test_execute_turn_integration(tmp_path: Path):
    from unittest.mock import MagicMock

    skill = Skill(
        name="test-runner",
        description="Runs tests.",
        content="Run pytest.",
        path=tmp_path / "SKILL.md",
        active=True,
    )

    mock_agent = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = "I ran the tests"
    mock_tool = MagicMock()
    mock_tool.tool_name = "activate_skill"
    mock_tool.tool_args = {"name": "test-runner"}
    mock_tool.result = "Instructions loaded"
    mock_resp.tools = [mock_tool]

    mock_agent.arun = AsyncMock(return_value=mock_resp)
    mock_agent.model.id = "test-model"

    session_file = tmp_path / "trace_turn.jsonl"
    turn = await execute_turn(
        "run tests",
        agent=mock_agent,
        session_id="s1",
        session_file=session_file,
        mcp_label="mock-mcp",
        skills=[skill],
    )

    assert turn.output == "I ran the tests"
    assert "test-runner" in turn.skills_active
    assert len(turn.tools_called) == 1
    assert turn.tools_called[0].name == "activate_skill"

    # Check session file output
    assert session_file.exists()
    trace_data = json.loads(session_file.read_text(encoding="utf-8").strip())
    assert trace_data["additional_metadata"]["skills_active"] == ["test-runner"]
    assert trace_data["tools_called"][0]["name"] == "activate_skill"

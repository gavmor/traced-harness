"""Tests for traced-agent CLI, target parsing, and telemetry setup."""

from mcp.client.stdio import StdioServerParameters

from traced_harness.client import parse_mcp_target
from traced_harness.main import parse_args
from traced_harness.telemetry import get_tracer, setup_telemetry


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.prompt is None
    assert not args.interactive
    assert args.mcp is None
    assert args.server is None


def test_parse_args_custom() -> None:
    args = parse_args(
        ["hello world", "--mcp", "npx test", "-m", "custom-model", "-s", "sess-1"]
    )
    assert args.prompt == "hello world"
    assert args.mcp == "npx test"
    assert args.model == "custom-model"
    assert args.session_id == "sess-1"


def test_parse_mcp_target_stdio() -> None:
    target, label = parse_mcp_target(mcp_cmd="uv run my_mcp")
    assert isinstance(target, StdioServerParameters)
    assert target.command == "uv"
    assert target.args == ["run", "my_mcp"]
    assert label == "stdio:uv run my_mcp"


def test_parse_mcp_target_url() -> None:
    target, label = parse_mcp_target(url="http://localhost:8000/sse")
    assert target == "http://localhost:8000/sse"
    assert label == "url:http://localhost:8000/sse"


def test_telemetry_tracer() -> None:
    setup_telemetry(service_name="test-service")
    tracer = get_tracer("test.tracer")
    assert tracer is not None
    with tracer.start_as_current_span("test.span") as span:
        assert span.is_recording()

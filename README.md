# traced-harness

Minimal, production-ready agent evaluation and testing harness powered by [Agno](https://github.com/agno-agi/agno) and instrumented with OpenTelemetry to connect to any Model Context Protocol (MCP) server.

## Features

- **Lightweight & Generic**: Connects to any MCP server over `stdio` (e.g. `uv run <project> mcp`), Python in-process (`module:server`), or HTTP/SSE.
- **Native Async MCP**: Uses `agno.tools.mcp.MCPTools` to directly discover and invoke tools without custom trajectory parsing.
- **OpenTelemetry Instrumentation**: Automatically records `agent.turn` parent spans and `tool_call.<name>` child spans with parameters, outputs, and model metrics.
- **Rich Terminal REPL**:
  - `/new [id]` - Reset conversation context and rotate session trace log without process restart.
  - `/status` - Inspect current MCP target, loaded tools count, model, and active trace log.
  - `/tools` - List all tools exposed by the connected MCP server.
  - `/clear` - Clear terminal screen.
- **Durable Trace Logging**: Writes clean turn-by-turn trace records into `sessions/trace_<id>.jsonl` for offline evaluation.

## Installation

```bash
# Install globally via uv
uv tool install --editable /home/user/Desktop/gavmor/traced-harness
```

This registers `traced-harness` (and aliases `traced-agent`, `traced-agno`, `traced-dspy`) on `$PATH`.

## Usage

### Auto-detection
If running in a project with `.mcp.json` or `MCP_SERVER` defined, simply run:
```bash
traced-harness
```

### Connect over stdio
```bash
traced-harness --mcp "uv run foxhole mcp"
```

### One-shot query
```bash
traced-harness "What is the health of a Devitt Mark III?"
```

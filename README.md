# traced-agent

Minimal DSPy ReAct agent instrumented with OpenTelemetry and designed to connect to any Model Context Protocol (MCP) server.

## Features

- **Minimal & Generic**: Connects to any MCP server over `stdio`, Python in-process (`module:server`), or HTTP/SSE.
- **OpenTelemetry Instrumentation**: Automatically emits `agent.turn` parent spans and `tool_call.<name>` child spans with full parameter payloads and completion metadata.
- **Interactive REPL**:
  - `/new [id]` - Reset conversation context and rotate session trace log without process restart.
  - `/status` - Inspect current MCP target, discovered tools, active model, and trace log location.
  - `/tools` - List all tools exposed by the connected MCP server.
  - `/clear` - Clear terminal screen.
- **Durable Trace Logging**: Writes clean, sequential turn traces into `sessions/trace_<id>.jsonl` for offline evaluation.

## Installation

```bash
# Run directly via uv
uv run traced-agent

# Or install as a global tool
uv tool install --editable .
```

## Usage

### Connect to an MCP server over stdio
```bash
traced-agent --mcp "uv run foxhole mcp"
```

### Connect to an in-process Python MCP server
```bash
traced-agent --server "foxhole.server:server"
```

### One-shot query
```bash
traced-agent --mcp "uv run foxhole mcp" "What are the specs of the Colonial Bardel?"
```

### Auto-detection
If running in a project with an `.mcp.json` or standard MCP server in the environment (`MCP_SERVER`), `traced-agent` auto-discovers and connects directly.

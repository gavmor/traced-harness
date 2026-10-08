# traced-harness

A test bed harness for evaluating agentic peripherals (MCP, skills, etc.)

## Features

- **Lightweight & Generic**: Connects to any MCP server over `stdio` (e.g. `uv run <project> mcp`), Python in-process (`module:server`), or HTTP/SSE.
- **Agent Skill Support (`SKILL.md`)**: Progressive loading and dynamic dispatch (`activate_skill`) for standard `SKILL.md` bundles with multi-tier discovery (`--skills-dir`, `./.skills/`, `./skills/`, `~/.agents/skills/`).
- **OpenTelemetry Instrumentation**: Automatically records `agent.turn` parent spans, `tool_call.<name>` child spans, `skill.discovery` scanning spans, and `skill.activate` spans with full parameter and metric attributes.
- **Rich Terminal REPL**:
  - `/new [id]` - Reset conversation context and rotate session trace log without process restart.
  - `/status` - Inspect current MCP target, loaded tools count, discovered/active skills, model, and active trace log.
  - `/tools` - List all tools exposed by the connected MCP server.
  - `/skills` - List all discovered skills with active/available status and descriptions.
  - `/skill <name>` - Inspect full markdown and toggle skill activation in current conversation.
  - `/clear` - Clear terminal screen.
- **Durable Trace Logging**: Writes clean turn-by-turn trace records into `sessions/trace_<id>.jsonl` including active skills and tool execution records for offline evaluation.
- **Memory Providers (`traced_harness.memory`)**: Durable-memory plugins as a first-class peripheral alongside MCP and skills — a `MemoryProviderAdapter` lifecycle ABC, a tool/context-hook contract injected into the system prompt, and telemetry for retrieval latency, injected-token overhead, and consolidation cost (wall/CPU time + store growth).
- **Multi-Session Benchmarking (`SessionRunner`)**: Runs an ordered `Scenario` of sessions with distinct `session_id`s, so in-context history is deliberately *not* carried across them and cross-session behaviour must come from the peripheral. Fires a `between_sessions` hook between them and measures its cost. The runner is domain-blind: it depends only on a structural `PeripheralLifecycle` protocol, so it works for any peripheral, not just memory.
- **Decision Backends (`traced_harness.decisions`)**: Turns a recorded trace into calibrated numbers — predicates with probabilities, choices with confidences, scores over ordered levels — by annotating it afterwards into `<stem>.decided.jsonl`. The harness ships the mechanism (question/answer types, a `DecisionBackend` protocol, a `decision.turn` span, the trace slot) and never a rubric: which questions to ask is always the caller's. Includes an offline replay backend, so the whole path runs with no credential.

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

### Agent Skills
```bash
# Pre-activate a specific skill by name or path
traced-harness --skill my-skill-name
traced-harness --skill ./path/to/my-skill/SKILL.md

# Provide custom skills directory
traced-harness --skills-dir ./custom-skills/

# Disable automatic local and global skill discovery
traced-harness --no-skills
```

### Decision backends

Annotate a recorded trace with calibrated decisions. This exact command runs
offline, with no API key:

```bash
traced-harness decide tests/fixtures/traces/clean_run.jsonl \
  --questions docs/examples/decision-questions.json \
  --decisions-backend replay \
  --replay-file docs/examples/recorded-decisions.json \
  --decisions-out /tmp/clean_run.decided.jsonl
```

It writes one annotated line per input line — the input itself is never
modified — with each turn's answers under
`additional_metadata.decisions`. Omit `--decisions-out` and the annotated copy
lands beside the input as `<stem>.decided.jsonl`. The `openai` backend talks to
the OpenAI Decisions endpoint instead and reads `OPENAI_API_KEY` at call time;
that path is **unexercised** (see the note in the docs).

Full reference, the questions-file format, the record shape and a real sample
line: [docs/decisions.md](docs/decisions.md).



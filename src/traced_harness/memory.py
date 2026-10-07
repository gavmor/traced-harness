"""Memory providers as a first-class agentic peripheral.

``traced-harness`` evaluates agentic peripherals. MCP servers and skills are
already first-class named concepts; durable-memory providers are the third, and
this module gives them the same treatment.

Layering
--------
The harness *core* stays domain-blind: :class:`~traced_harness.session_runner.SessionRunner`
depends only on the structural ``PeripheralLifecycle`` protocol and never
imports this module. What lives here is batteries — a generic memory
abstraction any memory study can build on, rather than rebuilding it:

* :class:`MemoryToolContract` — what a provider contributes to the agent:
  explicit recall tools, implicit context-engine hooks, and a system-prompt
  contract.
* :class:`MemoryProviderAdapter` — the lifecycle ABC (setup / consolidate /
  teardown). It satisfies ``PeripheralLifecycle`` via :meth:`between_sessions`.
* Memory telemetry — injection overhead, retrieval spans, and the
  consolidation measurement, all layered over the generic
  :func:`~traced_harness.telemetry.measured_span`.
* Prompt/registry wiring — :func:`register_memory_tools` and
  :func:`build_memory_instructions`, built on the generic skill registry.
* :func:`make_memory_session_runner` — a ``SessionRunner`` pre-wired with the
  memory vocabulary, so callers do not hand-thread metadata keys.

*Which* providers you benchmark is not the harness's business: concrete
adapters (Cashew, Chronicle, Memex8, Nachos, ...) belong to the study that
compares them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openinference.instrumentation import (
    Document,
    get_input_attributes,
    get_retriever_attributes,
    get_span_kind_attributes,
)

from traced_harness.session_runner import SessionRunner, TurnExecutor
from traced_harness.skills import (
    build_tool_instructions,
    clear_external_tools,
    get_registered_external_tools,
    register_external_tools,
)
from traced_harness.telemetry import (
    append_turn_record,
    estimate_tokens,
    get_tracer,
    measured_span,
)

__all__ = [
    "MEMORY_METADATA_KEY",
    "MemoryProviderAdapter",
    "MemoryToolContract",
    "RetrievalMeasurement",
    "build_memory_instructions",
    "clear_memory_tools",
    "consolidation_span",
    "get_registered_memory_tools",
    "make_memory_session_runner",
    "make_memory_tool_hook",
    "make_memory_turn_executor",
    "record_memory_injection",
    "register_memory_tools",
    "retrieval_span",
]

#: Slot in a turn's ``additional_metadata`` where memory records are written.
MEMORY_METADATA_KEY = "memory"

#: Turn-metadata keys this module writes through the harness's turn collector.
RETRIEVALS_KEY = "retrievals"
INJECTION_KEY = "injection"
PROVIDER_KEY = "provider"


@dataclass
class MemoryToolContract:
    """What a memory provider contributes to the agent at registration time.

    ``tools`` are explicit tool names the agent may call; ``context_hooks`` are
    implicit context-engine integration points (prefetch/compaction); and
    ``system_prompt`` is the durable-memory contract injected into the prompt.

    ``retrieval_tools`` narrows ``tools`` to the ones that *read* memory.
    Retrieval is what the benchmark counts and times — a write is not a
    recall, and counting one inflates the column that is supposed to prove
    the provider was consulted. A provider whose tools are all reads can
    leave it empty, and :meth:`recall_tools` falls back to ``tools``.
    """

    provider: str
    tools: list[str] = field(default_factory=list)
    context_hooks: list[str] = field(default_factory=list)
    system_prompt: str = ""
    retrieval_tools: list[str] = field(default_factory=list)

    def recall_tools(self) -> list[str]:
        """The tools whose calls count as retrievals."""
        return list(self.retrieval_tools or self.tools)


class MemoryProviderAdapter(ABC):
    """Lifecycle interface for a single memory provider under evaluation."""

    #: Short provider id, used in telemetry spans and trace metadata.
    name: str = "memory"

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        #: Recorded intended side effects (populated in dry_run, and also in
        #: live mode for observability). Each entry is ``(kind, detail)``.
        self.actions: list[tuple[str, Any]] = []
        #: Files/dirs whose byte footprint consolidation telemetry samples.
        self.store_paths: list[str] = []

    # -- lifecycle --------------------------------------------------------
    @abstractmethod
    def setup(self, workspace_dir: Path) -> dict[str, Any]:
        """Initialize ephemeral config, clean DBs, and return agent env/tool
        configs.

        Returns a dict with at least ``env`` (process env overlay),
        ``contract`` (:class:`MemoryToolContract`), and ``store_paths``.
        """

    @abstractmethod
    def trigger_consolidation(self) -> None:
        """Trigger offline sleep/consolidation routines between sessions."""

    def between_sessions(self) -> None:
        """Satisfy :class:`~traced_harness.session_runner.PeripheralLifecycle`.

        ``SessionRunner`` calls this between sessions without knowing what it
        means; for a memory provider it is the offline consolidation pass.
        """
        self.trigger_consolidation()

    @abstractmethod
    def teardown(self) -> None:
        """Clean containers, DB files, and lingering processes.

        Implementations own their on-disk state: ``SessionRunner`` will not
        delete a provider's stores, because it does not know what they mean.
        :meth:`_purge` is available as a helper.
        """

    # -- agent registration ----------------------------------------------
    @abstractmethod
    def contract(self) -> MemoryToolContract:
        """Return the tool/context-hook/prompt contract for agent wiring."""

    # -- shared helpers ---------------------------------------------------
    def _record(self, kind: str, detail: Any) -> None:
        self.actions.append((kind, detail))

    def _run(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess | None:
        """Run a subprocess unless in dry_run; always records the command."""
        self._record("exec", cmd)
        if self.dry_run:
            return None
        return subprocess.run(cmd, check=True, **kwargs)

    def _post_json(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any] | None:
        """POST JSON unless in dry_run; always records the request."""
        self._record("http_post", {"url": url, "payload": payload})
        if self.dry_run:
            return None
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        for k, v in headers.items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}

    @staticmethod
    def _purge(*paths: str | Path) -> None:
        for p in paths:
            path = Path(p)
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Memory telemetry.
# ---------------------------------------------------------------------------

MEMORY_TRACER_NAME = "traced.harness.memory"

MEMORY_INJECTION_SPAN = "memory.injection"
MEMORY_RETRIEVAL_SPAN = "memory.retrieval"
MEMORY_CONSOLIDATION_SPAN = "memory.consolidation"

MEMORY_PROVIDER_ATTR = "memory.provider"
MEM_BASE_TOKENS_ATTR = "memory.injection.base_prompt_tokens"
MEM_INJECTED_TOKENS_ATTR = "memory.injection.post_prompt_tokens"
MEM_OVERHEAD_TOKENS_ATTR = "memory.injection.overhead_tokens"
MEM_QUERY_ATTR = "memory.retrieval.query"
MEM_PASSAGES_ATTR = "memory.retrieval.passage_count"
MEM_LATENCY_ATTR = "memory.retrieval.latency_ms"

#: Where agno stows MCP's structured tool output on a ``ToolResult``.
MCP_STRUCTURED_CONTENT_KEY = "structured_content"


def record_memory_injection(
    base_prompt_tokens: int,
    post_injection_tokens: int,
    provider: str = "",
) -> dict[str, int]:
    """Record the prompt-token overhead added by memory injection on a turn.

    ``base_prompt_tokens`` is the pre-retrieval prompt; ``post_injection_tokens``
    is the prompt after the provider injected recalled context. The returned
    dict is meant to live under ``additional_metadata["memory"]``.
    """
    overhead = max(0, post_injection_tokens - base_prompt_tokens)
    tracer = get_tracer(MEMORY_TRACER_NAME)
    with tracer.start_as_current_span(MEMORY_INJECTION_SPAN) as span:
        span.set_attribute(MEMORY_PROVIDER_ATTR, provider)
        span.set_attribute(MEM_BASE_TOKENS_ATTR, base_prompt_tokens)
        span.set_attribute(MEM_INJECTED_TOKENS_ATTR, post_injection_tokens)
        span.set_attribute(MEM_OVERHEAD_TOKENS_ATTR, overhead)
    return {
        "base_tokens": base_prompt_tokens,
        "injected_tokens": post_injection_tokens,
        "overhead_tokens": overhead,
    }


@dataclass
class RetrievalMeasurement:
    """Mutable handle yielded by :func:`retrieval_span`.

    The caller sets ``documents`` (or the ``passages`` convenience alias, or
    ``passage_count``) inside the ``with`` block; latency is measured
    automatically on exit.

    ``documents`` are OpenInference :class:`~openinference.instrumentation.Document`
    mappings — ``content`` / ``id`` / ``score`` / ``metadata``. That is the
    upstream type, not a local mirror of it, so the span attributes come from
    ``get_retriever_attributes`` rather than a parallel encoder here.
    """

    query: str
    provider: str = ""
    documents: list[Document] = field(default_factory=list)
    passage_count: int | None = None
    latency_ms: float = 0.0

    @property
    def passages(self) -> list[str]:
        """Document contents, for callers that only want the text."""
        return [str(d.get("content", "")) for d in self.documents]

    @passages.setter
    def passages(self, values: list[str]) -> None:
        self.documents = [
            Document(content=str(v)) for v in values if str(v).strip()
        ]

    @property
    def count(self) -> int:
        if self.passage_count is not None:
            return self.passage_count
        return len(self.documents)

    def as_record(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "provider": self.provider,
            "passage_count": self.count,
            "documents": [dict(d) for d in self.documents],
            # Retained so trace consumers that only read text keep working.
            "passages": self.passages,
            "latency_ms": round(self.latency_ms, 3),
        }


@contextmanager
def retrieval_span(
    query: str,
    provider: str = "",
) -> Iterator[RetrievalMeasurement]:
    """Wrap an explicit or prefetch retrieval call in a telemetry span.

    Captures the query string, retrieved passage count, and millisecond
    latency::

        with retrieval_span("where does marc live", "cashew") as r:
            r.passages = store.query(...)
    """
    measurement = RetrievalMeasurement(query=query, provider=provider)
    tracer = get_tracer(MEMORY_TRACER_NAME)
    start = time.perf_counter()
    with tracer.start_as_current_span(MEMORY_RETRIEVAL_SPAN) as span:
        span.set_attributes(get_span_kind_attributes("retriever"))
        span.set_attributes(get_input_attributes(query))
        span.set_attribute(MEMORY_PROVIDER_ATTR, provider)
        span.set_attribute(MEM_QUERY_ATTR, query)
        try:
            yield measurement
        finally:
            measurement.latency_ms = (time.perf_counter() - start) * 1000.0
            span.set_attributes(
                get_retriever_attributes(documents=measurement.documents)
            )
            span.set_attribute(MEM_PASSAGES_ATTR, measurement.count)
            span.set_attribute(MEM_LATENCY_ATTR, measurement.latency_ms)


@contextmanager
def consolidation_span(
    provider: str = "",
    store_paths: list[str | Path] | None = None,
):
    """Measure a consolidation run: wall + CPU time and store growth.

    Memory-domain naming over the generic
    :func:`~traced_harness.telemetry.measured_span`. ``SessionRunner`` applies
    the same measurement itself between sessions; use this when driving
    consolidation directly.
    """
    with measured_span(
        MEMORY_CONSOLIDATION_SPAN,
        {MEMORY_PROVIDER_ATTR: provider},
        watch_paths=store_paths,
        tracer_name=MEMORY_TRACER_NAME,
    ) as m:
        yield m


# ---------------------------------------------------------------------------
# Agent wiring.
# ---------------------------------------------------------------------------


def register_memory_tools(tools: list[str], provider: str = "") -> None:
    """Register active memory tool names -> provider in the global registry."""
    register_external_tools(tools, provider)


def get_registered_memory_tools() -> dict[str, str]:
    """Return the active memory-tool -> provider dispatch registry."""
    return get_registered_external_tools()


def clear_memory_tools() -> None:
    """Clear the tool registry (used between benchmark suites)."""
    clear_external_tools()


def build_memory_instructions(
    provider: str,
    tools: list[str],
    context_hooks: list[str] | None = None,
    system_prompt: str = "",
) -> str:
    """System-prompt section describing the active memory provider.

    Covers its explicit recall tools and any implicit context hooks.
    """
    return build_tool_instructions(
        provider,
        tools,
        context_hooks=context_hooks,
        system_prompt=system_prompt,
        heading="Memory Provider",
    )


def make_memory_session_runner(
    adapter: MemoryProviderAdapter,
    turn_executor: TurnExecutor,
    workspace_dir: str | Path,
    trace_dir: str | Path | None = None,
) -> SessionRunner:
    """A ``SessionRunner`` pre-wired with the memory vocabulary.

    The runner itself stays domain-blind; this just saves every memory study
    from hand-threading the same metadata key and span name.
    """
    return SessionRunner(
        turn_executor=turn_executor,
        workspace_dir=workspace_dir,
        trace_dir=trace_dir,
        lifecycle=adapter,
        metadata_key=MEMORY_METADATA_KEY,
        between_sessions_span=MEMORY_CONSOLIDATION_SPAN,
    )


# ---------------------------------------------------------------------------
# Turn telemetry.
#
# Registering a provider's tool names is not enough to measure it: the recall
# happens inside the agent's tool loop, where neither the runner nor the
# caller can see it. These two pieces close that gap — a tool hook that times
# the real call, and a turn executor that totals what the turn cost.
# ---------------------------------------------------------------------------

#: Argument names a memory tool is likely to carry its query under, in
#: preference order. Falls back to the whole argument dict, so an unfamiliar
#: tool still records *something* identifying rather than an empty query.
QUERY_ARG_NAMES = ("query", "q", "question", "search", "text", "prompt", "key")


def _retrieval_query(arguments: dict[str, Any]) -> str:
    for name in QUERY_ARG_NAMES:
        value = arguments.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return json.dumps(arguments, default=str, sort_keys=True)


def _as_document(item: Any) -> Document | None:
    """One structured-output element -> one OpenInference document.

    Empty content yields no document at all — a provider that found nothing
    must not be recorded as having retrieved something.
    """
    if item is None:
        return None
    if isinstance(item, Mapping):
        return _mapping_document(item)
    content = str(item)
    return Document(content=content) if content.strip() else None


#: Optional document fields, with the cast each needs. ``content`` is required
#: and handled separately; ``metadata`` is a mapping, not a scalar.
_OPTIONAL_DOCUMENT_FIELDS: tuple[tuple[str, Any], ...] = (
    ("id", str),
    ("score", float),
)


def _mapping_document(item: Mapping[str, Any]) -> Document | None:
    """Read the document fields a mapping declares, casting as OpenInference wants."""
    raw = item.get("content", item.get("text"))
    content = (
        str(raw)
        if raw is not None
        else json.dumps(dict(item), default=str, sort_keys=True)
    )
    if not content.strip():
        return None
    doc = Document(content=content)
    for key, cast in _OPTIONAL_DOCUMENT_FIELDS:
        value = item.get(key)
        if value is not None:
            doc[key] = cast(value)  # type: ignore[literal-required]
    metadata = item.get("metadata")
    if isinstance(metadata, Mapping):
        doc["metadata"] = dict(metadata)
    return doc


def _documents(result: Any) -> list[Document]:
    """Read a memory tool's return value as structured retrieval documents.

    This reads the *specified* boundary rather than guessing at shapes. Agno
    hands a tool hook a ``ToolResult`` whose ``metadata`` carries MCP's
    ``structured_content`` — the tool's own declared output, validated against
    its ``outputSchema`` — alongside ``content``, the text blocks flattened to
    a string. So there are only two sources, in order of fidelity:

    1. ``structured_content``: a list is N documents, one per element;
       anything else is a single document.
    2. ``content``: the flattened text, as one document.

    A bare string (a non-MCP tool, or a test double) is one document. Nothing
    here inspects dictionary keys hoping to find the passage list, because the
    MCP result already says where it is.
    """
    structured = _structured_content(result)
    if structured is not None:
        items = structured if isinstance(structured, (list, tuple)) else [structured]
        return [d for d in (_as_document(i) for i in items) if d is not None]

    content = getattr(result, "content", result)
    doc = _as_document(content)
    return [doc] if doc is not None else []


def _structured_content(result: Any) -> Any | None:
    """MCP ``structured_content``, as agno stores it on a ``ToolResult``."""
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping):
        return metadata.get(MCP_STRUCTURED_CONTENT_KEY)
    return None


def make_memory_tool_hook(contract: MemoryToolContract) -> Any:
    """An Agno ``tool_hooks`` entry that measures this provider's recall calls.

    Wrapping the *actual* invocation is what makes ``latency_ms`` a measured
    number; reconstructing a retrieval record after the agent has returned
    would only ever report zero. Each measured call is appended to the active
    turn's ``retrievals`` list via the harness's turn collector.

    Only ``contract.recall_tools()`` is measured. Everything else — the
    provider's own writes included — passes straight through: a store or a
    delete is not a retrieval, and recording it as one would overstate how
    much memory the agent actually consulted. The hook is async because every
    memory tool reaches the agent over MCP, whose entrypoints are coroutines;
    Agno skips async hooks for synchronous tools, which by construction are
    never the provider's.
    """
    tool_names = set(contract.recall_tools())
    provider = contract.provider

    async def memory_retrieval_hook(
        function_name: str,
        function_call: Any,
        arguments: dict[str, Any],
    ) -> Any:
        if function_name not in tool_names:
            return await function_call(**arguments)
        with retrieval_span(_retrieval_query(arguments), provider) as measurement:
            result = await function_call(**arguments)
            measurement.documents = _documents(result)
        append_turn_record(RETRIEVALS_KEY, measurement.as_record())
        return result

    return memory_retrieval_hook


def make_memory_turn_executor(
    agent: Any,
    adapter: MemoryProviderAdapter,
) -> TurnExecutor:
    """A ``TurnExecutor`` whose turns carry this provider's memory telemetry.

    Drop-in replacement for
    :func:`~traced_harness.session_runner.make_agno_turn_executor` when the
    peripheral under test is a memory provider. The agent must have been built
    by :func:`~traced_harness.agent.create_agent` with the same adapter, so its
    retrieval hook is installed.

    What ends up on each turn's ``metadata``:

    ``retrievals``
        one record per measured memory-tool call (written by the hook).
    ``injection``
        the turn's prompt-token overhead: the provider's standing system-prompt
        contract plus the passages it put into the context this turn. That is
        what a memory provider costs per turn over asking the model cold.
    ``generated_tokens``
        written by ``execute_turn`` from the model's own usage metrics.
    """
    from traced_harness.agent import execute_turn

    contract = adapter.contract()
    provider = contract.provider
    # The contract prompt is re-injected on every turn, so it is a per-turn
    # cost, not a one-off.
    contract_tokens = estimate_tokens(
        build_memory_instructions(
            contract.provider,
            contract.tools,
            contract.context_hooks,
            contract.system_prompt,
        )
    )

    async def _executor(prompt: str, session_id: str, peripheral: str) -> Any:
        turn = await execute_turn(prompt, agent, session_id)
        retrievals = turn.metadata.get(RETRIEVALS_KEY) or []
        retrieved_tokens = sum(
            estimate_tokens(passage)
            for record in retrievals
            for passage in record.get("passages", []) or []
        )
        base_tokens = estimate_tokens(prompt)
        turn.metadata[PROVIDER_KEY] = provider
        turn.metadata[INJECTION_KEY] = record_memory_injection(
            base_tokens,
            base_tokens + contract_tokens + retrieved_tokens,
            provider,
        )
        return turn

    return _executor


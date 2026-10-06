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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traced_harness.session_runner import SessionRunner, TurnExecutor
from traced_harness.skills import (
    build_tool_instructions,
    clear_external_tools,
    get_registered_external_tools,
    register_external_tools,
)
from traced_harness.telemetry import get_tracer, measured_span

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
    "record_memory_injection",
    "register_memory_tools",
    "retrieval_span",
]

#: Slot in a turn's ``additional_metadata`` where memory records are written.
MEMORY_METADATA_KEY = "memory"


@dataclass
class MemoryToolContract:
    """What a memory provider contributes to the agent at registration time.

    ``tools`` are explicit tool names the agent may call; ``context_hooks`` are
    implicit context-engine integration points (prefetch/compaction); and
    ``system_prompt`` is the durable-memory contract injected into the prompt.
    """

    provider: str
    tools: list[str] = field(default_factory=list)
    context_hooks: list[str] = field(default_factory=list)
    system_prompt: str = ""


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

    The caller sets ``passages`` (or ``passage_count``) inside the ``with``
    block; latency is measured automatically on exit.
    """

    query: str
    provider: str = ""
    passages: list[str] = field(default_factory=list)
    passage_count: int | None = None
    latency_ms: float = 0.0

    @property
    def count(self) -> int:
        if self.passage_count is not None:
            return self.passage_count
        return len(self.passages)

    def as_record(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "provider": self.provider,
            "passage_count": self.count,
            "passages": list(self.passages),
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
        span.set_attribute(MEMORY_PROVIDER_ATTR, provider)
        span.set_attribute(MEM_QUERY_ATTR, query)
        try:
            yield measurement
        finally:
            measurement.latency_ms = (time.perf_counter() - start) * 1000.0
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

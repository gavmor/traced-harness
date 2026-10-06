"""OpenTelemetry configuration and tracer provider setup for traced-agent."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_initialized = False


def setup_telemetry(service_name: str = "traced-harness") -> TracerProvider:
    """Initialize OpenTelemetry TracerProvider with optional OTLP export."""
    global _initialized
    if _initialized:
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            return provider

    service = os.environ.get("OTEL_SERVICE_NAME", service_name)
    resource = Resource.create({"service.name": service})
    provider = TracerProvider(resource=resource)

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except (ImportError, RuntimeError, ValueError):
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except (ImportError, RuntimeError, ValueError):
                pass

    trace.set_tracer_provider(provider)
    _initialized = True
    return provider


def get_tracer(name: str = "traced.harness.agno") -> trace.Tracer:
    """Return a configured OpenTelemetry tracer."""
    if not _initialized:
        setup_telemetry()
    return trace.get_tracer(name)


SKILL_DISCOVERY_SPAN = "skill.discovery"
SKILL_ACTIVATE_SPAN = "skill.activate"

SKILL_NAME_ATTR = "skill.name"
SKILL_PATH_ATTR = "skill.path"
SKILL_CHARS_LOADED_ATTR = "skill.chars_loaded"
SKILL_MODE_ATTR = "skill.mode"


def record_skill_activation_span(
    name: str,
    path: str | Path,
    chars_loaded: int,
    mode: str = "preload",
    tracer_name: str = "traced.harness.skills",
) -> None:
    """Record an OpenTelemetry span for skill activation."""
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(SKILL_ACTIVATE_SPAN) as span:
        span.set_attribute(SKILL_NAME_ATTR, name)
        span.set_attribute(SKILL_PATH_ATTR, str(path))
        span.set_attribute(SKILL_CHARS_LOADED_ATTR, chars_loaded)
        span.set_attribute(SKILL_MODE_ATTR, mode)


# ---------------------------------------------------------------------------
# Memory-provider telemetry (text-retrieval only; no multimodal/visual spans).
#
# These helpers both (a) emit OpenTelemetry spans and (b) return plain-dict
# records so the Agno turn logger can embed them under a turn's
# ``additional_metadata["memory"]`` for offline evaluation by ``eval.py``.
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
MEM_WALL_ATTR = "memory.consolidation.wall_seconds"
MEM_CPU_ATTR = "memory.consolidation.cpu_seconds"
MEM_DB_BEFORE_ATTR = "memory.consolidation.db_bytes_before"
MEM_DB_AFTER_ATTR = "memory.consolidation.db_bytes_after"
MEM_DB_GROWTH_ATTR = "memory.consolidation.db_growth_bytes"


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Rough token count heuristic (~4 chars/token) for text-only prompts.

    Callers with an exact tokenizer should pass measured counts to the
    recording helpers instead of relying on this estimate.
    """
    if not text:
        return 0
    return max(1, round(len(text) / chars_per_token))


def directory_bytes(*paths: str | Path) -> int:
    """Sum the on-disk byte footprint of the given files/directories.

    Missing paths contribute zero so the helper is safe to call before a
    store has been created.
    """
    total = 0
    for p in paths:
        path = Path(p)
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    try:
                        total += child.stat().st_size
                    except OSError:
                        pass
    return total


def record_memory_injection(
    base_prompt_tokens: int,
    post_injection_tokens: int,
    provider: str = "",
) -> dict[str, int]:
    """Record the prompt-token overhead added by memory injection on a turn.

    ``base_prompt_tokens`` is the pre-retrieval prompt; ``post_injection_tokens``
    is the prompt after the memory provider injected recalled context. The
    returned dict is meant to live under ``additional_metadata["memory"]``.
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
    latency. Usage::

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


@dataclass
class ConsolidationMeasurement:
    """Handle yielded by :func:`consolidation_span`."""

    provider: str = ""
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    db_bytes_before: int = 0
    db_bytes_after: int = 0

    @property
    def db_growth_bytes(self) -> int:
        return self.db_bytes_after - self.db_bytes_before

    def as_record(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "wall_seconds": round(self.wall_seconds, 4),
            "cpu_seconds": round(self.cpu_seconds, 4),
            "db_bytes_before": self.db_bytes_before,
            "db_bytes_after": self.db_bytes_after,
            "db_growth_bytes": self.db_growth_bytes,
        }


@contextmanager
def consolidation_span(
    provider: str = "",
    store_paths: list[str | Path] | None = None,
) -> Iterator[ConsolidationMeasurement]:
    """Measure a background consolidation run: wall + CPU time and DB growth.

    ``store_paths`` are the SQLite/vector-store files or directories whose
    byte footprint is sampled before and after the run.
    """
    paths = store_paths or []
    measurement = ConsolidationMeasurement(provider=provider)
    measurement.db_bytes_before = directory_bytes(*paths)
    tracer = get_tracer(MEMORY_TRACER_NAME)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    with tracer.start_as_current_span(MEMORY_CONSOLIDATION_SPAN) as span:
        span.set_attribute(MEMORY_PROVIDER_ATTR, provider)
        try:
            yield measurement
        finally:
            measurement.wall_seconds = time.perf_counter() - wall_start
            measurement.cpu_seconds = time.process_time() - cpu_start
            measurement.db_bytes_after = directory_bytes(*paths)
            span.set_attribute(MEM_WALL_ATTR, measurement.wall_seconds)
            span.set_attribute(MEM_CPU_ATTR, measurement.cpu_seconds)
            span.set_attribute(MEM_DB_BEFORE_ATTR, measurement.db_bytes_before)
            span.set_attribute(MEM_DB_AFTER_ATTR, measurement.db_bytes_after)
            span.set_attribute(MEM_DB_GROWTH_ATTR, measurement.db_growth_bytes)

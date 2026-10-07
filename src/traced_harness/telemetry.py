"""OpenTelemetry configuration and tracer provider setup for traced-agent."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_initialized = False


def instrument_agno(tracer_provider: TracerProvider | None = None) -> bool:
    """Emit agent/LLM/tool spans via OpenInference, under its conventions.

    Replaces hand-written ``agent.turn`` / ``tool_call.<name>`` spans carrying
    ad-hoc ``gen_ai.*`` attributes. Returns False when the optional
    instrumentor is absent — a minimal install runs, it just emits no spans.
    """
    try:
        from openinference.instrumentation.agno import AgnoInstrumentor
    except ImportError:
        return False
    AgnoInstrumentor().instrument(
        tracer_provider=tracer_provider or trace.get_tracer_provider()
    )
    return True


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
    instrument_agno(provider)
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
# Generic measurement spans.
#
# The harness measures; it does not interpret. A study layered on this
# instrument supplies its own domain vocabulary (span names, attribute keys)
# and gets back a plain-dict record it can embed in a turn's
# ``additional_metadata`` for offline evaluation.
# ---------------------------------------------------------------------------

MEASURE_TRACER_NAME = "traced.harness.measure"


# ---------------------------------------------------------------------------
# Per-turn metadata collection.
#
# A peripheral runs *inside* a turn (an MCP tool call, a context-engine hook)
# and wants its measurements to land on that turn's result. It cannot return
# them: the agent returns prose. So the harness opens a turn-scoped slot that
# anything running under the turn may write into, and hands whatever is there
# to the ``TurnResult``.
#
# The harness supplies the plumbing and nothing else: keys are the caller's
# vocabulary, values are stored verbatim, and nothing here knows what a
# "retrieval" or an "injection" is. Writes outside a turn are no-ops rather
# than errors, so a peripheral stays usable when driven directly.
# ---------------------------------------------------------------------------

_turn_metadata: ContextVar[dict[str, Any] | None] = ContextVar(
    "traced_harness_turn_metadata", default=None
)


@contextmanager
def turn_telemetry() -> Iterator[dict[str, Any]]:
    """Open a turn-scoped metadata slot and yield the dict being filled.

    Nested turns each get their own slot; the previous one is restored on
    exit, so a turn executed inside another turn cannot steal its records::

        with turn_telemetry() as collected:
            await agent.arun(prompt)
        turn.metadata = collected
    """
    payload: dict[str, Any] = {}
    token = _turn_metadata.set(payload)
    try:
        yield payload
    finally:
        _turn_metadata.reset(token)


def record_turn_metadata(key: str, value: Any) -> bool:
    """Store ``value`` under ``key`` on the active turn, if there is one.

    Returns whether a turn was open — a peripheral driven outside a turn is
    not an error, it simply has nowhere to record.
    """
    payload = _turn_metadata.get()
    if payload is None:
        return False
    payload[key] = value
    return True


def append_turn_record(key: str, record: Any) -> bool:
    """Append ``record`` to the list under ``key`` on the active turn.

    Used for repeated measurements within one turn (several tool calls, say).
    Returns whether a turn was open.
    """
    payload = _turn_metadata.get()
    if payload is None:
        return False
    bucket = payload.setdefault(key, [])
    if not isinstance(bucket, list):
        raise TypeError(
            f"turn metadata key {key!r} already holds "
            f"{type(bucket).__name__}, not a list"
        )
    bucket.append(record)
    return True


def current_turn_metadata() -> dict[str, Any] | None:
    """The active turn's metadata dict, or ``None`` outside a turn."""
    return _turn_metadata.get()


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Rough token count heuristic (~4 chars/token) for text-only prompts.

    Callers with an exact tokenizer should pass measured counts instead of
    relying on this estimate.
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


@dataclass
class SpanMeasurement:
    """Handle yielded by :func:`measured_span`.

    Wall/CPU time are captured automatically. ``bytes_before``/``bytes_after``
    are sampled from ``watch_paths`` when supplied, so a caller can measure the
    on-disk growth caused by whatever ran inside the block. ``attributes`` may
    be extended inside the block; everything there is written to the span and
    returned by :meth:`as_record`.
    """

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    bytes_before: int = 0
    bytes_after: int = 0

    @property
    def bytes_growth(self) -> int:
        return self.bytes_after - self.bytes_before

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "wall_seconds": round(self.wall_seconds, 4),
            "cpu_seconds": round(self.cpu_seconds, 4),
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "bytes_growth": self.bytes_growth,
        }
        record.update(self.attributes)
        return record


@contextmanager
def measured_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    watch_paths: list[str | Path] | None = None,
    tracer_name: str = MEASURE_TRACER_NAME,
) -> Iterator[SpanMeasurement]:
    """Time an operation, optionally sampling on-disk growth around it.

    ``name`` is the OpenTelemetry span name; ``attributes`` are written onto
    the span verbatim, so the caller controls the vocabulary::

        with measured_span("memory.consolidation",
                           {"memory.provider": "cashew"},
                           watch_paths=store_paths) as m:
            adapter.trigger_consolidation()
        record = m.as_record()
    """
    paths = watch_paths or []
    measurement = SpanMeasurement(name=name, attributes=dict(attributes or {}))
    measurement.bytes_before = directory_bytes(*paths)
    tracer = get_tracer(tracer_name)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    with tracer.start_as_current_span(name) as span:
        try:
            yield measurement
        finally:
            measurement.wall_seconds = time.perf_counter() - wall_start
            measurement.cpu_seconds = time.process_time() - cpu_start
            measurement.bytes_after = directory_bytes(*paths)
            span.set_attribute("measure.wall_seconds", measurement.wall_seconds)
            span.set_attribute("measure.cpu_seconds", measurement.cpu_seconds)
            span.set_attribute("measure.bytes_before", measurement.bytes_before)
            span.set_attribute("measure.bytes_after", measurement.bytes_after)
            span.set_attribute("measure.bytes_growth", measurement.bytes_growth)
            for k, v in measurement.attributes.items():
                span.set_attribute(k, v)

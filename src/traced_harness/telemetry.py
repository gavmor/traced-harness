"""OpenTelemetry configuration and tracer provider setup for traced-agent."""

from __future__ import annotations

import os
from pathlib import Path

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

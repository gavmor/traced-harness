"""Tests for memory-provider telemetry helpers (text-only)."""

from __future__ import annotations

from traced_harness.telemetry import (
    ConsolidationMeasurement,
    RetrievalMeasurement,
    consolidation_span,
    directory_bytes,
    estimate_tokens,
    record_memory_injection,
    retrieval_span,
)


def test_estimate_tokens_heuristic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10


def test_directory_bytes_sums_files(tmp_path):
    (tmp_path / "a.db").write_bytes(b"x" * 100)
    sub = tmp_path / "vectors"
    sub.mkdir()
    (sub / "v.bin").write_bytes(b"y" * 50)
    assert directory_bytes(tmp_path / "a.db", sub) == 150
    # Missing paths contribute zero, not an error.
    assert directory_bytes(tmp_path / "missing") == 0


def test_record_memory_injection_overhead():
    rec = record_memory_injection(100, 160, provider="cashew")
    assert rec == {
        "base_tokens": 100,
        "injected_tokens": 160,
        "overhead_tokens": 60,
    }
    # Overhead floors at zero when injection shrinks the prompt.
    assert record_memory_injection(100, 80)["overhead_tokens"] == 0


def test_retrieval_span_captures_count_and_latency():
    with retrieval_span("where does marc live", provider="chronicle") as r:
        assert isinstance(r, RetrievalMeasurement)
        r.passages = ["marc lives in toronto", "marc moved in 2021"]
    rec = r.as_record()
    assert rec["passage_count"] == 2
    assert rec["query"] == "where does marc live"
    assert rec["provider"] == "chronicle"
    assert rec["latency_ms"] >= 0.0


def test_retrieval_span_explicit_count_overrides_passages():
    with retrieval_span("q") as r:
        r.passage_count = 7
    assert r.as_record()["passage_count"] == 7


def test_consolidation_span_measures_db_growth(tmp_path):
    db = tmp_path / "brain.db"
    db.write_bytes(b"x" * 10)
    with consolidation_span(provider="memex8", store_paths=[db]) as m:
        assert isinstance(m, ConsolidationMeasurement)
        db.write_bytes(b"x" * 110)  # grow during consolidation
    rec = m.as_record()
    assert rec["db_bytes_before"] == 10
    assert rec["db_bytes_after"] == 110
    assert rec["db_growth_bytes"] == 100
    assert rec["wall_seconds"] >= 0.0
    assert rec["cpu_seconds"] >= 0.0

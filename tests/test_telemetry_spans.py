"""Tests for the generic measurement span and its helpers.

``measured_span`` is domain-agnostic: the caller supplies the span name and
attribute vocabulary, and gets back timing plus optional on-disk growth.
"""

from __future__ import annotations

import time

from traced_harness.telemetry import (
    directory_bytes,
    estimate_tokens,
    measured_span,
)


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_directory_bytes_counts_files_and_dirs(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 10)
    d = tmp_path / "store"
    d.mkdir()
    (d / "b.bin").write_bytes(b"y" * 5)

    assert directory_bytes(f) == 10
    assert directory_bytes(d) == 5
    assert directory_bytes(f, d) == 15


def test_directory_bytes_tolerates_missing_paths(tmp_path):
    """Safe to call before a store exists."""
    assert directory_bytes(tmp_path / "nope") == 0


def test_measured_span_times_the_block():
    with measured_span("demo.op") as m:
        time.sleep(0.01)
    rec = m.as_record()
    assert rec["wall_seconds"] >= 0.01
    assert rec["cpu_seconds"] >= 0.0
    assert rec["bytes_growth"] == 0


def test_measured_span_samples_disk_growth(tmp_path):
    store = tmp_path / "store"
    store.mkdir()

    with measured_span("demo.compact", watch_paths=[store]) as m:
        (store / "grown.bin").write_bytes(b"z" * 128)

    rec = m.as_record()
    assert rec["bytes_before"] == 0
    assert rec["bytes_after"] == 128
    assert rec["bytes_growth"] == 128


def test_caller_owns_the_attribute_vocabulary():
    """The harness records whatever keys the caller supplies, uninterpreted."""
    with measured_span(
        "memory.consolidation", {"memory.provider": "cashew"}
    ) as m:
        m.attributes["memory.phases"] = 13

    rec = m.as_record()
    assert rec["memory.provider"] == "cashew"
    assert rec["memory.phases"] == 13
    # Timing keys are always present alongside the caller's vocabulary.
    assert {"wall_seconds", "cpu_seconds", "bytes_growth"} <= rec.keys()


def test_measurement_records_even_on_exception(tmp_path):
    try:
        with measured_span("demo.boom") as m:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert m.wall_seconds > 0.0

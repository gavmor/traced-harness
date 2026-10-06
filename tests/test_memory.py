"""Tests for memory as a first-class harness peripheral.

Covers the contract, the lifecycle ABC (including the structural bridge to the
agnostic ``PeripheralLifecycle`` protocol), the memory telemetry vocabulary,
and the prompt/registry wiring.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from traced_harness.memory import (
    MEMORY_METADATA_KEY,
    MemoryProviderAdapter,
    MemoryToolContract,
    build_memory_instructions,
    clear_memory_tools,
    consolidation_span,
    get_registered_memory_tools,
    make_memory_session_runner,
    record_memory_injection,
    register_memory_tools,
    retrieval_span,
)
from traced_harness.session_runner import PeripheralLifecycle, Scenario, Session


class _StubProvider(MemoryProviderAdapter):
    name = "stub"

    def setup(self, workspace_dir: Path) -> dict[str, Any]:
        self.store_paths = [str(workspace_dir / "stub.db")]
        self._record("setup", str(workspace_dir))
        return {"env": {}, "contract": self.contract(), "store_paths": self.store_paths}

    def trigger_consolidation(self) -> None:
        self._record("consolidate", self.name)

    def teardown(self) -> None:
        self._record("teardown", self.name)

    def contract(self) -> MemoryToolContract:
        return MemoryToolContract(
            provider=self.name,
            tools=["stub_recall"],
            context_hooks=["stub_prefetch"],
            system_prompt="Stub memory is active.",
        )


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_memory_tools()
    yield
    clear_memory_tools()


# -- contract + lifecycle ---------------------------------------------------
def test_abstract_adapter_cannot_instantiate():
    with pytest.raises(TypeError):
        MemoryProviderAdapter()  # type: ignore[abstract]


def test_contract_defaults():
    c = MemoryToolContract(provider="demo")
    assert c.tools == []
    assert c.context_hooks == []
    assert c.system_prompt == ""


def test_adapter_satisfies_agnostic_lifecycle_protocol():
    """The core runner's protocol is structural — memory just conforms to it."""
    assert isinstance(_StubProvider(dry_run=True), PeripheralLifecycle)


def test_between_sessions_delegates_to_consolidation():
    a = _StubProvider(dry_run=True)
    a.between_sessions()
    assert [k for k, _ in a.actions] == ["consolidate"]


def test_purge_removes_files_and_dirs(tmp_path):
    f = tmp_path / "a.db"
    f.write_text("x")
    d = tmp_path / "vectors"
    d.mkdir()
    (d / "v.bin").write_text("y")
    MemoryProviderAdapter._purge(f, d)
    assert not f.exists() and not d.exists()


# -- telemetry --------------------------------------------------------------
def test_record_memory_injection_overhead():
    rec = record_memory_injection(100, 160, provider="stub")
    assert rec == {
        "base_tokens": 100,
        "injected_tokens": 160,
        "overhead_tokens": 60,
    }


def test_injection_overhead_never_negative():
    assert record_memory_injection(200, 100)["overhead_tokens"] == 0


def test_retrieval_span_records_passages_and_latency():
    with retrieval_span("where does marc live", "stub") as r:
        r.passages = ["marc lives in Berlin", "marc moved in 2024"]
    rec = r.as_record()
    assert rec["query"] == "where does marc live"
    assert rec["provider"] == "stub"
    assert rec["passage_count"] == 2
    assert rec["latency_ms"] >= 0.0


def test_retrieval_span_honours_explicit_count():
    with retrieval_span("q") as r:
        r.passage_count = 7
    assert r.as_record()["passage_count"] == 7


def test_consolidation_span_measures_store_growth(tmp_path):
    store = tmp_path / "brain"
    store.mkdir()
    with consolidation_span("stub", [store]) as m:
        (store / "grown.bin").write_bytes(b"z" * 64)
    rec = m.as_record()
    assert rec["bytes_growth"] == 64
    assert rec["memory.provider"] == "stub"
    assert rec["wall_seconds"] >= 0.0


# -- agent wiring -----------------------------------------------------------
def test_register_memory_tools_populates_registry():
    register_memory_tools(["cashew_query", "recall"], "cashew")
    assert get_registered_memory_tools() == {
        "cashew_query": "cashew",
        "recall": "cashew",
    }


def test_build_memory_instructions_sections():
    text = build_memory_instructions(
        "cashew",
        ["cashew_query"],
        ["cashew_prefetch"],
        "Durable memory is provided by Cashew.",
    )
    assert "# Memory Provider: cashew" in text
    assert "Durable memory is provided by Cashew." in text
    assert "`cashew_query`" in text
    assert "cashew_prefetch" in text


def test_build_memory_instructions_empty_when_nothing_to_say():
    assert build_memory_instructions("none", [], [], "") == ""


# -- pre-wired runner -------------------------------------------------------
def test_make_memory_session_runner_uses_memory_vocabulary(tmp_path):
    import json
    from dataclasses import dataclass, field

    @dataclass
    class _Turn:
        prompt: str
        output: str
        session_id: str
        tools_called: list[Any] = field(default_factory=list)
        metadata: dict[str, Any] = field(default_factory=dict)

    adapter = _StubProvider(dry_run=True)

    async def _exec(prompt, session_id, peripheral):
        return _Turn(prompt=prompt, output="ok", session_id=session_id)

    runner = make_memory_session_runner(adapter, _exec, tmp_path)
    assert runner.metadata_key == MEMORY_METADATA_KEY

    scenario = Scenario("s", [Session("s1", ["a"]), Session("s2", ["b"])])
    result = asyncio.run(runner.run_scenario(scenario))

    entries = [
        json.loads(line)
        for line in result.trace_file.read_text().strip().splitlines()
    ]
    assert "memory" in entries[0]["additional_metadata"]
    # Consolidation ran once (between two sessions) and is attached to s2.
    assert "between_sessions" in entries[1]["additional_metadata"]["memory"]

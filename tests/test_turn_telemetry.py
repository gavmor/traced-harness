"""Turn-level telemetry: the generic collector and the memory vocabulary on it.

Two things are under test here, and the seam between them matters more than
either:

* ``traced_harness.telemetry`` provides a turn-scoped slot that anything
  running inside a turn can write into. It knows no domain words.
* ``traced_harness.memory`` supplies the memory words — ``retrievals``,
  ``injection`` — by writing into that slot from a tool hook and a turn
  executor.

``test_core_does_not_load_the_memory_module`` pins the seam: importing the
harness core must not drag the memory module in.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from traced_harness.agent import (
    GENERATED_TOKENS_KEY,
    PROMPT_TOKENS_KEY,
    TurnResult,
    execute_turn,
    log_turn_to_session,
)
from traced_harness.memory import (
    INJECTION_KEY,
    RETRIEVALS_KEY,
    MemoryProviderAdapter,
    MemoryToolContract,
    RetrievalMeasurement,
    RetrievedDocument,
    _documents,
    _retrieval_query,
    make_memory_tool_hook,
    make_memory_turn_executor,
)
from traced_harness.telemetry import (
    append_turn_record,
    current_turn_metadata,
    record_turn_metadata,
    turn_telemetry,
)


# ---------------------------------------------------------------------------
# The generic collector.
# ---------------------------------------------------------------------------
def test_records_land_in_the_open_turn():
    with turn_telemetry() as collected:
        assert record_turn_metadata("generated_tokens", 12) is True
        assert append_turn_record("retrievals", {"query": "a"}) is True
        assert append_turn_record("retrievals", {"query": "b"}) is True
    assert collected == {
        "generated_tokens": 12,
        "retrievals": [{"query": "a"}, {"query": "b"}],
    }


def test_recording_outside_a_turn_is_a_no_op_not_an_error():
    """A peripheral driven directly simply has nowhere to record."""
    assert current_turn_metadata() is None
    assert record_turn_metadata("generated_tokens", 1) is False
    assert append_turn_record("retrievals", {}) is False


def test_nested_turns_do_not_steal_each_others_records():
    with turn_telemetry() as outer:
        record_turn_metadata("who", "outer")
        with turn_telemetry() as inner:
            record_turn_metadata("who", "inner")
        record_turn_metadata("after", True)
    assert outer == {"who": "outer", "after": True}
    assert inner == {"who": "inner"}


def test_appending_to_a_non_list_key_is_rejected():
    with turn_telemetry():
        record_turn_metadata("retrievals", "not a list")
        with pytest.raises(TypeError, match="not a list"):
            append_turn_record("retrievals", {})


def test_core_does_not_load_the_memory_module():
    """The harness core stays domain-blind.

    Importing the runner, the agent and the telemetry must not pull in
    ``traced_harness.memory``: the memory vocabulary is a peripheral's, and a
    core that imported it would be a core that knows what a retrieval is.
    """
    probe = (
        "import sys;"
        "import traced_harness.session_runner, traced_harness.agent,"
        " traced_harness.telemetry;"
        "print('traced_harness.memory' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False", (
        f"core imported; traced_harness.memory loaded? {out.stdout.strip()}"
    )


# ---------------------------------------------------------------------------
# Fakes for the agent loop.
# ---------------------------------------------------------------------------
@dataclass
class _FakeMetrics:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _FakeToolCall:
    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    result: Any = ""


@dataclass
class _FakeResponse:
    content: str
    tools: list[_FakeToolCall] = field(default_factory=list)
    metrics: _FakeMetrics | None = None


class _FakeModel:
    id = "fake-model"


class _FakeAgent:
    """Minimal stand-in for an Agno agent.

    ``on_run`` lets a test act *during* the turn — which is where a memory
    tool would run, and therefore the only place its records can be written.
    """

    def __init__(self, response: _FakeResponse, on_run=None) -> None:
        self.model = _FakeModel()
        self._response = response
        self._on_run = on_run
        self.prompts: list[str] = []

    async def arun(self, prompt: str, session_id: str = "") -> _FakeResponse:
        self.prompts.append(prompt)
        if self._on_run is not None:
            await self._on_run()
        return self._response


class _FakeAdapter(MemoryProviderAdapter):
    name = "fakemem"

    def setup(self, workspace_dir: Path) -> dict[str, Any]:
        return {"env": {}, "contract": self.contract(), "store_paths": []}

    def trigger_consolidation(self) -> None:
        self._record("consolidate", self.name)

    def teardown(self) -> None:
        self._record("teardown", self.name)

    def contract(self) -> MemoryToolContract:
        return MemoryToolContract(
            provider=self.name,
            tools=["fakemem_recall"],
            system_prompt="Call `fakemem_recall` before answering.",
        )


# ---------------------------------------------------------------------------
# execute_turn populates TurnResult.metadata.
# ---------------------------------------------------------------------------
def test_execute_turn_prefers_the_models_own_token_counts():
    agent = _FakeAgent(
        _FakeResponse("four words of output", metrics=_FakeMetrics(31, 7))
    )
    turn = asyncio.run(execute_turn("a prompt", agent, "s1"))
    assert turn.metadata[PROMPT_TOKENS_KEY] == 31
    assert turn.metadata[GENERATED_TOKENS_KEY] == 7


def test_execute_turn_falls_back_to_the_estimate_without_metrics():
    """A model reporting no usage must still produce a usable token count."""
    agent = _FakeAgent(_FakeResponse("12345678", metrics=None))
    turn = asyncio.run(execute_turn("abcd", agent, "s1"))
    assert turn.metadata[PROMPT_TOKENS_KEY] == 1  # 4 chars / 4
    assert turn.metadata[GENERATED_TOKENS_KEY] == 2  # 8 chars / 4


def test_execute_turn_carries_peripheral_records_through():
    async def _peripheral_runs():
        append_turn_record("retrievals", {"query": "where do I live"})

    agent = _FakeAgent(_FakeResponse("Berlin."), on_run=_peripheral_runs)
    turn = asyncio.run(execute_turn("where do I live?", agent, "s1"))
    assert turn.metadata["retrievals"] == [{"query": "where do I live"}]


def test_session_log_carries_turn_metadata(tmp_path):
    turn = TurnResult(
        prompt="p",
        output="o",
        session_id="s1",
        metadata={"generated_tokens": 4, "retrievals": [{"query": "q"}]},
    )
    log_file = tmp_path / "session.jsonl"
    log_turn_to_session(turn, log_file)
    entry = json.loads(log_file.read_text().splitlines()[0])
    assert entry["additional_metadata"]["generated_tokens"] == 4
    assert entry["additional_metadata"]["retrievals"] == [{"query": "q"}]


# ---------------------------------------------------------------------------
# The memory tool hook.
# ---------------------------------------------------------------------------
def _hook_for(tools: tuple[str, ...] = ("fakemem_recall",)):
    return make_memory_tool_hook(
        MemoryToolContract(provider="fakemem", tools=list(tools))
    )


def test_hook_records_a_measured_retrieval():
    hook = _hook_for()

    async def _call(**kwargs):
        return json.dumps(["lives in Berlin", "moved in 2019"])

    async def _run():
        with turn_telemetry() as collected:
            result = await hook("fakemem_recall", _call, {"query": "where"})
        return result, collected

    result, collected = asyncio.run(_run())
    assert "Berlin" in result
    records = collected[RETRIEVALS_KEY]
    assert len(records) == 1
    assert records[0]["query"] == "where"
    assert records[0]["provider"] == "fakemem"
    # A tool that declares no outputSchema returns opaque text. The harness
    # reports ONE document rather than splitting the blob on a guess: what the
    # tool actually declared is one result, and inferring two would be the
    # instrument inventing structure the provider never claimed.
    assert records[0]["passage_count"] == 1
    assert records[0]["documents"] == [
        {"content": '["lives in Berlin", "moved in 2019"]'}
    ]
    # Latency is measured around the real call, not asserted as a constant.
    assert records[0]["latency_ms"] >= 0.0


def test_hook_splits_documents_when_the_tool_declares_them():
    """The same two memories, from a tool with an outputSchema."""
    hook = _hook_for()

    async def _call(**kwargs):
        return _tool_result(
            "lives in Berlin\nmoved in 2019",
            [
                {"id": "m1", "content": "lives in Berlin", "score": 0.9},
                {"id": "m2", "content": "moved in 2019"},
            ],
        )

    async def _run():
        with turn_telemetry() as collected:
            await hook("fakemem_recall", _call, {"query": "where"})
        return collected

    records = asyncio.run(_run())[RETRIEVALS_KEY]
    assert records[0]["passage_count"] == 2
    assert records[0]["passages"] == ["lives in Berlin", "moved in 2019"]
    assert records[0]["documents"][0] == {
        "content": "lives in Berlin",
        "id": "m1",
        "score": 0.9,
    }


def test_hook_ignores_tools_that_are_not_this_providers():
    hook = _hook_for()

    async def _call(**kwargs):
        return "weather is fine"

    async def _run():
        with turn_telemetry() as collected:
            await hook("get_weather", _call, {"city": "Berlin"})
        return collected

    assert RETRIEVALS_KEY not in asyncio.run(_run())


def test_hook_does_not_count_the_providers_own_writes_as_retrievals():
    """A store is not a recall. Counting it inflates `n_retrievals`, the
    column that exists to prove the provider was actually consulted."""
    hook = make_memory_tool_hook(
        MemoryToolContract(
            provider="fakemem",
            tools=["fakemem_recall", "fakemem_put"],
            retrieval_tools=["fakemem_recall"],
        )
    )

    async def _call(**kwargs):
        return "7"  # the new entry's id

    async def _run():
        with turn_telemetry() as collected:
            await hook("fakemem_put", _call, {"text": "a fact"})
        return collected

    assert RETRIEVALS_KEY not in asyncio.run(_run())


def test_a_contract_without_retrieval_tools_measures_all_of_them():
    """Backward compatible: a read-only provider need not restate its tools."""
    contract = MemoryToolContract(provider="p", tools=["a", "b"])
    assert contract.recall_tools() == ["a", "b"]
    assert MemoryToolContract(
        provider="p", tools=["a", "b"], retrieval_tools=["a"]
    ).recall_tools() == ["a"]


def test_hook_propagates_tool_failures():
    """A provider that errored did not retrieve; the turn must not say it did."""
    hook = _hook_for()

    async def _call(**kwargs):
        raise RuntimeError("backend not provisioned")

    async def _run():
        with turn_telemetry() as collected, pytest.raises(
            RuntimeError, match="not provisioned"
        ):
            await hook("fakemem_recall", _call, {"query": "where"})
        return collected

    assert asyncio.run(_run()).get(RETRIEVALS_KEY) is None


@dataclass
class _ToolResult:
    """The shape agno hands a tool hook: flattened text + MCP metadata."""

    content: str
    metadata: dict | None = None


def _tool_result(content, structured=None):
    meta = None if structured is None else {"structured_content": structured}
    return _ToolResult(content=content, metadata=meta)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (None, []),
        ("", []),
        ("   ", []),
        ("a single passage", ["a single passage"]),
        # An agno ToolResult with no structured output: the flattened text.
        (_tool_result("lives in Berlin"), ["lives in Berlin"]),
        (_tool_result(""), []),
        # MCP structured_content: a list is one document per element.
        (_tool_result("ignored", ["one", "two"]), ["one", "two"]),
        # ...and a non-list is a single document.
        (_tool_result("ignored", {"content": "only one"}), ["only one"]),
    ],
)
def test_documents_read_the_specified_boundary(result, expected):
    assert [d.content for d in _documents(result)] == expected


def test_structured_content_wins_over_flattened_text():
    """The tool's declared output is higher fidelity than agno's join."""
    docs = _documents(_tool_result("one\\ntwo", ["one", "two"]))
    assert [d.content for d in docs] == ["one", "two"]


def test_document_fields_are_carried_not_guessed():
    docs = _documents(
        _tool_result(
            "x",
            [{"id": "m1", "content": "lives in Berlin", "score": 0.91,
              "metadata": {"session": "s2"}}],
        )
    )
    assert len(docs) == 1
    d = docs[0]
    assert (d.id, d.content, d.score) == ("m1", "lives in Berlin", 0.91)
    assert d.metadata == {"session": "s2"}
    assert d.as_record()["score"] == 0.91


def test_a_mapping_without_content_is_serialized_whole():
    docs = _documents(_tool_result("x", [{"fact": "drives a Tesla"}]))
    assert docs[0].content == '{"fact": "drives a Tesla"}'
    assert docs[0].id is None and docs[0].score is None


def test_empty_documents_are_dropped_not_counted():
    docs = _documents(_tool_result("x", ["", "   ", "real"]))
    assert [d.content for d in docs] == ["real"]


def test_passages_alias_round_trips():
    m = RetrievalMeasurement(query="q")
    m.passages = ["a", "b", "  "]
    assert [d.content for d in m.documents] == ["a", "b"]
    assert m.passages == ["a", "b"]
    assert m.count == 2


def test_record_carries_documents_and_the_text_alias():
    m = RetrievalMeasurement(query="q", provider="p")
    m.documents = [RetrievedDocument(content="lives in Berlin", id="m1")]
    rec = m.as_record()
    assert rec["documents"] == [{"content": "lives in Berlin", "id": "m1"}]
    assert rec["passages"] == ["lives in Berlin"]
    assert rec["passage_count"] == 1


def test_retrieval_query_falls_back_to_the_whole_argument_dict():
    assert _retrieval_query({"query": "where do I live"}) == "where do I live"
    assert _retrieval_query({"topic": "vehicles"}) == '{"topic": "vehicles"}'



# ---------------------------------------------------------------------------
# The memory turn executor.
# ---------------------------------------------------------------------------
def test_turn_executor_prices_the_turn_against_what_memory_injected():
    adapter = _FakeAdapter(dry_run=True)

    async def _retrieve():
        append_turn_record(
            RETRIEVALS_KEY,
            {"query": "where", "passages": ["x" * 40], "passage_count": 1},
        )

    agent = _FakeAgent(
        _FakeResponse("Berlin.", metrics=_FakeMetrics(50, 3)),
        on_run=_retrieve,
    )
    executor = make_memory_turn_executor(agent, adapter)
    turn = asyncio.run(executor("where do I live?", "s1", adapter.name))

    injection = turn.metadata[INJECTION_KEY]
    assert injection["overhead_tokens"] > 0
    # Overhead is the contract prompt plus the 40-char passage (10 tokens),
    # and excludes the user's own prompt.
    assert injection["injected_tokens"] - injection["base_tokens"] == (
        injection["overhead_tokens"]
    )
    assert injection["overhead_tokens"] >= 10
    assert turn.metadata["provider"] == adapter.name
    assert turn.metadata[GENERATED_TOKENS_KEY] == 3


def test_turn_executor_charges_the_standing_contract_even_with_no_recall():
    """A provider that is wired in but never consulted still costs prompt."""
    adapter = _FakeAdapter(dry_run=True)
    agent = _FakeAgent(_FakeResponse("I don't know.", metrics=_FakeMetrics(9, 4)))
    executor = make_memory_turn_executor(agent, adapter)
    turn = asyncio.run(executor("where do I live?", "s1", adapter.name))

    assert RETRIEVALS_KEY not in turn.metadata
    assert turn.metadata[INJECTION_KEY]["overhead_tokens"] > 0

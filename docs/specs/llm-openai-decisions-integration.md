# Spec: calibrated decision backends in `traced-harness`

Integration of the approach behind [`simonw/llm-openai-decisions`][upstream] into
`gavmor/traced-harness`.

| | |
| --- | --- |
| Status | Ready to implement |
| Target repo | `gavmor/traced-harness` @ `838b0aa` ("refactor: separate the instrument from the study") |
| Upstream studied | `simonw/llm-openai-decisions` @ `4827324`, Apache-2.0 |
| Inputs | `docs/research/llm-openai-decisions.md` (task `t_58275be1`), `docs/research/traced-harness-architecture.md` (task `t_af5efea7`) |
| Implementer card | `t_4da11c3f` |
| Verifier card | `t_b9666490` |

Baseline measured on a clean clone of `838b0aa` while writing this spec, so the
implementer and the verifier start from the same numbers:

```
uv sync && uv run pytest -q     -> 60 passed in 2.40s
uv run ruff check src tests     -> All checks passed!
uv run ruff check .             -> Found 5 errors   (all inside vendored .agents/ skill templates)
python                          -> 3.12.13
httpx2                          -> 2.13.1   (already in uv.lock, transitive via mcp 2.3.0)
pydantic                        -> 2.13.5   (declared dependency, NOT imported anywhere in src/ or tests/)
```

**This document is the decision record. Nothing below is left to the
implementer's judgement.** Where the two research notes flagged an open
question, §0 answers it. If an answer turns out to be wrong during
implementation, change it here in a commit and say so in the PR — do not
silently diverge.

---

## 0. Decisions that were open, now closed

Both research notes ended on the same four unresolved questions. They are
settled here.

### D1. Does the concrete backend live in `traced-harness` or in a study repo? — **In `traced-harness`.**

The research note correctly flagged that HEAD is literally the commit that
separated instrument from study, and that `memory.py:26-29` refuses concrete
adapters:

> *Which* providers you benchmark is not the harness's business: concrete
> adapters (Cashew, Chronicle, Memex8, Nachos, ...) belong to the study that
> compares them.

That rule is about **peripherals under evaluation**. A decision backend is not
under evaluation; it is a measuring device the harness *uses* to turn free text
into calibrated numbers. The correct precedent is therefore not `memory.py` but
`telemetry.py:90-97`:

> The harness measures; it does not interpret. A study layered on this
> instrument supplies its own domain vocabulary (span names, attribute keys) and
> gets back a plain-dict record it can embed in a turn's
> `additional_metadata` for offline evaluation.

So: the harness ships the **mechanism** (question/answer primitives, the backend
protocol, the span, the metadata slot, the annotation driver, and one concrete
reading device). The harness never ships a **rubric** — *which* questions to ask
is the study's business and is always supplied by the caller. That line is what
keeps this change on the right side of `838b0aa`, and the implementer must not
cross it: **no default question set, no built-in quality rubric, no scoring, no
pass/fail opinion anywhere in `src/`.**

### D2. Is a decision an input to the run, a tool during it, or a judge after it? — **A post-hoc annotation over recorded traces.**

Decisions are produced by reading a finished JSONL trace and writing an
annotated copy alongside it. Reasons, in order of weight:

1. It is fully testable with zero network and zero credentials against the
   already-committed fixtures in `tests/fixtures/traces/`, which is what both
   the implementer and verifier cards require.
2. It does not touch `agent.py:181` (`await agent.arun(...)`), the single LLM
   call site that the research note flagged as the repo's collision hotspot.
3. It does not make a decision the model's choice, which is what an
   agent-callable `decide(...)` tool would do — reintroducing exactly the
   autoregressive variance a zero-decoding decision engine exists to remove.
4. Answers landing in `additional_metadata["decisions"]` are readable by
   `load_trace()` unchanged, so the study repo consumes them with the machinery
   it already uses for `["memory"]`.

Explicitly **not** chosen, and out of scope (§1.2): in-band capture during a
live agent run.

### D3. Vendor, depend on the package, or reimplement? — **Reimplement natively, with attribution.**

See §2 for the full justification and the rejected alternatives.

### D4. Where does the OpenAI key resolve? — **`OPENAI_API_KEY`, read only inside the OpenAI backend, only at call time.**

`agent.py:50-56` stays Google-only and is not touched. The harness never reads
`llm`'s key store, never writes a key anywhere, and never has a key present on
any default code path. See §5.3 and §7.

---

## 1. Scope

### 1.1 In scope

1. **`src/traced_harness/decisions.py`** — a new first-class module holding the
   backend-neutral vocabulary: `DecisionQuestion` / `Choice` / `Level` /
   `DecisionAnswer`, the `DecisionBackend` ABC, `DECISIONS_METADATA_KEY`,
   `decision_span()`, the deterministic context builder and `decision_id`
   function, the JSON-Schema → questions mapper, the `annotate_trace()` driver,
   and the offline `ReplayDecisionBackend`.
2. **`src/traced_harness/decisions_openai.py`** — `OpenAIDecisionsBackend`, a
   direct `POST https://api.openai.com/v1/decisions` client reimplementing the
   request shape and the five response invariants documented in
   `docs/research/llm-openai-decisions.md` §2.
3. **Trace capture** — the `decisions` slot in a turn's `additional_metadata`,
   with the field mapping, storage location, and correlation IDs fixed in §4.
4. **CLI enablement** — a `decide` subcommand plus the symmetric `--decide`
   flag in `main.py`, following the existing `eval` pattern exactly.
5. **Tests** — the 18 named tests in §6.2, all offline.
6. **CI** — `.github/workflows/test.yml`. The repo has none today; the
   acceptance criterion "CI green" is unmeetable without building it, so it is
   in scope (§6.3).
7. **Docs and attribution** — `docs/decisions.md`, a README section,
   two committed example files, and `NOTICE` (§5.5, §6.4).

### 1.2 Out of scope

Each of these is a deliberate exclusion, not an oversight. Do not implement
them in this PR; open a follow-up card if one is wanted.

| Excluded | Why |
| --- | --- |
| **The MediaPipe Decision Maker backend** | Nobody has read its API. `docs/research/llm-openai-decisions.md` §8 states plainly that the MediaPipe mapping restates a claim and that MediaPipe was never inspected. Guessing an on-device API in this PR would be fabrication. The interface in §3 is designed so MediaPipe drops in as a second `DecisionBackend` subclass with **no change to `decisions.py`**, and §3.6 lists the three asymmetries that must survive for that to hold. A separate research card covers it. |
| **In-band decisions during a live agent run** | Requires either surgery on `agent.py:158-226` or a `TurnResult.metadata` field, and the JSONL schema is duplicated across `agent.py:128` and `session_runner.py:181` — a schema change must touch both. Deferred by D2. |
| **An agent-callable `decide(...)` tool** | Rejected by D2.3. Also, answers would land in `tools_called[]`, which `eval.py:33-49` scans for *peripheral failures*; a refusal or an `{"error": ...}`-shaped answer would be misread as a peripheral failure and trigger cascade pruning. |
| **Any dependency on `llm`, `llm-openai-decisions`, or `pluggy`** | D3 / §2. |
| **Scoring, rubrics, pass/fail verdicts, DeepEval metrics** | D1. The study repo's job. |
| **Changing `eval.py`, `agent.py`, `session_runner.py`, `telemetry.py`, `skills.py`, `repl.py`, `client.py`** | `decisions.py` *reads* `eval.TraceTurn`/`load_trace` and *calls* `telemetry.measured_span`; neither is modified. The only edited existing files are `main.py` (subcommand wiring), `pyproject.toml` (one dependency line), `README.md`, and `__init__.py`. |
| **A live call to `api.openai.com/v1/decisions`** | The endpoint and the model `gpt-6-luna` are unverified — see R1 in §7. No acceptance criterion may require one. |
| **Choosing a license for `traced-harness` itself** | The repo has no root `LICENSE`. That is the owner's call, not this PR's. See Q1 in §7. |
| **Widening `testpaths`** | `pyproject.toml:41-45` pins it with a load-bearing comment. Do not touch it. |

---

## 2. Integration strategy: reimplement natively

**Chosen: reimplement.** `decisions_openai.py` is new code in the harness's own
style — stdlib + `dataclasses` + `httpx2` — that speaks the same wire protocol
and enforces the same validation rules as upstream, with upstream credited.

### Why

1. **The `llm` layer cannot host half the backends.** The stated goal is two
   interchangeable backends — the OpenAI Decisions endpoint and MediaPipe
   Decision Maker — behind one interface. MediaPipe is an on-device,
   zero-decoding engine; it is not an `llm` model plugin and never will be.
   Adopting `llm`'s plugin wiring buys a registry that exactly one backend can
   ever use, and forces the other into an adapter around an abstraction that
   does not fit it.
2. **Depending on the package means depending on `llm`.** `llm>=0.36` is a CLI
   framework: pluggy discovery, its own `logs.db`, its own key store, its own
   `LLM_USER_PATH` config convention. The harness's config surface is argparse
   flags plus environment variables and nothing else (`pyproject.toml` has no
   config section, there is no config file). Importing `llm` means a second
   model stack beside Agno/google-genai, a second credential store beside
   `GOOGLE_API_KEY`, and a second place where traces are written — for one HTTP
   POST.
3. **Vendoring degenerates into porting anyway.** `llm_openai_decisions.py`
   imports `llm` at module top level and builds its payload by walking
   `prompt.messages` with `llm.parts.TextPart` / `AttachmentPart`
   (`llm_openai_decisions.py:9`, `227-258`). A vendored copy still needs `llm`
   installed, or needs that half rewritten. So vendoring costs the full
   Apache-2.0 §4 obligation set and still does not avoid the rewrite.
4. **The portable core is small and the harness already has its dependencies.**
   The research note isolates five portable pieces (`docs/research/llm-openai-decisions.md` §6):
   strict JSON parsing, the question schema with its cross-field rules, the
   answer union, the payload-assembly rules, and the response-consistency
   checks. In harness style these are ~200 lines of dataclasses. `httpx2 2.13.1`
   is **already resolved in `uv.lock`** as a transitive dependency of
   `mcp 2.3.0`, so the transport costs no new resolution — and it ships
   `MockTransport`, which gives offline tests with no new dev dependency.
5. **The harness needs its own vocabulary regardless.** `DecisionAnswer` has to
   carry fields from backends that are not OpenAI. Owning the type is not
   incidental cost; it is the deliverable.

### Consequences the implementer must honour

- Declare `httpx2>=2.13.0` **explicitly** in `[project.dependencies]`. It is
  already locked, so `uv lock` must produce no change to resolved versions —
  if it does, stop and report it.
- Do **not** add `pydantic` models. `pydantic` is declared but imported nowhere
  in `src/` or `tests/`; the entire repo is dataclasses with `__post_init__`
  validation and `as_record()` serializers (`telemetry.py:133-166`,
  `memory.py:219-247`). Match it.
- Attribution is still required even though the code is new — see §5.5.

---

## 3. Design

### 3.1 `src/traced_harness/decisions.py`

Module docstring must carry a layering preamble in the style of `memory.py:1-30`
and `session_runner.py:1-24` — architectural intent lives in docstrings in this
repo, not a wiki. It must state D1 (mechanism here, rubric from the caller).

```python
DECISIONS_METADATA_KEY = "decisions"          # slot in additional_metadata
DECISIONS_SCHEMA_VERSION = 1                  # bumped on any record-shape change
DECISION_TRACER_NAME = "traced.harness.decisions"
DECISION_SPAN = "decision.turn"
```

**Question side** (frozen dataclasses, validated in `__post_init__`, raising
`ValueError`):

```python
QuestionType = Literal["predicate", "choice", "score"]

@dataclass(frozen=True)
class Choice:
    value: str | bool
    description: str | None = None

@dataclass(frozen=True)
class Level:
    label: str
    description: str | None = None

@dataclass(frozen=True)
class DecisionQuestion:
    name: str
    instructions: str
    type: QuestionType = "predicate"
    choices: tuple[Choice, ...] | None = None
    levels: tuple[Level, ...] | None = None
```

Validation rules, ported verbatim in behaviour from upstream
(`llm_openai_decisions.py:52-81`) — each is a required test (§6.2):

- `name` and `instructions` non-blank after `strip()`.
- `type == "choice"` ⇒ `choices` present with ≥2 entries, and all `(type(value), value)`
  pairs unique — so `True` and `"true"` are distinct, never coerced.
- `type == "choice"` ⇒ `levels` is `None`.
- `type == "score"` ⇒ `levels` present with ≥2 entries, ordered low→high as given.
- `type == "score"` ⇒ `choices` is `None`.
- `type == "predicate"` ⇒ both `None`.

**Answer side:**

```python
AnswerType = Literal["predicate", "choice", "score", "refusal"]

@dataclass(frozen=True)
class Probability:
    value: str | bool | int | None   # choice value, or 0-based level index for score
    label: str | None                # level label for score; None for choice
    probability: float               # [0.0, 1.0]

@dataclass
class DecisionAnswer:
    name: str
    type: AnswerType
    probability: float | None = None      # predicate only
    choice: str | bool | None = None      # choice only
    score: float | None = None            # score only
    confidence: float | None = None       # choice and score only; never predicate
    probabilities: tuple[Probability, ...] = ()
    backend: str = ""

    @property
    def refused(self) -> bool: ...
    def as_record(self) -> dict[str, Any]: ...   # JSON-serializable, see §4.2
```

All probability/confidence floats are validated into `[0.0, 1.0]`; out-of-range
raises `ValueError`.

**Backend protocol** — an ABC, mirroring `MemoryProviderAdapter` (`memory.py:89`):

```python
class DecisionBackend(ABC):
    name: str = "decisions"

    @abstractmethod
    async def ask(
        self, context: str, questions: Sequence[DecisionQuestion]
    ) -> DecisionResult: ...
```

```python
@dataclass
class DecisionResult:
    answers: tuple[DecisionAnswer, ...]
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
```

`ask` is `async` because the harness is async end to end and the OpenAI backend
does network I/O. A synchronous backend (MediaPipe) implements it as an `async
def` that does its work in `asyncio.to_thread`.

**Errors** (all defined in `decisions.py`, all subclasses of
`DecisionsError(Exception)`):

| Exception | Raised when |
| --- | --- |
| `DecisionsConfigError` | missing credential, unknown backend name, unreadable questions file — always **before** any request |
| `DecisionBackendError` | transport failure, non-2xx status, malformed or inconsistent response |

### 3.2 Context builder — exactly this, and deterministic

`build_turn_context(turn: TraceTurn) -> str` renders the text the questions are
asked about. The template is fixed so tests can assert it byte for byte:

```
### Input
{turn.input}

### Output
{turn.actual_output}

### Tools called
- {name}({json.dumps(input_parameters, sort_keys=True)}) -> {output[:2000]}
```

Rules:
- The `### Tools called` section and its list are omitted entirely when
  `tools_called` is empty.
- Tools appear in trace order; `json.dumps(..., sort_keys=True)` makes
  parameters stable.
- Tool output truncated at 2000 characters, matching `agent.py:207`.
- No trailing newline. No other whitespace normalisation.

A caller may pass its own `context_builder: Callable[[TraceTurn], str]` to
`annotate_trace`; the default is this function.

### 3.3 JSON-Schema mapping

`questions_from_json_schema(schema: dict) -> list[DecisionQuestion]`. This is
the "same json_schema in" contract both backends share.

Input is an object schema. Each entry of `properties` becomes one question, in
declaration order; the property name becomes `name`, its `description` becomes
`instructions` (empty description ⇒ `DecisionsConfigError`).

| JSON Schema property | Question |
| --- | --- |
| `{"type": "boolean"}` | `predicate` |
| `{"type": "string", "enum": [...]}` | `choice`, one `Choice(value=e)` per enum entry, order preserved |
| `{"type": "integer", "minimum": m, "maximum": n}` | `score`, `n - m + 1` levels labelled `str(m) … str(n)` |
| `{"type": "integer", ..., "x-levels": ["Bad", "OK", "Good"]}` | `score` with those labels; length must equal `n - m + 1` |
| anything else | `DecisionsConfigError` naming the property and its type |

`string` without `enum`, `number`, `array`, `object`, and nested schemas are
unsupported and must raise — silently dropping a question would corrupt the
answer-count parity check in §3.5.

### 3.4 Questions file format

`--questions FILE` takes one JSON document, detected by its top-level type:

- a **list** ⇒ native questions, each `{"name", "instructions", "type", "choices"?, "levels"?}`;
  `choices` entries may be a bare value or `{"value", "description"}`; `levels`
  entries may be a bare string or `{"label", "description"}`.
- an **object with `"type": "object"`** ⇒ a JSON Schema, routed through §3.3.
- anything else ⇒ `DecisionsConfigError`.

### 3.5 `annotate_trace` — the driver

```python
async def annotate_trace(
    trace_file: str | Path,
    questions: Sequence[DecisionQuestion],
    backend: DecisionBackend,
    out_file: str | Path | None = None,
    context_builder: Callable[[TraceTurn], str] = build_turn_context,
    on_error: Literal["record", "raise"] = "record",
) -> Path:
```

Behaviour:

1. Read with `eval.load_trace()` *and* keep the raw JSON lines (re-read the file
   with `json.loads` per line) so the output can reproduce every original key
   verbatim — `TraceTurn` is lossy for `additional_metadata` subkeys it does not
   model.
2. For each turn, in file order: build the context, open `decision_span()`, call
   `await backend.ask(...)`, build the record from §4.2, and merge it into
   `additional_metadata[DECISIONS_METADATA_KEY]`.
3. Turns are processed **sequentially**, not concurrently. Determinism and
   rate-limit safety beat speed here; a concurrency option is a follow-up.
4. Write to `out_file`, defaulting to `<stem>.decided.jsonl` beside the input.
   **The input file is never modified** — asserted by a test.
5. If the input already contains a `decisions` key on a turn, overwrite it and
   count it; report the count to the caller's logger. Do not merge.
6. `on_error="record"` (the default) catches `DecisionBackendError` per turn and
   writes `{"error": "<class>: <message>", "schema_version": N, ...}` into that
   turn's slot, then continues. `on_error="raise"` propagates.
   `DecisionsConfigError` always propagates regardless — it means the run was
   misconfigured, not that one turn failed.
7. Return the output `Path`.

> Note for §4.4: the per-turn error record is written under
> `additional_metadata["decisions"]["error"]`, **never** into `tools_called[]`.
> `eval.py:33-49` detects peripheral failures by scanning tool output for an
> `"error"` key; putting a decision error where that scanner can see it would
> fabricate a peripheral failure and trigger cascade pruning. A test asserts
> `evaluate_trace()` reports identical results for the input and the annotated
> output.

### 3.6 `ReplayDecisionBackend` — the offline backend

A first-class feature, not test scaffolding: deterministic re-annotation and
keyless CI/demo runs.

```python
class ReplayDecisionBackend(DecisionBackend):
    name = "replay"
    def __init__(self, path: str | Path, strict: bool = True) -> None: ...
```

`path` is a JSON file containing a list of recorded provider response bodies —
the same shape `OpenAIDecisionsBackend` would have received. The Nth `ask()`
call consumes the Nth entry and runs it through **the same validation path** as
a live response (§3.7), so a replay run exercises the real parser. With
`strict=True` (default), running out of entries raises `DecisionBackendError`.

### 3.7 `src/traced_harness/decisions_openai.py`

Never imported by `decisions.py` or by any core module. Imported lazily inside
the CLI backend factory, exactly as `agent.py:95-99` lazily imports `memory`.

```python
OPENAI_DECISIONS_URL = "https://api.openai.com/v1/decisions"
DEFAULT_DECISIONS_MODEL = "gpt-6-luna"

class OpenAIDecisionsBackend(DecisionBackend):
    name = "openai"
    def __init__(
        self,
        model: str = DEFAULT_DECISIONS_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        client: Any | None = None,     # httpx2.AsyncClient; injected by tests
    ) -> None: ...
```

**Request** (`POST`, JSON body):

```json
{
  "model": "<model>",
  "input": "<context string>",
  "questions": [{"name": "...", "instructions": "...", "type": "...",
                 "choices": [...], "levels": [...]}]
}
```

- `input` is always a plain string here. The harness annotates recorded text;
  upstream's multi-message/image path (`llm_openai_decisions.py:227-258`) has no
  counterpart in a trace turn and is **not** ported. Image attachments are out
  of scope.
- `choices` / `levels` are omitted when `None`, never sent as `null`.
- Headers: `Authorization: Bearer <key>` and `Content-Type: application/json`.
  Nothing else.

**Transport**: `httpx2.AsyncClient(timeout=timeout, follow_redirects=False)`.
`follow_redirects=False` is **load-bearing and must not be changed**: a 302 must
surface as an error rather than replay the `Authorization` header at the
redirect target. Upstream asserts this by including `302` in its error-status
matrix; §6.2 test 13 does the same here.

**Response validation** — all five upstream invariants
(`llm_openai_decisions.py:285-329`), each a required test:

1. Exactly one answer per question, in order.
2. `answers[i].name == questions[i].name`.
3. `answers[i].type == questions[i].type` **or** `"refusal"` — refusal is a
   legal answer to any question type.
4. Top-level `model` is a non-blank string.
5. `usage.input_tokens` and `usage.output_tokens` are non-negative and
   `type(x) is int` — this deliberately rejects `bool`, since `True` passes
   `isinstance(x, int)`.

Additionally: the body must parse as JSON with no duplicate keys and no
`NaN`/`Infinity` (upstream `parse_json`, `llm_openai_decisions.py:24-35`). Use
`json.loads(..., parse_constant=_reject, object_pairs_hook=_no_dupes)`.

Every failure becomes a `DecisionBackendError`. The message may contain the HTTP
status and at most 200 characters of the response body. It must **never**
contain request headers, and §6.2 test 14 asserts the key string appears in no
exception text and in no written artifact.

---

## 4. Mapping onto `traced-harness` traces

### 4.1 Storage location

- **Channel B (JSONL) is the durable artifact.** Decisions are written into
  `additional_metadata["decisions"]` of each turn in a **new file**,
  `<input-stem>.decided.jsonl`, in the input's directory unless `--out` says
  otherwise. The input is never mutated.
- Outer schema is untouched: `input`, `actual_output`, `tools_called`,
  `additional_metadata`. Every pre-existing key is reproduced verbatim, so
  `load_trace()` and `evaluate_trace()` work on the output with no change.
- **Channel A (OpenTelemetry)** gets a `decision.turn` span per turn, via
  `decision_span()` layered on `telemetry.measured_span` — the same relationship
  `memory.consolidation_span` has to it (`memory.py:277-295`). Spans are dropped
  unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set (`telemetry.py:33-51`); that is
  pre-existing behaviour and is not changed. **No new exporter, no new env
  plumbing.**

### 4.2 Record shape

`additional_metadata["decisions"]`, one object per turn:

```json
{
  "schema_version": 1,
  "backend": "openai",
  "model": "gpt-6-luna",
  "trace_id": "trace_memory_cashew",
  "turn_index": 3,
  "session_id": "s1",
  "answers": [
    {
      "decision_id": "9f2c41ab7e0d5c18",
      "name": "answered_the_question",
      "type": "predicate",
      "probability": 0.93,
      "choice": null,
      "score": null,
      "confidence": null,
      "probabilities": [],
      "refused": false
    }
  ],
  "usage": {"input_tokens": 812, "output_tokens": 14},
  "measure": {
    "wall_seconds": 0.4121,
    "cpu_seconds": 0.0039,
    "bytes_before": 0,
    "bytes_after": 0,
    "bytes_growth": 0,
    "decision.backend": "openai",
    "decision.question_count": 3,
    "decision.refusal_count": 0
  },
  "otel": {"trace_id": "4bf92f3577b34da6a3ce929d0e0e4736", "span_id": "00f067aa0ba902b7"},
  "raw": {}
}
```

Notes that are requirements, not commentary:

- `measure` is literally `SpanMeasurement.as_record()`. That method merges the
  caller's span attributes into the top level of the dict
  (`telemetry.py:154-166`), which is why dotted keys appear flat inside it. This
  matches how `between_sessions` records are stored (`session_runner.py:149-165`).
- `otel` is present only when the active span context is valid; otherwise the
  key is omitted. Values are lowercase hex, 32 and 16 characters.
- `raw` holds provider top-level keys other than `answers` / `model` / `usage`,
  preserving upstream's forward-compatible `extra="allow"` posture. If
  `json.dumps(raw)` exceeds 4096 characters it is replaced wholesale with
  `{"_truncated": true}`.
- Every value is JSON-serializable by construction; `json.dumps(record)` must
  never raise. Test 3 in §6.2.

### 4.3 Field mapping: OpenAI Decisions response → harness record

| Provider field | Harness field | Rule |
| --- | --- | --- |
| `answers[i].name` | `answers[i].name` | must equal `questions[i].name` (invariant 2) |
| `answers[i].type == "predicate"` | `type: "predicate"` | |
| `answers[i].probability` | `probability` | float, clamped-checked to `[0,1]`; `confidence` stays `null` — predicate has no confidence upstream |
| `answers[i].type == "choice"` | `type: "choice"` | |
| `answers[i].choice` | `choice` | `str` or `bool` preserved exactly; never coerced |
| `answers[i].confidence` | `confidence` | float `[0,1]` |
| `answers[i].probabilities[] {value, probability}` | `probabilities[] {value, label: null, probability}` | order preserved; ≥2 entries |
| `answers[i].type == "score"` | `type: "score"` | |
| `answers[i].score` | `score` | float; a probability-weighted mean of 0-based level indices, range `0 … len(levels)-1`. The harness does **not** recompute or validate it against `probabilities` — it is a provider-side property |
| `answers[i].probabilities[] {value:int, label, probability}` | `probabilities[] {value, label, probability}` | `value` is the 0-based level index |
| `answers[i].type == "refusal"` | `type: "refusal"`, `refused: true` | `probability`, `choice`, `score`, `confidence` all `null`; `probabilities` empty. Legal for any question type |
| top-level `model` | `decisions.model` | the **resolved** model, overwriting the requested one |
| `usage.input_tokens`, `usage.output_tokens` | `decisions.usage` | ints; `bool` rejected (invariant 5) |
| `usage.*_details` and other `usage` keys | `decisions.usage` | passed through unchanged |
| any other top-level key | `decisions.raw` | capped per §4.2 |

### 4.4 Correlation IDs

Four identifiers, with one authoritative join key.

| Id | Value | Purpose |
| --- | --- | --- |
| `session_id` | copied from the turn's existing `additional_metadata["session_id"]`, written by both JSONL writers (`agent.py:146`, `session_runner.py:207`) | joins decisions to the run that produced them |
| `turn_index` | `TraceTurn.index` — the 0-based line number assigned by `load_trace()` (`eval.py:124`) | identifies the turn within the trace file |
| `trace_id` | the input file's stem, e.g. `trace_memory_cashew` from `trace_memory_cashew.jsonl` | identifies the trace across files |
| `decision_id` | `sha256(f"{trace_id}:{turn_index}:{question.name}".encode()).hexdigest()[:16]` | stable per answer |

- **`(session_id, turn_index)` is the authoritative correlation.** It is already
  present in both writers and survives file copies.
- `decision_id` is **deterministic by construction** — no UUIDs, no timestamps,
  no randomness. Re-annotating the same trace with the same questions yields the
  same ids, so output files diff cleanly and tests assert exact values. Test 5
  in §6.2.
- `otel.trace_id` / `otel.span_id` are recorded for joining to Channel A when an
  OTLP endpoint is configured. They are **not** authoritative: with no exporter
  configured the spans are dropped, so nothing on the other side of that join
  exists. Never make a code path depend on them.

---

## 5. Configuration and enablement

### 5.1 Default behaviour: nothing changes

With no new flag and no new environment variable, `traced-harness` behaves
exactly as at `838b0aa`. Specifically, and testably (test 18):

- `parse_args([])` produces the pre-existing namespace plus decide-fields that
  are `None`/defaults.
- Importing `traced_harness.main` does **not** import
  `traced_harness.decisions` or `traced_harness.decisions_openai` —
  asserted against `sys.modules`.
- No environment variable is read at import time.
- No trace file is written, read, or modified by the new code.

### 5.2 CLI

Follow the existing `eval` pattern in `parse_args` (`main.py:104-112`), which
special-cases `effective_argv[0]` before delegating to argparse. Both forms must
work, exactly as for `eval`:

```
traced-harness decide TRACE --questions FILE [options]
traced-harness --decide TRACE --questions FILE [options]
```

| Flag | Dest | Default | Meaning |
| --- | --- | --- | --- |
| `--decide PATH` / positional after `decide` | `decide_trace` | `None` | input trace JSONL |
| `--questions PATH` | `decide_questions` | `None` | required with `--decide`; native list or JSON Schema (§3.4) |
| `--decisions-backend NAME` | `decide_backend` | `$TRACED_DECISIONS_BACKEND` or `openai` | `openai` \| `replay` |
| `--decisions-model NAME` | `decide_model` | `$TRACED_DECISIONS_MODEL` or `gpt-6-luna` | |
| `--replay-file PATH` | `decide_replay` | `None` | required when backend is `replay` |
| `--decisions-out PATH` | `decide_out` | `None` | default `<stem>.decided.jsonl` |
| `--decisions-fail-on-error` | `decide_fail_on_error` | `False` | `on_error="raise"` and exit 1 |

`--decide` without `--questions`, or `--decisions-backend replay` without
`--replay-file`, exits 2 with a message naming the missing flag — argparse
convention, no traceback.

Exit codes: `0` success; `1` a decision error was recorded and
`--decisions-fail-on-error` was given (mirrors `--fail-on-errors` at
`main.py:106-114`); `2` misconfiguration.

**Wiring in `async_main`:** add a `decide_trace` branch at the very top of
`async_main`, directly above the existing `eval_trace` branch
(`main.py:143-154`), which lazily imports `traced_harness.decisions`, calls
`setup_telemetry()`, runs `annotate_trace`, prints the output path, and
`return`s. It must short-circuit before MCP resolution, skill discovery, and
agent creation — the decide path never constructs an agent and never needs a
Gemini key. Unlike the `eval` branch it *does* call `setup_telemetry()`, because
decisions emit spans; that call is a no-op without
`OTEL_EXPORTER_OTLP_ENDPOINT` (`telemetry.py:33-51`).

### 5.3 Environment variables

| Variable | Default | Read where |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | `decisions_openai.py` only, at `ask()` time. Missing ⇒ `DecisionsConfigError` **before** any request |
| `TRACED_DECISIONS_BACKEND` | `openai` | CLI default only |
| `TRACED_DECISIONS_MODEL` | `gpt-6-luna` | CLI default only |
| `TRACED_DECISIONS_OPENAI_URL` | `https://api.openai.com/v1/decisions` | `decisions_openai.py`; for gateways and local record/replay servers |
| `TRACED_DECISIONS_TIMEOUT` | `60` | seconds, float; `decisions_openai.py` |

No config file. No `llm` key store. No reading of `GOOGLE_API_KEY` /
`GEMINI_API_KEY` by any decisions code path, and no writing of
`OPENAI_API_KEY` into `os.environ` by anything.

### 5.4 Dependency changes

One line added to `[project.dependencies]` in `pyproject.toml`:

```toml
"httpx2>=2.13.0",
```

`uv lock` must report no change to resolved versions — `httpx2 2.13.1` is
already in `uv.lock` via `mcp 2.3.0`. **No `[project.optional-dependencies]`
section is introduced**; the repo has none and this change does not need one.
No new dev dependencies: `httpx2.MockTransport` ships with `httpx2` and
`pytest-asyncio>=1.4.0` is already in the dev group.

### 5.5 Licensing and attribution

`traced-harness` has **no root `LICENSE`** at `838b0aa`; the only `LICENSE`
files in the tree are inside vendored skill bundles. That is not this PR's to
fix (Q1, §7), but the attribution below is required regardless.

Because §2 chose reimplementation, Apache-2.0 §4(a)/(b)/(c) obligations are
**not** triggered — no copyrightable expression from upstream is copied. What is
required:

1. **`NOTICE` at the repo root** containing exactly:

   ```
   This product includes a reimplementation of the OpenAI Decisions request and
   response protocol as documented in llm-openai-decisions (commit 4827324),
   Copyright Simon Willison, licensed under the Apache License 2.0.
   https://github.com/simonw/llm-openai-decisions

   No source code from that project is included. The wire protocol, question
   and answer types, and response-validation rules were derived from its
   documented behaviour.
   ```

2. **Module docstring credit** in `decisions_openai.py`, naming the project, the
   commit `4827324`, and Apache-2.0.

3. **If any code is copied verbatim after all** — including small helpers such
   as upstream's `parse_json` — the obligations become mandatory and this PR
   must additionally: ship the full Apache-2.0 text at
   `licenses/llm-openai-decisions-LICENSE`, mark each file containing copied
   code with a prominent "changed from" notice (§4(b)), and preserve any
   attribution notices (§4(c)). Upstream ships **no `NOTICE`** and its
   `LICENSE` appendix placeholder is unfilled (`LICENSE:189`), so there is no
   upstream copyright line to carry beyond the author metadata naming Simon
   Willison. **Prefer not copying.**

4. **Provenance, recorded once in `docs/decisions.md`:** upstream's
   implementation is machine-generated per its own git history (commit
   `eeaf9ea`, *"Plugin, built by GPT-6 Astra Medium"*; `c3a1dff`, *"README,
   hand-edited by me"*). This does not affect the Apache grant, but a repo with
   an AI-provenance policy needs the fact on record.

5. Apache-2.0 is compatible with permissive licenses and GPLv3, and grants a
   patent license; it is **not** compatible with GPLv2-only. Irrelevant while
   this repo is unlicensed, relevant the moment Q1 is answered.

---

## 6. Acceptance criteria

Every box is objectively checkable by the verifier (`t_b9666490`) from a clean
checkout. Commands are given where the check is a command.

### 6.1 Build and baseline

- [ ] **A1** `uv sync` succeeds on a clean clone of the PR branch.
- [ ] **A2** `uv run pytest -q` passes, with **60 pre-existing tests still
      passing** and the new tests added on top. Report the total.
- [ ] **A3** `uv run ruff check src tests` → "All checks passed!".
      (`ruff check .` is expected to report 5 errors in vendored
      `.agents/` skill templates — pre-existing at `838b0aa`, out of scope,
      must not be "fixed".)
- [ ] **A4** `uv lock --check` passes and `git diff` on `uv.lock` shows **no
      resolved-version changes**, only the explicit `httpx2` requirement.
- [ ] **A5** `testpaths` in `pyproject.toml` is unchanged.

### 6.2 Unit tests — all 18 present, named, and passing

`tests/test_decisions.py` — no network, no credentials:

- [ ] **T1** `test_question_validation` — choice with <2 choices, duplicate
      `(type, value)` pairs, `levels` on a choice, `choices` on a score, <2
      levels, blank name, blank instructions: each raises `ValueError`.
- [ ] **T2** `test_json_schema_mapping` — boolean→predicate; string+enum→choice
      with enum order preserved; integer min/max→score with `n-m+1` labelled
      levels; `x-levels` honoured and length-checked; `description`→
      `instructions`; property order preserved; `{"type":"number"}` and a bare
      `{"type":"string"}` raise `DecisionsConfigError` naming the property.
- [ ] **T3** `test_answer_record_is_json_serializable` — `json.dumps` of every
      answer type round-trips; a refusal record has `refused: true` and `null`
      in all four value fields.
- [ ] **T4** `test_context_builder_is_deterministic` — byte-exact expected
      string for a fixture turn; tool parameters sorted; tool output truncated at
      2000 chars; the `### Tools called` section absent when there are none.
- [ ] **T5** `test_decision_id_stable` — the exact expected 16-char digest for a
      known `(trace_id, turn_index, name)`; two calls equal; any component
      changing changes the id.
- [ ] **T6** `test_annotate_trace_with_fake_backend` — against
      `tests/fixtures/traces/clean_run.jsonl` with a `_FakeBackend` defined in
      the test file (**fakes, not mocks** — follow
      `tests/test_session_runner.py:24-39`): output has one line per input line;
      every pre-existing key is byte-identical; only `additional_metadata.decisions`
      is added.
- [ ] **T7** `test_annotate_trace_does_not_mutate_input` — the input file's
      sha256 is unchanged after the run.
- [ ] **T8** `test_annotated_trace_still_loads_and_evaluates` —
      `load_trace(output)` succeeds and `evaluate_trace()` returns the **same**
      `total_turns`, `evaluated_turns`, `failed_tool_calls`, and
      `first_failure_turn` as for the input, including for
      `failing_peripheral.jsonl` and `multi_turn_cascade.jsonl`. This is the
      guard against decision records being misread as peripheral failures.
- [ ] **T9** `test_decision_span_records_measurement` — `decision_span()`
      yields a `SpanMeasurement` whose `as_record()` carries `wall_seconds`,
      `cpu_seconds`, and the `decision.backend` / `decision.question_count` /
      `decision.refusal_count` attributes. Assert on the **yielded handle**, not
      on exported spans: the repo never asserts span export anywhere
      (`tests/test_telemetry_spans.py`, `tests/test_memory.py:110-133` both test
      the handle), because `telemetry.py:18` makes the provider a process-global
      initialised once, so installing a test exporter would mean touching
      `telemetry.py` — out of scope (§1.2).
- [ ] **T10** `test_replay_backend` — consumes recorded bodies in order, runs
      them through the real validation path, and raises
      `DecisionBackendError` when exhausted.
- [ ] **T11** `test_on_error_record_vs_raise` — a backend raising
      `DecisionBackendError` yields an `error` string in that turn's slot and
      processing continues (`record`), or propagates (`raise`); a
      `DecisionsConfigError` propagates in both modes.

`tests/test_decisions_openai.py` — `httpx2.MockTransport`, fake key, no network:

- [ ] **T12** `test_request_payload_shape` — `model`/`input`/`questions` exactly
      as §3.7; `choices`/`levels` omitted rather than `null`; headers are
      exactly `Authorization: Bearer <key>` and `Content-Type: application/json`.
- [ ] **T13** `test_all_three_answer_types` — predicate, choice, and score
      responses from committed fixtures under `tests/fixtures/decisions/` map to
      the §4.3 fields, with `bool` and `str` choice values staying distinct.
- [ ] **T14** `test_refusal_accepted_for_any_question_type` — parametrized over
      all three question types.
- [ ] **T15** `test_response_validation_errors` — parametrized: answer-count
      mismatch, name mismatch, type mismatch, blank `model`, `usage` value of
      `True`, non-int `usage`, negative `usage`, duplicate JSON keys, `NaN`,
      invalid UTF-8, `{}`, `[]` → each raises `DecisionBackendError`.
- [ ] **T16** `test_http_error_statuses` — 302, 401, 422, 429, 500 each raise
      `DecisionBackendError`; in the 302 case the transport recorded exactly one
      request, proving the redirect was not followed.
- [ ] **T17** `test_api_key_never_leaks` — with a sentinel key value: the
      sentinel appears in no exception `str()` or `repr()`, in no captured log
      record, and nowhere in the written output file, for both a success and
      every failure path in T15/T16. Also: `test_missing_key_raises_before_request`
      — `DecisionsConfigError` with zero requests recorded.

`tests/test_decisions_cli.py`:

- [ ] **T18** `test_cli_and_default_behaviour_unchanged` — both `decide
      TRACE ...` and `--decide TRACE ...` parse to the same namespace;
      `--decide` without `--questions` exits 2; `parse_args([])` leaves all
      decide-fields at defaults; and after `import traced_harness.main`,
      neither `traced_harness.decisions` nor `traced_harness.decisions_openai`
      is in `sys.modules`.

### 6.3 End-to-end, offline

- [ ] **E1** This exact command succeeds with no network and no `OPENAI_API_KEY`
      in the environment:

      ```
      uv run traced-harness decide tests/fixtures/traces/clean_run.jsonl \
        --questions docs/examples/decision-questions.json \
        --decisions-backend replay \
        --replay-file docs/examples/recorded-decisions.json \
        --decisions-out /tmp/clean_run.decided.jsonl
      ```

- [ ] **E2** `/tmp/clean_run.decided.jsonl` has the same number of lines as the
      input, and every line carries `additional_metadata.decisions` with
      `schema_version`, `backend`, `trace_id`, `turn_index`, `session_id`, and a
      non-empty `answers[]` whose entries match §4.2.
- [ ] **E3** Re-running E1 produces a **byte-identical** output file
      (determinism), apart from the `measure` timings — which the check must
      exclude explicitly, not hand-wave.
- [ ] **E4** `uv run traced-harness eval /tmp/clean_run.decided.jsonl` produces
      the same verdict as `uv run traced-harness eval tests/fixtures/traces/clean_run.jsonl`.
- [ ] **E5** "Feature disabled" check: `uv run traced-harness --help` lists the
      new flags, and a run with none of them touches no file and reaches no
      decisions code (T18 covers the import assertion).
- [ ] **E6** `tests/fixtures/traces/*.jsonl` are unchanged in `git status` after
      the full test run and after E1.

### 6.4 Docs and attribution

- [ ] **D1** `docs/decisions.md` exists covering: what a decision is here, the
      three question types, the questions-file format with a worked example of
      both forms, the backend table, every flag and environment variable from §5,
      a **real** sample annotated trace line copied from an actual E1 run (not
      hand-written), the §5.5 provenance note, and an explicit "not implemented
      yet: MediaPipe, in-band capture" section.
- [ ] **D2** `README.md` gains a short "Decision backends" section linking to it.
- [ ] **D3** `NOTICE` exists at the repo root with the §5.5 text.
- [ ] **D4** `docs/examples/decision-questions.json` and
      `docs/examples/recorded-decisions.json` are committed and are the files E1
      uses.
- [ ] **D5** `decisions_openai.py`'s module docstring credits upstream with the
      commit SHA and Apache-2.0; `decisions.py`'s docstring carries the layering
      preamble stating D1 (mechanism here, rubric from the caller).

### 6.5 Secrets

- [ ] **S1** `git log -p` on the branch contains no value matching
      `sk-[A-Za-z0-9_-]{16,}` and no real key in any fixture. Fixtures use the
      literal `test-key-not-a-real-key`.
- [ ] **S2** No test requires `OPENAI_API_KEY`; the full suite passes with it
      unset **and** with it set to a sentinel (verify both).
- [ ] **S3** No `.env`, `.envrc`, or credential file is added.

### 6.6 CI

- [ ] **C1** `.github/workflows/test.yml` exists, triggered on `push` and
      `pull_request`, running on `ubuntu-latest` with `astral-sh/setup-uv`,
      matrix `python-version: ["3.12", "3.13"]`, executing `uv sync` then
      `uv run pytest -q` then `uv run ruff check src tests`.
- [ ] **C2** The workflow sets no secrets and defines no `OPENAI_API_KEY`.
- [ ] **C3** CI is green on the PR head commit. Report the run URL. If the repo
      has Actions disabled, say so explicitly rather than marking this passed —
      it is then a blocker for the owner, not a silent skip.

### 6.7 PR hygiene

- [ ] **P1** The PR description links to this spec by path and states which
      acceptance boxes are covered by which test.
- [ ] **P2** Conventional-commit subjects with scopes, matching repo history —
      e.g. `feat(decisions): calibrated decision backends over recorded traces`.
- [ ] **P3** The only modified pre-existing files are `main.py`,
      `pyproject.toml`, `uv.lock`, `README.md`, `__init__.py`. Any other
      modification must be justified in the PR description.
- [ ] **P4** `__init__.py` re-exports only `DecisionQuestion`, `DecisionAnswer`,
      `DecisionBackend`, `DECISIONS_METADATA_KEY`, and `annotate_trace` —
      matching how `__init__.py` curates today, and **not** exporting
      `decisions_openai` (which must stay lazily imported).

---

## 7. Risks and open questions

### Risks

**R1 — The upstream endpoint and model are unverified. (highest)**
`api.openai.com/v1/decisions` and `gpt-6-luna` were never called by the research
task; `docs/research/llm-openai-decisions.md` says so explicitly. This spec
therefore describes **what the client expects**, not confirmed server behaviour.
*Mitigation:* every acceptance criterion is satisfied offline via
`MockTransport` and the replay backend; no criterion requires a live call. The
PR description must state that the live path is unexercised. If a live call is
ever made, record the response body as a new fixture and add it to T13.

**R2 — Secrets in tests and CI.** *Mitigation:* S1–S3 and C2. Concretely: the
sentinel key is `test-key-not-a-real-key`; the backend raises before any request
when no key is present; the key is never written to a trace, never put on a
span, never logged, and T17 asserts it. CI defines no secrets, so a leak has no
live credential to leak.

**R3 — Decision records misread as peripheral failures.** `eval.py:33-49`
flags a tool whose output contains an `"error"` key, and `evaluate_trace()`
prunes every turn after the first failure. A refusal or a recorded decision
error in the wrong place would fabricate a cascade. *Mitigation:* decisions
never enter `tools_called[]` (§3.5 note); T8 asserts `evaluate_trace()` output
is identical before and after annotation on all three committed fixtures.

**R4 — The instrument/study boundary.** D1 argues a decision backend is
instrument, not subject. The repo owner may disagree, since `838b0aa` is the
commit that drew that line. *Mitigation:* the mechanism/rubric split (no default
questions, no scoring anywhere in `src/`), and `decisions_openai.py` being
lazily imported and never touched by core. If the owner rules the other way, the
fix is mechanical: move `decisions_openai.py` to the study repo and keep
`decisions.py`. Design for that move; do not entangle the two modules.

**R5 — Schema duplication.** The JSONL schema lives in two writers
(`agent.py:128`, `session_runner.py:181`) with no shared serializer. This PR
avoids the hazard by writing a third, separate annotation file rather than
changing either writer — but any future in-band work will hit it. Flagged as a
hotspot, not fixed here.

**R6 — `httpx2` is currently a transitive dependency.** If `mcp` ever drops it,
an undeclared import would break. *Mitigation:* §5.4 declares it explicitly.

**R7 — Trace bloat.** A long trace with many questions multiplies record size.
*Mitigation:* the 4096-character `raw` cap and the 2000-character tool-output
truncation in the context builder. Not otherwise bounded; acceptable for
benchmark-sized traces.

**R8 — The repo has no CI today.** Building it is in scope (C1), but a first
workflow on a repo whose owner has never enabled Actions may simply not run.
*Mitigation:* C3 requires the verifier to report this honestly rather than tick
the box.

### Open questions — for the repo owner, none blocking implementation

**Q1 — `traced-harness` has no root `LICENSE`.** Adding `NOTICE` to an
unlicensed repo is legally odd. This PR adds the attribution anyway because it
is the honest record; choosing a license is the owner's call. *Does not block:
implement as specified.*

**Q2 — Should MediaPipe land before or after this?** This spec deliberately
ships one backend plus an interface designed for two. If the owner wants both
in one PR, that PR cannot be written until someone has actually read the
MediaPipe API. *Does not block: a separate research card covers it.*

**Q3 — Should annotation eventually move in-band?** Out of scope by D2. Worth
revisiting once the interface has a second backend and the duplicated JSONL
schema (R5) is consolidated.

**Q4 — Concurrency.** `annotate_trace` is sequential for determinism and
rate-limit safety (§3.5.3). A `--concurrency N` flag is an obvious follow-up;
it is omitted here because it would make E3's byte-identical check harder to
state.

---

## 8. Implementation order

Each step leaves the tree green; the implementer can stop and ship at any
boundary.

1. `decisions.py` — types, validation, `DecisionBackend` ABC, errors, span,
   `decision_id`, context builder. Tests T1, T3, T4, T5, T9.
2. `questions_from_json_schema` + the questions-file loader. Test T2.
3. `ReplayDecisionBackend` + `annotate_trace`. Tests T6, T7, T8, T10, T11.
4. `decisions_openai.py` + `pyproject.toml` dependency line + `uv lock`.
   Tests T12–T17, fixtures under `tests/fixtures/decisions/`.
5. `main.py` wiring + `__init__.py` re-exports. Test T18.
6. `docs/examples/*` → run E1 → paste the real output line into
   `docs/decisions.md`. Docs, README, `NOTICE`.
7. `.github/workflows/test.yml`. Push, confirm green, open the PR linking this
   spec.

[upstream]: https://github.com/simonw/llm-openai-decisions

---

## 9. Amendments made during implementation (`t_4da11c3f`)

§0 asks for divergences to be recorded here rather than applied silently. Three
were needed; none changes a design decision or an acceptance criterion.

**A1 — The response parser lives in `decisions.py`, not `decisions_openai.py`.**
§3.7 places response validation in the OpenAI module, but §3.6 requires
`ReplayDecisionBackend` to run recorded bodies through "the same validation path
as a live response", while §3.7 also forbids `decisions.py` from importing
`decisions_openai`. Both cannot hold with the parser in the OpenAI module. The
strict JSON loader (`loads_strict`) and the five-invariant validator
(`parse_decision_response`) therefore live in `decisions.py`;
`decisions_openai.py` owns the transport, the payload, the credential and the
HTTP status check, and delegates parsing. Replay and OpenAI consequently share
one parser, which is what §3.6 was asking for. The dependency direction §3.7
cares about is unchanged: `decisions.py` still never imports
`decisions_openai`.

**A2 — `OpenAIDecisionsBackend(timeout=...)` defaults to `None`, not `60.0`.**
§3.7 gives the signature `timeout: float = 60.0`, and §5.3 requires
`TRACED_DECISIONS_TIMEOUT` to be read in `decisions_openai.py`. A literal
default makes "the caller passed 60" indistinguishable from "the caller said
nothing", so the environment variable could never apply. The parameter is
`float | None = None` and resolves to `TRACED_DECISIONS_TIMEOUT`, else `60.0`.
The effective default is the documented one; `backend.timeout` exposes it.

**A3 — `__init__.py` re-exports lazily (PEP 562 `__getattr__`).** P4 requires
five names re-exported from the package; T18 requires that importing
`traced_harness.main` leaves `traced_harness.decisions` out of `sys.modules`.
Importing `traced_harness.main` executes the package `__init__`, so an eager
re-export makes T18 unsatisfiable. The five names resolve on first attribute
access instead. Both hold, and `tests/test_decisions_cli.py` asserts each in a
clean subprocess interpreter.

Two notes that are not divergences:

- **§6.2 names 18 tests; this PR adds 91 test cases** across the three files,
  with every named behaviour covered (several of the 18 are parametrized into
  families, and the committed example files get coverage of their own).
- **The live path remains unexercised.** R1 stands exactly as written: nothing
  in this PR has called `api.openai.com/v1/decisions` or `gpt-6-luna`.


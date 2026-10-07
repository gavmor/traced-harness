# Decision backends

A **decision** here is a calibrated answer to one explicit question asked about
one recorded turn. Not prose about the turn — a number, with the distribution
behind it:

```json
{"name": "answered_the_question", "type": "predicate", "probability": 0.93}
```

`traced-harness` ships the *mechanism*: question and answer types, a backend
protocol, a span, a trace slot, and a driver that walks a recorded trace. It
ships no *rubric*. Which questions to ask is the study's business, and there is
no default question set, no scoring and no pass/fail opinion anywhere in `src/`.

Decisions are a **post-hoc annotation**. You record a trace the way you always
have, then annotate it afterwards into a new file. Nothing runs during a live
agent turn and the input trace is never modified.

## Try it, with no credential

```bash
uv run traced-harness decide tests/fixtures/traces/clean_run.jsonl \
  --questions docs/examples/decision-questions.json \
  --decisions-backend replay \
  --replay-file docs/examples/recorded-decisions.json \
  --decisions-out /tmp/clean_run.decided.jsonl
```

That prints the output path and writes one annotated line per input line. The
`replay` backend reads recorded response bodies instead of calling anything,
and runs them through the same parser and the same validation a live response
gets — so a replay run is a real exercise of the client, not a stub.

## The three question types

| Type | Asks for | Answer carries |
| --- | --- | --- |
| `predicate` | a yes/no judgement | `probability` in `[0, 1]` |
| `choice` | one option from a fixed list | `choice`, `confidence`, and the full `probabilities` distribution |
| `score` | a rung on an ordered ladder | `score` (a probability-weighted mean of 0-based level indices), `confidence`, `probabilities` |

Any question may also be answered with a `refusal`, which sets `refused: true`
and leaves every value field `null`.

Cross-field rules are enforced when the question is constructed, not when the
request fails: a `choice` needs at least two choices and no levels, a `score`
needs at least two levels and no choices, a `predicate` takes neither. Choice
values are never coerced across types — `true` and `"true"` are two distinct
options.

## The questions file

`--questions FILE` takes one JSON document, routed by its top-level type.

**A list** is read as native questions:

```json
[
  {"name": "answered_the_question",
   "instructions": "Did the assistant's output directly answer the user's input question?",
   "type": "predicate"},

  {"name": "tool_use",
   "instructions": "Judge the tool calls this turn made.",
   "type": "choice",
   "choices": [
     {"value": "necessary", "description": "Every call was needed to answer."},
     "unnecessary",
     "missing"
   ]},

  {"name": "answer_quality",
   "instructions": "Rate how well the output serves the user.",
   "type": "score",
   "levels": ["Poor", "Adequate", "Good", {"label": "Excellent", "description": "Could not be improved."}]}
]
```

`choices` and `levels` entries may be a bare value or an object with a
`description`.

**An object with `"type": "object"`** is read as a JSON Schema, and each entry
of `properties` becomes one question in declaration order:

```json
{
  "type": "object",
  "properties": {
    "answered": {"type": "boolean", "description": "Did it answer?"},
    "tool_use": {"type": "string", "enum": ["necessary", "unnecessary", "missing"],
                 "description": "Judge the tools."},
    "quality": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "Rate 1-5."},
    "graded": {"type": "integer", "minimum": 0, "maximum": 2,
               "x-levels": ["Bad", "OK", "Good"],
               "description": "Graded rating."}
  }
}
```

| JSON Schema property | Question |
| --- | --- |
| `{"type": "boolean"}` | `predicate` |
| `{"type": "string", "enum": [...]}` | `choice`, one option per enum entry, order preserved |
| `{"type": "integer", "minimum": m, "maximum": n}` | `score` with `n - m + 1` levels labelled `str(m) … str(n)` |
| `{"type": "integer", ..., "x-levels": [...]}` | `score` with those labels; the list length must equal `n - m + 1` |
| anything else | an error naming the property |

A property's `description` becomes the question's `instructions` and is
required. Unsupported types raise rather than being dropped: a silently missing
question would break the answer-count check against the provider's response.

## Backends

| Name | What it does | Needs |
| --- | --- | --- |
| `replay` | reads recorded response bodies from `--replay-file`, in order | nothing |
| `openai` | `POST https://api.openai.com/v1/decisions` | `OPENAI_API_KEY` |

> **The OpenAI backend is unexercised.** Neither the endpoint nor the default
> model `gpt-6-luna` has ever been called from this repository. The client
> encodes what the protocol is documented to be; it is not confirmed server
> behaviour. Everything in this document that you can run offline has been run.

Adding a backend means subclassing `DecisionBackend` and implementing one
`async def ask(context, questions) -> DecisionResult`. A synchronous,
on-device engine does its work in `asyncio.to_thread` and nothing in
`decisions.py` changes.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--decide PATH` / `decide PATH` | — | the recorded trace JSONL to annotate |
| `--questions PATH` | — | required with `--decide` |
| `--decisions-backend NAME` | `$TRACED_DECISIONS_BACKEND` or `openai` | `openai` or `replay` |
| `--decisions-model NAME` | `$TRACED_DECISIONS_MODEL` or `gpt-6-luna` | |
| `--replay-file PATH` | — | required when the backend is `replay` |
| `--decisions-out PATH` | `<stem>.decided.jsonl` beside the input | |
| `--decisions-fail-on-error` | off | propagate a backend failure and exit 1 instead of recording it |

Exit codes: `0` success, `1` a decision failed and `--decisions-fail-on-error`
was given, `2` misconfiguration.

## Environment

| Variable | Default | Read where |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | the OpenAI backend only, at call time |
| `TRACED_DECISIONS_BACKEND` | `openai` | CLI default only |
| `TRACED_DECISIONS_MODEL` | `gpt-6-luna` | CLI default only |
| `TRACED_DECISIONS_OPENAI_URL` | `https://api.openai.com/v1/decisions` | the OpenAI backend; for gateways and local record/replay servers |
| `TRACED_DECISIONS_TIMEOUT` | `60` | the OpenAI backend, seconds |

There is no config file and no key store. The key is read from the environment
at call time, never written to a trace, never put on a span, never logged, and
never included in an exception message. With no decide flag, no decisions code
runs and no environment variable above is read.

## What an annotated line looks like

Copied verbatim from the run at the top of this page (reformatted for reading;
the file itself is one line per turn):

```json
{
  "input": "What is the health and armor of the Colonial Spatha tank?",
  "actual_output": "The Spatha has 3650 HP and Tier 2 Heavy Vehicle armor.",
  "tools_called": [
    {
      "name": "get_vehicle_stats",
      "input_parameters": {"vehicle_name": "Spatha"},
      "output": "{\"title\": \"Spatha\", \"hp\": 3650, \"armor_type\": \"Tier 2 Heavy Vehicle\"}"
    }
  ],
  "additional_metadata": {
    "session_id": "clean_session",
    "agent": "traced_agno",
    "turn": 0,
    "decisions": {
      "schema_version": 1,
      "backend": "replay",
      "trace_id": "clean_run",
      "turn_index": 0,
      "session_id": "clean_session",
      "model": "gpt-6-luna",
      "answers": [
        {
          "decision_id": "f261ff1032b66d89",
          "name": "answered_the_question",
          "type": "predicate",
          "probability": 0.93,
          "choice": null,
          "score": null,
          "confidence": null,
          "probabilities": [],
          "refused": false
        },
        {
          "decision_id": "0a96ec5274fa6fe5",
          "name": "tool_use",
          "type": "choice",
          "probability": null,
          "choice": "necessary",
          "score": null,
          "confidence": 0.88,
          "probabilities": [
            {"value": "necessary", "label": null, "probability": 0.88},
            {"value": "unnecessary", "label": null, "probability": 0.09},
            {"value": "missing", "label": null, "probability": 0.03}
          ],
          "refused": false
        },
        {
          "decision_id": "8826d58b747e3a1f",
          "name": "answer_quality",
          "type": "score",
          "probability": null,
          "choice": null,
          "score": 2.53,
          "confidence": 0.63,
          "probabilities": [
            {"value": 0, "label": "Poor", "probability": 0.01},
            {"value": 1, "label": "Adequate", "probability": 0.08},
            {"value": 2, "label": "Good", "probability": 0.28},
            {"value": 3, "label": "Excellent", "probability": 0.63}
          ],
          "refused": false
        }
      ],
      "usage": {"input_tokens": 812, "output_tokens": 14},
      "measure": {
        "wall_seconds": 0.0001,
        "cpu_seconds": 0.0001,
        "bytes_before": 0,
        "bytes_after": 0,
        "bytes_growth": 0,
        "decision.backend": "replay",
        "decision.question_count": 3,
        "decision.refusal_count": 0
      },
      "otel": {
        "trace_id": "2b62beffbd8d28f557292462d977374f",
        "span_id": "ca69f396c00e6f35"
      },
      "raw": {}
    }
  }
}
```

Reading that record:

- Every original key is reproduced verbatim. Only
  `additional_metadata.decisions` is added, so `load_trace()` and
  `evaluate_trace()` work on the output unchanged — and a decision error never
  lands in `tools_called`, where the peripheral-health scanner would misread it
  as a failing tool and prune everything after it.
- `measure` is a `SpanMeasurement.as_record()`, the same shape
  `between_sessions` records use. Its dotted keys are span attributes merged
  flat.
- `otel` appears only when a span context is active, and is **not** a reliable
  join key: with no `OTEL_EXPORTER_OTLP_ENDPOINT` configured the spans are
  dropped, so there is nothing on the other side of that join. The authoritative
  correlation is `(session_id, turn_index)`.
- `decision_id` is `sha256("{trace_id}:{turn_index}:{question_name}")[:16]`.
  No UUIDs, no timestamps — re-annotating the same trace with the same
  questions produces the same ids, so output files diff cleanly.
- `raw` carries provider top-level keys other than `answers`, `model` and
  `usage`. Over 4096 characters it is replaced by `{"_truncated": true}`.

Re-running the command produces a byte-identical file apart from the `measure`
timings.

## Errors

By default a backend failure on one turn is recorded in that turn's slot and
the run continues:

```json
{"schema_version": 1, "backend": "replay", "trace_id": "clean_run",
 "turn_index": 1, "session_id": "clean_session",
 "error": "DecisionBackendError: replay file one.json is exhausted after 1 response(s)",
 "measure": {"...": "..."}}
```

`--decisions-fail-on-error` propagates the first failure and exits 1 instead.
A *configuration* error — a missing credential, an unknown backend, an
unreadable questions file — always propagates and exits 2, in both modes: it
means the run was set up wrong, not that one turn failed.

## Not implemented yet

- **The MediaPipe Decision Maker backend.** Nobody has read its API, and
  guessing an on-device interface would be fabrication. The `DecisionBackend`
  protocol is shaped so it drops in as a second subclass with no change to
  `decisions.py`: `ask` is async so a synchronous engine can wrap itself in
  `asyncio.to_thread`, nothing in the core imports a concrete backend, and the
  JSON-Schema mapping is the shared "same schema in" contract.
- **In-band decisions during a live agent run.** That needs either surgery on
  the single LLM call site or a new field on `TurnResult`, and the JSONL schema
  is currently duplicated across two writers, so a schema change has to touch
  both. Annotating afterwards avoids all of it.
- **Concurrency.** `annotate_trace` processes turns sequentially: determinism
  and rate-limit safety beat speed, and a byte-identical re-run is easier to
  state without it.

## Provenance and attribution

The protocol this client speaks was derived from
[`llm-openai-decisions`](https://github.com/simonw/llm-openai-decisions)
(commit `4827324`, Apache-2.0, Copyright Simon Willison). No source code from
that project is included here — see `NOTICE` at the repository root.

Worth having on the record: upstream's implementation is machine-generated per
its own git history (commit `eeaf9ea`, *"Plugin, built by GPT-6 Astra Medium"*;
`c3a1dff`, *"README, hand-edited by me"*). That does not affect the Apache
grant, but a repository with an AI-provenance policy wants the fact written
down rather than inferred.

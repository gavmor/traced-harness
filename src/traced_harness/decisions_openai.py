"""OpenAI Decisions endpoint backend.

A direct client for ``POST https://api.openai.com/v1/decisions``.

Attribution
-----------
This module is an independent reimplementation of the request shape and the
response-validation rules of `llm-openai-decisions
<https://github.com/simonw/llm-openai-decisions>`_ (commit ``4827324``),
Copyright Simon Willison, licensed under the Apache License 2.0. No source code
from that project is included here; the wire protocol, the question and answer
types, and the response-consistency checks were derived from its documented
behaviour. See ``NOTICE`` at the repository root.

The endpoint and the default model are **unverified** — nothing in this
repository has ever called them. What follows is what the client expects, not
confirmed server behaviour. Every test exercises it through
``httpx2.MockTransport``; no test, and no acceptance criterion, requires a live
call or a credential.

This module is never imported by :mod:`traced_harness.decisions` or by any core
harness module at import time: the CLI backend factory imports it lazily, the
same way :mod:`traced_harness.agent` imports :mod:`traced_harness.memory`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import httpx2

from traced_harness.decisions import (
    DecisionBackend,
    DecisionBackendError,
    DecisionQuestion,
    DecisionResult,
    DecisionsConfigError,
    loads_strict,
    parse_decision_response,
)

__all__ = [
    "DEFAULT_DECISIONS_MODEL",
    "OPENAI_DECISIONS_URL",
    "OpenAIDecisionsBackend",
]

OPENAI_DECISIONS_URL = "https://api.openai.com/v1/decisions"
DEFAULT_DECISIONS_MODEL = "gpt-6-luna"
DEFAULT_TIMEOUT_SECONDS = 60.0

#: How much of a failing response body may appear in an error message.
ERROR_BODY_LIMIT = 200


def _env_timeout() -> float:
    raw = os.environ.get("TRACED_DECISIONS_TIMEOUT")
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError:
        raise DecisionsConfigError(
            f"TRACED_DECISIONS_TIMEOUT is not a number: {raw!r}"
        ) from None


class OpenAIDecisionsBackend(DecisionBackend):
    """Ask the OpenAI Decisions endpoint for calibrated answers.

    The credential is read from ``OPENAI_API_KEY`` at :meth:`ask` time and
    nowhere else. It is never written to a trace, never put on a span, never
    logged, and never included in an exception message.
    """

    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_DECISIONS_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self.base_url = (
            base_url
            or os.environ.get("TRACED_DECISIONS_OPENAI_URL")
            or OPENAI_DECISIONS_URL
        )
        self.timeout = _env_timeout() if timeout is None else timeout
        self._client = client

    def __repr__(self) -> str:  # never render the credential
        return (
            f"OpenAIDecisionsBackend(model={self.model!r}, "
            f"base_url={self.base_url!r})"
        )

    def _resolve_key(self) -> str:
        key = self._api_key or os.environ.get("OPENAI_API_KEY") or ""
        if not key.strip():
            raise DecisionsConfigError(
                "OPENAI_API_KEY is not set; the openai decisions backend needs a "
                "credential (use --decisions-backend replay for an offline run)"
            )
        return key

    def _payload(
        self, context: str, questions: Sequence[DecisionQuestion]
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": context,
            "questions": [question.as_payload() for question in questions],
        }

    async def ask(
        self, context: str, questions: Sequence[DecisionQuestion]
    ) -> DecisionResult:
        if not questions:
            raise DecisionsConfigError("no questions supplied")
        key = self._resolve_key()

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        client = self._client
        owns_client = client is None
        if owns_client:
            # follow_redirects=False is load-bearing: a 302 must surface as an
            # error rather than replay the Authorization header at the target.
            client = httpx2.AsyncClient(
                timeout=self.timeout, follow_redirects=False
            )
        try:
            try:
                response = await client.post(
                    self.base_url, json=self._payload(context, questions), headers=headers
                )
            except httpx2.HTTPError as exc:
                # Deliberately no exception chaining and no exception text:
                # the request object carries the Authorization header.
                raise DecisionBackendError(
                    f"request to the decisions endpoint failed "
                    f"({type(exc).__name__})"
                ) from None
        finally:
            if owns_client:
                await client.aclose()

        if not 200 <= response.status_code < 300:
            raise DecisionBackendError(
                f"decisions endpoint returned HTTP {response.status_code}: "
                f"{_safe_body(response)}"
            )

        body = loads_strict(response.content)
        return parse_decision_response(body, questions, self.name)


def _safe_body(response: Any) -> str:
    """At most ``ERROR_BODY_LIMIT`` characters of a response body, never headers."""
    try:
        text = response.content.decode("utf-8", errors="replace")
    except (AttributeError, UnicodeError):
        return "<unreadable body>"
    return text[:ERROR_BODY_LIMIT]

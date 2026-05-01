"""Anthropic ``check`` tests.

The status-code handling is shared with OpenAI and exhaustively tested
in ``tests/models/openai/test_check.py``. This file only confirms the
provider-specific 200 path so we know URL routing is wired up.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ai.models.anthropic import anthropic, check
from ai.models.core import client as client_

_MODEL = anthropic("claude-opus-4-6")


def _client_with_mock(
    status_code: int = 200,
    json_body: Any = None,
    base_url: str = "https://anthropic.test",
) -> client_.Client:
    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps(json_body or {}).encode()
        return httpx.Response(status_code, content=body)

    client = client_.Client(base_url=base_url, api_key="sk-test-key")
    client._http = httpx.AsyncClient(
        base_url=base_url,
        transport=httpx.MockTransport(_handler),
    )
    return client


async def test_200_returns_true() -> None:
    client = _client_with_mock(200, {"id": "claude-opus-4-6", "type": "model"})
    assert await check.check(client, _MODEL) is True

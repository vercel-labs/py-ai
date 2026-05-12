"""Anthropic ``check`` tests.

The status-code handling is shared with OpenAI and exhaustively tested
in ``tests/providers/openai/test_check.py``. This file only confirms the
provider-specific 200 path so we know URL routing is wired up.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ai.models.core import client as client_
from ai.providers.anthropic import anthropic, anthropic_like, check

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


async def test_custom_anthropic_version_header() -> None:
    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        body = json.dumps({"id": "custom-model", "type": "model"}).encode()
        return httpx.Response(200, content=body)

    provider = anthropic_like(
        name="custom-anthropic",
        base_url="https://anthropic.test",
        anthropic_version="2024-01-01",
    )
    client = client_.Client(base_url="https://anthropic.test", api_key="sk-test-key")
    client._http = httpx.AsyncClient(
        base_url="https://anthropic.test",
        transport=httpx.MockTransport(_handler),
    )

    assert await check.check(client, provider("custom-model")) is True
    assert captured_headers["anthropic-version"] == "2024-01-01"

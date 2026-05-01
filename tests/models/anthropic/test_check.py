from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

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


async def test_shared_status_logic_smoke() -> None:
    assert await check.check(_client_with_mock(401), _MODEL) is False
    with pytest.raises(httpx.HTTPStatusError):
        await check.check(_client_with_mock(500), _MODEL)

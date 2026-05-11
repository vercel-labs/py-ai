"""Google ``check`` tests."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ai.models.core import client as client_
from ai.models.google import check, google

_MODEL = google("gemini-2.5-flash")


def _client_with_mock(
    status_code: int = 200,
    json_body: Any = None,
    base_url: str = "https://google.test/v1beta",
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
    client = _client_with_mock(200, {"name": "models/gemini-2.5-flash"})
    assert await check.check(client, _MODEL) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_client_error_returns_false(status: int) -> None:
    assert await check.check(_client_with_mock(status), _MODEL) is False


async def test_500_raises() -> None:
    with pytest.raises(httpx.HTTPStatusError):
        await check.check(_client_with_mock(500), _MODEL)


async def test_no_api_key_returns_false() -> None:
    client = client_.Client(base_url="https://google.test/v1beta", api_key=None)
    assert await check.check(client, _MODEL) is False

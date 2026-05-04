from __future__ import annotations

import httpx

from ai.models.anthropic import anthropic
from ai.models.core import client as client_


async def test_list_gets_models_with_provider_headers_and_sorts_ids() -> None:
    captured_urls: list[str] = []
    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        captured_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "claude-z"},
                    {"id": "claude-a"},
                ]
            },
        )

    client = client_.Client(base_url="https://anthropic.test", api_key="sk-test")
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    try:
        ids = await anthropic.list(client=client)
    finally:
        await client.aclose()

    assert captured_urls == ["https://anthropic.test/v1/models"]
    assert captured_headers["x-api-key"] == "sk-test"
    assert captured_headers["anthropic-version"] == "2023-06-01"
    assert ids == ["claude-a", "claude-z"]

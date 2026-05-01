from __future__ import annotations

import httpx

from ai.models.core import client as client_
from ai.models.openai import openai


async def test_list_gets_models_with_auth_header_and_sorts_ids() -> None:
    captured_urls: list[str] = []
    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        captured_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "gpt-z"},
                    {"id": "gpt-a"},
                ]
            },
        )

    client = client_.Client(base_url="https://openai.test/v1", api_key="sk-test")
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    try:
        ids = await openai.list(client=client)
    finally:
        await client.aclose()

    assert captured_urls == ["https://openai.test/v1/models"]
    assert captured_headers["authorization"] == "Bearer sk-test"
    assert ids == ["gpt-a", "gpt-z"]

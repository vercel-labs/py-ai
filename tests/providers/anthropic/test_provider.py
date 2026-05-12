from __future__ import annotations

import httpx
import pytest

from ai.models.core import client as client_
from ai.providers.anthropic import anthropic, anthropic_like


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


def test_base_url_defaults_when_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert anthropic.base_url == "https://api.anthropic.com"


def test_base_url_reads_anthropic_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
    assert anthropic.base_url == "https://proxy.example.com"


def test_client_uses_anthropic_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    c = anthropic.client()
    assert c.base_url == "https://proxy.example.com"
    assert c.api_key == "sk-test"


def test_anthropic_like_creates_generic_compatible_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUSTOM_ANTHROPIC_API_KEY", "sk-custom")
    provider = anthropic_like(
        name="custom-anthropic",
        base_url="https://custom.example.com",
        api_key_env="CUSTOM_ANTHROPIC_API_KEY",
        anthropic_version="2024-01-01",
    )

    model = provider("custom-model")
    client = provider.client()

    assert repr(provider) == "custom-anthropic"
    assert provider.adapter == "anthropic"
    assert provider.base_url == "https://custom.example.com"
    assert provider.anthropic_version == "2024-01-01"
    assert client.base_url == "https://custom.example.com"
    assert client.api_key == "sk-custom"
    assert model.id == "custom-model"
    assert model.adapter == "anthropic"
    assert model.provider is provider


def test_anthropic_like_reads_custom_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUSTOM_ANTHROPIC_BASE_URL", "https://proxy.example.com")
    provider = anthropic_like(
        name="custom-anthropic",
        base_url="https://custom.example.com",
        base_url_env="CUSTOM_ANTHROPIC_BASE_URL",
    )

    assert provider.base_url == "https://proxy.example.com"

from __future__ import annotations

import anthropic
import httpx
import pytest

import ai
from ai.providers.anthropic import AnthropicCompatibleProvider, adapter


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

    provider = ai.get_provider(
        "anthropic",
        base_url="https://anthropic.test",
        api_key="sk-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    try:
        ids = await provider.list()
    finally:
        await provider.aclose()

    assert captured_urls == ["https://anthropic.test/v1/models"]
    assert captured_headers["x-api-key"] == "sk-test"
    assert captured_headers["anthropic-version"] == "2023-06-01"
    assert ids == ["claude-a", "claude-z"]


async def test_get_provider_accepts_anthropic_sdk_client() -> None:
    sdk_client = anthropic.AsyncAnthropic(api_key="sk-test")
    provider = ai.get_provider("anthropic", client=sdk_client)

    try:
        assert isinstance(provider, AnthropicCompatibleProvider)
        assert provider.sdk_client is sdk_client
        model = ai.Model("claude-sonnet-4-6", provider=provider)
        assert adapter._make_client(model) is sdk_client
    finally:
        await sdk_client.close()


def test_base_url_defaults_when_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert ai.get_provider("anthropic").base_url == "https://api.anthropic.com"


def test_base_url_reads_anthropic_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
    assert ai.get_provider("anthropic").base_url == "https://proxy.example.com"


def test_provider_uses_anthropic_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    provider = ai.get_provider("anthropic")
    assert provider.base_url == "https://proxy.example.com"
    assert provider.api_key == "sk-test"


def test_get_provider_accepts_base_url_and_api_key() -> None:
    provider = ai.get_provider(
        "anthropic",
        base_url="https://custom.example.com",
        api_key="sk-custom",
    )

    model = ai.Model("custom-model", provider=provider)
    assert repr(provider) == "anthropic"
    assert provider.adapter == "anthropic"
    assert provider.base_url == "https://custom.example.com"
    assert provider.api_key == "sk-custom"
    assert model.id == "custom-model"
    assert model.adapter == "anthropic"
    assert model.provider is provider


def test_get_provider_env_overrides_base_url_env() -> None:
    provider = ai.get_provider(
        "anthropic",
        env={"ANTHROPIC_BASE_URL": "https://proxy.example.com"},
    )

    assert provider.base_url == "https://proxy.example.com"

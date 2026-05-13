from __future__ import annotations

import httpx
import openai
import pytest

import ai
from ai.providers.openai import OpenAICompatibleProvider, adapter


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

    provider = ai.get_provider(
        "openai",
        base_url="https://openai.test/v1",
        api_key="sk-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    try:
        ids = await provider.list()
    finally:
        await provider.aclose()

    assert captured_urls == ["https://openai.test/v1/models"]
    assert captured_headers["authorization"] == "Bearer sk-test"
    assert ids == ["gpt-a", "gpt-z"]


async def test_get_provider_accepts_openai_sdk_client() -> None:
    sdk_client = openai.AsyncOpenAI(api_key="sk-test")
    provider = ai.get_provider("openai", client=sdk_client)

    try:
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.sdk_client is sdk_client
        assert provider.is_configured() is True
        model = ai.Model("gpt-5.4", provider=provider)
        assert adapter._make_client(model) is sdk_client
    finally:
        await sdk_client.close()


def test_base_url_defaults_when_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert ai.get_provider("openai").base_url == "https://api.openai.com/v1"


def test_base_url_reads_openai_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.com/v1")
    assert ai.get_provider("openai").base_url == "https://proxy.example.com/v1"


def test_provider_uses_openai_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = ai.get_provider("openai")
    assert provider.base_url == "https://proxy.example.com/v1"
    assert provider.api_key == "sk-test"
    assert provider.is_configured() is True


def test_provider_is_configured_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert ai.get_provider("openai").is_configured() is False
    assert ai.get_provider("openai", api_key="sk-test").is_configured() is True


def test_get_provider_accepts_base_url_and_api_key() -> None:
    provider = ai.get_provider(
        "openai",
        base_url="https://custom.example.com/v1",
        api_key="sk-custom",
    )

    model = ai.Model("custom-model", provider=provider)
    assert repr(provider) == "openai"
    assert provider.adapter == "openai"
    assert provider.base_url == "https://custom.example.com/v1"
    assert provider.api_key == "sk-custom"
    assert provider.is_configured() is True
    assert model.id == "custom-model"
    assert model.adapter == "openai"
    assert model.provider is provider
    assert isinstance(provider, ai.Provider)


def test_get_provider_env_overrides_base_url_env() -> None:
    provider = ai.get_provider(
        "openai",
        env={"OPENAI_BASE_URL": "https://proxy.example.com/v1"},
    )

    assert provider.base_url == "https://proxy.example.com/v1"


def test_get_provider_expands_config_envs_in_base_url() -> None:
    provider = ai.get_provider(
        "cloudflare-workers-ai",
        env={"CLOUDFLARE_ACCOUNT_ID": "account-123"},
    )

    assert provider.config_envs == ("CLOUDFLARE_ACCOUNT_ID",)
    assert provider.base_url == (
        "https://api.cloudflare.com/client/v4/accounts/account-123/ai/v1"
    )


def test_provider_is_configured_requires_config_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLOUDFLARE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)

    missing_account = ai.get_provider(
        "cloudflare-workers-ai",
        env={"CLOUDFLARE_API_KEY": "sk-test"},
    )
    configured = ai.get_provider(
        "cloudflare-workers-ai",
        env={
            "CLOUDFLARE_API_KEY": "sk-test",
            "CLOUDFLARE_ACCOUNT_ID": "account-123",
        },
    )

    assert missing_account.is_configured() is False
    assert configured.is_configured() is True
    assert configured.base_url == (
        "https://api.cloudflare.com/client/v4/accounts/account-123/ai/v1"
    )

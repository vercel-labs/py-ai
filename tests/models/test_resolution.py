import pytest

import ai
from ai import models
from ai.providers import ai_gateway, anthropic, openai


def test_get_resolves_provider_qualified_model_id() -> None:
    model = ai.get_model("openai:gpt-5")

    assert model.id == "gpt-5"
    assert model.adapter == "openai"
    assert model.provider is openai


def test_get_resolves_slash_qualified_model_id() -> None:
    model = models.get_model("anthropic/claude-sonnet-4-5")

    assert model.id == "claude-sonnet-4-5"
    assert model.adapter == "anthropic"
    assert model.provider is anthropic


def test_provider_from_id_resolves_openai_compatible_provider() -> None:
    provider = models.Provider.from_id("deepseek")

    assert provider.name == "deepseek"
    assert provider.adapter == "openai"
    assert provider.default_base_url == "https://api.deepseek.com"
    assert provider.api_key_env == "DEEPSEEK_API_KEY"
    assert provider.config_envs == ()


def test_provider_from_id_uses_template_envs_for_base_url() -> None:
    provider = models.Provider.from_id("cloudflare-workers-ai")

    assert provider.default_base_url == (
        "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/v1"
    )
    assert provider.api_key_env == "CLOUDFLARE_API_KEY"
    assert provider.config_envs == ("CLOUDFLARE_ACCOUNT_ID",)


def test_provider_from_id_detects_token_env_after_url_env() -> None:
    provider = models.Provider.from_id("databricks")

    assert (
        provider.default_base_url == "https://${DATABRICKS_HOST}/ai-gateway/mlflow/v1"
    )
    assert provider.api_key_env == "DATABRICKS_TOKEN"
    assert provider.config_envs == ("DATABRICKS_HOST",)


def test_provider_from_id_resolves_gateway_provider() -> None:
    assert models.Provider.from_id("vercel") is ai_gateway


def test_get_uses_model_provider_config_for_anthropic_compatibility() -> None:
    model = models.get_model("azure:claude-sonnet-4-5")

    assert model.id == "claude-sonnet-4-5"
    assert model.adapter == "anthropic"
    assert model.provider.name == "azure"
    assert model.provider.default_base_url == (
        "https://${AZURE_RESOURCE_NAME}.services.ai.azure.com/anthropic/v1"
    )
    assert model.provider.api_key_env == "AZURE_API_KEY"
    assert model.provider.config_envs == ("AZURE_RESOURCE_NAME",)


def test_get_uses_model_provider_config_for_openai_compatibility() -> None:
    model = models.get_model("azure:kimi-k2.5")

    assert model.id == "kimi-k2.5"
    assert model.adapter == "openai"
    assert model.provider.name == "azure"
    assert model.provider.default_base_url == (
        "https://${AZURE_RESOURCE_NAME}.services.ai.azure.com/models"
    )
    assert model.provider.api_key_env == "AZURE_API_KEY"
    assert model.provider.config_envs == ("AZURE_RESOURCE_NAME",)


def test_provider_from_id_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown provider id"):
        models.Provider.from_id("missing-provider")


def test_provider_from_id_rejects_unsupported_provider_package() -> None:
    with pytest.raises(ai.UnsupportedProviderError) as exc_info:
        models.Provider.from_id("google")

    assert exc_info.value.provider_id == "google"


def test_get_rejects_unsupported_provider_package() -> None:
    with pytest.raises(ai.errors.UnsupportedProviderError):
        models.get_model("google:gemini-2.5-pro")


def test_get_rejects_unqualified_model_id() -> None:
    with pytest.raises(ValueError, match="known provider id"):
        models.get_model("gpt-5")

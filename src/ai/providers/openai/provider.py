"""OpenAI-compatible providers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar

from .. import base

if TYPE_CHECKING:
    import httpx
    import modelsdotdev
    import openai

    OpenAIClient = httpx.AsyncClient | openai.AsyncOpenAI
else:
    OpenAIClient = object

_BASE_URL = "https://api.openai.com/v1"
_BASE_URL_ENV = "OPENAI_BASE_URL"
_API_KEY_ENV = "OPENAI_API_KEY"


class OpenAICompatibleProvider(base.Provider):
    """Provider configuration for OpenAI-compatible APIs."""

    handles: ClassVar[tuple[str, ...]] = (
        "openai",
        "@ai-sdk/openai",
        "@ai-sdk/openai-compatible",
    )

    def __init__(
        self,
        *,
        name: str,
        default_base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        base_url_env: str | None = None,
        config_envs: Iterable[str] | None = None,
        env: Mapping[str, str] | None = None,
        client: OpenAIClient | None = None,
    ) -> None:
        import httpx as _httpx
        import openai as _openai

        if isinstance(client, _openai.AsyncOpenAI):
            sdk_client = client
            http_client = None
        elif isinstance(client, _httpx.AsyncClient) or client is None:
            sdk_client = None
            http_client = client
        else:
            raise TypeError(
                "OpenAI providers require an httpx.AsyncClient or openai.AsyncOpenAI"
            )

        super().__init__(
            name=name,
            adapter="openai",
            base_url=default_base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            config_envs=config_envs,
            env=env,
            client=http_client,
        )
        self._sdk_client = sdk_client

    @property
    def sdk_client(self) -> openai.AsyncOpenAI | None:
        """User-provided OpenAI SDK client, if configured."""
        return self._sdk_client

    def is_configured(self) -> bool:
        if self.sdk_client is not None:
            return True
        if not self.api_key:
            return False
        return super().is_configured()

    @classmethod
    def from_modelsdev_provider(
        cls,
        provider: modelsdotdev.Provider,
        *,
        model_provider_config: modelsdotdev.ModelProviderConfig | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        env: Mapping[str, str] | None = None,
        client: OpenAIClient | None = None,
    ) -> base.Provider:
        resolved_base_url = base_url or base.provider_base_url(
            provider,
            model_provider_config,
        )
        if resolved_base_url is None and provider.id == "openai":
            resolved_base_url = _BASE_URL
        if resolved_base_url is None:
            raise ValueError(f"provider {provider.id!r} does not declare an API URL")
        api_key_env, config_envs = base.provider_config(provider, model_provider_config)
        return cls(
            name=provider.id,
            default_base_url=resolved_base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url_env=_BASE_URL_ENV
            if provider.id == "openai" and base_url is None
            else None,
            config_envs=config_envs,
            env=env,
            client=client,
        )

    @property
    def tools(self) -> ModuleType:
        """The provider's built-in tool factories.

        Convenience accessor: ``openai.tools.web_search(...)``. The
        chat-completions adapter currently raises if a built-in tool is
        passed; route via the AI Gateway provider until a Responses
        adapter ships.
        """
        from . import tools as tools_module

        return tools_module

    async def list(self) -> list[str]:
        """List available model IDs from the OpenAI-compatible API."""
        if self.sdk_client is not None:
            sdk_models = await self.sdk_client.models.list()
            return sorted(str(m.id) for m in sdk_models.data)

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = await self.http.get(
            f"{self.base_url.rstrip('/')}/models",
            headers=headers,
        )
        response.raise_for_status()
        response_data: list[dict[str, object]] = response.json().get("data", [])
        return sorted(str(m["id"]) for m in response_data)


__all__ = ["OpenAICompatibleProvider"]

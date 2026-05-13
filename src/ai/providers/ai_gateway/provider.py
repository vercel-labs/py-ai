"""AI Gateway provider.

Defines the callable :data:`ai_gateway` provider."""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

from .. import base

if TYPE_CHECKING:
    import httpx
    import modelsdotdev

_BASE_URL = "https://ai-gateway.vercel.sh/v3/ai"
_API_KEY_ENV = "AI_GATEWAY_API_KEY"


class GatewayProvider(base.Provider):
    """Provider configuration for the Vercel AI Gateway."""

    handles: ClassVar[tuple[str, ...]] = ("vercel", "@ai-sdk/gateway")

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _BASE_URL,
        env: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            name="ai-gateway",
            adapter="ai-gateway-v3",
            base_url=base_url,
            api_key=api_key,
            api_key_env=_API_KEY_ENV,
            env=env,
            client=client,
        )

    @classmethod
    def from_modelsdev_provider(
        cls,
        provider: modelsdotdev.Provider,
        *,
        model_provider_config: modelsdotdev.ModelProviderConfig | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        env: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> base.Provider:
        return cls(
            api_key=api_key,
            base_url=base_url or _BASE_URL,
            env=env,
            client=client,
        )

    @property
    def tools(self) -> ModuleType:
        """Gateway-native built-in tool factories.

        Convenience accessor: ``ai_gateway.tools.perplexity_search(...)``.
        These tools are executed server-side by the gateway and work
        with any gateway-routed model.
        """
        from . import tools as tools_module

        return tools_module

    async def list(self) -> list[str]:
        """List available model IDs from the AI Gateway."""
        from . import client as gateway_client

        gateway = gateway_client.GatewayClient(self)
        response = await gateway.get("config")
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return sorted(str(m["id"]) for m in data.get("models", []))


__all__ = ["GatewayProvider"]

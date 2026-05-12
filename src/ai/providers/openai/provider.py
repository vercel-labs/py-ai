"""OpenAI-compatible providers."""

import dataclasses
import os
from types import ModuleType

from ...models import core

_BASE_URL = "https://api.openai.com/v1"
_BASE_URL_ENV = "OPENAI_BASE_URL"
_API_KEY_ENV = "OPENAI_API_KEY"


@dataclasses.dataclass(frozen=True)
class OpenAICompatibleProvider:
    """Callable provider for OpenAI-compatible APIs.

    ``provider("gpt-5.4")`` returns a :class:`Model` that uses the OpenAI
    chat-completions adapter.
    """

    name: str
    default_base_url: str
    api_key_env: str | None = None
    base_url_env: str | None = None

    @property
    def adapter(self) -> str:
        return "openai"

    @property
    def base_url(self) -> str:
        if self.base_url_env:
            return os.environ.get(self.base_url_env) or self.default_base_url
        return self.default_base_url

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

    def client(self) -> core.client.Client:
        """Create a :class:`Client` from env-var credentials.

        ``base_url_env`` overrides the default base URL when configured.
        """
        return core.client.Client(
            base_url=self.base_url,
            api_key=os.environ.get(self.api_key_env) if self.api_key_env else None,
        )

    async def check(self, client: core.client.Client, model: core.model.Model) -> bool:
        """Delegate to :func:`openai.check.check`."""
        from . import check as check_

        return await check_.check(client, model)

    def __call__(
        self,
        model_id: str,
        *,
        client: core.client.Client | None = None,
    ) -> core.model.Model:
        return core.model.Model(
            id=model_id,
            adapter=self.adapter,
            provider=self,
            client=client,
        )

    async def list(self, *, client: core.client.Client | None = None) -> list[str]:
        """List available model IDs from the OpenAI-compatible API."""
        c = client or self.client()
        headers = {"Authorization": f"Bearer {c.api_key}"} if c.api_key else {}
        response = await c.http.get(f"{c.base_url.rstrip('/')}/models", headers=headers)
        response.raise_for_status()
        data: list[dict[str, object]] = response.json().get("data", [])
        return sorted(str(m["id"]) for m in data)

    def __repr__(self) -> str:
        return self.name


def openai_like(
    *,
    name: str,
    base_url: str,
    api_key_env: str | None = None,
    base_url_env: str | None = None,
) -> OpenAICompatibleProvider:
    """Create a provider for an OpenAI-compatible API."""
    return OpenAICompatibleProvider(
        name=name,
        default_base_url=base_url,
        api_key_env=api_key_env,
        base_url_env=base_url_env,
    )


openai = openai_like(
    name="openai",
    base_url=_BASE_URL,
    api_key_env=_API_KEY_ENV,
    base_url_env=_BASE_URL_ENV,
)

__all__ = ["OpenAICompatibleProvider", "openai", "openai_like"]

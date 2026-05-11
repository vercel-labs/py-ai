"""Google Gemini provider.

Defines the callable :data:`google` provider, which satisfies the
:class:`~ai.models.core.proto.Provider` protocol.
"""

from __future__ import annotations

import os
from types import ModuleType

from .. import core

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_BASE_URL_ENV = "GOOGLE_BASE_URL"
_PRIMARY_API_KEY_ENV = "GOOGLE_API_KEY"
_FALLBACK_API_KEY_ENV = "GEMINI_API_KEY"


def _api_key_from_env() -> str | None:
    """Return Gemini Developer API key using google-genai precedence."""
    return os.environ.get(_PRIMARY_API_KEY_ENV) or os.environ.get(_FALLBACK_API_KEY_ENV)


class _Google:
    """Callable provider factory for Google Gemini Developer API models.

    Satisfies the :class:`~ai.models.core.proto.Provider` protocol.
    """

    @property
    def api_key_env(self) -> str:
        return _PRIMARY_API_KEY_ENV

    @property
    def base_url(self) -> str:
        return os.environ.get(_BASE_URL_ENV) or _BASE_URL

    @property
    def adapter(self) -> str:
        return "google"

    @property
    def name(self) -> str:
        return "google"

    @property
    def tools(self) -> ModuleType:
        """The provider's built-in tool factories.

        Convenience accessor: ``google.tools.google_search(...)``.
        """
        from . import tools as tools_module

        return tools_module

    def client(self) -> core.client.Client:
        """Create a :class:`Client` from env-var credentials.

        ``GOOGLE_BASE_URL`` overrides the default Gemini Developer API base URL
        when set. ``GOOGLE_API_KEY`` takes precedence over ``GEMINI_API_KEY``,
        matching the first-party ``google-genai`` SDK.
        """
        return core.client.Client(
            base_url=self.base_url,
            api_key=_api_key_from_env(),
        )

    async def check(self, client: core.client.Client, model: core.model.Model) -> bool:
        """Delegate to :func:`google.check.check`."""
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
        """List available Gemini Developer API model IDs."""
        c = client or self.client()
        headers = {"x-goog-api-key": c.api_key or ""}
        response = await c.http.get(f"{c.base_url.rstrip('/')}/models", headers=headers)
        response.raise_for_status()
        data: list[dict[str, object]] = response.json().get("models", [])
        ids: list[str] = []
        for model in data:
            raw = str(model.get("name") or model.get("id") or "")
            if not raw:
                continue
            ids.append(raw.removeprefix("models/"))
        return sorted(ids)

    def __repr__(self) -> str:
        return "google"


google = _Google()

__all__ = ["google"]

"""AI Gateway provider.

Usage::

    from ai.models import ai_gateway

    model = ai_gateway("anthropic/claude-sonnet-4")
    ids = await ai_gateway.list()

Heavy adapter modules (``.generate``, ``.stream``) are loaded lazily so that
``import ai`` does not pull in ``httpx`` and other I/O libraries at import
time.  This matters for sandboxed runtimes (e.g. Temporal workflow workers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.model import Model
from . import errors

if TYPE_CHECKING:
    from ..core.client import Client

_BASE_URL = "https://ai-gateway.vercel.sh/v3/ai"
_API_KEY_ENV = "AI_GATEWAY_API_KEY"
_PROTOCOL_VERSION = "0.0.1"


class _AIGateway:
    """Callable provider factory for the Vercel AI Gateway."""

    def __call__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        client: Client | None = None,
    ) -> Model:
        return Model(
            id=model_id,
            adapter="ai-gateway-v3",
            provider="ai-gateway",
            base_url=base_url or _BASE_URL,
            api_key_env=_API_KEY_ENV,
            client=client,
        )

    async def list(self, *, client: Client | None = None) -> list[str]:
        """List available model IDs from the AI Gateway."""
        from ..core import client as client_

        c = client or client_.Client(
            base_url=_BASE_URL,
            api_key=__import__("os").environ.get(_API_KEY_ENV),
        )
        base_url = c.base_url.rstrip("/")
        headers: dict[str, str] = {
            "ai-gateway-protocol-version": _PROTOCOL_VERSION,
        }
        if c.api_key:
            headers["Authorization"] = f"Bearer {c.api_key}"
            headers["ai-gateway-auth-method"] = "api-key"

        config_url = f"{base_url}/config"
        response = await c.http.get(config_url, headers=headers)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return sorted(str(m["id"]) for m in data.get("models", []))

    def __repr__(self) -> str:
        return "ai_gateway"


ai_gateway = _AIGateway()

__all__ = [
    "ai_gateway",
    "errors",
]


def __getattr__(name: str) -> object:
    if name == "generate":
        from .generate import generate

        return generate
    if name == "stream":
        from .stream import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

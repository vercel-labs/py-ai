"""Anthropic provider.

Usage::

    from ai.models import anthropic

    model = anthropic("claude-sonnet-4-6")
    ids = await anthropic.list()

The adapter module is loaded lazily to avoid pulling in the ``anthropic``
SDK at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.model import Model

if TYPE_CHECKING:
    from ..core.client import Client

_BASE_URL = "https://api.anthropic.com/v1"
_API_KEY_ENV = "ANTHROPIC_API_KEY"
_ANTHROPIC_VERSION = "2023-06-01"


class _Anthropic:
    """Callable provider factory for Anthropic."""

    def __call__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        client: Client | None = None,
    ) -> Model:
        return Model(
            id=model_id,
            adapter="anthropic",
            provider="anthropic",
            base_url=base_url or _BASE_URL,
            api_key_env=_API_KEY_ENV,
            client=client,
        )

    async def list(self, *, client: Client | None = None) -> list[str]:
        """List available model IDs from the Anthropic API."""
        from ..core import client as client_

        c = client or client_.Client(
            base_url=_BASE_URL,
            api_key=__import__("os").environ.get(_API_KEY_ENV),
        )
        headers = {
            "x-api-key": c.api_key or "",
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        response = await c.http.get(f"{c.base_url.rstrip('/')}/models", headers=headers)
        response.raise_for_status()
        data: list[dict[str, object]] = response.json().get("data", [])
        return sorted(str(m["id"]) for m in data)

    def __repr__(self) -> str:
        return "anthropic"


anthropic = _Anthropic()

__all__ = ["anthropic"]


def __getattr__(name: str) -> object:
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

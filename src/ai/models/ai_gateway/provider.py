"""AI Gateway provider singleton.

Defines the callable :data:`ai_gateway` provider, which satisfies the
:class:`~ai.models.core.proto.Provider` protocol.  The singleton is
re-exported from this package's ``__init__`` for convenience.
"""

from __future__ import annotations

import os
from typing import Any

from ..core import client as client_
from ..core.model import Model

_BASE_URL = "https://ai-gateway.vercel.sh/v3/ai"
_API_KEY_ENV = "AI_GATEWAY_API_KEY"


class _AIGateway:
    """Callable provider factory for the Vercel AI Gateway.

    Satisfies the :class:`~ai.models.core.proto.Provider` protocol.
    """

    @property
    def api_key_env(self) -> str:
        return _API_KEY_ENV

    @property
    def base_url(self) -> str:
        return _BASE_URL

    @property
    def adapter(self) -> str:
        return "ai-gateway-v3"

    @property
    def name(self) -> str:
        return "ai-gateway"

    def client(self) -> client_.Client:
        """Create a :class:`Client` from env-var credentials."""
        return client_.Client(
            base_url=_BASE_URL,
            api_key=os.environ.get(_API_KEY_ENV),
        )

    async def check(self, client: client_.Client, model: Model) -> bool:
        """Delegate to :func:`ai_gateway.check.check`."""
        from . import check as check_

        return await check_.check(client, model)

    def __call__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        client: client_.Client | None = None,
    ) -> Model:
        return Model(
            id=model_id,
            adapter=self.adapter,
            provider=self,
            client=client,
        )

    async def list(self, *, client: client_.Client | None = None) -> list[str]:
        """List available model IDs from the AI Gateway."""
        from . import sdk

        c = client or self.client()
        gateway = sdk.GatewayClient(c)
        response = await gateway.get("config")
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return sorted(str(m["id"]) for m in data.get("models", []))

    def __repr__(self) -> str:
        return "ai_gateway"


ai_gateway = _AIGateway()

__all__ = ["ai_gateway"]

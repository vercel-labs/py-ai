"""Model metadata types."""

from __future__ import annotations

import dataclasses

from .client import Client


@dataclasses.dataclass(frozen=True)
class Model:
    """Lightweight reference to a model on a specific provider.

    * ``id`` — identifier sent to the provider (e.g. ``"claude-sonnet-4-6"``).
    * ``adapter`` — wire protocol key (e.g. ``"ai-gateway-v3"``, ``"anthropic"``).
    * ``provider`` — hosting service (e.g. ``"ai-gateway"``, ``"anthropic"``).
    * ``base_url`` — API endpoint for auto-client creation.
    * ``api_key_env`` — env var name to read for auto-client creation.
    * ``client`` — explicit :class:`Client` override (skips auto-client).
    """

    id: str
    adapter: str
    provider: str
    base_url: str | None = None
    api_key_env: str | None = None
    client: Client | None = dataclasses.field(default=None, repr=False)

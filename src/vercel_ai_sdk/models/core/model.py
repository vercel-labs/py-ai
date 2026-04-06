"""Model metadata types."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelCost:
    """Per-million-token pricing."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclasses.dataclass(frozen=True)
class Model:
    """Pure-data description of a model.

    * ``id`` — identifier sent to the provider (e.g. ``"claude-sonnet-4-20250514"``).
    * ``adapter`` — adapter key (e.g. ``"ai-gateway-v3"``, ``"anthropic-messages"``).
    * ``provider`` — hosting service (e.g. ``"ai-gateway"``, ``"anthropic"``).
    """

    id: str
    adapter: str
    provider: str
    name: str = ""
    capabilities: tuple[str, ...] = ("text",)
    context_window: int = 0
    max_output_tokens: int = 0
    cost: ModelCost | None = None

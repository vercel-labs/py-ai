"""Anthropic direct-API model catalog.

Model IDs are what the Anthropic Messages API expects.
Pricing is per million tokens (USD).
"""

from __future__ import annotations

from ..core.model import Model, ModelCost

_ADAPTER = "anthropic"
_PROVIDER = "anthropic"

CATALOG: dict[str, Model] = {
    "claude-opus-4-6": Model(
        id="claude-opus-4-6",
        adapter=_ADAPTER,
        provider=_PROVIDER,
        name="Claude Opus 4.6",
        capabilities=("text",),
        context_window=1_000_000,
        max_output_tokens=128_000,
        cost=ModelCost(input=5.0, output=25.0, cache_read=0.50, cache_write=6.25),
    ),
    "claude-sonnet-4-6": Model(
        id="claude-sonnet-4-6",
        adapter=_ADAPTER,
        provider=_PROVIDER,
        name="Claude Sonnet 4.6",
        capabilities=("text",),
        context_window=1_000_000,
        max_output_tokens=64_000,
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.30, cache_write=3.75),
    ),
    "claude-haiku-4-5": Model(
        id="claude-haiku-4-5",
        adapter=_ADAPTER,
        provider=_PROVIDER,
        name="Claude Haiku 4.5",
        capabilities=("text",),
        context_window=200_000,
        max_output_tokens=64_000,
        cost=ModelCost(input=1.0, output=5.0, cache_read=0.10, cache_write=1.25),
    ),
    "claude-sonnet-4-20250514": Model(
        id="claude-sonnet-4-20250514",
        adapter=_ADAPTER,
        provider=_PROVIDER,
        name="Claude Sonnet 4",
        capabilities=("text",),
        context_window=200_000,
        max_output_tokens=64_000,
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.30, cache_write=3.75),
    ),
}

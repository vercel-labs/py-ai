"""AI Gateway model catalog.

Model IDs use the gateway's ``provider/model-name`` format.
Pricing is per million tokens (USD).
"""

from __future__ import annotations

from ..core.model import Model, ModelCost

_ADAPTER = "ai-gateway-v3"
_PROVIDER = "ai-gateway"


def _text(
    id: str,
    name: str,
    *,
    context_window: int,
    max_output_tokens: int,
    cost: ModelCost,
) -> Model:
    return Model(
        id=id,
        adapter=_ADAPTER,
        provider=_PROVIDER,
        name=name,
        capabilities=("text",),
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        cost=cost,
    )


def _media(
    id: str,
    name: str,
    *,
    capabilities: tuple[str, ...],
    context_window: int = 0,
    max_output_tokens: int = 0,
    cost: ModelCost | None = None,
) -> Model:
    return Model(
        id=id,
        adapter=_ADAPTER,
        provider=_PROVIDER,
        name=name,
        capabilities=capabilities,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        cost=cost,
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

CATALOG: dict[str, Model] = {
    # -- Anthropic (via gateway) -------------------------------------------
    "anthropic/claude-opus-4-6": _text(
        "anthropic/claude-opus-4-6",
        "Claude Opus 4.6",
        context_window=1_000_000,
        max_output_tokens=128_000,
        cost=ModelCost(input=5.0, output=25.0, cache_read=0.50, cache_write=6.25),
    ),
    "anthropic/claude-sonnet-4-6": _text(
        "anthropic/claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        context_window=1_000_000,
        max_output_tokens=64_000,
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.30, cache_write=3.75),
    ),
    "anthropic/claude-haiku-4-5": _text(
        "anthropic/claude-haiku-4-5",
        "Claude Haiku 4.5",
        context_window=200_000,
        max_output_tokens=64_000,
        cost=ModelCost(input=1.0, output=5.0, cache_read=0.10, cache_write=1.25),
    ),
    "anthropic/claude-sonnet-4": _text(
        "anthropic/claude-sonnet-4",
        "Claude Sonnet 4",
        context_window=200_000,
        max_output_tokens=64_000,
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.30, cache_write=3.75),
    ),
    # -- OpenAI (via gateway) ----------------------------------------------
    "openai/gpt-5.4": _text(
        "openai/gpt-5.4",
        "GPT-5.4",
        context_window=1_000_000,
        max_output_tokens=128_000,
        cost=ModelCost(input=2.50, output=15.0, cache_read=0.25),
    ),
    "openai/gpt-5.4-mini": _text(
        "openai/gpt-5.4-mini",
        "GPT-5.4 Mini",
        context_window=400_000,
        max_output_tokens=128_000,
        cost=ModelCost(input=0.75, output=4.50, cache_read=0.075),
    ),
    "openai/gpt-5.4-nano": _text(
        "openai/gpt-5.4-nano",
        "GPT-5.4 Nano",
        context_window=400_000,
        max_output_tokens=128_000,
        cost=ModelCost(input=0.20, output=1.25, cache_read=0.02),
    ),
    # -- Google (via gateway) ----------------------------------------------
    "google/gemini-2.5-pro": _text(
        "google/gemini-2.5-pro",
        "Gemini 2.5 Pro",
        context_window=1_000_000,
        max_output_tokens=65_536,
        cost=ModelCost(input=1.25, output=10.0, cache_read=0.315),
    ),
    "google/gemini-2.5-flash": _text(
        "google/gemini-2.5-flash",
        "Gemini 2.5 Flash",
        context_window=1_000_000,
        max_output_tokens=65_536,
        cost=ModelCost(input=0.15, output=0.60, cache_read=0.0375),
    ),
    # -- Image / video models (via gateway) --------------------------------
    "google/gemini-3-pro-image": _media(
        "google/gemini-3-pro-image",
        "Gemini 3 Pro Image",
        capabilities=("text", "image"),
        context_window=1_000_000,
        max_output_tokens=65_536,
    ),
    "google/imagen-4.0-generate-001": _media(
        "google/imagen-4.0-generate-001",
        "Imagen 4.0",
        capabilities=("image",),
    ),
    "openai/gpt-image-1": _media(
        "openai/gpt-image-1",
        "GPT Image 1",
        capabilities=("image",),
    ),
    "google/veo-3.0-generate-001": _media(
        "google/veo-3.0-generate-001",
        "Veo 3.0",
        capabilities=("video",),
    ),
}

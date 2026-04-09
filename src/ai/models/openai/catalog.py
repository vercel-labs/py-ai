"""OpenAI direct-API model catalog.

Model IDs are what the OpenAI Chat Completions API expects.
Pricing is per million tokens (USD).
"""

from __future__ import annotations

from ..core.model import Model, ModelCost

_ADAPTER = "openai"
_PROVIDER = "openai"

CATALOG: dict[str, Model] = {
    "gpt-5.4": Model(
        id="gpt-5.4",
        adapter=_ADAPTER,
        provider=_PROVIDER,
        name="GPT-5.4",
        capabilities=("text",),
        context_window=1_000_000,
        max_output_tokens=128_000,
        cost=ModelCost(input=2.50, output=15.0, cache_read=0.25),
    ),
    "gpt-5.4-mini": Model(
        id="gpt-5.4-mini",
        adapter=_ADAPTER,
        provider=_PROVIDER,
        name="GPT-5.4 Mini",
        capabilities=("text",),
        context_window=400_000,
        max_output_tokens=128_000,
        cost=ModelCost(input=0.75, output=4.50, cache_read=0.075),
    ),
    "gpt-5.4-nano": Model(
        id="gpt-5.4-nano",
        adapter=_ADAPTER,
        provider=_PROVIDER,
        name="GPT-5.4 Nano",
        capabilities=("text",),
        context_window=400_000,
        max_output_tokens=128_000,
        cost=ModelCost(input=0.20, output=1.25, cache_read=0.02),
    ),
}

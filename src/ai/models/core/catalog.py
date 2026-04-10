"""Model catalog — per-provider registry of known models.

Usage::

    from ai.models.core.catalog import model

    opus = model("anthropic", "claude-opus-4-6")
    sonnet_gw = model("ai-gateway", "anthropic/claude-sonnet-4-6")
"""

from __future__ import annotations

from .model import Model

# ---------------------------------------------------------------------------
# Global registry: provider name -> {model_id -> Model}
# ---------------------------------------------------------------------------

_catalogs: dict[str, dict[str, Model]] = {}


def register_catalog(provider: str, models: dict[str, Model]) -> None:
    """Register a provider's model catalog.

    Called by each provider's ``catalog`` module during initialisation,
    and by users who want to add custom providers.
    """
    _catalogs[provider] = models


def _load_builtin_catalogs() -> None:
    """Import and register all built-in provider catalogs.

    Catalog submodules only depend on :mod:`..core.model` (pure
    dataclasses) — no ``httpx``, ``anthropic``, or ``openai`` imports.
    This makes it safe to run eagerly at import time, including inside
    sandboxed runtimes (e.g. Temporal workflow workers).
    """
    from ..ai_gateway.catalog import CATALOG as ai_gw_catalog
    from ..anthropic.catalog import CATALOG as anthropic_catalog
    from ..openai.catalog import CATALOG as openai_catalog

    register_catalog("ai-gateway", ai_gw_catalog)
    register_catalog("anthropic", anthropic_catalog)
    register_catalog("openai", openai_catalog)


_load_builtin_catalogs()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def model(provider: str, model_id: str) -> Model:
    """Look up a model by provider and model ID.

    Raises :class:`KeyError` with a helpful message if the provider or
    model ID is not found.

    Examples::

        model("ai-gateway", "anthropic/claude-opus-4-6")
        model("anthropic", "claude-opus-4-6")
        model("openai", "gpt-5.4")
    """
    provider_catalog = _catalogs.get(provider)
    if provider_catalog is None:
        available = ", ".join(sorted(_catalogs)) or "(none)"
        raise KeyError(
            f"Unknown provider {provider!r}. Registered providers: {available}"
        )

    entry = provider_catalog.get(model_id)
    if entry is None:
        available = ", ".join(sorted(provider_catalog)) or "(none)"
        raise KeyError(
            f"Unknown model {model_id!r} for provider {provider!r}. "
            f"Available models: {available}"
        )

    return entry


def get_providers() -> list[str]:
    """Return all registered provider names."""
    return sorted(_catalogs)


def get_models(provider: str) -> dict[str, Model]:
    """Return all models for a provider.

    Raises :class:`KeyError` if the provider is not registered.
    """
    provider_catalog = _catalogs.get(provider)
    if provider_catalog is None:
        available = ", ".join(sorted(_catalogs)) or "(none)"
        raise KeyError(
            f"Unknown provider {provider!r}. Registered providers: {available}"
        )

    return dict(provider_catalog)

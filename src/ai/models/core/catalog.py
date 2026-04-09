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
_catalogs_loaded = False


def register_catalog(provider: str, models: dict[str, Model]) -> None:
    """Register a provider's model catalog.

    Called by each provider's ``catalog`` module during lazy initialisation.
    """
    _catalogs[provider] = models


def _ensure_catalogs() -> None:
    """Lazily import and register all built-in provider catalogs."""
    global _catalogs_loaded  # noqa: PLW0603
    if _catalogs_loaded:
        return
    _catalogs_loaded = True

    from ..ai_gateway import catalog as ai_gw_catalog
    from ..anthropic import catalog as anthropic_catalog
    from ..openai import catalog as openai_catalog

    register_catalog("ai-gateway", ai_gw_catalog.CATALOG)
    register_catalog("anthropic", anthropic_catalog.CATALOG)
    register_catalog("openai", openai_catalog.CATALOG)


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
    _ensure_catalogs()

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
    _ensure_catalogs()
    return sorted(_catalogs)


def get_models(provider: str) -> dict[str, Model]:
    """Return all models for a provider.

    Raises :class:`KeyError` if the provider is not registered.
    """
    _ensure_catalogs()

    provider_catalog = _catalogs.get(provider)
    if provider_catalog is None:
        available = ", ".join(sorted(_catalogs)) or "(none)"
        raise KeyError(
            f"Unknown provider {provider!r}. Registered providers: {available}"
        )

    return dict(provider_catalog)

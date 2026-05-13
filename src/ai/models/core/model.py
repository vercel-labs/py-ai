"""Model metadata types."""

import dataclasses

from ... import _modelsdev
from ...providers import base
from .client import Client


@dataclasses.dataclass(frozen=True)
class Model:
    """Lightweight reference to a model on a specific provider.

    * ``id`` — identifier sent to the provider (e.g. ``"claude-sonnet-4-6"``).
    * ``adapter`` — wire protocol key (e.g. ``"ai-gateway-v3"``, ``"anthropic"``).
    * ``provider`` — :class:`Provider` that owns this model.
    * ``client`` — explicit :class:`Client` override (skips provider's default).
    """

    id: str
    adapter: str
    provider: base.Provider
    client: Client | None = dataclasses.field(default=None, repr=False)


def get_model(model_id: str) -> Model:
    """Resolve a provider-qualified model ID into a :class:`Model`.

    Args:
        model_id:
            Model ID in the format of `provider:model`.
            Example: ``"openai:gpt-5"``.

    Raises:
        Raises a :class:`ai.UnsupportedProviderError` when the provider is
        unrecognized or otherwise unsupported.
    """
    ref = _modelsdev.parse_model_id(model_id)
    if ref.provider_id is None:
        raise ValueError("model_id must include a known provider id")
    model_info = _modelsdev.get_model_by_id(f"{ref.provider_id}:{ref.model_id}")
    model_provider_config = None if model_info is None else model_info.provider_config
    return base.Provider.from_id(
        ref.provider_id,
        model_provider_config=model_provider_config,
    )(ref.model_id)

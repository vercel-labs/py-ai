"""Model metadata types."""

import dataclasses

import pydantic

from .client import Client
from .proto import Provider


@dataclasses.dataclass(frozen=True)
class Model[ProviderParamsT: pydantic.BaseModel]:
    """Lightweight reference to a model on a specific provider.

    * ``id`` — identifier sent to the provider (e.g. ``"claude-sonnet-4-6"``).
    * ``adapter`` — wire protocol key (e.g. ``"ai-gateway-v3"``, ``"anthropic"``).
    * ``provider`` — :class:`Provider` that owns this model.
    * ``client`` — explicit :class:`Client` override (skips provider's default).
    """

    id: str
    adapter: str
    provider: Provider[ProviderParamsT]
    client: Client | None = dataclasses.field(default=None, repr=False)

"""Model — pure data describing a model, no execution logic."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Model:
    """Immutable description of a model.

    ``id``
        The model identifier sent to the provider
        (e.g. ``"claude-sonnet-4-20250514"``, ``"gpt-4o"``).

    ``api``
        Wire protocol discriminator used to look up the execution function
        (e.g. ``"anthropic"``, ``"openai"``, ``"ai-gateway"``).
        A single ``api`` value may be shared by multiple providers that speak
        the same wire format.

    ``provider``
        The actual host / provider name
        (e.g. ``"anthropic"``, ``"azure"``, ``"ai-gateway"``).
    """

    id: str
    api: str
    provider: str

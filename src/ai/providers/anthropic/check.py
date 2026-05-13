"""Anthropic connection check.

Verifies that the provider credentials are valid and that the model
exists on the Anthropic API by hitting ``GET {base_url}/v1/models/{model_id}``.

The ``anthropic-version`` header is added by the check function itself.

This endpoint is free — no tokens or credits are consumed.
"""

import anthropic

from ...models import core
from . import provider as provider_

_ANTHROPIC_VERSION = "2023-06-01"

# HTTP status codes that indicate bad auth or a missing model.
_FAIL_STATUSES = frozenset({401, 403, 404})


async def check(model: core.model.Model) -> bool:
    """Return ``True`` if *client* can reach Anthropic and *model* exists."""
    provider = model.provider
    if isinstance(provider, provider_.AnthropicCompatibleProvider):
        client = provider.sdk_client
        if client is not None:
            try:
                await client.models.retrieve(model.id)
            except anthropic.APIStatusError as exc:
                if exc.status_code in _FAIL_STATUSES:
                    return False
                raise
            return True

    if not provider.is_configured():
        return False
    api_key = provider.api_key
    if not api_key:
        return False
    url = f"{provider.base_url.rstrip('/')}/v1/models/{model.id}"
    anthropic_version = getattr(model.provider, "anthropic_version", _ANTHROPIC_VERSION)
    headers = {
        "x-api-key": api_key,
        "anthropic-version": anthropic_version,
    }
    response = await provider.http.get(url, headers=headers)
    if response.status_code == 200:
        return True
    if response.status_code in _FAIL_STATUSES:
        return False
    # Unexpected status — let the caller handle it.
    response.raise_for_status()
    return False  # pragma: no cover

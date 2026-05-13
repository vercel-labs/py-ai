"""OpenAI connection check.

Verifies that the client's credentials are valid and that the model
exists on the OpenAI API by hitting ``GET /v1/models/{model_id}``.

This endpoint is free — no tokens or credits are consumed.
"""

import openai

from ...models import core
from . import provider as provider_

# HTTP status codes that indicate bad auth or a missing model.
_FAIL_STATUSES = frozenset({401, 403, 404})


async def check(model: core.model.Model) -> bool:
    """Return ``True`` if *client* can reach OpenAI and *model* exists."""
    provider = model.provider
    if isinstance(provider, provider_.OpenAICompatibleProvider):
        client = provider.sdk_client
        if client is not None:
            try:
                await client.models.retrieve(model.id)
            except openai.APIStatusError as exc:
                if exc.status_code in _FAIL_STATUSES:
                    return False
                raise
            return True

    if not provider.api_key:
        return False
    url = f"{provider.base_url.rstrip('/')}/models/{model.id}"
    headers = {"Authorization": f"Bearer {provider.api_key}"}
    response = await provider.http.get(url, headers=headers)
    if response.status_code == 200:
        return True
    if response.status_code in _FAIL_STATUSES:
        return False
    # Unexpected status — let the caller handle it.
    response.raise_for_status()
    return False  # pragma: no cover

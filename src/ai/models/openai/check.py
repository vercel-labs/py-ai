"""OpenAI connection check.

Verifies that the client's credentials are valid and that the model
exists on the OpenAI API by hitting ``GET /v1/models/{model_id}``.

This endpoint is free — no tokens or credits are consumed.
"""

from __future__ import annotations

from ..core import client as client_
from ..core import model as model_

# HTTP status codes that indicate bad auth or a missing model.
_FAIL_STATUSES = frozenset({401, 403, 404})


async def check(client: client_.Client, model: model_.Model) -> bool:
    """Return ``True`` if *client* can reach OpenAI and *model* exists."""
    if not client.api_key:
        return False
    url = f"{client.base_url.rstrip('/')}/models/{model.id}"
    headers = {"Authorization": f"Bearer {client.api_key}"}
    response = await client.http.get(url, headers=headers)
    if response.status_code == 200:
        return True
    if response.status_code in _FAIL_STATUSES:
        return False
    # Unexpected status — let the caller handle it.
    response.raise_for_status()
    return False  # pragma: no cover

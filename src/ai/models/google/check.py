"""Google Gemini connection check.

Verifies that the client's credentials are valid and that the model exists on
the Gemini Developer API by hitting ``GET {base_url}/models/{model_id}``.

This endpoint is free; no tokens or credits are consumed.
"""

from .. import core

# Google returns 400 for some bad-key / malformed-model cases.
_FAIL_STATUSES = frozenset({400, 401, 403, 404})


async def check(client: core.client.Client, model: core.model.Model) -> bool:
    """Return ``True`` if *client* can reach Google and *model* exists."""
    if not client.api_key:
        return False
    model_name = model.id if model.id.startswith("models/") else f"models/{model.id}"
    url = f"{client.base_url.rstrip('/')}/{model_name}"
    headers = {"x-goog-api-key": client.api_key}
    response = await client.http.get(url, headers=headers)
    if response.status_code == 200:
        return True
    if response.status_code in _FAIL_STATUSES:
        return False
    # Unexpected status - let the caller handle it.
    response.raise_for_status()
    return False  # pragma: no cover

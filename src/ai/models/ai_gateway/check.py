"""AI Gateway connection check.

Verifies **both** that the client's credentials are valid and that the
model exists in the gateway's catalogue.

* Auth is validated via ``GET {origin}/v1/credits`` which requires a
  valid API key (returns 401/403 otherwise).
* Model existence is validated via ``GET {baseURL}/config`` which is a
  public endpoint returning ``{"models": [{"id": "...", ...}, ...]}``.

Both endpoints are free — no tokens or credits are consumed.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from ..core import client as client_
from ..core import model as model_

_PROTOCOL_VERSION = "0.0.1"

# HTTP status codes that indicate bad auth.
_FAIL_STATUSES = frozenset({401, 403})


def _origin(base_url: str) -> str:
    """Extract the origin (scheme + host + port) from *base_url*."""
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


async def check(client: client_.Client, model: model_.Model) -> bool:
    """Return ``True`` if *client* can reach the gateway and *model* exists."""
    base_url = client.base_url.rstrip("/")
    headers: dict[str, str] = {
        "ai-gateway-protocol-version": _PROTOCOL_VERSION,
    }
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"
        headers["ai-gateway-auth-method"] = "api-key"

    # 1. Verify credentials via /v1/credits (requires valid auth).
    credits_url = f"{_origin(base_url)}/v1/credits"
    auth_resp = await client.http.get(credits_url, headers=headers)
    if auth_resp.status_code in _FAIL_STATUSES:
        return False
    if auth_resp.status_code != 200:
        auth_resp.raise_for_status()

    # 2. Verify model existence via /config (public catalogue).
    config_url = f"{base_url}/config"
    config_resp = await client.http.get(config_url, headers=headers)
    if config_resp.status_code != 200:
        config_resp.raise_for_status()
        return False  # pragma: no cover

    data: dict[str, Any] = config_resp.json()
    remote_ids: set[str] = {m["id"] for m in data.get("models", [])}
    return model.id in remote_ids

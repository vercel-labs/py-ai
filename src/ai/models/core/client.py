"""HTTP client for adapter functions."""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

    from . import model as model_


@dataclasses.dataclass
class Client:
    """Connection parameters for a provider API.

    Adapter functions receive a ``Client`` instead of creating their own HTTP
    session.  This keeps auth and base URL decoupled from the adapter logic.

    The :pyattr:`http` property lazily creates a shared
    :class:`httpx.AsyncClient` so that consecutive calls reuse the same
    connection pool.
    """

    base_url: str
    api_key: str | None = None
    headers: dict[str, str] = dataclasses.field(default_factory=dict)

    _http: Any = dataclasses.field(default=None, repr=False, compare=False)

    @property
    def http(self) -> httpx.AsyncClient:
        """Lazy-init shared httpx client."""
        import httpx as _httpx

        if self._http is None or self._http.is_closed:
            self._http = _httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=_httpx.Timeout(timeout=300.0, connect=10.0),
            )
        return self._http  # type: ignore[no-any-return]

    async def aclose(self) -> None:
        """Close the underlying HTTP client if open."""
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
            self._http = None


def auto_client(model: model_.Model) -> Client:
    """Create a :class:`Client` from the model's connection info.

    Uses ``model.client`` if set, otherwise builds one from
    ``model.base_url`` and the env var named by ``model.api_key_env``.
    """
    if model.client is not None:
        return model.client

    if model.base_url is None:
        raise ValueError(
            f"Model {model.id!r} (provider={model.provider!r}) has no "
            f"base_url and no explicit client. Pass a client= to the "
            f"provider factory or set base_url."
        )

    api_key: str | None = None
    if model.api_key_env:
        api_key = os.environ.get(model.api_key_env)

    return Client(base_url=model.base_url, api_key=api_key)

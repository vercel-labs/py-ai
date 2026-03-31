"""HTTP client for wire functions."""

from __future__ import annotations

import dataclasses

import httpx


@dataclasses.dataclass
class Client:
    """Connection parameters for a provider API.

    Wire functions receive a ``Client`` instead of creating their own HTTP
    session.  This keeps auth and base URL decoupled from the wire logic.

    The :pyattr:`http` property lazily creates a shared
    :class:`httpx.AsyncClient` so that consecutive calls reuse the same
    connection pool.
    """

    base_url: str
    api_key: str | None = None
    headers: dict[str, str] = dataclasses.field(default_factory=dict)

    _http: httpx.AsyncClient | None = dataclasses.field(
        default=None, repr=False, compare=False
    )

    @property
    def http(self) -> httpx.AsyncClient:
        """Lazy-init shared httpx client."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=httpx.Timeout(timeout=300.0, connect=10.0),
            )
        return self._http

    async def aclose(self) -> None:
        """Close the underlying HTTP client if open."""
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
            self._http = None

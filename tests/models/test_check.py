"""Tests for the connection-check layer (``check_connection``).

Each provider's check function is tested with mocked httpx responses.
OpenAI and Anthropic share the same status-code logic, so we test one
provider fully and use parametrize for the other.  The AI Gateway has
unique two-endpoint routing that needs dedicated tests.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import httpx
import pytest

from ai.models import anthropic, check_connection, openai
from ai.models.ai_gateway import ai_gateway
from ai.models.ai_gateway import check as ai_gw_check
from ai.models.anthropic import check as anthropic_check
from ai.models.core import client as client_
from ai.models.core import model as model_
from ai.models.openai import check as openai_check

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_OPENAI_MODEL = openai("gpt-5.4")
_ANTHROPIC_MODEL = anthropic("claude-opus-4-6")
_GATEWAY_MODEL = ai_gateway("anthropic/claude-opus-4-6")


class _FailProvider:
    """A provider whose check always raises KeyError (for testing unknown providers)."""

    @property
    def api_key_env(self) -> None:
        return None

    @property
    def base_url(self) -> str:
        return "http://unknown.test"

    @property
    def adapter(self) -> str:
        return "x"

    @property
    def name(self) -> str:
        return "unknown-provider"

    def client(self) -> client_.Client:
        return client_.Client(base_url=self.base_url)

    async def check(self, client: client_.Client, model: model_.Model) -> bool:
        raise KeyError(f"No check function registered for provider={self.name!r}.")

    async def list(self, *, client: client_.Client | None = None) -> list[str]:
        return []

    def __call__(
        self, model_id: str, *, client: client_.Client | None = None
    ) -> model_.Model:
        return model_.Model(
            id=model_id, adapter=self.adapter, provider=self, client=client
        )


_UNKNOWN_MODEL = _FailProvider()("x")


def _client_with_mock(
    status_code: int = 200,
    json_body: Any = None,
    base_url: str = "https://test.example.com",
) -> client_.Client:
    """Return a ``Client`` whose httpx session uses a mock transport."""

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps(json_body or {}).encode()
        return httpx.Response(status_code, content=body)

    c = client_.Client(base_url=base_url, api_key="sk-test-key")
    c._http = httpx.AsyncClient(
        base_url=base_url,
        transport=httpx.MockTransport(_handler),
    )
    return c


# ===================================================================
# OpenAI check (full coverage -- Anthropic shares the same logic)
# ===================================================================


class TestOpenAICheck:
    async def test_200_returns_true(self) -> None:
        c = _client_with_mock(200, {"id": "gpt-5.4", "object": "model"})
        assert await openai_check.check(c, _OPENAI_MODEL) is True

    @pytest.mark.parametrize("status", [401, 403, 404])
    async def test_client_error_returns_false(self, status: int) -> None:
        c = _client_with_mock(status)
        assert await openai_check.check(c, _OPENAI_MODEL) is False

    async def test_500_raises(self) -> None:
        c = _client_with_mock(500)
        with pytest.raises(httpx.HTTPStatusError):
            await openai_check.check(c, _OPENAI_MODEL)

    async def test_no_api_key_returns_false(self) -> None:
        c = client_.Client(base_url="https://test.example.com", api_key=None)
        assert await openai_check.check(c, _OPENAI_MODEL) is False


# ===================================================================
# Anthropic check (smoke test -- same status-code logic as OpenAI)
# ===================================================================


class TestAnthropicCheck:
    async def test_200_returns_true(self) -> None:
        c = _client_with_mock(200, {"id": "claude-opus-4-6", "type": "model"})
        assert await anthropic_check.check(c, _ANTHROPIC_MODEL) is True

    async def test_shared_status_logic_smoke(self) -> None:
        assert (
            await anthropic_check.check(_client_with_mock(401), _ANTHROPIC_MODEL)
            is False
        )
        with pytest.raises(httpx.HTTPStatusError):
            await anthropic_check.check(_client_with_mock(500), _ANTHROPIC_MODEL)


# ===================================================================
# AI Gateway check (unique two-endpoint routing)
# ===================================================================


def _gateway_client(
    *,
    credits_status: int = 200,
    config_status: int = 200,
    config_body: dict[str, Any] | None = None,
    api_key: str | None = "sk-test-key",
) -> client_.Client:
    credits_body = json.dumps({"balance": "10.00", "totalUsed": "5.00"})
    config_bytes = json.dumps(config_body or {"models": []}).encode()

    def _handler(request: httpx.Request) -> httpx.Response:
        if "/v1/credits" in str(request.url):
            return httpx.Response(credits_status, content=credits_body.encode())
        return httpx.Response(config_status, content=config_bytes)

    base_url = "https://test.example.com/v3/ai"
    c = client_.Client(base_url=base_url, api_key=api_key)
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    return c


class TestAIGatewayCheck:
    async def test_auth_ok_model_present(self) -> None:
        config = {"models": [{"id": "anthropic/claude-opus-4-6"}]}
        c = _gateway_client(config_body=config)
        assert await ai_gw_check.check(c, _GATEWAY_MODEL) is True

    async def test_auth_ok_model_absent(self) -> None:
        config = {"models": [{"id": "openai/gpt-5.4"}]}
        c = _gateway_client(config_body=config)
        assert await ai_gw_check.check(c, _GATEWAY_MODEL) is False

    @pytest.mark.parametrize("status", [401, 403])
    async def test_credits_auth_error_returns_false(self, status: int) -> None:
        c = _gateway_client(credits_status=status)
        assert await ai_gw_check.check(c, _GATEWAY_MODEL) is False

    async def test_credits_500_raises(self) -> None:
        c = _gateway_client(credits_status=500)
        with pytest.raises(httpx.HTTPStatusError):
            await ai_gw_check.check(c, _GATEWAY_MODEL)


# ===================================================================
# Top-level check_connection()
# ===================================================================


class TestCheckConnection:
    async def test_gateway_dispatches(self) -> None:
        config = {"models": [{"id": "anthropic/claude-opus-4-6"}]}
        c = _gateway_client(config_body=config)
        m = dataclasses.replace(_GATEWAY_MODEL, client=c)
        assert await check_connection(m) is True

    async def test_unknown_provider_raises(self) -> None:
        c = _client_with_mock(200)
        m = dataclasses.replace(_UNKNOWN_MODEL, client=c)
        with pytest.raises(KeyError, match="unknown-provider"):
            await check_connection(m)

    async def test_dispatch_false_propagates(self) -> None:
        c = _client_with_mock(401)
        m = dataclasses.replace(_OPENAI_MODEL, client=c)
        assert await check_connection(m) is False

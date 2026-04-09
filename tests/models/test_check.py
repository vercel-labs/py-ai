"""Tests for the connection-check layer (``check_connection``).

Each provider's check function is tested with mocked httpx responses so
that no real network calls are made.  The tests verify:

- 200 → ``True``
- 401/403/404 → ``False``
- 5xx → exception propagates
- AI Gateway: model presence vs absence in the config response
- Top-level ``check_connection`` dispatches correctly and raises on
  unknown providers.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ai.models import check_connection
from ai.models.ai_gateway import check as ai_gw_check
from ai.models.anthropic import check as anthropic_check
from ai.models.core.client import Client
from ai.models.core.model import Model
from ai.models.openai import check as openai_check

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_OPENAI_MODEL = Model(id="gpt-5.4", adapter="openai", provider="openai")
_ANTHROPIC_MODEL = Model(
    id="claude-opus-4-6", adapter="anthropic", provider="anthropic"
)
_GATEWAY_MODEL = Model(
    id="anthropic/claude-opus-4-6", adapter="ai-gateway-v3", provider="ai-gateway"
)
_UNKNOWN_MODEL = Model(id="x", adapter="x", provider="unknown-provider")


def _client(base_url: str = "https://test.example.com") -> Client:
    return Client(base_url=base_url, api_key="sk-test-key")


def _mock_transport(
    status_code: int = 200,
    json_body: Any = None,
) -> httpx.MockTransport:
    """Return an httpx MockTransport that always returns *status_code*."""

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps(json_body or {}).encode()
        return httpx.Response(status_code, content=body)

    return httpx.MockTransport(_handler)


def _client_with_mock(
    status_code: int = 200,
    json_body: Any = None,
    base_url: str = "https://test.example.com",
) -> Client:
    """Return a ``Client`` whose httpx session uses a mock transport."""
    c = Client(base_url=base_url, api_key="sk-test-key")
    c._http = httpx.AsyncClient(
        base_url=base_url,
        transport=_mock_transport(status_code, json_body),
    )
    return c


# ===================================================================
# OpenAI check
# ===================================================================


class TestOpenAICheck:
    @pytest.mark.asyncio
    async def test_200_returns_true(self) -> None:
        c = _client_with_mock(200, {"id": "gpt-5.4", "object": "model"})
        assert await openai_check.check(c, _OPENAI_MODEL) is True

    @pytest.mark.asyncio
    async def test_401_returns_false(self) -> None:
        c = _client_with_mock(401)
        assert await openai_check.check(c, _OPENAI_MODEL) is False

    @pytest.mark.asyncio
    async def test_403_returns_false(self) -> None:
        c = _client_with_mock(403)
        assert await openai_check.check(c, _OPENAI_MODEL) is False

    @pytest.mark.asyncio
    async def test_404_returns_false(self) -> None:
        c = _client_with_mock(404)
        assert await openai_check.check(c, _OPENAI_MODEL) is False

    @pytest.mark.asyncio
    async def test_500_raises(self) -> None:
        c = _client_with_mock(500)
        with pytest.raises(httpx.HTTPStatusError):
            await openai_check.check(c, _OPENAI_MODEL)

    @pytest.mark.asyncio
    async def test_no_api_key_returns_false(self) -> None:
        c = Client(base_url="https://test.example.com", api_key=None)
        assert await openai_check.check(c, _OPENAI_MODEL) is False


# ===================================================================
# Anthropic check
# ===================================================================


class TestAnthropicCheck:
    @pytest.mark.asyncio
    async def test_200_returns_true(self) -> None:
        c = _client_with_mock(200, {"id": "claude-opus-4-6", "type": "model"})
        assert await anthropic_check.check(c, _ANTHROPIC_MODEL) is True

    @pytest.mark.asyncio
    async def test_401_returns_false(self) -> None:
        c = _client_with_mock(401)
        assert await anthropic_check.check(c, _ANTHROPIC_MODEL) is False

    @pytest.mark.asyncio
    async def test_403_returns_false(self) -> None:
        c = _client_with_mock(403)
        assert await anthropic_check.check(c, _ANTHROPIC_MODEL) is False

    @pytest.mark.asyncio
    async def test_404_returns_false(self) -> None:
        c = _client_with_mock(404)
        assert await anthropic_check.check(c, _ANTHROPIC_MODEL) is False

    @pytest.mark.asyncio
    async def test_500_raises(self) -> None:
        c = _client_with_mock(500)
        with pytest.raises(httpx.HTTPStatusError):
            await anthropic_check.check(c, _ANTHROPIC_MODEL)

    @pytest.mark.asyncio
    async def test_no_api_key_returns_false(self) -> None:
        c = Client(base_url="https://test.example.com", api_key=None)
        assert await anthropic_check.check(c, _ANTHROPIC_MODEL) is False


# ===================================================================
# AI Gateway check
# ===================================================================


def _gateway_transport(
    *,
    credits_status: int = 200,
    config_status: int = 200,
    config_body: dict[str, Any] | None = None,
) -> httpx.MockTransport:
    """Mock transport that routes /v1/credits and /config separately."""
    credits_body = json.dumps({"balance": "10.00", "totalUsed": "5.00"})
    config_bytes = json.dumps(config_body or {"models": []}).encode()

    def _handler(request: httpx.Request) -> httpx.Response:
        if "/v1/credits" in str(request.url):
            return httpx.Response(credits_status, content=credits_body.encode())
        return httpx.Response(config_status, content=config_bytes)

    return httpx.MockTransport(_handler)


def _gateway_client(
    *,
    credits_status: int = 200,
    config_status: int = 200,
    config_body: dict[str, Any] | None = None,
    api_key: str | None = "sk-test-key",
) -> Client:
    base_url = "https://test.example.com/v3/ai"
    c = Client(base_url=base_url, api_key=api_key)
    c._http = httpx.AsyncClient(
        transport=_gateway_transport(
            credits_status=credits_status,
            config_status=config_status,
            config_body=config_body,
        ),
    )
    return c


class TestAIGatewayCheck:
    @pytest.mark.asyncio
    async def test_auth_ok_model_present(self) -> None:
        config = {"models": [{"id": "anthropic/claude-opus-4-6"}]}
        c = _gateway_client(config_body=config)
        assert await ai_gw_check.check(c, _GATEWAY_MODEL) is True

    @pytest.mark.asyncio
    async def test_auth_ok_model_absent(self) -> None:
        config = {"models": [{"id": "openai/gpt-5.4"}]}
        c = _gateway_client(config_body=config)
        assert await ai_gw_check.check(c, _GATEWAY_MODEL) is False

    @pytest.mark.asyncio
    async def test_auth_ok_empty_list(self) -> None:
        c = _gateway_client(config_body={"models": []})
        assert await ai_gw_check.check(c, _GATEWAY_MODEL) is False

    @pytest.mark.asyncio
    async def test_credits_401_returns_false(self) -> None:
        c = _gateway_client(credits_status=401)
        assert await ai_gw_check.check(c, _GATEWAY_MODEL) is False

    @pytest.mark.asyncio
    async def test_credits_403_returns_false(self) -> None:
        c = _gateway_client(credits_status=403)
        assert await ai_gw_check.check(c, _GATEWAY_MODEL) is False

    @pytest.mark.asyncio
    async def test_credits_500_raises(self) -> None:
        c = _gateway_client(credits_status=500)
        with pytest.raises(httpx.HTTPStatusError):
            await ai_gw_check.check(c, _GATEWAY_MODEL)

    @pytest.mark.asyncio
    async def test_no_api_key_fails_auth(self) -> None:
        """No key means /v1/credits returns 401 → False."""
        c = _gateway_client(credits_status=401, api_key=None)
        assert await ai_gw_check.check(c, _GATEWAY_MODEL) is False


# ===================================================================
# Top-level check_connection()
# ===================================================================


class TestCheckConnection:
    @pytest.mark.asyncio
    async def test_openai_dispatches(self) -> None:
        body = {"id": "gpt-5.4", "object": "model"}
        c = _client_with_mock(200, body)
        assert await check_connection(_OPENAI_MODEL, client=c) is True

    @pytest.mark.asyncio
    async def test_anthropic_dispatches(self) -> None:
        body = {"id": "claude-opus-4-6", "type": "model"}
        c = _client_with_mock(200, body)
        assert await check_connection(_ANTHROPIC_MODEL, client=c) is True

    @pytest.mark.asyncio
    async def test_gateway_dispatches(self) -> None:
        config = {"models": [{"id": "anthropic/claude-opus-4-6"}]}
        c = _gateway_client(config_body=config)
        assert await check_connection(_GATEWAY_MODEL, client=c) is True

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self) -> None:
        c = _client_with_mock(200)
        with pytest.raises(KeyError, match="unknown-provider"):
            await check_connection(_UNKNOWN_MODEL, client=c)

    @pytest.mark.asyncio
    async def test_false_propagates(self) -> None:
        c = _client_with_mock(401)
        assert await check_connection(_OPENAI_MODEL, client=c) is False

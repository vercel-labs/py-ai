"""Tests for the AI Gateway built-in tool surface.

The gateway adapter must:

1. Accept ``ai_gateway.tools.*`` instances and serialize them as v3
   ``provider`` blocks with the registered ``gateway_id``.
2. Reject native provider built-ins (``anthropic.tools.*``,
   ``openai.tools.*``) with a clear redirect message.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ai import models
from ai.models import anthropic, openai
from ai.models.ai_gateway import adapter, ai_gateway
from ai.models.ai_gateway import tools as gateway_tools

from .conftest import mock_client, sse, user_msg

_TEST_MODEL = ai_gateway("test-provider/test-model")


def _capture_body_handler(captured: dict[str, Any]) -> Any:
    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(
            200,
            text=sse({"type": "finish", "finishReason": "stop", "usage": {}}),
        )

    return handler


class TestGatewayBuiltins:
    async def test_anthropic_web_search_serializes(self) -> None:
        captured: dict[str, Any] = {}
        client = mock_client(httpx.MockTransport(_capture_body_handler(captured)))

        async for _ in adapter.stream(
            client,
            _TEST_MODEL,
            [user_msg("hi")],
            tools=[gateway_tools.anthropic_web_search(max_uses=3)],
        ):
            pass

        assert captured["tools"] == [
            {
                "type": "provider",
                "id": "anthropic.web_search_20260209",
                "args": {"max_uses": 3},
            }
        ]

    async def test_openai_mcp_serializes(self) -> None:
        captured: dict[str, Any] = {}
        client = mock_client(httpx.MockTransport(_capture_body_handler(captured)))

        async for _ in adapter.stream(
            client,
            _TEST_MODEL,
            [user_msg("hi")],
            tools=[
                gateway_tools.openai_mcp(
                    server_label="my-server",
                    server_url="https://mcp.example.com",
                ),
            ],
        ):
            pass

        assert captured["tools"] == [
            {
                "type": "provider",
                "id": "openai.mcp",
                "args": {
                    "server_label": "my-server",
                    "server_url": "https://mcp.example.com",
                },
            }
        ]

    async def test_all_gateway_ids_are_registered(self) -> None:
        """Every concrete _GatewayBuiltin subclass declares a non-empty id."""
        seen: set[str] = set()
        for cls in gateway_tools._GatewayBuiltin.__subclasses__():
            gateway_id = cls.gateway_id
            assert gateway_id, f"{cls.__name__} missing gateway_id"
            assert gateway_id not in seen, f"duplicate gateway_id {gateway_id!r}"
            seen.add(gateway_id)

    async def test_native_anthropic_tool_rejected(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            raise AssertionError("request should not be sent")

        client = mock_client(httpx.MockTransport(handler))
        stream = models.stream(
            ai_gateway("anthropic/claude-sonnet-4", client=client),
            [user_msg("hi")],
            tools=[anthropic.tools.web_search(max_uses=1)],
        )

        with pytest.raises(TypeError, match="ai_gateway.tools"):
            async for _ in stream:
                pass

    async def test_native_openai_tool_rejected(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            raise AssertionError("request should not be sent")

        client = mock_client(httpx.MockTransport(handler))
        stream = models.stream(
            ai_gateway("openai/gpt-5.4", client=client),
            [user_msg("hi")],
            tools=[openai.tools.web_search()],
        )

        with pytest.raises(TypeError, match="ai_gateway.tools"):
            async for _ in stream:
                pass

"""Tests for the AI Gateway built-in tool surface.

The gateway adapter accepts native ``anthropic.tools.*`` and
``openai.tools.*`` instances and serializes them as v3 ``provider``
blocks with id ``f"<provider>.<wire_type>"``. Foreign ``BuiltinTool``
subclasses are rejected with a clear redirect message.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ai.models import anthropic, openai
from ai.models.ai_gateway import adapter, ai_gateway
from ai.models.anthropic.tools import _AnthropicBuiltin
from ai.models.openai.tools import _OpenAIBuiltin
from ai.types import tools as tools_

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
            tools=[anthropic.tools.web_search(max_uses=3)],
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
                openai.tools.mcp(
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

    async def test_native_builtins_have_wire_types(self) -> None:
        """Every concrete native built-in declares a non-empty wire_type
        and the gateway-derived id is unique per provider."""
        for base, prefix in (
            (_AnthropicBuiltin, "anthropic"),
            (_OpenAIBuiltin, "openai"),
        ):
            seen: set[str] = set()
            for cls in base.__subclasses__():
                wire_type = cls.wire_type
                assert wire_type, f"{cls.__name__} missing wire_type"
                gateway_id = f"{prefix}.{wire_type}"
                assert gateway_id not in seen, f"duplicate id {gateway_id!r}"
                seen.add(gateway_id)

    async def test_foreign_builtin_tool_rejected(self) -> None:
        """A BuiltinTool subclass not in either provider hierarchy raises."""

        class CustomBuiltin(tools_.BuiltinTool):
            pass

        def handler(req: httpx.Request) -> httpx.Response:
            raise AssertionError("request should not be sent")

        client = mock_client(httpx.MockTransport(handler))
        stream = adapter.stream(
            client,
            _TEST_MODEL,
            [user_msg("hi")],
            tools=[CustomBuiltin()],
        )

        with pytest.raises(TypeError, match="anthropic.tools|openai.tools"):
            async for _ in stream:
                pass

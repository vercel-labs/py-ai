"""Tests for the AI Gateway built-in tool surface.

The gateway adapter accepts native ``anthropic.tools.*``, ``openai.tools.*``,
and ``ai_gateway.tools.*`` instances and serializes them as v3 ``provider``
blocks of the shape ``{type, id, name, args}`` with camelCase ``args`` keys.
Foreign ``BuiltinTool`` subclasses are rejected with a clear redirect message.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ai.models import ai_gateway as ai_gateway_pkg
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
                "name": "web_search",
                "args": {"maxUses": 3},
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
                "name": "mcp",
                "args": {
                    "serverLabel": "my-server",
                    "serverUrl": "https://mcp.example.com",
                },
            }
        ]

    async def test_gateway_perplexity_search_serializes(self) -> None:
        captured: dict[str, Any] = {}
        client = mock_client(httpx.MockTransport(_capture_body_handler(captured)))

        async for _ in adapter.stream(
            client,
            _TEST_MODEL,
            [user_msg("hi")],
            tools=[
                ai_gateway_pkg.tools.perplexity_search(
                    max_results=5,
                    search_domain_filter=["nature.com"],
                ),
            ],
        ):
            pass

        assert captured["tools"] == [
            {
                "type": "provider",
                "id": "gateway.perplexity_search",
                "name": "perplexity_search",
                "args": {
                    "maxResults": 5,
                    "searchDomainFilter": ["nature.com"],
                },
            }
        ]

    async def test_gateway_parallel_search_serializes(self) -> None:
        captured: dict[str, Any] = {}
        client = mock_client(httpx.MockTransport(_capture_body_handler(captured)))

        async for _ in adapter.stream(
            client,
            _TEST_MODEL,
            [user_msg("hi")],
            tools=[
                ai_gateway_pkg.tools.parallel_search(
                    mode="agentic",
                    source_policy={"include_domains": ["wikipedia.org"]},
                ),
            ],
        ):
            pass

        assert captured["tools"] == [
            {
                "type": "provider",
                "id": "gateway.parallel_search",
                "name": "parallel_search",
                "args": {
                    "mode": "agentic",
                    "sourcePolicy": {"includeDomains": ["wikipedia.org"]},
                },
            }
        ]

    async def test_native_builtins_have_types(self) -> None:
        """Every concrete native built-in declares a non-empty type_
        and the gateway-derived id is unique per provider."""
        for base, prefix in (
            (_AnthropicBuiltin, "anthropic"),
            (_OpenAIBuiltin, "openai"),
        ):
            seen: set[str] = set()
            for cls in base.__subclasses__():
                type_val = cls.type_
                assert type_val, f"{cls.__name__} missing type_"
                gateway_id = f"{prefix}.{type_val}"
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

        with pytest.raises(
            TypeError, match="anthropic.tools|openai.tools|ai_gateway.tools"
        ):
            async for _ in stream:
                pass

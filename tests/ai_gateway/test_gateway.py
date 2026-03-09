"""Integration tests for ``GatewayModel``.

Every test exercises the real ``model.stream()`` method with an injected
``httpx.MockTransport``, so the full production code path is covered:

    model.stream()
      → build_request_body()
      → httpx POST (mock)
      → SSE line parsing
      → parse_stream_part()
      → StreamHandler
      → yield Message
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.ai_gateway import GatewayModel, errors
from vercel_ai_sdk.core import messages

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse(*events: dict[str, Any]) -> str:
    """Build SSE response text from event dicts."""
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events)


def _gateway(
    handler: httpx.MockTransport,
    *,
    model: str = "test-provider/test-model",
    api_key: str = "test-key",
    provider_options: dict[str, Any] | None = None,
) -> GatewayModel:
    """Create a ``GatewayModel`` wired to a mock transport."""
    return GatewayModel(
        model=model,
        api_key=api_key,
        base_url="https://gw.test/v3/ai",
        provider_options=provider_options,
        _transport=handler,
    )


async def _collect(
    model: GatewayModel,
    msgs: list[messages.Message],
    **kwargs: Any,
) -> list[messages.Message]:
    """Drain ``model.stream()`` and return all yielded messages."""
    result: list[messages.Message] = []
    async for msg in model.stream(msgs, **kwargs):
        result.append(msg)
    return result


def _user(text: str) -> messages.Message:
    return messages.Message(
        role="user",
        parts=[messages.TextPart(text=text)],
    )


# ---------------------------------------------------------------------------
# Streaming: text, reasoning, tool calls
# ---------------------------------------------------------------------------


class TestStreaming:
    @pytest.mark.asyncio
    async def test_text_stream(self) -> None:
        body = _sse(
            {"type": "text-start", "id": "t1"},
            {"type": "text-delta", "id": "t1", "textDelta": "Hello"},
            {"type": "text-delta", "id": "t1", "textDelta": " World"},
            {"type": "text-end", "id": "t1"},
            {
                "type": "finish",
                "finishReason": "stop",
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                },
            },
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        model = _gateway(httpx.MockTransport(handler))
        msgs = await _collect(model, [_user("Hi")])

        final = msgs[-1]
        assert final.text == "Hello World"
        assert final.is_done
        assert final.usage is not None
        assert final.usage.input_tokens == 5
        assert final.usage.output_tokens == 2

    @pytest.mark.asyncio
    async def test_reasoning_then_text(self) -> None:
        body = _sse(
            {"type": "reasoning-start", "id": "r1"},
            {"type": "reasoning-delta", "id": "r1", "delta": "think"},
            {"type": "reasoning-end", "id": "r1"},
            {"type": "text-start", "id": "t1"},
            {"type": "text-delta", "id": "t1", "textDelta": "42"},
            {"type": "text-end", "id": "t1"},
            {"type": "finish", "finishReason": "stop", "usage": {}},
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        final = (await _collect(_gateway(httpx.MockTransport(handler)), [_user("?")]))[
            -1
        ]
        assert final.reasoning == "think"
        assert final.text == "42"

    @pytest.mark.asyncio
    async def test_streaming_tool_call(self) -> None:
        body = _sse(
            {
                "type": "tool-input-start",
                "id": "tc-1",
                "toolName": "search",
            },
            {"type": "tool-input-delta", "id": "tc-1", "delta": '{"q":'},
            {"type": "tool-input-delta", "id": "tc-1", "delta": '"hi"}'},
            {"type": "tool-input-end", "id": "tc-1"},
            {
                "type": "finish",
                "finishReason": "tool-calls",
                "usage": {},
            },
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        final = (
            await _collect(_gateway(httpx.MockTransport(handler)), [_user("search")])
        )[-1]
        tc = final.tool_calls
        assert len(tc) == 1
        assert tc[0].tool_name == "search"
        assert tc[0].tool_args == '{"q":"hi"}'

    @pytest.mark.asyncio
    async def test_inline_file_stream(self) -> None:
        """Models like Gemini-3-pro-image return inline file parts
        alongside text in the language model stream."""
        body = _sse(
            {"type": "text-start", "id": "t1"},
            {"type": "text-delta", "id": "t1", "textDelta": "Here is an image:"},
            {"type": "text-end", "id": "t1"},
            {
                "type": "file",
                "id": "f1",
                "mediaType": "image/png",
                "data": "iVBORw0KGgo=",
            },
            {
                "type": "finish",
                "finishReason": "stop",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        final = (
            await _collect(_gateway(httpx.MockTransport(handler)), [_user("draw me")])
        )[-1]
        assert final.text == "Here is an image:"
        assert len(final.images) == 1
        assert final.images[0].media_type == "image/png"
        assert final.images[0].data == "iVBORw0KGgo="
        assert final.is_done

    @pytest.mark.asyncio
    async def test_complete_tool_call_part(self) -> None:
        """Non-streaming ``tool-call`` part (one shot) must also work."""
        body = _sse(
            {
                "type": "tool-call",
                "toolCallId": "tc-1",
                "toolName": "get_weather",
                "input": {"city": "SF"},
            },
            {
                "type": "finish",
                "finishReason": "tool-calls",
                "usage": {},
            },
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        final = (
            await _collect(_gateway(httpx.MockTransport(handler)), [_user("weather")])
        )[-1]
        assert len(final.tool_calls) == 1
        assert json.loads(final.tool_calls[0].tool_args) == {"city": "SF"}


# ---------------------------------------------------------------------------
# Request: headers, body, tools
# ---------------------------------------------------------------------------


class TestRequest:
    @pytest.mark.asyncio
    async def test_protocol_headers(self) -> None:
        captured: dict[str, str] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured.update(dict(req.headers))
            return httpx.Response(
                200,
                text=_sse({"type": "finish", "finishReason": "stop", "usage": {}}),
            )

        model = _gateway(
            httpx.MockTransport(handler),
            model="anthropic/claude-sonnet-4",
            api_key="sk-test",
        )
        await _collect(model, [_user("Hi")])

        assert captured["authorization"] == "Bearer sk-test"
        assert captured["ai-gateway-protocol-version"] == "0.0.1"
        assert captured["ai-language-model-specification-version"] == "3"
        assert captured["ai-language-model-id"] == "anthropic/claude-sonnet-4"
        assert captured["ai-language-model-streaming"] == "true"
        assert captured["ai-gateway-auth-method"] == "api-key"

    @pytest.mark.asyncio
    async def test_body_prompt_format(self) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=_sse({"type": "finish", "finishReason": "stop", "usage": {}}),
            )

        await _collect(_gateway(httpx.MockTransport(handler)), [_user("Hello")])

        assert captured_body["prompt"] == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            }
        ]

    @pytest.mark.asyncio
    async def test_provider_options_in_body(self) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=_sse({"type": "finish", "finishReason": "stop", "usage": {}}),
            )

        opts = {"gateway": {"order": ["bedrock", "openai"]}}
        await _collect(
            _gateway(httpx.MockTransport(handler), provider_options=opts),
            [_user("Hi")],
        )

        assert captured_body["providerOptions"] == opts

    @pytest.mark.asyncio
    async def test_real_tool_in_request_body(self) -> None:
        """A real ``@tool``-decorated function must appear correctly
        in the request body sent to the gateway."""

        @ai.tool
        async def lookup(query: str) -> str:
            """Search the database."""
            return "result"

        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=_sse({"type": "finish", "finishReason": "stop", "usage": {}}),
            )

        await _collect(
            _gateway(httpx.MockTransport(handler)),
            [_user("find something")],
            tools=[lookup],
        )

        assert "tools" in captured_body
        td = captured_body["tools"][0]
        assert td["name"] == "lookup"
        assert td["type"] == "function"
        assert "query" in td["inputSchema"]["properties"]

    @pytest.mark.asyncio
    async def test_multi_turn_request_body(self) -> None:
        """A multi-turn conversation including a tool result must
        serialize correctly into the v3 prompt format."""
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=_sse({"type": "finish", "finishReason": "stop", "usage": {}}),
            )

        tool_part = messages.ToolPart(
            tool_call_id="tc-1",
            tool_name="search",
            tool_args='{"q": "weather"}',
            status="result",
            result={"temp": 72},
        )
        conversation = [
            _user("What's the weather?"),
            messages.Message(role="assistant", parts=[tool_part]),
            _user("Thanks, and tomorrow?"),
        ]

        await _collect(_gateway(httpx.MockTransport(handler)), conversation)

        prompt = captured_body["prompt"]
        # user → assistant (tool-call) → tool (tool-result) → user
        assert len(prompt) == 4
        assert prompt[0]["role"] == "user"
        assert prompt[1]["role"] == "assistant"
        assert prompt[1]["content"][0]["type"] == "tool-call"
        assert prompt[2]["role"] == "tool"
        assert prompt[2]["content"][0]["type"] == "tool-result"
        assert prompt[3]["role"] == "user"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    @pytest.mark.asyncio
    async def test_401_authentication_error(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={
                    "error": {
                        "message": "Invalid API key",
                        "type": "authentication_error",
                    }
                },
            )

        with pytest.raises(errors.GatewayAuthenticationError):
            await _collect(_gateway(httpx.MockTransport(handler)), [_user("Hi")])

    @pytest.mark.asyncio
    async def test_429_rate_limit_error(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_exceeded",
                    }
                },
            )

        with pytest.raises(errors.GatewayRateLimitError):
            await _collect(_gateway(httpx.MockTransport(handler)), [_user("Hi")])

    @pytest.mark.asyncio
    async def test_404_model_not_found(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={
                    "error": {
                        "message": "Model xyz not found",
                        "type": "model_not_found",
                        "param": {"modelId": "xyz"},
                    }
                },
            )

        with pytest.raises(errors.GatewayModelNotFoundError) as exc_info:
            await _collect(_gateway(httpx.MockTransport(handler)), [_user("Hi")])
        assert exc_info.value.model_id == "xyz"

    @pytest.mark.asyncio
    async def test_500_malformed_response(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Not JSON")

        with pytest.raises(errors.GatewayResponseError):
            await _collect(_gateway(httpx.MockTransport(handler)), [_user("Hi")])

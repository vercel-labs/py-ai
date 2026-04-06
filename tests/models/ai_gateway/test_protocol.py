"""Tests for the v3 protocol serialization and deserialization.

Focus areas:
- ``_messages_to_prompt``: the critical outgoing translation layer
- ``_build_request_body``: using real ``@tool``
- ``_parse_stream_part``: the critical incoming translation layer
- ``_parse_usage``: the two distinct wire formats
"""

from __future__ import annotations

import importlib
import json
from unittest.mock import AsyncMock, patch

import pydantic
import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.models.core.helpers import streaming
from vercel_ai_sdk.types import messages

# The ai_gateway __init__.py re-exports `stream` as a function, which
# shadows the module.  Use importlib to get the actual module.
stream_mod = importlib.import_module("vercel_ai_sdk.models.ai_gateway.stream")

# ---------------------------------------------------------------------------
# _messages_to_prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMessagesToPrompt:
    async def test_system_message(self) -> None:
        msgs = [
            messages.Message(
                role="system",
                parts=[messages.TextPart(text="You are helpful.")],
            )
        ]
        result = await stream_mod._messages_to_prompt(msgs)
        assert result == [{"role": "system", "content": "You are helpful."}]

    async def test_user_message(self) -> None:
        msgs = [
            messages.Message(
                role="user",
                parts=[messages.TextPart(text="Hello")],
            )
        ]
        result = await stream_mod._messages_to_prompt(msgs)
        assert result == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            }
        ]

    async def test_assistant_with_reasoning_and_text(self) -> None:
        msgs = [
            messages.Message(
                role="assistant",
                parts=[
                    messages.ReasoningPart(text="Let me think..."),
                    messages.TextPart(text="42"),
                ],
            )
        ]
        result = await stream_mod._messages_to_prompt(msgs)
        content = result[0]["content"]
        assert content[0] == {"type": "reasoning", "text": "Let me think..."}
        assert content[1] == {"type": "text", "text": "42"}

    async def test_tool_call_with_result_produces_two_messages(self) -> None:
        """A completed tool call must produce an assistant message
        (with the tool-call) AND a tool message (with the result)."""
        msgs = [
            messages.Message(
                role="assistant",
                parts=[
                    messages.ToolPart(
                        tool_call_id="tc-1",
                        tool_name="get_weather",
                        tool_args='{"city": "SF"}',
                        status="result",
                        result={"temp": 72},
                    )
                ],
            )
        ]
        result = await stream_mod._messages_to_prompt(msgs)
        assert len(result) == 2

        # Assistant message has the tool-call
        tc = result[0]["content"][0]
        assert tc["type"] == "tool-call"
        assert tc["toolCallId"] == "tc-1"
        assert tc["input"] == {"city": "SF"}

        # Tool message has the result
        tr = result[1]["content"][0]
        assert tr["type"] == "tool-result"
        assert tr["output"] == {"type": "json", "value": {"temp": 72}}

    async def test_tool_error_result(self) -> None:
        msgs = [
            messages.Message(
                role="assistant",
                parts=[
                    messages.ToolPart(
                        tool_call_id="tc-1",
                        tool_name="get_weather",
                        tool_args="{}",
                        status="error",
                        result="Connection timeout",
                    )
                ],
            )
        ]
        result = await stream_mod._messages_to_prompt(msgs)
        tr = result[1]["content"][0]
        assert tr["output"]["type"] == "error-text"
        assert tr["output"]["value"] == "Connection timeout"

    async def test_user_message_with_image_url(self) -> None:
        """FilePart with image URL -> downloaded and converted to data: URL."""
        fake_jpeg = b"\xff\xd8\xff\xe0"
        msgs = [
            messages.Message(
                role="user",
                parts=[
                    messages.TextPart(text="Look at this"),
                    messages.FilePart(
                        data="https://example.com/cat.jpg", media_type="image/jpeg"
                    ),
                ],
            )
        ]
        with patch(
            "vercel_ai_sdk.models.core.helpers.media.download",
            new_callable=AsyncMock,
            return_value=(fake_jpeg, "image/jpeg"),
        ):
            result = await stream_mod._messages_to_prompt(msgs)
        content = result[0]["content"]
        assert content[0] == {"type": "text", "text": "Look at this"}
        assert content[1]["type"] == "file"
        assert content[1]["mediaType"] == "image/jpeg"
        assert content[1]["data"].startswith("data:image/jpeg;base64,")

    async def test_user_message_with_file_bytes(self) -> None:
        """FilePart with bytes -> v3 file content part with data URL."""
        msgs = [
            messages.Message(
                role="user",
                parts=[
                    messages.FilePart(
                        data=b"\x89PNG", media_type="image/png", filename="pic.png"
                    ),
                ],
            )
        ]
        result = await stream_mod._messages_to_prompt(msgs)
        part = result[0]["content"][0]
        assert part["type"] == "file"
        assert part["mediaType"] == "image/png"
        assert part["data"].startswith("data:image/png;base64,")
        assert part["filename"] == "pic.png"

    async def test_user_message_text_only_unchanged(self) -> None:
        """Regression: text-only user messages still work."""
        msgs = [
            messages.Message(
                role="user",
                parts=[messages.TextPart(text="Hello")],
            )
        ]
        result = await stream_mod._messages_to_prompt(msgs)
        assert result == [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]

    async def test_pending_tool_call_no_tool_message(self) -> None:
        """A pending tool call should NOT produce a tool-result message."""
        msgs = [
            messages.Message(
                role="assistant",
                parts=[
                    messages.ToolPart(
                        tool_call_id="tc-1",
                        tool_name="search",
                        tool_args="{}",
                        status="pending",
                    )
                ],
            )
        ]
        result = await stream_mod._messages_to_prompt(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"


# ---------------------------------------------------------------------------
# _build_request_body — using real @tool
# ---------------------------------------------------------------------------


@ai.tool
async def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


@pytest.mark.asyncio
class TestBuildRequestBody:
    async def test_with_real_tool(self) -> None:
        """Verify @tool-produced schema round-trips through
        _build_request_body -> JSON -> gateway wire format."""
        msgs = [
            messages.Message(
                role="user",
                parts=[messages.TextPart(text="What's the weather?")],
            )
        ]
        body = await stream_mod._build_request_body(msgs, tools=[get_weather])

        assert "tools" in body
        tool_def = body["tools"][0]
        assert tool_def["type"] == "function"
        assert tool_def["name"] == "get_weather"
        assert tool_def["description"] == ("Get the current weather for a city.")
        # The schema comes from pydantic — verify structure, not exact dict
        schema = tool_def["inputSchema"]
        assert "properties" in schema
        assert "city" in schema["properties"]
        assert "units" in schema["properties"]
        # 'city' is required (no default), 'units' is not (has default)
        assert "city" in schema.get("required", [])

    async def test_with_output_type(self) -> None:
        class WeatherResult(pydantic.BaseModel):
            temp: float
            condition: str

        msgs = [
            messages.Message(
                role="user",
                parts=[messages.TextPart(text="Weather?")],
            )
        ]
        body = await stream_mod._build_request_body(msgs, output_type=WeatherResult)

        assert "responseFormat" in body
        rf = body["responseFormat"]
        assert rf["type"] == "json"
        assert rf["name"] == "WeatherResult"
        assert "properties" in rf["schema"]
        assert "temp" in rf["schema"]["properties"]

    async def test_provider_options_passthrough(self) -> None:
        msgs = [
            messages.Message(
                role="user",
                parts=[messages.TextPart(text="Hi")],
            )
        ]
        opts = {"gateway": {"order": ["bedrock", "openai"]}}
        body = await stream_mod._build_request_body(msgs, provider_options=opts)
        assert body["providerOptions"] == opts


# ---------------------------------------------------------------------------
# _parse_stream_part — parametrized simple 1:1 mappings
# ---------------------------------------------------------------------------

_SIMPLE_STREAM_PARTS = [
    (
        {"type": "text-start", "id": "t1"},
        streaming.TextStart(block_id="t1"),
    ),
    (
        {"type": "text-end", "id": "t1"},
        streaming.TextEnd(block_id="t1"),
    ),
    (
        {"type": "reasoning-start", "id": "r1"},
        streaming.ReasoningStart(block_id="r1"),
    ),
    (
        {"type": "reasoning-delta", "id": "r1", "delta": "hmm"},
        streaming.ReasoningDelta(block_id="r1", delta="hmm"),
    ),
    (
        {"type": "reasoning-end", "id": "r1"},
        streaming.ReasoningEnd(block_id="r1"),
    ),
    (
        {"type": "tool-input-start", "id": "tc-1", "toolName": "search"},
        streaming.ToolStart(tool_call_id="tc-1", tool_name="search"),
    ),
    (
        {"type": "tool-input-delta", "id": "tc-1", "delta": '{"q"'},
        streaming.ToolArgsDelta(tool_call_id="tc-1", delta='{"q"'),
    ),
    (
        {"type": "tool-input-end", "id": "tc-1"},
        streaming.ToolEnd(tool_call_id="tc-1"),
    ),
]


@pytest.mark.parametrize(
    ("wire", "expected"),
    _SIMPLE_STREAM_PARTS,
    ids=[w["type"] for w, _ in _SIMPLE_STREAM_PARTS],
)
def test_parse_stream_part_simple(
    wire: dict[str, object], expected: streaming.StreamEvent
) -> None:
    events = stream_mod._parse_stream_part(wire)
    assert len(events) == 1
    assert events[0] == expected


@pytest.mark.asyncio
class TestParseStreamPartComplex:
    async def test_text_delta_uses_textDelta_key(self) -> None:
        """The gateway sends ``textDelta`` (camelCase), not ``delta``."""
        events = stream_mod._parse_stream_part(
            {"type": "text-delta", "id": "t1", "textDelta": "Hello"}
        )
        assert isinstance(events[0], streaming.TextDelta)
        assert events[0].delta == "Hello"

    async def test_tool_call_expands_to_three_events(self) -> None:
        """A complete ``tool-call`` part must expand into
        ToolStart -> ToolArgsDelta -> ToolEnd."""
        events = stream_mod._parse_stream_part(
            {
                "type": "tool-call",
                "toolCallId": "tc-1",
                "toolName": "get_weather",
                "input": {"city": "SF"},
            }
        )
        assert len(events) == 3
        assert isinstance(events[0], streaming.ToolStart)
        assert events[0].tool_name == "get_weather"
        assert isinstance(events[1], streaming.ToolArgsDelta)
        assert json.loads(events[1].delta) == {"city": "SF"}
        assert isinstance(events[2], streaming.ToolEnd)

    async def test_finish_flat_usage(self) -> None:
        events = stream_mod._parse_stream_part(
            {
                "type": "finish",
                "finishReason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                },
            }
        )
        done = events[0]
        assert isinstance(done, streaming.MessageDone)
        assert done.finish_reason == "stop"
        assert done.usage is not None
        assert done.usage.input_tokens == 10
        assert done.usage.output_tokens == 20

    async def test_finish_v3_nested_usage(self) -> None:
        events = stream_mod._parse_stream_part(
            {
                "type": "finish",
                "finishReason": {
                    "unified": "tool-calls",
                    "raw": "tool_calls",
                },
                "usage": {
                    "inputTokens": {
                        "total": 100,
                        "cacheRead": 50,
                    },
                    "outputTokens": {
                        "total": 200,
                        "reasoning": 30,
                    },
                },
            }
        )
        done = events[0]
        assert isinstance(done, streaming.MessageDone)
        assert done.finish_reason == "tool-calls"
        assert done.usage is not None
        assert done.usage.input_tokens == 100
        assert done.usage.cache_read_tokens == 50
        assert done.usage.reasoning_tokens == 30

    async def test_file_part(self) -> None:
        """A ``file`` stream part (inline image from Gemini/GPT-5)
        must produce a FileEvent."""
        events = stream_mod._parse_stream_part(
            {
                "type": "file",
                "id": "f1",
                "mediaType": "image/png",
                "data": "iVBORw0KGgo=",
            }
        )
        assert len(events) == 1
        assert isinstance(events[0], streaming.FileEvent)
        assert events[0].block_id == "f1"
        assert events[0].media_type == "image/png"
        assert events[0].data == "iVBORw0KGgo="

    async def test_file_part_defaults(self) -> None:
        """A minimal ``file`` part uses sensible defaults."""
        events = stream_mod._parse_stream_part({"type": "file", "data": "somedata"})
        assert len(events) == 1
        assert isinstance(events[0], streaming.FileEvent)
        assert events[0].media_type == "application/octet-stream"

    async def test_unknown_types_produce_no_events(self) -> None:
        for t in ("stream-start", "raw", "response-metadata", "banana"):
            assert stream_mod._parse_stream_part({"type": t}) == []


# ---------------------------------------------------------------------------
# _parse_usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestParseUsage:
    async def test_flat_format(self) -> None:
        usage = stream_mod._parse_usage({"prompt_tokens": 10, "completion_tokens": 20})
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20

    async def test_v3_nested_format(self) -> None:
        usage = stream_mod._parse_usage(
            {
                "inputTokens": {
                    "total": 100,
                    "cacheRead": 30,
                    "cacheWrite": 5,
                },
                "outputTokens": {"total": 50, "reasoning": 10},
            }
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_tokens == 30
        assert usage.cache_write_tokens == 5
        assert usage.reasoning_tokens == 10

    async def test_non_dict_returns_empty(self) -> None:
        usage = stream_mod._parse_usage("not a dict")
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

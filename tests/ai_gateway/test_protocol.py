"""Tests for the v3 protocol serialization and deserialization.

Focus areas:
- ``messages_to_v3_prompt``: the critical outgoing translation layer
- ``tools_to_v3`` / ``build_request_body``: using real ``@tool``
- ``parse_stream_part``: the critical incoming translation layer
- ``parse_generate_result``: non-streaming response handling
- ``_parse_usage``: the two distinct wire formats
"""

from __future__ import annotations

import json

import pydantic
import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.ai_gateway import protocol
from vercel_ai_sdk.core import llm, messages

# ---------------------------------------------------------------------------
# messages_to_v3_prompt
# ---------------------------------------------------------------------------


class TestMessagesToV3Prompt:
    def test_system_message(self) -> None:
        msgs = [
            messages.Message(
                role="system",
                parts=[messages.TextPart(text="You are helpful.")],
            )
        ]
        result = protocol.messages_to_v3_prompt(msgs)
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_user_message(self) -> None:
        msgs = [
            messages.Message(
                role="user",
                parts=[messages.TextPart(text="Hello")],
            )
        ]
        result = protocol.messages_to_v3_prompt(msgs)
        assert result == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            }
        ]

    def test_assistant_with_reasoning_and_text(self) -> None:
        msgs = [
            messages.Message(
                role="assistant",
                parts=[
                    messages.ReasoningPart(text="Let me think..."),
                    messages.TextPart(text="42"),
                ],
            )
        ]
        result = protocol.messages_to_v3_prompt(msgs)
        content = result[0]["content"]
        assert content[0] == {"type": "reasoning", "text": "Let me think..."}
        assert content[1] == {"type": "text", "text": "42"}

    def test_tool_call_with_result_produces_two_messages(self) -> None:
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
        result = protocol.messages_to_v3_prompt(msgs)
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

    def test_tool_error_result(self) -> None:
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
        result = protocol.messages_to_v3_prompt(msgs)
        tr = result[1]["content"][0]
        assert tr["output"]["type"] == "error-text"
        assert tr["output"]["value"] == "Connection timeout"

    def test_pending_tool_call_no_tool_message(self) -> None:
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
        result = protocol.messages_to_v3_prompt(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"


# ---------------------------------------------------------------------------
# tools_to_v3 / build_request_body — using real @tool
# ---------------------------------------------------------------------------


@ai.tool
async def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


class TestBuildRequestBody:
    def test_with_real_tool(self) -> None:
        """Verify @tool-produced schema round-trips through
        build_request_body → JSON → gateway wire format."""
        msgs = [
            messages.Message(
                role="user",
                parts=[messages.TextPart(text="What's the weather?")],
            )
        ]
        body = protocol.build_request_body(msgs, tools=[get_weather])

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

    def test_with_output_type(self) -> None:
        class WeatherResult(pydantic.BaseModel):
            temp: float
            condition: str

        msgs = [
            messages.Message(
                role="user",
                parts=[messages.TextPart(text="Weather?")],
            )
        ]
        body = protocol.build_request_body(msgs, output_type=WeatherResult)

        assert "responseFormat" in body
        rf = body["responseFormat"]
        assert rf["type"] == "json"
        assert rf["name"] == "WeatherResult"
        assert "properties" in rf["schema"]
        assert "temp" in rf["schema"]["properties"]

    def test_provider_options_passthrough(self) -> None:
        msgs = [
            messages.Message(
                role="user",
                parts=[messages.TextPart(text="Hi")],
            )
        ]
        opts = {"gateway": {"order": ["bedrock", "openai"]}}
        body = protocol.build_request_body(msgs, provider_options=opts)
        assert body["providerOptions"] == opts


# ---------------------------------------------------------------------------
# parse_stream_part — parametrized simple 1:1 mappings
# ---------------------------------------------------------------------------

_SIMPLE_STREAM_PARTS = [
    (
        {"type": "text-start", "id": "t1"},
        llm.TextStart(block_id="t1"),
    ),
    (
        {"type": "text-end", "id": "t1"},
        llm.TextEnd(block_id="t1"),
    ),
    (
        {"type": "reasoning-start", "id": "r1"},
        llm.ReasoningStart(block_id="r1"),
    ),
    (
        {"type": "reasoning-delta", "id": "r1", "delta": "hmm"},
        llm.ReasoningDelta(block_id="r1", delta="hmm"),
    ),
    (
        {"type": "reasoning-end", "id": "r1"},
        llm.ReasoningEnd(block_id="r1"),
    ),
    (
        {"type": "tool-input-start", "id": "tc-1", "toolName": "search"},
        llm.ToolStart(tool_call_id="tc-1", tool_name="search"),
    ),
    (
        {"type": "tool-input-delta", "id": "tc-1", "delta": '{"q"'},
        llm.ToolArgsDelta(tool_call_id="tc-1", delta='{"q"'),
    ),
    (
        {"type": "tool-input-end", "id": "tc-1"},
        llm.ToolEnd(tool_call_id="tc-1"),
    ),
]


@pytest.mark.parametrize(
    ("wire", "expected"),
    _SIMPLE_STREAM_PARTS,
    ids=[w["type"] for w, _ in _SIMPLE_STREAM_PARTS],
)
def test_parse_stream_part_simple(
    wire: dict[str, object], expected: llm.StreamEvent
) -> None:
    events = protocol.parse_stream_part(wire)
    assert len(events) == 1
    assert events[0] == expected


class TestParseStreamPartComplex:
    def test_text_delta_uses_textDelta_key(self) -> None:
        """The gateway sends ``textDelta`` (camelCase), not ``delta``."""
        events = protocol.parse_stream_part(
            {"type": "text-delta", "id": "t1", "textDelta": "Hello"}
        )
        assert isinstance(events[0], llm.TextDelta)
        assert events[0].delta == "Hello"

    def test_tool_call_expands_to_three_events(self) -> None:
        """A complete ``tool-call`` part must expand into
        ToolStart → ToolArgsDelta → ToolEnd."""
        events = protocol.parse_stream_part(
            {
                "type": "tool-call",
                "toolCallId": "tc-1",
                "toolName": "get_weather",
                "input": {"city": "SF"},
            }
        )
        assert len(events) == 3
        assert isinstance(events[0], llm.ToolStart)
        assert events[0].tool_name == "get_weather"
        assert isinstance(events[1], llm.ToolArgsDelta)
        assert json.loads(events[1].delta) == {"city": "SF"}
        assert isinstance(events[2], llm.ToolEnd)

    def test_finish_flat_usage(self) -> None:
        events = protocol.parse_stream_part(
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
        assert isinstance(done, llm.MessageDone)
        assert done.finish_reason == "stop"
        assert done.usage is not None
        assert done.usage.input_tokens == 10
        assert done.usage.output_tokens == 20

    def test_finish_v3_nested_usage(self) -> None:
        events = protocol.parse_stream_part(
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
        assert isinstance(done, llm.MessageDone)
        assert done.finish_reason == "tool-calls"
        assert done.usage is not None
        assert done.usage.input_tokens == 100
        assert done.usage.cache_read_tokens == 50
        assert done.usage.reasoning_tokens == 30

    def test_unknown_types_produce_no_events(self) -> None:
        for t in ("stream-start", "raw", "response-metadata", "banana"):
            assert protocol.parse_stream_part({"type": t}) == []


# ---------------------------------------------------------------------------
# parse_generate_result
# ---------------------------------------------------------------------------


class TestParseGenerateResult:
    def test_text_content(self) -> None:
        events = protocol.parse_generate_result(
            {
                "content": [{"type": "text", "text": "Hello!"}],
                "finishReason": "stop",
                "usage": {"prompt_tokens": 4, "completion_tokens": 10},
            }
        )
        # TextStart + TextDelta + TextEnd + MessageDone
        assert len(events) == 4
        assert isinstance(events[1], llm.TextDelta)
        assert events[1].delta == "Hello!"
        assert isinstance(events[3], llm.MessageDone)

    def test_tool_call_content(self) -> None:
        events = protocol.parse_generate_result(
            {
                "content": [
                    {
                        "type": "tool-call",
                        "toolCallId": "tc-1",
                        "toolName": "search",
                        "input": {"query": "weather"},
                    }
                ],
                "finishReason": "tool-calls",
            }
        )
        assert isinstance(events[0], llm.ToolStart)
        assert isinstance(events[3], llm.MessageDone)
        assert events[3].finish_reason == "tool-calls"


# ---------------------------------------------------------------------------
# _parse_usage
# ---------------------------------------------------------------------------


class TestParseUsage:
    def test_flat_format(self) -> None:
        usage = protocol._parse_usage({"prompt_tokens": 10, "completion_tokens": 20})
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20

    def test_v3_nested_format(self) -> None:
        usage = protocol._parse_usage(
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

    def test_non_dict_returns_empty(self) -> None:
        usage = protocol._parse_usage("not a dict")
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

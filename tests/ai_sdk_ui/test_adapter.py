"""
Based on: .reference/ai/packages/ai/src/ui/process-ui-message-stream.test.ts
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.ai_sdk_ui.adapter import (
    to_ui_message_stream,
    to_messages,
    UIMessage,
    UITextPart,
    UIStepStartPart,
    UIToolPart,
)
from vercel_ai_sdk.core.messages import Message, TextPart, ToolPart


async def get_event_types(messages: list[Message]) -> list[str]:
    """Stream messages through adapter and return event type sequence."""

    async def stream():
        for m in messages:
            yield m

    return [p.type async for p in to_ui_message_stream(stream())]


# -----------------------------------------------------------------------------
# Event sequence tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_streaming():
    """Text: start -> start-step -> text-start/delta/end -> finish-step -> finish"""
    messages = [
        Message(
            id="msg-1",
            role="assistant",
            parts=[TextPart(text="Hello", delta="Hello", state="streaming")],
        ),
        Message(
            id="msg-1",
            role="assistant",
            parts=[TextPart(text="Hello, world!", delta=", world!", state="streaming")],
        ),
        Message(
            id="msg-1",
            role="assistant",
            parts=[TextPart(text="Hello, world!", state="done")],
        ),
    ]

    assert await get_event_types(messages) == [
        "start",
        "start-step",
        "text-start",
        "text-delta",
        "text-delta",
        "text-end",
        "finish-step",
        "finish",
    ]


@pytest.mark.asyncio
async def test_tool_roundtrip():
    """Server-side tool: input-available -> output-available -> text response.

    Reference: process-ui-message-stream.test.ts "server-side tool roundtrip"
    """
    messages = [
        # Tool pending
        Message(
            id="msg-1",
            role="assistant",
            parts=[
                ToolPart(
                    tool_call_id="tc-1",
                    tool_name="get_weather",
                    tool_args='{"city": "London"}',
                    status="pending",
                    state="done",
                ),
            ],
        ),
        # Tool result
        Message(
            id="msg-1",
            role="assistant",
            parts=[
                ToolPart(
                    tool_call_id="tc-1",
                    tool_name="get_weather",
                    tool_args='{"city": "London"}',
                    status="result",
                    result={"weather": "sunny"},
                    state="done",
                ),
            ],
        ),
        # Final text
        Message(
            id="msg-2",
            role="assistant",
            parts=[
                TextPart(text="The weather is sunny.", state="done"),
            ],
        ),
    ]

    assert await get_event_types(messages) == [
        "start",
        "start-step",
        "tool-input-start",
        "tool-input-available",
        "tool-output-available",
        "finish-step",
        "start-step",
        "text-start",
        "text-end",
        "finish-step",
        "finish",
    ]


@pytest.mark.asyncio
async def test_text_then_tool_then_text():
    """Full mothership scenario: text -> tool -> result -> final text.

    Input: "when will the robots take over?"
    1. Text: "I'll check with the mothership..."
    2. Tool: talk_to_mothership(question="...")
    3. Result: "Soon."
    4. Text: "According to the mothership: Soon."
    """
    messages = [
        # Streaming initial text
        Message(
            id="msg-1",
            role="assistant",
            parts=[
                TextPart(
                    text="I'll check with the mothership.",
                    delta="I'll check with the mothership.",
                    state="streaming",
                )
            ],
        ),
        # Text done + tool pending
        Message(
            id="msg-1",
            role="assistant",
            parts=[
                TextPart(text="I'll check with the mothership.", state="done"),
                ToolPart(
                    tool_call_id="tc-1",
                    tool_name="talk_to_mothership",
                    tool_args='{"question": "when?"}',
                    status="pending",
                    state="done",
                ),
            ],
        ),
        # Tool result
        Message(
            id="msg-1",
            role="assistant",
            parts=[
                TextPart(text="I'll check with the mothership.", state="done"),
                ToolPart(
                    tool_call_id="tc-1",
                    tool_name="talk_to_mothership",
                    tool_args='{"question": "when?"}',
                    status="result",
                    result={"answer": "Soon."},
                    state="done",
                ),
            ],
        ),
        # Final text (new message)
        Message(
            id="msg-2",
            role="assistant",
            parts=[
                TextPart(
                    text="According to the mothership: Soon.",
                    delta="According to the mothership: Soon.",
                    state="streaming",
                )
            ],
        ),
        Message(
            id="msg-2",
            role="assistant",
            parts=[TextPart(text="According to the mothership: Soon.", state="done")],
        ),
    ]

    # Per AI SDK protocol, tool-input-available and tool-output-available
    # are in the SAME step (one LLM turn). Reference:
    # process-ui-message-stream.test.ts "server-side tool roundtrip with multiple assistant texts"
    assert await get_event_types(messages) == [
        "start",
        "start-step",
        "text-start",
        "text-delta",
        "text-end",
        "tool-input-start",
        "tool-input-available",
        "tool-output-available",  # Same step as tool-input (AI SDK protocol)
        "finish-step",
        # New step for second LLM call (new message ID)
        "start-step",
        "text-start",
        "text-delta",
        "text-end",
        "finish-step",
        "finish",
    ]


# -----------------------------------------------------------------------------
# Integration tests - runtime-based execution
# -----------------------------------------------------------------------------


class MockLLM(ai.LanguageModel):
    """A mock LLM that yields pre-defined message sequences."""

    def __init__(self, responses: list[list[Message]]) -> None:
        """
        Args:
            responses: List of response sequences. Each call to stream() consumes
                       one sequence and yields its messages.
        """
        self._responses = list(responses)
        self._call_index = 0

    async def stream(
        self,
        messages: list[Message],
        tools: list[ai.Tool] | None = None,
    ) -> AsyncGenerator[Message, None]:
        if self._call_index >= len(self._responses):
            raise RuntimeError("MockLLM: no more responses configured")

        response_sequence = self._responses[self._call_index]
        self._call_index += 1

        for msg in response_sequence:
            yield msg


@ai.tool
async def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"


async def mock_agent(
    llm: ai.LanguageModel,
    user_query: str,
) -> ai.StreamResult:
    """Agent using stream_loop directly."""
    return await ai.stream_loop(
        llm,
        messages=ai.make_messages(system="You are helpful.", user=user_query),
        tools=[get_weather],
    )


@pytest.mark.asyncio
async def test_runtime_tool_roundtrip():
    """
    Integration test: run a mock agent loop through ai.run() and verify
    that tool-input-available and tool-output-available events are emitted.

    This test demonstrates the bug: the runtime yields the message with
    the tool call, but by the time it's yielded the tool has already been
    executed and the ToolPart has been mutated to status="result". The UI
    adapter never sees the intermediate status="pending" state.

    Root cause: stream_loop appends the message, then executes tools which
    mutate the message in-place. The message was already yielded with
    status="pending", but pydantic models are mutable so when we collect
    them at the end, we see the mutated state.
    """
    # First LLM call: returns a tool call
    tool_call_response = [
        Message(
            id="msg-1",
            role="assistant",
            parts=[
                ToolPart(
                    tool_call_id="tc-1",
                    tool_name="get_weather",
                    tool_args='{"city": "London"}',
                    status="pending",
                    state="done",
                ),
            ],
        ),
    ]

    # Second LLM call: returns final text
    final_text_response = [
        Message(
            id="msg-2",
            role="assistant",
            parts=[TextPart(text="The weather is sunny.", state="done")],
        ),
    ]

    mock_llm = MockLLM([tool_call_response, final_text_response])

    # Collect all messages from the runtime
    runtime_messages: list[Message] = []
    async for msg in ai.run(mock_agent, mock_llm, "What's the weather in London?"):
        runtime_messages.append(msg)

    # Stream through UI adapter
    event_types = [
        p.type async for p in to_ui_message_stream(_async_iter(runtime_messages))
    ]

    # This is what SHOULD happen:
    # 1. First step yields tool call with status="pending" -> tool-input-start, tool-input-available
    # 2. After tool execution, we yield the same message with status="result" -> tool-output-available
    #    (same step because same message ID)
    # 3. Second LLM step yields final text -> text-start, text-end
    expected = [
        "start",
        "start-step",
        "tool-input-start",
        "tool-input-available",
        "tool-output-available",  # Same step as input (same message ID)
        "finish-step",
        # Second LLM call (new message ID = new step)
        "start-step",
        "text-start",
        "text-end",
        "finish-step",
        "finish",
    ]

    assert event_types == expected


async def _async_iter(items: list[Message]) -> AsyncGenerator[Message, None]:
    """Helper to convert a list to an async generator."""
    for item in items:
        yield item


# -----------------------------------------------------------------------------
# UI → Internal conversion tests
# -----------------------------------------------------------------------------


def test_ui_to_internal_two_turn_with_tool():
    """Test converting a realistic two-turn conversation with tool call.

    This test uses the exact payload structure from a real AI SDK frontend
    that was causing 422 validation errors due to:
    1. step-start parts (boundary markers)
    2. tool-{toolName} dynamic type pattern (e.g., "tool-talk_to_mothership")
    """
    # Exact structure from a failing request
    raw_messages = [
        {
            "id": "lmaOqWZJdKOVUbYT",
            "role": "user",
            "parts": [{"type": "text", "text": "when will the robots take over?"}],
        },
        {
            "id": "d04b88d9a82e",
            "role": "assistant",
            "parts": [
                {"type": "step-start"},
                {
                    "type": "text",
                    "text": "I'll check with the mothership about this important question.",
                    "state": "done",
                },
                {
                    "type": "tool-talk_to_mothership",
                    "toolCallId": "toolu_01FiXNXhq1kHx4TegRjSaJyv",
                    "state": "output-available",
                    "input": '{"question": "when will the robots take over?"}',
                    "output": "Soon.",
                },
                {"type": "text", "text": "", "state": "done"},  # Empty text part
                {"type": "step-start"},
                {
                    "type": "text",
                    "text": "The mothership has spoken: Soon.",
                    "state": "done",
                },
                {"type": "text", "text": "", "state": "done"},  # Empty text part
            ],
        },
        {
            "id": "ZLi3qVpgZLBjwMxZ",
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "text": "this is a test run. can you remember the first turn?",
                }
            ],
        },
    ]

    # Parse UI messages - this should NOT raise validation errors
    ui_messages = [UIMessage.model_validate(m) for m in raw_messages]

    # Verify parsing worked
    assert len(ui_messages) == 3
    assert ui_messages[0].role == "user"
    assert ui_messages[1].role == "assistant"
    assert ui_messages[2].role == "user"

    # Check that step-start and tool parts were parsed correctly
    assistant_parts = ui_messages[1].parts
    assert isinstance(assistant_parts[0], UIStepStartPart)
    assert isinstance(assistant_parts[1], UITextPart)
    assert isinstance(assistant_parts[2], UIToolPart)
    assert assistant_parts[2].tool_name == "talk_to_mothership"
    assert assistant_parts[2].state == "output-available"

    # Convert to internal format
    internal = to_messages(ui_messages)

    # Verify conversion
    assert len(internal) == 3
    assert internal[0].role == "user"
    assert internal[0].text == "when will the robots take over?"

    assert internal[1].role == "assistant"
    # Should have text parts (non-empty) and tool part
    # step-start and empty text parts should be skipped
    text_parts = [p for p in internal[1].parts if isinstance(p, TextPart)]
    tool_parts = internal[1].tool_calls

    assert len(text_parts) == 2  # Two non-empty text parts
    assert (
        text_parts[0].text
        == "I'll check with the mothership about this important question."
    )
    assert text_parts[1].text == "The mothership has spoken: Soon."

    assert len(tool_parts) == 1
    assert tool_parts[0].tool_name == "talk_to_mothership"
    assert tool_parts[0].tool_call_id == "toolu_01FiXNXhq1kHx4TegRjSaJyv"
    assert tool_parts[0].status == "result"  # output-available maps to result
    # Non-dict results are wrapped in {"value": ...} for internal ToolPart compatibility
    assert tool_parts[0].result == {"value": "Soon."}

    assert internal[2].role == "user"
    assert internal[2].text == "this is a test run. can you remember the first turn?"


def test_ui_tool_part_with_dict_input():
    """Test that tool parts with dict input (not JSON string) are handled."""
    raw_message = {
        "id": "msg-1",
        "role": "assistant",
        "parts": [
            {
                "type": "tool-get_weather",
                "toolCallId": "tc-1",
                "state": "input-available",
                "input": {"city": "London"},  # Dict, not JSON string
            }
        ],
    }

    ui_msg = UIMessage.model_validate(raw_message)
    internal = to_messages([ui_msg])

    assert len(internal) == 1
    tool_part = internal[0].tool_calls[0]
    assert tool_part.tool_name == "get_weather"
    assert tool_part.tool_args == '{"city": "London"}'
    assert tool_part.status == "pending"  # input-available maps to pending


def test_ui_skips_unsupported_parts():
    """Test that unsupported part types are skipped gracefully."""
    raw_message = {
        "id": "msg-1",
        "role": "assistant",
        "parts": [
            {"type": "text", "text": "Hello"},
            {"type": "data-custom", "data": {"foo": "bar"}},  # Unsupported
            {"type": "unknown-type", "content": "xyz"},  # Unsupported
            {"type": "text", "text": "World"},
        ],
    }

    ui_msg = UIMessage.model_validate(raw_message)
    # Only text parts should be parsed (data-* and unknown skipped)
    assert len(ui_msg.parts) == 2
    assert all(isinstance(p, UITextPart) for p in ui_msg.parts)

    internal = to_messages([ui_msg])
    assert len(internal[0].parts) == 2

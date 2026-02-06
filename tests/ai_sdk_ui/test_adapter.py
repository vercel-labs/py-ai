"""
Based on: .reference/ai/packages/ai/src/ui/process-ui-message-stream.test.ts
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.ai_sdk_ui.adapter import to_ui_message_stream
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

"""
Based on: .reference/ai/packages/ai/src/ui/process-ui-message-stream.test.ts
"""

import pytest

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

    assert await get_event_types(messages) == [
        "start",
        "start-step",
        "text-start",
        "text-delta",
        "text-end",
        "tool-input-start",
        "tool-input-available",
        "finish-step",
        "start-step",
        "tool-output-available",
        "finish-step",
        "start-step",
        "text-start",
        "text-delta",
        "text-end",
        "finish-step",
        "finish",
    ]

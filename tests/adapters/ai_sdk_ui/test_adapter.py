"""
Based on: .reference/ai/packages/ai/src/ui/process-ui-message-stream.test.ts
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest

import vercel_ai_sdk as ai
from vercel_ai_sdk.adapters.ai_sdk_ui import adapter, ui_message
from vercel_ai_sdk.agents import hooks
from vercel_ai_sdk.types import messages

from ...conftest import MockLLM, tool_msg


async def get_event_types(msgs: list[messages.Message]) -> list[str]:
    """Stream messages through adapter and return event type sequence."""

    async def stream() -> AsyncGenerator[messages.Message]:
        for m in msgs:
            yield m

    return [p.type async for p in adapter.to_ui_message_stream(stream())]


# -----------------------------------------------------------------------------
# Event sequence tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_streaming() -> None:
    """Text: start -> start-step -> text-start/delta/end -> finish-step -> finish"""
    msgs = [
        messages.Message(
            id="msg-1",
            role="assistant",
            parts=[messages.TextPart(text="Hello", delta="Hello", state="streaming")],
        ),
        messages.Message(
            id="msg-1",
            role="assistant",
            parts=[
                messages.TextPart(
                    text="Hello, world!", delta=", world!", state="streaming"
                )
            ],
        ),
        messages.Message(
            id="msg-1",
            role="assistant",
            parts=[messages.TextPart(text="Hello, world!", state="done")],
        ),
    ]

    assert await get_event_types(msgs) == [
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
async def test_tool_roundtrip() -> None:
    """Server-side tool: input-available -> output-available -> text response.

    Reference: process-ui-message-stream.test.ts "server-side tool roundtrip"
    """
    msgs = [
        # Tool pending
        messages.Message(
            id="msg-1",
            role="assistant",
            parts=[
                messages.ToolPart(
                    tool_call_id="tc-1",
                    tool_name="get_weather",
                    tool_args='{"city": "London"}',
                    status="pending",
                    state="done",
                ),
            ],
        ),
        # Tool result
        messages.Message(
            id="msg-1",
            role="assistant",
            parts=[
                messages.ToolPart(
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
        messages.Message(
            id="msg-2",
            role="assistant",
            parts=[
                messages.TextPart(text="The weather is sunny.", state="done"),
            ],
        ),
    ]

    assert await get_event_types(msgs) == [
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
async def test_text_then_tool_then_text() -> None:
    """Full mothership scenario: text -> tool -> result -> final text.

    Input: "when will the robots take over?"
    1. Text: "I'll check with the mothership..."
    2. Tool: talk_to_mothership(question="...")
    3. Result: "Soon."
    4. Text: "According to the mothership: Soon."
    """
    msgs = [
        # Streaming initial text
        messages.Message(
            id="msg-1",
            role="assistant",
            parts=[
                messages.TextPart(
                    text="I'll check with the mothership.",
                    delta="I'll check with the mothership.",
                    state="streaming",
                )
            ],
        ),
        # Text done + tool pending
        messages.Message(
            id="msg-1",
            role="assistant",
            parts=[
                messages.TextPart(text="I'll check with the mothership.", state="done"),
                messages.ToolPart(
                    tool_call_id="tc-1",
                    tool_name="talk_to_mothership",
                    tool_args='{"question": "when?"}',
                    status="pending",
                    state="done",
                ),
            ],
        ),
        # Tool result
        messages.Message(
            id="msg-1",
            role="assistant",
            parts=[
                messages.TextPart(text="I'll check with the mothership.", state="done"),
                messages.ToolPart(
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
        messages.Message(
            id="msg-2",
            role="assistant",
            parts=[
                messages.TextPart(
                    text="According to the mothership: Soon.",
                    delta="According to the mothership: Soon.",
                    state="streaming",
                )
            ],
        ),
        messages.Message(
            id="msg-2",
            role="assistant",
            parts=[
                messages.TextPart(
                    text="According to the mothership: Soon.", state="done"
                )
            ],
        ),
    ]

    # Per AI SDK protocol, tool-input-available and tool-output-available
    # are in the SAME step (one LLM turn). Reference:
    # process-ui-message-stream.test.ts
    # "server-side tool roundtrip with multiple assistant texts"
    assert await get_event_types(msgs) == [
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
async def test_runtime_tool_roundtrip() -> None:
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
        messages.Message(
            id="msg-1",
            role="assistant",
            parts=[
                messages.ToolPart(
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
        messages.Message(
            id="msg-2",
            role="assistant",
            parts=[messages.TextPart(text="The weather is sunny.", state="done")],
        ),
    ]

    mock_llm = MockLLM([tool_call_response, final_text_response])

    # Collect all messages from the runtime
    runtime_messages: list[messages.Message] = []
    async for msg in ai.run(mock_agent, mock_llm, "What's the weather in London?"):
        runtime_messages.append(msg)

    # Stream through UI adapter
    event_types = [
        p.type
        async for p in adapter.to_ui_message_stream(_async_iter(runtime_messages))
    ]

    # This is what SHOULD happen:
    # 1. First step streams tool call args then completes
    #    -> tool-input-start, tool-input-delta, tool-input-available
    # 2. After tool execution, we yield the same message with
    #    status="result" -> tool-output-available
    #    (same step because same message ID)
    # 3. Second LLM step streams text then completes
    #    -> text-start, text-delta, text-end, (final done msg) text-start, text-end
    expected = [
        "start",
        "start-step",
        "tool-input-start",
        "tool-input-delta",
        "tool-input-available",
        "tool-output-available",  # Same step as input (same message ID)
        "finish-step",
        # Second LLM call (new message ID = new step)
        "start-step",
        "text-start",
        "text-delta",
        "text-end",
        "text-start",  # Final done message re-emits completed text
        "text-end",
        "finish-step",
        "finish",
    ]

    assert event_types == expected


async def _async_iter(
    items: list[messages.Message],
) -> AsyncGenerator[messages.Message]:
    """Helper to convert a list to an async generator."""
    for item in items:
        yield item


# -----------------------------------------------------------------------------
# UI → Internal conversion tests
# -----------------------------------------------------------------------------


def test_ui_to_internal_two_turn_with_tool() -> None:
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
                    "text": "I'll check with the mothership "
                    "about this important question.",
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
    ui_messages = [ui_message.UIMessage.model_validate(m) for m in raw_messages]

    # Verify parsing worked
    assert len(ui_messages) == 3
    assert ui_messages[0].role == "user"
    assert ui_messages[1].role == "assistant"
    assert ui_messages[2].role == "user"

    # Check that step-start and tool parts were parsed correctly
    assistant_parts = ui_messages[1].parts
    assert isinstance(assistant_parts[0], ui_message.UIStepStartPart)
    assert isinstance(assistant_parts[1], ui_message.UITextPart)
    assert isinstance(assistant_parts[2], ui_message.UIToolPart)
    assert assistant_parts[2].tool_name == "talk_to_mothership"
    assert assistant_parts[2].state == "output-available"

    # Convert to internal format
    internal = adapter.to_messages(ui_messages)

    # The single UI assistant message contains [text, tool(done), text] from
    # two stream_loop iterations.  to_messages splits at the tool-result
    # boundary so LLM adapters receive one message per iteration.
    assert len(internal) == 4
    assert internal[0].role == "user"
    assert internal[0].text == "when will the robots take over?"

    # First iteration: text + tool call
    assert internal[1].role == "assistant"
    assert internal[1].text == (
        "I'll check with the mothership about this important question."
    )
    assert len(internal[1].tool_calls) == 1
    assert internal[1].tool_calls[0].tool_name == "talk_to_mothership"
    assert internal[1].tool_calls[0].tool_call_id == "toolu_01FiXNXhq1kHx4TegRjSaJyv"
    assert internal[1].tool_calls[0].status == "result"
    assert internal[1].tool_calls[0].result == {"value": "Soon."}

    # Second iteration: follow-up text
    assert internal[2].role == "assistant"
    assert internal[2].text == "The mothership has spoken: Soon."
    assert len(internal[2].tool_calls) == 0

    assert internal[3].role == "user"
    assert internal[3].text == "this is a test run. can you remember the first turn?"


def test_ui_tool_part_with_dict_input() -> None:
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

    ui_msg = ui_message.UIMessage.model_validate(raw_message)
    internal = adapter.to_messages([ui_msg])

    assert len(internal) == 1
    tool_part = internal[0].tool_calls[0]
    assert tool_part.tool_name == "get_weather"
    assert tool_part.tool_args == '{"city": "London"}'
    assert tool_part.status == "pending"  # input-available maps to pending


def test_ui_file_part_converted_to_core_file_part() -> None:
    """UIFilePart from the frontend is converted to a core FilePart."""
    raw_message = {
        "id": "msg-1",
        "role": "user",
        "parts": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "file",
                "mediaType": "image/png",
                "url": "https://example.com/photo.png",
                "filename": "photo.png",
            },
        ],
    }
    ui_msg = ui_message.UIMessage.model_validate(raw_message)
    internal = adapter.to_messages([ui_msg])

    assert len(internal) == 1
    msg = internal[0]
    assert msg.role == "user"
    assert len(msg.parts) == 2
    assert isinstance(msg.parts[0], messages.TextPart)
    assert isinstance(msg.parts[1], messages.FilePart)
    fp = msg.parts[1]
    assert fp.data == "https://example.com/photo.png"
    assert fp.media_type == "image/png"
    assert fp.filename == "photo.png"


def test_ui_skips_unsupported_parts() -> None:
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

    ui_msg = ui_message.UIMessage.model_validate(raw_message)
    # Only text parts should be parsed (data-* and unknown skipped)
    assert len(ui_msg.parts) == 2
    assert all(isinstance(p, ui_message.UITextPart) for p in ui_msg.parts)

    internal = adapter.to_messages([ui_msg])
    assert len(internal[0].parts) == 2


# -----------------------------------------------------------------------------
# Tool approval (human-in-the-loop) tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_approval_hook_emits_approval_request() -> None:
    """Pending ToolApproval HookPart emits tool-approval-request on the wire.

    The HookPart message uses a *different* id from the tool message,
    matching what the Runtime actually does (it creates an ad-hoc Message
    with its own auto-generated id at runtime.py:452).  The adapter must
    keep both in the same step so the frontend's sendAutomaticallyWhen
    helper can find the tool part when the user responds to the approval.
    """
    msgs = [
        # Tool pending (args complete, awaiting approval)
        messages.Message(
            id="msg-1",
            role="assistant",
            parts=[
                messages.ToolPart(
                    tool_call_id="tc-1",
                    tool_name="rm_rf",
                    tool_args='{"path": "/"}',
                    status="pending",
                    state="done",
                ),
            ],
        ),
        # Hook pending (approval requested) — different message id,
        # just like the Runtime produces at runtime.py:452.
        messages.Message(
            id="hook-msg-1",
            role="assistant",
            parts=[
                messages.HookPart(
                    hook_id="approve_tc-1",
                    hook_type=hooks.ToolApproval.hook_type,  # type: ignore[attr-defined]
                    status="pending",
                    metadata={"tool_name": "rm_rf", "tool_args": '{"path": "/"}'},
                ),
            ],
        ),
    ]

    event_types = await get_event_types(msgs)
    # tool-approval-request must be in the SAME step as the tool input —
    # no extra start-step/finish-step between them.
    assert event_types == [
        "start",
        "start-step",
        "tool-input-start",
        "tool-input-available",
        "tool-approval-request",
        "finish-step",
        "finish",
    ]


def test_approval_responded_resolves_hook() -> None:
    """to_messages() resolves the ToolApproval hook for approval-responded parts."""
    label = "approve_tc-42"
    raw_messages = [
        {
            "id": "msg-1",
            "role": "assistant",
            "parts": [
                {
                    "type": "tool-dangerous_action",
                    "toolCallId": "tc-42",
                    "state": "approval-responded",
                    "input": '{"x": 1}',
                    "approval": {
                        "id": label,
                        "approved": True,
                        "reason": "looks safe",
                    },
                }
            ],
        },
    ]

    # Clean up any leftover state from other tests
    hooks._pending_resolutions.pop(label, None)

    ui_msgs = [ui_message.UIMessage.model_validate(m) for m in raw_messages]
    adapter.to_messages(ui_msgs)

    # The side-effect should have pre-registered the resolution
    assert label in hooks._pending_resolutions
    resolution = hooks._pending_resolutions.pop(label)
    assert resolution == {"granted": True, "reason": "looks safe"}


@pytest.mark.asyncio
async def test_runtime_tool_approval_same_step() -> None:
    """E2E: tool-approval-request must land in the same SSE step as the tool call.

    Runs a graph with ToolApproval (cancels_future=True) through ai.run(),
    collects runtime messages, streams through the adapter, and asserts
    that no spurious step boundary appears between tool-input-available
    and tool-approval-request.

    This is the test that would have caught the bug where the Runtime's
    HookPart message (which has a different id from the LLM message)
    caused the adapter to open a new step.
    """

    @ai.tool
    async def dangerous_action(path: str) -> str:
        """Do something dangerous."""
        return f"deleted {path}"

    async def graph(llm: ai.LanguageModel) -> None:
        result = await ai.stream_step(
            llm,
            ai.make_messages(system="You are helpful.", user="delete /tmp"),
            [dangerous_action],
        )
        if not result.tool_calls:
            return

        last_msg = result.last_message
        assert last_msg is not None

        async def approve_and_execute(tc: ai.ToolPart) -> None:
            approval = await ai.ToolApproval.create(  # type: ignore[attr-defined]
                f"approve_{tc.tool_call_id}",
                metadata={"tool_name": tc.tool_name},
            )
            if approval.granted:
                await ai.execute_tool(tc, message=last_msg)
            else:
                tc.set_error("denied")

        await asyncio.gather(*(approve_and_execute(tc) for tc in result.tool_calls))

    mock_llm = MockLLM(
        [
            [
                tool_msg(
                    tc_id="tc-1",
                    name="dangerous_action",
                    args='{"path": "/tmp"}',
                )
            ],
        ]
    )

    runtime_messages: list[messages.Message] = []
    result = ai.run(graph, mock_llm)
    async for msg in result:
        runtime_messages.append(msg)

    # The run should have a pending hook (approval not yet granted)
    assert "approve_tc-1" in result.pending_hooks

    # Stream through UI adapter
    event_types = [
        p.type
        async for p in adapter.to_ui_message_stream(_async_iter(runtime_messages))
    ]

    # tool-approval-request must be in the SAME step as tool-input.
    # If a spurious step boundary sneaks in, we'd see:
    #   [..., "tool-input-available", "finish-step", "start-step",
    #    "tool-approval-request", ...]
    # which breaks the frontend's sendAutomaticallyWhen helper.
    assert event_types == [
        "start",
        "start-step",
        "tool-input-start",
        "tool-input-delta",
        "tool-input-available",
        "tool-approval-request",
        "finish-step",
        "finish",
    ]

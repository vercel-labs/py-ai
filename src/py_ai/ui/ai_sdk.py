"""
Reference: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal

import py_ai as ai

# necessary headers for the streaming integration to work

UI_MESSAGE_STREAM_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "x-vercel-ai-ui-message-stream": "v1",
    "x-accel-buffering": "no",
}


# different kinds of messages expected by the frontend

FinishReason = Literal[
    "stop", "length", "content-filter", "tool-calls", "error", "other"
]


@dataclasses.dataclass
class StartPart:
    """Indicates the beginning of a new message with metadata."""

    type: Literal["start"] = dataclasses.field(default="start", init=False)
    message_id: str | None = None
    message_metadata: Any | None = None


@dataclasses.dataclass
class TextStartPart:
    """Indicates the beginning of a text block."""

    id: str
    type: Literal["text-start"] = dataclasses.field(default="text-start", init=False)
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class TextDeltaPart:
    """Contains incremental text content for the text block."""

    id: str
    delta: str
    type: Literal["text-delta"] = dataclasses.field(default="text-delta", init=False)
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class TextEndPart:
    """Indicates the completion of a text block."""

    id: str
    type: Literal["text-end"] = dataclasses.field(default="text-end", init=False)
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class ToolInputStartPart:
    """Indicates the beginning of tool input streaming."""

    tool_call_id: str
    tool_name: str
    type: Literal["tool-input-start"] = dataclasses.field(
        default="tool-input-start", init=False
    )
    provider_executed: bool | None = None
    dynamic: bool | None = None
    title: str | None = None


@dataclasses.dataclass
class ToolInputAvailablePart:
    """Indicates that tool input is complete and ready for execution."""

    tool_call_id: str
    tool_name: str
    input: Any
    type: Literal["tool-input-available"] = dataclasses.field(
        default="tool-input-available", init=False
    )
    provider_executed: bool | None = None
    provider_metadata: dict[str, Any] | None = None
    dynamic: bool | None = None
    title: str | None = None


@dataclasses.dataclass
class ToolOutputAvailablePart:
    """Contains the result of tool execution."""

    tool_call_id: str
    output: Any
    type: Literal["tool-output-available"] = dataclasses.field(
        default="tool-output-available", init=False
    )
    provider_executed: bool | None = None
    dynamic: bool | None = None
    preliminary: bool | None = None


@dataclasses.dataclass
class StartStepPart:
    """A part indicating the start of a step."""

    type: Literal["start-step"] = dataclasses.field(default="start-step", init=False)


@dataclasses.dataclass
class FinishStepPart:
    """A part indicating that a step has been completed."""

    type: Literal["finish-step"] = dataclasses.field(default="finish-step", init=False)


@dataclasses.dataclass
class FinishPart:
    """A part indicating the completion of a message."""

    type: Literal["finish"] = dataclasses.field(default="finish", init=False)
    finish_reason: FinishReason | None = None
    message_metadata: Any | None = None


@dataclasses.dataclass
class ErrorPart:
    """The error parts are appended to the message as they are received."""

    error_text: str
    type: Literal["error"] = dataclasses.field(default="error", init=False)


UIMessageStreamPart = (
    StartPart
    | TextStartPart
    | TextDeltaPart
    | TextEndPart
    | ToolInputStartPart
    | ToolInputAvailablePart
    | ToolOutputAvailablePart
    | StartStepPart
    | FinishStepPart
    | FinishPart
    | ErrorPart
)



# utils for serialization

def _to_camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def _generate_id(prefix: str = "id") -> str:
    """Generate a unique ID with prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def serialize_part(part: UIMessageStreamPart) -> str:
    """Serialize a stream part to JSON with camelCase keys."""
    d = dataclasses.asdict(part)
    camel_dict = {_to_camel_case(k): v for k, v in d.items() if v is not None}
    return json.dumps(camel_dict)


def format_sse(part: UIMessageStreamPart) -> str:
    """Format a stream part as an SSE data line."""
    return f"data: {serialize_part(part)}\n\n"


async def to_ui_message_stream(
    messages: AsyncGenerator[ai.Message, None],
) -> AsyncGenerator[UIMessageStreamPart, None]:
    """
    Convert a proto_sdk message stream into AI SDK UI message stream parts.

    This adapter transforms the internal message format into the AI SDK
    protocol that can be consumed by useChat and other AI SDK UI hooks.
    """
    # Track state for proper event sequencing
    current_text_id: str | None = None
    current_label: str | None = None
    emitted_start: bool = False
    in_step: bool = False

    async for msg in messages:
        # Emit start part on first message or label change (new agent)
        if not emitted_start or (msg.label and msg.label != current_label):
            if in_step:
                yield FinishStepPart()
                in_step = False
            if emitted_start:
                yield FinishPart(finish_reason="stop")

            yield StartPart(message_id=msg.id)
            yield StartStepPart()
            emitted_start = True
            in_step = True
            current_label = msg.label
            current_text_id = None

        # Handle text streaming (deltas)
        if msg.text_delta:
            if not current_text_id:
                current_text_id = _generate_id("text")
                yield TextStartPart(id=current_text_id)

            yield TextDeltaPart(id=current_text_id, delta=msg.text_delta)

        # Handle completed messages
        if msg.is_done:
            # Close any open text block
            if current_text_id:
                yield TextEndPart(id=current_text_id)
                current_text_id = None

            # Emit tool-related parts
            has_tool_calls = False
            for part in msg.parts:
                if isinstance(part, ai.ToolCallPart):
                    has_tool_calls = True
                    yield ToolInputStartPart(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                    )
                    yield ToolInputAvailablePart(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                        input=part.tool_args,
                    )

            # Handle tool results
            if msg.role == "tool":
                for part in msg.parts:
                    if isinstance(part, ai.ToolResultPart):
                        yield ToolOutputAvailablePart(
                            tool_call_id=part.tool_call_id,
                            output=part.result,
                        )

            # Finish step if we had tool calls (will continue with tool execution)
            if has_tool_calls:
                yield FinishStepPart()
                yield FinishPart(finish_reason="tool-calls")
                in_step = False
                emitted_start = False

    # Final cleanup
    if current_text_id:
        yield TextEndPart(id=current_text_id)
    if in_step:
        yield FinishStepPart()
    if emitted_start:
        yield FinishPart(finish_reason="stop")


async def to_sse_stream(
    messages: AsyncGenerator[ai.Message, None],
) -> AsyncGenerator[str, None]:
    """
    Convert a proto_sdk message stream directly into SSE-formatted strings.

    Use this with streaming HTTP responses (e.g., FastAPI StreamingResponse).

    Example:
        from fastapi.responses import StreamingResponse
        from proto_sdk.ui import ai_sdk

        @app.post("/chat")
        async def chat():
            return StreamingResponse(
                ai_sdk.to_sse_stream(ai.execute(my_agent, llm, query)),
                media_type="text/event-stream",
                headers=ai_sdk.UI_MESSAGE_STREAM_HEADERS,
            )
    """
    async for part in to_ui_message_stream(messages):
        yield format_sse(part)

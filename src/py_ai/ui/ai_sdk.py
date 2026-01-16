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
class ReasoningStartPart:
    """Indicates the beginning of a reasoning block."""

    id: str
    type: Literal["reasoning-start"] = dataclasses.field(
        default="reasoning-start", init=False
    )
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class ReasoningDeltaPart:
    """Contains incremental reasoning content for the reasoning block."""

    id: str
    delta: str
    type: Literal["reasoning-delta"] = dataclasses.field(
        default="reasoning-delta", init=False
    )
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class ReasoningEndPart:
    """Indicates the completion of a reasoning block."""

    id: str
    type: Literal["reasoning-end"] = dataclasses.field(
        default="reasoning-end", init=False
    )
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class SourceUrlPart:
    """References to external URLs."""

    source_id: str
    url: str
    type: Literal["source-url"] = dataclasses.field(default="source-url", init=False)
    title: str | None = None
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class SourceDocumentPart:
    """References to documents or files."""

    source_id: str
    media_type: str
    title: str
    type: Literal["source-document"] = dataclasses.field(
        default="source-document", init=False
    )
    filename: str | None = None
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class FilePart:
    """The file parts contain references to files with their media type."""

    url: str
    media_type: str
    type: Literal["file"] = dataclasses.field(default="file", init=False)
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class DataPart:
    """Custom data parts allow streaming of arbitrary structured data with type-specific handling.

    The type will be formatted as 'data-{data_type}' in the output.
    """

    data_type: str
    data: Any
    id: str | None = None
    transient: bool | None = None


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
class ToolInputDeltaPart:
    """Incremental chunks of tool input as it's being generated."""

    tool_call_id: str
    input_text_delta: str
    type: Literal["tool-input-delta"] = dataclasses.field(
        default="tool-input-delta", init=False
    )


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
class ToolInputErrorPart:
    """Indicates an error occurred during tool input processing."""

    tool_call_id: str
    tool_name: str
    input: Any
    error_text: str
    type: Literal["tool-input-error"] = dataclasses.field(
        default="tool-input-error", init=False
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
class ToolOutputErrorPart:
    """Indicates an error occurred during tool execution."""

    tool_call_id: str
    error_text: str
    type: Literal["tool-output-error"] = dataclasses.field(
        default="tool-output-error", init=False
    )
    provider_executed: bool | None = None
    dynamic: bool | None = None


@dataclasses.dataclass
class ToolOutputDeniedPart:
    """Indicates tool execution was denied."""

    tool_call_id: str
    type: Literal["tool-output-denied"] = dataclasses.field(
        default="tool-output-denied", init=False
    )


@dataclasses.dataclass
class ToolApprovalRequestPart:
    """Requests approval for tool execution."""

    approval_id: str
    tool_call_id: str
    type: Literal["tool-approval-request"] = dataclasses.field(
        default="tool-approval-request", init=False
    )


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
class AbortPart:
    """Indicates the message was aborted."""

    type: Literal["abort"] = dataclasses.field(default="abort", init=False)


@dataclasses.dataclass
class MessageMetadataPart:
    """Contains message metadata."""

    message_metadata: Any
    type: Literal["message-metadata"] = dataclasses.field(
        default="message-metadata", init=False
    )


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
    | ReasoningStartPart
    | ReasoningDeltaPart
    | ReasoningEndPart
    | SourceUrlPart
    | SourceDocumentPart
    | FilePart
    | DataPart
    | ToolInputStartPart
    | ToolInputDeltaPart
    | ToolInputAvailablePart
    | ToolInputErrorPart
    | ToolOutputAvailablePart
    | ToolOutputErrorPart
    | ToolOutputDeniedPart
    | ToolApprovalRequestPart
    | StartStepPart
    | FinishStepPart
    | FinishPart
    | AbortPart
    | MessageMetadataPart
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
    current_reasoning_id: str | None = None
    current_label: str | None = None
    emitted_start: bool = False
    in_step: bool = False
    started_tool_calls: set[str] = set()  # track which tool calls we've started

    async for msg in messages:
        # Emit start part on first message or label change (new agent)
        if not emitted_start or (msg.label and msg.label != current_label):
            # Close any open blocks before switching
            if current_reasoning_id:
                yield ReasoningEndPart(id=current_reasoning_id)
                current_reasoning_id = None
            if current_text_id:
                yield TextEndPart(id=current_text_id)
                current_text_id = None
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
            started_tool_calls = set()

        # Handle reasoning streaming (deltas) - reasoning comes before text
        if msg.reasoning_delta:
            if not current_reasoning_id:
                current_reasoning_id = _generate_id("reasoning")
                yield ReasoningStartPart(id=current_reasoning_id)

            yield ReasoningDeltaPart(id=current_reasoning_id, delta=msg.reasoning_delta)

        # Handle text streaming (deltas)
        if msg.text_delta:
            # Close reasoning block when text starts (reasoning precedes text)
            if current_reasoning_id:
                yield ReasoningEndPart(id=current_reasoning_id)
                current_reasoning_id = None

            if not current_text_id:
                current_text_id = _generate_id("text")
                yield TextStartPart(id=current_text_id)

            yield TextDeltaPart(id=current_text_id, delta=msg.text_delta)

        # Handle streaming tool call arguments
        for delta in msg.tool_call_deltas:
            if delta.tool_call_id not in started_tool_calls:
                started_tool_calls.add(delta.tool_call_id)
                yield ToolInputStartPart(
                    tool_call_id=delta.tool_call_id,
                    tool_name=delta.tool_name,
                )
            yield ToolInputDeltaPart(
                tool_call_id=delta.tool_call_id,
                input_text_delta=delta.args_delta,
            )

        # Handle completed messages
        if msg.is_done:
            # Close any open reasoning block
            if current_reasoning_id:
                yield ReasoningEndPart(id=current_reasoning_id)
                current_reasoning_id = None

            # Close any open text block
            if current_text_id:
                yield TextEndPart(id=current_text_id)
                current_text_id = None

            # Emit tool-related parts
            has_tool_calls = False
            for part in msg.parts:
                if isinstance(part, ai.ToolCallPart):
                    has_tool_calls = True
                    # Emit start if we haven't seen this tool call streaming
                    if part.tool_call_id not in started_tool_calls:
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
    if current_reasoning_id:
        yield ReasoningEndPart(id=current_reasoning_id)
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

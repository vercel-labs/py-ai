"""
Reference: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal

import pydantic

from .. import core

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
    messages: AsyncGenerator[core.messages.Message, None],
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
    emitted_tool_results: set[str] = set()  # track which tool results we've emitted

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
            emitted_tool_results = set()

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
        for delta in msg.tool_deltas:
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

            # Emit tool-related parts (unified model: ToolPart contains both call and result)
            has_pending_tool_calls = False
            for part in msg.parts:
                if isinstance(part, core.messages.ToolPart):
                    if part.status == "pending":
                        has_pending_tool_calls = True
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
                    elif part.status == "result":
                        # Tool result - emit output if we haven't already
                        if part.tool_call_id not in emitted_tool_results:
                            emitted_tool_results.add(part.tool_call_id)
                            yield ToolOutputAvailablePart(
                                tool_call_id=part.tool_call_id,
                                output=part.result,
                            )

            # Finish step if we had pending tool calls (will continue with tool execution)
            if has_pending_tool_calls:
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
    messages: AsyncGenerator[core.messages.Message, None],
) -> AsyncGenerator[str, None]:
    """Convert a proto_sdk message stream directly into SSE-formatted strings."""
    async for part in to_ui_message_stream(messages):
        yield format_sse(part)


# ============================================================================
# UI Message → Internal Message Conversion
# ============================================================================
#
# Reference: https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message
#
# Pydantic models for parsing AI SDK v6 UI messages. These can be used directly
# with FastAPI for automatic request body parsing.
#
# AI SDK v6 uses a `parts` array instead of legacy `content` string.


class UITextPart(pydantic.BaseModel):
    """Text content part in AI SDK v6 format."""

    type: Literal["text"]
    text: str


class UIReasoningPart(pydantic.BaseModel):
    """Reasoning/thinking content part in AI SDK v6 format."""

    type: Literal["reasoning"]
    reasoning: str


# Tool invocation states in AI SDK v6:
# - "input-streaming": Tool arguments are being streamed
# - "input-available": Tool arguments are complete, ready for execution
# - "output-available": Tool has been executed, result is available
# - "output-error": Tool execution failed
UIToolInvocationState = Literal[
    "input-streaming", "input-available", "output-available", "output-error"
]


class UIToolInvocationPart(pydantic.BaseModel):
    """Tool invocation part in AI SDK v6 format.

    Reference: https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message
    """

    model_config = pydantic.ConfigDict(populate_by_name=True)

    type: Literal["tool-invocation"]
    tool_invocation_id: str = pydantic.Field(alias="toolInvocationId")
    tool_name: str = pydantic.Field(alias="toolName")
    args: dict[str, Any] = pydantic.Field(default_factory=dict)
    state: UIToolInvocationState = "input-available"
    result: Any | None = None


# Union of all supported part types
UIMessagePart = UITextPart | UIReasoningPart | UIToolInvocationPart


class UIMessage(pydantic.BaseModel):
    """Message in AI SDK v6 format.

    Reference: https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message
    """

    model_config = pydantic.ConfigDict(populate_by_name=True)

    id: str = pydantic.Field(default_factory=lambda: _generate_id("msg"))
    role: Literal["user", "assistant", "system"]
    parts: list[UIMessagePart] = pydantic.Field(default_factory=list)


def to_messages(ui_messages: list[UIMessage]) -> list[core.messages.Message]:
    """Convert AI SDK v6 UI messages to internal Message format.

    Args:
        ui_messages: List of UIMessage objects from the AI SDK v6 frontend.

    Returns:
        List of internal Message objects ready for use with the runtime.
    """
    result: list[core.messages.Message] = []

    for ui_msg in ui_messages:
        internal_parts: list[core.messages.Part] = []

        for part in ui_msg.parts:
            if isinstance(part, UITextPart):
                internal_parts.append(core.messages.TextPart(text=part.text))

            elif isinstance(part, UIReasoningPart):
                internal_parts.append(
                    core.messages.ReasoningPart(reasoning=part.reasoning)
                )

            elif isinstance(part, UIToolInvocationPart):
                # Convert args dict to JSON string (internal format)
                tool_args = json.dumps(part.args) if part.args else "{}"

                # Map AI SDK v6 states to internal status
                status: Literal["pending", "result"] = "pending"
                if part.state in ("output-available", "output-error"):
                    status = "result"

                internal_parts.append(
                    core.messages.ToolPart(
                        tool_call_id=part.tool_invocation_id,
                        tool_name=part.tool_name,
                        tool_args=tool_args,
                        status=status,
                        result=part.result,
                    )
                )

        # Validate user/system messages have content - OpenAI requires it for these roles.
        # Assistant messages can have empty content if they have tool calls.
        if ui_msg.role in ("user", "system") and not internal_parts:
            raise ValueError(
                f"Message '{ui_msg.id}' has role '{ui_msg.role}' but no content. "
                "User and system messages require non-empty content."
            )

        result.append(
            core.messages.Message(
                id=ui_msg.id,
                role=ui_msg.role,
                parts=internal_parts,
                is_done=True,
            )
        )

    return result

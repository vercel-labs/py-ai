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
from . import protocol


# utils for serialization


def _to_camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def _generate_id(prefix: str = "id") -> str:
    """Generate a unique ID with prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def serialize_part(part: protocol.UIMessageStreamPart) -> str:
    """Serialize a stream part to JSON with camelCase keys."""
    d = dataclasses.asdict(part)
    camel_dict = {_to_camel_case(k): v for k, v in d.items() if v is not None}
    return json.dumps(camel_dict)


def format_sse(part: protocol.UIMessageStreamPart) -> str:
    """Format a stream part as an SSE data line."""
    return f"data: {serialize_part(part)}\n\n"


async def to_ui_message_stream(
    messages: AsyncGenerator[core.messages.Message, None],
) -> AsyncGenerator[protocol.UIMessageStreamPart, None]:
    """
    Convert a proto_sdk message stream into AI SDK UI message stream parts.

    This adapter transforms the internal message format into the AI SDK
    protocol that can be consumed by useChat and other AI SDK UI hooks.
    """
    # Track state for proper event sequencing
    current_text_id: str | None = None
    current_reasoning_id: str | None = None
    current_label: str | None = None
    current_message_id: str | None = None
    emitted_start: bool = False
    in_step: bool = False
    started_tool_calls: set[str] = set()  # track which tool calls we've started
    emitted_tool_results: set[str] = set()  # track which tool results we've emitted
    pending_tool_calls: set[str] = set()  # track tool calls waiting for results

    async for msg in messages:
        # Determine if we need to start a new step (new message ID means new step)
        is_new_message = current_message_id is not None and msg.id != current_message_id

        # Emit start part on first message or label change (new agent)
        if not emitted_start or (msg.label and msg.label != current_label):
            # Close any open blocks before switching
            if current_reasoning_id:
                yield protocol.ReasoningEndPart(id=current_reasoning_id)
                current_reasoning_id = None
            if current_text_id:
                yield protocol.TextEndPart(id=current_text_id)
                current_text_id = None
            if in_step:
                yield protocol.FinishStepPart()
                in_step = False
            if emitted_start:
                yield protocol.FinishPart(finish_reason="stop")

            yield protocol.StartPart(message_id=msg.id)
            yield protocol.StartStepPart()
            emitted_start = True
            in_step = True
            current_label = msg.label
            current_message_id = msg.id
            started_tool_calls = set()
            emitted_tool_results = set()
            pending_tool_calls = set()
        elif is_new_message:
            # New message ID within the same stream = new step
            # Close any open blocks
            if current_reasoning_id:
                yield protocol.ReasoningEndPart(id=current_reasoning_id)
                current_reasoning_id = None
            if current_text_id:
                yield protocol.TextEndPart(id=current_text_id)
                current_text_id = None
            if in_step:
                yield protocol.FinishStepPart()
            yield protocol.StartStepPart()
            in_step = True
            current_message_id = msg.id

        # Handle reasoning streaming (deltas) - reasoning comes before text
        if msg.reasoning_delta:
            if not current_reasoning_id:
                current_reasoning_id = _generate_id("reasoning")
                yield protocol.ReasoningStartPart(id=current_reasoning_id)

            yield protocol.ReasoningDeltaPart(
                id=current_reasoning_id, delta=msg.reasoning_delta
            )

        # Handle text streaming (deltas)
        if msg.text_delta:
            # Close reasoning block when text starts (reasoning precedes text)
            if current_reasoning_id:
                yield protocol.ReasoningEndPart(id=current_reasoning_id)
                current_reasoning_id = None

            if not current_text_id:
                current_text_id = _generate_id("text")
                yield protocol.TextStartPart(id=current_text_id)

            yield protocol.TextDeltaPart(id=current_text_id, delta=msg.text_delta)

        # Handle streaming tool call arguments
        for delta in msg.tool_deltas:
            if delta.tool_call_id not in started_tool_calls:
                started_tool_calls.add(delta.tool_call_id)
                yield protocol.ToolInputStartPart(
                    tool_call_id=delta.tool_call_id,
                    tool_name=delta.tool_name,
                )
            yield protocol.ToolInputDeltaPart(
                tool_call_id=delta.tool_call_id,
                input_text_delta=delta.args_delta,
            )

        # Handle completed messages
        if msg.is_done:
            # Track if we had an active text block before closing
            had_active_text = current_text_id is not None

            # Close any open reasoning block
            if current_reasoning_id:
                yield protocol.ReasoningEndPart(id=current_reasoning_id)
                current_reasoning_id = None

            # Close any open text block
            if current_text_id:
                yield protocol.TextEndPart(id=current_text_id)
                current_text_id = None

            # Collect tool parts for processing
            has_new_pending_tools = False
            has_new_tool_results = False

            for part in msg.parts:
                if isinstance(part, core.messages.ToolPart):
                    if (
                        part.status == "pending"
                        and part.tool_call_id not in pending_tool_calls
                    ):
                        has_new_pending_tools = True
                    elif (
                        part.status == "result"
                        and part.tool_call_id not in emitted_tool_results
                    ):
                        has_new_tool_results = True

            # Process parts in two passes:
            # 1. First handle text and pending tools
            # 2. Then handle tool results (which may need their own step)

            # Pass 1: Text and pending tool inputs
            for part in msg.parts:
                if isinstance(part, core.messages.TextPart):
                    # For text parts that weren't streamed (no active text block),
                    # AND this message doesn't have new pending tool calls or results,
                    # emit text-start and text-end
                    if (
                        part.text
                        and not had_active_text
                        and not has_new_pending_tools
                        and not has_new_tool_results
                    ):
                        text_id = _generate_id("text")
                        yield protocol.TextStartPart(id=text_id)
                        yield protocol.TextEndPart(id=text_id)
                elif isinstance(part, core.messages.ToolPart):
                    if part.status == "pending":
                        # Emit start if we haven't seen this tool call streaming
                        if part.tool_call_id not in started_tool_calls:
                            started_tool_calls.add(part.tool_call_id)
                            yield protocol.ToolInputStartPart(
                                tool_call_id=part.tool_call_id,
                                tool_name=part.tool_name,
                            )
                        if part.tool_call_id not in pending_tool_calls:
                            pending_tool_calls.add(part.tool_call_id)
                            yield protocol.ToolInputAvailablePart(
                                tool_call_id=part.tool_call_id,
                                tool_name=part.tool_name,
                                input=part.tool_args,
                            )

            # Pass 2: Tool results (same step as tool input per AI SDK protocol)
            # Tool input and output are part of the same "step" (one LLM turn)
            if has_new_tool_results:
                for part in msg.parts:
                    if (
                        isinstance(part, core.messages.ToolPart)
                        and part.status == "result"
                    ):
                        if part.tool_call_id not in emitted_tool_results:
                            emitted_tool_results.add(part.tool_call_id)
                            pending_tool_calls.discard(part.tool_call_id)
                            yield protocol.ToolOutputAvailablePart(
                                tool_call_id=part.tool_call_id,
                                output=part.result,
                            )

    # Final cleanup
    if current_reasoning_id:
        yield protocol.ReasoningEndPart(id=current_reasoning_id)
    if current_text_id:
        yield protocol.TextEndPart(id=current_text_id)
    if in_step:
        yield protocol.FinishStepPart()
    if emitted_start:
        yield protocol.FinishPart(finish_reason="stop")


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
# - "approval-requested": Tool requires user approval (TODO: approval workflow)
# - "approval-responded": User has responded to approval (TODO: approval workflow)
# - "output-available": Tool has been executed, result is available
# - "output-error": Tool execution failed
# - "output-denied": Tool execution was denied by user (TODO: approval workflow)
UIToolInvocationState = Literal[
    "input-streaming",
    "input-available",
    "approval-requested",
    "approval-responded",
    "output-available",
    "output-error",
    "output-denied",
]


class UIToolInvocationPart(pydantic.BaseModel):
    """Tool invocation part in AI SDK v6 format (legacy type: "tool-invocation").

    Note: The AI SDK frontend typically sends tool-{toolName} format instead.
    This model is kept for backwards compatibility.

    Reference: https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message
    """

    model_config = pydantic.ConfigDict(populate_by_name=True)

    type: Literal["tool-invocation"]
    tool_invocation_id: str = pydantic.Field(alias="toolInvocationId")
    tool_name: str = pydantic.Field(alias="toolName")
    args: dict[str, Any] = pydantic.Field(default_factory=dict)
    state: UIToolInvocationState = "input-available"
    result: Any | None = None


class UIStepStartPart(pydantic.BaseModel):
    """Step boundary marker. Skipped during conversion to internal format."""

    type: Literal["step-start"]


class UIToolPart(pydantic.BaseModel):
    """Tool part with dynamic type pattern: tool-{toolName}.

    The AI SDK frontend sends tool parts with type like "tool-get_weather"
    where the tool name is embedded in the type string.
    """

    model_config = pydantic.ConfigDict(populate_by_name=True)

    # The actual type string (e.g., "tool-talk_to_mothership")
    # We store this to extract the tool name
    type: str
    tool_call_id: str = pydantic.Field(alias="toolCallId")
    state: UIToolInvocationState
    input: str | dict[str, Any] | None = None  # JSON string or parsed dict
    output: Any | None = None
    error_text: str | None = pydantic.Field(default=None, alias="errorText")
    # TODO: title, providerExecuted, preliminary fields
    # TODO: approval workflow (approval object)

    @property
    def tool_name(self) -> str:
        """Extract tool name from the type string (e.g., 'tool-get_weather' -> 'get_weather')."""
        if self.type.startswith("tool-"):
            return self.type[5:]
        return self.type


class UIFilePart(pydantic.BaseModel):
    """File part. TODO: FilePart not yet supported in core messages."""

    model_config = pydantic.ConfigDict(populate_by_name=True)

    type: Literal["file"]
    media_type: str = pydantic.Field(alias="mediaType")
    url: str
    filename: str | None = None


class UISourceUrlPart(pydantic.BaseModel):
    """Source URL part. TODO: SourceUrlPart not yet supported."""

    model_config = pydantic.ConfigDict(populate_by_name=True)

    type: Literal["source-url"]
    source_id: str = pydantic.Field(alias="sourceId")
    url: str
    title: str | None = None


class UISourceDocumentPart(pydantic.BaseModel):
    """Source document part. TODO: SourceDocumentPart not yet supported."""

    model_config = pydantic.ConfigDict(populate_by_name=True)

    type: Literal["source-document"]
    source_id: str = pydantic.Field(alias="sourceId")
    media_type: str = pydantic.Field(alias="mediaType")
    title: str
    filename: str | None = None


# Union of all supported part types (used for type hints)
UIMessagePart = (
    UITextPart
    | UIReasoningPart
    | UIToolInvocationPart
    | UIStepStartPart
    | UIToolPart
    | UIFilePart
    | UISourceUrlPart
    | UISourceDocumentPart
)


def _parse_ui_part(part_data: dict[str, Any]) -> UIMessagePart | None:
    """Parse a UI part dict, handling dynamic type patterns.

    Returns None for unsupported part types (they will be skipped).
    """
    part_type = part_data.get("type", "")

    if part_type == "text":
        return UITextPart.model_validate(part_data)
    elif part_type == "reasoning":
        return UIReasoningPart.model_validate(part_data)
    elif part_type == "tool-invocation":
        return UIToolInvocationPart.model_validate(part_data)
    elif part_type == "step-start":
        return UIStepStartPart.model_validate(part_data)
    elif part_type == "file":
        return UIFilePart.model_validate(part_data)
    elif part_type == "source-url":
        return UISourceUrlPart.model_validate(part_data)
    elif part_type == "source-document":
        return UISourceDocumentPart.model_validate(part_data)
    elif part_type.startswith("tool-"):
        # Dynamic tool type: tool-{toolName} (e.g., "tool-get_weather")
        return UIToolPart.model_validate(part_data)
    elif part_type.startswith("data-"):
        # TODO: data-{name} parts not yet supported
        return None
    elif part_type == "dynamic-tool":
        # TODO: dynamic-tool type not yet supported
        return None
    else:
        # Unknown part type - skip gracefully
        return None


class UIMessage(pydantic.BaseModel):
    """Message in AI SDK v6 format.

    Reference: https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message
    """

    model_config = pydantic.ConfigDict(populate_by_name=True)

    id: str = pydantic.Field(default_factory=lambda: _generate_id("msg"))
    role: Literal["user", "assistant", "system"]
    parts: list[UIMessagePart] = pydantic.Field(default_factory=list)

    @pydantic.field_validator("parts", mode="before")
    @classmethod
    def parse_parts(cls, v: list[dict[str, Any]]) -> list[UIMessagePart]:
        """Parse parts using custom logic to handle dynamic type patterns."""
        if not isinstance(v, list):
            return v
        result: list[UIMessagePart] = []
        for part_data in v:
            if isinstance(part_data, dict):
                parsed = _parse_ui_part(part_data)
                if parsed is not None:
                    result.append(parsed)
            else:
                # Already parsed (e.g., in tests)
                result.append(part_data)
        return result


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
                # Skip empty text parts (AI SDK sometimes sends empty text)
                if part.text:
                    internal_parts.append(core.messages.TextPart(text=part.text))

            elif isinstance(part, UIReasoningPart):
                internal_parts.append(core.messages.ReasoningPart(text=part.reasoning))

            elif isinstance(part, UIToolInvocationPart):
                # Legacy tool-invocation type
                # Convert args dict to JSON string (internal format)
                tool_args = json.dumps(part.args) if part.args else "{}"

                # Map AI SDK v6 states to internal status
                status: Literal["pending", "result"] = "pending"
                if part.state in ("output-available", "output-error", "output-denied"):
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

            elif isinstance(part, UIToolPart):
                # Dynamic tool-{toolName} type (e.g., "tool-get_weather")
                # Input can be a JSON string or already parsed dict
                if isinstance(part.input, str):
                    tool_args = part.input
                elif part.input is not None:
                    tool_args = json.dumps(part.input)
                else:
                    tool_args = "{}"

                # Map AI SDK v6 states to internal status
                status: Literal["pending", "result"] = "pending"
                if part.state in ("output-available", "output-error", "output-denied"):
                    status = "result"

                # The internal ToolPart.result expects dict | None, but AI SDK
                # output can be any type. Wrap non-dict results for compatibility.
                tool_result: dict[str, Any] | None = None
                if part.output is not None:
                    if isinstance(part.output, dict):
                        tool_result = part.output
                    else:
                        tool_result = {"value": part.output}

                internal_parts.append(
                    core.messages.ToolPart(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                        tool_args=tool_args,
                        status=status,
                        result=tool_result,
                    )
                )

            elif isinstance(part, UIStepStartPart):
                # Skip step-start boundary markers
                pass

            elif isinstance(part, (UIFilePart, UISourceUrlPart, UISourceDocumentPart)):
                # TODO: these part types not yet supported in core messages
                pass

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
            )
        )

    return result

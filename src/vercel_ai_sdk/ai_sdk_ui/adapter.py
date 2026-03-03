"""
Reference: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
"""

from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any, Literal

from .. import core
from ..core import hooks
from . import protocol, ui_message

logger = logging.getLogger(__name__)

# ============================================================================
# Serialization utilities
# ============================================================================


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
    if isinstance(part, protocol.DataPart):
        # DataPart's wire type is computed (``data-{data_type}``); replace
        # the raw ``data_type`` field with the protocol ``type`` key.
        d["type"] = part.type
        del d["data_type"]
    camel_dict = {_to_camel_case(k): v for k, v in d.items() if v is not None}
    return json.dumps(camel_dict)


def format_sse(part: protocol.UIMessageStreamPart) -> str:
    """Format a stream part as an SSE data line."""
    return f"data: {serialize_part(part)}\n\n"


# ============================================================================
# Internal Message → UI Message Stream Conversion
# ============================================================================


class _StreamState:
    """Tracks state for UI message stream event sequencing.

    Encapsulates the mutable state needed to properly sequence events
    (reasoning blocks, text blocks, steps, tool calls) when converting
    an internal message stream to the AI SDK UI protocol.
    """

    def __init__(self) -> None:
        self.text_id: str | None = None
        self.reasoning_id: str | None = None
        self.label: str | None = None
        self.message_id: str | None = None
        self.emitted_start: bool = False
        self.in_step: bool = False
        self.started_tool_calls: set[str] = set()
        self.emitted_tool_results: set[str] = set()
        self.pending_tool_calls: set[str] = set()
        self.emitted_approval_requests: set[str] = set()

    def close_open_blocks(self) -> list[protocol.UIMessageStreamPart]:
        """Close any open reasoning/text blocks, returning parts to emit."""
        parts: list[protocol.UIMessageStreamPart] = []
        if self.reasoning_id:
            parts.append(protocol.ReasoningEndPart(id=self.reasoning_id))
            self.reasoning_id = None
        if self.text_id:
            parts.append(protocol.TextEndPart(id=self.text_id))
            self.text_id = None
        return parts

    def finish_step(self) -> list[protocol.UIMessageStreamPart]:
        """Close open blocks and finish the current step if active."""
        parts = self.close_open_blocks()
        if self.in_step:
            parts.append(protocol.FinishStepPart())
            self.in_step = False
        return parts

    def reset_tool_tracking(self) -> None:
        """Reset tool tracking sets (for new message/agent boundaries)."""
        self.started_tool_calls = set()
        self.emitted_tool_results = set()
        self.pending_tool_calls = set()
        self.emitted_approval_requests = set()

    def begin_message(
        self, msg: core.messages.Message
    ) -> list[protocol.UIMessageStreamPart]:
        """Handle message/step boundaries, returning parts to emit.

        Decides whether to start a new message (first message or agent switch)
        or a new step (same stream, different message ID), closing any open
        blocks and steps as needed.
        """
        parts: list[protocol.UIMessageStreamPart] = []
        is_new_message = self.message_id is not None and msg.id != self.message_id

        if not self.emitted_start or (msg.label and msg.label != self.label):
            # First message or label change (new agent)
            parts.extend(self.finish_step())
            if self.emitted_start:
                parts.append(protocol.FinishPart(finish_reason="stop"))

            parts.append(protocol.StartPart(message_id=msg.id))
            parts.append(protocol.StartStepPart())
            self.emitted_start = True
            self.in_step = True
            self.label = msg.label
            self.message_id = msg.id
            self.reset_tool_tracking()
        elif is_new_message:
            # New message ID within the same stream = new step
            parts.extend(self.finish_step())
            parts.append(protocol.StartStepPart())
            self.in_step = True
            self.message_id = msg.id

        return parts


def _tool_call_id_from_approval_hook(
    hook_part: core.messages.HookPart,
) -> str | None:
    """Extract tool_call_id from a ToolApproval HookPart.

    Returns the tool_call_id if this is a ToolApproval hook whose hook_id
    follows the ``approve_{tool_call_id}`` convention, otherwise None.
    """
    if hook_part.hook_type != hooks.ToolApproval.hook_type:  # type: ignore[attr-defined]
        return None
    prefix = "approve_"
    if hook_part.hook_id.startswith(prefix):
        return hook_part.hook_id[len(prefix) :]
    return None


def _is_tool_approval_hook_message(msg: core.messages.Message) -> bool:
    """True if this message contains only ToolApproval HookParts."""
    if not msg.parts:
        return False
    return all(
        isinstance(p, core.messages.HookPart)
        and _tool_call_id_from_approval_hook(p) is not None
        for p in msg.parts
    )


async def to_ui_message_stream(
    messages: AsyncIterable[core.messages.Message],
) -> AsyncGenerator[protocol.UIMessageStreamPart]:
    """
    Convert a proto_sdk message stream into AI SDK UI message stream parts.

    This adapter transforms the internal message format into the AI SDK
    protocol that can be consumed by useChat and other AI SDK UI hooks.
    """
    state = _StreamState()

    async for msg in messages:
        # Tool-approval hook messages are emitted by the Runtime as
        # separate Message objects (with their own id).  To the frontend
        # they belong to the *same* step as the tool call, so we pin
        # the message id to avoid creating a spurious step boundary.
        if _is_tool_approval_hook_message(msg) and state.message_id:
            msg = msg.model_copy(update={"id": state.message_id})

        for part in state.begin_message(msg):
            yield part

        # Handle reasoning streaming (deltas) - reasoning comes before text
        if delta := msg.reasoning_delta:
            if not state.reasoning_id:
                state.reasoning_id = _generate_id("reasoning")
                yield protocol.ReasoningStartPart(id=state.reasoning_id)
            yield protocol.ReasoningDeltaPart(id=state.reasoning_id, delta=delta)

        # Handle text streaming (deltas)
        if delta := msg.text_delta:
            # Close reasoning block when text starts (reasoning precedes text)
            if state.reasoning_id:
                yield protocol.ReasoningEndPart(id=state.reasoning_id)
                state.reasoning_id = None

            if not state.text_id:
                state.text_id = _generate_id("text")
                yield protocol.TextStartPart(id=state.text_id)
            yield protocol.TextDeltaPart(id=state.text_id, delta=delta)

        # Handle streaming tool call arguments
        for tool_delta in msg.tool_deltas:
            if tool_delta.tool_call_id not in state.started_tool_calls:
                state.started_tool_calls.add(tool_delta.tool_call_id)
                yield protocol.ToolInputStartPart(
                    tool_call_id=tool_delta.tool_call_id,
                    tool_name=tool_delta.tool_name,
                )
            yield protocol.ToolInputDeltaPart(
                tool_call_id=tool_delta.tool_call_id,
                input_text_delta=tool_delta.args_delta,
            )

        # Handle completed messages
        if msg.is_done:
            had_active_text = state.text_id is not None
            for part in state.close_open_blocks():
                yield part

            # Scan tool parts for new pending/completed states
            has_new_pending_tools = False
            has_new_tool_results = False

            for msg_part in msg.parts:
                if isinstance(msg_part, core.messages.ToolPart):
                    if (
                        msg_part.status == "pending"
                        and msg_part.tool_call_id not in state.pending_tool_calls
                    ):
                        has_new_pending_tools = True
                    elif (
                        msg_part.status in ("result", "error")
                        and msg_part.tool_call_id not in state.emitted_tool_results
                    ):
                        has_new_tool_results = True

            # Process parts in two passes:
            # 1. First handle text and pending tools
            # 2. Then handle tool results (which may need their own step)

            # Pass 1: Text and pending tool inputs
            for msg_part in msg.parts:
                match msg_part:
                    case core.messages.TextPart(text=text) if (
                        text
                        and not had_active_text
                        and not has_new_pending_tools
                        and not has_new_tool_results
                    ):
                        text_id = _generate_id("text")
                        yield protocol.TextStartPart(id=text_id)
                        yield protocol.TextEndPart(id=text_id)
                    case core.messages.ToolPart(
                        status="pending",
                        tool_call_id=tc_id,
                        tool_name=name,
                        tool_args=args,
                    ):
                        if tc_id not in state.started_tool_calls:
                            state.started_tool_calls.add(tc_id)
                            yield protocol.ToolInputStartPart(
                                tool_call_id=tc_id,
                                tool_name=name,
                            )
                        if tc_id not in state.pending_tool_calls:
                            state.pending_tool_calls.add(tc_id)
                            yield protocol.ToolInputAvailablePart(
                                tool_call_id=tc_id,
                                tool_name=name,
                                input=args,
                            )

            # Pass 2: Tool outputs (same step as tool input per AI SDK protocol)
            # Tool input and output are part of the same "step" (one LLM turn)
            if has_new_tool_results:
                for msg_part in msg.parts:
                    match msg_part:
                        case core.messages.ToolPart(
                            tool_call_id=tc_id,
                            result=result,
                            status=status,
                        ) if (
                            status in ("result", "error")
                            and tc_id not in state.emitted_tool_results
                        ):
                            state.emitted_tool_results.add(tc_id)
                            state.pending_tool_calls.discard(tc_id)
                            yield protocol.ToolOutputAvailablePart(
                                tool_call_id=tc_id,
                                output=result,
                            )

            # Pass 3: Hook-based tool approvals
            for msg_part in msg.parts:
                if not isinstance(msg_part, core.messages.HookPart):
                    continue
                approval_tc_id = _tool_call_id_from_approval_hook(msg_part)
                if approval_tc_id is None:
                    continue

                if msg_part.status == "pending":
                    if approval_tc_id not in state.emitted_approval_requests:
                        state.emitted_approval_requests.add(approval_tc_id)
                        yield protocol.ToolApprovalRequestPart(
                            approval_id=msg_part.hook_id,
                            tool_call_id=approval_tc_id,
                        )
                elif msg_part.status == "resolved":
                    resolution = msg_part.resolution or {}
                    if not resolution.get("granted", False):
                        yield protocol.ToolOutputDeniedPart(
                            tool_call_id=approval_tc_id,
                        )
                elif msg_part.status == "cancelled":
                    yield protocol.ToolOutputErrorPart(
                        tool_call_id=approval_tc_id,
                        error_text="Hook cancelled",
                    )

    # Final cleanup
    for part in state.finish_step():
        yield part
    if state.emitted_start:
        yield protocol.FinishPart(finish_reason="stop")


async def filter_by_label(
    messages: AsyncIterable[core.messages.Message],
    label: str | None = None,
) -> AsyncGenerator[core.messages.Message]:
    """Filter a message stream to a single agent label.

    If label is provided, only messages with that label pass through.
    If label is None, auto-locks to whichever label arrives first.
    """
    async for msg in messages:
        if label is None:
            label = msg.label
        if msg.label == label:
            yield msg


async def to_sse_stream(
    messages: AsyncIterable[core.messages.Message],
) -> AsyncGenerator[str]:
    """Convert a proto_sdk message stream directly into SSE-formatted strings."""
    async for part in to_ui_message_stream(messages):
        yield format_sse(part)


# ============================================================================
# Tool conversion helpers
# ============================================================================

_TOOL_RESULT_STATES: frozenset[str] = frozenset({"output-available"})
_TOOL_ERROR_STATES: frozenset[str] = frozenset({"output-error", "output-denied"})


def _map_tool_status(
    state: ui_message.UIToolInvocationState,
) -> Literal["pending", "result", "error"]:
    """Map AI SDK v6 tool invocation state to internal status."""
    if state in _TOOL_ERROR_STATES:
        return "error"
    if state in _TOOL_RESULT_STATES:
        return "result"
    return "pending"


def _normalize_tool_args(tool_input: str | dict[str, Any] | None) -> str:
    """Normalize tool input (JSON string, dict, or None) to a JSON string."""
    match tool_input:
        case str():
            return tool_input
        case dict():
            return json.dumps(tool_input)
        case _:
            return "{}"


def _normalize_tool_result(output: Any) -> dict[str, Any] | None:
    """Normalize tool output to dict format for internal ToolPart.

    The internal ToolPart.result expects dict | None, but AI SDK
    output can be any type. Wrap non-dict results for compatibility.
    """
    if output is None:
        return None
    return output if isinstance(output, dict) else {"value": output}


def to_messages(
    ui_messages: list[ui_message.UIMessage],
) -> list[core.messages.Message]:
    """Convert AI SDK v6 UI messages to internal Message format.

    As a side-effect, tool parts in ``approval-responded`` state trigger
    ``ToolApproval.resolve()`` so the agent loop can resume execution
    without the caller needing to handle approval routing explicitly.

    When approvals are resolved, the trailing assistant message is
    automatically stripped.  The checkpoint will replay that step, so
    including it would send duplicate tool-use content to the LLM.

    Args:
        ui_messages: List of UIMessage objects from the AI SDK v6 frontend.

    Returns:
        List of internal Message objects ready for use with the runtime.
    """
    result: list[core.messages.Message] = []
    resolved_any_approval = False

    for ui_msg in ui_messages:
        internal_parts: list[core.messages.Part] = []

        for part in ui_msg.parts:
            match part:
                case ui_message.UITextPart(text=text) if text:
                    internal_parts.append(core.messages.TextPart(text=text))

                case ui_message.UIReasoningPart(reasoning=reasoning):
                    internal_parts.append(core.messages.ReasoningPart(text=reasoning))

                case ui_message.UIToolInvocationPart() as inv:
                    # Legacy tool-invocation type
                    internal_parts.append(
                        core.messages.ToolPart(
                            tool_call_id=inv.tool_invocation_id,
                            tool_name=inv.tool_name,
                            tool_args=json.dumps(inv.args) if inv.args else "{}",
                            status=_map_tool_status(inv.state),
                            result=inv.result,
                        )
                    )

                case ui_message.UIToolPart() as tp:
                    # Dynamic tool-{toolName} type (e.g., "tool-get_weather")
                    internal_parts.append(
                        core.messages.ToolPart(
                            tool_call_id=tp.tool_call_id,
                            tool_name=tp.tool_name,
                            tool_args=_normalize_tool_args(tp.input),
                            status=_map_tool_status(tp.state),
                            result=_normalize_tool_result(tp.output),
                        )
                    )
                    # Side-effect: resolve ToolApproval hooks from approval
                    # responses so the agent loop can resume execution.
                    if (
                        tp.state == "approval-responded"
                        and tp.approval is not None
                        and tp.approval.approved is not None
                    ):
                        hooks.ToolApproval.resolve(  # type: ignore[attr-defined]
                            tp.approval.id,
                            {
                                "granted": tp.approval.approved,
                                "reason": tp.approval.reason,
                            },
                        )
                        resolved_any_approval = True

                case (
                    ui_message.UIStepStartPart()
                    | ui_message.UIFilePart()
                    | ui_message.UISourceUrlPart()
                    | ui_message.UISourceDocumentPart()
                ):
                    pass  # Skip unsupported/boundary parts

        # Validate user/system messages have content - OpenAI requires it there.
        # Assistant messages can have empty content if they have tool calls.
        if ui_msg.role in ("user", "system") and not internal_parts:
            raise ValueError(
                f"Message '{ui_msg.id}' has role '{ui_msg.role}' but no content. "
                "User and system messages require non-empty content."
            )

        # The UI sends one assistant message per conversation turn, but a
        # single turn may span multiple stream_loop iterations (e.g.
        # [text, tool(done), text, tool(done), text]).  LLM APIs expect
        # one message per iteration, so split at completed-tool boundaries.
        if ui_msg.role == "assistant":
            result.extend(_split_assistant_parts(internal_parts, msg_id=ui_msg.id))
        else:
            result.append(
                core.messages.Message(
                    id=ui_msg.id,
                    role=ui_msg.role,
                    parts=internal_parts,
                )
            )

    # When resuming from a checkpoint (approvals were resolved above),
    # the frontend sends the full history including the assistant message
    # from the interrupted run.  The checkpoint replays that step, so
    # strip the trailing assistant message to avoid duplicate tool-use.
    if resolved_any_approval and result and result[-1].role == "assistant":
        logger.info(
            "Stripping trailing assistant message (checkpoint will replay this step)"
        )
        result = result[:-1]

    return result


def _split_assistant_parts(
    parts: list[core.messages.Part],
    msg_id: str,
) -> list[core.messages.Message]:
    """Split assistant parts at completed-tool → non-tool boundaries.

    Returns one ``Message`` per ``stream_loop`` iteration so that LLM
    adapters receive correctly-shaped single-iteration messages.
    """
    messages: list[core.messages.Message] = []
    current: list[core.messages.Part] = []
    has_completed_tool = False

    for part in parts:
        if has_completed_tool and not isinstance(part, core.messages.ToolPart):
            messages.append(
                core.messages.Message(role="assistant", parts=current, id=msg_id)
            )
            current = []
            has_completed_tool = False

        current.append(part)

        if isinstance(part, core.messages.ToolPart) and part.status in (
            "result",
            "error",
        ):
            has_completed_tool = True

    if current:
        messages.append(
            core.messages.Message(role="assistant", parts=current, id=msg_id)
        )

    return messages

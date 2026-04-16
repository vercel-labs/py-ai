"""
Inbound: UI -> internal message conversion.

Converts AI SDK v6 UIMessages into internal Message objects
for use with the runtime/agent loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, NamedTuple

from ...types import messages as messages_
from . import ui_message

logger = logging.getLogger(__name__)


_TOOL_RESULT_STATES: frozenset[str] = frozenset({"output-available"})
_TOOL_ERROR_STATES: frozenset[str] = frozenset({"output-error", "output-denied"})


def _is_tool_completed(state: ui_message.UIToolInvocationState) -> bool:
    """Return True if the tool invocation state indicates a completed tool."""
    return state in _TOOL_RESULT_STATES or state in _TOOL_ERROR_STATES


def _is_tool_error(state: ui_message.UIToolInvocationState) -> bool:
    """Return True if the tool invocation state indicates an error."""
    return state in _TOOL_ERROR_STATES


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
    """Normalize tool output to dict format for internal ToolResultPart.

    The internal ToolResultPart.result expects dict | None, but AI SDK
    output can be any type. Wrap non-dict results for compatibility.
    """
    if output is None:
        return None
    return output if isinstance(output, dict) else {"value": output}


def _error_result(error_text: str | None, output: Any) -> dict[str, Any] | None:
    """Normalize an error-state tool result."""
    normalized = _normalize_tool_result(output)
    if error_text:
        if normalized is None:
            return {"error": error_text}
        if isinstance(normalized, dict) and "error" not in normalized:
            return {"error": error_text, **normalized}
    return normalized


def _approval_signal_part(tp: ui_message.UIToolPart) -> messages_.HookPart | None:
    """Reconstruct approval signal state from a UI tool part when possible."""
    approval = tp.approval
    if approval is None:
        return None

    if tp.state == "approval-requested":
        return messages_.HookPart(
            hook_id=approval.id,
            hook_type="ToolApproval",
            status="pending",
        )

    if tp.state == "approval-responded" and approval.approved is not None:
        return messages_.HookPart(
            hook_id=approval.id,
            hook_type="ToolApproval",
            status="resolved",
            resolution={
                "granted": approval.approved,
                "reason": approval.reason,
            },
        )

    if tp.state == "output-denied":
        return messages_.HookPart(
            hook_id=approval.id,
            hook_type="ToolApproval",
            status="resolved",
            resolution={
                "granted": False,
                "reason": approval.reason,
            },
        )

    return None


# ============================================================================
# Approval extraction
# ============================================================================


class ApprovalResponse(NamedTuple):
    """Extracted approval response from a UIToolPart in approval-responded state."""

    hook_id: str
    granted: bool
    reason: str | None


def extract_approvals(
    ui_messages: list[ui_message.UIMessage],
) -> list[ApprovalResponse]:
    """Extract approval responses from UI messages.

    Walks UIMessages looking for UIToolParts in ``approval-responded`` state
    and returns the approval data as a list. Pure function -- does not
    resolve hooks or trigger any side-effects.

    Args:
        ui_messages: List of UIMessage objects from the AI SDK v6 frontend.

    Returns:
        List of ApprovalResponse tuples with hook_id, granted, and reason.
    """
    approvals: list[ApprovalResponse] = []
    for ui_msg in ui_messages:
        for part in ui_msg.parts:
            if not isinstance(part, ui_message.UIToolPart):
                continue
            if (
                part.state == "approval-responded"
                and part.approval is not None
                and part.approval.approved is not None
            ):
                approvals.append(
                    ApprovalResponse(
                        hook_id=part.approval.id,
                        granted=part.approval.approved,
                        reason=part.approval.reason,
                    )
                )
    return approvals


# ============================================================================
# UI message normalization (heal stale tool states)
# ============================================================================


def normalize_ui_messages(
    ui_messages: list[ui_message.UIMessage],
) -> list[ui_message.UIMessage]:
    """Heal stale tool-part states from previously persisted assistant history.

    Tool parts may be stored in transient states (e.g. ``"call"``) if the
    stream was interrupted. This normalizes them to consistent terminal
    states based on what data is actually present.
    """
    normalized: list[ui_message.UIMessage] = []
    for message in ui_messages:
        new_parts = []
        changed = False
        for part in message.parts:
            part_type = getattr(part, "type", None)
            state = getattr(part, "state", None)
            if isinstance(part_type, str) and part_type.startswith("tool-"):
                output = getattr(part, "output", None)
                approval = getattr(part, "approval", None)
                approved = approval.approved if approval is not None else None
                error_text = getattr(part, "error_text", None)

                next_state = state
                if output is not None:
                    if state == "output-error" or error_text is not None:
                        next_state = "output-error"
                    elif state == "output-denied" or approved is False:
                        next_state = "output-denied"
                    else:
                        next_state = "output-available"
                elif state == "call":
                    next_state = "input-available"

                if next_state != state:
                    part = part.model_copy(update={"state": next_state})
                    changed = True

            new_parts.append(part)

        normalized.append(
            message.model_copy(update={"parts": new_parts}) if changed else message
        )
    return normalized


# ============================================================================
# UI -> internal message conversion
# ============================================================================


def ui_to_messages(
    ui_messages: list[ui_message.UIMessage],
) -> list[messages_.Message]:
    """Convert AI SDK v6 UI messages to internal Message format.

    This is a pure data transformation. It does not resolve hooks or
    trigger any side-effects. Use ``extract_approvals()`` separately
    to obtain approval responses for hook resolution.

    When the last message is an assistant message that contains
    approval-responded tool parts, it is automatically stripped to
    avoid sending duplicate tool-use content to the LLM on re-entry.
    The caller should check ``extract_approvals()`` to determine
    whether this stripping occurred.

    Args:
        ui_messages: List of UIMessage objects from the AI SDK v6 frontend.

    Returns:
        List of internal Message objects ready for use with the runtime.
    """
    result: list[messages_.Message] = []
    has_approval_responses = False

    for ui_msg in ui_messages:
        assistant_parts: list[messages_.Part] = []
        tool_result_parts: list[messages_.ToolResultPart] = []
        signal_parts: list[messages_.HookPart] = []

        for part in ui_msg.parts:
            match part:
                case ui_message.UITextPart(text=text) if text:
                    assistant_parts.append(messages_.TextPart(text=text))

                case ui_message.UIReasoningPart(reasoning=reasoning):
                    assistant_parts.append(messages_.ReasoningPart(text=reasoning))

                case ui_message.UIToolInvocationPart() as inv:
                    # Legacy tool-invocation type
                    tool_args = json.dumps(inv.args) if inv.args else "{}"
                    assistant_parts.append(
                        messages_.ToolCallPart(
                            tool_call_id=inv.tool_invocation_id,
                            tool_name=inv.tool_name,
                            tool_args=tool_args,
                        )
                    )
                    if _is_tool_completed(inv.state):
                        tool_result_parts.append(
                            messages_.ToolResultPart(
                                tool_call_id=inv.tool_invocation_id,
                                tool_name=inv.tool_name,
                                result=inv.result,
                                is_error=_is_tool_error(inv.state),
                            )
                        )

                case ui_message.UIToolPart() as tp:
                    # Dynamic tool-{toolName} type (e.g., "tool-get_weather")
                    assistant_parts.append(
                        messages_.ToolCallPart(
                            tool_call_id=tp.tool_call_id,
                            tool_name=tp.tool_name,
                            tool_args=_normalize_tool_args(tp.input),
                        )
                    )
                    approval_signal = _approval_signal_part(tp)
                    if approval_signal is not None:
                        signal_parts.append(approval_signal)

                    if tp.state in _TOOL_RESULT_STATES:
                        tool_result_parts.append(
                            messages_.ToolResultPart(
                                tool_call_id=tp.tool_call_id,
                                tool_name=tp.tool_name,
                                result=_normalize_tool_result(tp.output),
                                is_error=False,
                            )
                        )
                    elif tp.state == "output-error":
                        tool_result_parts.append(
                            messages_.ToolResultPart(
                                tool_call_id=tp.tool_call_id,
                                tool_name=tp.tool_name,
                                result=_error_result(tp.error_text, tp.output),
                                is_error=True,
                            )
                        )
                    if tp.state == "approval-responded":
                        has_approval_responses = True

                case ui_message.UIFilePart() as fp:
                    assistant_parts.append(
                        messages_.FilePart(
                            data=fp.url,
                            media_type=fp.media_type,
                            filename=fp.filename,
                        )
                    )

                case (
                    ui_message.UIStepStartPart()
                    | ui_message.UISourceUrlPart()
                    | ui_message.UISourceDocumentPart()
                ):
                    pass  # Skip unsupported/boundary parts

        # Validate user/system messages have content - OpenAI requires it.
        if ui_msg.role in ("user", "system") and not assistant_parts:
            raise ValueError(
                f"Message '{ui_msg.id}' has role '{ui_msg.role}' but no content. "
                "User and system messages require non-empty content."
            )

        # The UI sends one assistant message per conversation turn, but a
        # single turn may span multiple loop iterations (e.g.
        # [text, tool_call, tool_result, text, tool_call, tool_result, text]).
        # LLM APIs expect one message per iteration, so split into
        # assistant + tool message pairs at tool-result boundaries.
        if ui_msg.role == "assistant":
            split_messages = _split_assistant_parts(
                assistant_parts, tool_result_parts, msg_id=ui_msg.id
            )
            result.extend(split_messages)
            if signal_parts:
                result.extend(
                    messages_.Message(
                        id=ui_msg.id,
                        role="signal",
                        parts=[part],
                    )
                    for part in signal_parts
                )
        else:
            result.append(
                messages_.Message(
                    id=ui_msg.id,
                    role=ui_msg.role,
                    parts=assistant_parts,
                )
            )

    # When resuming after approval responses, the frontend sends the full
    # history including the assistant message from the interrupted run.
    # Strip it to avoid sending duplicate tool-use content to the LLM.
    if has_approval_responses and result and result[-1].role == "assistant":
        logger.info("Stripping trailing assistant message (approval responses present)")
        result = result[:-1]

    return result


def _split_assistant_parts(
    parts: list[messages_.Part],
    tool_results: list[messages_.ToolResultPart],
    msg_id: str,
) -> list[messages_.Message]:
    """Split assistant parts into assistant + tool message pairs.

    The UI sends one big assistant message per turn, but internally each
    loop iteration produces an assistant message (with tool calls) followed
    by a tool message (with results).  This reconstructs that structure.

    Returns a list of Messages: alternating assistant and tool messages,
    split at tool-call boundaries when results are available.
    """
    # Index tool results by their tool_call_id for lookup
    results_by_id = {tr.tool_call_id: tr for tr in tool_results}

    messages: list[messages_.Message] = []
    current: list[messages_.Part] = []
    pending_results: list[messages_.ToolResultPart] = []

    for part in parts:
        current.append(part)

        # When we see a ToolCallPart that has a result, accumulate it
        if (
            isinstance(part, messages_.ToolCallPart)
            and part.tool_call_id in results_by_id
        ):
            pending_results.append(results_by_id[part.tool_call_id])

    # If there are pending results and more parts follow, we need to split.
    # Walk again, splitting at boundaries where all accumulated tool calls
    # have results and a non-tool part follows.
    if not pending_results:
        # No completed tools -- single assistant message
        if current:
            messages.append(
                messages_.Message(role="assistant", parts=current, id=msg_id)
            )
        return messages

    # Re-walk to split at tool-call boundaries
    messages = []
    current = []
    current_results: list[messages_.ToolResultPart] = []
    seen_tool_call = False

    for part in parts:
        # If we had a completed tool call group and now see a non-tool part,
        # split here
        if (
            seen_tool_call
            and current_results
            and not isinstance(part, messages_.ToolCallPart)
        ):
            messages.append(
                messages_.Message(role="assistant", parts=current, id=msg_id)
            )
            messages.append(messages_.Message(role="tool", parts=list(current_results)))
            current = []
            current_results = []
            seen_tool_call = False

        current.append(part)

        if isinstance(part, messages_.ToolCallPart):
            seen_tool_call = True
            if part.tool_call_id in results_by_id:
                current_results.append(results_by_id[part.tool_call_id])

    # Flush remaining
    if current:
        messages.append(messages_.Message(role="assistant", parts=current, id=msg_id))
    if current_results:
        messages.append(messages_.Message(role="tool", parts=list(current_results)))

    return messages

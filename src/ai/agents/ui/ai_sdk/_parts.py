"""Shared conversions between internal Part objects and UIMessagePart objects.

Used by ``outbound.history`` to reconstruct UIMessages from persisted
``ai.messages.Message`` lists. The live outbound stream does not use these; it
emits wire-protocol deltas directly from event streams.
"""

from __future__ import annotations

import json
from typing import Any, cast

from ....types import messages as messages_
from . import _approvals, ui_message

_TOOL_STATE_RANK: dict[ui_message.UIToolInvocationState, int] = {
    "input-streaming": 0,
    "input-available": 1,
    "approval-requested": 2,
    "approval-responded": 3,
    "output-denied": 4,
    "output-error": 5,
    "output-available": 6,
}


def _normalize_tool_input(raw: str) -> str | dict[str, Any]:
    """Parse tool args JSON string into a dict; fall back to raw string.

    TODO(datamodel-rework §4): once ``ToolCallPart.tool_args`` has a
    canonical shape, drop this helper.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    return parsed if isinstance(parsed, dict) else raw


def to_ui_parts(parts: list[messages_.Part]) -> list[ui_message.UIMessagePart]:
    """Convert internal Part objects to UIMessagePart objects."""
    result: list[ui_message.UIMessagePart] = []
    for part in parts:
        if isinstance(part, messages_.TextPart) and part.text:
            result.append(ui_message.UITextPart(type="text", text=part.text))
        elif isinstance(part, messages_.ReasoningPart) and part.text:
            result.append(
                ui_message.UIReasoningPart(type="reasoning", text=part.text)
            )
        elif isinstance(part, messages_.ToolCallPart):
            result.append(
                ui_message.UIToolPart.model_validate(
                    {
                        "type": f"tool-{part.tool_name}",
                        "toolCallId": part.tool_call_id,
                        "state": "input-available",
                        "input": _normalize_tool_input(part.tool_args),
                    }
                )
            )
        elif isinstance(part, messages_.FilePart):
            result.append(
                ui_message.UIFilePart.model_validate(
                    {
                        "type": "file",
                        "mediaType": part.media_type,
                        "url": part.data if isinstance(part.data, str) else "",
                        "filename": part.filename,
                    }
                )
            )
    return result


def _merge_tool_part(
    existing: ui_message.UIToolPart,
    candidate: ui_message.UIToolPart,
) -> ui_message.UIToolPart:
    """Merge duplicate UI tool parts, keeping the first display position."""
    existing_rank = _TOOL_STATE_RANK.get(existing.state, 0)
    candidate_rank = _TOOL_STATE_RANK.get(candidate.state, 0)
    updates: dict[str, Any] = {}

    if candidate_rank >= existing_rank:
        updates["state"] = candidate.state
        if candidate.state == "output-denied":
            updates["output"] = None

    if existing.input is None and candidate.input is not None:
        updates["input"] = candidate.input
    if candidate.output is not None:
        updates["output"] = candidate.output
    if candidate.error_text is not None:
        updates["error_text"] = candidate.error_text
    if candidate.approval is not None:
        updates["approval"] = candidate.approval

    return existing.model_copy(update=updates) if updates else existing


def dedupe_tool_parts(
    ui_parts: list[ui_message.UIMessagePart],
) -> list[ui_message.UIMessagePart]:
    """Collapse duplicate UIToolParts by tool_call_id."""
    result: list[ui_message.UIMessagePart] = []
    tool_index: dict[str, int] = {}

    for part in ui_parts:
        if not isinstance(part, ui_message.UIToolPart):
            result.append(part)
            continue

        idx = tool_index.get(part.tool_call_id)
        if idx is None:
            tool_index[part.tool_call_id] = len(result)
            result.append(part)
            continue

        existing = result[idx]
        if isinstance(existing, ui_message.UIToolPart):
            result[idx] = _merge_tool_part(existing, part)

    return result


def merge_tool_results(
    ui_parts: list[ui_message.UIMessagePart],
    tool_parts: list[messages_.Part],
) -> None:
    """Merge ToolResultParts into existing UIToolParts in-place."""
    tool_index: dict[str, int] = {}
    for idx, ui_part in enumerate(ui_parts):
        if isinstance(ui_part, ui_message.UIToolPart):
            tool_index[ui_part.tool_call_id] = idx

    for part in tool_parts:
        if not isinstance(part, messages_.ToolResultPart):
            continue
        # Hook-abort placeholders are internal: the corresponding
        # HookPart(pending) carries the user-visible state via
        # merge_approval_signals.
        if part.is_hook_pending:
            continue
        idx_opt = tool_index.get(part.tool_call_id)
        if idx_opt is None:
            continue
        idx = idx_opt
        existing = ui_parts[idx]
        if not isinstance(existing, ui_message.UIToolPart):
            continue
        if existing.state == "output-denied":
            continue
        state = "output-error" if part.is_error else "output-available"
        ui_parts[idx] = existing.model_copy(
            update={"state": state, "output": part.result}
        )


def merge_approval_signals(
    ui_parts: list[ui_message.UIMessagePart],
    internal_parts: list[messages_.Part],
) -> None:
    """Merge HookPart approval state into existing UIToolParts in-place."""
    tool_index: dict[str, int] = {}
    for idx, ui_part in enumerate(ui_parts):
        if isinstance(ui_part, ui_message.UIToolPart):
            tool_index[ui_part.tool_call_id] = idx

    for part in internal_parts:
        if not isinstance(part, messages_.HookPart):
            continue

        tool_call_id = _approvals.tool_call_id_for(part)
        if tool_call_id is None:
            continue

        idx_opt = tool_index.get(tool_call_id)
        if idx_opt is None:
            continue
        idx = idx_opt

        existing = ui_parts[idx]
        if not isinstance(existing, ui_message.UIToolPart):
            continue

        updates: dict[str, Any] = {}
        if part.status == "pending":
            updates["state"] = "approval-requested"
            updates["approval"] = ui_message.UIToolApproval(id=part.hook_id)
        elif part.status == "resolved":
            resolution = cast(
                "dict[str, Any]",
                part.resolution if isinstance(part.resolution, dict) else {},
            )
            updates["approval"] = ui_message.UIToolApproval(
                id=part.hook_id,
                approved=resolution.get("granted"),
                reason=resolution.get("reason"),
            )
            if resolution.get("granted", False):
                updates["state"] = "approval-responded"
            else:
                updates["state"] = "output-denied"
                updates["output"] = None
        elif part.status == "cancelled":
            updates["state"] = "output-error"
            updates["error_text"] = "Hook cancelled"

        if updates:
            ui_parts[idx] = existing.model_copy(update=updates)

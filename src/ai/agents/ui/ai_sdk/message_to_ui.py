"""
Message to UI: internal Message -> UI format conversion.

Converts internal Message objects back into AI SDK v6 UI format,
both for persistence (DB storage, GET endpoints) and for
accumulating streaming assistant turns.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ...types import messages as messages_
from . import ui_message

logger = logging.getLogger(__name__)


# ============================================================================
# Part-level conversions
# ============================================================================


def parts_to_ui(parts: list[messages_.Part]) -> list[dict[str, Any]]:
    """Convert internal Part objects to UI-compatible dicts.

    The frontend expects the AI SDK UI protocol shape (``type``, ``text``,
    ``toolCallId``, ``toolName``, ``input``, ``output``, ``state``, etc.)
    which differs from the internal model.

    Handles: TextPart, ReasoningPart, ToolCallPart, ToolResultPart, FilePart.
    ToolResultParts are returned as standalone dicts (with ``output`` and
    terminal state); callers that want merged tool-call+result dicts should
    use ``UIMessageBuilder`` instead.
    """
    result: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, messages_.TextPart):
            if part.text:
                result.append({"type": "text", "text": part.text})
        elif isinstance(part, messages_.ReasoningPart):
            if part.text:
                result.append({"type": "reasoning", "reasoning": part.text})
        elif isinstance(part, messages_.ToolCallPart):
            result.append(
                {
                    "type": f"tool-{part.tool_name}",
                    "toolCallId": part.tool_call_id,
                    "toolName": part.tool_name,
                    "state": "input-available",
                    "input": _normalize_tool_input(part.tool_args),
                }
            )
        elif isinstance(part, messages_.ToolResultPart):
            state = "output-error" if part.is_error else "output-available"
            result.append(
                {
                    "type": f"tool-{part.tool_name}",
                    "toolCallId": part.tool_call_id,
                    "toolName": part.tool_name,
                    "state": state,
                    "output": part.result,
                }
            )
        elif isinstance(part, messages_.FilePart):
            entry: dict[str, Any] = {
                "type": "file",
                "mediaType": part.media_type,
                "url": part.data if isinstance(part.data, str) else "",
            }
            if part.filename:
                entry["filename"] = part.filename
            result.append(entry)
    return result


def ui_parts_to_dicts(
    parts: list[ui_message.UIMessagePart],
) -> list[dict[str, Any]]:
    """Serialize UIMessage parts to plain dicts for DB storage."""
    return [
        part.model_dump() if hasattr(part, "model_dump") else dict(part)  # type: ignore[call-overload]
        for part in parts
    ]


# ============================================================================
# Message-level conversion (batch, for history loading)
# ============================================================================


def messages_to_ui(
    messages: list[messages_.Message],
) -> list[ui_message.UIMessage]:
    """Convert internal Messages to UIMessages.

    This is the inverse of ``inbound.ui_to_messages()``. It merges
    consecutive assistant + tool + signal message groups back into single
    UIMessages with the correct tool states, producing the format
    expected by the AI SDK frontend and suitable for DB persistence.

    User/system messages are converted directly. Assistant messages
    are accumulated until a non assistant/tool/signal message is seen (or
    the list ends), merging tool results and approval signals into the
    preceding assistant's tool-call parts.
    """
    result: list[ui_message.UIMessage] = []

    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role in ("user", "system"):
            result.append(
                ui_message.UIMessage(
                    id=msg.id,
                    role=msg.role,
                    parts=_internal_parts_to_ui_parts(msg.parts),
                )
            )
            i += 1
            continue

        if msg.role == "assistant":
            # Accumulate: merge this assistant message with any following
            # tool/signal messages, and possibly more assistant+tool pairs that
            # belong to the same UI turn.
            ui_parts: list[ui_message.UIMessagePart] = []
            turn_id = msg.id

            while i < len(messages) and messages[i].role in (
                "assistant",
                "tool",
                "signal",
            ):
                current = messages[i]
                if current.role == "assistant":
                    ui_parts.extend(_internal_parts_to_ui_parts(current.parts))
                elif current.role == "tool":
                    _merge_tool_results(ui_parts, current.parts)
                elif current.role == "signal":
                    _merge_signal_parts(ui_parts, current.parts)
                i += 1

            result.append(
                ui_message.UIMessage(
                    id=turn_id,
                    role="assistant",
                    parts=ui_parts,
                )
            )
            continue

        # Skip signal, tool (orphaned), or unknown roles
        i += 1

    return result


def _internal_parts_to_ui_parts(
    parts: list[messages_.Part],
) -> list[ui_message.UIMessagePart]:
    """Convert internal Part objects to UIMessagePart objects."""
    result: list[ui_message.UIMessagePart] = []
    for part in parts:
        if isinstance(part, messages_.TextPart) and part.text:
            result.append(ui_message.UITextPart(type="text", text=part.text))
        elif isinstance(part, messages_.ReasoningPart) and part.text:
            result.append(
                ui_message.UIReasoningPart(type="reasoning", reasoning=part.text)
            )
        elif isinstance(part, messages_.ToolCallPart):
            tool_input = _normalize_tool_input(part.tool_args)
            result.append(
                ui_message.UIToolPart(
                    type=f"tool-{part.tool_name}",
                    tool_call_id=part.tool_call_id,
                    state="input-available",
                    input=tool_input,
                )
            )
        elif isinstance(part, messages_.FilePart):
            result.append(
                ui_message.UIFilePart(
                    type="file",
                    media_type=part.media_type,
                    url=part.data if isinstance(part.data, str) else "",
                    filename=part.filename,
                )
            )
    return result


def _merge_tool_results(
    ui_parts: list[ui_message.UIMessagePart],
    tool_parts: list[messages_.Part],
) -> None:
    """Merge tool result parts into existing UI tool-call parts in-place.

    Finds the matching UIToolPart by tool_call_id and updates its state
    and output to reflect the tool result.
    """
    # Index existing tool parts by tool_call_id
    tool_index: dict[str, int] = {}
    for idx, ui_part in enumerate(ui_parts):
        if isinstance(ui_part, ui_message.UIToolPart):
            tool_index[ui_part.tool_call_id] = idx

    for part in tool_parts:
        if not isinstance(part, messages_.ToolResultPart):
            continue
        idx = tool_index.get(part.tool_call_id)
        if idx is None:
            continue
        existing = ui_parts[idx]
        if not isinstance(existing, ui_message.UIToolPart):
            continue
        if existing.state == "output-denied":
            continue
        state = "output-error" if part.is_error else "output-available"
        ui_parts[idx] = existing.model_copy(
            update={"state": state, "output": part.result}
        )


def _merge_signal_parts(
    ui_parts: list[ui_message.UIMessagePart],
    signal_parts: list[messages_.Part],
) -> None:
    """Merge HookPart approval state into existing UI tool-call parts."""
    tool_index: dict[str, int] = {}
    for idx, ui_part in enumerate(ui_parts):
        if isinstance(ui_part, ui_message.UIToolPart):
            tool_index[ui_part.tool_call_id] = idx

    for part in signal_parts:
        if not isinstance(part, messages_.HookPart):
            continue

        tool_call_id = _tool_call_id_from_approval_id(part.hook_id)
        if tool_call_id is None:
            continue

        idx = tool_index.get(tool_call_id)
        if idx is None:
            continue

        existing = ui_parts[idx]
        if not isinstance(existing, ui_message.UIToolPart):
            continue

        updates: dict[str, Any] = {}
        if part.status == "pending":
            updates["state"] = "approval-requested"
            updates["approval"] = {"id": part.hook_id}
        elif part.status == "resolved":
            resolution = part.resolution or {}
            updates["approval"] = {
                "id": part.hook_id,
                "approved": resolution.get("granted"),
                "reason": resolution.get("reason"),
            }
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


# ============================================================================
# UIMessageBuilder: streaming accumulator for assistant turns
# ============================================================================


class UIMessageBuilder:
    """Accumulate streaming runtime messages into a single UI assistant message.

    Processes internal Message objects as they arrive from the agent loop
    and builds up a single UIMessage with all parts in the AI SDK UI format.
    Handles text, reasoning, tool calls, tool results, and approval signals.

    Usage::

        builder = UIMessageBuilder()
        async for msg in agent_stream:
            builder.ingest(msg)
        ui_msg = builder.build()
    """

    def __init__(self, message_id: str | None = None) -> None:
        self.message_id = message_id
        self.parts: list[dict[str, Any]] = []
        self._tool_indexes: dict[str, int] = {}

    @classmethod
    def from_ui_message(cls, message: ui_message.UIMessage) -> UIMessageBuilder:
        """Seed the builder from an existing UI assistant message (for resume)."""
        builder = cls(message_id=message.id)
        builder.parts = ui_parts_to_dicts(message.parts)
        for index, part in enumerate(builder.parts):
            part_type = part.get("type")
            tool_call_id = part.get("toolCallId")
            if (
                isinstance(part_type, str)
                and part_type.startswith("tool-")
                and isinstance(tool_call_id, str)
            ):
                builder._tool_indexes[tool_call_id] = index
        return builder

    def ingest(self, message: messages_.Message) -> None:
        """Consume one runtime message.

        Routes by role:
        - ``assistant`` (done): appends text, reasoning, and tool-call parts
        - ``tool``: updates existing tool parts with results
        - ``signal``: updates tool parts with approval state
        """
        if message.role == "assistant" and message.is_done:
            self._ingest_assistant(message)
        elif message.role == "tool":
            self._ingest_tool(message)
        elif message.role == "signal":
            self._ingest_signal(message)

    def build(self) -> ui_message.UIMessage | None:
        """Return the accumulated UIMessage, or None if nothing was ingested."""
        if not self.parts or self.message_id is None:
            return None
        # Parse the accumulated dicts back into typed UIMessageParts
        parsed_parts: list[ui_message.UIMessagePart] = []
        for part_dict in self.parts:
            parsed = ui_message._parse_ui_part(part_dict)
            if parsed is not None:
                parsed_parts.append(parsed)
        return ui_message.UIMessage(
            id=self.message_id,
            role="assistant",
            parts=parsed_parts,
        )

    @property
    def raw_parts(self) -> list[dict[str, Any]]:
        """Access the accumulated parts as raw dicts (for direct DB storage)."""
        return self.parts

    # -- Private ingest handlers --

    def _ingest_assistant(self, message: messages_.Message) -> None:
        if self.message_id is None:
            self.message_id = message.id
        for part in message.parts:
            if isinstance(part, messages_.ReasoningPart) and part.text:
                candidate = {"type": "reasoning", "reasoning": part.text}
                if self.parts[-1:] != [candidate]:
                    self.parts.append(candidate)
            elif isinstance(part, messages_.TextPart) and part.text:
                candidate = {"type": "text", "text": part.text}
                if self.parts[-1:] != [candidate]:
                    self.parts.append(candidate)
            elif isinstance(part, messages_.ToolCallPart):
                if part.tool_call_id in self._tool_indexes:
                    continue
                self._tool_indexes[part.tool_call_id] = len(self.parts)
                self.parts.append(
                    {
                        "type": f"tool-{part.tool_name}",
                        "toolCallId": part.tool_call_id,
                        "toolName": part.tool_name,
                        "state": "input-available",
                        "input": _normalize_tool_input(part.tool_args),
                    }
                )

    def _ingest_tool(self, message: messages_.Message) -> None:
        for part in message.parts:
            if not isinstance(part, messages_.ToolResultPart):
                continue
            index = self._tool_indexes.get(part.tool_call_id)
            if index is None:
                continue
            tool_part = dict(self.parts[index])
            if tool_part.get("state") != "output-denied":
                tool_part["state"] = (
                    "output-error" if part.is_error else "output-available"
                )
            tool_part["output"] = part.result
            self.parts[index] = tool_part

    def _ingest_signal(self, message: messages_.Message) -> None:
        hook_part = message.get_hook_part()
        if hook_part is None:
            return
        tool_call_id = _tool_call_id_from_approval_id(hook_part.hook_id)
        if tool_call_id is None:
            return
        index = self._tool_indexes.get(tool_call_id)
        if index is None:
            return

        tool_part = dict(self.parts[index])
        if hook_part.status == "pending":
            tool_part["state"] = "approval-requested"
            tool_part["approval"] = {"id": hook_part.hook_id}
        elif hook_part.status == "resolved":
            resolution = hook_part.resolution or {}
            tool_part["approval"] = {
                "id": hook_part.hook_id,
                "approved": resolution.get("granted"),
                "reason": resolution.get("reason"),
            }
            if resolution.get("granted", False):
                tool_part["state"] = "approval-responded"
            else:
                tool_part["state"] = "output-denied"
        elif hook_part.status == "cancelled":
            tool_part["state"] = "output-error"
            tool_part["errorText"] = "Hook cancelled"
        self.parts[index] = tool_part


# ============================================================================
# Shared helpers
# ============================================================================


def _tool_call_id_from_approval_id(approval_id: str) -> str | None:
    """Extract the tool_call_id from a ToolApproval hook label.

    E.g. ``"approve_tc_abc123"`` -> ``"tc_abc123"``.
    """
    prefix = "approve_"
    if approval_id.startswith(prefix):
        return approval_id[len(prefix) :]
    return None


def _normalize_tool_input(raw: str) -> str | dict[str, Any]:
    """Normalize tool input for the UI's accepted string-or-dict shape.

    Tries to parse the JSON string into a dict. Falls back to the
    raw string if parsing fails or the result isn't a dict.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    return parsed if isinstance(parsed, dict) else raw
